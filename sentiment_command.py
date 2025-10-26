# --- sentiment_command.py ---
# Standalone module for the /sentiment command.

import asyncio
import json
import re
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
import configparser
import os
from bs4 import BeautifulSoup
from urllib.parse import quote_plus
from urllib3.exceptions import InsecureRequestWarning
import urllib3
from tabulate import tabulate
from typing import Optional, Dict, Any, List # Added List

# --- Module-Specific Configuration ---
urllib3.disable_warnings(InsecureRequestWarning)
GEMINI_API_LOCK = asyncio.Lock() # Keep lock within the module
YFINANCE_API_SEMAPHORE = asyncio.Semaphore(8)

# Initialize gemini_model within the module
gemini_model = None
try:
    config = configparser.ConfigParser()
    # Ensure config.ini exists in the parent directory relative to this script
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.ini')
    if not os.path.exists(config_path):
        # If running standalone or config is elsewhere, adjust path as needed
        config_path = 'config.ini' # Fallback to current directory

    config.read(config_path)
    GEMINI_API_KEY = config.get('API_KEYS', 'GEMINI_API_KEY', fallback=None)
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using the correct model name (adjust if necessary, e.g., 'gemini-pro')
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
except Exception as e:
    print(f"Warning: Could not configure Gemini model in sentiment_command.py: {e}")

# --- Helper Functions ---

async def get_yfinance_info_robustly(ticker: str) -> Optional[Dict[str, Any]]:
    """A robust, centralized function to fetch yfinance .info data."""
    async with YFINANCE_API_SEMAPHORE:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(np.random.uniform(0.2, 0.5)) # Slight delay
                # Run blocking yf call in thread
                stock_info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)
                # Check for essential data presence
                if stock_info and ('regularMarketPrice' in stock_info or 'currentPrice' in stock_info):
                    return stock_info
                else:
                    # Raise error if essential data missing, triggers retry
                    raise ValueError(f"Incomplete data received for {ticker}")
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep((attempt + 1) * 2) # Exponential backoff
                # Optional: Log final failure
                # else: print(f"   -> ❌ ERROR: All attempts to fetch .info for {ticker} failed. Last error: {type(e).__name__}")
    return None # Return None if all retries fail


async def get_company_name(ticker: str) -> str:
    """Fetches the long name of a company from its ticker, falling back gracefully."""
    try:
        # Use the robust info fetcher
        stock_info = await get_yfinance_info_robustly(ticker)
        if stock_info:
            # Prioritize longName, then shortName, then ticker itself
            return stock_info.get('longName') or stock_info.get('shortName') or ticker
        return ticker # Fallback to ticker if info fetch fails
    except Exception:
        return ticker # Fallback in case of unexpected errors

async def scrape_finviz_headlines(ticker: str) -> list[str]:
    """Scrapes news headlines for a given ticker from Finviz."""
    headlines = []
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'} # Standard User-Agent
        # Run blocking requests call in thread
        response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=15, verify=False) # verify=False for potential SSL issues
        response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

        soup = BeautifulSoup(response.text, 'html.parser')
        news_table = soup.find(id='news-table')

        if news_table:
            # Find all relevant links within the table
            for link in news_table.find_all('a', class_='news-link-left'):
                 if link:
                     headlines.append(link.get_text(strip=True))

    # Specific handling for HTTP errors vs. general errors
    except requests.exceptions.HTTPError as http_err:
        # Finviz often returns 404 for invalid tickers, ignore those silently
        if http_err.response.status_code != 404:
            print(f"   -> [DEBUG Sentiment] Finviz HTTP error for {ticker}: {http_err}")
    except Exception as e:
        # Log other types of errors if needed for debugging
        # print(f"   -> [DEBUG Sentiment] Finviz scraping error for {ticker}: {type(e).__name__}")
        pass # Silently fail on other scrape errors (timeout, connection error, parsing error)
    # Limit number of headlines if needed (e.g., return headlines[:20])
    return headlines[:15] # Limit to 15 headlines


async def scrape_google_news(ticker: str, company_name: str) -> list[str]:
    """Scrapes Google News headlines for a ticker and company name."""
    headlines = set()
    try:
        # Search for ticker AND company name for relevance
        search_query = f'"{ticker}" OR "{company_name}" stock news'
        encoded_query = quote_plus(search_query)
        # Use tbm=nws for News tab, tbs=qdr:w for past week
        url = f"https://www.google.com/search?q={encoded_query}&tbm=nws&tbs=qdr:w"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')

        # Google selectors can change, this targets common headline elements
        # Looking for divs that often contain news titles
        for item in soup.select('div[role="heading"]'):
             title = item.get_text(strip=True)
             if title:
                  headlines.add(title)

        # Fallback/Alternative selector if the primary one fails
        if not headlines:
             for link in soup.find_all('a'):
                 h3 = link.find('h3')
                 if h3:
                     title = h3.get_text(strip=True)
                     if title: headlines.add(title)


    except requests.exceptions.HTTPError as http_err:
        print(f"   -> [DEBUG Sentiment] Google News HTTP error for {ticker}: {http_err}")
    except Exception as e:
        # print(f"   -> [DEBUG Sentiment] Google News scraping error for {ticker}: {type(e).__name__}")
        pass # Fail silently
    # Limit number of headlines
    return list(headlines)[:15] # Limit to 15 unique headlines


async def scrape_reddit_combined(ticker: str, company_name: str) -> list[str]:
    """Searches specified Reddit subreddits for a ticker and company name."""
    post_titles = set() # Use set to avoid duplicates automatically
    # More descriptive User-Agent
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; MarketInsightsCenterBot/1.0; +http://example.com/bot)'}

    # Prepare search query for URL encoding (quotes help find exact matches)
    search_query = f'"{ticker}" OR "{company_name}"'
    encoded_query = quote_plus(search_query)

    # List of subreddits to search
    subreddits = ['wallstreetbets', 'stocks', 'investing', 'finance'] # Added 'finance'

    # Asynchronously search each subreddit
    search_tasks = []
    for subreddit in subreddits:
        # Search within the specific subreddit for the query, sort by relevance/hot in past week
        search_url = f"https://old.reddit.com/r/{subreddit}/search/?q={encoded_query}&restrict_sr=1&sort=hot&t=week"
        # Create a task for each request
        search_tasks.append(asyncio.to_thread(requests.get, search_url, headers=headers, timeout=15, verify=False)) # verify=False

    # Execute searches concurrently
    responses = await asyncio.gather(*search_tasks, return_exceptions=True)

    # Process successful responses
    titles_from_subreddit = 0
    max_titles_per_sub = 5 # Limit titles per subreddit

    for i, response in enumerate(responses):
        if isinstance(response, Exception) or response.status_code != 200:
            # Log errors if needed, but continue processing others
            # print(f"   -> [DEBUG Sentiment] Reddit scrape failed for r/{subreddits[i]}: {response}")
            continue

        titles_from_subreddit = 0 # Reset counter for this subreddit
        try:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Find post titles using a more specific selector if possible
            # Example: finding 'a' tags with class 'search-title'
            for post_link in soup.find_all('a', class_='search-title'):
                title = post_link.get_text(strip=True)
                if title:
                    post_titles.add(title)
                    titles_from_subreddit += 1
                    if titles_from_subreddit >= max_titles_per_sub:
                        break # Stop processing this subreddit after reaching limit

        except Exception as parse_err:
            # Log parsing errors if they occur
             # print(f"   -> [DEBUG Sentiment] Reddit parsing error for r/{subreddits[i]}: {parse_err}")
             pass

    # Limit total number of titles returned
    return list(post_titles)[:20] # Limit to 20 unique titles overall


async def get_ai_sentiment_analysis(
    text_to_analyze: str,
    topic_name: str,
    model_to_use: Any, # Expects an initialized GenerativeModel instance
    lock_to_use: asyncio.Lock # Expects an asyncio.Lock instance
) -> Optional[Dict[str, Any]]:
    """Core AI logic for sentiment analysis, using a provided model and lock."""
    if not model_to_use:
        print("-> AI model is not configured. Cannot perform sentiment analysis.")
        return None

    # Truncate text to avoid exceeding model limits (adjust limit if needed)
    max_text_length = 8000 # Increased limit for potentially better context
    truncated_text = text_to_analyze[:max_text_length]

    # Enhanced prompt for better JSON structure and keyword relevance
    prompt = f"""
    Analyze the sentiment expressed in the following text specifically regarding '{topic_name}'.
    Consider the overall tone, specific opinions, and frequency of positive/negative language.

    Return ONLY a valid JSON object with this EXACT structure:
    {{
      "sentiment_score": float,
      "summary": "string",
      "positive_keywords": ["string"],
      "negative_keywords": ["string"]
    }}

    - 'sentiment_score': A float between -1.0 (very negative) and 1.0 (very positive). Neutral is 0.0.
    - 'summary': A concise 1-2 sentence summary of the overall sentiment towards '{topic_name}'.
    - 'positive_keywords': A list of up to 5 single words or short phrases from the text strongly indicating positive sentiment *about the topic*.
    - 'negative_keywords': A list of up to 5 single words or short phrases from the text strongly indicating negative sentiment *about the topic*.

    --- TEXT TO ANALYZE ---
    {truncated_text}
    --- END TEXT ---

    JSON Response:
    """

    # Ensure thread-safety for API calls if necessary (using the provided lock)
    async with lock_to_use:
        try:
            # Run the blocking API call in a separate thread
            response = await asyncio.to_thread(
                model_to_use.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2, # Lower temperature for more deterministic JSON output
                    response_mime_type="application/json" # Request JSON directly
                )
            )

            # --- Robust JSON Parsing ---
            if response and response.text:
                 try:
                      # Attempt to parse the text directly as JSON
                      analysis_result = json.loads(response.text)
                      # Basic validation of the parsed structure
                      if not all(k in analysis_result for k in ["sentiment_score", "summary", "positive_keywords", "negative_keywords"]):
                           raise ValueError("Parsed JSON missing required keys.")
                      if not isinstance(analysis_result["sentiment_score"], (int, float)):
                           raise ValueError("'sentiment_score' must be a number.")
                      return analysis_result
                 except (json.JSONDecodeError, ValueError) as json_err:
                      print(f"-> AI sentiment analysis: Failed to parse valid JSON. Error: {json_err}")
                      print(f"   Raw AI Response Text: {response.text[:200]}...") # Log beginning of raw response
                      return None # Indicate failure if JSON is bad
            else:
                 print("-> AI sentiment analysis: Received empty response from model.")
                 return None

        except Exception as e:
            # Log the specific error during the API call
            print(f"-> An error occurred during AI sentiment analysis API call: {type(e).__name__} - {e}")
            return None # Indicate failure

# --- Main Command Handler (Modified) ---
async def handle_sentiment_command(
    args: list = None,
    ai_params: dict = None,
    is_called_by_ai: bool = False,
    gemini_model_override: Optional[Any] = None, # Renamed
    api_lock_override: Optional[asyncio.Lock] = None, # Renamed
    **kwargs # Add **kwargs to accept unexpected arguments
):
    """
    Performs AI-driven sentiment analysis on a stock by scraping news and social media.
    Provides a raw score and rounded scores. Accepts optional model/lock overrides.
    Ignores unexpected keyword arguments via **kwargs.
    """
    ticker = None
    if is_called_by_ai and ai_params:
        ticker = ai_params.get("ticker")
    elif args: # CLI path
        ticker = args[0]
    # Handle case where neither AI nor CLI provides a ticker
    if not ticker:
        message = "Please provide a stock ticker. Usage: /sentiment <TICKER>"
        if not is_called_by_ai: print(message)
        # Return error structure for AI, None for CLI failure
        return {"status": "error", "message": message} if is_called_by_ai else None

    ticker = ticker.upper().strip() # Sanitize ticker

    if not is_called_by_ai:
        print(f"\n--- AI Sentiment Analysis for {ticker} ---")
        print("-> Scraping recent news and social media mentions (Finviz, Google News, Reddit)...")

    # Fetch company name asynchronously
    company_name = await get_company_name(ticker)

    # Run scraping tasks concurrently
    finviz_task = scrape_finviz_headlines(ticker)
    google_news_task = scrape_google_news(ticker, company_name) # ADDED Google News
    reddit_task = scrape_reddit_combined(ticker, company_name)

    # Gather results from all scraping tasks
    # return_exceptions=True prevents one failure from stopping others
    scrape_results = await asyncio.gather(
        finviz_task,
        google_news_task, # ADDED Google News
        reddit_task,
        return_exceptions=True
    )

    # Safely extract results
    headlines_finviz = scrape_results[0] if isinstance(scrape_results[0], list) else []
    headlines_google = scrape_results[1] if isinstance(scrape_results[1], list) else [] # ADDED Google News
    reddit_titles = scrape_results[2] if isinstance(scrape_results[2], list) else []

    # Combine text from all sources
    combined_text = "\n".join(headlines_finviz) + "\n" + "\n".join(headlines_google) + "\n" + "\n".join(reddit_titles)

    if not combined_text.strip():
        message = f"-> Could not find any recent text (Finviz/Google/Reddit) for {ticker} ({company_name})."
        if not is_called_by_ai: print(message)
        return {"status": "error", "message": message} if is_called_by_ai else None

    if not is_called_by_ai:
         # Updated message
         print(f"-> Data gathered ({len(headlines_finviz)} Finviz, {len(headlines_google)} Google, {len(reddit_titles)} Reddit). Sending to AI...")

    # Use override model/lock if provided, otherwise use module's own globals
    model_to_use = gemini_model_override or gemini_model
    lock_to_use = api_lock_override or GEMINI_API_LOCK

    # Perform AI analysis
    analysis_result = await get_ai_sentiment_analysis(combined_text, f"{ticker} ({company_name})", model_to_use, lock_to_use)

    if not analysis_result:
        message = "-> AI analysis failed or returned no result."
        if not is_called_by_ai: print(message)
        # Return error structure for AI, None for CLI
        return {"status": "error", "message": message} if is_called_by_ai else None

    # --- Process and Display Results ---
    try:
        raw_score = float(analysis_result.get("sentiment_score", 0.0))
        # Ensure score is within expected range
        raw_score = np.clip(raw_score, -1.0, 1.0)
    except (ValueError, TypeError):
        raw_score = 0.0 # Default to neutral if score is invalid

    # Calculate rounded scores
    score_rounded_01 = round(raw_score * 10) / 10.0
    score_rounded_025 = round(raw_score * 4) / 4.0

    summary = analysis_result.get("summary", "No summary provided.")
    pos_keys = analysis_result.get("positive_keywords", [])
    neg_keys = analysis_result.get("negative_keywords", [])

    # --- CLI Output ---
    if not is_called_by_ai:
        print("\n--- Sentiment Analysis Results ---")
        # Visual score bar (Bull/Bear)
        # Scale score from [-1, 1] to [0, 10] for bear side, and [0, 10] for bull side
        bear_intensity = int(max(0, ((-raw_score + 1) / 2) * 10))
        bull_intensity = int(max(0, ((raw_score + 1) / 2) * 10))
        # Adjusted padding for better alignment
        score_bar = ("🐻" * bear_intensity).ljust(10) + " | " + ("🐂" * bull_intensity).ljust(10)


        score_data = [
            ["Raw Score", f"{raw_score:.4f}", score_bar],
            ["Nearest 0.25", f"{score_rounded_025:.2f}", ""], # No bar needed
            ["Nearest 0.1", f"{score_rounded_01:.1f}", ""]  # No bar needed
        ]
        print(tabulate(score_data, headers=["Score Type", "Value", "Visual"], tablefmt="grid"))

        print("\n  AI Summary:")
        print(f"    {summary}")
        print("\n  Positive Keywords:", ", ".join(pos_keys) if pos_keys else "None identified")
        print("  Negative Keywords:", ", ".join(neg_keys) if neg_keys else "None identified")
        # Use a dynamic separator length
        separator_line = "-" * len("--- Sentiment Analysis Results ---")
        print(separator_line)


    # --- Return Value for AI/Prometheus ---
    # Return a structured dictionary including all calculated scores
    return {
        "status": "success", # Indicate success
        "ticker": ticker,
        "sentiment_score_raw": raw_score,
        "sentiment_score_rounded_01": score_rounded_01,
        "sentiment_score_rounded_025": score_rounded_025,
        "summary": summary,
        "positive_keywords": pos_keys,
        "negative_keywords": neg_keys
    }