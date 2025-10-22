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
from typing import Optional, Dict, Any

# --- Module-Specific Configuration ---
# This makes the module self-sufficient
urllib3.disable_warnings(InsecureRequestWarning)
GEMINI_API_LOCK = asyncio.Lock()
YFINANCE_API_SEMAPHORE = asyncio.Semaphore(8)

# Initialize gemini_model within the module
gemini_model = None
try:
    config = configparser.ConfigParser()
    config.read('config.ini')
    GEMINI_API_KEY = config.get('API_KEYS', 'GEMINI_API_KEY', fallback=None)
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
except Exception as e:
    print(f"Warning: Could not configure Gemini model in sentiment_command.py: {e}")

# --- Helper Functions for this Command ---

async def get_yfinance_info_robustly(ticker: str) -> Optional[Dict[str, Any]]:
    """A robust, centralized function to fetch yfinance .info data."""
    async with YFINANCE_API_SEMAPHORE:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                await asyncio.sleep(np.random.uniform(0.2, 0.5))
                stock_info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)
                if stock_info and not stock_info.get('regularMarketPrice'):
                    raise ValueError(f"Incomplete data received for {ticker}")
                return stock_info
            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep((attempt + 1) * 2)
                else:
                    return None
    return None

async def get_company_name(ticker: str) -> str:
    """Fetches the long name of a company from its ticker."""
    try:
        stock_info = await get_yfinance_info_robustly(ticker)
        if stock_info:
            return stock_info.get('longName') or stock_info.get('shortName') or ticker
        return ticker
    except Exception:
        return ticker

async def scrape_finviz_headlines(ticker: str) -> list[str]:
    """Scrapes news headlines for a given ticker from Finviz."""
    headlines = []
    try:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=15, verify=False)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        news_table = soup.find(id='news-table')
        if news_table:
            for row in news_table.find_all('tr'):
                title_element = row.find('a', class_='news-link-left')
                if title_element:
                    headlines.append(title_element.get_text(strip=True))
    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code != 404:
            print(f"   -> Finviz HTTP error for {ticker}: {http_err}")
    except Exception:
        pass # Silently fail on other scrape errors
    return headlines

async def scrape_reddit_combined(ticker: str, company_name: str) -> list[str]:
    """Searches Reddit for a ticker and company name."""
    post_titles = set()
    headers = {'User-Agent': 'Mozilla/5.0 SingularityBot/1.0'}
    search_query = f'"{ticker}" OR "{company_name}"'
    encoded_query = quote_plus(search_query)
    subreddits = ['wallstreetbets', 'stocks', 'investing', ticker.lower()]

    for subreddit in subreddits:
        try:
            search_url = f"https://old.reddit.com/r/{subreddit}/search/?q={encoded_query}&restrict_sr=1&sort=hot&t=week"
            response = await asyncio.to_thread(requests.get, search_url, headers=headers, timeout=15, verify=False)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                for i, post in enumerate(soup.find_all('a', class_='search-title may-blank')):
                    if i >= 5: break
                    post_titles.add(post.get_text(strip=True))
        except Exception:
            continue # Silently fail
    return list(post_titles)

async def get_ai_sentiment_analysis(
    text_to_analyze: str, 
    topic_name: str, 
    model_to_use: Any, 
    lock_to_use: asyncio.Lock
) -> Optional[Dict[str, Any]]:
    """Core AI logic for sentiment analysis, using a provided model instance."""
    if not model_to_use:
        print("-> AI model is not configured. Cannot perform sentiment analysis.")
        return None
    prompt = f"""
    Analyze the sentiment of the following text regarding '{topic_name}'.
    Return a JSON object with this exact structure:
    {{
      "sentiment_score": float, "summary": "string",
      "positive_keywords": ["string"], "negative_keywords": ["string"]
    }}
    - 'sentiment_score' must be a float between -1.0 and 1.0.
    ---
    {text_to_analyze[:4000]}
    ---
    """
    async with lock_to_use:
        try:
            response = await asyncio.to_thread(
                model_to_use.generate_content, prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2, response_mime_type="application/json"
                )
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"-> An error occurred during AI sentiment analysis: {e}")
            return None

# --- Main Command Handler (Modified) ---
async def handle_sentiment_command(
    args: list = None, 
    ai_params: dict = None, 
    is_called_by_ai: bool = False,
    gemini_model_override: Optional[Any] = None,
    api_lock_override: Optional[asyncio.Lock] = None
):
    """
    Performs AI-driven sentiment analysis on a stock by scraping news and social media.
    Provides a raw score and scores rounded to the nearest 0.1 and 0.25.
    Accepts an optional override for the gemini model and api lock.
    """
    ticker = None
    if is_called_by_ai and ai_params:
        ticker = ai_params.get("ticker")
    elif args:
        ticker = args[0]
    
    if not ticker:
        message = "Please provide a stock ticker. Usage: /sentiment <TICKER>"
        if not is_called_by_ai: print(message)
        return {"error": message} if is_called_by_ai else None

    ticker = ticker.upper()
    if not is_called_by_ai:
        print(f"\n--- AI Sentiment Analysis for {ticker} ---")
        print("-> Scraping recent news and social media mentions...")

    company_name = await get_company_name(ticker)
    
    finviz_task = scrape_finviz_headlines(ticker)
    reddit_task = scrape_reddit_combined(ticker, company_name)
    headlines, reddit_titles = await asyncio.gather(finviz_task, reddit_task)
    
    combined_text = "\n".join(headlines) + "\n" + "\n".join(reddit_titles)
    
    if not combined_text.strip():
        message = f"-> Could not find any recent text for {ticker}."
        if not is_called_by_ai: print(message)
        return {"error": message} if is_called_by_ai else None
        
    if not is_called_by_ai: print(f"-> Data gathered for '{company_name}'. Sending to AI for analysis...")

    # Use override model/lock if provided, otherwise use module's own globals
    model_to_use = gemini_model_override or gemini_model
    lock_to_use = api_lock_override or GEMINI_API_LOCK
    
    analysis_result = await get_ai_sentiment_analysis(combined_text, f"{ticker} ({company_name})", model_to_use, lock_to_use)

    if not analysis_result:
        message = "-> AI analysis failed or returned no result."
        if not is_called_by_ai: print(message)
        return {"error": message}

    raw_score = analysis_result.get("sentiment_score", 0.0)
    score_rounded_01 = round(raw_score * 10) / 10.0
    score_rounded_025 = round(raw_score * 4) / 4.0

    summary = analysis_result.get("summary", "No summary provided.")
    pos_keys = analysis_result.get("positive_keywords", [])
    neg_keys = analysis_result.get("negative_keywords", [])

    if not is_called_by_ai:
        print("\n--- Sentiment Analysis Results ---")
        score_bar = "🐻" * int(max(0, ((-raw_score + 1)/2) * 10)) + " | " + "🐂" * int(max(0, ((raw_score + 1)/2) * 10))
        
        score_data = [
            ["Raw Score", f"{raw_score:.4f}", score_bar],
            ["Nearest 0.25", f"{score_rounded_025:.2f}", ""],
            ["Nearest 0.1", f"{score_rounded_01:.1f}", ""]
        ]
        print(tabulate(score_data, headers=["Score Type", "Value", "Visual"], tablefmt="grid"))

        print("\n  AI Summary:")
        print(f"    {summary}")
        print("\n  Positive Keywords:", ", ".join(pos_keys) if pos_keys else "None identified")
        print("  Negative Keywords:", ", ".join(neg_keys) if neg_keys else "None identified")
        print("---------------------------------")

    # Return all scores for potential AI use
    return {
        "ticker": ticker,
        "sentiment_score_raw": raw_score,
        "sentiment_score_rounded_01": score_rounded_01,
        "sentiment_score_rounded_025": score_rounded_025,
        "summary": summary,
        "positive_keywords": pos_keys,
        "negative_keywords": neg_keys
    }