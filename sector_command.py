# --- Imports for sector_command ---
import asyncio
import os
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus

import yfinance as yf
import pandas as pd
import humanize
import requests
from bs4 import BeautifulSoup
from tabulate import tabulate

# --- Imports from other command modules ---
from fundamentals_command import handle_fundamentals_command
from invest_command import calculate_ema_invest
from sentiment_command import get_ai_sentiment_analysis

# --- Global Variables & Constants ---
YFINANCE_API_SEMAPHORE = asyncio.Semaphore(8)

# --- Helper Functions (copied or moved for self-containment) ---

def get_gics_map(filepath="gics_map.txt") -> Dict[str, str]:
    if not os.path.exists(filepath): return {}
    gics_map = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if ':' in line:
                    code, name = line.strip().split(':', 1)
                    gics_map[code] = name
    except Exception: pass
    return gics_map

def get_all_gics_tickers(txt_path: str = 'gics_database.txt') -> set:
    if not os.path.exists(txt_path): return set()
    all_tickers = set()
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                if ':' in line:
                    _, tickers_str = line.split(':', 1)
                    all_tickers.update(t.strip().upper() for t in tickers_str.split(',') if t.strip())
    except Exception: return set()
    return all_tickers

def filter_stocks_by_gics(user_inputs_str: str, txt_path: str = 'gics_database.txt') -> set:
    if not os.path.exists(txt_path): return set()
    user_inputs_list = [item.strip() for item in user_inputs_str.split(',')]
    gics_map = get_gics_map()
    name_to_code_map = {name.lower(): code for code, name in gics_map.items()}
    gics_data = {}
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                if ':' in line: code, tickers = line.split(':', 1); gics_data[code.strip()] = tickers.strip()
    except Exception: return set()
    target_codes = set()
    for item in user_inputs_list:
        item_lower = item.lower()
        if item.isdigit(): target_codes.add(item)
        else:
            for name, code in name_to_code_map.items():
                if item_lower in name: target_codes.add(code)
    selected_tickers = set()
    for user_code in target_codes:
        for db_code, tickers_str in gics_data.items():
            if db_code.startswith(user_code):
                selected_tickers.update(t.strip().upper() for t in tickers_str.split(',') if t.strip())
    return selected_tickers

async def get_yfinance_info_robustly(ticker: str) -> Optional[Dict[str, Any]]:
    async with YFINANCE_API_SEMAPHORE:
        for attempt in range(3):
            try:
                stock_info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)
                if stock_info and stock_info.get('regularMarketPrice'): return stock_info
            except Exception:
                if attempt < 2: await asyncio.sleep((attempt + 1) * 2)
    return None

async def get_top_constituents_by_market_cap(tickers: List[str], top_n: int = 10) -> List[Dict[str, Any]]:
    async def fetch_market_cap(ticker):
        stock_info = await get_yfinance_info_robustly(ticker)
        if stock_info and stock_info.get('marketCap'):
            return {'ticker': ticker, 'market_cap': stock_info['marketCap']}
        return None
    results = await asyncio.gather(*[fetch_market_cap(t) for t in tickers])
    valid_results = [res for res in results if res]
    return sorted(valid_results, key=lambda x: x['market_cap'], reverse=True)[:top_n]

async def get_sector_performance_change(tickers: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    if not tickers: return {}
    data = await asyncio.to_thread(yf.download, tickers=tickers, period="1y", progress=False)
    if data.empty: return {t: {'1M': None, '1Y': None} for t in tickers}
    close_data = data.get('Close')
    results = {}
    for ticker in tickers:
        prices = close_data[ticker] if isinstance(close_data, pd.DataFrame) else close_data
        if prices is not None and not prices.dropna().empty:
            p = prices.dropna()
            change_1m = ((p.iloc[-1] / p.iloc[-22]) - 1) * 100 if len(p) >= 22 else None
            change_1y = ((p.iloc[-1] / p.iloc[0]) - 1) * 100
            results[ticker] = {'1M': change_1m, '1Y': change_1y}
        else:
            results[ticker] = {'1M': None, '1Y': None}
    return results

async def calculate_market_invest_scores_singularity(tickers: List[str], ema_sens: int, is_called_by_ai: bool = False) -> List[Dict[str, Any]]:
    tasks = [calculate_ema_invest(ticker, ema_sens, is_called_by_ai=True) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    output = [{'ticker': tickers[i], 'live_price': res[0], 'score': res[1]} for i, res in enumerate(results) if res]
    return sorted(output, key=lambda x: x.get('score', -1), reverse=True)

async def scrape_sector_news_headlines(sector_name: str) -> List[str]:
    headlines = []
    try:
        query = quote_plus(f'"{sector_name}" industry news')
        url = f"https://www.google.com/search?q={query}&tbm=nws"
        response = await asyncio.to_thread(requests.get, url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        soup = BeautifulSoup(response.text, 'html.parser')
        for item in soup.select('div.n0jPhd'):
            headlines.append(item.get_text(strip=True))
        return list(dict.fromkeys(headlines))[:20]
    except Exception:
        return []

# --- Main Command Handler ---

async def handle_sector_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    if is_called_by_ai:
        return {"status": "error_invalid_tool", "message": "This function is not for AI use. Use 'find_and_screen_stocks'."}

    print("\n--- Industry & Sector Analysis Engine ---")
    if not args:
        print("Usage: /sector <GICS_CODE | Sector/Industry Name | \"Market\">")
        return

    input_str = " ".join(args)
    sector_name = input_str.strip('\"')
    all_tickers = get_all_gics_tickers() if sector_name.lower() == 'market' else filter_stocks_by_gics(sector_name)

    if not all_tickers:
        print(f"-> No tickers found for '{sector_name}'.")
        return

    print(f"-> Found {len(all_tickers)} tickers. Analyzing top constituents...")
    
    top_10 = await get_top_constituents_by_market_cap(list(all_tickers))
    if not top_10:
        print("-> Could not determine top constituents. Aborting.")
        return
        
    top_10_tickers = [c['ticker'] for c in top_10]
    print("\n--- Top 10 by Market Cap ---")
    print(tabulate([[c['ticker'], f"${humanize.intword(c['market_cap'])}"] for c in top_10], headers=["Ticker", "Market Cap"]))

    perf_data = await get_sector_performance_change(top_10_tickers)
    print("\n--- Top 10 Performance ---")
    perf_table = [[t, f"{perf_data.get(t, {}).get('1M'):.2f}%", f"{perf_data.get(t, {}).get('1Y'):.2f}%"] for t in top_10_tickers]
    print(tabulate(perf_table, headers=["Ticker", "1M Change", "1Y Change"]))

    funda_scores = [res['fundamental_score'] for t in top_10_tickers if (res := await handle_fundamentals_command(ai_params={'ticker': t}, is_called_by_ai=True)) and 'fundamental_score' in res]
    invest_scores = [res[1] for t in top_10_tickers if (res := await calculate_ema_invest(t, 2, is_called_by_ai=True)) and res[1] is not None]
    print(f"\n--- Health Scores (Top 10) ---")
    print(f"-> Avg. Fundamental Score: {sum(funda_scores)/len(funda_scores):.2f}/100" if funda_scores else "N/A")
    print(f"-> Avg. Invest Score: {sum(invest_scores)/len(invest_scores):.2f}%" if invest_scores else "N/A")

    print(f"\n--- Top/Bottom 5 by Invest Score (All {len(all_tickers)} Tickers) ---")
    all_scores = await calculate_market_invest_scores_singularity(list(all_tickers), 2, is_called_by_ai=True)
    if all_scores:
        top_5, bottom_5 = all_scores[:5], sorted(all_scores[-5:], key=lambda x: x.get('score', float('inf')))
        print("\n**Top 5**"); print(tabulate(top_5, headers="keys"))
        print("\n**Bottom 5**"); print(tabulate(bottom_5, headers="keys"))

    print(f"\n--- Sentiment Analysis for '{sector_name}' ---")
    headlines = await scrape_sector_news_headlines(sector_name)
    if headlines:
        sentiment = await get_ai_sentiment_analysis("\n".join(headlines), sector_name)
        if sentiment: print(f"  Score: {sentiment.get('sentiment_score', 0.0):.2f} | Summary: {sentiment.get('summary', 'N/A')}")