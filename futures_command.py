# --- Imports for futures_command ---
import asyncio
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from dateutil.relativedelta import relativedelta
from tabulate import tabulate

# --- Imports from other command modules ---
from invest_command import calculate_ema_invest, plot_ticker_graph
from strategies_command import get_futures_specs

# --- Global Variables & Constants ---
YFINANCE_API_SEMAPHORE = asyncio.Semaphore(8)

# --- Helper Functions (copied or moved for self-containment) ---

async def get_yfinance_info_robustly(ticker: str) -> Optional[Dict[str, Any]]:
    """A robust, centralized function to fetch yfinance .info data."""
    async with YFINANCE_API_SEMAPHORE:
        for attempt in range(3):
            try:
                await asyncio.sleep(random.uniform(0.2, 0.5))
                stock_info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)
                if stock_info and not stock_info.get('regularMarketPrice'):
                    raise ValueError(f"Incomplete data for {ticker}")
                return stock_info
            except Exception:
                if attempt < 2:
                    await asyncio.sleep((attempt + 1) * 2)
                else:
                    return None
    return None

async def get_yf_download_robustly(tickers: list, **kwargs) -> pd.DataFrame:
    """A robust wrapper for yf.download with retry logic."""
    for attempt in range(3):
        try:
            data = await asyncio.to_thread(yf.download, tickers=tickers, progress=False, **kwargs)
            if not data.empty:
                return data
        except Exception:
            if attempt < 2:
                await asyncio.sleep((attempt + 1) * 2)
    return pd.DataFrame()

async def get_futures_info(symbol_root: str) -> Optional[Dict[str, Any]]:
    """Fetches and formats contract specifications for a given futures symbol."""
    specs = get_futures_specs().get(symbol_root)
    if not specs: return None
    
    yf_ticker = specs['ticker']
    live_data = await get_yfinance_info_robustly(yf_ticker)
    if not live_data: return None
    
    point_value = specs.get('point_value', 0.0)
    tick_size = specs.get('tick_size', 0.0)
    tick_value = point_value * tick_size
    
    return {
        "Product Name": specs.get('name'),
        "yfinance Ticker": yf_ticker,
        "Current Price": live_data.get('regularMarketPrice'),
        "Point Value": f"${point_value:,.2f}",
        "Tick Size": tick_size,
        "Tick Value": f"${tick_value:,.2f}",
        "Volume": live_data.get('volume'),
        "Open Interest": live_data.get('openInterest')
    }

async def analyze_futures_contract(symbol_root: str) -> Optional[Dict[str, str]]:
    """Calculates INVEST scores and signals for a futures contract."""
    specs = get_futures_specs().get(symbol_root)
    if not specs: return None
    
    yf_ticker = specs['ticker']
    tasks = [calculate_ema_invest(yf_ticker, i, is_called_by_ai=True) for i in range(1, 4)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    scores = {
        'weekly': results[0][1] if not isinstance(results[0], Exception) else None,
        'daily': results[1][1] if not isinstance(results[1], Exception) else None,
        'hourly': results[2][1] if not isinstance(results[2], Exception) else None,
    }

    def get_signal(score: Optional[float]) -> str:
        if score is None: return "N/A"
        return "SELL 🔴" if score < 40 else "BUY 🟢" if score > 60 else "HOLD 🟡"

    graph_file = await asyncio.to_thread(plot_ticker_graph, yf_ticker, 2, is_called_by_ai=True)
    
    return {
        "weekly_score": f"{scores['weekly']:.2f}%" if scores['weekly'] is not None else "N/A",
        "weekly_signal": get_signal(scores['weekly']),
        "daily_score": f"{scores['daily']:.2f}%" if scores['daily'] is not None else "N/A",
        "daily_signal": get_signal(scores['daily']),
        "hourly_score": f"{scores['hourly']:.2f}%" if scores['hourly'] is not None else "N/A",
        "hourly_signal": get_signal(scores['hourly']),
        "graph_file": graph_file or "Failed to generate graph."
    }

def _generate_futures_tickers(symbol_root: str, num: int, cycle: str, exchange: str) -> List[str]:
    """Generates a list of valid yfinance-compatible futures tickers."""
    month_map = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M', 7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
    quarterly = [3, 6, 9, 12]
    tickers, current_date, offset = [], datetime.now(), 0
    while len(tickers) < num and offset < 24:
        target_date = current_date + relativedelta(months=offset)
        month, year = target_date.month, str(target_date.year)[-2:]
        if cycle == 'monthly' or (cycle == 'quarterly' and month in quarterly):
            ticker = f"{symbol_root}{month_map[month]}{year}.{exchange}"
            if ticker not in tickers:
                tickers.append(ticker)
        offset += 1
    return tickers

async def analyze_term_structure(symbol_root: str) -> Optional[Dict[str, Any]]:
    """Fetches data for multiple contract months to analyze the term structure."""
    specs = get_futures_specs().get(symbol_root)
    if not specs or not all(k in specs for k in ['ticker', 'cycle', 'exchange']):
        return {"error": f"Incomplete contract specs for {symbol_root}."}

    yf_root = specs['ticker'].replace('=F', '')
    contract_tickers = _generate_futures_tickers(yf_root, 4, specs['cycle'], specs['exchange'])
    if not contract_tickers: return {"error": "Could not generate valid contract tickers."}
    
    data = await get_yf_download_robustly(contract_tickers, period="5d", group_by='ticker')
    if data.empty: return None

    contracts_data, prices = [], []
    for ticker in contract_tickers:
        if isinstance(data.columns, pd.MultiIndex) and (ticker, 'Close') in data.columns:
            price = data[(ticker, 'Close')].dropna().iloc[-1]
            contracts_data.append({"contract": ticker, "price": price}); prices.append(price)

    if len(prices) < 2: return {"error": "Not enough valid contract data fetched."}
    
    p1, p2 = prices[0], prices[1]
    state = "Contango" if p2 > p1 else "Backwardation"
    slope = ((p2 - p1) / p1) * 100
    sentiment = 100 - np.clip(50 - (slope * 15), 0, 100)
    return {"contracts": contracts_data, "state": state, "sentiment_score": sentiment}

# --- Main Command Handler ---

async def handle_futures_command(args: List[str]):
    """Main handler for the /futures command."""
    if len(args) < 2:
        print("Usage: /futures <info|analyze|termstructure> <symbol>")
        return

    subcommand, symbol_root = args[0].lower(), args[1].upper().replace('/', '')

    if subcommand == 'info':
        info = await get_futures_info(symbol_root)
        if info:
            print(tabulate(info.items(), headers=["Metric", "Value"], tablefmt="pretty"))
        else:
            print(f"❌ Could not retrieve info for '{symbol_root}'.")
    
    elif subcommand == 'analyze':
        results = await analyze_futures_contract(symbol_root)
        if results:
            table = [["Weekly", results['weekly_score'], results['weekly_signal']], ["Daily", results['daily_score'], results['daily_signal']], ["Hourly", results['hourly_score'], results['hourly_signal']]]
            print(tabulate(table, headers=["Timeframe", "INVEST Score", "Signal"], tablefmt="pretty"))
            print(f"\n📂 Daily Chart Generated: {results['graph_file']}")
        else:
            print(f"❌ Could not perform analysis for '{symbol_root}'.")

    elif subcommand == 'termstructure':
        results = await analyze_term_structure(symbol_root)
        if results:
            if "error" in results: print(f"❌ {results['error']}")
            else:
                print(tabulate(results['contracts'], headers="keys", tablefmt="pretty"))
                print(f"\nMarket State: {results['state']}")
                print(f"Sentiment Score: {results['sentiment_score']:.2f} / 100")
        else:
            print(f"❌ Could not analyze term structure for '{symbol_root}'.")
    else:
        print(f"Error: Unknown subcommand '{subcommand}'.")