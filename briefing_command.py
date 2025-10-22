# --- Imports for briefing_command ---
import asyncio
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta 
from io import StringIO
import pytz
from bs4 import BeautifulSoup
import requests

import yfinance as yf
import pandas as pd
from tabulate import tabulate
import humanize

# --- Imports from other command modules ---
from risk_command import perform_risk_calculations_singularity
from breakout_command import run_breakout_analysis_singularity

# --- Helper Functions ---

EST_TIMEZONE = pytz.timezone('US/Eastern')
USER_PREFERENCES_FILE = 'user_preferences.json'

def load_user_preferences() -> Dict[str, Any]:
    if not os.path.exists(USER_PREFERENCES_FILE): return {}
    try:
        with open(USER_PREFERENCES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError): return {}
    
async def get_daily_change_for_tickers(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    if not tickers: return {}
    results = {}
    
    try:
        # Attempt to download all ticker data in a single batch
        data = await asyncio.to_thread(
            yf.download,
            tickers=tickers,
            period="5d",
            interval="1d",
            progress=False,
            auto_adjust=True,
            timeout=30
        )
        if data.empty:
            return {ticker: {'error': 'No data returned'} for ticker in tickers}
    except Exception as e:
        # If the entire download fails, return an error for every requested ticker
        return {ticker: {'error': str(e)} for ticker in tickers}

    close_prices = data.get('Close')
    # Check if the result is a DataFrame (multi-ticker) or Series (single-ticker)
    is_dataframe = isinstance(close_prices, pd.DataFrame)

    # Process each ticker individually to prevent one failure from affecting others
    for ticker in tickers:
        try:
            series = close_prices[ticker] if is_dataframe else close_prices
            
            if series is not None and not series.empty:
                valid_closes = series.dropna()
                if len(valid_closes) > 1:
                    live_price, prev_close = valid_closes.iloc[-1], valid_closes.iloc[-2]
                    # Ensure data is valid for calculation
                    if pd.notna(live_price) and pd.notna(prev_close) and prev_close != 0:
                        results[ticker] = {'live_price': live_price, 'change_pct': ((live_price - prev_close) / prev_close) * 100}
                        continue # Skip to the next ticker on success
            
            results[ticker] = {'error': 'Insufficient data'}
        except (KeyError, IndexError, TypeError):
            # If an error occurs for this specific ticker, assign an error and continue
            results[ticker] = {'error': 'Data processing failed'}
            
    return results

async def get_multi_period_change(tickers: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    if not tickers: return {}
    results = {}
    end_date = datetime.now()
    start_date_1m = end_date - timedelta(days=40)
    try:
        data = await asyncio.to_thread(yf.download, tickers=tickers, start=start_date_1m.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), progress=False, auto_adjust=True, timeout=30)
        if data.empty: return {t: {'1D': None, '1W': None, '1M': None} for t in tickers}
        
        close_data = data.get('Close')
        for ticker in tickers:
            series = close_data if isinstance(close_data, pd.Series) else close_data.get(ticker)
            if series is not None:
                series = series.dropna()
                d1 = ((series.iloc[-1] - series.iloc[-2]) / series.iloc[-2]) * 100 if len(series) >= 2 and series.iloc[-2] != 0 else None
                w1 = ((series.iloc[-1] - series.iloc[-6]) / series.iloc[-6]) * 100 if len(series) >= 6 and series.iloc[-6] != 0 else None
                m1 = ((series.iloc[-1] - series.iloc[-22]) / series.iloc[-22]) * 100 if len(series) >= 22 and series.iloc[-22] != 0 else None
                results[ticker] = {'1D': d1, '1W': w1, '1M': m1}
            else:
                results[ticker] = {'1D': None, '1W': None, '1M': None}
    except Exception:
        results = {t: {'1D': None, '1W': None, '1M': None} for t in tickers}
    return results

def get_sp500_symbols_singularity(is_called_by_ai: bool = False) -> List[str]:
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        df = pd.read_html(StringIO(response.text))[0]
        if 'Symbol' not in df.columns: return []
        symbols = [str(s).replace('.', '-') for s in df['Symbol'].tolist() if isinstance(s, str)]
        return sorted(list(set(s for s in symbols if s)))
    except Exception: return []

async def get_sp500_movers(is_called_by_ai: bool = False) -> Dict[str, List[Dict]]:
    if not is_called_by_ai: print("  Briefing: Fetching S&P 500 movers...")
    sp500_symbols = await asyncio.to_thread(get_sp500_symbols_singularity, is_called_by_ai=True)
    if not sp500_symbols: return {'top': [], 'bottom': [], 'error': 'Could not fetch S&P 500 symbol list.'}
    
    all_changes, chunk_size = {}, 75
    for i in range(0, len(sp500_symbols), chunk_size):
        chunk = sp500_symbols[i:i + chunk_size]
        print(f"[BRIEFING_DEBUG] Fetching S&P movers chunk {i//chunk_size + 1}/{(len(sp500_symbols) + chunk_size - 1)//chunk_size}...")
        try:
            chunk_changes = await asyncio.wait_for(get_daily_change_for_tickers(chunk), timeout=60.0)
            all_changes.update(chunk_changes)
        except asyncio.TimeoutError:
            print(f"[BRIEFING_DEBUG]   ! Chunk timed out. Skipping this chunk.")
        except Exception as e:
            print(f"[BRIEFING_DEBUG]   ! An error occurred on this chunk: {e}")
        await asyncio.sleep(1)
        
    if not all_changes: return {'top': [], 'bottom': [], 'error': 'Failed to fetch any S&P 500 price data.'}
    valid_performers = [{'ticker': t, **d} for t, d in all_changes.items() if 'change_pct' in d and pd.notna(d['change_pct'])]
    if not valid_performers: return {'top': [], 'bottom': [], 'error': 'Could not calculate daily changes.'}
    valid_performers.sort(key=lambda x: x['change_pct'], reverse=True)
    return {'top': valid_performers[:3], 'bottom': list(reversed(valid_performers[-3:]))}

async def _scrape_cnbc_quote(url: str) -> Optional[float]:
    """A helper to scrape the 'last price' from a CNBC quote page."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        price_span = soup.find('span', class_='QuoteStrip-lastPrice')
        if price_span:
            price_text = price_span.text.strip().replace('%', '').replace(',', '')
            return float(price_text)
    except Exception as e:
        print(f"  [BRIEFING_DEBUG] Failed to scrape {url}: {e}")
    return None

async def get_treasury_yield_data() -> Dict[str, Optional[float]]:
    """
    Fetches key Treasury yields by scraping CNBC for maximum reliability.
    """
    # Define CNBC URLs for the yields
    urls = {
        '10Y': 'https://www.cnbc.com/quotes/US10Y',
        '2Y': 'https://www.cnbc.com/quotes/US2Y'
    }

    # Asynchronously fetch both yields using the scraper
    y10_val, y2_val = await asyncio.gather(
        _scrape_cnbc_quote(urls['10Y']),
        _scrape_cnbc_quote(urls['2Y'])
    )

    return {'10Y': y10_val, '2Y': y2_val}

async def get_major_futures_data() -> Dict[str, Dict[str, Any]]:
    """
    Fetches prices for major futures, using a hybrid of scraping (for Oil's live price)
    and yfinance (for history and other tickers) to ensure reliability.
    """
    results = {}
    
    # --- Part 1: Get Crude Oil data using a hybrid approach ---
    try:
        # Scrape CNBC for the most reliable live price
        oil_url = 'https://www.cnbc.com/quotes/@CL.1'
        scraped_price = await _scrape_cnbc_quote(oil_url)

        # Use yfinance to get recent history for the % change calculation
        oil_hist = await asyncio.to_thread(yf.download, tickers=['CL=F'], period="5d", progress=False)
        
        if not oil_hist.empty and 'Close' in oil_hist:
            close_data = oil_hist['Close']
            series = None

            # THE FIX: Robustly handle if yfinance returns a DataFrame or a Series
            if isinstance(close_data, pd.DataFrame):
                if 'CL=F' in close_data.columns:
                    series = close_data['CL=F'].dropna()
            elif isinstance(close_data, pd.Series):
                series = close_data.dropna()

            if series is not None and len(series) > 1:
                # Now we are guaranteed 'series' is a pd.Series, so .iloc returns a float
                yf_live_price = series.iloc[-1]
                prev_close = series.iloc[-2]
                
                # Prioritize the more accurate scraped price, but use yfinance as a fallback
                final_live_price = scraped_price if scraped_price is not None else yf_live_price
                
                change_pct = ((final_live_price - prev_close) / prev_close) * 100
                results['CL=F'] = {'live_price': final_live_price, 'change_pct': change_pct}
            
            elif scraped_price is not None:
                # Fallback if history fails but scraping works (no % change available)
                results['CL=F'] = {'live_price': scraped_price}
            
    except Exception as e:
        print(f"  [BRIEFING_DEBUG] Failed to fetch Crude Oil data: {e}")

    # --- Part 2: Fetch other futures using the standard method ---
    other_tickers = {'Gold': 'GC=F', 'Nasdaq': 'NQ=F', 'Bitcoin': 'BTC-USD'}
    other_data = await get_daily_change_for_tickers(list(other_tickers.values()))
    
    # Combine all results, giving priority to the new oil data
    results.update(other_data)
    return results

async def get_specific_economic_data() -> dict[str, str]:
    """
    Fetches key economic data by scraping Trading Economics and getting DXY from yfinance.
    """
    # Initialize dictionary with default "N/A" values
    data = {
        "Dollar (1Y)": "N/A",
        "GDP Growth Rate": "N/A",
        "Unemployment Rate": "N/A",
        "Interest Rate": "N/A",
        "Govt Debt to GDP": "N/A",
        "Business Confidence": "N/A",
        "Consumer Confidence": "N/A"
    }

    # --- Part 1: Fetch Dollar Index (DXY) from yfinance ---
    try:
        dxy = await asyncio.to_thread(yf.Ticker("DX-Y.NYB").history, period="1y")
        # Ensure a full year of trading data is available for an accurate calculation
        if not dxy.empty and len(dxy) > 250:
            yoy_change = ((dxy['Close'].iloc[-1] - dxy['Close'].iloc[0]) / dxy['Close'].iloc[0]) * 100
            six_month_change = ((dxy['Close'].iloc[-1] - dxy['Close'].iloc[-126]) / dxy['Close'].iloc[-126]) * 100
            data["Dollar (1Y)"] = f"{yoy_change:+.2f}% (6M: {six_month_change:+.2f}%)"
    except Exception as e:
        print(f"  [BRIEFING_DEBUG] Failed to fetch DXY data: {e}")

    # --- Part 2: Scrape Trading Economics ---
    try:
        url = "https://tradingeconomics.com/united-states/indicators"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        
        response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=20, verify=False)
        response.raise_for_status() # Raise an error for bad responses (like 404 or 500)

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Map the exact text from the website to the desired key in our 'data' dictionary
        indicators_to_scrape = {
            "GDP Growth Rate": "GDP Growth Rate",
            "Unemployment Rate": "Unemployment Rate",
            "Interest Rate": "Interest Rate",
            "Government Debt to GDP": "Govt Debt to GDP",
            "Business Confidence": "Business Confidence",
            "Consumer Confidence": "Consumer Confidence"
        }

        for row in soup.find_all('tr'):
            cells = row.find_all('td')
            if not cells or len(cells) < 6:
                continue
            
            indicator_name = cells[0].text.strip()
            
            if indicator_name in indicators_to_scrape:
                # Extract the Last, Previous, and Unit values from the correct cells
                actual_val, prev_val = cells[1].text.strip(), cells[2].text.strip()
                unit = cells[5].text.strip().lower()
                
                # Append the correct unit symbol (%) to the values
                suffix = "%" if "percent" in unit else ""
                
                data_key = indicators_to_scrape[indicator_name]
                data[data_key] = f"{actual_val}{suffix} (Prev: {prev_val}{suffix})"
                
    except Exception as e:
        print(f"  [BRIEFING_DEBUG] Failed to scrape Trading Economics: {e}")

    return data

async def handle_briefing_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    if not is_called_by_ai: print("\n--- Generating Daily Market Briefing ---")
    watchlist_tickers = load_user_preferences().get('favorite_tickers', [])
    
    # --- Step 1: Market, R.I.S.K., and Macro Data ---
    if not is_called_by_ai: print("➪ Briefing Step 1/4: Calculating Market, R.I.S.K., and Macro Data...")
    tasks1 = {
        "spy_vix": get_daily_change_for_tickers(['SPY', '^VIX']), "risk": perform_risk_calculations_singularity(is_called_by_ai=True),
        "yields": get_treasury_yield_data(), "futures": get_major_futures_data(), "economic": get_specific_economic_data()
    }
    results1 = await asyncio.gather(*tasks1.values(), return_exceptions=True)
    results_dict = dict(zip(tasks1.keys(), results1))

    # --- Steps 2 & 3: S&P Movers, Breakouts, and Watchlist ---
    if not is_called_by_ai: print("➪ Briefing Step 2/4: Analyzing S&P 500 Movers...")
    results_dict['sp500_movers'] = await get_sp500_movers(is_called_by_ai=True)

    if not is_called_by_ai: print("➪ Briefing Step 3/4: Running Breakout Analysis and Checking Watchlist...")
    tasks3 = {"breakouts": run_breakout_analysis_singularity(is_called_by_ai=True)}
    if watchlist_tickers:
        tasks3["watchlist"] = get_daily_change_for_tickers(watchlist_tickers)
    results3 = await asyncio.gather(*tasks3.values(), return_exceptions=True)
    results_dict.update(dict(zip(tasks3.keys(), results3)))
    
    if not is_called_by_ai: print("➪ Briefing Step 4/4: Compiling Final Report...")
    
    # Process results, gracefully handling potential errors from gather
    def get_result(key):
        res = results_dict.get(key)
        return res if not isinstance(res, Exception) else {}

    spy_vix_data = get_result('spy_vix')
    spy_data, vix_data = spy_vix_data.get('SPY', {}), spy_vix_data.get('^VIX', {})
    risk_results, yield_results = get_result('risk'), get_result('yields')
    futures_results, sp_movers = get_result('futures'), get_result('sp500_movers')
    breakout_results = get_result('breakouts')
    
    top_3_breakout_tickers = [item.get('Ticker') for item in breakout_results.get('current_breakout_stocks', [])[:3]]
    top_3_perf_data = await get_multi_period_change(top_3_breakout_tickers)

    # --- Final Report Printout ---
    print("\n" + "="*60)
    print("DAILY MARKET BRIEFING")
    print(f"{datetime.now(EST_TIMEZONE).strftime('%B %d, %Y - %I:%M %p EST')}")
    print("="*60)

    spy_p, spy_c = (f"${spy_data.get('live_price'):.2f}", f"({spy_data.get('change_pct'):+.2f}%)") if 'live_price' in spy_data else ("N/A", "")
    vix_p, vix_c = (f"{vix_data.get('live_price'):.2f}", f"({vix_data.get('change_pct'):+.2f}%)") if 'live_price' in vix_data else ("N/A", "")
    print(f"S&P 500: {spy_p} {spy_c} | VIX: {vix_p} {vix_c}")
    
    oil, gold = futures_results.get('CL=F', {}), futures_results.get('GC=F', {})
    oil_p, oil_c = (f"${oil.get('live_price'):.2f}", f"({oil.get('change_pct'):+.2f}%)") if 'live_price' in oil else ("N/A", "")
    gold_p, gold_c = (f"${gold.get('live_price'):.2f}", f"({gold.get('change_pct'):+.2f}%)") if 'live_price' in gold else ("N/A", "")
    print(f"Crude Oil: {oil_p} {oil_c} | Gold: {gold_p} {gold_c}")

    y10_val, y2_val = yield_results.get('10Y'), yield_results.get('2Y')
    y10 = f"{y10_val:.3f}%" if y10_val is not None else "N/A"
    y2 = f"{y2_val:.3f}%" if y2_val is not None else "N/A"
    print(f"US Treasury Yields -> 10Y: {y10} | 2Y: {y2}")
    
    print(f"\nRISK Scores -> General: {risk_results.get('general_score', 'N/A')} | Market Invest: {risk_results.get('market_invest_score', 'N/A')}")
    
    print("\n--- S&P 500 Movers ---")
    top_sp = ", ".join([f"{t['ticker']} ({t.get('change_pct', 0):+.2f}%)" for t in sp_movers.get('top', [])])
    bottom_sp = ", ".join([f"{t['ticker']} ({t.get('change_pct', 0):+.2f}%)" for t in sp_movers.get('bottom', [])])
    print(f"  Top 3: {top_sp if top_sp else 'N/A'}")
    print(f"  Bottom 3: {bottom_sp if bottom_sp else 'N/A'}")

    print("\n--- Breakout Analysis ---")
    top_3_stocks = breakout_results.get('current_breakout_stocks', [])[:3]
    if top_3_stocks:
        for stock in top_3_stocks:
            ticker = stock.get('Ticker', 'N/A')
            perf = top_3_perf_data.get(ticker, {})
            d1 = f"{perf.get('1D'):+.2f}%" if perf.get('1D') is not None else 'N/A'
            w1 = f"{perf.get('1W'):+.2f}%" if perf.get('1W') is not None else 'N/A'
            m1 = f"{perf.get('1M'):+.2f}%" if perf.get('1M') is not None else 'N/A'
            print(f"    - {ticker} (Score: {stock.get('Invest Score', 'N/A')}) | 1D: {d1}, 1W: {w1}, 1M: {m1}")
    else:
        print("    No current breakout stocks.")
    
    # --- NEW: Economic Snapshot Table ---
    print("\n--- Economic Snapshot ---")
    econ_results = get_result('economic')
    if isinstance(econ_results, dict) and econ_results:
        print(tabulate(econ_results.items(), headers=["Indicator", "Latest Value"], tablefmt="pretty"))
    else:
        print("  Economic data not available.")
        
    watchlist_data = results_dict.get('watchlist')
    if isinstance(watchlist_data, dict):
        valid_watchlist = [{'ticker': t, **d} for t, d in watchlist_data.items() if 'change_pct' in d and pd.notna(d['change_pct'])]
        if valid_watchlist:
            valid_watchlist.sort(key=lambda x: x['change_pct'], reverse=True)
            print("\n--- Your Watchlist Movers ---")
            top_w = ", ".join([f"{t['ticker']} ({t['change_pct']:.2f}%)" for t in valid_watchlist[:3]])
            bottom_w = ", ".join([f"{t['ticker']} ({t['change_pct']:.2f}%)" for t in list(reversed(valid_watchlist[-3:]))])
            print(f"  Top 3: {top_w if top_w else 'N/A'}")
            print(f"  Bottom 3: {bottom_w if bottom_w else 'N/A'}")
            
    print("\n" + "="*60)