# --- Imports for risk_command ---
import asyncio
import os
import csv
import logging
import uuid
from typing import Optional, List, Dict, Any
from io import StringIO
from datetime import datetime, timedelta
import tabulate

import yfinance as yf
import pandas as pd
import numpy as np
import pytz
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore

# --- Global Constants and Configuration ---
EST_TIMEZONE = pytz.timezone('US/Eastern')
RISK_CSV_FILE = 'market_data.csv'
RISK_EOD_CSV_FILE = 'risk_eod_data.csv'
RISK_LOG_FILE = 'risk_calculations.log'
SP500_CACHE_FILE = 'sp500_risk_cache.csv'
SP100_CACHE_FILE = 'sp100_risk_cache.csv'

# --- Logging Setup ---
risk_logger = logging.getLogger('RISK_MODULE_EXTERNAL')
risk_logger.setLevel(logging.INFO)
if not risk_logger.hasHandlers():
    risk_file_handler = logging.FileHandler(RISK_LOG_FILE)
    risk_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
    risk_file_handler.setFormatter(risk_formatter)
    risk_logger.addHandler(risk_file_handler)

# --- NEW: Download Helper with Timeout and Retry Logic ---
async def _download_with_retry(tickers: List[str], timeout: int, **kwargs) -> pd.DataFrame:
    """Wraps yf.download with a strict timeout and a retry loop."""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Create the blocking yfinance task
            download_task = asyncio.to_thread(yf.download, tickers=tickers, progress=False, **kwargs)
            
            # Wrap the task in asyncio.wait_for to enforce the timeout
            data = await asyncio.wait_for(download_task, timeout=timeout)
            return data # Success
            
        except asyncio.TimeoutError:
            print(f"[RISK_DEBUG]     ! Download for {len(tickers)} tickers timed out (attempt {attempt + 1}/{max_retries})...")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 * (attempt + 1)) # Exponential backoff (2s, 4s)
            else:
                print(f"[RISK_DEBUG]     ! All retry attempts failed for this batch.")
        except Exception as e:
            print(f"[RISK_DEBUG]     ! An unexpected download error occurred: {e}")
            break # Do not retry on other errors
    
    return pd.DataFrame() # Return empty DataFrame on failure


# --- Helper Functions ---

def get_sp500_symbols_singularity() -> List[str]:
    print("[RISK_DEBUG] Fetching S&P 500 symbols...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        df = pd.read_html(StringIO(response.text))[0]
        symbols = [str(s).replace('.', '-') for s in df['Symbol'].tolist() if isinstance(s, str)]
        print(f"[RISK_DEBUG] Successfully fetched {len(symbols)} S&P 500 symbols.")
        return sorted(list(set(symbols)))
    except Exception as e:
        print(f"[RISK_DEBUG] FAILED to fetch S&P 500 symbols: {e}")
        return []

def get_sp100_symbols_risk() -> list:
    print("[RISK_DEBUG] Fetching S&P 100 symbols...")
    try:
        url = 'https://en.wikipedia.org/wiki/S%26P_100'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        df = pd.read_html(StringIO(response.text))[2]
        symbols = df['Symbol'].tolist()
        print(f"[RISK_DEBUG] Successfully fetched {len(symbols)} S&P 100 symbols.")
        return [s.replace('.', '-') for s in symbols if isinstance(s, str)]
    except Exception as e:
        print(f"[RISK_DEBUG] FAILED to fetch S&P 100 symbols: {e}")
        return []

async def fetch_and_cache_data(symbols: List[str], cache_filename: str, period: str):
    print(f"[RISK_DEBUG] Managing cache for {cache_filename}...")
    
    cached_tickers = set()
    if os.path.exists(cache_filename):
        print(f"[RISK_DEBUG] Reading existing cache file: {cache_filename}")
        cached_df = pd.read_csv(cache_filename, header=[0, 1], index_col=0, parse_dates=True)
        cached_tickers = set(cached_df.columns.get_level_values(1))
        
        last_date = cached_df.index.max()
        if last_date.date() >= datetime.now().date() - timedelta(days=1) and len(cached_tickers) >= len(symbols):
            print("[RISK_DEBUG] Cache is complete and up to date. Skipping download.")
            return cached_df

    symbols_to_download = [s for s in symbols if s not in cached_tickers]
    
    if symbols_to_download:
        print(f"[RISK_DEBUG] Cache incomplete. Needing to download {len(symbols_to_download)} tickers...")
        chunk_size = 50
        for i in range(0, len(symbols_to_download), chunk_size):
            chunk = symbols_to_download[i:i + chunk_size]
            print(f"[RISK_DEBUG]   Downloading and caching chunk {i//chunk_size + 1}/{(len(symbols_to_download) + chunk_size - 1)//chunk_size}...")
            
            chunk_data = await _download_with_retry(tickers=chunk, timeout=60, period=period)
            
            if chunk_data.empty:
                print(f"[RISK_DEBUG]     Chunk failed after all retries. Moving to next chunk.")
                continue

            if os.path.exists(cache_filename):
                existing_data = pd.read_csv(cache_filename, header=[0, 1], index_col=0, parse_dates=True)
                combined_data = pd.concat([existing_data, chunk_data], axis=1)
                combined_data = combined_data.loc[:, ~combined_data.columns.duplicated()]
                combined_data.to_csv(cache_filename)
            else:
                chunk_data.to_csv(cache_filename)
            print(f"[RISK_DEBUG]     Successfully saved chunk to {cache_filename}.")
            await asyncio.sleep(1)

    if os.path.exists(cache_filename):
        print(f"[RISK_DEBUG] Loading final data from {cache_filename}.")
        return pd.read_csv(cache_filename, header=[0, 1], index_col=0, parse_dates=True)
    else:
        return pd.DataFrame()

def _calculate_ma_percentage_from_data(symbols: List[str], data: pd.DataFrame, ma_window: int) -> float:
    print(f"[RISK_DEBUG] Calculating MA% for {len(symbols)} symbols, MA Window: {ma_window}")
    if not symbols or data.empty:
        print("[RISK_DEBUG] MA% calculation skipped: No symbols or data.")
        return 0.0
    
    above_ma_count, valid_stocks_count = 0, 0
    
    # --- START: Robust Data Extraction Logic ---
    try:
        # FIX: The slice must be ['Close', :] to select the 'Close' metric from the top level.
        close_prices_df = data.loc[:, pd.IndexSlice['Close', :]]
        # Drop the top-level column name ('Close') to get columns with just ticker names
        close_prices_df.columns = close_prices_df.columns.droplevel(0)
    except KeyError:
        print("[RISK_DEBUG]   ! 'Close' column not found in downloaded data. Cannot calculate MA%.")
        return 0.0
    # --- END: Robust Data Extraction Logic ---

    for symbol in symbols:
        if symbol in close_prices_df.columns:
            close_prices = close_prices_df[symbol].dropna()
            if len(close_prices) >= ma_window:
                valid_stocks_count += 1
                ma = close_prices.rolling(window=ma_window).mean().iloc[-1]
                if pd.notna(ma) and close_prices.iloc[-1] > ma:
                    above_ma_count += 1
    
    result = (above_ma_count / valid_stocks_count) * 100 if valid_stocks_count > 0 else 0.0
    print(f"[RISK_DEBUG] MA% calculation complete. Result: {result:.2f}%")
    return result

async def get_live_price_and_ma_risk(ticker: str, ma_windows: List[int], is_called_by_ai: bool = False) -> tuple[Optional[float], Dict[int, Optional[float]]]:
    print(f"[RISK_DEBUG] Fetching live price & MAs for {ticker}...")
    ma_values = {ma: None for ma in ma_windows}
    try:
        period = '2y' if max(ma_windows, default=0) >= 200 else '1y'
        hist = await asyncio.to_thread(yf.Ticker(ticker).history, period=period)
        if hist.empty:
            print(f"[RISK_DEBUG] FAILED for {ticker}: No history returned.")
            return None, ma_values
        
        price = hist['Close'].iloc[-1]
        for window in ma_windows:
            if len(hist) >= window:
                ma_values[window] = hist['Close'].rolling(window=window).mean().iloc[-1]
        print(f"[RISK_DEBUG] Success for {ticker}. Price: {price:.2f}")
        return price, ma_values
    except Exception as e:
        print(f"[RISK_DEBUG] FAILED for {ticker}: {e}")
        return None, ma_values

def calculate_ema_score_risk(ticker: str ="SPY", is_called_by_ai: bool = False) -> Optional[float]:
    print(f"[RISK_DEBUG] Calculating EMA score for {ticker}...")
    try:
        data = yf.Ticker(ticker).history(period="1y")
        if data.empty or len(data) < 55: return None
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()
        ema_8, ema_55 = data['EMA_8'].iloc[-1], data['EMA_55'].iloc[-1]
        if pd.isna(ema_55) or ema_55 == 0: return None
        score = (((ema_8 - ema_55) / ema_55) * 5 + 0.5) * 100
        print(f"[RISK_DEBUG] EMA score for {ticker} is {score:.2f}")
        return float(np.clip(score, 0, 100))
    except Exception as e:
        print(f"[RISK_DEBUG] FAILED EMA score calculation for {ticker}: {e}")
        return None

async def calculate_risk_scores_singularity(is_called_by_ai: bool = False) -> tuple:
    print("[RISK_DEBUG] --- Starting calculate_risk_scores_singularity ---")
    
    sp500_symbols = get_sp500_symbols_singularity()
    sp100_symbols = get_sp100_symbols_risk()
    
    sp500_data = await fetch_and_cache_data(sp500_symbols, SP500_CACHE_FILE, '2y')
    sp100_data = await fetch_and_cache_data(sp100_symbols, SP100_CACHE_FILE, '6mo')
    
    s5tw_val = _calculate_ma_percentage_from_data(sp500_symbols, sp500_data, 20)
    s5th_val = _calculate_ma_percentage_from_data(sp500_symbols, sp500_data, 200)
    s1fd_val = _calculate_ma_percentage_from_data(sp100_symbols, sp100_data, 5)
    s1tw_val = _calculate_ma_percentage_from_data(sp100_symbols, sp100_data, 20)
    
    print("[RISK_DEBUG] Fetching individual index MAs...")
    (spy_live_price, spy_mas), (vix_live_price, _), (rut_live_price, rut_mas), (oex_live_price, oex_mas) = await asyncio.gather(
        get_live_price_and_ma_risk('SPY', [20, 50], is_called_by_ai=True),
        get_live_price_and_ma_risk('^VIX', [], is_called_by_ai=True),
        get_live_price_and_ma_risk('^RUT', [20, 50], is_called_by_ai=True),
        get_live_price_and_ma_risk('^OEX', [20, 50], is_called_by_ai=True)
    )
    print("[RISK_DEBUG] Individual index MAs fetched.")
    
    critical_data_map = {'SPY Price': spy_live_price, 'VIX Price': vix_live_price}
    if any(v is None for v in critical_data_map.values()):
        print("[RISK_DEBUG] FAILED: Critical data (SPY or VIX price) is missing.")
        return None, None, None, None, spy_live_price, vix_live_price
        
    try:
        print("[RISK_DEBUG] Calculating component scores...")
        spy20 = np.clip(((spy_live_price - spy_mas.get(20, spy_live_price)) / 20) + 50, 0, 100)
        spy50 = np.clip(((spy_live_price - spy_mas.get(50, spy_live_price) - 150) / 20) + 50, 0, 100)
        vix_score = np.clip((((vix_live_price - 15) * -5) + 50), 0, 100)
        rut20 = np.clip(((rut_live_price - rut_mas.get(20, rut_live_price)) / 10) + 50, 0, 100)
        rut50 = np.clip(((rut_live_price - rut_mas.get(50, rut_live_price)) / 5) + 50, 0, 100)
        s5tw_score = np.clip(((s5tw_val - 60) + 50), 0, 100)
        s5th_score = np.clip(((s5th_val - 70) + 50), 0, 100)
        oex20_score = np.clip(((oex_live_price - oex_mas.get(20, oex_live_price)) / 100) + 50, 0, 100)
        oex50_score = np.clip(((oex_live_price - oex_mas.get(50, oex_live_price) - 25) / 100) + 50, 0, 100)
        s1fd_score = np.clip(((s1fd_val - 60) + 50), 0, 100)
        s1tw_score = np.clip(((s1tw_val - 70) + 50), 0, 100)
        print("[RISK_DEBUG] Component scores calculated.")
    except (TypeError, KeyError):
        print("[RISK_DEBUG] FAILED: Type or Key error during component score calculation.")
        return None, None, None, None, spy_live_price, vix_live_price
        
    ema_score_val_risk = await asyncio.to_thread(calculate_ema_score_risk, "SPY", is_called_by_ai=True)
    if ema_score_val_risk is None: ema_score_val_risk = 50.0
    
    print("[RISK_DEBUG] Calculating final weighted scores...")
    general_score = np.clip(((3*spy20)+spy50+(3*vix_score)+(3*rut50)+rut20+(2*s5tw_score)+s5th_score)/13.0, 0, 100)
    large_cap_score = np.clip(((3*oex20_score)+oex50_score+(2*s1fd_score)+s1tw_score)/7.0, 0, 100)
    combined_score = np.clip((general_score + large_cap_score + ema_score_val_risk) / 3.0, 0, 100)
    print(f"[RISK_DEBUG] Final scores: General={general_score:.2f}, LargeCap={large_cap_score:.2f}, Combined={combined_score:.2f}")
    
    return general_score, large_cap_score, ema_score_val_risk, combined_score, spy_live_price, vix_live_price

def calculate_recession_likelihood_ema_risk(ticker:str ="SPY", interval:str ="1mo", period:str ="5y", is_called_by_ai: bool = False) -> Optional[float]:
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        if data.empty or len(data) < 55: return None
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()
        ema_8, ema_55 = data['EMA_8'].iloc[-1], data['EMA_55'].iloc[-1]
        if pd.isna(ema_8) or pd.isna(ema_55) or ema_55 == 0: return None
        x_value = (((ema_8 - ema_55) / ema_55) + 0.5) * 100
        likelihood = 100 * np.exp(-((45.622216 * x_value / 2750) ** 4))
        return float(np.clip(likelihood, 0, 100))
    except Exception:
        return None

def calculate_recession_likelihood_vix_risk(vix_price: Optional[float], is_called_by_ai: bool = False) -> Optional[float]:
    if vix_price is not None and not pd.isna(vix_price):
        try:
            likelihood = 0.01384083 * (float(vix_price) ** 2)
            return float(np.clip(likelihood, 0, 100))
        except ValueError:
            return None
    return None

def calculate_market_invest_score_risk(vix_contraction_chance: Optional[float], ema_contraction_chance: Optional[float], is_called_by_ai: bool = False) -> tuple:
    if vix_contraction_chance is None or ema_contraction_chance is None: return None, None, None
    try:
        ratio = vix_contraction_chance / ema_contraction_chance if ema_contraction_chance != 0 else float('inf')
        uncapped_score_mis = 50.0 - (ratio - 1.0) * 100.0
        capped_score_for_signal_mis = float(np.clip(uncapped_score_mis, 0, 100))
        rounded_capped_score_for_display_mis = int(round(capped_score_for_signal_mis))
        return uncapped_score_mis, capped_score_for_signal_mis, rounded_capped_score_for_display_mis
    except Exception:
        return None, None, None

# --- Add this new helper function to risk_command.py ---
def _save_risk_data_to_csv(data_row: Dict[str, Any]):
    """Appends a row of risk data to the market_data.csv file."""
    market_data_csv_file = 'market_data.csv'
    file_exists = os.path.isfile(market_data_csv_file)
    
    header = [
        "Timestamp", "General Market Score", "Large Market Cap Score", "EMA Score", 
        "Combined Score", "Live SPY Price", "Live VIX Price", "Momentum Based Recession Chance", 
        "VIX Based Recession Chance", "Raw Market Invest Score", "Market Score Percentile", 
        "Market IV", "Market IVR"
    ]
    
    try:
        with open(market_data_csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            if not file_exists:
                writer.writeheader()
            writer.writerow(data_row)
        risk_logger.info(f"Successfully saved risk data to {market_data_csv_file}")
    except Exception as e:
        risk_logger.error(f"Failed to save risk data to CSV: {e}")

def calculate_market_score_percentile_risk(capped_mis_signal: Optional[float], is_called_by_ai: bool = False) -> Optional[float]:
    """Calculates the percentile of the current market score against historical data."""
    if capped_mis_signal is None: return None
    try:
        if not os.path.exists(RISK_CSV_FILE): return 50.0
        df = pd.read_csv(RISK_CSV_FILE)
        
        if 'Raw Market Invest Score' not in df.columns or df['Raw Market Invest Score'].isnull().all():
            return 50.0

        historical_scores = df['Raw Market Invest Score'].dropna()
        if len(historical_scores) < 10: return 50.0
        
        percentile = percentileofscore(historical_scores, capped_mis_signal, kind='rank')
        return float(percentile)
    except Exception as e:
        risk_logger.warning(f"Percentile Calculation Error: {e}")
        return None

def calculate_iv_and_ivr_risk(is_called_by_ai: bool = False) -> tuple[Optional[float], Optional[float]]:
    """
    Calculates the current market IV and its IV Rank based on historical data
    from the main market_data.csv file.
    """
    try:
        # Step 1: Fetch current implied volatility from live SPY options
        spy_options = yf.Ticker('SPY').option_chain()
        calls, puts = spy_options.calls, spy_options.puts
        if calls.empty or puts.empty:
            return None, None
        
        current_iv = (calls['impliedVolatility'].mean() + puts['impliedVolatility'].mean()) / 2 * 100
        
        # Step 2: Use market_data.csv for historical data
        if not os.path.exists(RISK_CSV_FILE):
            # If no history, return current IV but a neutral 50% IVR
            return current_iv, 50.0
        
        df_hist = pd.read_csv(RISK_CSV_FILE, parse_dates=['Timestamp'])
        df_hist = df_hist.set_index('Timestamp').sort_index()
        
        # Step 3: Prepare the historical IV data
        if 'Market IV' not in df_hist.columns:
            return current_iv, 50.0
            
        df_hist['Market IV'] = pd.to_numeric(df_hist['Market IV'], errors='coerce')
        df_hist.dropna(subset=['Market IV'], inplace=True)

        # Step 4: Filter for the last year and check for minimum data
        one_year_ago = datetime.now(pytz.utc) - timedelta(days=365)
        df_1y = df_hist[df_hist.index >= one_year_ago]

        # Use a minimum of 20 data points for a meaningful rank
        MIN_DATA_POINTS = 20
        if df_1y.empty or len(df_1y) < MIN_DATA_POINTS:
            return current_iv, 50.0 # Return neutral IVR if not enough data

        # Step 5: Calculate IVR
        min_iv = df_1y['Market IV'].min()
        max_iv = df_1y['Market IV'].max()

        # Also add the current IV to the calculation to ensure it's within the bounds
        min_iv = min(min_iv, current_iv)
        max_iv = max(max_iv, current_iv)

        if (max_iv - min_iv) > 0:
            ivr = ((current_iv - min_iv) / (max_iv - min_iv)) * 100
        else:
            ivr = 50.0 # If min and max are the same, rank is neutral
        
        return float(current_iv), float(np.clip(ivr, 0, 100))

    except Exception as e:
        risk_logger.warning(f"IV/IVR Calculation Error: {e}")
        return None, None
    
# --- Replace the existing perform_risk_calculations_singularity function in risk_command.py ---
async def perform_risk_calculations_singularity(is_eod_save: bool = False, is_called_by_ai: bool = False):
    print("[RISK_DEBUG] --- Starting perform_risk_calculations_singularity ---")
    risk_logger.info(f"--- Singularity: Performing R.I.S.K. calculations cycle (EOD Save: {is_eod_save}) ---")
    
    general, large_cap, ema, combined, spy_p, vix_p = await calculate_risk_scores_singularity(is_called_by_ai=True)
    print("[RISK_DEBUG] Main risk scores calculated. Now calculating recession likelihoods.")
    
    likelihood_ema_val = await asyncio.to_thread(calculate_recession_likelihood_ema_risk, is_called_by_ai=True)
    likelihood_vix_val = calculate_recession_likelihood_vix_risk(vix_p, is_called_by_ai=True)
    uncapped_mis, capped_mis_signal, _ = calculate_market_invest_score_risk(likelihood_vix_val, likelihood_ema_val, is_called_by_ai=True)
    print("[RISK_DEBUG] Recession likelihoods and Market Invest Score calculated.")

    # --- NEW: Calculate Percentile and IV/IVR ---
    print("[RISK_DEBUG] Calculating percentile and IV/IVR.")
    market_percentile = await asyncio.to_thread(calculate_market_score_percentile_risk, capped_mis_signal, is_called_by_ai=True)
    market_iv, market_ivr = await asyncio.to_thread(calculate_iv_and_ivr_risk, is_called_by_ai=True)
    print("[RISK_DEBUG] Percentile and IV/IVR calculated.")

    # Dictionary for screen display
    results_summary = {
        "general_score": f"{general:.2f}" if general is not None else 'N/A',
        "large_cap_score": f"{large_cap:.2f}" if large_cap is not None else 'N/A',
        "ema_score": f"{ema:.2f}" if ema is not None else 'N/A',
        "combined_score": f"{combined:.2f}" if combined is not None else 'N/A',
        "market_invest_score": f"{capped_mis_signal:.2f}" if capped_mis_signal is not None else 'N/A',
        "market_score_percentile": f"{market_percentile:.2f}%" if market_percentile is not None else 'N/A',
        "market_iv": f"{market_iv:.2f}" if market_iv is not None else 'N/A',
        "market_ivr": f"{market_ivr:.2f}%" if market_ivr is not None else 'N/A',
        "recession_chance_ema": f"{likelihood_ema_val:.2f}%" if likelihood_ema_val is not None else 'N/A',
        "recession_chance_vix": f"{likelihood_vix_val:.2f}%" if likelihood_vix_val is not None else 'N/A',
    }

    # Dictionary formatted for the CSV file
    data_for_csv = {
        "Timestamp": datetime.now(pytz.utc).isoformat(),
        "General Market Score": f"{general:.2f}" if general is not None else 'N/A',
        "Large Market Cap Score": f"{large_cap:.2f}" if large_cap is not None else 'N/A',
        "EMA Score": f"{ema:.2f}" if ema is not None else 'N/A',
        "Combined Score": f"{combined:.2f}" if combined is not None else 'N/A',
        "Live SPY Price": f"{spy_p:.2f}" if spy_p is not None else 'N/A',
        "Live VIX Price": f"{vix_p:.2f}" if vix_p is not None else 'N/A',
        "Momentum Based Recession Chance": f"{likelihood_ema_val:.1f}%" if likelihood_ema_val is not None else 'N/A',
        "VIX Based Recession Chance": f"{likelihood_vix_val:.1f}%" if likelihood_vix_val is not None else 'N/A',
        "Raw Market Invest Score": f"{uncapped_mis:.2f}" if uncapped_mis is not None else 'N/A',
        "Market Score Percentile": f"{market_percentile:.1f}%" if market_percentile is not None else 'N/A',
        "Market IV": f"{market_iv:.2f}" if market_iv is not None else 'N/A',
        "Market IVR": f"{market_ivr:.1f}%" if market_ivr is not None else 'N/A'
    }
    
    print("[RISK_DEBUG] --- Finished perform_risk_calculations_singularity ---")
    return results_summary, data_for_csv

# --- Replace the existing handle_risk_command function in risk_command.py ---
async def handle_risk_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    is_eod = (args and args[0].lower() == 'eod') or \
             (ai_params and ai_params.get("assessment_type") == "eod")
    
    # Unpack both the display results and the data for saving
    results, data_to_save = await perform_risk_calculations_singularity(is_eod_save=is_eod, is_called_by_ai=True)

    # Save the data to CSV every time, regardless of other parameters
    if data_to_save:
        _save_risk_data_to_csv(data_to_save)

    if is_called_by_ai:
        return results

    if results:
        print("\n--- R.I.S.K. Assessment ---")
        table_data = [[key.replace('_', ' ').title(), val] for key, val in results.items()]
        print(tabulate.tabulate(table_data, headers=["Metric", "Value"], tablefmt="heavy_outline", stralign="center"))
    else:
        print("\n[RISK] Failed to calculate risk assessment scores.")