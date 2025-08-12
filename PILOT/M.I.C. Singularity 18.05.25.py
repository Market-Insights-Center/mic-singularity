import yfinance as yf
import pandas as pd
import math
from tabulate import tabulate
import os
import time # Added for pauses
import sys # Added for printing characters with delay
import matplotlib.pyplot as plt
import numpy as np
# Ensure this library is compatible or consider if it needs similar session handling
# For tradingview_screener, if it makes HTTP requests that get blocked,
# it might need its own session handling or a different approach.
from tradingview_screener import Query, Column
from tradingview_ta import TA_Handler, Interval, Exchange # Ensure this library is compatible
import csv
from datetime import datetime, timedelta, time as dt_time, date # Renamed time import to dt_time
import pytz
from typing import Optional
import asyncio

# --- yfinance fix: Import requests from curl_cffi and create a session ---
# Using the approach from M.I.C. Singularity 08.05.25.py for potentially better data fetching
YFINANCE_SESSION = None
IS_CURL_CFFI_ACTIVE = False
try:
    from curl_cffi import requests as cffi_requests # Use an alias to avoid confusion
    YFINANCE_SESSION = cffi_requests.Session(impersonate="chrome")
    IS_CURL_CFFI_ACTIVE = True
    print("Successfully imported curl_cffi.requests and created a session with impersonation for yfinance.")
except ImportError:
    print("Warning: curl_cffi not found. yfinance calls will use standard requests and might fail due to API restrictions. Please install with: pip install curl_cffi")
    import requests # Fallback to standard requests
    YFINANCE_SESSION = requests.Session()
    IS_CURL_CFFI_ACTIVE = False
except Exception as e:
    print(f"Error setting up curl_cffi session: {e}. Using standard requests session.")
    import requests # Fallback
    YFINANCE_SESSION = requests.Session()
    IS_CURL_CFFI_ACTIVE = False

if YFINANCE_SESSION is not None:
    if IS_CURL_CFFI_ACTIVE:
        print("yfinance will use a curl_cffi session with impersonation.")
    else:
        print("yfinance will use a standard requests session (curl_cffi not active or failed to load). Data fetching may be unreliable.")
else:
    # This case should ideally not happen if the try-except block is structured correctly.
    print("CRITICAL: YFINANCE_SESSION could not be initialized. yfinance calls will likely fail.")
    # Attempt a final fallback to ensure YFINANCE_SESSION is at least a basic session object.
    import requests
    YFINANCE_SESSION = requests.Session()
    print("Initialized YFINANCE_SESSION with a basic requests.Session as a last resort.")


# --- Terminal Startup Sequence ---

# ASCII art placeholder as requested
MIC_LOGO_ASCII_PLACEHOLDER = "" \
"--- ASCII ART LOGO PLACEHOLDER ---"

def print_slowly(text, delay=0.01):
    """Prints text character by character with a delay."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print() # Newline at the end

def run_startup_sequence():
    """Runs the initial terminal startup sequence."""
    print_slowly("Welcome to the Market Insights Center Singularity", delay=0.03)
    time.sleep(1) # Short pause
    print(MIC_LOGO_ASCII_PLACEHOLDER) # Print logo placeholder
    time.sleep(1.5) # Longer pause
    print_slowly("INVEST has been activated", delay=0.03)
    time.sleep(1) # Short pause
    print("\nWhich command will you like to use?")
    print("Available commands:")
    print("  /custom")
    print("  /invest")
    print("  /market")
    print("  /breakout")
    print("  /assess")
    print("  /cultivate")
    print("  /exit (to quit)")
    print("-" * 30) # Separator

# --- Utility Functions ---

# Define the EST timezone
est_timezone = pytz.timezone('US/Eastern')

# Utility to safely handle NaN or None values and convert to float
def safe_score(value):
    """Returns 0.0 if value is NaN, None, or cannot be converted to float."""
    try:
        if pd.isna(value) or value is None:
            return 0.0
        # Handle potential percentage strings before converting
        if isinstance(value, str):
            value = value.replace('%', '').replace('$', '').strip()
        return float(value)
    except (ValueError, TypeError):
        return 0.0

# Helper function check_if_saved_today (synchronous version for terminal)
def check_if_saved_today(file_path: str, date_str: str) -> bool:
    """Checks if a specific date string exists in the 'DATE' column of a CSV file."""
    from pandas.errors import EmptyDataError
    if not os.path.exists(file_path):
        return False
    try:
        # Read only the 'DATE' column to be efficient
        df_dates = pd.read_csv(file_path, usecols=['DATE'], dtype={'DATE': str})
        if df_dates.empty:
            return False
        # Check if the date_str exists in the 'DATE' column values
        return date_str in df_dates['DATE'].values
    except FileNotFoundError:
        return False
    except EmptyDataError:
        return False
    except ValueError as ve:
        print(f"Warning: ValueError checking '{file_path}' for date '{date_str}'. Assuming not saved. Error: {ve}")
        return False
    except Exception as e:
        print(f"Error checking save status for '{file_path}' on date '{date_str}': {e}")
        return False

# Fetch the EMAs and calculate EMA Invest
def calculate_ema_invest(ticker, ema_interval):
    """Calculates EMA Invest score based on ticker and interval."""
    ticker_str = ticker.replace('.', '-')
    # Use the global session for yfinance
    stock = yf.Ticker(ticker_str, session=YFINANCE_SESSION)
    interval_mapping = {1: "1wk", 2: "1d", 3: "1h"}
    interval = interval_mapping.get(ema_interval, "1h")

    # Adjust period based on interval
    if interval == "1wk": period = "max"
    elif interval == "1d": period = "10y"
    elif interval == "1h": period = "730d" # Max allowed for 1h is ~2y
    else: period = "max" # Default

    try:
        # Attempt to fetch history data using the configured session
        data = stock.history(period=period, interval=interval)

        # Adjust period if data is insufficient for the chosen interval
        if data.empty:
             print(f"Warning: No history data returned for {ticker_str} with interval {interval} and period {period}. Trying shorter periods.")
             if interval == "1h":
                 data = stock.history(period="3mo", interval=interval)
                 if data.empty:
                     data = stock.history(period="1mo", interval=interval)
             elif interval == "1d":
                  data = stock.history(period="2y", interval=interval)
                  if data.empty:
                      data = stock.history(period="1y", interval=interval)
             elif interval == "1wk": # Corrected from '1w' to '1wk' to match mapping
                  data = stock.history(period="5y", interval=interval)

        if data.empty or 'Close' not in data.columns:
            print(f"Warning: Still no valid data or 'Close' column for {ticker_str} after period adjustments. This might be due to API issues, network problems, or the ticker being delisted.")
            return None, None

    except Exception as e:
        print(f"Error fetching history for {ticker_str} (Interval {interval}, Period {period}): {e}. This could be an API or network issue.")
        return None, None

    try:
        # Ensure enough data points for EMA calculation (at least 55 for EMA_55)
        if len(data) < 55:
             print(f"Warning: Insufficient data ({len(data)} points) for 55-period EMA for {ticker_str}. Cannot calculate EMA Invest.")
             live_price_fallback = data['Close'].iloc[-1] if not data.empty and not pd.isna(data['Close'].iloc[-1]) else None
             return live_price_fallback, None

        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_13'] = data['Close'].ewm(span=13, adjust=False).mean()
        data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()

    except Exception as e:
        print(f"Error calculating EMAs for {ticker_str}: {e}")
        return None, None

    if data.iloc[-1].isna().any():
       print(f"Warning: Latest EMAs are invalid for {ticker_str}")
       live_price_fallback = data['Close'].iloc[-1] if not data.empty and not pd.isna(data['Close'].iloc[-1]) else None
       return live_price_fallback, None

    latest_data = data.iloc[-1]
    live_price = latest_data['Close']
    ema_8 = latest_data['EMA_8']
    ema_55 = latest_data['EMA_55']

    if safe_score(ema_55) == 0:
        print(f"Warning: EMA_55 is zero for {ticker_str}, cannot calculate score.")
        return live_price, None

    ema_enter = (safe_score(ema_8) - safe_score(ema_55)) / safe_score(ema_55)
    ema_invest = ((ema_enter * 4) + 0.5) * 100
    ema_invest = max(0, min(ema_invest, 100))
    return live_price, ema_invest

# Calculate the one-year percent change and invest_per
def calculate_one_year_invest(ticker):
    """Calculates one-year percentage change and Invest percentage."""
    ticker_str = ticker.replace('.', '-')
    stock = yf.Ticker(ticker_str, session=YFINANCE_SESSION)
    try:
        data = stock.history(period="1y")
        if data.empty or len(data) < 2 or 'Close' not in data.columns:
             print(f"Warning: Insufficient 1-year data or missing 'Close' column for {ticker_str}.")
             return 0.0, 50.0
    except Exception as e:
        print(f"Error fetching 1-year history for {ticker_str}: {e}")
        return 0.0, 50.0

    start_price = safe_score(data['Close'].iloc[0])
    end_price = safe_score(data['Close'].iloc[-1])

    if start_price <= 0:
        print(f"Warning: Invalid or zero start price for 1-year calc on {ticker_str}. Cannot calculate change.")
        return 0.0, 50.0

    one_year_change = ((end_price - start_price) / start_price) * 100
    invest_per = 50.0
    if one_year_change < 0:
        invest_per = (one_year_change / 2) + 50
    else:
        try:
            invest_per = math.sqrt(max(0, one_year_change * 5)) + 50
        except ValueError:
            print(f"Warning: ValueError in sqrt for {ticker_str}. Using default invest_per.")
            invest_per = 50.0
    invest_per = max(0, min(invest_per, 100))
    return one_year_change, invest_per

# Calculate S&P 500 and S&P 100 symbols
def get_sp500_symbols():
    """Fetches S&P 500 ticker symbols from Wikipedia."""
    try:
        sp500_list_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # pd.read_html does not directly use the yfinance session.
        # If this fails due to blocking, a manual fetch using the session might be needed.
        dfs = pd.read_html(sp500_list_url)
        if not dfs:
             print("Warning: No tables found on S&P 500 Wikipedia page.")
             return []
        df = dfs[0]
        if 'Symbol' not in df.columns:
             print("Warning: 'Symbol' column not found in S&P 500 table.")
             return []
        symbols = df['Symbol'].tolist()
        symbols = [s.replace('.', '-') for s in symbols if isinstance(s, str)]
        return symbols
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}")
        return []

def get_sp100_symbols():
    """Fetches S&P 100 ticker symbols from Wikipedia."""
    try:
        sp100_list_url = 'https://en.wikipedia.org/wiki/S%26P_100'
        dfs = pd.read_html(sp100_list_url)
        if len(dfs) < 3:
             print("Warning: Not enough tables found on S&P 100 Wikipedia page.")
             return []
        df = dfs[2]
        if 'Symbol' not in df.columns:
             print("Warning: 'Symbol' column not found in S&P 100 table.")
             return []
        symbols = df['Symbol'].tolist()
        symbols = [s.replace('.', '-') for s in symbols if isinstance(s, str)]
        return symbols
    except Exception as e:
        print(f"Error fetching S&P 100 symbols: {e}")
        return []

def get_spy_symbols():
    """Returns S&P 500 symbols (using the same source as get_sp500_symbols)."""
    return get_sp500_symbols()


# Calculate market risk
def calculate_market_risk():
    """
    Calculates various market indicators and combines them into scores.
    Returns (None, None, None) on critical failure.
    """
    print("Fetching data for market risk calculation...")
    try:
        def calculate_ma_above(symbol, ma_window):
            """Checks if the latest price is above the specified moving average."""
            try:
                data = yf.download(symbol, period='1y', interval='1d', progress=False, session=YFINANCE_SESSION)
                if data.empty or len(data) < ma_window or 'Close' not in data.columns:
                    return None
                rolling_mean = data['Close'].rolling(window=ma_window).mean()
                latest_price = data['Close'].iloc[-1]
                latest_ma = rolling_mean.iloc[-1]
                if pd.isna(latest_ma) or pd.isna(latest_price):
                    return None
                return latest_price > latest_ma
            except Exception: # Minimal printing for sub-function
                return None

        def calculate_percentage_above_ma(symbols, ma_window):
            """Calculates the percentage of symbols above their moving average."""
            above_ma_count = 0
            valid_stocks = 0
            if not symbols: return 0.0
            for symbol in symbols:
                result = calculate_ma_above(symbol, ma_window)
                if result is not None:
                    valid_stocks += 1
                    if result:
                        above_ma_count += 1
            return (above_ma_count / valid_stocks) * 100 if valid_stocks > 0 else 0.0

        def calculate_s5tw():
             sp500_symbols = get_sp500_symbols()
             if not sp500_symbols: print("Warning: S&P 500 symbols list is empty for s5tw.")
             return calculate_percentage_above_ma(sp500_symbols, 20)

        def calculate_s5th():
             spy_symbols = get_spy_symbols()
             if not spy_symbols: print("Warning: SPY (S&P 500) symbols list is empty for s5th.")
             return calculate_percentage_above_ma(spy_symbols, 200)

        def calculate_s1fd():
            sp100_symbols = get_sp100_symbols()
            if not sp100_symbols: print("Warning: S&P 100 symbols list is empty for s1fd.")
            return calculate_percentage_above_ma(sp100_symbols, 5)

        def calculate_s1tw():
             sp100_symbols = get_sp100_symbols()
             if not sp100_symbols: print("Warning: S&P 100 symbols list is empty for s1tw.")
             return calculate_percentage_above_ma(sp100_symbols, 20)

        def get_live_price_and_ma(ticker_str_param):
            """Fetches live price and specified moving averages for a ticker."""
            try:
                stock = yf.Ticker(ticker_str_param, session=YFINANCE_SESSION)
                # Using a slightly longer period to ensure enough data for MAs
                hist = stock.history(period="260d", interval="1d") # Approx 1 year of trading days
                if hist.empty or 'Close' not in hist.columns:
                     print(f"Warning: No history data for {ticker_str_param} in get_live_price_and_ma.")
                     return None, None, None
                if len(hist) < 50: # Need at least 50 days for 50-day MA
                     print(f"Warning: Insufficient history data ({len(hist)} days) for MAs on {ticker_str_param}.")
                     # Try fetching more data if possible, or return None
                     hist_longer = stock.history(period="1y", interval="1d") # Try 1 year
                     if len(hist_longer) < 50:
                        print(f"Still insufficient data for {ticker_str_param} even with 1y period.")
                        return None, None, None
                     hist = hist_longer


                live_price = safe_score(hist['Close'].iloc[-1])
                ma_20 = safe_score(hist['Close'].rolling(window=20).mean().iloc[-1]) if len(hist) >= 20 else None
                ma_50 = safe_score(hist['Close'].rolling(window=50).mean().iloc[-1]) if len(hist) >= 50 else None

                if pd.isna(live_price) or (ma_20 is not None and pd.isna(ma_20)) or (ma_50 is not None and pd.isna(ma_50)):
                    print(f"Warning: NaN values encountered for price/MA on {ticker_str_param}")
                    return None, None, None
                return live_price, ma_20, ma_50
            except Exception as e:
                print(f"Error getting price/MA for {ticker_str_param}: {e}")
                return None, None, None

        spy_live_price, spy_ma_20, spy_ma_50 = get_live_price_and_ma('SPY')
        vix_live_price, _, _ = get_live_price_and_ma('^VIX')
        rut_live_price, rut_ma_20, rut_ma_50 = get_live_price_and_ma('^RUT')
        oex_live_price, oex_ma_20, oex_ma_50 = get_live_price_and_ma('^OEX')

        essential_prices = [spy_live_price, vix_live_price, rut_live_price, oex_live_price]
        if any(p is None for p in essential_prices):
            missing_indices = [name for name, val in zip(['SPY', 'VIX', 'RUT', 'OEX'], essential_prices) if val is None]
            print(f"Warning: Missing key index live prices for market risk calculation: {', '.join(missing_indices)}. Cannot reliably calculate market risk.")
            return None, None, None

        s5tw = calculate_s5tw()
        s5th = calculate_s5th()
        s1fd = calculate_s1fd()
        s1tw = calculate_s1tw()

        spy20 = ((safe_score(spy_live_price) - safe_score(spy_ma_20)) / 20) + 50 if spy_ma_20 is not None else 50.0
        spy50 = ((safe_score(spy_live_price) - safe_score(spy_ma_50) - 150) / 20) + 50 if spy_ma_50 is not None else 50.0
        vix_calc = (((safe_score(vix_live_price) - 15) * -5) + 50) if vix_live_price is not None else 50.0
        rut20 = ((safe_score(rut_live_price) - safe_score(rut_ma_20)) / 10) + 50 if rut_ma_20 is not None else 50.0
        rut50 = ((safe_score(rut_live_price) - safe_score(rut_ma_50)) / 5) + 50 if rut_ma_50 is not None else 50.0
        s5tw_calc = ((safe_score(s5tw) - 60) + 50) if s5tw is not None else 50.0 # Handle None from calculate_percentage_above_ma
        s5th_calc = ((safe_score(s5th) - 70) + 50) if s5th is not None else 50.0

        general_components = [
            (spy20, 3), (spy50, 1), (vix_calc, 3), (rut50, 3), (rut20, 1),
            (s5tw_calc, 2), (s5th_calc, 1)
        ]
        general_sum = sum(safe_score(score) * weight for score, weight in general_components)
        general_weights = sum(weight for _, weight in general_components)
        general_score = general_sum / general_weights if general_weights > 0 else 50.0

        oex20_calc = ((safe_score(oex_live_price) - safe_score(oex_ma_20)) / 10) + 50 if oex_ma_20 is not None else 50.0
        oex50_calc = ((safe_score(oex_live_price) - safe_score(oex_ma_50)) / 5) + 50 if oex_ma_50 is not None else 50.0
        s1fd_calc = ((safe_score(s1fd) - 60) + 50) if s1fd is not None else 50.0
        s1tw_calc = ((safe_score(s1tw) - 70) + 50) if s1tw is not None else 50.0

        large_components = [
            (oex20_calc, 3), (oex50_calc, 1), (s1fd_calc, 2), (s1tw_calc, 1)
        ]
        large_sum = sum(safe_score(score) * weight for score, weight in large_components)
        large_weights = sum(weight for _, weight in large_components)
        large_score = large_sum / large_weights if large_weights > 0 else 50.0

        combined_score = (general_score + large_score) / 2.0
        general_score = max(0, min(100, general_score))
        large_score = max(0, min(100, large_score))
        combined_score = max(0, min(100, combined_score))
        return combined_score, general_score, large_score
    except Exception as e:
        print(f"Error calculating market risk: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

# Calculate MACD signal
def calculate_macd_signal(ticker, ema_interval, fast_period=12, slow_period=26, signal_period=9):
    """Calculates MACD signal and strength based on ticker and interval."""
    ticker_str = ticker.replace('.', '-')
    stock = yf.Ticker(ticker_str, session=YFINANCE_SESSION)
    interval_mapping = {1: "1wk", 2: "1d", 3: "1h"}
    interval = interval_mapping.get(ema_interval, "1h")
    if interval == "1wk": period = "5y"
    elif interval == "1d": period = "2y"
    elif interval == "1h": period = "730d"
    else: period = "2y"

    try:
        data = stock.history(period=period, interval=interval)
        required_points = slow_period + signal_period
        if data.empty or len(data) < required_points or 'Close' not in data.columns:
             print(f"Warning: Insufficient data for MACD on {ticker_str} ({interval}). Required: {required_points}.")
             return "Neutral", 50.0
    except Exception as e:
         print(f"Error fetching history for MACD on {ticker_str}: {e}")
         return "Neutral", 50.0
    try:
        data['fast_ema'] = data['Close'].ewm(span=fast_period, adjust=False).mean()
        data['slow_ema'] = data['Close'].ewm(span=slow_period, adjust=False).mean()
        data['macd'] = data['fast_ema'] - data['slow_ema']
        data['signal'] = data['macd'].ewm(span=signal_period, adjust=False).mean()
        data['histogram'] = data['macd'] - data['signal']
    except Exception as e:
        print(f"Error calculating MACD components for {ticker_str}: {e}")
        return "Neutral", 50.0

    if len(data['histogram']) < 3 or data['histogram'].isnull().tail(3).any():
         print(f"Warning: Not enough valid histogram points for MACD signal on {ticker_str}.")
         return "Neutral", 50.0
    last_three_hist = data['histogram'].dropna().tail(3).tolist()
    if len(last_three_hist) < 3:
        print(f"Warning: Not enough non-NaN histogram points for MACD signal on {ticker_str}.")
        return "Neutral", 50.0

    signal = "Neutral"
    if last_three_hist[2] < last_three_hist[1] < last_three_hist[0] and last_three_hist[2] < 0 : signal = "Sell"
    elif last_three_hist[2] > last_three_hist[1] > last_three_hist[0] and last_three_hist[2] > 0: signal = "Buy"
    elif last_three_hist[2] > last_three_hist[1] and last_three_hist[1] <= last_three_hist[0] and last_three_hist[2] > 0: signal = "Buy"
    elif last_three_hist[2] < last_three_hist[1] and last_three_hist[1] >= last_three_hist[0] and last_three_hist[2] < 0: signal = "Sell"

    macd_strength_percent_adjusted = 50.0
    if not any(pd.isna(h) for h in last_three_hist):
         strength_change1 = abs(last_three_hist[2] - last_three_hist[1])
         strength_change2 = abs(last_three_hist[1] - last_three_hist[0])
         macd_strength_val = (strength_change1 + strength_change2) / 2.0 # Renamed to avoid conflict
         macd_strength_percent_adjusted = ((macd_strength_val / 2.0) * 100.0) + 50.0
    macd_strength_percent_adjusted = max(0, min(100, macd_strength_percent_adjusted))
    return signal, macd_strength_percent_adjusted

# Plot ticker graph
def plot_ticker_graph(ticker, ema_interval):
    """Plots ticker price and EMAs and saves to a file."""
    ticker_str = ticker.replace('.', '-')
    stock = yf.Ticker(ticker_str, session=YFINANCE_SESSION)
    interval_mapping = {1: "1wk", 2: "1d", 3: "1h"}
    interval = interval_mapping.get(ema_interval, "1h")
    if ema_interval == 3: period = "6mo"
    elif ema_interval == 1: period = "5y"
    elif ema_interval == 2: period = "1y"
    else: period = "1y"

    try:
        data = stock.history(period=period, interval=interval)
        if data.empty or 'Close' not in data.columns:
            raise ValueError(f"No data or 'Close' column for {ticker_str} (Period: {period}, Interval: {interval})")
        if len(data) < 55: data['EMA_55'] = np.nan
        else: data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()
        if len(data) < 8: data['EMA_8'] = np.nan
        else: data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Close'], color='grey', label='Price', linewidth=1.0)
        if 'EMA_55' in data.columns and not data['EMA_55'].isnull().all():
             ax.plot(data.index, data['EMA_55'], color='darkgreen', label='EMA 55', linewidth=1.5)
        if 'EMA_8' in data.columns and not data['EMA_8'].isnull().all():
             ax.plot(data.index, data['EMA_8'], color='firebrick', label='EMA 8', linewidth=1.5)
        ax.set_title(f"{ticker_str} Price and EMAs ({interval})", color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Price', color='white')
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax.grid(True, color='dimgray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
        fig.tight_layout()
        filename = f"{ticker_str}_graph.png"
        plt.savefig(filename, facecolor='black', edgecolor='black'); plt.close(fig)
        return filename
    except Exception as e:
        print(f"Error plotting graph for {ticker_str}: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None

# Function to get allocation score
def get_allocation_score():
    """Reads market data and returns allocation scores."""
    market_data_file = 'market_data.csv'
    avg_score, risk_gen_score, mkt_inv_score = 50.0, 50.0, 50.0
    if not os.path.exists(market_data_file):
        print(f"Warning: Market data file '{market_data_file}' not found. Using default scores (50).")
        return avg_score, risk_gen_score, mkt_inv_score
    try:
        df = pd.read_csv(market_data_file)
        if df.empty:
            print(f"Warning: Market data file '{market_data_file}' is empty. Using default scores (50).")
            return avg_score, risk_gen_score, mkt_inv_score
        last_row = df.iloc[-1]
        risk_gen_score = safe_score(last_row.get('RISK_GEN_SCORE', 50.0))
        mkt_inv_score = safe_score(last_row.get('MKT_INV_SCORE', 50.0))
        avg_score = safe_score(last_row.get('AVG_SCORE', (risk_gen_score + mkt_inv_score) / 2.0))
        return max(0,min(100,avg_score)), max(0,min(100,risk_gen_score)), max(0,min(100,mkt_inv_score))
    except Exception as e:
        print(f"Error reading market data file '{market_data_file}': {e}. Using default scores (50).")
        return 50.0, 50.0, 50.0

# --- process_custom_portfolio function (Adapted for Terminal) ---
# This function now takes no 'interaction' object and prints directly
async def process_custom_portfolio(portfolio_data, tailor_portfolio, frac_shares,
                                   total_value=None, is_custom_command_without_save=False):
    """
    Processes custom or /invest portfolio requests, calculates scores, allocations,
    and generates output tables and graphs for the terminal.
    """
    sell_to_cash_active = False
    avg_score, _, _ = get_allocation_score()
    if avg_score is not None and avg_score < 50.0:
        sell_to_cash_active = True
        print(f"INFO: Sell-to-Cash feature ACTIVE (Avg Market Score: {avg_score:.2f} < 50). Allocations for tickers with Raw Score < 50 are adjusted, difference moved to Cash.")

    # Default values (can be overridden by portfolio_data if needed, but typically fixed for this logic)
    risk_type = portfolio_data.get('risk_type', 'stock')
    ema_sensitivity = int(portfolio_data.get('ema_sensitivity', 3))
    try:
        amplification = float(portfolio_data.get('amplification', 1.0))
    except ValueError:
        print("Warning: Invalid amplification value found in portfolio_data. Defaulting to 1.0")
        amplification = 1.0
    num_portfolios = int(portfolio_data.get('num_portfolios', 0))

    portfolio_results = [] # Stores lists of ticker data for each sub-portfolio
    all_entries_for_graphs = [] # Stores dicts {'ticker': T, 'ema_sensitivity': S} for graph generation

    # Initialize new return values for Assess Code C (even if not used by /invest or /custom directly)
    tailored_portfolio_entries_intermediate_for_return = []
    final_cash_value_tailored_for_return = 0.0


    print("--- Starting Initial Ticker Calculations ---")
    for i in range(num_portfolios):
        portfolio_index = i + 1
        tickers_str = portfolio_data.get(f'tickers_{portfolio_index}', '')
        weight_str = portfolio_data.get(f'weight_{portfolio_index}', '0')
        try:
            weight = float(weight_str)
        except ValueError:
            print(f"Warning: Invalid weight '{weight_str}' for portfolio {portfolio_index}. Setting to 0.")
            weight = 0.0

        tickers = [ticker.strip().upper() for ticker in tickers_str.split(',') if ticker.strip()]
        if not tickers:
            print(f"Warning: No valid tickers for portfolio {portfolio_index}. Skipping.")
            continue

        current_portfolio_list = [] # Data for tickers in the current sub-portfolio
        for ticker in tickers:
            try:
                # Calculate live price and EMA Invest score using helper functions
                live_price, ema_invest = calculate_ema_invest(ticker, ema_sensitivity)

                if live_price is None and ema_invest is None: # Both failed
                     print(f"Warning: Failed to get base data (price and score) for {ticker}. Skipping.")
                     current_portfolio_list.append({'ticker': ticker, 'error': "Failed to get base data", 'portfolio_weight': weight})
                     all_entries_for_graphs.append({'ticker': ticker, 'error': "Failed to get base data"})
                     continue

                if ema_invest is None: # Score calculation failed, but price might be available
                     print(f"Warning: EMA Invest score calculation failed for {ticker}, using neutral 50.")
                     ema_invest = 50.0
                if live_price is None: # Price calculation failed, but score might be available
                     print(f"Warning: Live price calculation failed for {ticker}. Assigning 0 price for allocation purposes.")
                     live_price = 0.0 # Critical for allocation if score is present

                # Calculate one-year invest score (not directly used in combined score here, but part of original logic)
                _, invest_per = calculate_one_year_invest(ticker)

                # Ensure scores are numeric
                ema_invest = safe_score(ema_invest)
                # invest_per = safe_score(invest_per) # Not used in this version's raw_combined_invest

                # Simplified raw combined score (as per user's original script structure for /invest)
                stock_invest = ema_invest # The primary score component
                raw_combined_invest = stock_invest

                # Determine score for allocation (may be adjusted by Sell-to-Cash)
                score_for_allocation = raw_combined_invest
                score_was_adjusted = False # Flag to track if Sell-to-Cash adjusted this ticker's score

                if sell_to_cash_active and raw_combined_invest < 50.0:
                    score_for_allocation = 50.0 # Adjust score to 50 for allocation calculation
                    score_was_adjusted = True

                # Calculate amplified scores (both original and adjusted for Sell-to-Cash)
                # Formula: (Score * Amplification) - (Amplification - 1) * 50
                # This formula centers the amplification effect around 50%
                amplified_score_adjusted = safe_score((score_for_allocation * amplification) - (amplification - 1) * 50)
                amplified_score_adjusted_clamped = max(0, amplified_score_adjusted) # Clamp at 0

                amplified_score_original = safe_score((raw_combined_invest * amplification) - (amplification - 1) * 50)
                amplified_score_original_clamped = max(0, amplified_score_original) # Clamp at 0

                entry_data = {
                    'ticker': ticker, 'live_price': live_price, 'raw_invest_score': raw_combined_invest,
                    'amplified_score_adjusted': amplified_score_adjusted_clamped, # Used for adjusted allocation
                    'amplified_score_original': amplified_score_original_clamped, # Used for original allocation (if Sell-to-Cash active)
                    'portfolio_weight': weight, 'score_was_adjusted': score_was_adjusted,
                    'portfolio_allocation_percent_adjusted': None, # To be calculated later
                    'portfolio_allocation_percent_original': None, # To be calculated later
                    'combined_percent_allocation_adjusted': None, # To be calculated later
                    'combined_percent_allocation_original': None, # To be calculated later
                }
                current_portfolio_list.append(entry_data)
                if live_price > 0: # Only add to graph list if price is valid
                    all_entries_for_graphs.append({'ticker': ticker, 'ema_sensitivity': ema_sensitivity})
            except Exception as e:
                print(f"Error processing ticker {ticker} in portfolio {portfolio_index}: {e}")
                current_portfolio_list.append({'ticker': ticker, 'error': str(e), 'portfolio_weight': weight})
                all_entries_for_graphs.append({'ticker': ticker, 'error': str(e)}) # Add error entry for graph handling
        portfolio_results.append(current_portfolio_list)
    print("--- Finished Initial Ticker Calculations ---")

    print("Generating ticker graphs (saved as PNG files)...")
    sent_graphs = set() # To avoid generating duplicate graphs
    for graph_entry in all_entries_for_graphs:
        ticker_key = graph_entry.get('ticker')
        if not ticker_key or ticker_key in sent_graphs: continue # Skip if no ticker or already sent

        if 'error' not in graph_entry: # Only plot if no initial error
            try:
                graph_filename = plot_ticker_graph(ticker_key, graph_entry['ema_sensitivity'])
                if graph_filename and os.path.exists(graph_filename):
                    print(f"  Graph saved to {graph_filename}")
                    sent_graphs.add(ticker_key)
                    # Keep the file for the user to view later in the terminal environment
                else:
                    print(f"  Failed to generate graph for {ticker_key}.")
            except Exception as plot_error:
                print(f"  Error plotting graph for {ticker_key}: {plot_error}")
        else: # Ticker had an initial error
             print(f"  Skipping graph for {ticker_key} due to earlier error: {graph_entry['error']}")
    print("--- Finished Graph Generation ---")


    print("--- Calculating Sub-Portfolio Allocations (Adjusted & Original) ---")
    for portfolio_index, portfolio in enumerate(portfolio_results):
        # Calculate total amplified score for the current sub-portfolio (ADJUSTED for Sell-to-Cash)
        portfolio_amplified_total_adjusted = safe_score(sum(entry.get('amplified_score_adjusted', 0) for entry in portfolio if 'error' not in entry))
        for entry in portfolio:
            if 'error' not in entry:
                if portfolio_amplified_total_adjusted > 0:
                     amplified_score_adj = safe_score(entry.get('amplified_score_adjusted', 0))
                     # Calculate allocation within this sub-portfolio based on adjusted scores
                     portfolio_allocation_percent_adj = safe_score((amplified_score_adj / portfolio_amplified_total_adjusted) * 100)
                     entry['portfolio_allocation_percent_adjusted'] = round(portfolio_allocation_percent_adj, 2)
                else: entry['portfolio_allocation_percent_adjusted'] = 0.0 # No allocation if total score is zero
            else: entry['portfolio_allocation_percent_adjusted'] = None # Error case

        # Calculate total amplified score for the current sub-portfolio (ORIGINAL, before Sell-to-Cash adjustment)
        portfolio_amplified_total_original = safe_score(sum(entry.get('amplified_score_original', 0) for entry in portfolio if 'error' not in entry))
        for entry in portfolio:
            if 'error' not in entry:
                if portfolio_amplified_total_original > 0:
                    amplified_score_orig = safe_score(entry.get('amplified_score_original', 0))
                    # Calculate allocation within this sub-portfolio based on original scores
                    portfolio_allocation_percent_orig = safe_score((amplified_score_orig / portfolio_amplified_total_original) * 100)
                    entry['portfolio_allocation_percent_original'] = round(portfolio_allocation_percent_orig, 2)
                else: entry['portfolio_allocation_percent_original'] = 0.0
            else: entry['portfolio_allocation_percent_original'] = None


    # Output Sub-Portfolios (if not a 'custom command without save' which implies simplified output)
    if not is_custom_command_without_save: # Typically for /invest or /custom with save code
        for i, portfolio in enumerate(portfolio_results, 1):
            portfolio.sort(key=lambda x: x.get('portfolio_allocation_percent_adjusted', -1) if x.get('portfolio_allocation_percent_adjusted') is not None else -1, reverse=True)
            portfolio_weight_display = portfolio[0].get('portfolio_weight', 'N/A') if portfolio and 'error' not in portfolio[0] else 'N/A'
            print(f"\n--- Sub-Portfolio {i} (Weight: {portfolio_weight_display}%) ---")
            table_data = []
            for entry in portfolio:
                 if 'error' not in entry:
                    live_price_f = f"${entry.get('live_price', 0):.2f}"
                    invest_score_val = safe_score(entry.get('raw_invest_score', 0))
                    invest_score_f = f"{invest_score_val:.2f}%" if invest_score_val is not None else "N/A"
                    # Display adjusted amplified score and original portfolio allocation %
                    amplified_score_f = f"{entry.get('amplified_score_adjusted', 0):.2f}%"
                    port_alloc_val_original = safe_score(entry.get('portfolio_allocation_percent_original', 0))
                    port_alloc_f = f"{port_alloc_val_original:.2f}%" if port_alloc_val_original is not None else "N/A"
                    table_data.append([entry.get('ticker', 'ERR'), live_price_f, invest_score_f, amplified_score_f, port_alloc_f])
                 else:
                     table_data.append([entry.get('ticker', 'ERR'), "Error", "Error", "Error", "Error"]) # Display error row

            if not table_data: table_str = "No valid data for this sub-portfolio."
            else: table_str = tabulate(table_data, headers=["Ticker", "Live Price", "Raw Score", "Adj Amplified %", "Portfolio % Alloc (Original)"], tablefmt="pretty")

            print(f"```{table_str}```")

            # Display errors if any
            error_messages = [f"Error for {entry.get('ticker', 'UNKNOWN')}: {entry.get('error', 'Unknown error')}" for entry in portfolio if 'error' in entry]
            if error_messages:
                print("Errors in this Sub-Portfolio:")
                for msg in error_messages: print(msg)


    print("--- Calculating Combined Portfolio Allocations (Adjusted & Original) ---")
    combined_result_intermediate = [] # Stores entries with their final combined allocation percentages
    for portfolio in portfolio_results:
        for entry in portfolio:
            if 'error' not in entry:
                port_weight = entry.get('portfolio_weight', 0)
                # Calculate combined allocation based on ADJUSTED sub-portfolio allocation
                sub_alloc_adj = entry.get('portfolio_allocation_percent_adjusted', 0)
                combined_percent_allocation_adjusted = round(safe_score((sub_alloc_adj * port_weight) / 100.0), 4) # Use 100.0 for float division
                entry['combined_percent_allocation_adjusted'] = combined_percent_allocation_adjusted
                # Calculate combined allocation based on ORIGINAL sub-portfolio allocation
                sub_alloc_orig = entry.get('portfolio_allocation_percent_original', 0)
                combined_percent_allocation_original = round(safe_score((sub_alloc_orig * port_weight) / 100.0), 4) # Use 100.0 for float division
                entry['combined_percent_allocation_original'] = combined_percent_allocation_original
                combined_result_intermediate.append(entry) # Add to list for final processing
            else:
                 # Include error entries in intermediate list as well, for completeness if needed later
                 combined_result_intermediate.append(entry)


    print("--- Constructing Final Combined Portfolio (with Cash if applicable) ---")
    final_combined_portfolio_data = [] # This will be used for the "Final Combined Portfolio" table
    total_cash_diff_percent = 0.0 # Accumulates cash from Sell-to-Cash adjustments

    valid_combined_intermediate = [e for e in combined_result_intermediate if 'error' not in e]

    for entry in valid_combined_intermediate: # Iterate through all processed tickers (excluding errors for this step)
        final_combined_portfolio_data.append({
            'ticker': entry['ticker'], 'live_price': entry['live_price'],
            'raw_invest_score': entry['raw_invest_score'],
            'amplified_score_adjusted': entry['amplified_score_adjusted'], # For display
            'combined_percent_allocation': entry['combined_percent_allocation_adjusted'] # This is the key allocation %
        })
        # If Sell-to-Cash was active and this ticker's score was adjusted, calculate the difference
        if sell_to_cash_active and entry.get('score_was_adjusted', False):
            # Difference between what it would have been allocated (adjusted) vs. original
            # This difference is positive if the adjusted score (50) led to a higher allocation than the original low score
            adj_alloc = entry['combined_percent_allocation_adjusted']
            orig_alloc = entry['combined_percent_allocation_original']
            difference = adj_alloc - orig_alloc
            total_cash_diff_percent += max(0.0, difference) # Ensure only positive differences contribute to cash

    # Normalize stock allocations if Sell-to-Cash is active and cash was generated
    # The total allocation to stocks should be (100 - total_cash_diff_percent)
    current_stock_total_alloc = sum(item['combined_percent_allocation'] for item in final_combined_portfolio_data if item['ticker'] != 'Cash')
    target_stock_alloc = 100.0 - total_cash_diff_percent

    # Only normalize if there are stocks and the current total is significantly different from the target
    if current_stock_total_alloc > 1e-9 and not math.isclose(current_stock_total_alloc, target_stock_alloc, abs_tol=0.01):
        if current_stock_total_alloc > 1e-9: # Avoid division by zero
             norm_factor = target_stock_alloc / current_stock_total_alloc
             for item in final_combined_portfolio_data:
                 if item['ticker'] != 'Cash':
                     item['combined_percent_allocation'] *= norm_factor
        else: print("    Warning: Cannot normalize zero stock allocations for Sell-to-Cash.")
    elif current_stock_total_alloc == 0 and target_stock_alloc > 0:
         # This means no stocks were processed successfully, but the target wasn't 100% cash initially.
         # This scenario is tricky. If no stocks were processed, the portfolio is effectively 100% cash.
         # Let's ensure the cash allocation reflects the full 100% in this case.
         pass # Handled below when adding the Cash row


    # Add Cash row if any cash was generated by Sell-to-Cash OR if no stocks were processed
    cash_alloc_percent = total_cash_diff_percent
    if not final_combined_portfolio_data and math.isclose(target_stock_alloc, 100.0, abs_tol=0.01):
        # If no stocks were processed and the target was 100% cash (meaning Sell-to-Cash didn't create diff,
        # but perhaps all tickers failed), the portfolio should be 100% cash.
        cash_alloc_percent = 100.0
    elif not final_combined_portfolio_data and not math.isclose(target_stock_alloc, 100.0, abs_tol=0.01):
         # If no stocks were processed and the target wasn't 100% cash, something went wrong.
         # We'll still show 100% cash as no stocks were allocated.
         cash_alloc_percent = 100.0


    if cash_alloc_percent > 1e-4 or (not final_combined_portfolio_data and cash_alloc_percent == 0):
        # Add Cash row if there's a significant cash amount OR if no stocks were processed at all
        # and a cash row isn't already implicitly there from a non-zero total_cash_diff_percent
        if not any(item['ticker'] == 'Cash' for item in final_combined_portfolio_data):
            final_combined_portfolio_data.append({
                'ticker': 'Cash', 'live_price': 1.0, 'raw_invest_score': None,
                'amplified_score_adjusted': None, 'combined_percent_allocation': cash_alloc_percent
            })
            print(f"    Added Cash row to Final Combined Portfolio due to Sell-to-Cash or no stock data: {cash_alloc_percent:.2f}%")
        else:
             # Update the existing cash row's allocation if it was added implicitly
             for item in final_combined_portfolio_data:
                 if item['ticker'] == 'Cash':
                     item['combined_percent_allocation'] = cash_alloc_percent
                     print(f"    Updated existing Cash row allocation: {cash_alloc_percent:.2f}%")
                     break


    # Sort final combined portfolio by raw score (descending), Cash last
    final_combined_portfolio_data.sort(
        key=lambda x: safe_score(x.get('raw_invest_score', -float('inf'))) if x['ticker'] != 'Cash' else -float('inf')-1,
        reverse=True
    )


    # Output Final Combined Portfolio (if not a 'custom command without save')
    if not is_custom_command_without_save:
        print("\n--- Final Combined Portfolio (Sorted by Raw Score)---")
        if sell_to_cash_active: print(f"*(Sell-to-Cash Active: Difference allocated to Cash)*")
        combined_data_display = []
        for entry in final_combined_portfolio_data:
            ticker = entry.get('ticker', 'ERR')
            if ticker == 'Cash':
                live_price_f, invest_score_f, amplified_score_f = '-', '-', '-'
            else:
                live_price_f = f"${entry.get('live_price', 0):,.2f}" # Added comma formatting
                invest_score_f = f"{entry.get('raw_invest_score', 0):.2f}%"
                amplified_score_f = f"{entry.get('amplified_score_adjusted', 0):.2f}%"
            comb_alloc_f = f"{round(entry.get('combined_percent_allocation', 0), 2):.2f}%"
            combined_data_display.append([ticker, live_price_f, invest_score_f, amplified_score_f, comb_alloc_f])

        if not combined_data_display: combined_table_str = "No valid data for the combined portfolio."
        else: combined_table_str = tabulate(combined_data_display, headers=["Ticker", "Live Price", "Raw Score", "Adj Amplified %", "Final % Alloc"], tablefmt="pretty")

        print(f"```{combined_table_str}```")

    print("--- Calculating Tailored Portfolio ---")
    tailored_portfolio_output_list = [] # For simplified output (e.g., /custom without save)
    remaining_buying_power = None # For /custom without save, when Sell-to-Cash is inactive
    # final_cash_value_tailored is calculated below

    if tailor_portfolio and total_value is not None and safe_score(total_value) > 0: # True if total_value is provided and valid
        total_value = safe_score(total_value) # Ensure it's a float

        # This list will hold the structured data needed for Assess Code C
        current_tailored_entries = [] # Stores dicts for each stock holding
        total_actual_money_allocated_stocks = 0.0

        # Iterate through the final_combined_portfolio_data (which includes allocations and potentially a Cash row from Sell-to-Cash)
        for entry in final_combined_portfolio_data:
            if entry['ticker'] == 'Cash': continue # Skip the 'Cash' row from Sell-to-Cash for stock allocation

            final_stock_alloc_pct = safe_score(entry.get('combined_percent_allocation', 0.0))
            live_price = safe_score(entry.get('live_price', 0.0))

            if final_stock_alloc_pct > 1e-9 and live_price > 0: # Allocate if % > 0 and price > 0
                target_allocation_value_for_ticker = total_value * (final_stock_alloc_pct / 100.0)
                shares = 0.0
                try:
                    exact_shares = target_allocation_value_for_ticker / live_price
                    if frac_shares: shares = round(exact_shares, 4) # Use 4 decimal places for fractional shares
                    else: shares = float(math.floor(exact_shares))
                except ZeroDivisionError: shares = 0.0 # Should be caught by live_price > 0
                except Exception: shares = 0.0
                shares = max(0.0, shares) # Ensure non-negative shares
                actual_money_allocation_for_ticker = shares * live_price

                share_threshold = 0.0001 if frac_shares else 1.0 # Min shares to buy
                if shares >= share_threshold:
                    # Actual percent of the *total_value* this ticker represents
                    actual_percent_of_total_value = (actual_money_allocation_for_ticker / total_value) * 100.0 if total_value > 0 else 0.0
                    current_tailored_entries.append({
                        'ticker': entry.get('ticker','ERR'),
                        'raw_invest_score': entry.get('raw_invest_score', -float('inf')), # For sorting
                        'shares': shares,
                        'actual_money_allocation': actual_money_allocation_for_ticker, # For Assess C
                        'actual_percent_allocation': actual_percent_of_total_value # For display
                    })
                    total_actual_money_allocated_stocks += actual_money_allocation_for_ticker

        # Calculate remaining/final cash for the tailored portfolio
        raw_remaining_value_after_stocks = total_value - total_actual_money_allocated_stocks

        if not sell_to_cash_active: # Sell-to-Cash was NOT active during initial score processing
            remaining_buying_power = raw_remaining_value_after_stocks # Can be negative if over-allocated
            final_cash_value_tailored_for_return = max(0.0, raw_remaining_value_after_stocks) # Actual cash cannot be negative
        else: # Sell-to-Cash WAS active
            # The 'final_combined_portfolio_data' already accounted for cash generated from score adjustments.
            # The allocations in 'final_combined_portfolio_data' were normalized to (100 - total_cash_diff_percent).
            # So, raw_remaining_value_after_stocks is the cash remaining *after* buying stocks based on those normalized allocations.
            # This value should ideally be close to the 'total_cash_diff_percent' * total_value if allocations were perfect.
            final_cash_value_tailored_for_return = max(0.0, raw_remaining_value_after_stocks)
            remaining_buying_power = None # Not typically shown when Sell-to-Cash is active

        # Calculate final cash percentage for display purposes
        final_cash_percent_display = (final_cash_value_tailored_for_return / total_value) * 100.0 if total_value > 0 else 0.0
        final_cash_percent_display = max(0.0, min(100.0, final_cash_percent_display)) # Clamp 0-100

        # Sort tailored entries by raw score for display consistency
        current_tailored_entries.sort(key=lambda x: safe_score(x.get('raw_invest_score', -float('inf'))), reverse=True)

        # Assign to the variable that will be returned for Assess C
        tailored_portfolio_entries_intermediate_for_return = current_tailored_entries # List of dicts
        # final_cash_value_tailored_for_return is already set above

        # Prepare data for table output (uses final_cash_value_tailored_for_return for cash row)
        tailored_portfolio_table_data_display = [
             [item['ticker'], f"{item['shares']:.4f}" if frac_shares else f"{int(item['shares'])}", # Format shares
              f"${safe_score(item['actual_money_allocation']):,.2f}", f"{safe_score(item['actual_percent_allocation']):.2f}%"]
             for item in current_tailored_entries
        ]
        tailored_portfolio_table_data_display.append(['Cash', '-', f"${safe_score(final_cash_value_tailored_for_return):,.2f}", f"{safe_score(final_cash_percent_display):.2f}%"])

        # Prepare data for simplified list output
        if frac_shares:
            tailored_portfolio_output_list = [f"{item['ticker']} - {item['shares']:.4f}" for item in current_tailored_entries] # Format shares
        else:
            tailored_portfolio_output_list = [f"{item['ticker']} - {int(item['shares'])}" for item in current_tailored_entries]

        # Output Tailored Portfolio (Full table or Simplified list)
        if is_custom_command_without_save: # Simplified output for /custom without save code
            print("\n--- Tailored Portfolio Allocation (Sorted by Raw Score) ---")
            if tailored_portfolio_output_list:
                for line in tailored_portfolio_output_list: print(line)
            else:
                print("No stocks allocated in the tailored portfolio based on the provided value and strategy.")
            # Send final cash value and remaining buying power (if applicable)
            print(f"Final Cash Value: ${safe_score(final_cash_value_tailored_for_return):,.2f}")
            if remaining_buying_power is not None: # Only show if Sell-to-Cash was inactive
                print(f"Remaining Buying Power: ${safe_score(remaining_buying_power):,.2f}")
        else: # Full table output for /invest or /custom with save code
            print("\n--- Tailored Portfolio (Sorted by Raw Score) ---")
            if not tailored_portfolio_table_data_display: tailored_table_str = "No stocks allocated." # Should at least have Cash
            else: tailored_table_str = tabulate(tailored_portfolio_table_data_display, headers=["Ticker", "Shares", "Actual $ Allocation", "Actual % of Total"], tablefmt="pretty")

            print(f"```{tailored_table_str}```")
            if remaining_buying_power is not None: # Only show if Sell-to-Cash was inactive
                print(f"Remaining Buying Power: ${safe_score(remaining_buying_power):,.2f}")

    print("--- Finished Tailored Portfolio ---")

    # Return values:
    # tailored_portfolio_output_list: Simplified list for /custom without save
    # combined_result_intermediate: List of dicts with combined allocations (used by some internal saves)
    # portfolio_results: List of lists, raw sub-portfolio processing results
    # tailored_portfolio_entries_intermediate_for_return: List of dicts for Assess C (ticker, shares, actual_money_allocation)
    # final_cash_value_tailored_for_return: Float, final cash for Assess C
    return (
        tailored_portfolio_output_list,
        combined_result_intermediate,
        portfolio_results,
        tailored_portfolio_entries_intermediate_for_return, # New for Assess C
        final_cash_value_tailored_for_return # New for Assess C
    )

# --- Terminal Input Handling Functions (from M.I.C. Singularity 08.05.25.py) ---
def get_user_input(prompt, validation=None, error_message="Invalid input. Please try again."):
    """Gets user input from the terminal with optional validation."""
    while True:
        user_input = input(prompt).strip()
        if not validation:
            return user_input
        if validation(user_input):
            return user_input
        else:
            print(error_message)

def get_float_input(prompt, min_value=None, max_value=None):
    """Gets a float input from the user with optional min/max validation."""
    def validate(value_str):
        try:
            val = float(value_str)
            if min_value is not None and val < min_value:
                print(f"Value must be at least {min_value}.")
                return False
            if max_value is not None and val > max_value:
                print(f"Value must be at most {max_value}.")
                return False
            return True
        except ValueError:
            return False
    return float(get_user_input(prompt, validate, "Invalid number format."))

def get_int_input(prompt, min_value=None, max_value=None):
    """Gets an integer input from the user with optional min/max validation."""
    def validate(value_str):
        try:
            val = int(value_str)
            if min_value is not None and val < min_value:
                print(f"Value must be at least {min_value}.")
                return False
            if max_value is not None and val > max_value:
                print(f"Value must be at most {max_value}.")
                return False
            return True
        except ValueError:
            return False
    return int(get_user_input(prompt, validate, "Invalid integer format."))

def get_yes_no_input(prompt):
    """Gets a yes/no input from the user."""
    def validate(response):
        return response.lower() in ['yes', 'no', 'y', 'n']
    response = get_user_input(prompt, validate, "Please enter 'yes' or 'no'.")
    return response.lower() in ['yes', 'y']


# --- Terminal Command Handlers ---

# --- /invest Command Handler (Adapted for Terminal) ---
async def handle_invest_command(args):
    """
    Handles the /invest command for the terminal. Collects user inputs
    via terminal and calls process_custom_portfolio.
    """
    print("\n--- Running /invest Command ---")

    # Collect inputs via terminal
    ema_sensitivity = get_int_input("Enter EMA sensitivity (1: Weekly, 2: Daily, 3: Hourly): ", min_value=1, max_value=3)

    valid_amplifications = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    amplification_input = 0.0 # Initialize
    while True:
        amplification_str = get_user_input(f"Enter amplification ({', '.join(map(str, valid_amplifications))}): ")
        try:
            amplification_input = float(amplification_str)
            if amplification_input in valid_amplifications:
                break
            else:
                print(f"Invalid amplification value. Please enter one of: {', '.join(map(str, valid_amplifications))}")
        except ValueError:
            print(f"Invalid number format. Please enter one of: {', '.join(map(str, valid_amplifications))}")
    amplification = amplification_input # Use the validated float

    num_portfolios = get_int_input("Enter the number of portfolios to analyze: ", min_value=1)

    tailor_portfolio = get_yes_no_input("Would you like to tailor the table to your portfolio value? (yes/no): ")

    frac_shares = get_yes_no_input("Would you like to tailor the table using fractional shares? (yes/no): ")

    total_value = None
    if tailor_portfolio:
        total_value = get_float_input("Enter the total value for the combined portfolio: ", min_value=0.01) # Ensure positive value

    all_portfolio_inputs = []
    portfolio_weights = []

    # Portfolio Input Loop (Weight and Tickers)
    for i in range(1, num_portfolios + 1):
        current_portfolio_input = {}
        # Weight Input
        if i == num_portfolios: # Last portfolio
            if num_portfolios == 1: current_portfolio_weight = 100.0
            else: current_portfolio_weight = 100.0 - sum(portfolio_weights)

            if current_portfolio_weight < -0.01: # Allow for small floating point inaccuracies
                print(f"Error: Previous weights exceed 100% ({sum(portfolio_weights):.2f}%). Please try again.")
                return # Exit command handler
            current_portfolio_weight = max(0.0, current_portfolio_weight) # Ensure weight isn't negative
            if num_portfolios > 1: print(f"Weight for final Portfolio {i} automatically set to {current_portfolio_weight:.2f}%.")
        else: # Not the last portfolio
            remaining_weight = 100.0 - sum(portfolio_weights)
            current_portfolio_weight = get_float_input(f"Enter weight for Portfolio {i} (0-{remaining_weight:.2f}). Remaining: {remaining_weight:.2f}%: ", min_value=0, max_value=remaining_weight + 0.01) # Allow slight overshoot for float math
            portfolio_weights.append(current_portfolio_weight)

        current_portfolio_input['weight'] = current_portfolio_weight

        # Ticker Input
        tickers_str = get_user_input(f"Enter tickers for Portfolio {i} (comma-separated): ", validation=lambda r: r and r.strip(), error_message="Tickers cannot be empty.")
        tickers = [ticker.strip().upper() for ticker in tickers_str.split(',') if ticker.strip()]
        if not tickers:
             print(f"Tickers cannot be empty for Portfolio {i}. Please try again.")
             return # Exit command handler
        current_portfolio_input['tickers'] = tickers
        all_portfolio_inputs.append(current_portfolio_input)

    # Validate total weight sums to 100% (within tolerance)
    total_weight_check = sum(p['weight'] for p in all_portfolio_inputs)
    if not math.isclose(total_weight_check, 100.0, abs_tol=0.1):
         print(f"Warning: Total portfolio weight sums to {total_weight_check:.2f}%, not 100%. Proceeding anyway, but results may be unexpected.")

    # Convert collected inputs to the dictionary format expected by process_custom_portfolio
    portfolio_data_dict = {
         'risk_type': 'stock',             # FIXED as per original logic
         'risk_tolerance': '10',           # FIXED as per original logic
         'ema_sensitivity': str(ema_sensitivity),
         'amplification': str(amplification),
         'num_portfolios': str(num_portfolios),
         'frac_shares': str(frac_shares).lower(), # Convert bool to 'true'/'false' string
         'remove_amplification_cap': 'true' # FIXED as per original logic
    }
    # Add the collected tickers and weights for each portfolio
    for i, p_data in enumerate(all_portfolio_inputs):
         portfolio_data_dict[f'tickers_{i+1}'] = ",".join(p_data['tickers'])
         portfolio_data_dict[f'weight_{i+1}'] = f"{p_data['weight']:.2f}" # Format weight

    # Call the main processing function
    try:
         # For /invest, is_custom_command_without_save is always False, so it always shows full tables
         await process_custom_portfolio(
             portfolio_data=portfolio_data_dict,
             tailor_portfolio=tailor_portfolio,    # Pass boolean from command input
             frac_shares=frac_shares,         # Pass boolean from command input
             total_value=total_value,         # Pass collected value or None
             is_custom_command_without_save=False # Always False for /invest, so full output is shown
         )
         print("--- /invest analysis complete. ---")
    except Exception as e:
         print(f"An error occurred during the analysis: {e}")
         import traceback
         traceback.print_exc()


# --- /custom Command Handler (Adapted for Terminal) ---
async def handle_custom_command(args):
    """
    Handles the /custom command for the terminal. Allows running or saving
    a custom portfolio configuration.
    """
    print("\n--- Running /custom Command ---")

    portfolio_db_file = 'portfolio_codes_database.csv'
    portfolio_code = None
    save_code = None

    if not args:
        print("Usage: /custom <portfolio_code> [save_code=3725]")
        return

    portfolio_code = args[0].strip()

    if len(args) > 1:
        save_arg = args[1].split('=')
        if len(save_arg) == 2 and save_arg[0].lower() == 'save_code':
            save_code = save_arg[1]
        else:
            print(f"Invalid argument: {args[1]}. Usage: /custom <portfolio_code> [save_code=3725]")
            return

    is_new_code_auto = False # Flag for auto-generated code

    # --- Handle '#' input for portfolio_code ---
    if portfolio_code == '#':
        next_code_num = 1 # Default if file doesn't exist or has no numeric codes
        if os.path.exists(portfolio_db_file):
            max_code = 0
            try:
                with open(portfolio_db_file, 'r', encoding='utf-8', newline='') as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        code_val = row.get('portfolio_code','').strip()
                        if code_val.isdigit():
                            max_code = max(max_code, int(code_val))
                next_code_num = max_code + 1
            except Exception as e:
                print(f"Error reading portfolio db to find next code: {e}. Defaulting to 1.")
                next_code_num = 1
        portfolio_code = str(next_code_num)
        is_new_code_auto = True
        print(f"Using next available portfolio code: `{portfolio_code}`")

    # --- Save Data Action ---
    if save_code == "3725":
        if is_new_code_auto:
            print("Cannot use '#' with save_code. Please provide an existing code to save.")
            return
        # Call the save function directly. It will read the config and process internally.
        await save_portfolio_data_terminal(portfolio_code)
        print("--- /custom save process completed. ---")
        return # Exit after attempting save

    # --- Run Analysis Action ---
    portfolio_data = None # This will store the configuration read from the CSV
    file_exists = os.path.isfile(portfolio_db_file)

    # --- Attempt to Read Existing Config ---
    if file_exists and not is_new_code_auto: # Don't read if we just generated the code
        try:
             with open(portfolio_db_file, 'r', encoding='utf-8', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row.get('portfolio_code', '').strip().lower() == portfolio_code.lower():
                        portfolio_data = row # Store original row data (dictionary)
                        break
        except Exception as e:
            print(f"Error reading existing portfolio database {portfolio_db_file}: {e}")
            print("Error accessing portfolio database.")
            return

    # --- Create New Config if Not Found (or if '#' was used) ---
    if portfolio_data is None:
        create_message = f"Portfolio code '{portfolio_code}' not found." if file_exists and not is_new_code_auto else \
                         f"Portfolio database '{portfolio_db_file}' not found." if not file_exists else \
                         f"Creating new portfolio with auto-generated code '{portfolio_code}'." if is_new_code_auto else \
                         f"Portfolio code '{portfolio_code}' not found." # Fallback

        print(f"{create_message} Let's create it.")

        # Collect inputs for the new portfolio configuration
        new_portfolio_data = await collect_portfolio_inputs_terminal(portfolio_code) # Pass the determined code

        if new_portfolio_data:
             try:
                # Save the new configuration to the database CSV
                await save_portfolio_to_csv(portfolio_db_file, new_portfolio_data)
                portfolio_data = new_portfolio_data # Use the newly collected data for processing
                print(f"New portfolio configuration '{portfolio_code}' saved.")
             except Exception as e:
                 print(f"Error saving new portfolio config for {portfolio_code}: {e}")
                 print("Error saving new portfolio configuration. Cannot proceed with analysis.")
                 return
        else:
            # If collect_portfolio_inputs_terminal returns None, it means the user cancelled.
            print("Portfolio configuration creation cancelled.")
            return # Exit if collection failed or was cancelled

    # --- Process the Found or Newly Created Portfolio ---
    if portfolio_data:
        try:
            # Determine if tailoring is requested by asking for a total value.
            tailor_portfolio_requested = False
            total_value = None

            # Prompt for total value only if save_code was NOT entered (as per original v2.5.4.0 logic)
            # This implies that if save_code is entered, tailoring is NOT done via this command execution.
            # If the user wants to save the *tailored* output, a separate command or flow would be needed.
            # Sticking to the original v2.5.4.0 logic for now: tailoring only happens if save_code is NOT provided.
            tailor_portfolio = (save_code != "3725") # True if save_code is NOT entered

            # Prompt for total value only if tailoring is enabled (i.e., save_code was NOT entered)
            if tailor_portfolio:
                value_str = get_user_input("Enter the total portfolio value to tailor (enter 0 or skip to skip tailoring): ", validation=lambda r: r.replace('.', '', 1).isdigit() and float(r) >= 0, error_message="Invalid number format.")
                total_value = float(value_str)
                if total_value > 0:
                    tailor_portfolio_requested = True # User successfully provided a positive value
                else:
                    print("Proceeding without tailoring.")
                    tailor_portfolio = False # Turn off tailoring *for this run*
                    total_value = None
            else:
                 # If tailoring is not enabled (save_code was 3725), total_value remains None
                 pass


            # Get frac_shares from the loaded portfolio_data config
            frac_shares = portfolio_data.get('frac_shares', 'no').lower() == 'yes'

            # Notify user before processing
            print(f"Processing custom portfolio code: `{portfolio_code}`...")

            # Call the main processing function
            # Pass parameters correctly based on whether tailoring was requested (by providing value)
            # The is_custom_command_without_save flag controls whether simplified output is shown
            # and is True if tailoring is happening AND save_code was NOT 3725.
            # This matches the original v2.5.4.0 logic for custom command output.
            await process_custom_portfolio(
                portfolio_data=portfolio_data, # Pass the config read from CSV
                tailor_portfolio=tailor_portfolio_requested, # Pass boolean indicating if value was provided
                frac_shares=frac_shares, # Pass the correctly determined boolean
                total_value=total_value, # Pass collected value or None
                is_custom_command_without_save=(tailor_portfolio_requested and save_code != "3725")
            )

            print(f"--- Custom portfolio analysis for `{portfolio_code}` complete. ---")

        except KeyError as e:
            print(f"Error: Incomplete configuration for portfolio code {portfolio_code}. Missing key: {e}")
            print(f"Configuration for portfolio code '{portfolio_code}' seems incomplete. Please check the `{portfolio_db_file}` or recreate the code.")
        except Exception as e:
            print(f"An unexpected error occurred while processing portfolio '{portfolio_code}': {e}")
            import traceback
            traceback.print_exc()

# --- Portfolio Saving Functions (Adapted for Terminal) ---

async def save_portfolio_to_csv(file_path, portfolio_data):
    # Saves the config from collect_portfolio_inputs to portfolio_codes_database.csv
    file_exists = os.path.isfile(file_path)
    fieldnames = list(portfolio_data.keys())

    # Exclude 'tailor_portfolio' field if present (shouldn't be in terminal version anyway)
    if 'tailor_portfolio' in fieldnames:
        fieldnames.remove('tailor_portfolio')

    try:
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
             # Ensure portfolio_code is the first column if it exists
             if 'portfolio_code' in fieldnames:
                 fieldnames.insert(0, fieldnames.pop(fieldnames.index('portfolio_code')))

             writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore') # Use extrasaction='ignore'
             if not file_exists or os.path.getsize(file_path) == 0:
                 writer.writeheader()
             # Create a temporary dict excluding the unwanted field
             data_to_save = {k: v for k, v in portfolio_data.items() if k in fieldnames}
             writer.writerow(data_to_save)
    except IOError as e:
        print(f"Error writing to CSV {file_path}: {e}")
        raise # Re-raise to indicate save failure
    except Exception as e:
        print(f"Unexpected error saving portfolio config to CSV: {e}")
        raise # Re-raise

async def save_portfolio_data_internal(portfolio_code, date_str):
    """
    Internal function to save portfolio data without interaction, using a provided date.
    """
    portfolio_db_file = 'portfolio_codes_database.csv'
    portfolio_data = None # Config read from CSV

    # --- Read Portfolio Config ---
    try:
        if not os.path.exists(portfolio_db_file):
             print(f"Error [Save]: Portfolio database '{portfolio_db_file}' not found.")
             raise FileNotFoundError(f"Portfolio database '{portfolio_db_file}' not found.") # Indicate failure

        with open(portfolio_db_file, 'r', encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file)
            found = False
            for row in reader:
                if row.get('portfolio_code', '').strip().lower() == portfolio_code.lower():
                    portfolio_data = row # Store original row data
                    found = True
                    break
            if not found:
                 print(f"Error [Save]: Portfolio code '{portfolio_code}' not found in '{portfolio_db_file}'.")
                 raise ValueError(f"Portfolio code '{portfolio_code}' not found.") # Indicate failure
    except Exception as e:
        print(f"Error [Save]: Reading portfolio database {portfolio_db_file} for {portfolio_code}: {e}")
        raise # Re-raise

    # --- Process and Save Combined Data ---
    if portfolio_data and date_str:
        try:
            frac_shares = portfolio_data.get('frac_shares', 'false').lower() == 'true'

            # Process portfolio WITHOUT tailoring to get the combined result
            # Suppress output during saving by passing tailor_portfolio=False, total_value=None, is_custom_command_without_save=False
            # MODIFIED: Unpack all 5 return values from process_custom_portfolio
            _, combined_result, _, _, _ = await process_custom_portfolio(
                portfolio_data=portfolio_data, # Pass the config read from CSV
                tailor_portfolio=False,        # Force False for saving combined data
                frac_shares=frac_shares,
                total_value=None,
                is_custom_command_without_save=False # Ensure full processing for data generation
            )

            if combined_result:
                valid_combined = [entry for entry in combined_result if 'error' not in entry]
                # Sort by combined allocation for saving (as per original intent before request change)
                sorted_combined = sorted(valid_combined, key=lambda x: safe_score(x.get('combined_percent_allocation', -1)), reverse=True)

                # --- Perform the Save ---
                # This saves the COMBINED PORTFOLIO RESULTS generated by the portfolio code's config for the given date.
                save_file = f"portfolio_code_{portfolio_code}_data.csv" # v2.5.2.0 filename
                file_exists = os.path.isfile(save_file)
                save_count = 0
                # Headers: DATE, TICKER, PRICE, COMBINED_ALLOCATION_PERCENT
                headers = ['DATE', 'TICKER', 'PRICE', 'COMBINED_ALLOCATION_PERCENT']
                with open(save_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    if not file_exists or os.path.getsize(save_file) == 0: writer.writeheader()
                    for item in sorted_combined:
                        ticker = item.get('ticker', 'ERR')
                        price_val = item.get('live_price')
                        alloc_val = item.get('combined_percent_allocation')

                        price_str = f"{safe_score(price_val):.2f}" if price_val is not None else "N/A"
                        alloc_str = f"{safe_score(alloc_val):.2f}" if alloc_val is not None else "N/A"

                        writer.writerow({
                            'DATE': date_str, # Use the provided date
                            'TICKER': ticker,
                            'PRICE': price_str,
                            'COMBINED_ALLOCATION_PERCENT': alloc_str
                        })
                        save_count += 1
                print(f"[Save]: Saved {save_count} rows of combined portfolio data for code '{portfolio_code}' to '{save_file}' for date {date_str}.")
            else:
                print(f"[Save]: No valid combined portfolio data generated for code '{portfolio_code}'.")
                raise ValueError("No valid combined portfolio data generated for save.") # Indicate failure

        except Exception as e:
            print(f"Error [Save]: Processing/saving for code {portfolio_code}: {e}")
            raise # Re-raise the exception to be caught by the calling function

async def save_portfolio_data_terminal(portfolio_code):
    """
    Saves the combined portfolio output of process_custom_portfolio for a given code.
    Prompts user for date via terminal.
    """
    print(f"Attempting to save combined data for portfolio code: `{portfolio_code}`...")

    # --- Get Save Date ---
    save_date_str = get_user_input("Enter the date to save the data under (MM/DD/YYYY): ", validation=lambda d: datetime.strptime(d, '%m/%d/%Y') if True else False, error_message="Invalid date format. Please use MM/DD/YYYY.")
    # Validation function above is a bit hacky, but works with get_user_input. A better way would be:
    # def validate_date(date_str):
    #     try:
    #         datetime.strptime(date_str, '%m/%d/%Y')
    #         return True
    #     except ValueError:
    #         return False
    # save_date_str = get_user_input("Enter the date to save the data under (MM/DD/YYYY): ", validate_date, "Invalid date format. Please use MM/DD/YYYY.")


    # --- Call internal save logic ---
    try:
        # Use the internal function which handles reading config and processing
        await save_portfolio_data_internal(portfolio_code, save_date_str)
        # Success message is printed by internal function
    except Exception as e:
        # Error handling is done inside the internal function, but catch any unexpected ones
        print(f"An error occurred while saving data for portfolio code '{portfolio_code}': {e}")


# --- /market Command Handler (Adapted for Terminal) ---
async def handle_market_command(args):
    """
    Handles the /market command for the terminal. Gets top/bottom SPY stocks
    or saves full market data.
    """
    print("\n--- Running /market Command ---")

    save_code = None
    if args:
        save_arg = args[0].split('=')
        if len(save_arg) == 2 and save_arg[0].lower() == 'save_code':
            save_code = save_arg[1]
        else:
            print(f"Invalid argument: {args[0]}. Usage: /market [save_code=3725]")
            return

    # In the terminal, we need to ask for sensitivity
    sensitivity_value = get_int_input("Enter Market Sensitivity (1, 2, or 3): ", min_value=1, max_value=3)

    # Fetch symbols once needed
    spy_symbols = get_spy_symbols()
    if not spy_symbols:
        print("Error: Failed to retrieve S&P 500 symbols.")
        return

    if save_code == "3725":
        # --- Manual Save Action (Save FULL Data) ---
        save_date_str = get_user_input("Enter date (MM/DD/YYYY) to save FULL market data under: ", validation=lambda d: datetime.strptime(d, '%m/%d/%Y') if True else False, error_message="Invalid date format. Use MM/DD/YYYY.")
        if save_date_str:
             try:
                 print(f"Calculating & saving FULL market data (Sens: {sensitivity_value}). This may take a while...")
                 # Use the internal save function, ensuring save_full_data=True (default)
                 # Pass a dummy channel_id as it's not used in the terminal version's save_market_data_internal
                 await save_market_data_internal(0, save_date_str, save_full_data=True, symbols=spy_symbols) # Pass symbols
                 print(f"Save process completed for FULL market data (Sens {sensitivity_value}) for {save_date_str}.")
             except Exception as e:
                 print(f"Error saving full market data: {e}.")
        print("--- /market save process completed. ---")
        return # Exit after save attempt

    else: # Display Action
        try:
            print(f"Calculating market scores (Sens: {sensitivity_value}). This may take some time...")
            # Use the internal save function with save_full_data=False to get the data without saving
            # Pass a dummy channel_id and date_str as they are not used when save_full_data=False
            # Pass symbols
            all_scores_data = calculate_market_invest_scores(spy_symbols, sensitivity_value) # Direct call to calculation

            if not all_scores_data:
                 print("Error calculating market scores or no data returned."); return
            valid_scores = [item for item in all_scores_data if item.get('score') is not None]
            if not valid_scores:
                print("No valid scores could be calculated for display."); return

            top_scores = valid_scores[:10]
            bottom_scores = valid_scores[-10:]

            # Calculate SPY score separately for display
            spy_result_list = calculate_market_invest_scores(['SPY'], sensitivity_value)
            spy_result = spy_result_list[0] if spy_result_list and spy_result_list[0].get('score') is not None else None

            def format_row(item_dict):
                ticker = item_dict.get('ticker', 'ERR')
                price_val = item_dict.get('live_price')
                score_val = item_dict.get('score') # Potentially negative raw score
                price = f"${safe_score(price_val):,.2f}" if isinstance(price_val, (int, float)) else str(price_val) # Added comma formatting
                # Format the potentially negative score
                score = f"{safe_score(score_val):.2f}%" if score_val is not None else "N/A"
                return [ticker, price, score]

            top_table_data = [format_row(r) for r in top_scores]
            bottom_table_data = [format_row(r) for r in bottom_scores]

            print(f"\n**Top 10 SPY Stocks (Sens: {sensitivity_value})**")
            if top_table_data:
                print(tabulate(top_table_data, headers=["Ticker", "Price", "Score"], tablefmt="pretty"))
            else:
                print("No top scores data.")

            print(f"\n**Bottom 10 SPY Stocks (Sens: {sensitivity_value})**")
            if bottom_table_data:
                print(tabulate(bottom_table_data, headers=["Ticker", "Price", "Score"], tablefmt="pretty"))
            else:
                print("No bottom scores data.")

            print(f"\n**SPY Score (Sens: {sensitivity_value})**")
            if spy_result:
                print(tabulate([format_row(spy_result)], headers=["Ticker", "Price", "Score"], tablefmt="pretty"))
            else:
                print("Error calculating SPY score.")

        except Exception as e:
            print(f"Error during market display: {e}")
            import traceback; traceback.print_exc()

        print("--- /market analysis complete. ---")


# --- Breakout Related Functions (Adapted for Terminal) ---

async def save_breakout_data_internal(date_str):
    """Internal function to save breakout data without interaction, using a provided date."""
    file_path = "breakout_tickers.csv"
    save_file = "breakout_historical_database.csv"
    save_count = 0

    # Check if the current breakout data exists
    if not os.path.exists(file_path):
        print(f"Error [Save]: Breakout data file '{file_path}' not found.")
        raise FileNotFoundError(f"Breakout data file '{file_path}' not found.") # Indicate failure

    try:
        # Read the current breakout data
        df_current = pd.read_csv(file_path)
        # Ensure correct types for saving (remove % and $ signs)
        for col in ["Highest Invest Score", "Lowest Invest Score", "Live Price", "1Y% Change", "Invest Score"]:
             if col in df_current.columns and df_current[col].dtype == 'object':
                 df_current[col] = df_current[col].astype(str).str.replace('%', '', regex=False).str.replace('$', '', regex=False).str.strip()
                 df_current[col] = pd.to_numeric(df_current[col], errors='coerce') # Convert to numeric, coercing errors

        file_exists = os.path.isfile(save_file)
        headers = ['DATE', 'TICKER', 'PRICE', 'INVEST_SCORE']

        with open(save_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists or os.path.getsize(save_file) == 0:
                writer.writeheader()

            for index, row in df_current.iterrows():
                ticker = row.get('Ticker', 'ERR')
                price_val = row.get('Live Price')
                score_val = row.get('Invest Score')

                # Format data for saving
                price_str = f"{safe_score(price_val):.2f}" if price_val is not None else "N/A"
                score_str = f"{safe_score(score_val):.2f}" if score_val is not None else "N/A" # Save raw score

                writer.writerow({
                    'DATE': date_str,
                    'TICKER': ticker,
                    'PRICE': price_str,
                    'INVEST_SCORE': score_str
                })
                save_count += 1
        print(f"[Save]: Saved {save_count} rows to '{save_file}' for breakout data for date {date_str}.")

    except pd.errors.EmptyDataError:
        print(f"Warning [Save]: Breakout source file '{file_path}' is empty. Nothing saved.")
        raise ValueError(f"Breakout source file '{file_path}' is empty.") # Indicate failure
    except KeyError as e:
        print(f"Warning [Save]: Missing expected column in '{file_path}': {e}. Cannot save breakout data.")
        raise KeyError(f"Missing expected column in '{file_path}': {e}") # Indicate failure
    except IOError as e:
        print(f"Error [Save]: Writing to breakout save file '{save_file}': {e}")
        raise # Re-raise for the main loop to catch
    except Exception as e:
        print(f"Error [Save]: Processing/saving breakout data: {e}")
        raise # Re-raise for the main loop to catch

async def save_breakout_data_terminal():
    """Saves the current breakout data after prompting the user for a date via terminal."""
    print("Attempting to save current breakout data...")

    save_date_str = get_user_input("Enter the date to save this data under (MM/DD/YYYY): ", validation=lambda d: datetime.strptime(d, '%m/%d/%Y') if True else False, error_message="Invalid date format. Use MM/DD/YYYY.")

    # Call the internal saving logic
    try:
        await save_breakout_data_internal(save_date_str)
        # Success message is printed by internal function
    except Exception as e:
        print(f"An error occurred while saving breakout data: {e}")


# --- Breakout Logic (Main Function - Adapted for Terminal) ---
# This function now takes no 'interaction' object and prints directly
async def breakout_logic(save_data_override: bool = False):
    """
    Performs breakout analysis, updates breakout_tickers.csv, and optionally saves historical data.
    Args:
        save_data_override: If True, triggers the save data logic (used by /breakout save_code).
    """
    file_path = "breakout_tickers.csv"
    invest_score_threshold = 100
    fraction_threshold = 3 / 4

    # --- Save Action (if triggered by command) ---
    if save_data_override:
        await save_breakout_data_terminal()
        return # Exit after saving

    # --- Analysis Action ---
    print("--- Running Breakout Analysis ---")
    try:
        existing_tickers_data = {}
        if os.path.exists(file_path):
            try:
                df_existing = pd.read_csv(file_path)
                # Convert relevant columns safely
                for col in ["Highest Invest Score", "Lowest Invest Score", "Live Price", "1Y% Change", "Invest Score"]:
                    if col in df_existing.columns:
                        if df_existing[col].dtype == 'object':
                             # Remove formatting before converting
                             df_existing[col] = df_existing[col].astype(str).str.replace('%', '', regex=False).str.replace('$', '', regex=False).str.strip()
                        df_existing[col] = pd.to_numeric(df_existing[col], errors='coerce')
                existing_tickers_data = df_existing.set_index('Ticker').to_dict('index')
            except pd.errors.EmptyDataError: print(f"Warning: {file_path} is empty.")
            except KeyError: print(f"Warning: 'Ticker' column likely missing in {file_path}")
            except Exception as read_err: print(f"Error reading {file_path}: {read_err}.")

        new_tickers = []
        print("Running TradingView Screening...")
        try:
            # Adjusted query based on provided file
            query = Query().select('name', 'close', 'change', 'volume', 'market_cap_basic', 'change|1W', 'average_volume_90d_calc'
            ).where(Column('market_cap_basic') >= 1_000_000_000, Column('volume') >= 1_000_000, Column('change|1W') >= 20, Column('close') >= 1, Column('average_volume_90d_calc') >= 1_000_000
            ).order_by('change', ascending=False) # Kept order by change
            scanner_results = query.get_scanner_data()
            if scanner_results and isinstance(scanner_results, tuple) and len(scanner_results) == 2 and isinstance(scanner_results[1], pd.DataFrame):
                new_tickers_df = scanner_results[1]
                if 'name' in new_tickers_df.columns:
                    new_tickers = new_tickers_df['name'].tolist()
                    print(f"Screening found {len(new_tickers)} potential tickers.")
                else:
                    print("Warning: 'name' column not found in screening results.")
            else:
                print(f"Warning: TradingView screening returned unexpected data format or no data: {scanner_results}")
        except Exception as screen_err:
            print(f"Error during TradingView screening: {screen_err}")

        updated_data = []
        all_tickers_to_process = set(list(existing_tickers_data.keys()) + new_tickers)
        print(f"Processing {len(all_tickers_to_process)} unique tickers...")
        processed_count = 0

        for ticker in all_tickers_to_process:
            try:
                live_price, current_invest_score_raw = calculate_ema_invest(ticker, 2) # Use Daily sens (2)
                one_year_change_raw, _ = calculate_one_year_invest(ticker)
                # Store potentially negative score, handle None
                current_invest_score = current_invest_score_raw if current_invest_score_raw is not None else None
                live_price = safe_score(live_price) if live_price is not None else None
                one_year_change = safe_score(one_year_change_raw) if one_year_change_raw is not None else None
                existing_entry = existing_tickers_data.get(ticker, {})
                # Use raw scores for comparison, handle None safely
                highest_score_prev = existing_entry.get("Highest Invest Score") if pd.notna(existing_entry.get("Highest Invest Score")) else -float('inf')
                lowest_score_prev = existing_entry.get("Lowest Invest Score") if pd.notna(existing_entry.get("Lowest Invest Score")) else float('inf')
                highest_invest_score = highest_score_prev
                lowest_invest_score = lowest_score_prev

                if current_invest_score is not None:
                    # Use raw score for tracking highest/lowest
                    if highest_invest_score == -float('inf') or current_invest_score > highest_invest_score: highest_invest_score = current_invest_score
                    if lowest_invest_score == float('inf') or current_invest_score < lowest_invest_score: lowest_invest_score = current_invest_score

                remove_ticker = False
                if current_invest_score is None:
                    remove_ticker = True; print(f"Removing {ticker} due to failed score calculation.")
                # Use raw scores for removal logic checks
                elif highest_invest_score > -float('inf') and highest_invest_score > 0: # Check if a valid high exists and > 0
                    if (current_invest_score > 600 or current_invest_score < invest_score_threshold or current_invest_score < fraction_threshold * highest_invest_score):
                        remove_ticker = True; print(f"Removing {ticker}: Current={safe_score(current_invest_score):.2f}, Highest={safe_score(highest_invest_score):.2f}, Threshold={invest_score_threshold}, Fraction={fraction_threshold}")
                elif current_invest_score is not None and current_invest_score < invest_score_threshold: # Check if current score is below threshold when no valid high exists
                    remove_ticker = True; print(f"Removing {ticker}: Current={safe_score(current_invest_score):.2f}, Threshold={invest_score_threshold} (no previous high)")
                elif current_invest_score is None and ticker in existing_tickers_data:
                     # If score calculation failed for an existing ticker, remove it.
                     remove_ticker = True; print(f"Removing {ticker} (existing) due to failed score calculation.")


                if not remove_ticker:
                    status = "Repeat" if ticker in existing_tickers_data else "New"
                    updated_data.append({
                        "Ticker": ticker,
                        "Live Price": f"{live_price:.2f}" if live_price is not None else "N/A",
                        # Format potentially negative score for CSV/display
                        "Invest Score": f"{current_invest_score:.2f}%" if current_invest_score is not None else "N/A",
                        "Highest Invest Score": f"{highest_invest_score:.2f}%" if highest_invest_score > -float('inf') else "N/A",
                        "Lowest Invest Score": f"{lowest_invest_score:.2f}%" if lowest_invest_score < float('inf') else "N/A",
                        "1Y% Change": f"{one_year_change:.2f}%" if one_year_change is not None else "N/A",
                        "Status": status,
                    })
            except Exception as e: print(f"Error processing breakout logic for {ticker}: {e}")
            processed_count += 1
            if processed_count % 25 == 0: print(f"  ...processed {processed_count}/{len(all_tickers_to_process)}")

        print("Sorting and saving breakout data...")
        # Sort based on the potentially negative raw score before formatting
        for item in updated_data:
            score_str = item["Invest Score"].replace('%', '') if isinstance(item["Invest Score"], str) else None
            item['_sort_score'] = safe_score(score_str) # Use safe_score to handle N/A
        updated_data.sort(key=lambda x: x['_sort_score'] if x['_sort_score'] is not None else -float('inf'), reverse=True)
        for item in updated_data: del item['_sort_score'] # Remove temporary sort key

        final_columns = ["Ticker", "Live Price", "Invest Score", "Highest Invest Score", "Lowest Invest Score", "1Y% Change", "Status"]
        final_df = pd.DataFrame(updated_data, columns=final_columns)
        try: final_df.to_csv(file_path, index=False)
        except IOError as e: print(f"Error writing breakout data to {file_path}: {e}")

        # Send output to Terminal
        print("Formatting output table...")
        # Format the potentially negative score for display
        table_data_output = [[row["Ticker"], f"${safe_score(row['Live Price']):.2f}" if row['Live Price'] != "N/A" else "N/A", row["Invest Score"], row["Highest Invest Score"], row["Lowest Invest Score"], row["Status"]] for row in updated_data] # Use safe_score for price formatting
        if not table_data_output: result_table = "No tickers currently meet the breakout criteria."
        else: result_table = tabulate(table_data_output, headers=["Ticker", "Price", "Score", "Highest Score", "Lowest Score", "Status"], tablefmt="pretty")

        print("\n**Breakout Tickers**")
        print(f"```{result_table}```")

    except Exception as e:
        print(f"Error during breakout analysis: {e}")
        import traceback; traceback.print_exc()

    print("--- Breakout analysis complete. ---")


# --- /breakout Command Handler (Adapted for Terminal) ---
async def handle_breakout_command(args):
    """
    Handles the /breakout command for the terminal. Runs breakout analysis
    or saves the current breakout data.
    """
    print("\n--- Running /breakout Command ---")

    save_code = None
    if args:
        save_arg = args[0].split('=')
        if len(save_arg) == 2 and save_arg[0].lower() == 'save_code':
            save_code = save_arg[1]
        else:
            print(f"Invalid argument: {args[0]}. Usage: /breakout [save_code=3725]")
            return

    if save_code == "3725":
        await breakout_logic(save_data_override=True)
        print("--- /breakout save process completed. ---")
    else:
        await breakout_logic(save_data_override=False)
        print("--- /breakout analysis completed. ---")


# --- Assess Related Functions (Adapted for Terminal) ---

# Need functions for Assess Code A, B, C, D logic
# These are adapted from the provided INVEST and Singularity files.

# Assess Code A Helper
def calculate_volatility_and_correspondence(ticker, timeframe_code, risk_tolerance):
    """Calculates volatility (AAPC) and checks correspondence for Assess Code A."""
    ticker_str = ticker.replace('.', '-')
    stock = yf.Ticker(ticker_str, session=YFINANCE_SESSION)

    period_map = {'1': '1y', '2': '3mo', '3': '1mo'}
    selected_period = period_map.get(timeframe_code, '1y')

    try:
        hist = stock.history(period=selected_period, interval="1d")
        if hist.empty or 'Close' not in hist.columns:
            print(f"Warning: Insufficient data for {ticker_str} over {selected_period} for Assess A.")
            return None, None, None, None, None # Return None for all results

        # Calculate AAPC (Average Absolute Percentage Change)
        hist['Daily_Change'] = hist['Close'].pct_change()
        # Handle potential inf values from division by zero (e.g., penny stocks)
        hist.replace([np.inf, -np.inf], np.nan, inplace=True)
        hist.dropna(subset=['Daily_Change'], inplace=True) # Drop rows where daily change is NaN (first row) or inf

        if hist['Daily_Change'].empty:
             print(f"Warning: No valid daily changes for {ticker_str} over {selected_period} for Assess A.")
             return None, None, None, None, None

        aabc = hist['Daily_Change'].abs().mean() * 100 # Average Absolute Daily Change in percentage

        # Map AABC to Volatility Score (0-10)
        # This mapping needs to be consistent with the original logic
        # Assuming a linear mapping or thresholds based on typical volatility ranges
        # This mapping is derived from the logic in the provided Assess Code A snippet
        aabc_val = safe_score(aabc)
        score = 0
        if aabc_val > 0: # Only score if AABC is positive
            if aabc_val <= 1: score = 0
            elif aabc_val <= 2: score = 1
            elif aabc_val <= 3: score = 2
            elif aabc_val <= 4: score = 3
            elif aabc_val <= 5: score = 4
            elif aabc_val <= 6: score = 5
            elif aabc_val <= 7: score = 6
            elif aabc_val <= 8: score = 7
            elif aabc_val <= 9: score = 8
            elif aabc_val <= 10: score = 9
            else: score = 10

        # Calculate Period Change
        start_price = safe_score(hist['Close'].iloc[0])
        end_price = safe_score(hist['Close'].iloc[-1])
        period_change = ((end_price - start_price) / start_price) * 100 if start_price != 0 else 0.0


        # Check Correspondence
        correspondence = "No Match"
        # Risk tolerance mapping to volatility score ranges (from provided snippet)
        risk_map = {1: [0, 1], 2: [2, 3], 3: [4, 5], 4: [6, 7], 5: [8, 9, 10]}
        if risk_tolerance in risk_map and score in risk_map[risk_tolerance]:
            correspondence = "Matches"

        return period_change, aabc_val, score, correspondence, selected_period

    except Exception as e:
        print(f"Error assessing ticker {ticker_str} for Code A: {e}")
        return None, None, None, None, None # Return None on error


# Assess Code B Helper (Manual Portfolio)
def calculate_portfolio_metrics(portfolio_tickers, shares_owned, cash_amount, backtest_period):
    """Calculates weighted Beta and Correlation for a manual portfolio."""
    print(f"Calculating portfolio metrics over {backtest_period}...")

    all_tickers_for_data = list(set(portfolio_tickers + ['SPY'])) # Ensure SPY is included
    hist_data, live_prices = None, {}

    try:
        # Fetch historical data for all tickers + SPY
        hist_data = yf.download(all_tickers_for_data, period=backtest_period, interval="1d", progress=False, session=YFINANCE_SESSION)
        if hist_data.empty or 'Close' not in hist_data.columns:
             print("Error fetching historical data for Code B.")
             return None, None, None # Indicate failure

        # Fetch live prices separately for current value calculation
        # Use a short period like 1 day with 1 minute interval to get recent price
        live_prices_data = yf.download(portfolio_tickers, period="1d", interval="1m", progress=False, session=YFINANCE_SESSION)

        if isinstance(live_prices_data.columns, pd.MultiIndex):
             for ticker_lp in portfolio_tickers:
                 if ('Close', ticker_lp) in live_prices_data.columns:
                     # Get the last valid price, handling potential NaNs
                     latest_price = live_prices_data['Close'][ticker_lp].dropna().iloc[-1] if not live_prices_data['Close'][ticker_lp].dropna().empty else np.nan
                     live_prices[ticker_lp] = latest_price
                 else:
                     live_prices[ticker_lp] = np.nan # Ticker column not found
        elif 'Close' in live_prices_data.columns and len(portfolio_tickers) == 1:
             # Case for a single ticker where yfinance might not return MultiIndex
             latest_price = live_prices_data['Close'].dropna().iloc[-1] if not live_prices_data['Close'].dropna().empty else np.nan
             live_prices[portfolio_tickers[0]] = latest_price
        else:
             print("Warning: Could not fetch live prices for all tickers for Code B.")
             for ticker_lp in portfolio_tickers: live_prices[ticker_lp] = np.nan # Initialize with NaN

    except Exception as e_data:
        print(f"Error fetching market data for Code B: {e_data}")
        return None, None, None # Indicate failure

    # Extract Close prices, handling MultiIndex vs single column
    if isinstance(hist_data.columns, pd.MultiIndex):
        close_prices = hist_data['Close']
    elif 'Close' in hist_data.columns:
         # Handle case where only one ticker + SPY was requested and SPY failed,
         # or only one ticker succeeded, resulting in a single 'Close' column.
         # Need to ensure the column name is the ticker name if it's not SPY.
         if len(all_tickers_for_data) == 1 and all_tickers_for_data[0] in hist_data.columns:
              close_prices = hist_data[[all_tickers_for_data[0]]] # Select the single ticker column
         elif 'Close' in hist_data.columns and len(all_tickers_for_data) > 0:
              # If there's a 'Close' column but not MultiIndex, and we requested tickers,
              # assume it's the first ticker requested if only one succeeded. This is fragile.
              # A more robust approach would iterate through requested tickers and check if they exist as columns.
              # For simplicity here, we'll just select the 'Close' column and hope it's the relevant one if not SPY.
              # A better fix would involve checking column names carefully.
              close_prices = hist_data[['Close']]
              # Attempt to rename the column if it's not SPY and there's only one stock requested
              if len(portfolio_tickers) == 1 and portfolio_tickers[0] != 'SPY' and close_prices.columns[0] == 'Close':
                   close_prices.columns = [portfolio_tickers[0]]
         else:
              print("Error: Unexpected historical data format after fetch for Code B.")
              return None, None, None


    # Clean column names (remove . if present)
    close_prices.columns = [col.replace('.', '-') for col in close_prices.columns]

    # Drop tickers with insufficient data points for the period
    min_data_points = 20 # Arbitrary minimum threshold
    close_prices = close_prices.dropna(axis=1, thresh=min_data_points)

    if 'SPY' not in close_prices.columns:
        print("Error: SPY historical data insufficient for Code B calculation.")
        return None, None, None # Indicate failure

    # Calculate daily returns
    daily_returns = close_prices.pct_change().dropna()

    if daily_returns.empty or 'SPY' not in daily_returns.columns:
        print("Error calculating daily returns for Code B.")
        return None, None, None # Indicate failure

    spy_returns = daily_returns['SPY']
    if spy_returns.std() == 0:
        print("SPY had no price variance in the period. Cannot calculate metrics for Code B.")
        return None, None, None # Indicate failure

    stock_metrics = {} # To store beta and correlation for each stock
    for ticker_met in portfolio_tickers:
        if ticker_met in daily_returns.columns:
            ticker_returns = daily_returns[ticker_met]
            if ticker_returns.std() == 0: # Check for zero variance in ticker returns
                beta, correlation = 0.0, 0.0
            else:
                # Calculate Beta and Correlation
                # Ensure both series are aligned by index before calculating covariance/correlation
                aligned_returns = pd.concat([ticker_returns, spy_returns], axis=1).dropna()
                if aligned_returns.empty or len(aligned_returns.columns) < 2:
                    print(f"Warning: Insufficient aligned data for {ticker_met} vs SPY for Code B metrics.")
                    beta, correlation = np.nan, np.nan # Cannot calculate
                else:
                    covariance_matrix = np.cov(aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1])
                    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1] if covariance_matrix[1, 1] != 0 else 0.0
                    correlation = np.corrcoef(aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1])[0, 1]

            stock_metrics[ticker_met] = {'beta': beta, 'correlation': correlation}
        else:
            # Ticker data was insufficient after dropna
            stock_metrics[ticker_met] = {'beta': np.nan, 'correlation': np.nan}
            print(f"Warning: Insufficient data for {ticker_met} in Code B metrics calculation.")

    stock_metrics['Cash'] = {'beta': 0.0, 'correlation': 0.0} # Cash has 0 beta and correlation

    # Calculate total portfolio value and individual holding values
    portfolio_holdings = []
    total_portfolio_value = safe_score(cash_amount) # Start with cash

    for ticker_h in portfolio_tickers:
        shares = safe_score(shares_owned.get(ticker_h, 0))
        # Use fetched live price if available, otherwise fallback to last historical price
        live_price = safe_score(live_prices.get(ticker_h, np.nan))
        if pd.isna(live_price) and ticker_h in close_prices.columns:
            # Fallback to last historical price if live price fetch failed
            live_price = safe_score(close_prices[ticker_h].dropna().iloc[-1] if not close_prices[ticker_h].dropna().empty else np.nan)

        holding_value = 0.0
        if not pd.isna(live_price) and shares > 0:
            holding_value = shares * live_price

        total_portfolio_value += holding_value

        portfolio_holdings.append({
            'ticker': ticker_h,
            'shares': shares,
            'live_price': live_price,
            'holding_value': holding_value,
            **stock_metrics.get(ticker_h, {}) # Add calculated metrics
        })

    # Add Cash holding
    portfolio_holdings.append({
        'ticker': 'Cash',
        'shares': np.nan, # Shares not applicable for cash
        'live_price': 1.0, # Price is 1 for cash
        'holding_value': safe_score(cash_amount),
        **stock_metrics['Cash'] # Add cash metrics
    })

    if total_portfolio_value <= 0:
        print("Total portfolio value is zero or negative for Code B. Cannot calculate weighted averages.")
        return None, None, None # Indicate failure

    # Calculate weighted averages
    weighted_beta_sum, weighted_correlation_sum = 0.0, 0.0
    for holding in portfolio_holdings:
        # Ensure holding_value is a number before calculating weight
        holding_value = safe_score(holding.get('holding_value', 0.0))
        weight = holding_value / total_portfolio_value if total_portfolio_value > 0 else 0

        # Ensure beta and correlation are numbers before summing
        beta_val = safe_score(holding.get('beta', np.nan))
        correlation_val = safe_score(holding.get('correlation', np.nan))

        if not pd.isna(beta_val):
            weighted_beta_sum += weight * beta_val
        else:
            print(f"Warning: Skipping {holding.get('ticker', 'Unknown')} in Beta calculation due to invalid Beta.")

        if not pd.isna(correlation_val):
            weighted_correlation_sum += weight * correlation_val
        else:
            print(f"Warning: Skipping {holding.get('ticker', 'Unknown')} in Correlation calculation due to invalid Correlation.")


    return total_portfolio_value, weighted_beta_sum, weighted_correlation_sum


# Assess Code C Helper (Custom Portfolio Risk)
# This logic is integrated into the handle_assess_command for Code C

# Assess Code D Helper (Cultivate Portfolio Risk)
# This logic is integrated into the handle_assess_command for Code D
# Requires Cultivate functions: calculate_cultivate_formulas, screen_stocks, get_yf_data,
# calculate_metrics, select_tickers, build_and_process_portfolios, HEDGING_TICKERS

# Define HEDGING_TICKERS (from Cultivate logic)
HEDGING_TICKERS = ['SH', 'PSQ', 'REMX', 'GLD', 'SLV', 'USO', 'VXX'] # Example list, adjust as needed

# Cultivate Helper Functions (from Cultivate logic)
def calculate_cultivate_formulas(sigma):
    """Calculates Invest Omega formula variables based on Sigma."""
    # This is a simplified placeholder based on the structure, actual formulas needed
    sigma = safe_score(sigma)
    if sigma is None: return None

    # Placeholder formulas - replace with actual Invest Omega formulas
    delta = max(1.0, min(5.0, sigma / 20.0)) # Example: Delta between 1 and 5 based on Sigma
    omega = max(0.0, min(100.0, 100.0 - sigma)) # Example: Omega decreases as Sigma increases
    lambda_val = 100.0 - omega # Lambda is the non-cash portion
    lambda_hedge = max(0.0, min(lambda_val, sigma)) # Example: Lambda Hedge related to Sigma and Lambda
    kappa = max(0.0, min(100.0, sigma * 1.5)) # Example: Kappa increases with Sigma
    eta = max(0.0, min(100.0, 100.0 - kappa)) # Example: Eta is inverse of Kappa

    # Placeholder for other Greeks if needed by build_and_process_portfolios or output
    alpha = 50.0
    beta_alloc = 1.0
    mu_center, mu_range = 50.0, (40.0, 60.0)
    rho_center, rho_range = 0.5, (0.4, 0.6)
    omega_target_center, omega_target_range = omega, (max(0, omega-10), min(100, omega+10))


    formula_results = {
        'sigma': sigma,
        'delta': delta, # Amplification
        'lambda': lambda_val, # Non-cash allocation percentage
        'omega': omega, # Cash allocation percentage
        'lambda_hedge': lambda_hedge, # Percentage allocated to hedging within non-cash
        'kappa': kappa, # Related to market risk appetite?
        'eta': eta, # Related to non-hedging allocation within non-cash
        'alpha': alpha, # Placeholder
        'beta_alloc': beta_alloc, # Placeholder
        'mu_center': mu_center, # Placeholder
        'mu_range': mu_range, # Placeholder
        'rho_center': rho_center, # Placeholder
        'rho_range': rho_range, # Placeholder
        'omega_target_center': omega_target_center, # Placeholder
        'omega_target_range': omega_target_range, # Placeholder
    }
    return formula_results

def screen_stocks():
    """Screens stocks using TradingView criteria for Cultivate Code A."""
    print("Running TradingView Screener for Cultivate Code A...")
    try:
        # Use the TradingView screener query from the INVEST file
        query = Query().select('name', 'close', 'change', 'volume', 'market_cap_basic', 'change|1W', 'average_volume_90d_calc'
        ).where(Column('market_cap_basic') >= 1_000_000_000, Column('volume') >= 1_000_000, Column('change|1W') >= 20, Column('close') >= 1, Column('average_volume_90d_calc') >= 1_000_000
        ).order_by('change', ascending=False) # Kept order by change
        scanner_results = query.get_scanner_data()
        if scanner_results and isinstance(scanner_results, tuple) and len(scanner_results) == 2 and isinstance(scanner_results[1], pd.DataFrame):
            new_tickers_df = scanner_results[1]
            if 'name' in new_tickers_df.columns:
                tickers = new_tickers_df['name'].tolist()
                print(f"Screener found {len(tickers)} tickers.")
                return tickers
            else:
                print("Warning: 'name' column not found in screening results.")
                return []
        else:
            print(f"Warning: TradingView screening returned unexpected data format or no data: {scanner_results}")
            return []
    except Exception as e:
        print(f"Error during TradingView screening: {e}")
        return []

def get_yf_data(tickers, period, interval):
    """Helper to fetch yfinance data for a list of tickers."""
    if not tickers: return pd.DataFrame()
    try:
        # Use the global session for yfinance
        data = yf.download(tickers, period=period, interval=interval, progress=False, session=YFINANCE_SESSION)
        return data
    except Exception as e:
        print(f"Error fetching yfinance data for {tickers} (Period: {period}, Interval: {interval}): {e}")
        return pd.DataFrame()

def calculate_metrics(tickers, spy_hist_data):
    """Calculates Beta, Correlation, and Leverage (placeholder) for tickers."""
    metrics_dict = {}
    if spy_hist_data.empty or 'Close' not in spy_hist_data.columns:
        print("Error: SPY historical data is required for metrics calculation.")
        return {}

    # Ensure SPY data is a single column Series
    if isinstance(spy_hist_data.columns, pd.MultiIndex):
        if ('Close', 'SPY') in spy_hist_data.columns:
            spy_close = spy_hist_data['Close']['SPY']
        else:
            print("Error: 'Close' column for SPY not found in MultiIndex.")
            return {}
    elif 'Close' in spy_hist_data.columns:
         # If it's a single column df and SPY was the only ticker requested,
         # assume the 'Close' column is SPY.
         if len(spy_hist_data.columns) == 1 and spy_hist_data.columns[0] == 'Close':
             spy_close = spy_hist_data['Close']
         elif 'SPY' in spy_hist_data.columns: # Check if SPY is a direct column name
              spy_close = spy_hist_data['SPY'] # Assuming 'SPY' column exists directly
         else:
              print("Error: Unexpected SPY historical data format for metrics.")
              return {}
    else:
        print("Error: 'Close' column not found in SPY historical data for metrics.")
        return {}

    # Calculate SPY daily returns
    spy_returns = spy_close.pct_change().dropna()
    if spy_returns.empty or spy_returns.std() == 0:
        print("Error: SPY returns data is invalid or has no variance for metrics calculation.")
        return {}

    # Fetch historical data for the list of tickers
    # Use the same period as SPY data for alignment
    ticker_hist_data = get_yf_data(tickers, period=spy_hist_data.index[0].strftime('%Y-%m-%d') + '::' + spy_hist_data.index[-1].strftime('%Y-%m-%d'), interval="1d")

    if ticker_hist_data.empty or 'Close' not in ticker_hist_data.columns:
        print("Warning: No historical data fetched for tickers for metrics calculation.")
        return {}

    # Extract Close prices, handling MultiIndex
    if isinstance(ticker_hist_data.columns, pd.MultiIndex):
        ticker_close_prices = ticker_hist_data['Close']
    elif 'Close' in ticker_hist_data.columns:
         # Handle single ticker case - assuming the single 'Close' column is the ticker
         if len(tickers) == 1 and ticker_hist_data.columns[0] == 'Close':
              ticker_close_prices = ticker_hist_data[['Close']]
              ticker_close_prices.columns = [tickers[0]] # Rename column to ticker
         else:
              print("Warning: Unexpected ticker historical data format for metrics.")
              return {}
    else:
        print("Warning: 'Close' column not found in ticker historical data for metrics.")
        return {}

    # Calculate daily returns for tickers
    ticker_daily_returns = ticker_close_prices.pct_change().dropna()

    if ticker_daily_returns.empty:
        print("Warning: No valid daily returns calculated for tickers for metrics.")
        return {}

    # Align ticker returns and SPY returns by index
    aligned_returns = pd.concat([ticker_daily_returns, spy_returns], axis=1).dropna()

    if aligned_returns.empty:
        print("Warning: No aligned data for tickers vs SPY for metrics calculation.")
        return {}

    for ticker in tickers:
        if ticker in aligned_returns.columns:
            ticker_returns = aligned_returns[ticker]
            if ticker_returns.std() == 0:
                beta, correlation = 0.0, 0.0
            else:
                # Calculate Beta and Correlation using aligned data
                try:
                    covariance_matrix = np.cov(ticker_returns, aligned_returns['SPY'])
                    beta = covariance_matrix[0, 1] / covariance_matrix[1, 1] if covariance_matrix[1, 1] != 0 else 0.0
                    correlation = np.corrcoef(ticker_returns, aligned_returns['SPY'])[0, 1]
                except Exception as e:
                     print(f"Error calculating Beta/Correlation for {ticker}: {e}")
                     beta, correlation = np.nan, np.nan # Indicate calculation failure

            # Placeholder for Leverage calculation - needs actual logic
            leverage = 1.0 # Example placeholder

            metrics_dict[ticker] = {'beta': beta, 'correlation': correlation, 'leverage': leverage}
        else:
            # Ticker was in the input list but dropped due to insufficient data or alignment issues
            metrics_dict[ticker] = {'beta': np.nan, 'correlation': np.nan, 'leverage': np.nan}
            print(f"Warning: Could not calculate metrics for {ticker} due to insufficient or misaligned data.")

    return metrics_dict

def select_tickers(tickers_to_filter, metrics, invest_scores_all, formula_results, portfolio_value):
    """Selects final common stock tickers based on metrics and scores."""
    # This is a placeholder based on the structure, actual selection logic needed
    print("Selecting final tickers based on criteria...")
    final_common_stock_tickers = []
    selection_warning_msg = None

    # Example selection criteria (replace with actual Invest Omega logic)
    # Filter based on having valid metrics and Invest Score
    eligible_tickers = [
        ticker for ticker in tickers_to_filter
        if ticker in metrics and not pd.isna(metrics[ticker]['beta']) and not pd.isna(metrics[ticker]['correlation'])
        and ticker in invest_scores_all and invest_scores_all[ticker]['score'] is not None
    ]

    if not eligible_tickers:
        selection_warning_msg = "No eligible tickers found after filtering by metrics and scores."
        print(selection_warning_msg)
        return [], selection_warning_msg, invest_scores_all # Return empty list, warning, and scores

    # Sort eligible tickers by Invest Score (descending)
    eligible_tickers.sort(key=lambda t: safe_score(invest_scores_all[t]['score']), reverse=True)

    # Select top N tickers based on Sigma Portfolio calculation (placeholder)
    # Sigma Portfolio calculation needs actual logic from Invest Omega
    sigma_val = safe_score(formula_results.get('sigma', 50.0))
    # Example: Number of tickers based on Sigma and Portfolio Value (epsilon)
    # This formula is from the provided Cultivate snippet
    num_tickers_to_select = max(1, math.ceil(0.3 * math.sqrt(portfolio_value))) # Use portfolio_value (epsilon)

    final_common_stock_tickers = eligible_tickers[:num_tickers_to_select]

    if not final_common_stock_tickers:
         selection_warning_msg = "No tickers selected for Common Stock portfolio after applying selection logic."
         print(selection_warning_msg)

    print(f"Selected {len(final_common_stock_tickers)} final tickers.")
    return final_common_stock_tickers, selection_warning_msg, invest_scores_all # Return selected tickers, warning, and scores

def build_and_process_portfolios(common_stock_tickers, amplification, portfolio_weight_lambda, total_portfolio_value, cash_allocation_omega, frac_shares, invest_scores_all, eta, kappa, lambda_hedge):
    """
    Builds and processes the Common Stock, Market Hedging, and Resource Hedging portfolios,
    calculates allocations, and determines final cash.
    MODIFIED to return tailored_portfolio_entries and final_cash_value for Assess Code D.
    Also calculates actual value of each portfolio type.
    """
    print("Building and processing portfolios...")
    combined_portfolio_list = [] # For combined allocation output
    tailored_portfolio_entries = [] # For tailored output and Assess Code D

    # Calculate the value allocated to stocks and hedges (non-cash portion)
    # This is Epsilon * Lambda (as a percentage)
    value_for_stocks_hedges = safe_score(total_portfolio_value) * (safe_score(portfolio_weight_lambda) / 100.0)
    initial_omega_cash = safe_score(total_portfolio_value) * (safe_score(cash_allocation_omega) / 100.0) # Initial cash based on Omega

    # Calculate the value allocated to hedging within the non-cash portion
    # This is value_for_stocks_hedges * (Lambda Hedge / Lambda) if Lambda > 0, otherwise 0
    market_resource_hedge_value_target = value_for_stocks_hedges * (safe_score(lambda_hedge) / safe_score(portfolio_weight_lambda) if safe_score(portfolio_weight_lambda) > 0 else 0.0)

    # Calculate the value allocated to Common Stock within the non-cash portion
    # This is value_for_stocks_hedges * (Eta / Lambda) if Lambda > 0, otherwise 0
    common_stock_value_target = value_for_stocks_hedges * (safe_score(eta) / safe_score(portfolio_weight_lambda) if safe_score(portfolio_weight_lambda) > 0 else 0.0)


    # --- Process Common Stock Portfolio ---
    print(f" Processing Common Stock Portfolio (Target Value: ${common_stock_value_target:,.2f})...")
    common_stock_data = []
    if common_stock_tickers:
        # Calculate total amplified score for selected common stocks
        total_amplified_score_common = safe_score(sum(
            safe_score((safe_score(invest_scores_all.get(t, {}).get('score', 50.0)) * safe_score(amplification)) - (safe_score(amplification) - 1) * 50)
            for t in common_stock_tickers if t in invest_scores_all and invest_scores_all[t]['score'] is not None
        ))

        for ticker in common_stock_tickers:
            if ticker in invest_scores_all and invest_scores_all[ticker]['score'] is not None:
                live_price = safe_score(invest_scores_all[ticker].get('live_price', 0.0))
                raw_invest_score = safe_score(invest_scores_all[ticker]['score'])
                # Calculate amplified score for this ticker
                amplified_score = safe_score((raw_invest_score * safe_score(amplification)) - (safe_score(amplification) - 1) * 50)
                amplified_score_clamped = max(0, min(100, amplified_score))

                # Calculate allocation percentage within the Common Stock portfolio (based on amplified score)
                common_stock_alloc_pct_internal = (amplified_score_clamped / total_amplified_score_common) * 100.0 if total_amplified_score_common > 0 else 0.0

                # Calculate combined allocation percentage (relative to the total portfolio value Epsilon)
                # This is (Internal Allocation % / 100) * (Common Stock Target Value / Epsilon) * 100
                combined_alloc_pct = (common_stock_alloc_pct_internal / 100.0) * (common_stock_value_target / safe_score(total_portfolio_value) if safe_score(total_portfolio_value) > 0 else 0.0) * 100.0

                common_stock_data.append({
                    'ticker': ticker,
                    'portfolio': 'Common Stock',
                    'live_price': live_price,
                    'raw_invest_score': raw_invest_score,
                    'amplified_score': amplified_score_clamped,
                    'portfolio_allocation_percent': common_stock_alloc_pct_internal, # Allocation within Common Stock
                    'combined_percent_allocation': combined_alloc_pct # Allocation relative to Total Portfolio Value (Epsilon)
                })
                combined_portfolio_list.append(common_stock_data[-1]) # Add to combined list


    # --- Process Market Hedging Portfolio ---
    print(f" Processing Market Hedging Portfolio (Target Value: ~50% of Market/Resource Hedge Target: ${safe_score(market_resource_hedge_value_target)/2.0:,.2f})...")
    market_hedging_tickers = [t for t in HEDGING_TICKERS if t in invest_scores_all and invest_scores_all[t]['score'] is not None and t in ['SH', 'PSQ']] # Example Market Hedging Tickers
    market_hedging_data = []
    if market_hedging_tickers:
        # Calculate total amplified score for Market Hedging tickers (using the same amplification)
        total_amplified_score_market_hedge = safe_score(sum(
            safe_score((safe_score(invest_scores_all.get(t, {}).get('score', 50.0)) * safe_score(amplification)) - (safe_score(amplification) - 1) * 50)
            for t in market_hedging_tickers if t in invest_scores_all and invest_scores_all[t]['score'] is not None
        ))

        for ticker in market_hedging_tickers:
             if ticker in invest_scores_all and invest_scores_all[ticker]['score'] is not None:
                live_price = safe_score(invest_scores_all[ticker].get('live_price', 0.0))
                raw_invest_score = safe_score(invest_scores_all[ticker]['score'])
                amplified_score = safe_score((raw_invest_score * safe_score(amplification)) - (safe_score(amplification) - 1) * 50)
                amplified_score_clamped = max(0, min(100, amplified_score))

                # Calculate allocation percentage within the Market Hedging portfolio
                market_hedge_alloc_pct_internal = (amplified_score_clamped / total_amplified_score_market_hedge) * 100.0 if total_amplified_score_market_hedge > 0 else 0.0

                # Calculate combined allocation percentage (relative to Total Portfolio Value Epsilon)
                # This is (Internal Allocation % / 100) * (Market/Resource Hedge Target Value / Epsilon) * 100 * 0.5 (assuming 50% to Market Hedge)
                combined_alloc_pct = (market_hedge_alloc_pct_internal / 100.0) * (market_resource_hedge_value_target / safe_score(total_portfolio_value) if safe_score(total_portfolio_value) > 0 else 0.0) * 100.0 * 0.5

                market_hedging_data.append({
                    'ticker': ticker,
                    'portfolio': 'Market Hedging',
                    'live_price': live_price,
                    'raw_invest_score': raw_invest_score,
                    'amplified_score': amplified_score_clamped,
                    'portfolio_allocation_percent': market_hedge_alloc_pct_internal, # Allocation within Market Hedging
                    'combined_percent_allocation': combined_alloc_pct # Allocation relative to Total Portfolio Value (Epsilon)
                })
                combined_portfolio_list.append(market_hedging_data[-1]) # Add to combined list


    # --- Process Resource Hedging Portfolio ---
    print(f" Processing Resource Hedging Portfolio (Target Value: ~50% of Market/Resource Hedge Target: ${safe_score(market_resource_hedge_value_target)/2.0:,.2f})...")
    resource_hedging_tickers = [t for t in HEDGING_TICKERS if t in invest_scores_all and invest_scores_all[t]['score'] is not None and t not in ['SH', 'PSQ']] # Example Resource Hedging Tickers
    resource_hedging_data = []
    if resource_hedging_tickers:
        # Calculate total amplified score for Resource Hedging tickers
        total_amplified_score_resource_hedge = safe_score(sum(
            safe_score((safe_score(invest_scores_all.get(t, {}).get('score', 50.0)) * safe_score(amplification)) - (safe_score(amplification) - 1) * 50)
            for t in resource_hedging_tickers if t in invest_scores_all and invest_scores_all[t]['score'] is not None
        ))

        for ticker in resource_hedging_tickers:
             if ticker in invest_scores_all and invest_scores_all[ticker]['score'] is not None:
                live_price = safe_score(invest_scores_all[ticker].get('live_price', 0.0))
                raw_invest_score = safe_score(invest_scores_all[ticker]['score'])
                amplified_score = safe_score((raw_invest_score * safe_score(amplification)) - (safe_score(amplification) - 1) * 50)
                amplified_score_clamped = max(0, min(100, amplified_score))

                # Calculate allocation percentage within the Resource Hedging portfolio
                resource_hedge_alloc_pct_internal = (amplified_score_clamped / total_amplified_score_resource_hedge) * 100.0 if total_amplified_score_resource_hedge > 0 else 0.0

                # Calculate combined allocation percentage (relative to Total Portfolio Value Epsilon)
                # This is (Internal Allocation % / 100) * (Market/Resource Hedge Target Value / Epsilon) * 100 * 0.5 (assuming 50% to Resource Hedge)
                combined_alloc_pct = (resource_hedge_alloc_pct_internal / 100.0) * (market_resource_hedge_value_target / safe_score(total_portfolio_value) if safe_score(total_portfolio_value) > 0 else 0.0) * 100.0 * 0.5

                resource_hedging_data.append({
                    'ticker': ticker,
                    'portfolio': 'Resource Hedging',
                    'live_price': live_price,
                    'raw_invest_score': raw_invest_score,
                    'amplified_score': amplified_score_clamped,
                    'portfolio_allocation_percent': resource_hedge_alloc_pct_internal, # Allocation within Resource Hedging
                    'combined_percent_allocation': combined_alloc_pct # Allocation relative to Total Portfolio Value (Epsilon)
                })
                combined_portfolio_list.append(resource_hedging_data[-1]) # Add to combined list


    # --- Calculate Tailored Portfolio Holdings ---
    print(" Calculating Tailored Portfolio Holdings...")
    total_actual_money_allocated_stocks_hedges = 0.0 # Track actual money allocated to stocks/hedges

    # Iterate through the combined portfolio list (stocks and hedges)
    for entry in combined_portfolio_list:
        combined_alloc_pct = safe_score(entry.get('combined_percent_allocation', 0.0))
        live_price = safe_score(entry.get('live_price', 0.0))

        if combined_alloc_pct > 1e-9 and live_price > 0: # Allocate if % > 0 and price > 0
            # Target value for this specific ticker based on its combined allocation percentage of the TOTAL portfolio value (Epsilon)
            target_allocation_value_for_ticker = safe_score(total_portfolio_value) * (combined_alloc_pct / 100.0)

            shares = 0.0
            try:
                exact_shares = target_allocation_value_for_ticker / live_price
                if frac_shares: shares = round(exact_shares, 4) # Use 4 decimal places for fractional shares
                else: shares = float(math.floor(exact_shares))
            except ZeroDivisionError: shares = 0.0 # Should be caught by live_price > 0
            except Exception: shares = 0.0

            shares = max(0.0, shares) # Ensure non-negative shares
            actual_money_allocation = shares * live_price

            share_threshold = 0.0001 if frac_shares else 1.0 # Min shares to buy
            if shares >= share_threshold:
                 # Actual percent of the *total_portfolio_value* this holding represents
                 actual_percent_allocation_total = (actual_money_allocation / safe_score(total_portfolio_value)) * 100.0 if safe_score(total_portfolio_value) > 0 else 0.0

                 tailored_portfolio_entries.append({
                    'ticker': entry.get('ticker','ERR'),
                    'portfolio': entry.get('portfolio', '?'),
                    'shares': shares,
                    'actual_money_allocation': actual_money_allocation, # For Assess D
                    'actual_percent_allocation_total': actual_percent_allocation_total # For display
                 })
                 total_actual_money_allocated_stocks_hedges += actual_money_allocation

    # Calculate Final Cash Value and Percentage
    # Final cash is the initial Omega cash + the remaining value from the stocks/hedges portion
    # The remaining value from the stocks/hedges portion is the target value for stocks/hedges minus the actual money allocated to them.
    remaining_from_stocks_hedges = value_for_stocks_hedges - total_actual_money_allocated_stocks_hedges
    final_cash_value = initial_omega_cash + remaining_from_stocks_hedges
    final_cash_value = max(0.0, final_cash_value) # Ensure cash is not negative

    # Calculate the final cash percentage of the total portfolio value (epsilon)
    final_cash_percent = (final_cash_value / safe_score(total_portfolio_value)) * 100.0 if safe_score(total_portfolio_value) > 0 else 0.0
    final_cash_percent = max(0.0, min(100.0, final_cash_percent)) # Clamp cash percent

    # Sort tailored portfolio entries by actual money allocation descending for table output
    tailored_portfolio_entries.sort(key=lambda x: x.get('actual_money_allocation', 0.0), reverse=True)

    # Calculate Common Value, Market Hedge Value, and Resource Hedge Value from tailored_portfolio_entries
    common_value_actual = sum(entry['actual_money_allocation'] for entry in tailored_portfolio_entries if entry.get('portfolio') == 'Common Stock')
    market_hedge_value_actual = sum(entry['actual_money_allocation'] for entry in tailored_portfolio_entries if entry.get('portfolio') == 'Market Hedging')
    resource_hedge_value_actual = sum(entry['actual_money_allocation'] for entry in tailored_portfolio_entries if entry.get('portfolio') == 'Resource Hedging')


    # Return values:
    # combined_portfolio_list: List of dicts with combined allocations (used by some internal saves)
    # tailored_portfolio_entries: List of dicts for tailored portfolio output/Assess D
    # final_cash_value: Float, final total cash for tailored portfolio
    # final_cash_percent: Float, final total cash percentage for tailored portfolio
    # value_for_stocks_hedges: Float, the value allocated to stocks/hedges (Non-Cash portion) - used for 'Tailored Portfolio Value' in Greeks table
    # common_value_actual: Float, total value of common stock holdings in tailored portfolio
    # market_hedge_value_actual: Float, total value of Market Hedging holdings in tailored portfolio
    # resource_hedge_value_actual: Float, total value of Resource Hedging holdings in tailored portfolio
    # initial_omega_cash: Float, the initial Omega-based cash value (for Greeks table)
    return combined_portfolio_list, tailored_portfolio_entries, final_cash_value, final_cash_percent, value_for_stocks_hedges, common_value_actual, market_hedge_value_actual, resource_hedge_value_actual, initial_omega_cash


# --- run_cultivate_analysis function (Adapted for Terminal) ---
# This function orchestrates the Cultivate analysis and output for the terminal.
async def run_cultivate_analysis(portfolio_value: float, frac_shares: bool, cultivate_code: str, is_saving: bool = False):
    """
    Orchestrates the Cultivate portfolio analysis based on the cultivate code for the terminal.
    Handles input, calls logic functions, and formats output for the terminal.
    Returns the combined_portfolio_data list and relevant parameters for saving.
    Args:
        portfolio_value: The total portfolio value (epsilon).
        frac_shares: Boolean indicating if fractional shares are allowed.
        cultivate_code: 'A' or 'B' (string).
        is_saving: Flag to indicate if this run is primarily for saving data (suppresses some output).
    """
    code_str = cultivate_code.upper() # Ensure uppercase string
    epsilon = safe_score(portfolio_value)

    if code_str not in ['A', 'B']:
        print("Error: Invalid Cultivate Code. Please use 'A' or 'B'.")
        return [], code_str, epsilon, frac_shares, "Error: Invalid Cultivate Code" # Return empty data and error message

    if epsilon <= 0:
        print("Error: Portfolio value must be a positive number.")
        return [], code_str, epsilon, frac_shares, "Error: Invalid portfolio value" # Return error message, but include basic params

    if not is_saving:
        print(f"Starting Cultivate analysis (Code: {code_str}, Epsilon: ${epsilon:,.2f})...")
    else:
        print(f"[Save Cultivate]: Running analysis for Code {code_str} (Epsilon {epsilon:,.2f})...")


    # --- Step 1: Get Allocation Score (Sigma) ---
    if not is_saving: print("Step 1/7: Getting Allocation Score...")
    allocation_score, risk_general_score, market_invest_score = calculate_market_risk() # Use calculate_market_risk for Sigma
    if allocation_score is None:
        print("Error: Failed to retrieve or calculate Allocation Score (Sigma). Aborting.")
        return [], code_str, epsilon, frac_shares, "Error: Failed to get Allocation Score"
    sigma = allocation_score


    # --- Step 2: Calculate Formula Variables ---
    if not is_saving: print("Step 2/7: Calculating Formula Variables...")
    formula_results = calculate_cultivate_formulas(sigma)
    if formula_results is None:
        print("Error: Failed to calculate portfolio structure variables from formulas. Aborting.")
        return [], code_str, epsilon, frac_shares, "Error: Formula calculation failed"

    # Extract key formula results needed later
    amplification_delta = safe_score(formula_results.get('delta', 0.0))
    portfolio_weight_lambda = safe_score(formula_results.get('lambda', 0.0)) # This is Lambda (100-Omega)
    cash_allocation_omega = safe_score(formula_results.get('omega', 0.0)) # Still needed for Greeks table
    eta = safe_score(formula_results.get('eta', 0.0)) # Get Eta
    kappa = safe_score(formula_results.get('kappa', 0.0)) # Get Kappa
    lambda_hedge = safe_score(formula_results.get('lambda_hedge', 0.0)) # Get Lambda Hedge


    # --- Step 3: Determine Tickers to Process (Code A or B) ---
    tickers_to_process = []
    if code_str == 'A':
        if not is_saving: print("Step 3/7: Screening stocks (Code A)...")
        tickers_to_process = screen_stocks()
        if not tickers_to_process and not is_saving:
            print("Warning: No stocks passed initial screening criteria. Proceeding with only hedging and cash.")
    elif code_str == 'B':
        if not is_saving: print("Step 3/7: Getting all SPY tickers (Code B)...")
        tickers_to_process = get_spy_symbols()
        if not tickers_to_process:
            print("Error: Failed to retrieve SPY symbols. Aborting.")
            return [], code_str, epsilon, frac_shares, "Error: Failed to get SPY symbols"

    # Ensure tickers_to_process is not empty before proceeding to metrics, unless it's Code A and no stocks screened
    if not tickers_to_process and code_str == 'B':
        print("Error: No SPY symbols found.")
        return [], code_str, epsilon, frac_shares, "Error: No SPY symbols found"


    # --- Step 4: Fetch SPY Data for Metrics & Calculate Metrics ---
    if not is_saving: print("Step 4/7: Calculating Metrics (Beta/Corr/Leverage)...")
    # Fetch SPY data needed for metrics calculation regardless of code A or B
    spy_hist_data_for_metrics = get_yf_data(['SPY'], period="10y", interval="1d")
    if spy_hist_data_for_metrics.empty:
        print("Error: Failed to download SPY data required for metric calculation. Aborting.")
        return [], code_str, epsilon, frac_shares, "Error: Failed to get SPY data for metrics"

    # Calculate metrics for the determined list of tickers (screened or SPY)
    # Only calculate metrics if there are tickers to process (unless it's Code A with no screened tickers)
    metrics_dict = {}
    if tickers_to_process:
        metrics_dict = calculate_metrics(tickers_to_process, spy_hist_data_for_metrics)
        if not metrics_dict and tickers_to_process and not is_saving:
            # Only warn if there were tickers to process but metrics failed for all
            print("Warning: Metrics calculation failed for ALL selected tickers. Filtering by metrics will be skipped.")


    # --- Step 5: Calculate ACTUAL Invest Scores for Filtering & Portfolio Building ---
    if not is_saving: print("Step 5/7: Calculating Invest Scores...")
    invest_scores_dict = {}
    # Combine the tickers to process and ALL hedging tickers for scoring
    # Ensure tickers_to_process is not empty before combining, handle case where Code A screened 0 stocks
    tickers_to_score = list(set((tickers_to_process if tickers_to_process else []) + HEDGING_TICKERS)) # Use the updated HEDGING_TICKERS constant

    # Use the INVEST calculate_ema_invest function with Daily interval (2)
    if tickers_to_score:
        score_count, successful_score_count = 0, 0
        for ticker in tickers_to_score:
            score_count += 1
            live_price, score = calculate_ema_invest(ticker, 2) # Use Daily sensitivity (2)
            if score is not None and live_price is not None:
                # Store raw score and price
                invest_scores_dict[ticker] = {'score': safe_score(score), 'live_price': safe_score(live_price)}
                successful_score_count += 1
            # else: print(f" Warning: Failed to get score/price for {ticker}.") # Verbose
            # if score_count % 50 == 0 or score_count == len(tickers_to_score):
            # print(f" Scoring progress: {score_count}/{len(tickers_to_score)} (Successful: {successful_score_count})") # Commented out
        # print(f" Invest Score calculation complete for {len(invest_scores_dict)} tickers.") # Commented out
    elif not is_saving:
        print("Warning: No tickers to score. Cannot proceed with portfolio building.")
        return [], code_str, epsilon, frac_shares, "Warning: No tickers to score" # Return empty data


    # --- Step 6: Select Final Tickers (Tf) ---
    if not is_saving: print("Step 6/7: Selecting Final Tickers...")
    # Pass the list of tickers determined by the cultivate code (screened or SPY) to select_tickers
    # If cultivate_code A resulted in 0 screened tickers, pass an empty list to select_tickers
    tickers_to_filter_for_selection = tickers_to_process if tickers_to_process else []
    final_common_stock_tickers, selection_warning_msg, invest_scores_all_used = select_tickers(
        tickers_to_filter=tickers_to_filter_for_selection, # Pass the list to filter from
        metrics=metrics_dict,
        invest_scores_all=invest_scores_dict, # Use the calculated scores
        formula_results=formula_results,
        portfolio_value=epsilon # Pass epsilon for Sigma Portfolio calculation
    )
    if selection_warning_msg and not is_saving:
        print(f"*** {selection_warning_msg} ***")

    if not final_common_stock_tickers and not any(t in HEDGING_TICKERS for t in tickers_to_score):
         # If no common stocks were selected AND no hedging tickers had valid scores,
         # the portfolio will be 100% cash.
         if not is_saving: print("No common stock or valid hedging tickers selected. Portfolio will be 100% Cash.")


    # --- Step 7: Build and Process Portfolios ---
    if not is_saving: print("Step 7/7: Building & Processing Final Portfolios...")
    # Pass Eta, Kappa, AND Lambda Hedge to build_and_process_portfolios and handle new return values
    combined_portfolio_data, tailored_portfolio_data, final_cash_value, final_cash_percent, value_for_stocks_hedges, common_value_actual, market_hedge_value_actual, resource_hedge_value_actual, initial_omega_cash = build_and_process_portfolios(
        common_stock_tickers=final_common_stock_tickers, # Use the selected common stocks
        amplification=amplification_delta,
        portfolio_weight_lambda=portfolio_weight_lambda, # This is Lambda (100-Omega), still passed but not used for split
        total_portfolio_value=epsilon, # Pass epsilon as the total value
        cash_allocation_omega=cash_allocation_omega, # Pass Omega for cash calculations and Greeks table
        frac_shares=frac_shares,
        invest_scores_all=invest_scores_all_used, # Pass the scores dictionary
        eta=eta, # Pass Eta
        kappa=kappa, # Pass Kappa
        lambda_hedge=lambda_hedge # Pass Lambda Hedge
    )

    if not is_saving:
        print("Portfolio processing complete. Formatting results...")

        # --- Output Combined Portfolio Table ---
        combined_table_display_data = []
        if combined_portfolio_data:
            # Sort by combined allocation descending for display
            sorted_combined_data = sorted(combined_portfolio_data, key=lambda x: safe_score(x.get('combined_percent_allocation', 0.0)), reverse=True)
            for entry in sorted_combined_data:
                alloc_pct = safe_score(entry.get('combined_percent_allocation', 0.0))
                # Only include if allocation is >= 0.01% to avoid clutter
                if alloc_pct >= 0.01:
                    live_price_f = f"${safe_score(entry.get('live_price', 0.0)):,.2f}" if safe_score(entry.get('live_price')) > 0 else "N/A" # Added comma formatting
                    # Display raw invest score (can be negative)
                    raw_invest_val = safe_score(entry.get('raw_invest_score'))
                    raw_invest_f = f"{raw_invest_val:.2f}%" if raw_invest_val is not None and not pd.isna(raw_invest_val) else "N/A"
                    combined_alloc_f = f"{alloc_pct:.2f}%"
                    combined_table_display_data.append([entry.get('ticker', 'ERR'), entry.get('portfolio', '?'), live_price_f, raw_invest_f, combined_alloc_f])

        print("\n**Combined Portfolio Allocation (Relative to Non-Cash Portion)**")
        headers_comb = ["Ticker", "Portfolio", "Live Price", "Raw Score", "Combined % Alloc"]
        if combined_table_display_data:
            print(tabulate(combined_table_display_data, headers=headers_comb, tablefmt="pretty", numalign="right", stralign="center", colalign=("center", "center", "right", "right", "right")))
        else:
            print("```\nNo target allocation calculated (excluding cash).\n```")


        # --- Output Tailored Portfolio Table ---
        tailored_table_display_data = []
        if tailored_portfolio_data:
            # The tailored_portfolio_data is already sorted by actual money allocation descending
            for item in tailored_portfolio_data:
                shares = item['shares'] # Format shares based on frac_shares flag
                shares_f = f"{shares:.4f}" if frac_shares else f"{int(shares)}" # Use 4 decimal places for frac shares
                money_f = f"${safe_score(item.get('actual_money_allocation', 0.0)):,.2f}" # Added comma formatting
                percent_val = safe_score(item.get('actual_percent_allocation_total', 0.0))
                percent_f = f"{percent_val:.2f}%"
                tailored_table_display_data.append([item.get('ticker', 'ERR'), item.get('portfolio', '?'), shares_f, money_f, percent_f])

        # Add Cash row to tailored table data
        tailored_table_display_data.append(['Cash', 'Cash', '-', f"${safe_score(final_cash_value):,.2f}", f"{safe_score(final_cash_percent):.2f}%"]) # Added comma formatting

        print("\n**Tailored Portfolio**")
        headers_tail = ["Ticker", "Portfolio", "Shares", "Actual $ Allocation", "Actual % of Total"]
        if tailored_table_display_data: # Always has at least the Cash row
            print(tabulate(tailored_table_display_data, headers=headers_tail, tablefmt="pretty", numalign="right", stralign="center", colalign=("center", "center", "right", "right", "right")))
        else:
             print("No tailored portfolio data generated.")


        # --- Output The Invest Greeks Table ---
        print("\n**The Invest Greeks**")
        # Recalculate Sigma_count for the table
        num_tickers_sigma_report = max(1, math.ceil(0.3 * math.sqrt(epsilon))) # Use epsilon

        # Prepare data for The Invest Greeks table in the specified order
        greek_data = [
             ["Sigma", f"{safe_score(sigma):.2f}"],
             ["Lambda", f"{safe_score(formula_results.get('lambda', 0.0)):.2f}"],
             # MODIFIED: Output Lambda Hedge, Kappa, and Eta in the new order
             ["Lambda Hedge", f"{safe_score(formula_results.get('lambda_hedge', 0.0)):.2f}"],
             ["Kappa", f"{safe_score(formula_results.get('kappa', 0.0)):.2f}"],
             ["Eta", f"{safe_score(formula_results.get('eta', 0.0)):.2f}"],
             ["Omega", f"{safe_score(formula_results.get('omega', 0.0)):.2f}"],
             ["Alpha", f"{safe_score(formula_results.get('alpha', 0.0)):.2f}"],
             ["Beta", f"{safe_score(formula_results.get('beta_alloc', 0.0)):.2f}"],
             ["Mu Target", f"{safe_score(formula_results.get('mu_center', 0.0)):.2f}"],
             ["Mu Low", f"{safe_score(formula_results.get('mu_range', (0.0, 0.0))[0]):.2f}"],
             ["Mu High", f"{safe_score(formula_results.get('mu_range', (0.0, 0.0))[1]):.2f}"],
             ["Rho Target", f"{safe_score(formula_results.get('rho_center', 0.0)):.2f}"],
             ["Rho Low", f"{safe_score(formula_results.get('rho_range', (0.0, 0.0))[0]):.2f}"],
             ["Rho High", f"{safe_score(formula_results.get('rho_range', (0.0, 0.0))[1]):.2f}"],
             ["Omega Target", f"{safe_score(formula_results.get('omega_target_center', 0.0)):.2f}"],
             ["Omega Low", f"{safe_score(formula_results.get('omega_target_range', (0.0, 0.0))[0]):.2f}"],
             ["Omega High", f"{safe_score(formula_results.get('omega_target_range', (0.0, 0.0))[1]):.2f}"],
             ["Delta", f"{safe_score(formula_results.get('delta', 0.0)):.2f}"],
             ["Epsilon", f"{safe_score(epsilon):,.2f}"], # Format with commas
             ["Sigma Portfolio", f"{num_tickers_sigma_report:.0f}"], # Display as integer
             # Use the value_for_stocks_hedges for 'Tailored Portfolio Value' in the Greeks table
             ["Tailored Portfolio Value", f"{safe_score(value_for_stocks_hedges):,.2f}"], # Format with commas
             ["Common Value", f"{safe_score(common_value_actual):,.2f}"], # Format with commas
             ["Market Hedge Value", f"{safe_score(market_hedge_value_actual):,.2f}"], # Format with commas
             ["Resource Hedge Value", f"{safe_score(resource_hedge_value_actual):,.2f}"], # Format with commas
             ["Cash Value", f"${safe_score(final_cash_value):,.2f}"], # Format with commas
        ]
        greek_table_str = tabulate(greek_data, headers=["Variable", "Value"], tablefmt="pretty", numalign="right", stralign="left")
        print(f"```{greek_table_str}```")


    # Return combined data and parameters for saving purposes
    # MODIFIED: Return the updated set of values from build_and_process_portfolios
    return combined_portfolio_data, code_str, epsilon, frac_shares, None # Return None for error message on success


# --- save_cultivate_data_internal function (Adapted for Terminal) ---
# This function remains largely the same as it correctly handles cultivate_code as a string.
async def save_cultivate_data_internal(combined_portfolio_data, date_str, cultivate_code, epsilon):
    """
    Internal function to save combined Cultivate portfolio data.
    Saves data in the format: DATE, TICKER, PRICE, RAW_INVEST_SCORE, COMBINED_ALLOCATION_PERCENT.
    """
    if not combined_portfolio_data:
        print(f"[Save Cultivate]: No valid combined portfolio data to save for Cultivate Code {cultivate_code} and Epsilon {epsilon}.")
        raise ValueError("No valid combined portfolio data to save.") # Indicate failure

    # Sort by combined allocation descending for saving consistency
    sorted_combined = sorted(combined_portfolio_data, key=lambda x: safe_score(x.get('combined_percent_allocation', 0.0)), reverse=True)

    # --- Perform the Save ---
    # Filename format: cultivate_combined_[Code]_[Epsilon].csv
    # Note: Filename includes epsilon for potential future analysis of different value effects,
    # but auto-save will always use 10000.
    # Using integer part of epsilon for filename for simplicity.
    epsilon_int = int(epsilon)
    # cultivate_code is expected to be a string here
    save_file = f"cultivate_combined_{cultivate_code.upper()}_{epsilon_int}.csv"
    file_exists = os.path.isfile(save_file)
    save_count = 0
    # Headers: DATE, TICKER, PRICE, RAW_INVEST_SCORE, COMBINED_ALLOCATION_PERCENT
    headers = ['DATE', 'TICKER', 'PRICE', 'RAW_INVEST_SCORE', 'COMBINED_ALLOCATION_PERCENT']
    try:
        with open(save_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists or os.path.getsize(save_file) == 0: writer.writeheader()
            for item in sorted_combined:
                # Only save entries with positive combined allocation percentage (excluding cash)
                if safe_score(item.get('combined_percent_allocation', 0.0)) > 0:
                    ticker = item.get('ticker', 'ERR')
                    price_val = item.get('live_price')
                    raw_score_val = item.get('raw_invest_score') # Use the raw score
                    alloc_val = item.get('combined_percent_allocation')

                    # Format data for saving
                    price_str = f"{safe_score(price_val):.2f}" if price_val is not None else "N/A"
                    # Format raw score (can be negative)
                    raw_score_str = f"{safe_score(raw_score_val):.2f}" if raw_score_val is not None else "N/A"
                    alloc_str = f"{safe_score(alloc_val):.2f}" if alloc_val is not None else "N/A"

                    writer.writerow({
                        'DATE': date_str, # Use the provided date
                        'TICKER': ticker,
                        'PRICE': price_str,
                        'RAW_INVEST_SCORE': raw_score_str,
                        'COMBINED_ALLOCATION_PERCENT': alloc_str
                    })
                    save_count += 1
        print(f"[Save Cultivate]: Saved {save_count} rows of combined portfolio data for Cultivate Code '{cultivate_code.upper()}' (Epsilon: {epsilon_int}) to '{save_file}' for date {date_str}.")
    except IOError as e:
        print(f"Error [Save Cultivate]: Writing to save file '{save_file}': {e}")
        raise # Re-raise for the calling function to catch
    except Exception as e:
        print(f"Error [Save Cultivate]: Processing/saving data for Cultivate Code '{cultivate_code.upper()}' (Epsilon: {epsilon}): {e}")
        import traceback
        traceback.print_exc()
        raise # Re-raise for the calling function to catch


# --- /cultivate Command Handler (Adapted for Terminal) ---
async def handle_cultivate_command(args):
    """
    Handles the /cultivate command for the terminal. Crafts a portfolio
    based on Invest Omega logic.
    """
    print("\n--- Running /cultivate Command ---")

    # Collect inputs via terminal
    portfolio_value = get_float_input("Enter your total portfolio value (Epsilon): ", min_value=0.01)
    frac_shares = get_yes_no_input("Allow fractional shares in the tailored portfolio? (yes/no): ")

    cultivate_code = get_user_input("Enter Cultivate Code ('A' for Screener, 'B' for SPY): ", validation=lambda r: r.upper() in ['A', 'B'], error_message="Invalid Cultivate Code. Please use 'A' or 'B'.")

    save_code = None
    if args:
        save_arg = args[0].split('=')
        if len(save_arg) == 2 and save_arg[0].lower() == 'save_code':
            save_code = save_arg[1]
        else:
            print(f"Invalid argument: {args[0]}. Usage: /cultivate [save_code=3725]")
            # Continue without save if argument is invalid, or return? Let's continue.


    # --- Handle Save Code ---
    if save_code == "3725":
        print(f"Attempting to save combined Cultivate data (Code: {cultivate_code.upper()}, Epsilon: ${portfolio_value:,.2f})...")

        # Prompt user for date
        save_date_str = get_user_input("Enter date (MM/DD/YYYY) to save combined Cultivate data under: ", validation=lambda d: datetime.strptime(d, '%m/%d/%Y') if True else False, error_message="Invalid date format. Use MM/DD/YYYY.")

        # Run the analysis internally to get the data
        try:
            combined_portfolio_data, code_used, epsilon_used, frac_shares_used, error_msg = await run_cultivate_analysis(
                portfolio_value=portfolio_value,
                frac_shares=frac_shares,
                cultivate_code=cultivate_code.upper(), # Pass the string value
                is_saving=True # Set flag to indicate this run is for saving
            )
            if error_msg:
                print(f"Error during data generation for save: {error_msg}")
                print("--- /cultivate save process failed. ---")
                return # Exit if analysis failed

            # Call the internal save function
            await save_cultivate_data_internal(
                combined_portfolio_data=combined_portfolio_data,
                date_str=save_date_str,
                cultivate_code=code_used, # Use code returned by analysis (which is now guaranteed to be a string)
                epsilon=epsilon_used # Use epsilon returned by analysis
            )
            # Success message is printed by internal save function
            print("--- /cultivate save process completed. ---")

        except Exception as e:
            print(f"An error occurred while saving Cultivate data: {e}")
            import traceback
            traceback.print_exc()
            print("--- /cultivate save process failed. ---")

        return # Exit after save attempt

    # --- Run Analysis Action (if save_code is not 3725) ---
    # Call the main analysis function for display output
    try:
        await run_cultivate_analysis(portfolio_value, frac_shares, cultivate_code.upper(), is_saving=False)
        print("--- /cultivate analysis complete. ---")
    except Exception as e:
        print(f"An error occurred during the analysis: {e}")
        import traceback
        traceback.print_exc()
        print("--- /cultivate analysis failed. ---")


# --- /assess Command Handler (Adapted for Terminal) ---
async def handle_assess_command(args):
    """
    Handles the /assess command for the terminal.
    """
    print("\n--- Running /assess Command ---")

    if not args:
        print("Usage: /assess <assess_code>")
        print("Assess Codes: A (Stock), B (Manual Portfolio), C (Custom Portfolio Risk), D (Cultivate Portfolio Risk)")
        return

    assess_code = args[0].upper()

    if assess_code == 'A':
        print("--- Assess Code A: Stock Volatility and Risk Correspondence ---")
        tickers_str = get_user_input("Enter tickers (comma-separated): ", validation=lambda r: r and r.strip(), error_message="Tickers cannot be empty.")
        tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()]
        if not tickers:
            print("No valid tickers provided.")
            return

        timeframe_code = get_user_input("Enter timeframe (1: 1Y, 2: 3M, 3: 1M): ", validation=lambda r: r in ['1', '2', '3'], error_message="Invalid timeframe. Use 1, 2, or 3.")

        risk_tolerance = get_int_input("Enter risk tolerance (1-5): ", min_value=1, max_value=5)

        results_a = []
        for ticker in tickers:
            period_change, aabc, score, correspondence, selected_period = calculate_volatility_and_correspondence(ticker, timeframe_code, risk_tolerance)
            if period_change is not None: # Check if calculation was successful
                 results_a.append([ticker, period_change, aabc, score, correspondence])
            else:
                 results_a.append([ticker, "Error", "Error", "Error", "Error"]) # Add error row

        # Sort by Vol Score
        results_a.sort(key=lambda x: x[3] if isinstance(x[3], (int, float)) else float('inf'))

        table_data_a = [[r[0], f"{r[1]:.2f}%" if isinstance(r[1], float) else r[1], f"{r[2]:.2f}%" if isinstance(r[2], float) else r[2], r[3], r[4]] for r in results_a]

        print(f"\n**Stock Assessment Results (Code A - {selected_period}):**")
        if table_data_a:
            print(tabulate(table_data_a, headers=["Ticker", f"{selected_period} Change", "AAPC", "Vol Score", "Risk Match"], tablefmt="pretty"))
        else:
            print("No assessment data generated.")
        print("--- Stock assessment (Code A) complete. ---")

    elif assess_code == 'B':
        print("--- Assess Code B: Manual Portfolio Assessment ---")
        portfolio_tickers_str = get_user_input("Enter the tickers in your portfolio (comma-separated): ", validation=lambda r: r and r.strip(), error_message="Tickers cannot be empty.")
        portfolio_tickers = [t.strip().upper() for t in portfolio_tickers_str.split(',') if t.strip()]
        if not portfolio_tickers:
            print("No valid tickers provided.")
            return

        shares_owned = {}
        for ticker_shares in portfolio_tickers:
            shares = get_float_input(f"Enter number of shares for {ticker_shares}: ", min_value=0)
            shares_owned[ticker_shares] = shares

        cash_amount = get_float_input("Enter cash amount in portfolio ($): ", min_value=0)

        backtest_period = get_user_input("Enter backtesting period (1y, 5y, or 10y): ", validation=lambda r: r.lower() in ['1y', '5y', '10y'], error_message="Invalid period. Use '1y', '5y', or '10y'.")

        total_portfolio_value, weighted_beta_sum, weighted_correlation_sum = calculate_portfolio_metrics(portfolio_tickers, shares_owned, cash_amount, backtest_period)

        if total_portfolio_value is not None: # Check if calculation was successful
            print(f"\n**Manual Portfolio Assessment (Code B - {backtest_period}):**")
            print(f"Total Portfolio Value: ${total_portfolio_value:,.2f}") # Added comma formatting
            print(f"Weighted Average Beta: {weighted_beta_sum:.4f}")
            print(f"Weighted Average Correlation to SPY: {weighted_correlation_sum:.4f}")
        else:
            print("\nError calculating portfolio metrics for Code B.")

        print("--- Manual Portfolio assessment (Code B) complete. ---")

    elif assess_code == 'C':
        print("--- Assess Code C: Custom Portfolio Risk Assessment ---")
        portfolio_db_file = 'portfolio_codes_database.csv'

        # 1. Ask for custom portfolio code
        custom_portfolio_code_str = get_user_input("Enter the custom portfolio code to assess: ", validation=lambda r: r and r.strip(), error_message="Custom portfolio code cannot be empty.")

        # 2. Ask for portfolio value
        user_portfolio_value = get_float_input(f"Enter the total portfolio value for code '{custom_portfolio_code_str}': ", min_value=0.01) # Ensure positive number

        # 3. Ask for backtesting period
        backtest_period_str = get_user_input("Enter the backtesting period for Beta/Correlation (e.g., 1y, 3y, 5y, 10y): ", validation=lambda r: r.lower() in ['1y', '2y', '3y', '5y', '10y'], error_message="Invalid period. Use '1y', '2y', '3y', '5y', or '10y'.")

        # 4. Retrieve Custom Portfolio Configuration from CSV
        custom_portfolio_config = None
        if not os.path.exists(portfolio_db_file):
            print(f"Error: Portfolio database file '{portfolio_db_file}' not found.")
            print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
            return

        try:
            with open(portfolio_db_file, 'r', encoding='utf-8', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row.get('portfolio_code', '').strip().lower() == custom_portfolio_code_str.lower():
                        custom_portfolio_config = row # Store the entire row (dict)
                        break
            if not custom_portfolio_config:
                print(f"Error: Custom portfolio code '{custom_portfolio_code_str}' not found in the database.")
                print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
                return
        except Exception as e_csv:
            print(f"Error reading portfolio database: {e_csv}")
            print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
            return

        # Get frac_shares from the loaded portfolio_data config
        frac_shares_from_config = custom_portfolio_config.get('frac_shares', 'false').lower() == 'true'
        print(f"Processing custom portfolio '{custom_portfolio_code_str}' to get holdings...")

        # 5. Execute Custom Portfolio Logic using process_custom_portfolio
        try:
            # Call process_custom_portfolio. It now returns 5 values.
            # We need the 4th (tailored_holdings_data) and 5th (cash_remaining).
            # Pass is_custom_command_without_save=False to ensure full processing and calculation of tailored holdings.
            _, _, _, tailored_holdings_data, cash_remaining = await process_custom_portfolio(
                portfolio_data=custom_portfolio_config, # The config dict from CSV
                tailor_portfolio=True, # We need tailored output for Assess C
                frac_shares=frac_shares_from_config, # Use frac_shares from config
                total_value=user_portfolio_value, # The value provided by the user for this assessment
                is_custom_command_without_save=False # Ensure full processing, not simplified output
            )
        except Exception as e_proc:
            print(f"An error occurred while processing the custom portfolio: {e_proc}")
            import traceback
            traceback.print_exc()
            print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
            return

        # Check if any holdings were generated (tailored_holdings_data is a list of dicts)
        if not tailored_holdings_data and safe_score(cash_remaining) == 0.0:
            print(f"The custom portfolio '{custom_portfolio_code_str}' resulted in no stock holdings and no cash. Cannot assess.")
            print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
            return

        # 6. Extract Tickers from tailored_holdings_data
        # tailored_holdings_data contains dicts like: {'ticker': T, 'shares': S, 'actual_money_allocation': V, ...}
        portfolio_tickers_from_custom = [item['ticker'] for item in tailored_holdings_data if item['ticker'] != 'Cash'] # Exclude cash if it's there
        print(f"Calculating Beta and Correlation for the portfolio holdings over {backtest_period_str}...")

        # 7. Calculate Beta and Correlation (adapted from Code B's logic)
        all_tickers_for_hist_data_c = list(set(portfolio_tickers_from_custom + ['SPY']))
        stock_metrics_c = {} # To store beta and correlation for each stock

        # Fetch historical data for all tickers + SPY
        try:
            hist_data_yf_c = yf.download(all_tickers_for_hist_data_c, period=backtest_period_str, interval="1d", progress=False, session=YFINANCE_SESSION)
            if hist_data_yf_c.empty or 'Close' not in hist_data_yf_c.columns:
                print("Failed to download historical data or 'Close' column missing for Assess C.")
                print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
                return

            # Extract Close prices, handling MultiIndex vs single column
            if isinstance(hist_data_yf_c.columns, pd.MultiIndex):
                close_prices_df_c = hist_data_yf_c['Close']
            elif 'Close' in hist_data_yf_c.columns:
                 # Handle single ticker case - assuming the single 'Close' column is the ticker
                 if len(all_tickers_for_hist_data_c) == 1 and hist_data_yf_c.columns[0] == 'Close':
                      close_prices_df_c = hist_data_yf_c[['Close']]
                      close_prices_df_c.columns = [all_tickers_for_hist_data_c[0]] # Rename column to ticker
                 else:
                      print("Warning: Unexpected historical data format after fetch for Code C.")
                      print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
                      return
            else:
                print("Warning: 'Close' column not found in historical data for Code C.")
                print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
                return

            # Clean column names (remove . if present)
            close_prices_df_c.columns = [col.replace('.', '-') for col in close_prices_df_c.columns]

            # Drop tickers with insufficient data points for the period
            min_data_points_c = 20 # Arbitrary minimum threshold
            close_prices_df_c = close_prices_df_c.dropna(axis=1, thresh=min_data_points_c)

            if 'SPY' not in close_prices_df_c.columns:
                print("Error: Could not get sufficient historical data for SPY for Assess C.")
                print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
                return

            # Calculate daily returns
            daily_returns_df_c = close_prices_df_c.pct_change().dropna()

            if daily_returns_df_c.empty or 'SPY' not in daily_returns_df_c.columns:
                print("Error calculating daily returns or SPY returns missing for Assess C.")
                print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
                return

            spy_returns_series_c = daily_returns_df_c['SPY']
            if spy_returns_series_c.std() == 0: # Check for zero variance in SPY returns
                print("SPY had no price variance in the period. Cannot calculate metrics for Assess C.")
                print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
                return

            for ticker_c in portfolio_tickers_from_custom: # Iterate through tickers from tailored output
                if ticker_c in daily_returns_df_c.columns:
                    ticker_returns_series_c = daily_returns_df_c[ticker_c]
                    if ticker_returns_series_c.std() == 0: # Check for zero variance in ticker returns
                        beta_c, correlation_c = 0.0, 0.0
                    else:
                        # Calculate Beta and Correlation
                        # Ensure both series are aligned by index before calculating covariance/correlation
                        aligned_returns_c = pd.concat([ticker_returns_series_c, spy_returns_series_c], axis=1).dropna()
                        if aligned_returns_c.empty or len(aligned_returns_c.columns) < 2:
                             print(f"Warning: Insufficient aligned data for {ticker_c} vs SPY for Code C metrics.")
                             beta_c, correlation_c = np.nan, np.nan # Cannot calculate
                        else:
                            covariance_matrix_c = np.cov(aligned_returns_c.iloc[:, 0], aligned_returns_c.iloc[:, 1])
                            beta_c = covariance_matrix_c[0, 1] / covariance_matrix_c[1, 1] if covariance_matrix_c[1, 1] != 0 else 0.0
                            correlation_c = np.corrcoef(aligned_returns_c.iloc[:, 0], aligned_returns_c.iloc[:, 1])[0, 1]

                    stock_metrics_c[ticker_c] = {'beta': beta_c, 'correlation': correlation_c}
                else:
                    # Ticker data was insufficient after dropna
                    stock_metrics_c[ticker_c] = {'beta': np.nan, 'correlation': np.nan}
                    print(f"Warning: Insufficient data for {ticker_c} in Assess Code C metrics calculation.")

        except Exception as e_hist_c:
            print(f"Error fetching/processing historical data for Assess C: {e_hist_c}")
            import traceback
            traceback.print_exc()
            print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
            return


# 8. Calculate Weighted Averages using tailored holding values
        # (Assuming this is where the 'try' block should start based on the context)
        try:
            weighted_beta_sum_c = 0.0
            weighted_correlation_sum_c = 0.0

            # Process stock holdings from tailored_holdings_data
            for holding_item_c in tailored_holdings_data: # This is the list of dicts
                ticker_val_c = holding_item_c.get('ticker')
                # ***** IMPORTANT ASSUMPTION: *****
                # The image shows 'custom_value_c' being used, but it's not defined in the visible snippet.
                # I am assuming 'custom_value' is a key in 'holding_item_c' dict.
                # If it's calculated differently, you'll need to adjust this part.
                custom_value_c = holding_item_c.get('custom_value', 0.0)

                if ticker_val_c and custom_value_c > 0: # Ensure ticker is valid and custom_value is positive
                    metrics_c = stock_metrics_c.get(ticker_val_c)
                    if metrics_c:
                        beta_val_c = metrics_c.get('beta', np.nan)
                        correlation_val_c = metrics_c.get('correlation', np.nan)

                        if not np.isnan(beta_val_c):
                            weighted_beta_sum_c += beta_val_c * custom_value_c
                        if not np.isnan(correlation_val_c):
                            weighted_correlation_sum_c += correlation_val_c * custom_value_c

            total_custom_value_c = sum(item.get('custom_value', 0.0) for item in tailored_holdings_data if item.get('ticker') and item.get('custom_value', 0.0) > 0)

            average_beta_c = weighted_beta_sum_c / total_custom_value_c if total_custom_value_c > 0 else 0.0
            average_correlation_c = weighted_correlation_sum_c / total_custom_value_c if total_custom_value_c > 0 else 0.0

            print("\n--- Custom Portfolio Risk Metrics (Code C) ---")

            # 9. Calculate Risk Score based on Weighted Averages
            # Risk Score Logic (Example - this needs to be defined based on your specific risk model)
            # A higher beta generally means higher systematic risk.
            # A higher average correlation might indicate less diversification benefit.
            # This is a simplified example; a real risk model would be more complex.

            risk_score_c = 0.0
            risk_factors_c = []

            # Example Risk Score Calculation based on Beta and Correlation
            # Adjust these weights and thresholds based on your risk tolerance and model
            beta_weight_c = 0.6
            correlation_weight_c = 0.4

            # Normalize or scale beta and correlation if necessary, depending on expected ranges
            # For simplicity, using raw values with thresholds
            beta_risk_c = average_beta_c * beta_weight_c
            correlation_risk_c = average_correlation_c * correlation_weight_c

            risk_score_c = beta_risk_c + correlation_risk_c

            risk_level_c = "Low Risk"
            if risk_score_c > 1.0: # Example threshold
                risk_level_c = "Medium Risk"
            if risk_score_c > 2.0: # Example threshold
                risk_level_c = "High Risk"

            # Add details to risk_factors_c
            risk_factors_c.append(f"Average Beta (weighted by value): {average_beta_c:.4f}")
            risk_factors_c.append(f"Average Correlation to S&P 500 (weighted by value): {average_correlation_c:.4f}")
            risk_factors_c.append(f"Calculated Risk Score (Beta * {beta_weight_c} + Correlation * {correlation_weight_c}): {risk_score_c:.4f}")
            risk_factors_c.append(f"Overall Risk Level: {risk_level_c}")

            print("\n--- Custom Portfolio Risk Assessment (Code C) Results ---")
            for factor in risk_factors_c:
                print(factor)

            print("\n--- Custom Portfolio Risk Assessment (Code C) Complete. ---")

        # Corrected indentation for the except block:
        except Exception as e:
            print(f"An unexpected error occurred during Custom Portfolio Risk Assessment (Code C): {e}")
            import traceback # Keep import here if only used in except block
            traceback.print_exc()
            print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")

async def assess_custom_portfolio(tailored_holdings_data: list, interaction: Optional[any] = None):
    """
    Performs a risk assessment for a custom portfolio based on its holdings.
    Calculates weighted average Beta and Correlation to S&P 500.
    (This is for Assess Code C in the original document)
    """
    print("\n--- Starting Custom Portfolio Risk Assessment (Code C) ---")
    if interaction:
        print(f"Acknowledging interaction for Assess Custom Portfolio Code C.")

    start_time = time.time()
    print(f"Custom Portfolio Risk Assessment (Code C) started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not tailored_holdings_data:
        error_message = "Error [Assess Custom Portfolio]: No tailored holdings data provided. Cannot assess."
        print(error_message)
        if interaction:
            print(error_message)
        print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
        return

    # Extract tickers and their custom values from the tailored_holdings_data
    # tailored_holdings_data is expected to be a list of dicts like:
    # [{'ticker': 'AAPL', 'custom_value': 1000.0}, {'ticker': 'MSFT', 'custom_value': 500.0}]
    # Ensure 'custom_value' key exists and is used for weighting.

    portfolio_tickers_c = [item['ticker'] for item in tailored_holdings_data if 'ticker' in item and item.get('custom_value', 0.0) > 0]
    if not portfolio_tickers_c:
        error_message = "Error [Assess Custom Portfolio]: No valid tickers with positive values found in tailored holdings data."
        print(error_message)
        if interaction:
            print(error_message)
        print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
        return

    print(f"Assessing custom portfolio with tickers: {', '.join(portfolio_tickers_c)}")

    # Define the backtesting period (e.g., 1 year, 5 years)
    # For consistency, let's use a predefined period or prompt the user if interactive.
    # For now, using a fixed 1-year period.
    backtest_period_c = "5y" # Example: 5-year backtesting period
    print(f"Using backtesting period: {backtest_period_c}")

    # --- Fetch Historical Data for Portfolio Tickers and SPY ---
    all_tickers_for_hist_data_c = list(set(portfolio_tickers_c + ['SPY']))
    stock_metrics_c = {} # To store beta and correlation for each stock

    try:
        print("Fetching historical data for Assess Code C...")
        hist_data_yf_c = yf.download(
            all_tickers_for_hist_data_c,
            period=backtest_period_c,
            interval="1d",
            progress=False,
            session=YFINANCE_SESSION if IS_CURL_CFFI_ACTIVE else None
        )

        if hist_data_yf_c.empty or 'Close' not in hist_data_yf_c.columns:
            error_message = "Failed to download historical data or 'Close' column missing for Assess C."
            print(error_message)
            if interaction:
                print(error_message)
            print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
            return

        # Extract Close prices, handling MultiIndex vs single column
        if isinstance(hist_data_yf_c.columns, pd.MultiIndex):
            close_prices_df_c = hist_data_yf_c['Close']
        elif 'Close' in hist_data_yf_c.columns:
            if len(all_tickers_for_hist_data_c) == 1 and hist_data_yf_c.columns[0] == 'Close':
                close_prices_df_c = hist_data_yf_c[['Close']]
                close_prices_df_c.columns = [all_tickers_for_hist_data_c[0]]
            else:
                # This part is tricky if not MultiIndex and multiple tickers were requested.
                # Assume yfinance returns individual columns for each ticker if not MultiIndex.
                # We need to select the correct columns.
                # For simplicity, let's try to select by ticker name if they exist as columns.
                # This might need adjustment based on yfinance's behavior for multiple tickers without MultiIndex.
                # A safer way is to ensure yfinance always returns MultiIndex or handle it more robustly.
                # For now, proceeding with the assumption that direct column selection might work.
                # This is a common point of failure if yfinance API changes or returns unexpected format.

                # Check if SPY and portfolio tickers are in columns
                missing_cols = [col for col in all_tickers_for_hist_data_c if col not in hist_data_yf_c.columns]
                if missing_cols:
                    print(f"Warning: Missing data columns for: {', '.join(missing_cols)} in Assess Code C historical data.")
                
                # Attempt to select the relevant columns
                cols_to_select = [col for col in all_tickers_for_hist_data_c if col in hist_data_yf_c.columns]
                if not cols_to_select:
                    print("Error: No relevant ticker columns found in historical data for Assess Code C.")
                    print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
                    return
                close_prices_df_c = hist_data_yf_c[cols_to_select]


        else:
            error_message = "Warning: 'Close' column not found in historical data for Code C."
            print(error_message)
            if interaction:
                print(error_message)
            print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
            return

        # Clean column names (remove . if present)
        close_prices_df_c.columns = [col.replace('.', '-') for col in close_prices_df_c.columns]

        # Drop tickers with insufficient data points for the period
        min_data_points_c = 20 # Arbitrary minimum threshold
        close_prices_df_c = close_prices_df_c.dropna(axis=1, thresh=min_data_points_c)

        if 'SPY' not in close_prices_df_c.columns:
            error_message = "Error: Could not get sufficient historical data for SPY for Assess C."
            print(error_message)
            if interaction:
                print(error_message)
            print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
            return

        # Calculate daily returns
        daily_returns_df_c = close_prices_df_c.pct_change().dropna()

        if daily_returns_df_c.empty or 'SPY' not in daily_returns_df_c.columns:
            error_message = "Error calculating daily returns or SPY returns missing for Assess C."
            print(error_message)
            if interaction:
                print(error_message)
            print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
            return

        spy_returns_series_c = daily_returns_df_c['SPY']
        if spy_returns_series_c.std() == 0: # Check for zero variance in SPY returns
            error_message = "SPY had no price variance in the period. Cannot calculate metrics for Assess C."
            print(error_message)
            if interaction:
                print(error_message)
            print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
            return

        for ticker_c in portfolio_tickers_c: # Iterate through tickers from tailored output
            if ticker_c in daily_returns_df_c.columns:
                ticker_returns_series_c = daily_returns_df_c[ticker_c]
                if ticker_returns_series_c.std() == 0: # Check for zero variance in ticker returns
                    beta_c, correlation_c = 0.0, 0.0
                else:
                    # Calculate Beta and Correlation
                    # Ensure both series are aligned by index before calculating covariance/correlation
                    aligned_returns_c = pd.concat([ticker_returns_series_c, spy_returns_series_c], axis=1).dropna()
                    if aligned_returns_c.empty or len(aligned_returns_c.columns) < 2:
                        print(f"Warning: Insufficient aligned data for {ticker_c} vs SPY for Code C metrics.")
                        beta_c, correlation_c = np.nan, np.nan # Cannot calculate
                    else:
                        covariance_matrix_c = np.cov(aligned_returns_c.iloc[:, 0], aligned_returns_c.iloc[:, 1])
                        beta_c = covariance_matrix_c[0, 1] / covariance_matrix_c[1, 1] if covariance_matrix_c[1, 1] != 0 else 0.0
                        correlation_c = np.corrcoef(aligned_returns_c.iloc[:, 0], aligned_returns_c.iloc[:, 1])[0, 1]

                stock_metrics_c[ticker_c] = {'beta': beta_c, 'correlation': correlation_c}
            else:
                # Ticker data was insufficient after dropna
                stock_metrics_c[ticker_c] = {'beta': np.nan, 'correlation': np.nan}
                print(f"Warning: Insufficient data for {ticker_c} in Assess Code C metrics calculation.")

    except Exception as e_hist_c:
        print(f"An error occurred during historical data fetching/processing for Assess C: {e_hist_c}")
        import traceback
        traceback.print_exc()
        if interaction:
            print(f"An error occurred during historical data fetching/processing for Assess C: {e_hist_c}")
        print("--- Custom Portfolio Risk Assessment (Code C) failed. ---")
        return

    # --- Calculate Weighted Averages using tailored holding values ---
    weighted_beta_sum_c = 0.0
    weighted_correlation_sum_c = 0.0
    total_custom_value_c = 0.0

    # Iterate through the provided tailored_holdings_data to get custom_value for each ticker
    for holding_item_c in tailored_holdings_data:
        ticker_val_c = holding_item_c.get('ticker')
        custom_value_c = holding_item_c.get('custom_value', 0.0) # Get custom_value, default to 0.0

        if ticker_val_c and custom_value_c > 0: # Ensure ticker is valid and custom_value is positive
            total_custom_value_c += custom_value_c # Sum up total value of holdings
            metrics_c = stock_metrics_c.get(ticker_val_c)
            if metrics_c:
                beta_val_c = metrics_c.get('beta', np.nan)
                correlation_val_c = metrics_c.get('correlation', np.nan)

                if not np.isnan(beta_val_c):
                    weighted_beta_sum_c += beta_val_c * custom_value_c
                else:
                    print(f"Warning: Beta not available for {ticker_val_c}, excluded from weighted beta sum.")

                if not np.isnan(correlation_val_c):
                    weighted_correlation_sum_c += correlation_val_c * custom_value_c
                else:
                    print(f"Warning: Correlation not available for {ticker_val_c}, excluded from weighted correlation sum.")
            else:
                print(f"Warning: Metrics not found for {ticker_val_c}. It might have been dropped due to insufficient historical data.")


    average_beta_c = weighted_beta_sum_c / total_custom_value_c if total_custom_value_c > 0 else 0.0
    average_correlation_c = weighted_correlation_sum_c / total_custom_value_c if total_custom_value_c > 0 else 0.0

    # --- Calculate Risk Score based on Weighted Averages ---
    # This is a simplified example; a real risk model would be more complex.
    risk_score_c = 0.0
    risk_factors_c = []

    # Example Risk Score Calculation based on Beta and Correlation
    # Adjust these weights and thresholds based on your risk tolerance and model
    beta_weight_c = 0.6
    correlation_weight_c = 0.4

    beta_risk_component_c = average_beta_c * beta_weight_c
    correlation_risk_component_c = average_correlation_c * correlation_weight_c

    risk_score_c = beta_risk_component_c + correlation_risk_component_c # Simple additive model

    risk_level_c = "Low Risk"
    if risk_score_c > 0.75 and risk_score_c <= 1.25 : risk_level_c = "Medium Risk" # Example thresholds
    elif risk_score_c > 1.25 : risk_level_c = "High Risk"


    print("\n--- Custom Portfolio Risk Assessment (Code C) Results ---")
    if interaction:
        results_message = f"--- Custom Portfolio Risk Assessment (Code C) for {len(portfolio_tickers_c)} tickers ---\n"
        results_message += f"Backtesting Period: {backtest_period_c}\n"
        results_message += f"Total Portfolio Value Assessed: ${total_custom_value_c:,.2f}\n"
        results_message += f"Weighted Average Beta: {average_beta_c:.4f}\n"
        results_message += f"Weighted Average Correlation to S&P 500: {average_correlation_c:.4f}\n"
        results_message += f"Calculated Risk Score (Beta * {beta_weight_c:.2f} + Corr * {correlation_weight_c:.2f}): {risk_score_c:.4f}\n"
        results_message += f"Overall Risk Level: {risk_level_c}\n"
        results_message += "--- End of Custom Portfolio Risk Assessment (Code C) ---\n"
        print(results_message)
    else: # Print to console if no interaction object
        print(f"Backtesting Period: {backtest_period_c}")
        print(f"Total Portfolio Value Assessed: ${total_custom_value_c:,.2f}")
        print(f"Weighted Average Beta: {average_beta_c:.4f}")
        print(f"Weighted Average Correlation to S&P 500: {average_correlation_c:.4f}")
        print(f"Calculated Risk Score (Beta * {beta_weight_c:.2f} + Corr * {correlation_weight_c:.2f}): {risk_score_c:.4f}")
        print(f"Overall Risk Level: {risk_level_c}")

    end_time = time.time()
    print(f"Custom Portfolio Risk Assessment (Code C) finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
    print("--- Custom Portfolio Risk Assessment (Code C) Complete. ---")
# --- END OF ASSESS_CUSTOM_PORTFOLIO (CODE C) ---

# --- START OF ASSESS_MARKET (CODE M) ---

async def assess_market(time_horizon_m: str, interaction: Optional[any] = None):
    """
    Performs a market assessment using TradingView screener data.
    Identifies top stocks based on a custom scoring system.
    """
    print("\n--- Starting Market Assessment (Code M) ---")
    if interaction:
         print(f"Acknowledging interaction for Assess Market Code M.")

    start_time = time.time()
    print(f"Market Assessment (Code M) started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Map time horizon string to TradingView Interval
    tv_interval_map = {
        "1m": Interval.INTERVAL_1_MINUTE,
        "5m": Interval.INTERVAL_5_MINUTES,
        "15m": Interval.INTERVAL_15_MINUTES,
        "30m": Interval.INTERVAL_30_MINUTES,
        "1h": Interval.INTERVAL_1_HOUR,
        "4h": Interval.INTERVAL_4_HOURS,
        "1d": Interval.INTERVAL_1_DAY,
        "1w": Interval.INTERVAL_1_WEEK,
        "1M": Interval.INTERVAL_1_MONTH,
    }

    interval_m = tv_interval_map.get(time_horizon_m)

    if not interval_m:
        error_message = f"Invalid time horizon '{time_horizon_m}' for Market Assessment (Code M). Supported: {', '.join(tv_interval_map.keys())}"
        print(error_message)
        if interaction:
             print(error_message)
        print("--- Market Assessment (Code M) failed due to invalid time horizon. ---")
        return

    print(f"Using time horizon (interval): {time_horizon_m}")

    # Define screener query - focusing on US stocks, major exchanges
    # Adding more columns for potentially better scoring
    screener_query = Query(
        market="america",
        screener="america",
        symbols={"query": ["."]}, # Query all symbols
        columns=[
            Column.performance_ytd,
            Column.performance_ytd_abs,
            Column.relative_volume_10d_avg, # Relative Volume
            Column.volume,
            Column.volume_abs,
            Column.basic_eps_diluted_y, # Annual EPS
            Column.earnings_per_share_forecast_y, # Annual EPS Forecast
            Column.price_earnings_ttm, # P/E Ratio
            Column.price_to_earnings_growth_ttm, # PEG Ratio
            Column.dividend_yield_paid, # Dividend Yield
            Column.market_capitalization, # Market Cap
            Column.beta, # Beta
            Column.volatility_week, # Weekly Volatility
            Column.volatility_month, # Monthly Volatility
            Column.technical_rating, # Technical Rating
            Column.return_on_equity, # ROE
            Column.return_on_assets, # ROA
            Column.return_on_invested_capital, # ROIC
            Column.gross_margin, # Gross Margin
            Column.operating_margin, # Operating Margin
            Column.net_margin, # Net Margin
            Column.total_revenue_y, # Annual Revenue
            Column.revenue_growth_yoy, # Revenue Growth YoY
            Column.earnings_growth_yoy, # Earnings Growth YoY
            Column.number_of_employees, # Number of Employees
            Column.recommendation_trend, # Recommendation Trend
            Column.average_recommendation, # Average Recommendation
            Column.description, # Company Description
            Column.country, # Country
            Column.sector, # Sector
            Column.industry, # Industry
            Column.float_shares, # Float
            Column.outstanding_shares, # Outstanding Shares
            Column.shares_float_vs_outstanding, # Float vs Outstanding
            Column.price_to_sales_ttm, # Price to Sales
            Column.price_to_book_mrq, # Price to Book
            Column.current_ratio_mrq, # Current Ratio
            Column.debt_to_equity_mrq, # Debt to Equity
            Column.free_cash_flow_per_share_ttm, # FCF per Share
            Column.quick_ratio_mrq, # Quick Ratio
            Column.holders_major, # Major Holders
            Column.holders_mutual_funds, # Mutual Fund Holders
            Column.holders_other_institutions, # Other Institutions Holders
            Column.holders_insiders, # Insiders Holders
            Column.analyst_ratings, # Analyst Ratings
            Column.average_volume_10d_calc, # Average Volume 10 Day
            Column.average_volume_30d_calc, # Average Volume 30 Day
            Column.average_volume_90d_calc, # Average Volume 90 Day
            Column.earnings_date, # Earnings Date
            Column.fundamental_currency_code, # Currency
            Column.dividend_yield, # Dividend Yield (different source?)
            Column.dividend_date, # Dividend Date
            Column.dividends_paid, # Total Dividends Paid
            Column.weekly_performance, # Weekly Performance
            Column.monthly_performance, # Monthly Performance
            Column.three_month_performance, # 3 Month Performance
            Column.six_month_performance, # 6 Month Performance
            Column.year_to_date_performance, # YTD Performance
            Column.one_year_performance, # 1 Year Performance
            Column.three_year_performance, # 3 Year Performance
            Column.five_year_performance, # 5 Year Performance
            Column.ten_year_performance, # 10 Year Performance
            Column.fifty_two_week_high, # 52 Week High
            Column.fifty_two_week_low, # 52 Week Low
            Column.fifty_two_week_high_low_delta, # 52 Week High/Low Delta
            Column.fifty_two_week_high_low_delta_abs, # 52 Week High/Low Delta Abs
            Column.moving_averages_buy, # Moving Averages Buy signal (Technical)
            Column.moving_averages_sell, # Moving Averages Sell signal (Technical)
            Column.moving_averages_neutral, # Moving Averages Neutral signal (Technical)
            Column.oscillators_buy, # Oscillators Buy signal (Technical)
            Column.oscillators_sell, # Oscillators Sell signal (Technical)
            Column.oscillators_neutral, # Oscillators Neutral signal (Technical)
            Column.technical_buy, # Technical Buy signal (Overall Technical)
            Column.technical_sell, # Technical Sell signal (Overall Technical)
            Column.technical_neutral, # Technical Neutral signal (Overall Technical)
            Column.candlestick_patterns, # Candlestick Patterns
            Column.price, # Current Price
            Column.change, # Change
            Column.change_abs, # Change Absolute
        ]
    )

    try:
        print("Fetching data from TradingView Screener...")
        if interaction:
             print("Fetching data from TradingView Screener...")
        screener_data = screener_query.get_data()
        print(f"Fetched {len(screener_data)} potential tickers from TradingView Screener.")
        if interaction:
             print(f"Fetched {len(screener_data)} potential tickers from TradingView Screener.")

        if not screener_data:
            no_data_message = "No data returned from TradingView Screener."
            print(no_data_message)
            if interaction:
                 print(no_data_message)
            print("--- Market Assessment (Code M) failed due to no screener data. ---")
            return

        # Filter out non-stock entries if necessary (e.g., indices, funds)
        # Basic filtering based on common patterns or known non-stock symbols might be needed.
        # For now, we'll process what the screener returns, assuming it's mostly stocks for the 'america' market/screener.

        # Create a pandas DataFrame for easier manipulation
        screener_df = pd.DataFrame(screener_data)

        # Clean column names (remove spaces, special chars)
        screener_df.columns = screener_df.columns.str.replace('[^A-Za-z0-9_]+', '', regex=True)

        # Convert relevant columns to numeric, coercing errors to NaN
        numeric_cols = [
            'performance_ytd', 'performance_ytd_abs', 'relative_volume_10d_avg',
            'volume', 'volume_abs', 'basic_eps_diluted_y', 'earnings_per_share_forecast_y',
            'price_earnings_ttm', 'price_to_earnings_growth_ttm', 'dividend_yield_paid',
            'market_capitalization', 'beta', 'volatility_week', 'volatility_month',
            'return_on_equity', 'return_on_assets', 'return_on_invested_capital',
            'gross_margin', 'operating_margin', 'net_margin', 'total_revenue_y',
            'revenue_growth_yoy', 'earnings_growth_yoy', 'number_of_employees',
            'float_shares', 'outstanding_shares', 'shares_float_vs_outstanding',
            'price_to_sales_ttm', 'price_to_book_mrq', 'current_ratio_mrq',
            'debt_to_equity_mrq', 'free_cash_flow_per_share_ttm', 'quick_ratio_mrq',
            'average_volume_10d_calc', 'average_volume_30d_calc', 'average_volume_90d_calc',
            'dividend_yield', 'dividends_paid', 'weekly_performance',
            'monthly_performance', 'three_month_performance', 'six_month_performance',
            'year_to_date_performance', 'one_year_performance', 'three_year_performance',
            'five_year_performance', 'ten_year_performance', 'fifty_two_week_high',
            'fifty_two_week_low', 'fifty_two_week_high_low_delta', 'fifty_two_week_high_low_delta_abs',
            'price', 'change', 'change_abs',
        ]
        for col in numeric_cols:
            if col in screener_df.columns:
                screener_df[col] = pd.to_numeric(screener_df[col], errors='coerce')
                # Fill NaN values with a neutral value (e.g., 0 or median), depending on the metric
                # For now, let's fill with 0, but this might need refinement
                screener_df[col] = screener_df[col].fillna(0)


        # --- Custom Scoring Logic ---
        # This is the core of the market assessment. Define weights and criteria
        # based on what factors you believe are important for identifying promising stocks.
        # This is a highly customizable section.

        print("Applying custom scoring logic...")
        if interaction:
             print("Applying custom scoring logic...")

        # Initialize 'custom_invest_score' column
        screener_df['custom_invest_score'] = 0.0

        # Define weights for different factors
        # These weights are examples and should be adjusted based on your strategy.
        weights = {
            'performance_ytd': 0.1, # Higher YTD performance is good
            'relative_volume_10d_avg': 0.05, # Higher relative volume can indicate interest
            'basic_eps_diluted_y': 0.1, # Positive EPS is good
            'earnings_per_share_forecast_y': 0.1, # Positive EPS forecast is good
            'price_earnings_ttm': -0.05, # Lower P/E might be better (handle high/negative P/E)
            'price_to_earnings_growth_ttm': -0.05, # Lower PEG might be better (handle high/negative PEG)
            'dividend_yield_paid': 0.03, # Dividend yield can be a factor
            'market_capitalization': 0.02, # Market cap can be a factor (e.g., favor certain sizes)
            'beta': -0.05, # Lower beta for lower volatility (depending on strategy)
            'return_on_equity': 0.07, # Higher ROE is generally good
            'return_on_assets': 0.05, # Higher ROA is generally good
            'return_on_invested_capital': 0.05, # Higher ROIC is generally good
            'gross_margin': 0.03, # Higher margins are generally good
            'operating_margin': 0.03,
            'net_margin': 0.05,
            'revenue_growth_yoy': 0.07, # Higher revenue growth is good
            'earnings_growth_yoy': 0.07, # Higher earnings growth is good
            'recommendation_trend': 0.05, # Positive recommendation trend is good (need to interpret this column)
            'average_recommendation': -0.05, # Lower average recommendation (closer to 1 or 2) is good
            'price_to_sales_ttm': -0.03, # Lower P/S might be better
            'price_to_book_mrq': -0.03, # Lower P/B might be better
            'current_ratio_mrq': 0.03, # Higher current ratio is generally good
            'debt_to_equity_mrq': -0.03, # Lower debt to equity is generally good
            'free_cash_flow_per_share_ttm': 0.05, # Positive FCF per share is good
            'quick_ratio_mrq': 0.03, # Higher quick ratio is generally good
            # Technical indicators from Technical Rating (need to interpret this)
            # For now, let's use the overall 'technical_rating' if it's a score, or interpret 'technical_buy', 'technical_sell', 'technical_neutral'
            'technical_buy': 0.05, # Higher technical buy signals are good
            'technical_sell': -0.05, # Higher technical sell signals are bad
            # candlestick_patterns can be a factor, but harder to incorporate in a simple score
        }

        # Apply weights to calculate the score
        for col, weight in weights.items():
            if col in screener_df.columns:
                # Add scoring logic for each column based on its nature (higher=better or lower=better)
                if weight > 0: # Higher is better
                    screener_df['custom_invest_score'] += screener_df[col] * weight
                else: # Lower is better (handle potential division by zero or extreme values for ratios)
                    # For ratios like P/E, PEG, P/S, P/B, handle infinity or very large values resulting from 0 or negative denominators
                    if col in ['price_earnings_ttm', 'price_to_earnings_growth_ttm', 'price_to_sales_ttm', 'price_to_book_mrq', 'debt_to_equity_mrq']:
                         # Cap extreme values or use inverse
                         # Example: using inverse with a small epsilon to avoid division by zero
                         epsilon = 1e-6
                         screener_df['custom_invest_score'] += (1 / (screener_df[col].replace(0, np.nan).fillna(epsilon))) * abs(weight) * np.sign(weight) # Use sign to maintain direction
                    else: # For metrics like beta, technical_sell where lower is better
                         screener_df['custom_invest_score'] += screener_df[col] * weight

        # Add score based on Technical Rating interpretation (example)
        # Assuming technical_rating is a string like "Strong Buy", "Buy", "Neutral", "Sell", "Strong Sell"
        if 'technical_rating' in screener_df.columns:
            rating_scores = {"Strong Buy": 2.0, "Buy": 1.0, "Neutral": 0.0, "Sell": -1.0, "Strong Sell": -2.0}
            screener_df['technical_rating_score'] = screener_df['technical_rating'].map(rating_scores).fillna(0.0)
            screener_df['custom_invest_score'] += screener_df['technical_rating_score'] * 0.1 # Example weight for technical rating

        # Add score based on Analyst Recommendation interpretation (example)
        # Assuming average_recommendation is a number (e.g., 1.5 for Buy)
        if 'average_recommendation' in screener_df.columns:
            # Lower average recommendation number is better (closer to 1)
            screener_df['custom_invest_score'] += (5 - screener_df['average_recommendation']) * 0.05 # Example weight

        # Consider adding scores based on volatility (lower volatility might be preferred depending on strategy)
        if 'volatility_week' in screener_df.columns:
            screener_df['custom_invest_score'] += -screener_df['volatility_week'] * 0.02 # Example weight

        if 'volatility_month' in screener_df.columns:
            screener_df['custom_invest_score'] += -screener_df['volatility_month'] * 0.02 # Example weight

# --- End of Custom Scoring Logic ---

        # Sort by the custom score in descending order
        scored_stocks_m = screener_df.sort_values(by='custom_invest_score', ascending=False).reset_index(drop=True)

        print("Custom scoring applied and stocks sorted.")
        if interaction:
             print("Custom scoring applied and stocks sorted.")

        # Select the top N stocks (e.g., top 10 or 20)
        top_n = 20  # You can adjust this number
        top_stocks_m = scored_stocks_m.head(top_n)

        print(f"\n--- Top {top_n} Market Assessment (Code M) Results ({time_horizon_m}) ---")
        if interaction:
             print(f"\n--- Top {top_n} Market Assessment (Code M) Results ({time_horizon_m}) ---")

        # Format and display the results
        if not top_stocks_m.empty:
            # Select relevant columns for display
            display_cols = [
                'symbol',
                'name',
                'exchange',
                'price',
                'change',
                'change_abs',
                'volume',
                'relative_volume_10d_avg',
                'market_capitalization',
                'sector',
                'industry',
                'technical_rating',
                'custom_invest_score'
            ]
            # Ensure display_cols exist in the dataframe before selecting
            display_df_m = top_stocks_m[[col for col in display_cols if col in top_stocks_m.columns]]

            # Format numerical columns for better readability
            for col in ['price', 'change', 'change_abs', 'volume', 'market_capitalization', 'custom_invest_score', 'relative_volume_10d_avg']:
                if col in display_df_m.columns:
                    if col in ['price', 'change', 'change_abs', 'custom_invest_score']:
                        display_df_m[col] = display_df_m[col].map(lambda x: f'{x:.2f}' if pd.notnull(x) else 'N/A')
                    elif col in ['volume', 'market_capitalization']:
                         display_df_m[col] = display_df_m[col].apply(lambda x: f'{x:,.0f}' if pd.notnull(x) else 'N/A')
                    elif col in ['relative_volume_10d_avg']:
                         display_df_m[col] = display_df_m[col].map(lambda x: f'{x:.2f}' if pd.notnull(x) else 'N/A')


            # Rename columns for display
            display_df_m.rename(columns={
                'symbol': 'Ticker',
                'name': 'Name',
                'exchange': 'Exchange',
                'price': 'Price',
                'change': 'Change (%)',
                'change_abs': 'Change ($)',
                'volume': 'Volume',
                'relative_volume_10d_avg': 'Rel Volume (10d Avg)',
                'market_capitalization': 'Market Cap',
                'sector': 'Sector',
                'industry': 'Industry',
                'technical_rating': 'Tech Rating',
                'custom_invest_score': 'INVEST Score'
            }, inplace=True)


            table_m = tabulate(display_df_m, headers='keys', tablefmt='psql')
            print(table_m)
            if interaction:
                 # Discord message limit is ~2000 characters, may need to split large tables
                 if len(table_m) > 1900:
                     # Split table into chunks
                     lines = table_m.split('\n')
                     chunk_size = 40 # Approx number of rows per message
                     for i in range(0, len(lines), chunk_size):
                         chunk = '\n'.join(lines[i:i+chunk_size])
                         print(f"```psql\n{chunk}\n```")
                 else:
                    print(f"```psql\n{table_m}\n```")


            # Optional: Save the full scored data to a file (e.g., CSV) for later analysis
            # This can be controlled by a flag or command parameter if needed
            # save_market_data_full(scored_stocks_m, time_horizon_m) # Implement this function if needed

             # Auto-Save Market Data (Top 10 ticker and price only)
            # This is for the auto-save feature, saving a simplified view
            top_10_for_save = top_stocks_m.head(10)
            if not top_10_for_save.empty:
                # Ensure 'symbol' and 'price' columns exist before saving
                if 'Ticker' in top_10_for_save.columns and 'Price' in top_10_for_save.columns:
                     # Rename columns back to original or save with simple names
                     save_df_m = top_10_for_save[['Ticker', 'Price']].copy()
                     # Convert Price back to numeric for consistent saving format if needed, or keep as string
                     # save_df_m['Price'] = pd.to_numeric(save_df_m['Price'], errors='coerce')
                     save_market_data_top_ten(save_df_m)
                else:
                     print("Warning: 'Ticker' or 'Price' column not found in top_10_for_save DataFrame. Cannot save market data.")


        else:
            no_results_message = f"No top {top_n} stocks found based on the custom scoring criteria for {time_horizon_m}."
            print(no_results_message)
            if interaction:
                 print(no_results_message)

    except Exception as e:
        print(f"An error occurred during Market Assessment (Code M): {e}")
        import traceback
        traceback.print_exc()
        if interaction:
             print(f"An error occurred during Market Assessment (Code M): {e}")

    end_time = time.time()
    print(f"Market Assessment (Code M) finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken for Market Assessment (Code M): {end_time - start_time:.2f} seconds.")
    print("--- Market Assessment (Code M) Complete. ---")

# --- END OF ASSESS_MARKET (CODE M) ---


# --- START OF SAVE_MARKET_DATA_TOP_TEN (CODE SM10) ---

def save_market_data_top_ten(top_ten_df_m: pd.DataFrame):
    """
    Saves the top 10 market assessment results (Ticker, Price) to a CSV file.
    This is intended for the auto-save feature's simplified output.
    """
    print("\n--- Starting Save Market Data Top Ten (Code SM10) ---")
    try:
        # Define the save directory and filename
        save_dir = "INVEST_Data/Market_Assessment"
        os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist

        # Generate a unique filename with a timestamp and a sens code (if applicable, using a default here)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sens_code = 3725 # Using a default sens code for the auto-save top 10
        save_file = os.path.join(save_dir, f"market_top_ten_sens_{sens_code}_{timestamp}.csv")

        # Ensure the DataFrame has the expected columns
        if 'Ticker' not in top_ten_df_m.columns or 'Price' not in top_ten_df_m.columns:
            print("Error [Save Market Top Ten]: Input DataFrame missing 'Ticker' or 'Price' column.")
            print("--- Save Market Data Top Ten (Code SM10) failed. ---")
            return

        # Save to CSV
        top_ten_df_m.to_csv(save_file, index=False)

        print(f"--- Successfully saved top 10 market data to '{save_file}'. ---")

    except Exception as e:
        print(f"An error occurred during Save Market Data Top Ten (Code SM10): {e}")
        import traceback
        traceback.print_exc()
        print("--- Save Market Data Top Ten (Code SM10) failed. ---")

# --- END OF SAVE_MARKET_DATA_TOP_TEN (CODE SM10) ---


# --- START OF SAVE_MARKET_DATA_FULL (CODE SMF) ---
# This function is for saving the FULL market data, potentially triggered manually.
# Implement this if you need a separate command to save the entire screener output.
# For now, it's just a placeholder as the auto-save only saves the top 10.
# If you implement this, uncomment the call in assess_market.

# def save_market_data_full(full_df_m: pd.DataFrame, time_horizon_m: str):
#     """
#     Saves the full market assessment results to a CSV file.
#     Intended for manual save command.
#     """
#     print("\n--- Starting Save Market Data Full (Code SMF) ---")
#     try:
#         save_dir = "INVEST_Data/Market_Assessment"
#         os.makedirs(save_dir, exist_ok=True)
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         # Use a different sens code or indicator for the full save
#         sens_code = 7357 # Example sens code for full save
#         save_file = os.path.join(save_dir, f"market_full_sens_{sens_code}_{timestamp}_{time_horizon_m}.csv")

#         # Save the full DataFrame
#         full_df_m.to_csv(save_file, index=False)

#         print(f"--- Successfully saved full market data to '{save_file}'. ---")

#     except Exception as e:
#         print(f"An error occurred during Save Market Data Full (Code SMF): {e}")
#         import traceback
#         traceback.print_exc()
#         print("--- Save Market Data Full (Code SMF) failed. ---")

# --- END OF SAVE_MARKET_DATA_FULL (CODE SMF) ---


# --- START OF ASSESS_BREAKOUT (CODE B) ---

async def assess_breakout(time_horizon_b: str, interaction: Optional[any] = None):
    """
    Performs a breakout assessment using technical analysis.
    Identifies potential breakout stocks based on specified criteria.
    """
    print("\n--- Starting Breakout Assessment (Code B) ---")
    if interaction:
         print(f"Acknowledging interaction for Assess Breakout Code B.")

    start_time = time.time()
    print(f"Breakout Assessment (Code B) started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Mapping for TradingView TA intervals
    tv_ta_interval_map = {
        "1m": Interval.INTERVAL_1_MINUTE,
        "5m": Interval.INTERVAL_5_MINUTES,
        "15m": Interval.INTERVAL_15_MINUTES,
        "30m": Interval.INTERVAL_30_MINUTES,
        "1h": Interval.INTERVAL_1_HOUR,
        "4h": Interval.INTERVAL_4_HOURS,
        "1d": Interval.INTERVAL_1_DAY,
        "1w": Interval.INTERVAL_1_WEEK,
        "1M": Interval.INTERVAL_1_MONTH,
    }

    interval_b = tv_ta_interval_map.get(time_horizon_b)

    if not interval_b:
        error_message = f"Invalid time horizon '{time_horizon_b}' for Breakout Assessment (Code B). Supported: {', '.join(tv_ta_interval_map.keys())}"
        print(error_message)
        if interaction:
             print(error_message)
        print("--- Breakout Assessment (Code B) failed due to invalid time horizon. ---")
        return

    print(f"Using time horizon (interval): {time_horizon_b}")

    # --- Breakout Strategy Parameters ---
    # Define the criteria for identifying potential breakouts.
    # This is a highly customizable section based on your breakout strategy.

    # Example Criteria (these are illustrative and should be refined):
    # 1. Price crosses above a key moving average (e.g., 50-day or 200-day).
    # 2. Increased volume accompanying the price move.
    # 3. Technical indicators showing bullish momentum (e.g., RSI, MACD).
    # 4. Price breaking above a recent resistance level or pattern.

    # For this implementation using tradingview_ta, we can leverage its analysis results.
    # The TA_Handler provides 'technical_indicators' which include ratings like 'BUY', 'SELL', 'NEUTRAL'.
    # We can look for stocks with a "Strong Buy" or "Buy" rating based on the chosen interval.

    # --- Get a list of potential tickers ---
    # Since tradingview_ta analyzes individual tickers, we need a source for tickers.
    # We can use the results from the Market Assessment (Code M) if it was run,
    # or use a predefined list, or fetch a list from another source (e.g., a list of S&P 500 stocks).
    # For demonstration, let's use a small predefined list of tickers.
    # In a real application, you would integrate this with a ticker universe source.

    # Option 1: Use a predefined list (simple for testing)
    # potential_tickers_b = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "JPM", "V"]

    # Option 2: (Recommended) Integrate with a broader ticker source or the Market Assessment results
    # If Market Assessment (Code M) was run prior, you could potentially access its results here.
    # However, to keep functions independent, fetching a list of tickers within this function might be better.
    # A comprehensive list of US stocks can be large, so consider focusing on major exchanges or indices.

    # Let's fetch a list of S&P 500 tickers using yfinance as an example
    print("Fetching S&P 500 tickers...")
    sp500_ticker = "^GSPC"
    try:
        sp500_info = yf.Ticker(sp500_ticker, session=YFINANCE_SESSION) if IS_CURL_CFFI_ACTIVE else yf.Ticker(sp500_ticker)
        # Note: yfinance Ticker object for index like GSPC doesn't directly provide constituent tickers.
        # Need an alternative method to get S&P 500 list. A static list or a web scrape might be necessary.
        # For simplicity and to proceed, let's use a predefined list for now.
        potential_tickers_b = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "JPM", "V", "UNH", "XOM", "LLY", "JNJ", "V"] # Extended list
        print(f"Using a predefined list of {len(potential_tickers_b)} potential tickers for Breakout Assessment.")

    except Exception as e:
        print(f"Error fetching S&P 500 tickers (using predefined list): {e}")
        potential_tickers_b = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"] # Fallback to smaller list
        print(f"Using a fallback predefined list of {len(potential_tickers_b)} potential tickers.")


    breakout_candidates_b = []
    analyzed_count = 0
    print(f"Analyzing {len(potential_tickers_b)} tickers for breakout signals...")
    if interaction:
         print(f"Analyzing {len(potential_tickers_b)} tickers for breakout signals using {time_horizon_b} interval...")


    # Process tickers in smaller batches to manage requests and potential rate limits
    batch_size = 10
    for i in range(0, len(potential_tickers_b), batch_size):
        batch_tickers = potential_tickers_b[i:i + batch_size]
        print(f"Processing batch {int(i/batch_size) + 1}/{(len(potential_tickers_b) + batch_size - 1) // batch_size}...")

        # tradingview_ta.TA_Handler can process multiple symbols in one go
        handlers = [TA_Handler(
            symbol=ticker,
            exchange="NASDAQ" if ticker in ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META"] else "NYSE", # Basic exchange assumption
            screener="america",
            interval=interval_b
        ) for ticker in batch_tickers]

        for handler in handlers:
            try:
                analysis = handler.get_analysis()
                analyzed_count += 1

                # --- Breakout Signal Detection ---
                # Interpret the analysis results to find breakout signals.
                # This part is crucial and depends on how you define a breakout using the available data.

                # Example: Check the Technical Rating summary
                technical_summary = analysis.summary
                # print(f"Analyzing {handler.symbol}: {technical_summary}") # Debugging

                # Criteria 1: Overall Technical Rating is "BUY" or "STRONG BUY"
                # This is a simplified approach. A more sophisticated strategy would look at specific indicators.
                if technical_summary and technical_summary['RECOMMENDATION'] in ['BUY', 'STRONG BUY']:

                    # Optional: Add more specific checks using individual indicators if needed
                    # Example: Check RSI value, Moving Average crossovers etc.
                    # indicators = analysis.indicators
                    # rsi = indicators.get('RSI')
                    # ma_50 = indicators.get('MA50')
                    # ma_200 = indicators.get('MA200')

                    # if rsi is not None and ma_50 is not None and ma_200 is not None:
                    #     # Add more conditions here based on RSI thresholds, MA crossovers, etc.
                    #     pass # Add logic here

                    # If the stock meets the criteria, add it to the breakout candidates list
                    breakout_candidates_b.append({
                        'ticker': handler.symbol,
                        'recommendation': technical_summary['RECOMMENDATION'],
                        'time_horizon': time_horizon_b,
                        # Add other relevant data from analysis if needed
                        # 'RSI': rsi,
                        # 'MA50': ma_50,
                        # 'MA200': ma_200,
                    })
                    print(f"Identified potential breakout candidate: {handler.symbol} ({technical_summary['RECOMMENDATION']})")


                # Add a small delay to avoid hitting API rate limits
                await asyncio.sleep(1) # Sleep for 1 second between ticker analysis

            except Exception as e:
                print(f"Error analyzing {handler.symbol} for breakout signals: {e}")
                # import traceback
                # traceback.print_exc() # Uncomment for detailed error during debugging

        # Add a larger delay between batches if needed
        await asyncio.sleep(5) # Sleep for 5 seconds between batches


    print(f"\n--- Breakout Assessment (Code B) Results ({time_horizon_b}) ---")
    if interaction:
         print(f"\n--- Breakout Assessment (Code B) Results ({time_horizon_b}) ---")

    if breakout_candidates_b:
        # Sort candidates (optional, e.g., by recommendation strength)
        # breakout_candidates_b.sort(key=lambda x: x['recommendation'], reverse=True) # If you assign numerical values to recommendation

        # Format and display the results
        breakout_df_b = pd.DataFrame(breakout_candidates_b)
        print(tabulate(breakout_df_b, headers='keys', tablefmt='psql'))
        if interaction:
             table_b = tabulate(breakout_df_b, headers='keys', tablefmt='psql')
             if len(table_b) > 1900:
                lines = table_b.split('\n')
                chunk_size = 40
                for i in range(0, len(lines), chunk_size):
                    chunk = '\n'.join(lines[i:i+chunk_size])
                    print(f"```psql\n{chunk}\n```")
             else:
                print(f"```psql\n{table_b}\n```")

        # Auto-Save Breakout Data
        save_breakout_data(breakout_df_b)

    else:
        no_candidates_message = f"No potential breakout candidates found based on the criteria for {time_horizon_b}."
        print(no_candidates_message)
        if interaction:
             print(no_candidates_message)

    end_time = time.time()
    print(f"Breakout Assessment (Code B) finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken for Breakout Assessment (Code B): {end_time - start_time:.2f} seconds.")
    print("--- Breakout Assessment (Code B) Complete. ---")

# --- END OF ASSESS_BREAKOUT (CODE B) ---


# --- START OF SAVE_BREAKOUT_DATA (CODE SB) ---

def save_breakout_data(breakout_df_b: pd.DataFrame):
    """
    Saves the breakout assessment results to a CSV file.
    """
    print("\n--- Starting Save Breakout Data (Code SB) ---")
    try:
        # Define the save directory and filename
        save_dir = "INVEST_Data/Breakout_Assessment"
        os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist

        # Generate a unique filename with a timestamp and a sens code (if applicable)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sens_code = 1785 # Example sens code for breakout save
        save_file = os.path.join(save_dir, f"breakout_sens_{sens_code}_{timestamp}.csv")

        # Save to CSV
        breakout_df_b.to_csv(save_file, index=False)

        print(f"--- Successfully saved breakout data to '{save_file}'. ---")

    except Exception as e:
        print(f"An error occurred during Save Breakout Data (Code SB): {e}")
        import traceback
        traceback.print_exc()
        print("--- Save Breakout Data (Code SB) failed. ---")

# --- END OF SAVE_BREAKOUT_DATA (CODE SB) ---


# --- START OF ASSESS_CULTIVATE (CODE E) ---

async def assess_cultivate(cultivate_code: str, epsilon: float, interaction: Optional[any] = None):
    """
    Cultivates a portfolio based on a specific strategy code and epsilon value.
    Identifies potential investment opportunities based on predefined criteria.
    """
    print("\n--- Starting Cultivate Assessment (Code E) ---")
    if interaction:
         print(f"Acknowledging interaction for Cultivate Assessment Code E with code '{cultivate_code}' and epsilon '{epsilon}'.")

    start_time = time.time()
    print(f"Cultivate Assessment (Code E) started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # --- Cultivate Strategy Logic ---
    # This is the core of the cultivate assessment. The logic here depends on the
    # `cultivate_code` and `epsilon` parameters. You need to define what each
    # `cultivate_code` means and how `epsilon` influences the strategy.

    # Example Strategy (Illustrative - replace with your actual cultivate strategies):
    # cultivate_code 'alpha': Identify stocks with high recent performance and strong fundamentals.
    # cultivate_code 'beta': Identify stocks with low volatility and consistent dividends.
    # cultivate_code 'gamma': Identify undervalued stocks based on P/E, P/B, etc., possibly using epsilon as a threshold.

    # For this example, let's implement a simple strategy for `cultivate_code = 'alpha'`
    # based on recent performance (using yfinance) and a basic fundamental check (e.g., positive EPS).
    # Epsilon could be used as a threshold for performance or another metric.

    print(f"Executing Cultivate Strategy for code: '{cultivate_code}' with epsilon: {epsilon}")
    if interaction:
         print(f"Executing Cultivate Strategy for code: '{cultivate_code}' with epsilon: {epsilon}")


    cultivate_candidates_e = []
    analysis_failed_count = 0

    # --- Get a list of potential tickers ---
    # Similar to Breakout Assessment, we need a source of tickers.
    # Using the predefined S&P 500 list again for consistency in this example.
    potential_tickers_e = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "BRK-B", "JPM", "V", "UNH", "XOM", "LLY", "JNJ", "V", "HD", "PG", "DIS", "NKE", "VZ"] # Extended list
    print(f"Using a predefined list of {len(potential_tickers_e)} potential tickers for Cultivate Assessment.")

    # Fetch data for potential tickers
    # We'll use yfinance for historical data (for recent performance) and potentially
    # some fundamental data if available via yfinance or other means.

    tickers_data = {}
    print("Fetching historical and fundamental data for tickers...")
    if interaction:
        print("Fetching historical and fundamental data for tickers...")

    # Define the period for historical data (e.g., last 3 months for recent performance)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90) # Last 3 months

    # Fetch data in batches to manage requests
    batch_size = 10
    for i in range(0, len(potential_tickers_e), batch_size):
        batch_tickers = potential_tickers_e[i:i + batch_size]
        print(f"Fetching data for batch {int(i/batch_size) + 1}/{(len(potential_tickers_e) + batch_size - 1) // batch_size}...")
        try:
            # Use the global YFINANCE_SESSION if available
            tickers_info = yf.download(
                batch_tickers,
                start=start_date,
                end=end_date,
                interval="1d", # Daily data
                progress=False,
                session=YFINANCE_SESSION if IS_CURL_CFFI_ACTIVE else None # Pass the session if active
            )
            if not tickers_info.empty:
                # Store data for each ticker
                for ticker in batch_tickers:
                    if ticker in tickers_info['Adj Close'].columns:
                        ticker_close_prices = tickers_info['Adj Close'][ticker].dropna()
                        if not ticker_close_prices.empty:
                            tickers_data[ticker] = {
                                'close_prices': ticker_close_prices,
                                # Fetch some basic fundamental data if available via yfinance Ticker object
                                # Note: This might require separate Ticker(ticker).info calls which can be slow.
                                # For a simple example, we'll skip detailed fundamentals here.
                            }
                            # Optional: Fetch info for basic fundamentals like EPS if needed
                            # try:
                            #     ticker_yf = yf.Ticker(ticker, session=YFINANCE_SESSION if IS_CURL_CFFI_ACTIVE else None)
                            #     ticker_info = ticker_yf.info
                            #     tickers_data[ticker]['info'] = ticker_info
                            # except Exception as e_info:
                            #     print(f"Could not fetch info for {ticker}: {e_info}")
                            #     tickers_data[ticker]['info'] = {}

                        else:
                             print(f"Warning: No valid historical data for {ticker} in Cultivate Assessment.")
                             analysis_failed_count += 1
                    else:
                         print(f"Warning: Ticker {ticker} not found in downloaded data columns.")
                         analysis_failed_count += 1

        except Exception as e:
            print(f"Error fetching data for batch starting with {batch_tickers[0]}: {e}")
            # Continue with the next batch
            analysis_failed_count += len(batch_tickers) # Assume all in batch failed to fetch

        # Add a delay between batches
        await asyncio.sleep(5) # Sleep for 5 seconds between batches


    print(f"Finished fetching data. Analyzing {len(tickers_data)} tickers for Cultivate Strategy.")
    if interaction:
         print(f"Finished fetching data. Analyzing {len(tickers_data)} tickers for Cultivate Strategy.")


    # --- Apply Cultivate Strategy based on cultivate_code ---

    if cultivate_code.lower() == 'alpha':
        print("Applying 'alpha' cultivate strategy (high recent performance, positive EPS).")
        # Epsilon can be used as a minimum recent performance threshold (percentage)

        for ticker, data in tickers_data.items():
            try:
                close_prices = data['close_prices']
                if len(close_prices) > 1:
                    # Calculate recent performance (e.g., percentage change over the period)
                    start_price = close_prices.iloc[0]
                    end_price = close_prices.iloc[-1]
                    recent_performance = ((end_price - start_price) / start_price) * 100 if start_price != 0 else 0.0

                    # Basic check for positive EPS (if info was fetched)
                    # For this example, we'll just use recent performance and epsilon
                    # has_positive_eps = data['info'].get('trailingEps', 0) > 0 if 'info' in data else False

                    # Apply epsilon as a minimum performance threshold
                    if recent_performance >= epsilon: # Check if performance meets or exceeds epsilon
                         cultivate_candidates_e.append({
                             'ticker': ticker,
                             'strategy': 'alpha',
                             'recent_performance (%)': recent_performance,
                             'epsilon_threshold (%)': epsilon
                             # Add other relevant data
                         })
                         print(f"Candidate found for 'alpha': {ticker} (Performance: {recent_performance:.2f}%)")
                    else:
                         print(f"Skipping {ticker} for 'alpha' (Performance: {recent_performance:.2f}%) - Below epsilon {epsilon}%.")


                else:
                    print(f"Warning: Insufficient historical data for {ticker} to calculate performance.")
                    analysis_failed_count += 1

            except Exception as e_ticker:
                print(f"Error processing {ticker} for 'alpha' strategy: {e_ticker}")
                analysis_failed_count += 1

    elif cultivate_code.lower() == 'beta':
        print("Applying 'beta' cultivate strategy (low volatility, consistent dividends).")
        # Epsilon could be used as a maximum volatility threshold or minimum dividend yield.

        for ticker, data in tickers_data.items():
             try:
                 # To calculate volatility, we need daily returns
                 close_prices = data['close_prices']
                 if len(close_prices) > 1:
                     returns = close_prices.pct_change().dropna()
                     if not returns.empty:
                         # Calculate historical volatility (annualized standard deviation of returns)
                         daily_volatility = returns.std()
                         annualized_volatility = daily_volatility * np.sqrt(252) # Assuming 252 trading days

                         # Check for consistent dividends (requires dividend data which is not reliably in basic yfinance info)
                         # For simplicity, let's just use volatility and epsilon as max volatility threshold.
                         # has_consistent_dividends = ... # Logic to check dividend history

                         if annualized_volatility <= epsilon: # Check if volatility is below or equal to epsilon
                             cultivate_candidates_e.append({
                                 'ticker': ticker,
                                 'strategy': 'beta',
                                 'annualized_volatility (%)': annualized_volatility * 100, # Display as percentage
                                 'epsilon_threshold (%)': epsilon * 100
                                 # Add dividend info if available
                             })
                             print(f"Candidate found for 'beta': {ticker} (Annualized Volatility: {annualized_volatility * 100:.2f}%)")
                         else:
                             print(f"Skipping {ticker} for 'beta' (Annualized Volatility: {annualized_volatility * 100:.2f}%) - Above epsilon {epsilon * 100:.2f}%.")

                     else:
                          print(f"Warning: Insufficient return data for {ticker} to calculate volatility.")
                          analysis_failed_count += 1

                 else:
                      print(f"Warning: Insufficient historical data for {ticker} for volatility calculation.")
                      analysis_failed_count += 1

             except Exception as e_ticker:
                 print(f"Error processing {ticker} for 'beta' strategy: {e_ticker}")
                 analysis_failed_count += 1


    elif cultivate_code.lower() == 'gamma':
        print("Applying 'gamma' cultivate strategy (undervalued based on ratios).")
        # Epsilon could be used as a threshold for P/E, P/B, etc.

        # For this strategy, we need fundamental data which is harder to get reliably
        # and consistently for many tickers using only yfinance.
        # This part is illustrative and would require a more robust fundamental data source.
        print("Note: 'gamma' strategy requires fundamental data which is not fully implemented in this example.")
        print("Skipping 'gamma' strategy execution.")
        analysis_failed_count += len(potential_tickers_e) # Assume all fail for gamma


    else:
        error_message = f"Unknown cultivate code: '{cultivate_code}'. Available codes: alpha, beta, gamma (gamma not fully implemented)."
        print(error_message)
        if interaction:
            print(error_message)
        print("--- Cultivate Assessment (Code E) failed due to unknown cultivate code. ---")
        return # Exit the function for unknown code


    print(f"\n--- Cultivate Assessment (Code E) Results ({cultivate_code.upper()}, Epsilon: {epsilon}) ---")
    if interaction:
         print(f"\n--- Cultivate Assessment (Code E) Results ({cultivate_code.upper()}, Epsilon: {epsilon}) ---")


    if cultivate_candidates_e:
        # Sort candidates (optional, e.g., by performance for alpha, volatility for beta)
        if cultivate_code.lower() == 'alpha':
            cultivate_candidates_e.sort(key=lambda x: x.get('recent_performance (%)', 0), reverse=True)
        elif cultivate_code.lower() == 'beta':
             cultivate_candidates_e.sort(key=lambda x: x.get('annualized_volatility (%)', float('inf'))) # Sort by volatility ascending

        # Format and display the results
        cultivate_df_e = pd.DataFrame(cultivate_candidates_e)

        # Select and rename columns for display based on strategy
        if cultivate_code.lower() == 'alpha':
             display_cols_e = ['ticker', 'strategy', 'recent_performance (%)', 'epsilon_threshold (%)']
             rename_map_e = {
                 'ticker': 'Ticker',
                 'strategy': 'Strategy',
                 'recent_performance (%)': f'Recent Performance (%) (>{epsilon:.2f}%)',
                 'epsilon_threshold (%)': 'Epsilon (%)'
             }
        elif cultivate_code.lower() == 'beta':
             display_cols_e = ['ticker', 'strategy', 'annualized_volatility (%)', 'epsilon_threshold (%)']
             rename_map_e = {
                 'ticker': 'Ticker',
                 'strategy': 'Strategy',
                 'annualized_volatility (%)': f'Annualized Volatility (%) (<={epsilon*100:.2f}%)', # Display epsilon * 100
                 'epsilon_threshold (%)': 'Epsilon (%)'
             }
        else: # Default or for 'gamma' if implemented
             display_cols_e = ['ticker', 'strategy']
             rename_map_e = {'ticker': 'Ticker', 'strategy': 'Strategy'}


        display_df_e = cultivate_df_e[display_cols_e].rename(columns=rename_map_e)

        # Format numerical columns
        for col in display_df_e.columns:
            if 'Performance (%)' in col or 'Volatility (%)' in col or 'Epsilon (%)' in col:
                 display_df_e[col] = display_df_e[col].map(lambda x: f'{x:.2f}' if pd.notnull(x) else 'N/A')


        print(tabulate(display_df_e, headers='keys', tablefmt='psql'))
        if interaction:
             table_e = tabulate(display_df_e, headers='keys', tablefmt='psql')
             if len(table_e) > 1900:
                lines = table_e.split('\n')
                chunk_size = 40
                for i in range(0, len(lines), chunk_size):
                    chunk = '\n'.join(lines[i:i+chunk_size])
                    print(f"```psql\n{chunk}\n```")
             else:
                print(f"```psql\n{table_e}\n```")

        # Auto-Save Cultivate Data (simplified save similar to market top 10)
        # Save Ticker, Raw INVEST Score (e.g., performance or volatility value), Combined Allocation (placeholder 0 for now)
        # The 'Raw INVEST Score' could be the performance (%) for 'alpha' or volatility (%) for 'beta'.
        # Let's create a 'raw_invest_score' column for saving.
        if cultivate_code.lower() == 'alpha':
            cultivate_df_e['raw_invest_score'] = cultivate_df_e['recent_performance (%)']
        elif cultivate_code.lower() == 'beta':
             cultivate_df_e['raw_invest_score'] = cultivate_df_e['annualized_volatility (%)']
             # For beta, lower volatility is better, so maybe inverse or negate for a 'score' where higher is better?
             # Or save the raw value and interpret later. Let's save the raw value.


        # Add a placeholder for 'combined_allocation_percent'
        cultivate_df_e['combined_allocation_percent'] = 0.0

        # Select columns for saving
        save_df_e = cultivate_df_e[['ticker', 'raw_invest_score', 'combined_allocation_percent']].copy()
        save_df_e.rename(columns={'ticker': 'TICKER', 'raw_invest_score': 'RAW_INVEST_SCORE', 'combined_allocation_percent': 'COMBINED_ALLOCATION_PERCENT'}, inplace=True)


        # Need to pass the epsilon value to the save function as an integer if needed for filename
        epsilon_int_for_save = int(epsilon * 100) if cultivate_code.lower() == 'alpha' else int(epsilon * 100) # Example: convert 0.05 to 5, 0.1 to 10

        save_cultivate_data(save_df_e, cultivate_code, epsilon_int_for_save)


    else:
        no_candidates_message = f"No potential cultivate candidates found based on code '{cultivate_code}' and epsilon {epsilon}."
        print(no_candidates_message)
        if interaction:
             print(no_candidates_message)

    if analysis_failed_count > 0:
         print(f"\nWarning: Analysis failed for {analysis_failed_count} tickers during Cultivate Assessment (Code E).")
         if interaction:
              print(f"\nWarning: Analysis failed for {analysis_failed_count} tickers during Cultivate Assessment (Code E).")


    end_time = time.time()
    print(f"Cultivate Assessment (Code E) finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time taken for Cultivate Assessment (Code E): {end_time - start_time:.2f} seconds.")
    print("--- Cultivate Assessment (Code E) Complete. ---")

# --- END OF ASSESS_CULTIVATE (CODE E) ---


# --- START OF SAVE_CULTIVATE_DATA (CODE SE) ---

def save_cultivate_data(cultivate_df_e: pd.DataFrame, cultivate_code: str, epsilon_int: int):
    """
    Saves the cultivate assessment results (simplified) to a CSV file.
    Intended for the auto-save feature. Saves Ticker, Raw Score, Allocation.
    """
    print("\n--- Starting Save Cultivate Data (Code SE) ---")
    try:
        # Define the save directory and filename
        save_dir = "INVEST_Data/Cultivate_Assessment"
        os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist

        # Generate a unique filename with a timestamp, cultivate code, and epsilon
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_file = os.path.join(save_dir, f"cultivate_{cultivate_code.lower()}_eps_{epsilon_int}_{timestamp}.csv")

        # Ensure the DataFrame has the expected columns (should be handled before calling)
        if 'TICKER' not in cultivate_df_e.columns or 'RAW_INVEST_SCORE' not in cultivate_df_e.columns or 'COMBINED_ALLOCATION_PERCENT' not in cultivate_df_e.columns:
            print("Error [Save Cultivate Data]: Input DataFrame missing required columns.")
            print("--- Save Cultivate Data (Code SE) failed. ---")
            return

        # Save to CSV
        cultivate_df_e.to_csv(save_file, index=False)

        print(f"--- Successfully saved cultivate data to '{save_file}'. ---")

    except Exception as e:
        print(f"An error occurred during Save Cultivate Data (Code SE): {e}")
        import traceback
        traceback.print_exc()
        print("--- Save Cultivate Data (Code SE) failed. ---")

# --- END OF SAVE_CULTIVATE_DATA (CODE SE) ---


# --- START OF LOAD_ASSESSMENT_DATA (CODE L) ---

def load_assessment_data(assessment_type: str, sens_code: Optional[int] = None) -> pd.DataFrame:
    """
    Loads the latest saved assessment data for a given type and optional sens code.
    Looks for the most recent file matching the pattern in the respective directory.
    Returns a pandas DataFrame or an empty DataFrame if no data is found.
    """
    print(f"\n--- Starting Load Assessment Data (Code L) for '{assessment_type}' (Sens Code: {sens_code}) ---")
    try:
        data_dir = f"INVEST_Data/{assessment_type.replace(' ', '_')}_Assessment"
        print(f"Looking for data in directory: {data_dir}")

        if not os.path.exists(data_dir):
            print(f"Data directory not found: {data_dir}")
            print(f"--- Load Assessment Data (Code L) failed: Directory not found. ---")
            return pd.DataFrame() # Return empty DataFrame if directory doesn't exist

        # Define the file pattern to look for
        # Patterns:
        # Custom Portfolio: custom_portfolio_sens_<sens_code>_<timestamp>.csv
        # Market Assessment (Top 10): market_top_ten_sens_<sens_code>_<timestamp>.csv
        # Market Assessment (Full): market_full_sens_<sens_code>_<timestamp>_<time_horizon>.csv (if implemented)
        # Breakout Assessment: breakout_sens_<sens_code>_<timestamp>.csv
        # Cultivate Assessment: cultivate_<cultivate_code>_eps_<epsilon>_<timestamp>.csv (Note: cultivate uses code and epsilon)

        file_pattern = None
        if assessment_type.lower() == 'custom portfolio':
            if sens_code is None:
                print("Error [Load Assessment Data]: 'sens_code' is required for 'Custom Portfolio' loading.")
                print("--- Load Assessment Data (Code L) failed. ---")
                return pd.DataFrame()
            file_pattern = f"custom_portfolio_sens_{sens_code}_*.csv"
        elif assessment_type.lower() == 'market assessment':
             # Prioritize loading 'market_top_ten' if sens_code is the default auto-save one (e.g., 3725)
             # Otherwise, if full save is implemented and needed, adjust logic here.
             # For now, let's load the latest market_top_ten if sens_code matches or is None (assuming we want the latest)
             if sens_code is None or sens_code == 3725: # Assuming 3725 is the default auto-save sens code
                 file_pattern = "market_top_ten_sens_*.csv"
                 print(f"Searching for market_top_ten files with pattern: {file_pattern}")
             # Add logic here for 'market_full' if implemented and needed
             # elif sens_code == 7357: # Example sens code for full save
             #     file_pattern = f"market_full_sens_{sens_code}_*.csv"

             if file_pattern is None:
                  print(f"Warning [Load Assessment Data]: No specific file pattern defined for Market Assessment with sens code {sens_code}. Searching for latest market_top_ten.")
                  file_pattern = "market_top_ten_sens_*.csv" # Default to searching for top ten

        elif assessment_type.lower() == 'breakout assessment':
             # Assuming breakout saves use a specific sens code (e.g., 1785) or just a pattern
             # If sens_code is provided, use it, otherwise look for any breakout file
             if sens_code is None:
                 file_pattern = "breakout_sens_*.csv"
                 print(f"Searching for any breakout files with pattern: {file_pattern}")
             else:
                 file_pattern = f"breakout_sens_{sens_code}_*.csv"
                 print(f"Searching for breakout files with sens code {sens_code} pattern: {file_pattern}")

        elif assessment_type.lower() == 'cultivate assessment':
             # Cultivate requires cultivate_code and epsilon, not a single sens_code in the filename
             # The sens_code parameter here might be used to filter cultivate files later if needed,
             # but the filename structure is based on cultivate_code and epsilon.
             # For loading, we might need cultivate_code and epsilon to find the right file,
             # or just load the latest cultivate file based on a general pattern.
             # Let's assume for loading cultivate data, we just look for the latest cultivate file
             # matching the pattern 'cultivate_*_*.csv'
             file_pattern = "cultivate_*_eps_*.csv" # Pattern for cultivate files
             print(f"Searching for cultivate files with pattern: {file_pattern}")

        else:
            print(f"Error [Load Assessment Data]: Unknown assessment type: '{assessment_type}'.")
            print("Supported types: 'Custom Portfolio', 'Market Assessment', 'Breakout Assessment', 'Cultivate Assessment'.")
            print("--- Load Assessment Data (Code L) failed. ---")
            return pd.DataFrame() # Return empty DataFrame for unknown type

        if file_pattern is None:
             print(f"Error [Load Assessment Data]: Could not determine file pattern for assessment type '{assessment_type}' and sens code {sens_code}.")
             print("--- Load Assessment Data (Code L) failed. ---")
             return pd.DataFrame()

        # Find all files matching the pattern
        import glob
        search_path = os.path.join(data_dir, file_pattern)
        list_of_files = glob.glob(search_path)

        if not list_of_files:
            print(f"No files found matching the pattern '{file_pattern}' in '{data_dir}'.")
            print(f"--- Load Assessment Data (Code L) failed: No matching files found. ---")
            return pd.DataFrame() # Return empty DataFrame if no files found

        # Find the most recent file based on modification time
        latest_file = max(list_of_files, key=os.path.getmtime)
        print(f"Found latest file: {latest_file}")

        # Load the data from the latest file
        loaded_df = pd.read_csv(latest_file)
        print(f"Successfully loaded data from '{latest_file}'. Shape: {loaded_df.shape}")

        # Optional: Filter loaded data based on sens_code if needed (especially for cultivate)
        # For cultivate, the sens_code parameter here might be used to select which cultivate file to load
        # based on the epsilon or other criteria encoded in the filename.
        # This requires parsing the filename, which can be added here if needed.
        # Example for cultivate: extract epsilon from filename and match with provided sens_code if sens_code is used for epsilon.
        # if assessment_type.lower() == 'cultivate assessment' and sens_code is not None:
        #     # Parse epsilon from filename and filter
        #     epsilon_from_filename = ... # Logic to extract epsilon from filename
        #     if epsilon_from_filename != sens_code: # Assuming sens_code is used for epsilon value here
        #         print(f"Warning: Loaded file '{latest_file}' has epsilon {epsilon_from_filename} which doesn't match provided sens_code (epsilon) {sens_code}.")
        #         # Decide whether to return empty or the loaded data with a warning


        print(f"--- Load Assessment Data (Code L) Complete. ---")
        return loaded_df

    except Exception as e:
        print(f"An error occurred during Load Assessment Data (Code L): {e}")
        import traceback
        traceback.print_exc()
        print("--- Load Assessment Data (Code L) failed. ---")
        return pd.DataFrame() # Return empty DataFrame on error

# --- END OF LOAD_ASSESSMENT_DATA (CODE L) ---


# --- START OF COMBINE_DATA (CODE CO) ---

def combine_data(cultivate_df: pd.DataFrame, market_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines data from Cultivate and Market assessments to produce a unified view
    with allocation suggestions.
    This is a core function for generating the final output for the user.
    """
    print("\n--- Starting Combine Data (Code CO) ---")
    try:
        # --- Data Combination and Allocation Logic ---
        # This is where you combine the insights from different assessments
        # and determine suggested allocations. This logic is highly customizable.

        # Example Combination Strategy:
        # - Start with candidates from the Cultivate Assessment.
        # - If a ticker is also present in the Market Assessment (Top N),
        #   potentially increase its allocation or confidence score.
        # - Use the 'raw_invest_score' from Cultivate and 'custom_invest_score'
        #   from Market Assessment as inputs for allocation.
        # - Consider other factors if available (e.g., risk assessment results,
        #   user preferences, current portfolio holdings).

        print("Combining Cultivate and Market data...")

        if cultivate_df.empty and market_df.empty:
            print("Warning [Combine Data]: Both Cultivate and Market DataFrames are empty. Cannot combine.")
            print("--- Combine Data (Code CO) failed. ---")
            return pd.DataFrame()

        # Ensure column names are consistent for merging/comparison
        # Assuming Cultivate DataFrame has 'TICKER' and 'RAW_INVEST_SCORE'
        # Assuming Market DataFrame has 'Ticker' and 'INVEST Score' ('custom_invest_score' before renaming)
        # Let's rename Market DataFrame columns to match for easier merging
        market_df_renamed = market_df.rename(columns={
            'Ticker': 'TICKER',
            'INVEST Score': 'MARKET_INVEST_SCORE' # Rename to avoid confusion with Cultivate score
        })

        # Select only necessary columns from market_df_renamed to avoid merging all columns
        market_df_slim = market_df_renamed[['TICKER', 'MARKET_INVEST_SCORE']].copy()


        # Merge Cultivate data with Market data based on Ticker
        # Use a left merge to keep all Cultivate candidates and add Market data where available
        combined_df = pd.merge(
            cultivate_df,
            market_df_slim,
            on='TICKER',
            how='left'
        )

        # Fill NaN values in 'MARKET_INVEST_SCORE' for tickers not in Market Assessment
        combined_df['MARKET_INVEST_SCORE'] = combined_df['MARKET_INVEST_SCORE'].fillna(0) # Assume a score of 0 if not in market top N

        print(f"Combined data shape: {combined_df.shape}")
        print("Calculating combined INVEST score and allocation...")

        # --- Calculate Combined INVEST Score and Allocation ---
        # This is the core allocation logic. Define how the scores are combined
        # and how allocations are determined.

        # Example Allocation Logic:
        # - Combine 'RAW_INVEST_SCORE' from Cultivate and 'MARKET_INVEST_SCORE'.
        # - Assign higher allocation to tickers with higher combined scores.
        # - The method of combination and allocation can vary greatly based on strategy.

        # Let's create a simple combined score by weighting the two scores.
        # Assuming 'RAW_INVEST_SCORE' from cultivate (e.g., performance or inverse volatility)
        # and 'MARKET_INVEST_SCORE' (TradingView Screener based score).

        # Need to normalize or scale the scores if their ranges are significantly different.
        # For simplicity here, let's assume they are somewhat comparable or can be weighted directly.
        # A more robust approach would involve scaling (e.g., min-max scaling, z-score).

        # Example: Simple Weighted Average
        cultivate_weight = 0.6 # Weight for Cultivate score
        market_weight = 0.4    # Weight for Market score

        # Ensure 'RAW_INVEST_SCORE' is treated appropriately based on cultivate strategy
        # For 'beta' (low volatility), a lower raw score is better. Need to adjust this for combination.
        # If saving raw volatility, a higher *inverse* volatility might be better for combining.
        # Let's assume 'RAW_INVEST_SCORE' as saved is such that higher is better for combination.
        # If 'beta' saves raw volatility, you might need to do: 1 / (combined_df['RAW_INVEST_SCORE'] + epsilon) * weight
        # For now, assuming 'RAW_INVEST_SCORE' is adjusted or naturally aligns (higher is better).

        # Calculate a 'Combined_Score'
        combined_df['Combined_Score'] = (
            combined_df['RAW_INVEST_SCORE'] * cultivate_weight +
            combined_df['MARKET_INVEST_SCORE'] * market_weight
        )

        # Sort by the Combined Score in descending order to prioritize
        combined_df = combined_df.sort_values(by='Combined_Score', ascending=False).reset_index(drop=True)

        # --- Determine Allocations ---
        # Allocate a percentage of the total portfolio value to each stock based on its combined score.
        # Methods: equal weighting, weighted by score, tiered allocation, etc.

        # Example: Simple Score-Based Allocation
        # Sum of all positive combined scores
        total_positive_score = combined_df['Combined_Score'][combined_df['Combined_Score'] > 0].sum()

        # Calculate allocation percentage for each ticker
        if total_positive_score > 0:
            combined_df['SUGGESTED_ALLOCATION (%)'] = (combined_df['Combined_Score'] / total_positive_score) * 100
            # Ensure allocation is 0 for negative or zero scores
            combined_df.loc[combined_df['Combined_Score'] <= 0, 'SUGGESTED_ALLOCATION (%)'] = 0.0
        else:
            combined_df['SUGGESTED_ALLOCATION (%)'] = 0.0 # No allocation if total score is zero or negative

        # Cap or adjust allocations if needed (e.g., max allocation per stock)
        max_single_allocation = 20.0 # Example: Max 20% in a single stock
        combined_df['SUGGESTED_ALLOCATION (%)'] = combined_df['SUGGESTED_ALLOCATION (%)'].clip(upper=max_single_allocation)

        # Re-normalize allocations so they sum to 100% (or less if some are capped)
        total_allocated = combined_df['SUGGESTED_ALLOCATION (%)'].sum()
        if total_allocated > 0:
             combined_df['SUGGESTED_ALLOCATION (%)'] = (combined_df['SUGGESTED_ALLOCATION (%)'] / total_allocated) * 100

        # Ensure 'COMBINED_ALLOCATION_PERCENT' is updated with the calculated allocation
        combined_df['COMBINED_ALLOCATION_PERCENT'] = combined_df['SUGGESTED_ALLOCATION (%)']


        print("\n--- Combined Assessment and Suggested Allocation Results ---")

        if not combined_df.empty:
            # Select and format columns for display
            display_cols_co = [
                'TICKER',
                'RAW_INVEST_SCORE', # Score from Cultivate (performance or volatility)
                'MARKET_INVEST_SCORE', # Score from Market Assessment
                'Combined_Score',
                'SUGGESTED_ALLOCATION (%)'
            ]
            display_df_co = combined_df[display_cols_co].copy()

            # Format numerical columns
            for col in ['RAW_INVEST_SCORE', 'MARKET_INVEST_SCORE', 'Combined_Score', 'SUGGESTED_ALLOCATION (%)']:
                if col in display_df_co.columns:
                    display_df_co[col] = display_df_co[col].map(lambda x: f'{x:.2f}' if pd.notnull(x) else 'N/A')

            print(tabulate(display_df_co, headers='keys', tablefmt='psql'))
            # Note: For terminal output, splitting might be needed if the table is very wide/long.
            # If this function is called by a command handler, that handler might manage printing chunks.


        else:
            print("No combined candidates found.")

        print("--- Combine Data (Code CO) Complete. ---")
        return combined_df # Return the DataFrame with calculated allocations

    except Exception as e:
        print(f"An error occurred during Combine Data (Code CO): {e}")
        import traceback
        traceback.print_exc()
        print("--- Combine Data (Code CO) failed. ---")
        return pd.DataFrame() # Return empty DataFrame on error

# --- END OF COMBINE_DATA (CODE CO) ---


# --- START OF VISUALIZE_DATA (CODE V) ---

# This function is commented out as matplotlib is not fully set up for interactive
# terminal use and saving plots would require more configuration.
# If needed, uncomment and adapt for saving plots to files.

# def visualize_data(data_df: pd.DataFrame, visualization_type: str):
#     """
#     Creates visualizations from assessment data.
#     (Requires matplotlib or similar library setup)
#     """
#     print(f"\n--- Starting Visualize Data (Code V) for '{visualization_type}' ---")
#     try:
#         if data_df.empty:
#             print("Error [Visualize Data]: Input DataFrame is empty. Cannot visualize.")
#             print("--- Visualize Data (Code V) failed. ---")
#             return

#         if visualization_type.lower() == 'allocation_pie':
#             print("Creating Allocation Pie Chart...")
#             # Requires a DataFrame with 'TICKER' and 'SUGGESTED_ALLOCATION (%)'
#             if 'TICKER' in data_df.columns and 'SUGGESTED_ALLOCATION (%)' in data_df.columns:
#                 # Filter out 0% allocations for the pie chart
#                 allocations_df = data_df[data_df['SUGGESTED_ALLOCATION (%)'] > 0].copy()

#                 if not allocations_df.empty:
#                     plt.figure(figsize=(10, 10))
#                     plt.pie(allocations_df['SUGGESTED_ALLOCATION (%)'], labels=allocations_df['TICKER'], autopct='%1.1f%%', startangle=140)
#                     plt.title('Suggested Portfolio Allocation')
#                     plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
#                     plt.tight_layout()

#                     # Save the plot to a file instead of showing
#                     save_dir = "INVEST_Data/Visualizations"
#                     os.makedirs(save_dir, exist_ok=True)
#                     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                     save_file = os.path.join(save_dir, f"allocation_pie_{timestamp}.png")
#                     plt.savefig(save_file)
#                     print(f"--- Saved Allocation Pie Chart to '{save_file}'. ---")
#                     plt.close() # Close the plot to free up memory

#                 else:
#                     print("No positive allocations to visualize.")
#             else:
#                 print("Error [Visualize Data]: Input DataFrame missing 'TICKER' or 'SUGGESTED_ALLOCATION (%)' columns for pie chart.")

#         elif visualization_type.lower() == 'market_score_bar':
#              print("Creating Market Score Bar Chart...")
#              # Requires a DataFrame with 'TICKER' and 'MARKET_INVEST_SCORE'
#              if 'TICKER' in data_df.columns and 'MARKET_INVEST_SCORE' in data_df.columns:
#                  # Sort by market score for better visualization
#                  market_scores_df = data_df.sort_values(by='MARKET_INVEST_SCORE', ascending=False).copy()

#                  if not market_scores_df.empty:
#                      plt.figure(figsize=(12, 6))
#                      plt.bar(market_scores_df['TICKER'], market_scores_df['MARKET_INVEST_SCORE'])
#                      plt.xlabel('Ticker')
#                      plt.ylabel('Market INVEST Score')
#                      plt.title('Market Assessment INVEST Scores')
#                      plt.xticks(rotation=45, ha='right')
#                      plt.tight_layout()

#                      # Save the plot to a file
#                      save_dir = "INVEST_Data/Visualizations"
#                      os.makedirs(save_dir, exist_ok=True)
#                      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#                      save_file = os.path.join(save_dir, f"market_score_bar_{timestamp}.png")
#                      plt.savefig(save_file)
#                      print(f"--- Saved Market Score Bar Chart to '{save_file}'. ---")
#                      plt.close() # Close the plot

#                  else:
#                      print("No market scores to visualize.")
#              else:
#                 print("Error [Visualize Data]: Input DataFrame missing 'TICKER' or 'MARKET_INVEST_SCORE' columns for bar chart.")


#         # Add more visualization types as needed (e.g., performance charts, volatility plots)

#         print("--- Visualize Data (Code V) Complete. ---")

#     except Exception as e:
#         print(f"An error occurred during Visualize Data (Code V): {e}")
#         import traceback
#         traceback.print_exc()
#         print("--- Visualize Data (Code V) failed. ---")

# --- END OF VISUALIZE_DATA (CODE V) ---


# --- START OF SAVE_COMBINED_DATA (CODE SC) ---

def save_combined_data(combined_df_co: pd.DataFrame, sens_code: Optional[int] = None):
    """
    Saves the combined assessment data with allocations to a CSV file.
    """
    print("\n--- Starting Save Combined Data (Code SC) ---")
    try:
        # Define the save directory and filename
        save_dir = "INVEST_Data/Combined_Assessment"
        os.makedirs(save_dir, exist_ok=True) # Create directory if it doesn't exist

        # Generate a unique filename with a timestamp and a sens code
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Use the provided sens_code or a default if none is given
        file_sens_code = sens_code if sens_code is not None else 9999 # Default sens code for combined data
        save_file = os.path.join(save_dir, f"combined_sens_{file_sens_code}_{timestamp}.csv")

        # Ensure the DataFrame has the expected columns (should be handled before calling)
        if combined_df_co.empty:
            print("Warning [Save Combined Data]: Input DataFrame is empty. Nothing to save.")
            print("--- Save Combined Data (Code SC) finished (nothing saved). ---")
            return

        # Select relevant columns for saving
        save_cols = ['TICKER', 'RAW_INVEST_SCORE', 'MARKET_INVEST_SCORE', 'Combined_Score', 'SUGGESTED_ALLOCATION (%)', 'COMBINED_ALLOCATION_PERCENT']
        # Ensure selected columns exist in the DataFrame before saving
        save_df_co = combined_df_co[[col for col in save_cols if col in combined_df_co.columns]].copy()


        # Save to CSV
        save_df_co.to_csv(save_file, index=False)

        print(f"--- Successfully saved combined data to '{save_file}'. ---")

    except Exception as e:
        print(f"An error occurred during Save Combined Data (Code SC): {e}")
        import traceback
        traceback.print_exc()
        print("--- Save Combined Data (Code SC) failed. ---")

# --- END OF SAVE_COMBINED_DATA (CODE SC) ---


# --- START OF EXECUTE_TRADE (CODE T) ---

# This is a placeholder function for trade execution.
# IMPLEMENT WITH CAUTION AND THOROUGH TESTING.
# Requires integration with a brokerage API (e.g., Alpaca, Interactive Brokers, etc.)
# Ensure proper error handling, order sizing, risk management, and confirmation.
# This function is NOT functional as provided and needs significant development.

# async def execute_trade(ticker: str, quantity: float, order_type: str, limit_price: Optional[float] = None, interaction: Optional[Any] = None):
#     """
#     Executes a trade for a given ticker and quantity.
#     (Placeholder - REQUIRES BROKERAGE API INTEGRATION)
#     """
#     print(f"\n--- Starting Trade Execution (Code T) for {ticker} ---")
#     if interaction:
#          print(f"Acknowledging interaction for Trade Execution Code T for {ticker}.")

#     print("!!! WARNING: TRADE EXECUTION FUNCTION IS A PLACEHOLDER !!!")
#     print("!!! It is NOT connected to a brokerage and will NOT execute real trades. !!!")
#     print("!!! Implement with EXTREME CAUTION and THOROUGH TESTING if connecting to a live account. !!!")

#     try:
#         # --- BROKERAGE API INTEGRATION REQUIRED HERE ---
#         # Example (pseudo-code):
#         # if brokerage_client is None:
#         #     print("Error: Brokerage client not initialized.")
#         #     return

#         # try:
#         #     order = brokerage_client.submit_order(
#         #         symbol=ticker,
#         #         qty=quantity,
#         #         side='buy' if order_type.lower() == 'buy' else 'sell',
#         #         type='limit' if order_type.lower() == 'limit' else 'market',
#         #         limit_price=limit_price,
#         #         time_in_force='gtc' # Good 'Til Cancelled
#         #     )
#         #     print(f"Pseudo-code: Submitted {order_type.upper()} order for {quantity} shares of {ticker}.")
#         #     print(f"Pseudo-code: Order ID: {order.id}, Status: {order.status}")
#         #     if interaction:
#         #          print(f"Pseudo-code: Submitted {order_type.upper()} order for {quantity} shares of {ticker}.")
#         #          print(f"Pseudo-code: Order ID: {order.id}, Status: {order.status}")

#         # except Exception as e_trade:
#         #     print(f"Pseudo-code Error submitting trade for {ticker}: {e_trade}")
#         #     if interaction:
#         #          print(f"Pseudo-code Error submitting trade for {ticker}: {e_trade}")
#         #     print("--- Trade Execution (Code T) failed. ---")
#         #     return

#         print(f"Pseudo-code: Simulated {order_type.upper()} trade for {quantity} shares of {ticker}.")
#         if limit_price is not None:
#              print(f"Pseudo-code: Limit Price: {limit_price}")
#         print("!!! Simulated Trade Execution Complete (No real trade occurred). !!!")
#         if interaction:
#             print(f"Pseudo-code: Simulated {order_type.upper()} trade for {quantity} shares of {ticker}.")
#             if limit_price is not None:
#                 print(f"Pseudo-code: Limit Price: {limit_price}")
#             print("!!! Simulated Trade Execution Complete (No real trade occurred). !!!")


#     except Exception as e:
#         print(f"An error occurred during Simulated Trade Execution (Code T): {e}")
#         import traceback
#         traceback.print_exc()
#         if interaction:
#             print(f"An error occurred during Simulated Trade Execution (Code T): {e}")
#         print("--- Simulated Trade Execution (Code T) failed. ---")

# --- END OF EXECUTE_TRADE (CODE T) ---


# --- START OF PORTFOLIO_STATUS (CODE P) ---

# This is a placeholder function for checking portfolio status.
# Requires integration with a brokerage API to get real-time holdings and balances.

# async def portfolio_status(interaction: Optional[Any] = None):
#     """
#     Retrieves and displays the current portfolio status (holdings, value, etc.).
#     (Placeholder - REQUIRES BROKERAGE API INTEGRATION)
#     """
#     print("\n--- Starting Portfolio Status Check (Code P) ---")
#     if interaction:
#          print(f"Acknowledging interaction for Portfolio Status Code P.")

#     print("!!! WARNING: PORTFOLIO STATUS FUNCTION IS A PLACEHOLDER !!!")
#     print("!!! It is NOT connected to a brokerage and will NOT show real portfolio data. !!!")

#     try:
#         # --- BROKERAGE API INTEGRATION REQUIRED HERE ---
#         # Example (pseudo-code):
#         # if brokerage_client is None:
#         #     print("Error: Brokerage client not initialized.")
#         #     return

#         # try:
#         #     account = brokerage_client.get_account()
#         #     holdings = brokerage_client.list_positions()

#         #     print("\n--- Pseudo-code Portfolio Status ---")
#         #     print(f"Account Value: ${float(account.equity):,.2f}")
#         #     print(f"Cash Balance: ${float(account.cash):,.2f}")
#         #     print("\nHoldings:")
#         #     if holdings:
#         #         for position in holdings:
#         #             print(f"- {position.symbol}: {position.qty} shares @ ${position.avg_entry_price} (Current Value: ${float(position.market_value):,.2f})")
#         #     else:
#         #         print("No current holdings.")
#         #     print("--- Pseudo-code Portfolio Status Complete ---")

#         #     if interaction:
#         #          status_message = f"--- Pseudo-code Portfolio Status ---\n"
#         #          status_message += f"Account Value: ${float(account.equity):,.2f}\n"
#         #          status_message += ff"Cash Balance: ${float(account.cash):,.2f}\n"
#         #          status_message += "\nHoldings:\n"
#         #          if holdings:
#         #              for position in holdings:
#         #                  status_message += f"- {position.symbol}: {position.qty} shares @ ${position.avg_entry_price} (Current Value: ${float(position.market_value):,.2f})\n"
#         #          else:
#         #              status_message += "No current holdings.\n"
#         #          status_message += "--- Pseudo-code Portfolio Status Complete ---"
#         #          print(status_message)

#         # except Exception as e_portfolio:
#         #     print(f"Pseudo-code Error retrieving portfolio status: {e_portfolio}")
#         #     if interaction:
#         #         print(f"Pseudo-code Error retrieving portfolio status: {e_portfolio}")
#         #     print("--- Pseudo-code Portfolio Status Check (Code P) failed. ---")
#         #     return

#         print("\n--- Simulated Portfolio Status ---")
#         print("Account Value: $100,000.00")
#         print("Cash Balance: $50,000.00")
#         print("\nHoldings:")
#         print("- AAPL: 100 shares @ $170.00 (Current Value: $17,500.00)")
#         print("- MSFT: 50 shares @ $400.00 (Current Value: $20,200.00)")
#         print("- GOOGL: 20 shares @ $150.00 (Current Value: $3,100.00)")
#         print("--- Simulated Portfolio Status Complete ---")
#         if interaction:
#              status_message = f"--- Simulated Portfolio Status ---\n"
#              status_message += f"Account Value: $100,000.00\n"
#              status_message += f"Cash Balance: $50,000.00\n"
#              status_message += "\nHoldings:\n"
#              status_message += f"- AAPL: 100 shares @ $170.00 (Current Value: $17,500.00)\n"
#              status_message += f"- MSFT: 50 shares @ $400.00 (Current Value: $20,200.00)\n"
#              status_message += f"- GOOGL: 20 shares @ $150.00 (Current Value: $3,100.00)\n"
#              status_message += "--- Simulated Portfolio Status Complete ---"
#              print(status_message)


#     except Exception as e:
#         print(f"An error occurred during Simulated Portfolio Status Check (Code P): {e}")
#         import traceback
#         traceback.print_exc()
#         if interaction:
#             print(f"An error occurred during Simulated Portfolio Status Check (Code P): {e}")
#         print("--- Simulated Portfolio Status Check (Code P) failed. ---")

# --- END OF PORTFOLIO_STATUS (CODE P) ---


# --- START OF AUTO_SAVE_TASK (CODE AS) ---

# This is the background task for auto-saving assessment results.
# It will run periodically (e.g., every 5 minutes) and trigger saves
# if the current time is within the auto-save window (e.g., 17:00-18:00 EST).

# Note: This function needs to be integrated into the main application loop
# using something like asyncio or a dedicated task scheduler.
# For a simple script, this might run in a loop or be triggered by a system scheduler.

# Define the auto-save window (EST)
auto_save_start_hour = 17 # 5 PM
auto_save_end_hour = 18   # 6 PM
auto_save_check_interval_minutes = 5 # Check every 5 minutes

# Keep track of the last date auto-save was performed to avoid multiple saves on the same day
last_auto_save_date: Optional[date] = None

async def auto_save_task():
    """
    Background task to automatically save assessment data at a scheduled time.
    """
    global last_auto_save_date
    print("\n--- Auto-Save Task (Code AS) Started ---")

    while True:
        try:
            # Get the current time in EST
            est = pytz.timezone('US/Eastern')
            now_est = datetime.now(est)
            current_time = now_est.time()
            current_date = now_est.date()

            print(f"Auto-Save Task: Checking time at {now_est.strftime('%Y-%m-%d %H:%M:%S %Z%z')}")

            # Define the auto-save time window
            # Check if current time is within the auto-save window (e.g., 17:00 to 18:00 EST)
            # and if auto-save hasn't been performed today yet.
            if (current_time >= dt_time(auto_save_start_hour, 0) and
                current_time < dt_time(auto_save_end_hour, 0) and
                last_auto_save_date != current_date):

                print(f"Auto-Save window open and not saved today. Initiating auto-save...")

                # --- Perform Auto-Save in the specified order ---
                # Order: Breakout -> Custom -> Market (Top 10) -> Cultivate

                print("Auto-Save: Running Breakout Assessment (Code B) for auto-save...")
                try:
                    # Run Breakout Assessment - specify time horizon (e.g., '1d' for daily breakout)
                    # The assess_breakout function should handle saving the results.
                    await assess_breakout(time_horizon_b='1d') # Example: Use daily interval for auto-save breakout
                    print("Auto-Save: Breakout Assessment complete.")
                except Exception as e_auto_b:
                    print(f"Auto-Save Error: Breakout Assessment failed: {e_auto_b}")
                    import traceback
                    traceback.print_exc()


                print("Auto-Save: Running Custom Portfolio Risk Assessment (Code C) for auto-save...")
                try:
                    # Run Custom Portfolio Risk Assessment - requires portfolio holdings.
                    # This assumes you have a way to load or access the current portfolio holdings for the auto-save.
                    # If auto-save should include a custom portfolio risk assessment, you need to define
                    # how the 'tailored_holdings_data' is obtained in this automated context.
                    # For now, let's skip this in auto-save or use a predefined/loaded portfolio if available.
                    # await assess_custom_portfolio(tailored_holdings_data=...) # Requires holdings data
                     print("Auto-Save: Skipping Custom Portfolio Risk Assessment in auto-save (requires specific holdings).")
                except Exception as e_auto_c:
                    print(f"Auto-Save Error: Custom Portfolio Risk Assessment failed: {e_auto_c}")
                    import traceback
                    traceback.print_exc()


                print("Auto-Save: Running Market Assessment (Code M) for auto-save...")
                try:
                    # Run Market Assessment - specify time horizon (e.g., '1d' for daily market scan)
                    # The assess_market function is modified to save only the top 10 in auto-save mode.
                    await assess_market(time_horizon_m='1d') # Example: Use daily interval for auto-save market
                    print("Auto-Save: Market Assessment complete.")
                except Exception as e_auto_m:
                    print(f"Auto-Save Error: Market Assessment failed: {e_auto_m}")
                    import traceback
                    traceback.print_exc()

                print("Auto-Save: Running Cultivate Assessment (Code E) for auto-save...")
                try:
                    # Run Cultivate Assessment - requires cultivate_code and epsilon.
                    # Define default auto-save parameters for cultivate.
                    auto_save_cultivate_code = 'alpha' # Example default code
                    auto_save_epsilon = 0.05          # Example default epsilon (5% performance)
                    await assess_cultivate(cultivate_code=auto_save_cultivate_code, epsilon=auto_save_epsilon)
                    print("Auto-Save: Cultivate Assessment complete.")
                except Exception as e_auto_e:
                    print(f"Auto-Save Error: Cultivate Assessment failed: {e_auto_e}")
                    import traceback
                    traceback.print_exc()


                # Mark auto-save as done for today
                last_auto_save_date = current_date
                print(f"Auto-save completed for {current_date}.")

            else:
                # print("Auto-Save Task: Outside save window or already saved today.") # Keep this quiet usually

                 # If it's after the save window (e.g., after 18:00 EST), reset last_auto_save_date for the next day
                 if current_time >= dt_time(auto_save_end_hour, 0) and last_auto_save_date == current_date:
                     print(f"Auto-Save Task: Resetting auto-save flag for next day.")
                     last_auto_save_date = None


        except Exception as e:
            print(f"An unexpected error occurred in Auto-Save Task: {e}")
            import traceback
            traceback.print_exc()

        # Wait for the next check interval
        await asyncio.sleep(auto_save_check_interval_minutes * 60) # Convert minutes to seconds

# --- END OF AUTO_SAVE_TASK (CODE AS) ---


# --- START OF MAIN EXECUTION LOOP / COMMAND HANDLING ---

# This part simulates the main loop that would listen for commands in a terminal environment.
# In a real application, this would be replaced by the logic that reads user input
# from the terminal, parses commands, and calls the appropriate async functions.

async def main_terminal_loop():
    print("\n--- M.I.C. Singularity Terminal Interface ---")
    print("Enter commands (e.g., assess custom, assess market 1d, assess breakout 4h, assess cultivate alpha 0.05, load market, combine, save combined 1234):")
    print("Type 'exit' to quit.")

    # Initialize variables to store the latest loaded/combined data
    latest_cultivate_df: pd.DataFrame = pd.DataFrame()
    latest_market_df: pd.DataFrame = pd.DataFrame()
    latest_combined_df: pd.DataFrame = pd.DataFrame()

    # Start the auto-save task in the background
    # asyncio.create_task(auto_save_task()) # Uncomment to enable auto-save in this loop

    while True:
        command_input = await asyncio.to_thread(input, "> ") # Use to_thread for blocking input

        if command_input.lower() == 'exit':
            print("Exiting M.I.C. Singularity.")
            break

        try:
            # Basic command parsing (split command and arguments)
            parts = command_input.strip().split()
            if not parts:
                continue # Ignore empty input

            command = parts[0].lower()
            args = parts[1:]

            if command == 'assess':
                if len(args) >= 1:
                    assessment_type = args[0].lower()
                    if assessment_type == 'custom':
                        # Assess Custom Portfolio command - requires user input for holdings
                        print("Enter custom portfolio holdings in the format: TICKER:VALUE, TICKER:VALUE (e.g., AAPL:1000, MSFT:500)")
                        holdings_input = await asyncio.to_thread(input, "Holdings: ")
                        try:
                            tailored_holdings_data = []
                            holding_entries = holdings_input.strip().split(',')
                            for entry in holding_entries:
                                if ':' in entry:
                                    ticker, value_str = entry.split(':')
                                    try:
                                        value = float(value_str.strip())
                                        if value > 0:
                                             tailored_holdings_data.append({'ticker': ticker.strip().upper(), 'custom_value': value})
                                        else:
                                             print(f"Warning: Skipping holding '{entry}' with non-positive value.")
                                    except ValueError:
                                        print(f"Warning: Could not parse value for holding '{entry}'. Skipping.")
                                else:
                                    print(f"Warning: Invalid holding format '{entry}'. Skipping.")

                            if tailored_holdings_data:
                                await assess_custom_portfolio(tailored_holdings_data=tailored_holdings_data)
                            else:
                                print("No valid custom holdings entered.")

                        except Exception as e_input:
                             print(f"Error processing holdings input: {e_input}")
                             import traceback
                             traceback.print_exc()


                    elif assessment_type == 'market':
                        # Assess Market command - requires time horizon
                        if len(args) >= 2:
                            time_horizon_m = args[1]
                            await assess_market(time_horizon_m=time_horizon_m)
                        else:
                            print("Usage: assess market <time_horizon> (e.g., assess market 1d)")

                    elif assessment_type == 'breakout':
                        # Assess Breakout command - requires time horizon
                        if len(args) >= 2:
                            time_horizon_b = args[1]
                            await assess_breakout(time_horizon_b=time_horizon_b)
                        else:
                            print("Usage: assess breakout <time_horizon> (e.g., assess breakout 4h)")

                    elif assessment_type == 'cultivate':
                        # Assess Cultivate command - requires code and epsilon
                        if len(args) >= 3:
                            cultivate_code = args[1]
                            try:
                                epsilon = float(args[2])
                                await assess_cultivate(cultivate_code=cultivate_code, epsilon=epsilon)
                            except ValueError:
                                print("Error: Epsilon must be a number.")
                                print("Usage: assess cultivate <code> <epsilon> (e.g., assess cultivate alpha 0.05)")
                        else:
                            print("Usage: assess cultivate <code> <epsilon> (e.g., assess cultivate alpha 0.05)")

                    else:
                        print(f"Unknown assessment type: {assessment_type}")
                        print("Supported types: custom, market, breakout, cultivate")
                else:
                    print("Usage: assess <type> [args]")

            elif command == 'load':
                 if len(args) >= 1:
                     assessment_type_load = args[0].lower()
                     sens_code_load = None
                     if len(args) >= 2:
                         try:
                              sens_code_load = int(args[1])
                         except ValueError:
                              print("Error: Sens code must be an integer.")
                              continue

                     # Call the load function
                     loaded_data_df = load_assessment_data(assessment_type=assessment_type_load, sens_code=sens_code_load)
                     if not loaded_data_df.empty:
                         print(f"Successfully loaded {len(loaded_data_df)} rows of '{assessment_type_load}' data.")
                         # Store the loaded data in the respective latest_ variables
                         if assessment_type_load == 'cultivate assessment':
                              latest_cultivate_df = loaded_data_df
                              print("Loaded cultivate data stored.")
                         elif assessment_type_load == 'market assessment':
                              latest_market_df = loaded_data_df
                              print("Loaded market data stored.")
                         # Add other types as needed
                     else:
                          print(f"Could not load '{assessment_type_load}' data.")

                 else:
                      print("Usage: load <assessment_type> [sens_code]")
                      print("Supported types: 'Custom Portfolio', 'Market Assessment', 'Breakout Assessment', 'Cultivate Assessment'")

            elif command == 'combine':
                 # Command to combine the latest loaded Cultivate and Market data
                 # Use the latest loaded data stored in the respective variables

                 if not latest_cultivate_df.empty and not latest_market_df.empty:
                      print("Combining the latest loaded Cultivate and Market data...")
                      combined_results_df = combine_data(cultivate_df=latest_cultivate_df, market_df=latest_market_df)
                      if not combined_results_df.empty:
                           print("Combination complete.")
                           # Store the combined results for saving
                           latest_combined_df = combined_results_df
                           print("Combined data stored.")
                      else:
                           print("Combination resulted in empty data.")
                 else:
                     print("Cannot combine: Ensure both 'cultivate assessment' and 'market assessment' data are loaded first using the 'load' command.")


            elif command == 'save':
                 if len(args) >= 1:
                     save_type = args[0].lower()
                     sens_code_save = None
                     if len(args) >= 2:
                         try:
                             sens_code_save = int(args[1])
                         except ValueError:
                              print("Error: Sens code must be an integer.")
                              continue

                     if save_type == 'combined':
                         # Save the latest combined data
                         # Use the latest combined data stored in the variable
                         if not latest_combined_df.empty:
                              save_combined_data(combined_df_co=latest_combined_df, sens_code=sens_code_save)
                         else:
                             print("No combined data available to save. Run 'combine' first.")

                     # Add options to save other assessment types manually if needed
                     # elif save_type == 'market_full':
                     #     # Save full market data - requires having the full data available
                     #     pass # Implement if full market save is needed

                     else:
                          print(f"Unknown save type: {save_type}")
                          print("Supported types: combined")

                 else:
                      print("Usage: save <type> [sens_code]")
                      print("Supported types: combined")


            # Add other commands here (e.g., for trade execution, portfolio status - if implemented)
            # elif command == 'trade':
            #     # Requires implementation of execute_trade and parsing trade parameters
            #     print("Trade execution command is not yet functional.")

            # elif command == 'portfolio':
            #     # Requires implementation of portfolio_status
            #     print("Portfolio status command is not yet functional.")

            else:
                print(f"Unknown command: {command}")
                print("Supported commands: assess, load, combine, save, exit")


        except Exception as e_cmd:
            print(f"An error occurred while processing command '{command_input}': {e_cmd}")
            import traceback
            traceback.print_exc()

# --- END OF MAIN EXECUTION LOOP / COMMAND HANDLING ---

# --- MAIN APPLICATION ENTRY POINT ---

if __name__ == "__main__":
    # Check if running in an environment that supports asyncio.run
    if sys.version_info >= (3, 7):
        asyncio.run(main_terminal_loop())
    else:
        print("Python 3.7 or higher is required to run the asyncio terminal loop.")
        print("Please run commands manually or use a different execution method.")
        # Fallback for older Python versions (manual execution example)
        # print("\n--- Manual Execution Example ---")
        # # Example: Manually run assess_market
        # # loop = asyncio.get_event_loop()
        # # loop.run_until_complete(assess_market(time_horizon_m='1d'))
        # # loop.close()

# --- END OF MAIN APPLICATION ENTRY POINT ---