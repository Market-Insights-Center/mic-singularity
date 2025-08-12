import yfinance as yf
import pandas as pd
import math
from tabulate import tabulate
import os
import time # Added for pauses
import sys # Added for printing characters with delay
import matplotlib.pyplot as plt
import numpy as np
from tradingview_screener import Query, Column
from tradingview_ta import TA_Handler, Interval, Exchange
import csv
from datetime import datetime, timedelta, time as dt_time, date # Renamed time import to dt_time
import pytz
from typing import Optional # Keep Optional for type hinting, though not strictly needed for terminal input
import asyncio # Added for running async functions

# --- yfinance fix: Import requests from curl_cffi and create a session ---
try:
    from curl_cffi import requests
    # Create a global session to be used by all yfinance calls
    YFINANCE_SESSION = requests.Session(impersonate="chrome")
    print("Successfully imported curl_cffi.requests and created a session for yfinance.")
except ImportError:
    print("Warning: curl_cffi not found. yfinance calls might fail. Please install with: pip install curl_cffi")
    # Fallback to standard requests if curl_cffi is not available, though this won't have the impersonate feature
    import requests
    YFINANCE_SESSION = requests.Session()
except Exception as e:
    print(f"Error setting up curl_cffi session: {e}. Using standard requests session.")
    import requests # Fallback
    YFINANCE_SESSION = requests.Session()

# --- Terminal Startup Sequence ---

# ASCII art representation of the MIC logo (Eye within a diamond)
MIC_LOGO_ASCII = r"""                                                                                                            
                                                                         @%@@%                                                                        
                                                                        %@@@@@@@                                                                      
                                                                       %@@@@@@@@@                                                                     
                                                                      @@@@@@@@@@@%                                                                    
                                                                    #@@@@@@@@@@@@@@                                                                   
                                                                   %@@@@@@@@@@@@@@@%@                                                                 
                                                                  %@@@@@@@@@@@@@@@@@@#                                                                
                                                                *%@@@@@@@@@@@@@@@@@@@@%                                                               
                                                               %@@@@@@@@@@@@@@@@@@@@@@@@@                                                             
                                                              #@@@@@@@@@@@@@@@@@@@@@@@@@@#                                                            
                                                            @%@@@@@@@@@@@@@@@@@@@@@@@@@@@@#                                                           
                                                           =@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                                          
                                                          @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                                         
                                                         @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*                                                       
                                                       +@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%                                                      
                                                      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%@                                                    
                                                     %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@*                                                   
                                                   @%@@@@@@@@@@@@@@@@@@@@@%@@@@@@@@@@@@@@@@@@@@@@@@#                                                  
                                                  %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                                
                                                 %@@@@@@@@@@@@@@@@@@@@@@##   @#%@@@@@@@@@@@@@@@@@@@@@@#                                               
                                                #@@@@@@@@@@@@@@@@@@@#@           @%%@@@@@@@@@@@@@@@@@@@%                                              
                                              @@@@@@@@@@@@@@@@@@#@                    %@@@@@@@@@@@@@@@@@%                                             
                                             %@@@@@@@@@@@@@@#@                            *@@@@@@@@@@@@@@@@                                           
                                            %@@@@@@@@@@@%*         @%%@@@@@%@@@@%%##@         %@@@@@@@@@@@@%                                          
                                          @@@@@@@@@@%+        ##%@@@@@@@@@@@@@@@@@@@@@@%%@       @%@@@@@@@@@%                                         
                                         %@@@@@@@@       -%@@@@@@@@@@@%#       @*@@@@@@@@@@%%%       @#%@@@@@@@                                       
                                        @@@@%%@       *@@@@@@@@@@@@%               @#@@@@@@@@@@@%+       *%@@@@%                                      
                                      +@@#@       #%@@@@@@@@@@@@@%   %@%*            @#@@@@@@@@@@@@@*        ##@%                                     
                                      @        #%@@@@@@@%%@@@@@@#   %@@@@@%            %@@@@##%@@@@@@@@*@        @                                    
                                            *@@@@@@@@*@   @@@@@#   %@@@@@@@             %@@@@    *@@@@@@@@@@                                          
                                         +%@@@@@@@+       %@@@@     #@@@@%              @@@@@+      =@@@@@@@@%#                                       
                                      *%@@@@@@@-          @@@@*                          @@@@@          %@@@@@@@@#                                    
                                     @@@@@@@%@            @@@@@                          %@@@@            #@@@@@@@@                                   
                                      #%@@@@@@@##         %@@@@                          @@@@*        @#%@@@@@@@%@                                    
                                         @%@@@@@@@@#@     @@@@@@                        @@@@@      @#%%@@@@@@%                                        
                                            @%%@@@@@@@%%  @@@@@@                       *@@@@%  @*%@@@@@@@%%                                           
                                      @         #%@@@@@@@@%=@@@@%@                    %@@@@%+@@@@@@@@%@@         @                                    
                                     =@@%%@        @#@@@@@@@@@@@@@#                 =%@@@@@@@@@@@@@@         %%@@%                                    
                                       *@@@@@#         @#@@@@@@@@%%%%@            #@@@@@@@@@@@%*         *#%%@@@*                                     
                                        @%@@@@@@%%@        -#@@@@@@@@@@@@**@*#%@@@@@@@@@@@%*         %%@@@@@@@#                                       
                                          #@@@@@@@@@#%*         #*@@@@@@@@@@@@@@@@@@@@*          %#%%@@@@@@@%%                                        
                                           +@@@@@@@@@@@@##@            .%#@+#@*@.            ##%%@@@@@@%@@@#                                          
                                             %@@@@@@@@@@@@@@%#@                          ##%%@@@@@@@@@@@@@#                                           
                                              *@@@@@@@@@@@@@@@@@%#@                  *#%@@@@@@@@@@@@@@@@%                                             
                                                %@@@@@@@@@@@@@@@@@@@%*           =##@@@@@@@@@@@@@@@@@@@#                                              
                                                 *@@@@@@@@@@@@@@@@@@@@@%%#@  *%@@@@@@@@@@@@@@@@@@@@@@%                                                
                                                   %@@@@@@@@@@@@@@@@@@@@@@%@@@@@@@@@@@@@@@@@@@@@@@@@#                                                 
                                                    *@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%                                                   
                                                      %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                                    
                                                       *@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%                                                      
                                                         %@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%                                                       
                                                          %%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                                        
                                                            @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@%                                                          
                                                             *@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                                                           
                                                               %@@@@@@@@@@@@@@@@@@@@@@@@%                                                             
                                                                %@@@@@@@@@@@@@@@@@@@@@@*                                                              
                                                                 @%@@@@@@@@@@@@@@@@@@@                                                                
                                                                   @@@@@@@@@@@@@@@@@#                                                                 
                                                                    #@@@@@@@@@@@@@@                                                                   
                                                                      %@@@@@@@@@@#                                                                    
                                                                       *@@@@@@@%@                                                                     
                                                                         %@@@@%                                                                       
                                                                          +##                 


                                                                                      
                                                                                                                                                      
                                                                                                                                                      
                    @@@@@                   @@@@                                                                                                      
                    @@@@@@                @@@@@@                                                                       @@@                            
                   @@@@@@@                @@@@@@                              @@@@@@@@@@@@@@@@                    @@@@@@@@@@@                         
                   @@@@@@@              @@@@@@@@                   @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@          @@@@@@   @@@                         
                   @@@@@@@             @@@@@@@@                     @@@@@@@    @@@@@               @@@@@@       @@@@@       @                         
                  @@@@@@@@            @@@@@@@@@                                @@@@                           @@@@@@                                  
                  @@@@@@@@@          @@@@ @@@@@                                @@@@                           @@@@@                                   
                 @@@@@@@@@@         @@@@  @@@@                                 @@@@                          @@@@@                                    
                 @@@@  @@@@        @@@@   @ @@                                @@@@@                         @@@@@                                     
                 @@@@  @@@@       @@@@    @@@@                                @@@@                          @@@@                                      
                 @@@@  @@@@      @@@@     @@@                                 @@@@                         @@@@                                       
                 @@@    @@@@    @@@@      @@@                                @@@@@                        @@@@                                        
                 @@@    @@@@   @@@@      @@@@                                @@@@                         @@@@                                        
                @@@@    @@@@ @@@@@       @@@@                                @@@@                        @@@@                                         
                @@@@     @@@@@@@@         @@@                                @@@@                        @@@@                                         
                @@@@     @@@@@@@          @@@                                @@@                         @@@                                          
                @@@@      @@@@@           @@                                @@@@                        @@@@                                          
               @@@@@                      @@                                @@@@                        @@@@                  @@                      
               @@@@                      @@@                                @@@@                        @@@                   @@                      
               @@@@                       @@                                @@@@                        @@@                  @@@@                     
               @@@@                      @@@                                @@@@                        @@@                 @@@@                      
               @@@@                     @@@@                                @@@@                        @@@                 @@@@                      
               @@@@                     @@@           @@@      @@@@@@@@     @@@@      @@      @@@       @@@                @@@@    @@@@               
               @@@@                     @@@          @@@@@   @@@@@   @@@@   @@@@  @@@@@      @@@@@      @@@               @@@@    @@@@@               
               @@@@                     @@@           @@@   @@@@       @@@@@@@@@@@@@          @@@       @@@@             @@@@      @@@                
               @@@@                     @@@                   @@@@@@@@@@@@@@@@@@                         @@@           @@@@@                          
               @@@@                     @@@                       @@@@@@    @@@@                          @@@        @@@@@                            
               @@@@                     @@@                                                                @@@@@  @@@@@@                              
                                        @@@                                                                  @@@@@@@@@                                
                                       @@@@                                                                                                           
                                        @@                                                                                                            
"""

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
    print_slowly(MIC_LOGO_ASCII, delay = 0.00025) # Print logo instantly
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

# Add the new helper function check_if_saved_today
# Made synchronous as it doesn't perform async I/O in this context
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
             elif interval == "1wk":
                  data = stock.history(period="5y", interval=interval)

        if data.empty or 'Close' not in data.columns:
            print(f"Warning: Still no valid data or 'Close' column for {ticker_str} after period adjustments.")
            return None, None

    except Exception as e:
        print(f"Error fetching history for {ticker_str} (Interval {interval}, Period {period}): {e}")
        return None, None

    try:
        # Ensure enough data points for EMA calculation (at least 55 for EMA_55)
        if len(data) < 55:
             print(f"Warning: Insufficient data ({len(data)} points) for 55-period EMA for {ticker_str}. Cannot calculate EMA Invest.")
             # Return live price if available, but no EMA invest score
             live_price_fallback = data['Close'].iloc[-1] if not data.empty and not pd.isna(data['Close'].iloc[-1]) else None
             return live_price_fallback, None

        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_13'] = data['Close'].ewm(span=13, adjust=False).mean()
        data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()

    except Exception as e:
        print(f"Error calculating EMAs for {ticker_str}: {e}")
        return None, None

    # Check for NaN in the latest EMA values
    if data.iloc[-1].isna().any():
       print(f"Warning: Latest EMAs are invalid for {ticker_str}")
       # Return live price if available, but no EMA invest score
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

    # Clamp score
    ema_invest = max(0, min(ema_invest, 100))

    return live_price, ema_invest

# Calculate the one-year percent change and invest_per
def calculate_one_year_invest(ticker):
    """Calculates one-year percentage change and Invest percentage."""
    ticker_str = ticker.replace('.', '-')
    # Use the global session for yfinance
    stock = yf.Ticker(ticker_str, session=YFINANCE_SESSION)
    try:
        data = stock.history(period="1y")
        if data.empty or len(data) < 2 or 'Close' not in data.columns:
             print(f"Warning: Insufficient 1-year data or missing 'Close' column for {ticker_str}.")
             return 0.0, 50.0 # Neutral default
    except Exception as e:
        print(f"Error fetching 1-year history for {ticker_str}: {e}")
        return 0.0, 50.0 # Neutral default on error

    start_price = safe_score(data['Close'].iloc[0])
    end_price = safe_score(data['Close'].iloc[-1])

    if start_price <= 0: # Ensure start_price is positive before division
        print(f"Warning: Invalid or zero start price for 1-year calc on {ticker_str}. Cannot calculate change.")
        return 0.0, 50.0

    one_year_change = ((end_price - start_price) / start_price) * 100

    invest_per = 50.0 # Default
    if one_year_change < 0:
        invest_per = (one_year_change / 2) + 50
    else:
        try:
            # Ensure non-negative input to sqrt
            invest_per = math.sqrt(max(0, one_year_change * 5)) + 50
        except ValueError:
            print(f"Warning: ValueError in sqrt for {ticker_str}. Using default invest_per.")
            invest_per = 50.0

    # Clamp score
    invest_per = max(0, min(invest_per, 100))

    return one_year_change, invest_per

# Calculate S&P 500 and S&P 100 symbols
def get_sp500_symbols():
    """Fetches S&P 500 ticker symbols from Wikipedia."""
    try:
        sp500_list_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # For pd.read_html, session injection is not directly supported in the same way as yfinance.
        # If this part fails due to similar request issues, it might need a different approach,
        # e.g., fetching the HTML content using the curl_cffi session then passing it to pandas.
        # For now, assuming pd.read_html works or the issue is specific to yfinance's direct API calls.
        dfs = pd.read_html(sp500_list_url)
        if not dfs:
             print("Warning: No tables found on S&P 500 Wikipedia page.")
             return []
        df = dfs[0] # Assuming the first table contains the list
        if 'Symbol' not in df.columns:
             print("Warning: 'Symbol' column not found in S&P 500 table.")
             return []
        symbols = df['Symbol'].tolist()
        symbols = [s.replace('.', '-') for s in symbols if isinstance(s, str)] # Handle potential non-strings and '.'
        return symbols
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}")
        return []

def get_sp100_symbols():
    """Fetches S&P 100 ticker symbols from Wikipedia."""
    try:
        sp100_list_url = 'https://en.wikipedia.org/wiki/S%26P_100'
        # Similar to get_sp500_symbols, pd.read_html session injection needs consideration if issues arise.
        dfs = pd.read_html(sp100_list_url)
        if len(dfs) < 3:
             print("Warning: Not enough tables found on S&P 100 Wikipedia page.")
             return [] # Check if table exists
        df = dfs[2] # Usually the 3rd table
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
    Returns a tuple: (combined_score, general_score, large_score).
    Returns (None, None, None) on critical failure.
    """
    print("Fetching data for market risk calculation...")
    try:
        def calculate_ma_above(symbol, ma_window):
            """Checks if the latest price is above the specified moving average."""
            try:
                # Use yf.download with the global session
                data = yf.download(symbol, period='1y', interval='1d', progress=False, session=YFINANCE_SESSION)
                if data.empty or len(data) < ma_window or 'Close' not in data.columns:
                    return None
                rolling_mean = data['Close'].rolling(window=ma_window).mean()
                latest_price = data['Close'].iloc[-1]
                latest_ma = rolling_mean.iloc[-1]

                if pd.isna(latest_ma) or pd.isna(latest_price):
                    return None
                return latest_price > latest_ma
            except Exception as e:
                # print(f"Error calculating MA for {symbol}: {e}") # Reduced verbosity
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
             return calculate_percentage_above_ma(sp500_symbols, 20)

        def calculate_s5th():
             spy_symbols = get_spy_symbols() # Uses SP500
             return calculate_percentage_above_ma(spy_symbols, 200)

        def calculate_s1fd():
            sp100_symbols = get_sp100_symbols()
            return calculate_percentage_above_ma(sp100_symbols, 5)

        def calculate_s1tw():
             sp100_symbols = get_sp100_symbols()
             return calculate_percentage_above_ma(sp100_symbols, 20)

        def get_live_price_and_ma(ticker_str_param):
            """Fetches live price and specified moving averages for a ticker."""
            try:
                # Use the global session for yfinance
                stock = yf.Ticker(ticker_str_param, session=YFINANCE_SESSION)
                hist = stock.history(period="260d") # Approx 1 year of trading days
                if hist.empty or len(hist) < 50 or 'Close' not in hist.columns:
                     print(f"Warning: Insufficient data for MAs on {ticker_str_param}")
                     return None, None, None

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


        # --- Calculate Individual Components ---
        spy_live_price, spy_ma_20, spy_ma_50 = get_live_price_and_ma('SPY')
        vix_live_price, _, _ = get_live_price_and_ma('^VIX') # VIX only needs price
        rut_live_price, rut_ma_20, rut_ma_50 = get_live_price_and_ma('^RUT')
        oex_live_price, oex_ma_20, oex_ma_50 = get_live_price_and_ma('^OEX')

        essential_indices = [spy_live_price, vix_live_price, rut_live_price, oex_live_price]
        if any(d is None for d in essential_indices):
            print("Warning: Missing key index live prices for market risk calculation.")
            return None, None, None

        s5tw = calculate_s5tw()
        s5th = calculate_s5th()
        s1fd = calculate_s1fd()
        s1tw = calculate_s1tw()

        spy20 = ((safe_score(spy_live_price) - safe_score(spy_ma_20)) / 20) + 50 if spy_ma_20 is not None else 50
        spy50 = ((safe_score(spy_live_price) - safe_score(spy_ma_50) - 150) / 20) + 50 if spy_ma_50 is not None else 50
        vix_calc = (((safe_score(vix_live_price) - 15) * -5) + 50) if vix_live_price is not None else 50
        rut20 = ((safe_score(rut_live_price) - safe_score(rut_ma_20)) / 10) + 50 if rut_ma_20 is not None else 50
        rut50 = ((safe_score(rut_live_price) - safe_score(rut_ma_50)) / 5) + 50 if rut_ma_50 is not None else 50
        s5tw_calc = ((safe_score(s5tw) - 60) + 50)
        s5th_calc = ((safe_score(s5th) - 70) + 50)

        general_components = [
            (spy20, 3), (spy50, 1), (vix_calc, 3), (rut50, 3), (rut20, 1),
            (s5tw_calc, 2), (s5th_calc, 1)
        ]
        general_sum = sum(safe_score(score) * weight for score, weight in general_components)
        general_weights = sum(weight for _, weight in general_components)
        general_score = general_sum / general_weights if general_weights > 0 else 50.0

        oex20_calc = ((safe_score(oex_live_price) - safe_score(oex_ma_20)) / 10) + 50 if oex_ma_20 is not None else 50
        oex50_calc = ((safe_score(oex_live_price) - safe_score(oex_ma_50)) / 5) + 50 if oex_ma_50 is not None else 50
        s1fd_calc = ((safe_score(s1fd) - 60) + 50)
        s1tw_calc = ((safe_score(s1tw) - 70) + 50)

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
    # Use the global session for yfinance
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
             print(f"Warning: Insufficient data ({len(data)} points) or missing 'Close' column for MACD on {ticker_str} ({interval}). Required: {required_points}. Returning Neutral.")
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
         print(f"Warning: Not enough valid histogram points ({len(data['histogram'])}) for MACD signal on {ticker_str}. Returning Neutral.")
         return "Neutral", 50.0

    last_three_hist = data['histogram'].dropna().tail(3).tolist()

    if len(last_three_hist) < 3:
        print(f"Warning: Not enough non-NaN histogram points ({len(last_three_hist)}) for MACD signal on {ticker_str}. Returning Neutral.")
        return "Neutral", 50.0

    signal = "Neutral"
    if last_three_hist[2] < last_three_hist[1] < last_three_hist[0] and last_three_hist[2] < 0 :
        signal = "Sell"
    elif last_three_hist[2] > last_three_hist[1] > last_three_hist[0] and last_three_hist[2] > 0:
        signal = "Buy"
    elif last_three_hist[2] > last_three_hist[1] and last_three_hist[1] <= last_three_hist[0] and last_three_hist[2] > 0:
         signal = "Buy"
    elif last_three_hist[2] < last_three_hist[1] and last_three_hist[1] >= last_three_hist[0] and last_three_hist[2] < 0:
         signal = "Sell"

    macd_strength = 0.0
    if not any(pd.isna(h) for h in last_three_hist):
         strength_change1 = abs(last_three_hist[2] - last_three_hist[1])
         strength_change2 = abs(last_three_hist[1] - last_three_hist[0])
         macd_strength = (strength_change1 + strength_change2) / 2.0
         macd_strength_percent_adjusted = ((macd_strength / 2.0) * 100.0) + 50.0
    else:
        macd_strength_percent_adjusted = 50.0

    macd_strength_percent_adjusted = max(0, min(100, macd_strength_percent_adjusted))
    return signal, macd_strength_percent_adjusted


# Plot ticker graph
def plot_ticker_graph(ticker, ema_interval):
    """Plots ticker price and EMAs and saves to a file."""
    ticker_str = ticker.replace('.', '-')
    # Use the global session for yfinance
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
            raise ValueError(f"No data or 'Close' column returned for {ticker_str} (Period: {period}, Interval: {interval})")

        if len(data) < 55:
             print(f"Warning: Not enough data ({len(data)}) for 55-period EMA for plotting {ticker_str}. Plotting price only.")
             data['EMA_55'] = np.nan
        else:
             data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()

        if len(data) < 8:
             print(f"Warning: Not enough data ({len(data)}) for 8-period EMA for plotting {ticker_str}. Plotting price and EMA55 only.")
             data['EMA_8'] = np.nan
        else:
             data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()


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
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        fig.tight_layout()

        filename = f"{ticker_str}_graph.png"
        plt.savefig(filename, facecolor='black', edgecolor='black')
        plt.close(fig)
        return filename
    except Exception as e:
        print(f"Error plotting graph for {ticker_str}: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)
        return None

# Function to get allocation score (reads from market_data.csv)
def get_allocation_score():
    """
    Reads market data from market_data.csv and calculates allocation scores.
    Returns (avg_score, risk_gen_score, mkt_inv_score). Defaults to 50 if file missing.
    """
    market_data_file = 'market_data.csv'
    avg_score = 50.0
    risk_gen_score = 50.0
    mkt_inv_score = 50.0

    if not os.path.exists(market_data_file):
        return avg_score, risk_gen_score, mkt_inv_score

    try:
        df = pd.read_csv(market_data_file)
        if df.empty:
            return avg_score, risk_gen_score, mkt_inv_score

        last_row = df.iloc[-1]
        risk_gen_score = safe_score(last_row.get('RISK_GEN_SCORE', 50.0))
        mkt_inv_score = safe_score(last_row.get('MKT_INV_SCORE', 50.0))
        avg_score = safe_score(last_row.get('AVG_SCORE', 50.0))

        if 'AVG_SCORE' not in df.columns:
             avg_score = (risk_gen_score + mkt_inv_score) / 2.0

        avg_score = max(0, min(100, avg_score))
        risk_gen_score = max(0, min(100, risk_gen_score))
        mkt_inv_score = max(0, min(100, mkt_inv_score))
        return avg_score, risk_gen_score, mkt_inv_score

    except Exception as e:
        print(f"Error reading market data file '{market_data_file}': {e}. Using default scores (50).")
        return 50.0, 50.0, 50.0


# --- Modified process_custom_portfolio function ---
async def process_custom_portfolio(portfolio_data, tailor_portfolio, frac_shares,
                                   total_value=None, is_custom_command_without_save=False):
    """
    Processes custom or /invest portfolio requests, calculates scores, allocations,
    and generates output tables and graphs for the TERMINAL.
    """
    sell_to_cash_active = False
    avg_score, risk_gen_score, mkt_inv_score = get_allocation_score()

    if avg_score is not None and avg_score < 50.0:
        sell_to_cash_active = True
        print(f"INFO: Sell-to-Cash feature ACTIVE (Avg Market Score: {avg_score:.2f} < 50)")

    risk_tolerance = int(portfolio_data.get('risk_tolerance', 10))
    risk_type = portfolio_data.get('risk_type', 'stock')
    remove_amplification_cap = portfolio_data.get('remove_amplification_cap', 'true').lower() == 'true'
    ema_sensitivity = int(portfolio_data.get('ema_sensitivity', 3))
    try:
        amplification = float(portfolio_data.get('amplification', 1.0))
    except ValueError:
        print("Warning: Invalid amplification value found. Defaulting to 1.0")
        amplification = 1.0
    num_portfolios = int(portfolio_data.get('num_portfolios', 0))

    portfolio_results = []
    all_entries_for_graphs = []

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

        current_portfolio_list = []
        for ticker_item in tickers: # Renamed to avoid conflict with yf.Ticker
            try:
                live_price, ema_invest = calculate_ema_invest(ticker_item, ema_sensitivity)

                if live_price is None and ema_invest is None:
                     print(f"Warning: Failed to get base data for {ticker_item}. Skipping.")
                     current_portfolio_list.append({'ticker': ticker_item, 'error': f"Failed to get base data", 'portfolio_weight': weight})
                     all_entries_for_graphs.append({'ticker': ticker_item, 'error': f"Failed to get base data"})
                     continue

                if ema_invest is None:
                     print(f"Warning: EMA Invest score calculation failed for {ticker_item}, using neutral 50.")
                     ema_invest = 50.0
                if live_price is None:
                     print(f"Warning: Live price calculation failed for {ticker_item}. Assigning 0 price.")
                     live_price = 0.0

                _, one_year_invest = calculate_one_year_invest(ticker_item)
                ema_invest = safe_score(ema_invest)
                one_year_invest = safe_score(one_year_invest)

                if risk_type.lower() == 'stock':
                     raw_combined_invest = ema_invest
                elif risk_type.lower() == 'market':
                     raw_combined_invest = avg_score
                elif risk_type.lower() == 'both':
                     raw_combined_invest = (ema_invest + avg_score) / 2.0
                else:
                     raw_combined_invest = ema_invest
                     print(f"Warning: Invalid risk_type '{risk_type}' in config. Defaulting to 'stock'.")

                score_for_allocation = raw_combined_invest
                score_was_adjusted = False

                if sell_to_cash_active and raw_combined_invest < 50.0:
                    score_for_allocation = 50.0
                    score_was_adjusted = True

                amplified_score_adjusted = safe_score((score_for_allocation * amplification) - (amplification - 1) * 50)
                amplified_score_original = safe_score((raw_combined_invest * amplification) - (amplification - 1) * 50)

                amplified_score_adjusted_clamped = max(0, min(100, amplified_score_adjusted))
                amplified_score_original_clamped = max(0, min(100, amplified_score_original))

                entry_data = {
                    'ticker': ticker_item,
                    'live_price': live_price,
                    'raw_invest_score': raw_combined_invest,
                    'amplified_score_adjusted': amplified_score_adjusted_clamped,
                    'amplified_score_original': amplified_score_original_clamped,
                    'portfolio_weight': weight,
                    'score_was_adjusted': score_was_adjusted,
                    'portfolio_allocation_percent_adjusted': None,
                    'portfolio_allocation_percent_original': None,
                    'combined_percent_allocation_adjusted': None,
                    'combined_percent_allocation_original': None,
                }
                current_portfolio_list.append(entry_data)
                if live_price is not None and live_price > 0:
                    all_entries_for_graphs.append({'ticker': ticker_item, 'ema_sensitivity': ema_sensitivity})

            except Exception as e:
                print(f"Error processing ticker {ticker_item} in portfolio {portfolio_index}: {e}")
                import traceback
                traceback.print_exc()
                current_portfolio_list.append({'ticker': ticker_item, 'error': str(e), 'portfolio_weight': weight})
                all_entries_for_graphs.append({'ticker': ticker_item, 'error': str(e)})
        portfolio_results.append(current_portfolio_list)
    print("--- Finished Initial Ticker Calculations ---")

    print("Generating ticker graphs (saved as PNG files)...")
    sent_graphs = set()
    for graph_entry in all_entries_for_graphs:
        ticker_key = graph_entry.get('ticker')
        if not ticker_key or ticker_key in sent_graphs: continue
        if 'error' not in graph_entry:
            try:
                graph_filename = plot_ticker_graph(ticker_key, graph_entry['ema_sensitivity'])
                if graph_filename and os.path.exists(graph_filename):
                    print(f"  Graph saved to {graph_filename}")
                    sent_graphs.add(ticker_key)
                elif graph_filename:
                    print(f"  Could not generate graph file for {ticker_key}: File not found after creation attempt.")
                else:
                    print(f"  Failed to generate graph for {ticker_key}.")
            except Exception as plot_error:
                print(f"  Error plotting graph for {ticker_key}: {plot_error}")
        else:
             print(f"  Skipping graph for {ticker_key} due to earlier error: {graph_entry['error']}")
    print("--- Finished Graph Generation ---")

    print("--- Calculating Sub-Portfolio Allocations (Adjusted & Original) ---")
    for portfolio_index, portfolio in enumerate(portfolio_results):
        portfolio_amplified_total_adjusted = safe_score(sum(entry.get('amplified_score_adjusted', 0) for entry in portfolio if 'error' not in entry))
        for entry in portfolio:
            if 'error' not in entry:
                amplified_score_adj = safe_score(entry.get('amplified_score_adjusted', 0))
                if portfolio_amplified_total_adjusted > 0:
                     portfolio_allocation_percent_adj = safe_score((amplified_score_adj / portfolio_amplified_total_adjusted) * 100)
                     entry['portfolio_allocation_percent_adjusted'] = round(portfolio_allocation_percent_adj, 4)
                else: entry['portfolio_allocation_percent_adjusted'] = 0.0
            else: entry['portfolio_allocation_percent_adjusted'] = None

        portfolio_amplified_total_original = safe_score(sum(entry.get('amplified_score_original', 0) for entry in portfolio if 'error' not in entry))
        for entry in portfolio:
            if 'error' not in entry:
                amplified_score_orig = safe_score(entry.get('amplified_score_original', 0))
                if portfolio_amplified_total_original > 0:
                    portfolio_allocation_percent_orig = safe_score((amplified_score_orig / portfolio_amplified_total_original) * 100)
                    entry['portfolio_allocation_percent_original'] = round(portfolio_allocation_percent_orig, 4)
                else: entry['portfolio_allocation_percent_original'] = 0.0
            else: entry['portfolio_allocation_percent_original'] = None

    if not is_custom_command_without_save:
        for i, portfolio in enumerate(portfolio_results, 1):
            portfolio.sort(key=lambda x: safe_score(x.get('portfolio_allocation_percent_adjusted', -1)), reverse=True)
            portfolio_weight_display = safe_score(portfolio[0].get('portfolio_weight', 0)) if portfolio else 0.0
            print(f"\n--- Sub-Portfolio {i} (Weight: {portfolio_weight_display:.2f}%) ---")
            table_data = []
            for entry in portfolio:
                 if 'error' not in entry:
                    live_price_f = f"${safe_score(entry.get('live_price', 0)):,.2f}"
                    invest_score_val = safe_score(entry.get('raw_invest_score', 0))
                    invest_score_f = f"{invest_score_val:.2f}%"
                    amplified_score_f = f"{safe_score(entry.get('amplified_score_adjusted', 0)):.2f}%"
                    port_alloc_val_original = safe_score(entry.get('portfolio_allocation_percent_original', 0))
                    port_alloc_f = f"{port_alloc_val_original:.2f}%"
                    table_data.append([entry.get('ticker', 'ERR'), live_price_f, invest_score_f, amplified_score_f, port_alloc_f])
                 else:
                     table_data.append([entry.get('ticker', 'ERR'), 'Error', entry.get('error', 'Unknown error'), '-', '-'])
            if not table_data: print("No valid data for this sub-portfolio.")
            else: print(tabulate(table_data, headers=["Ticker", "Live Price", "Raw Score", "Adj Amplified %", "Portfolio % Alloc (Original)"], tablefmt="pretty"))

    print("--- Calculating Combined Portfolio Allocations (Adjusted & Original) ---")
    combined_result_intermediate = []
    for portfolio in portfolio_results:
        for entry in portfolio:
            if 'error' not in entry:
                port_weight = safe_score(entry.get('portfolio_weight', 0))
                sub_alloc_adj = safe_score(entry.get('portfolio_allocation_percent_adjusted', 0))
                combined_percent_allocation_adjusted = safe_score((sub_alloc_adj * port_weight) / 100.0)
                entry['combined_percent_allocation_adjusted'] = combined_percent_allocation_adjusted
                sub_alloc_orig = safe_score(entry.get('portfolio_allocation_percent_original', 0))
                combined_percent_allocation_original = safe_score((sub_alloc_orig * port_weight) / 100.0)
                entry['combined_percent_allocation_original'] = combined_percent_allocation_original
                combined_result_intermediate.append(entry)
            else:
                 combined_result_intermediate.append(entry)

    print("--- Constructing Final Combined Portfolio (with Cash if applicable) ---")
    final_combined_portfolio_data = []
    total_cash_diff_percent = 0.0
    valid_combined_intermediate = [entry for entry in combined_result_intermediate if 'error' not in entry]

    for entry in valid_combined_intermediate:
        if entry.get('score_was_adjusted', False):
            adj_alloc = safe_score(entry.get('combined_percent_allocation_adjusted', 0))
            orig_alloc = safe_score(entry.get('combined_percent_allocation_original', 0))
            difference = orig_alloc - adj_alloc
            total_cash_diff_percent += max(0.0, difference)

    for entry in valid_combined_intermediate:
        final_combined_portfolio_data.append({
            'ticker': entry['ticker'],
            'live_price': entry['live_price'],
            'raw_invest_score': entry['raw_invest_score'],
            'amplified_score_adjusted': entry['amplified_score_adjusted'],
            'combined_percent_allocation': safe_score(entry['combined_percent_allocation_adjusted'])
        })

    current_stock_total_alloc = sum(item['combined_percent_allocation'] for item in final_combined_portfolio_data)
    target_stock_alloc = 100.0 - total_cash_diff_percent

    if not math.isclose(current_stock_total_alloc, target_stock_alloc, abs_tol=0.01):
        print(f"    Normalizing final stock allocations slightly. Current sum: {current_stock_total_alloc:.4f}%, Target: {target_stock_alloc:.4f}%")
        if current_stock_total_alloc > 1e-9:
             norm_factor = target_stock_alloc / current_stock_total_alloc
             for item in final_combined_portfolio_data:
                 item['combined_percent_allocation'] *= norm_factor
        else:
             print("    Warning: Cannot normalize zero stock allocations.")

    if total_cash_diff_percent > 1e-4:
        final_combined_portfolio_data.append({
            'ticker': 'Cash',
            'live_price': 1.0,
            'raw_invest_score': None,
            'amplified_score_adjusted': None,
            'combined_percent_allocation': total_cash_diff_percent
        })
        print(f"    Added Cash row to Final Combined Portfolio: {total_cash_diff_percent:.2f}%")

    final_combined_portfolio_data.sort(
        key=lambda x: safe_score(x.get('raw_invest_score', -float('inf'))) if x['ticker'] != 'Cash' else -float('inf')-1,
        reverse=True
    )

    if not is_custom_command_without_save:
        print("\n--- Final Combined Portfolio (Sorted by Raw Score)---")
        if sell_to_cash_active: print("*(Sell-to-Cash Active: Difference allocated to Cash)*")
        combined_data_display = []
        for entry in final_combined_portfolio_data:
            ticker_disp = entry.get('ticker', 'ERR') # Renamed to avoid conflict
            if ticker_disp == 'Cash':
                live_price_f = '-'
                invest_score_f = '-'
                amplified_score_f = '-'
            else:
                live_price_f = f"${safe_score(entry.get('live_price', 0)):,.2f}"
                invest_score_f = f"{safe_score(entry.get('raw_invest_score', 0)):.2f}%"
                amplified_score_f = f"{safe_score(entry.get('amplified_score_adjusted', 0)):.2f}%"
            comb_alloc_f = f"{safe_score(entry.get('combined_percent_allocation', 0)):.2f}%"
            combined_data_display.append([ticker_disp, live_price_f, invest_score_f, amplified_score_f, comb_alloc_f])
        if not combined_data_display: print("No valid data for the combined portfolio.")
        else: print(tabulate(combined_data_display, headers=["Ticker", "Live Price", "Raw Score", "Adj Amplified %", "Final % Alloc"], tablefmt="pretty"))

    print("--- Calculating Tailored Portfolio ---")
    tailored_portfolio_output_list = []
    tailored_portfolio_table_data = []
    remaining_buying_power = None
    final_cash_value_tailored = 0.0

    if tailor_portfolio:
        if total_value is None or safe_score(total_value) <= 0:
            print("Error: Tailored portfolio requested but total value is missing or invalid. Cannot proceed with tailoring.")
            return tailored_portfolio_output_list, combined_result_intermediate, portfolio_results
        else:
            total_value = safe_score(total_value)

        tailored_portfolio_entries_intermediate = []
        total_actual_money_allocated_stocks = 0.0

        for entry in final_combined_portfolio_data:
            if entry['ticker'] == 'Cash':
                 if sell_to_cash_active:
                     final_cash_value_tailored += total_value * (safe_score(entry.get('combined_percent_allocation', 0.0)) / 100.0)
                 continue

            final_stock_alloc_pct = safe_score(entry.get('combined_percent_allocation', 0.0))
            live_price = safe_score(entry.get('live_price', 0.0))

            if final_stock_alloc_pct > 1e-9 and live_price > 0:
                target_allocation_value = total_value * (final_stock_alloc_pct / 100.0)
                shares = 0.0
                try:
                    exact_shares = target_allocation_value / live_price
                    if frac_shares:
                        shares = round(exact_shares, 4)
                    else:
                        shares = float(math.floor(exact_shares))
                except ZeroDivisionError:
                    print(f"Warning: ZeroDivisionError encountered for {entry.get('ticker','ERR')} with price {live_price}. Shares set to 0.")
                    shares = 0.0
                except Exception as e_shares:
                    print(f"Error calculating shares for {entry.get('ticker', 'ERR')}: {e_shares}")
                    shares = 0.0

                shares = max(0.0, shares)
                actual_money_allocation = shares * live_price
                share_threshold = 0.0001 if frac_shares else 1.0

                if shares >= share_threshold:
                    actual_percent_allocation = (actual_money_allocation / total_value) * 100.0 if total_value > 0 else 0.0
                    tailored_portfolio_entries_intermediate.append({
                        'ticker': entry.get('ticker','ERR'),
                        'raw_invest_score': entry.get('raw_invest_score', -float('inf')),
                        'shares': shares,
                        'actual_money_allocation': actual_money_allocation,
                        'actual_percent_allocation': actual_percent_allocation
                    })
                    total_actual_money_allocated_stocks += actual_money_allocation

        remaining_value_after_stocks = total_value - total_actual_money_allocated_stocks
        final_cash_value_tailored += remaining_value_after_stocks
        remaining_buying_power = total_value - total_actual_money_allocated_stocks
        final_cash_value_tailored = max(0.0, final_cash_value_tailored)
        final_cash_percent_tailored = (final_cash_value_tailored / total_value) * 100.0 if total_value > 0 else 0.0
        final_cash_percent_tailored = max(0.0, min(100.0, final_cash_percent_tailored))

        print(f"    Tailored - Total Initial Value: ${total_value:,.2f}")
        print(f"    Tailored - Total Allocated to Stocks: ${total_actual_money_allocated_stocks:,.2f}")
        print(f"    Tailored - Remaining Value After Stocks: ${remaining_value_after_stocks:,.2f}")
        print(f"    Tailored - Final Clamped Cash Value: ${final_cash_value_tailored:,.2f}")
        print(f"    Tailored - Final Cash Percent: {final_cash_percent_tailored:.2f}%")
        print(f"    Tailored - Remaining Buying Power: ${remaining_buying_power:,.2f}")

        tailored_portfolio_entries_intermediate.sort(
            key=lambda x: safe_score(x.get('raw_invest_score', -float('inf'))),
            reverse=True
        )

        tailored_portfolio_table_data = [
             [item['ticker'], f"{item['shares']:.4f}" if frac_shares and item['shares'] > 0 else f"{int(item['shares'])}",
              f"${safe_score(item['actual_money_allocation']):,.2f}", f"{safe_score(item['actual_percent_allocation']):.2f}%"]
             for item in tailored_portfolio_entries_intermediate
        ]
        tailored_portfolio_table_data.append(['Cash', '-', f"${safe_score(final_cash_value_tailored):,.2f}", f"{safe_score(final_cash_percent_tailored):.2f}%"])

        if frac_shares:
            tailored_portfolio_output_list = ["{} - {:.4f} shares (${:,.2f})".format(item['ticker'], item['shares'], item['actual_money_allocation']) for item in tailored_portfolio_entries_intermediate]
        else:
            tailored_portfolio_output_list = ["{} - {:.0f} shares (${:,.2f})".format(item['ticker'], item['shares'], item['actual_money_allocation']) for item in tailored_portfolio_entries_intermediate]
        tailored_portfolio_output_list.append("Cash - ${:,.2f} ({:.2f}%)".format(safe_score(final_cash_value_tailored), safe_score(final_cash_percent_tailored)))

        if is_custom_command_without_save:
            if tailored_portfolio_output_list:
                print("\n--- Tailored Portfolio Allocation (Sorted by Raw Score) ---")
                for line in tailored_portfolio_output_list: print(line)
            else:
                print("No stocks allocated in the tailored portfolio based on the provided value and strategy.")
        else:
            print("\n--- Tailored Portfolio (Sorted by Raw Score) ---")
            if not tailored_portfolio_table_data:
                 print("No stocks allocated based on the provided value and strategy.")
            else:
                print(tabulate(tailored_portfolio_table_data, headers=["Ticker", "Shares", "Actual $ Allocation", "Actual % Allocation"], tablefmt="pretty"))

    print("--- Finished Tailored Portfolio ---")
    return tailored_portfolio_output_list, combined_result_intermediate, portfolio_results

# --- Terminal Input Handling Functions ---
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
async def collect_portfolio_inputs_terminal(portfolio_code):
    """Collects portfolio configuration inputs from the terminal."""
    inputs = {'portfolio_code': portfolio_code}
    portfolio_weights = []
    print(f"\nLet's set up portfolio code '{portfolio_code}'. Please answer the following questions.")

    ema_sensitivity = get_int_input("Enter EMA sensitivity (1: Weekly, 2: Daily, 3: Hourly): ", min_value=1, max_value=3)
    inputs['ema_sensitivity'] = str(ema_sensitivity)

    valid_amplifications = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    amplification_str = "" # Initialize
    while True:
        amplification_str = get_user_input(f"Enter amplification ({', '.join(map(str, valid_amplifications))}): ")
        try:
            amplification = float(amplification_str)
            if amplification in valid_amplifications:
                break
            else:
                print(f"Invalid amplification value. Please enter one of: {', '.join(map(str, valid_amplifications))}")
        except ValueError:
            print(f"Invalid number format. Please enter one of: {', '.join(map(str, valid_amplifications))}")
    inputs['amplification'] = str(amplification)


    num_portfolios = get_int_input("Enter the number of portfolios to analyze (e.g., 2): ", min_value=1)
    inputs['num_portfolios'] = str(num_portfolios)

    frac_shares = get_yes_no_input("Allow fractional shares? (yes/no): ")
    inputs['frac_shares'] = str(frac_shares).lower()

    inputs['risk_tolerance'] = '10'
    inputs['risk_type'] = 'stock'
    inputs['remove_amplification_cap'] = 'true'

    for i in range(num_portfolios):
        portfolio_num = i + 1
        tickers = get_user_input(f"Enter tickers for Portfolio {portfolio_num} (comma-separated): ", validation=lambda r: r and r.strip(), error_message="Tickers cannot be empty.")
        inputs[f'tickers_{portfolio_num}'] = tickers.upper()

        if portfolio_num == num_portfolios:
            if num_portfolios == 1:
                 weight = 100.0
            else:
                remaining_weight = 100.0 - sum(portfolio_weights)
                weight = remaining_weight
            if weight < -0.01:
                 print(f"Error: Previous weights exceed 100% ({sum(portfolio_weights):.2f}%). Cannot set weight for Portfolio {portfolio_num}. Please start over.")
                 return None
            weight = max(0.0, weight)
            inputs[f'weight_{portfolio_num}'] = f"{weight:.2f}"
            if num_portfolios > 1:
                print(f"Weight for final Portfolio {portfolio_num} automatically set to {weight:.2f}%.")
        else:
            remaining_weight = 100.0 - sum(portfolio_weights)
            weight = get_float_input(f"Enter weight for Portfolio {portfolio_num} (0-{remaining_weight:.2f}). Remaining: {remaining_weight:.2f}%: ", min_value=0, max_value=remaining_weight + 0.01)
            portfolio_weights.append(weight)
            inputs[f'weight_{portfolio_num}'] = f"{weight:.2f}"

    final_total_weight = sum(float(inputs.get(f'weight_{p+1}', 0)) for p in range(num_portfolios))
    if not math.isclose(final_total_weight, 100.0, abs_tol=0.1):
        print(f"Warning: Final weights sum to {final_total_weight:.2f}%, not exactly 100%. This may affect combined allocation percentages.")
    return inputs

async def save_portfolio_to_csv(file_path, portfolio_data):
    """
    Saves the portfolio configuration dictionary to a CSV file.
    """
    file_exists = os.path.isfile(file_path)
    fieldnames = list(portfolio_data.keys())
    if 'tailor_portfolio' in fieldnames:
        fieldnames.remove('tailor_portfolio')
    if 'portfolio_code' in fieldnames:
        fieldnames.insert(0, fieldnames.pop(fieldnames.index('portfolio_code')))

    try:
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
             writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
             if not file_exists or os.path.getsize(file_path) == 0:
                 writer.writeheader()
             data_to_save = {k: v for k, v in portfolio_data.items() if k in fieldnames}
             writer.writerow(data_to_save)
        print(f"Portfolio configuration saved to {file_path}")
    except IOError as e:
        print(f"Error writing to CSV {file_path}: {e}")
    except Exception as e:
        print(f"Unexpected error saving portfolio config to CSV: {e}")
        import traceback
        traceback.print_exc()

async def save_portfolio_data_internal(portfolio_code, date_str):
    """
    Internal function to save combined portfolio data for a given code and date.
    """
    portfolio_db_file = 'portfolio_codes_database.csv'
    portfolio_data_config = None # Renamed to avoid conflict

    try:
        if not os.path.exists(portfolio_db_file):
             print(f"Error [Save]: Portfolio database '{portfolio_db_file}' not found. Cannot save.")
             return
        with open(portfolio_db_file, 'r', encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file)
            found = False
            for row in reader:
                if row.get('portfolio_code', '').strip().lower() == portfolio_code.lower():
                    portfolio_data_config = row
                    found = True
                    break
            if not found:
                 print(f"Error [Save]: Portfolio code '{portfolio_code}' not found in '{portfolio_db_file}'. Cannot save.")
                 return
    except Exception as e:
        print(f"Error [Save]: Reading portfolio database {portfolio_db_file} for {portfolio_code}: {e}")
        import traceback
        traceback.print_exc()
        return

    if portfolio_data_config and date_str:
        try:
            frac_shares = portfolio_data_config.get('frac_shares', 'false').lower() == 'true'
            num_portfolios = int(portfolio_data_config.get('num_portfolios', 0))
            ema_sensitivity = int(portfolio_data_config.get('ema_sensitivity', 3))
            amplification = float(portfolio_data_config.get('amplification', 1.0))
            risk_tolerance = int(portfolio_data_config.get('risk_tolerance', 10))
            remove_amplification_cap = portfolio_data_config.get('remove_amplification_cap', 'true').lower() == 'true'

            processed_portfolio_data = {
                'portfolio_code': portfolio_code,
                'ema_sensitivity': ema_sensitivity,
                'amplification': amplification,
                'num_portfolios': num_portfolios,
                'frac_shares': frac_shares,
                'risk_tolerance': risk_tolerance,
                'risk_type': portfolio_data_config.get('risk_type', 'stock'),
                'remove_amplification_cap': remove_amplification_cap
            }
            for i in range(1, num_portfolios + 1):
                 processed_portfolio_data[f'tickers_{i}'] = portfolio_data_config.get(f'tickers_{i}', '')
                 processed_portfolio_data[f'weight_{i}'] = float(portfolio_data_config.get(f'weight_{i}', '0'))

            _, combined_result, _ = await process_custom_portfolio(
                portfolio_data=processed_portfolio_data,
                tailor_portfolio=False,
                frac_shares=frac_shares,
                total_value=None,
                is_custom_command_without_save=False
            )

            if combined_result:
                valid_combined = [entry for entry in combined_result if 'error' not in entry]
                sorted_combined = sorted(valid_combined, key=lambda x: safe_score(x.get('combined_percent_allocation', -1)), reverse=True)
                save_file = f"portfolio_code_{portfolio_code}_data.csv"
                file_exists = os.path.isfile(save_file)
                save_count = 0
                headers = ['DATE', 'TICKER', 'PRICE', 'COMBINED_ALLOCATION_PERCENT']
                try:
                    with open(save_file, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.DictWriter(f, fieldnames=headers)
                        if not file_exists or os.path.getsize(save_file) == 0:
                            writer.writeheader()
                        for item in sorted_combined:
                            ticker_save = item.get('ticker', 'ERR') # Renamed
                            price_val = item.get('live_price')
                            alloc_val = item.get('combined_percent_allocation')
                            price_str = f"{safe_score(price_val):.2f}" if price_val is not None else "N/A"
                            alloc_str = f"{safe_score(alloc_val):.4f}" if alloc_val is not None else "N/A"
                            writer.writerow({
                                'DATE': date_str,
                                'TICKER': ticker_save,
                                'PRICE': price_str,
                                'COMBINED_ALLOCATION_PERCENT': alloc_str
                            })
                            save_count += 1
                    print(f"[Save]: Saved {save_count} rows of combined portfolio data for code '{portfolio_code}' to '{save_file}' for date {date_str}.")
                except IOError as e:
                     print(f"Error [Save]: Writing combined portfolio data to CSV {save_file}: {e}")
                except Exception as e:
                     print(f"Unexpected error [Save]: Saving combined portfolio data: {e}")
                     import traceback
                     traceback.print_exc()
            else:
                print(f"[Save]: No valid combined portfolio data generated for code '{portfolio_code}'.")
        except Exception as e:
            print(f"Error [Save]: Processing/saving for code {portfolio_code}: {e}")
            import traceback
            traceback.print_exc()

async def save_portfolio_data_terminal(portfolio_code):
    """Prompts user for date and saves the *combined portfolio output* for terminal."""
    print(f"Attempting to save combined data for portfolio code: '{portfolio_code}'...")
    save_date_str = get_user_input("Enter the date to save the data under (MM/DD/YYYY): ", validation=lambda d: d and d.strip(), error_message="Date cannot be empty.")
    try:
        await save_portfolio_data_internal(portfolio_code, save_date_str)
        print(f"Save process completed for portfolio '{portfolio_code}' for date {save_date_str}. Check logs for details.")
    except Exception as e:
        print(f"An error occurred while saving data for portfolio code '{portfolio_code}'. Check logs.")
        import traceback
        traceback.print_exc()

async def handle_custom_command(args):
    """Handles the /custom command from the terminal."""
    print("\n--- /custom Command ---")
    if not args:
        print("Usage: /custom <portfolio_code> [save_code=3725]")
        return

    portfolio_code = args[0].strip()
    save_code = None
    if len(args) > 1:
        if args[1].lower().startswith("save_code="):
            save_code = args[1][len("save_code="):].strip()
        else:
            print("Invalid argument format. Use: /custom <portfolio_code> [save_code=3725]")
            return

    portfolio_db_file = 'portfolio_codes_database.csv'
    is_new_code_auto = False

    if portfolio_code == '#':
        next_code_num = 1
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
                import traceback
                traceback.print_exc()
                next_code_num = 1
        portfolio_code = str(next_code_num)
        is_new_code_auto = True
        print(f"Using next available portfolio code: '{portfolio_code}'")

    if save_code == "3725":
        if is_new_code_auto:
            print("Cannot use '#' with save_code. Please provide an existing code to save.")
            return
        await save_portfolio_data_terminal(portfolio_code)
        return

    portfolio_data_from_csv = None # Renamed
    file_exists = os.path.isfile(portfolio_db_file)

    if file_exists and not is_new_code_auto:
        try:
             with open(portfolio_db_file, 'r', encoding='utf-8', newline='') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row.get('portfolio_code', '').strip().lower() == portfolio_code.lower():
                        portfolio_data_from_csv = row
                        break
        except Exception as e:
            print(f"Error reading existing portfolio database {portfolio_db_file}: {e}")
            import traceback
            traceback.print_exc()
            print("Error accessing portfolio database.")
            return

    if portfolio_data_from_csv is None:
        create_message = f"Portfolio code '{portfolio_code}' not found." if file_exists and not is_new_code_auto else \
                         f"Portfolio database '{portfolio_db_file}' not found." if not file_exists else \
                         f"Creating new portfolio with auto-generated code '{portfolio_code}'." if is_new_code_auto else \
                         f"Portfolio code '{portfolio_code}' not found."
        print(f"{create_message} Let's create it.")
        new_portfolio_data = await collect_portfolio_inputs_terminal(portfolio_code)
        if new_portfolio_data:
             try:
                await save_portfolio_to_csv(portfolio_db_file, new_portfolio_data)
                portfolio_data_from_csv = new_portfolio_data
                print(f"New portfolio configuration '{portfolio_code}' saved.")
             except Exception as e:
                 print(f"Error saving new portfolio config for {portfolio_code}: {e}")
                 import traceback
                 traceback.print_exc()
                 print("Error saving new portfolio configuration. Cannot proceed with analysis.")
                 return
        else:
            print("Portfolio configuration collection cancelled.")
            return

    if portfolio_data_from_csv:
        try:
            tailor_portfolio_requested = False
            total_value = None
            allow_tailoring_prompt = (save_code != "3725")

            if allow_tailoring_prompt:
                value_str = get_user_input("Enter the total portfolio value to tailor (leave blank to skip tailoring): ")
                if value_str:
                    try:
                        total_value = float(value_str)
                        if total_value <= 0:
                            print("Value must be positive. Proceeding without tailoring.")
                            total_value = None
                            tailor_portfolio_requested = False
                        else:
                            tailor_portfolio_requested = True
                    except ValueError:
                        print("Invalid number format for value. Proceeding without tailoring.")
                        total_value = None
                        tailor_portfolio_requested = False
                else:
                    print("Skipping tailoring.")
                    tailor_portfolio_requested = False
                    total_value = None
            else:
                 print("Tailoring skipped because save_code=3725 was provided.")

            frac_shares = portfolio_data_from_csv.get('frac_shares', 'false').lower() == 'true'
            print(f"Processing custom portfolio code: '{portfolio_code}'...")
            await process_custom_portfolio(
                portfolio_data=portfolio_data_from_csv,
                tailor_portfolio=tailor_portfolio_requested,
                frac_shares=frac_shares,
                total_value=total_value,
                is_custom_command_without_save=(tailor_portfolio_requested and save_code != "3725")
            )
            print(f"Custom portfolio analysis for '{portfolio_code}' complete.")
        except KeyError as e:
            print(f"Incomplete configuration for portfolio code {portfolio_code}. Missing key: {e}")
            print(f"Error: Configuration for portfolio code '{portfolio_code}' seems incomplete. Please check the 'portfolio_codes_database.csv' or recreate the code.")
            import traceback
            traceback.print_exc()
        except Exception as e:
            print(f"Error processing custom portfolio {portfolio_code}: {e}")
            import traceback
            traceback.print_exc()
            print(f"An unexpected error occurred while processing portfolio '{portfolio_code}'. Check logs.")

async def handle_invest_command(args):
    """Handles the /invest command from the terminal."""
    print("\n--- /invest Command ---")
    ema_sensitivity = get_int_input("Enter EMA sensitivity (1: Weekly, 2: Daily, 3: Hourly): ", min_value=1, max_value=3)

    valid_amplifications = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    amplification_val = 0.0 # Initialize
    while True:
        amplification_str = get_user_input(f"Enter amplification ({', '.join(map(str, valid_amplifications))}): ")
        try:
            amplification_val = float(amplification_str)
            if amplification_val in valid_amplifications:
                break
            else:
                print(f"Invalid amplification value. Please enter one of: {', '.join(map(str, valid_amplifications))}")
        except ValueError:
             print(f"Invalid number format. Please enter one of: {', '.join(map(str, valid_amplifications))}")


    num_portfolios = get_int_input("How many portfolios would you like to calculate? (enter a number > 0): ", min_value=1)
    tailor_portfolio = get_yes_no_input("Would you like to tailor the table to your portfolio value? (yes/no): ")
    frac_shares = get_yes_no_input("Would you like to tailor the table using fractional shares? (yes/no): ")
    total_value = None
    if tailor_portfolio:
         value_str = get_user_input("Enter the total value for the combined portfolio: ")
         if value_str:
             try:
                 total_value = float(value_str)
                 if total_value <= 0:
                     print("Value must be positive. Proceeding without tailoring.")
                     total_value = None
                     tailor_portfolio = False
             except ValueError:
                 print("Invalid number format for value. Proceeding without tailoring.")
                 total_value = None
                 tailor_portfolio = False
         else:
             print("Skipping tailoring.")
             tailor_portfolio = False
             total_value = None

    all_portfolio_inputs = []
    portfolio_weights = []

    for i in range(1, num_portfolios + 1):
        current_portfolio_input = {}
        current_portfolio_weight = 0.0
        if i == num_portfolios:
            if num_portfolios == 1:
                current_portfolio_weight = 100.0
            else:
                remaining_weight = 100.0 - sum(portfolio_weights)
                current_portfolio_weight = remaining_weight
            if current_portfolio_weight < -0.01:
                print(f"Error: Previous weights exceed 100% ({sum(portfolio_weights):.2f}%). Please try again.")
                return
            current_portfolio_weight = max(0.0, current_portfolio_weight)
            if num_portfolios > 1:
                print(f"Weight for final Portfolio {i} automatically set to {current_portfolio_weight:.2f}%.")
            weight = current_portfolio_weight
        else:
            remaining_weight = 100.0 - sum(portfolio_weights)
            weight = get_float_input(f"Enter weight for Portfolio {i} (0-{remaining_weight:.2f}). Remaining: {remaining_weight:.2f}%: ", min_value=0, max_value=remaining_weight + 0.01)
            portfolio_weights.append(weight)
        current_portfolio_input['weight'] = weight
        tickers_str = get_user_input(f"Enter tickers for Portfolio {i} (comma-separated): ", validation=lambda r: r and r.strip(), error_message="Tickers cannot be empty.")
        tickers_list = [ticker.strip().upper() for ticker in tickers_str.split(',') if ticker.strip()] # Renamed
        if not tickers_list:
             print(f"Tickers cannot be empty for Portfolio {i}. Please try again.")
             return
        current_portfolio_input['tickers'] = tickers_list
        all_portfolio_inputs.append(current_portfolio_input)

    total_weight_check = sum(p['weight'] for p in all_portfolio_inputs)
    if not math.isclose(total_weight_check, 100.0, abs_tol=0.1):
         print(f"Warning: Total portfolio weight should sum to 100%. Current sum is {total_weight_check:.2f}%.")

    print(f"Processing /invest request with {num_portfolios} portfolio(s)...")
    portfolio_data_dict = {
         'risk_type': 'stock',
         'risk_tolerance': '10',
         'ema_sensitivity': str(ema_sensitivity),
         'amplification': str(amplification_val), # Use the validated float
         'num_portfolios': str(num_portfolios),
         'frac_shares': str(frac_shares).lower(),
         'remove_amplification_cap': 'true'
    }
    for i, p_data in enumerate(all_portfolio_inputs):
         portfolio_data_dict[f'tickers_{i+1}'] = ",".join(p_data['tickers'])
         portfolio_data_dict[f'weight_{i+1}'] = f"{p_data['weight']:.2f}"
    try:
        await process_custom_portfolio(
            portfolio_data=portfolio_data_dict,
            tailor_portfolio=tailor_portfolio,
            frac_shares=frac_shares,
            total_value=total_value,
            is_custom_command_without_save=False
        )
        print(f"/invest analysis complete.")
    except Exception as e:
         print(f"Error during /invest processing: {e}")
         import traceback
         traceback.print_exc()
         print(f"An error occurred during the analysis. Check logs.")

async def handle_market_command(args):
    """Handles the /market command from the terminal."""
    print("\n--- /market Command ---")
    save_code = None
    if args:
        if args[0].lower().startswith("save_code="):
            save_code = args[0][len("save_code="):].strip()
        else:
            print("Invalid argument format. Usage: /market [save_code=3725]")
            return

    print("Calculating market risk scores...")
    try:
        combined_score, general_score, large_score = calculate_market_risk()
        if combined_score is None:
            print("Failed to calculate market risk scores.")
            print("Attempting to load latest saved market scores...")
            avg_score_file, risk_gen_score_file, mkt_inv_score_file = get_allocation_score()
            if avg_score_file == 50.0 and risk_gen_score_file == 50.0 and mkt_inv_score_file == 50.0:
                 print("Could not load scores from file either. Market data unavailable.")
                 return
            else:
                 print("Using scores from existing market data file:")
                 print(f"  Average Score: {avg_score_file:.2f}%")
                 print(f"  Risk General Score: {risk_gen_score_file:.2f}%")
                 print(f"  Market Invest Score: {mkt_inv_score_file:.2f}%")
                 if save_code == "3725":
                      print("Cannot save new market data as calculation failed.")
                 return

        print("\n--- Market Insights ---")
        print(f"Combined Market Risk Score: {safe_score(combined_score):.2f}%")
        print(f"Risk General Score: {safe_score(general_score):.2f}%")
        print(f"Market Invest Score: {safe_score(large_score):.2f}%")

        if save_code == "3725":
            print("Save code provided. Saving market data...")
            save_file = 'market_data.csv'
            file_exists = os.path.isfile(save_file)
            save_date_str = datetime.now(est_timezone).strftime('%m/%d/%Y')
            if check_if_saved_today(save_file, save_date_str):
                print(f"Market data for {save_date_str} already saved to '{save_file}'. Skipping save.")
            else:
                try:
                    with open(save_file, 'a', newline='', encoding='utf-8') as f:
                        headers = ['DATE', 'RISK_GEN_SCORE', 'MKT_INV_SCORE', 'AVG_SCORE']
                        writer = csv.DictWriter(f, fieldnames=headers)
                        if not file_exists or os.path.getsize(save_file) == 0:
                            writer.writeheader()
                        writer.writerow({
                            'DATE': save_date_str,
                            'RISK_GEN_SCORE': f"{safe_score(general_score):.2f}",
                            'MKT_INV_SCORE': f"{safe_score(large_score):.2f}",
                            'AVG_SCORE': f"{safe_score(combined_score):.2f}"
                        })
                    print(f"Market data for {save_date_str} saved to '{save_file}'.")
                except IOError as e:
                    print(f"Error writing market data to CSV {save_file}: {e}")
                except Exception as e:
                    print(f"Unexpected error saving market data: {e}")
                    import traceback
                    traceback.print_exc()
    except Exception as e:
        print(f"Error calculating or displaying market data: {e}")
        import traceback
        traceback.print_exc()

async def handle_breakout_command(args):
    """Handles the /breakout command from the terminal."""
    print("\n--- /breakout Command ---")
    save_code = None
    if args:
        if args[0].lower().startswith("save_code="):
            save_code = args[0][len("save_code="):].strip()
        else:
            print("Invalid argument format. Usage: /breakout [save_code=3725]")
            return

    print("Running Breakout Scan...")
    try:
        # Note: tradingview_screener might also benefit from session impersonation if it makes HTTP requests
        # that are being blocked. This would require investigating its API or source.
        # For now, assuming it works or the yfinance fix is the primary concern.
        screen_query = Query(
            screener_type="america",
            markets=["stock"],
            filters=[
                {"left": "close", "operation": "greater", "right": 5},
                {"left": "volume", "operation": "greater", "right": 100000},
                {"left": "change", "operation": "greater", "right": 1},
            ],
            columns=[
                "name", "close", "change", "change_abs", "volume", "RSI", "MACD.macd", "MACD.signal",
                "EMA50", "EMA200", "exchange", "sector", "industry", "ticker" # Ensure 'ticker' is requested
            ],
        )
        screener_results_tuple = screen_query.get_scanner_data()

        if screener_results_tuple is None or not isinstance(screener_results_tuple, tuple) or len(screener_results_tuple) < 2:
             print("Breakout scan returned no results or failed to get data.")
             return
        screener_results_list, df_results = screener_results_tuple
        if df_results is None or df_results.empty:
            print("Breakout scan returned an empty DataFrame.")
            return
        print(f"Found {len(df_results)} potential breakout candidates from the initial scan.")

        breakout_candidates = []
        print("Calculating breakout scores...")
        for index, row in df_results.iterrows():
            ticker_scan = row.get('ticker', '').strip() # Renamed
            exchange = row.get('exchange', '').strip()
            if not ticker_scan:
                continue
            try:
                price = safe_score(row.get('close'))
                change_pct = safe_score(row.get('change'))
                volume = safe_score(row.get('volume'))
                rsi = safe_score(row.get('RSI'))
                macd_line = safe_score(row.get('MACD.macd'))
                signal_line = safe_score(row.get('MACD.signal'))
                ema50 = safe_score(row.get('EMA50'))
                ema200 = safe_score(row.get('EMA200'))
                macd_signal_calc, macd_strength_calc = calculate_macd_signal(ticker_scan, ema_interval=2)
                simple_score = 50.0
                if price > 0:
                    if ema50 > 0 and price > ema50: simple_score += 5
                    if ema200 > 0 and price > ema200: simple_score += 5
                    if macd_signal_calc == "Buy": simple_score += 10
                    simple_score += (safe_score(macd_strength_calc) - 50) * 0.2
                    if rsi is not None:
                         simple_score += (safe_score(rsi) - 50) * 0.2
                    if change_pct > 0: simple_score += min(15, change_pct * 1.0)
                    if volume is not None and volume > 0:
                        simple_score += min(10, safe_score(volume) / 500000 * 5)
                breakout_score = max(0, min(100, simple_score))
                breakout_candidates.append({
                    'ticker': ticker_scan,
                    'score': breakout_score,
                    'live_price': price,
                    'change_pct': change_pct,
                    'volume': volume,
                    'macd_signal': macd_signal_calc,
                    'macd_strength': macd_strength_calc,
                    'exchange': exchange,
                    'sector': row.get('sector', 'N/A'),
                    'industry': row.get('industry', 'N/A')
                })
            except Exception as e:
                print(f"Error processing breakout candidate {ticker_scan}: {e}")
                import traceback
                traceback.print_exc()
                continue

        breakout_candidates.sort(key=lambda x: safe_score(x.get('score', 0)), reverse=True)
        print("\n--- Breakout Scan Results (Top 10) ---")
        if not breakout_candidates:
            print("No breakout candidates found with calculated scores.")
        else:
            display_count = min(10, len(breakout_candidates))
            table_data = []
            for i in range(display_count):
                candidate = breakout_candidates[i]
                table_data.append([
                    candidate.get('ticker', 'ERR'),
                    f"{safe_score(candidate.get('score', 0)):.2f}%",
                    f"${safe_score(candidate.get('live_price', 0)):,.2f}",
                    f"{safe_score(candidate.get('change_pct', 0)):.2f}%",
                    f"{safe_score(candidate.get('volume', 0)):,.0f}",
                    candidate.get('macd_signal', 'N/A'),
                    f"{safe_score(candidate.get('macd_strength', 0)):.2f}%",
                    candidate.get('exchange', 'N/A')
                ])
            print(tabulate(table_data, headers=["Ticker", "Score", "Price", "Change %", "Volume", "MACD Signal", "MACD Strength", "Exchange"], tablefmt="pretty"))

        if save_code == "3725":
            print("Save code provided. Saving breakout data...")
            save_file = 'breakout_data.csv'
            file_exists = os.path.isfile(save_file)
            save_date_str = datetime.now(est_timezone).strftime('%m/%d/%Y')
            try:
                with open(save_file, 'a', newline='', encoding='utf-8') as f:
                    headers = ['DATE', 'TICKER', 'SCORE', 'PRICE', 'CHANGE_PCT', 'VOLUME', 'MACD_SIGNAL', 'MACD_STRENGTH', 'EXCHANGE', 'SECTOR', 'INDUSTRY']
                    writer = csv.DictWriter(f, fieldnames=headers)
                    if not file_exists or os.path.getsize(save_file) == 0:
                        writer.writeheader()
                    for candidate in breakout_candidates:
                        writer.writerow({
                            'DATE': save_date_str,
                            'TICKER': candidate.get('ticker', 'ERR'),
                            'SCORE': f"{safe_score(candidate.get('score', 0)):.2f}",
                            'PRICE': f"{safe_score(candidate.get('live_price', 0)):.2f}",
                            'CHANGE_PCT': f"{safe_score(candidate.get('change_pct', 0)):.2f}",
                            'VOLUME': f"{safe_score(candidate.get('volume', 0)):.0f}",
                            'MACD_SIGNAL': candidate.get('macd_signal', 'N/A'),
                            'MACD_STRENGTH': f"{safe_score(candidate.get('macd_strength', 0)):.2f}",
                            'EXCHANGE': candidate.get('exchange', 'N/A'),
                            'SECTOR': candidate.get('sector', 'N/A'),
                            'INDUSTRY': candidate.get('industry', 'N/A')
                        })
                print(f"Breakout data for {save_date_str} saved to '{save_file}'.")
            except IOError as e:
                print(f"Error writing breakout data to CSV {save_file}: {e}")
            except Exception as e:
                print(f"Unexpected error saving breakout data: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"Error during breakout scan: {e}")
        import traceback
        traceback.print_exc()

async def handle_assess_command(args):
    """Handles the /assess command from the terminal."""
    print("\n--- /assess Command ---")
    if not args:
        print("Usage: /assess <assess_code> [tickers=<...>] [timeframe=<...>] [risk_tolerance=<...>]")
        return

    assess_code = args[0].strip()
    tickers_str = None
    timeframe_str = None
    risk_tolerance_str = None

    for arg in args[1:]:
        arg_lower = arg.lower()
        if arg_lower.startswith("tickers="):
            tickers_str = arg[len("tickers="):].strip()
        elif arg_lower.startswith("timeframe="):
            timeframe_str = arg[len("timeframe="):].strip()
        elif arg_lower.startswith("risk_tolerance="):
            risk_tolerance_str = arg[len("risk_tolerance="):].strip()
        else:
            print(f"Warning: Unknown argument '{arg}'. Ignoring.")

    print(f"Running Assess for code: '{assess_code}'...")
    default_tickers = []
    default_timeframe = "1d"
    default_risk_tolerance = 10

    if assess_code.lower() == 'a':
        default_tickers = ['AAPL', 'MSFT', 'GOOGL']
        default_timeframe = "1d"
        default_risk_tolerance = 5
    elif assess_code.lower() == 'b':
        default_tickers = ['TSLA', 'AMZN', 'NVDA']
        default_timeframe = "1h"
        default_risk_tolerance = 15
    else:
        print(f"Warning: Unknown assess_code '{assess_code}'. Using default parameters.")
        default_tickers = ['SPY', 'QQQ']
        default_timeframe = "1d"
        default_risk_tolerance = 10

    tickers_assess = [t.strip().upper() for t in tickers_str.split(',') if t.strip()] if tickers_str is not None else default_tickers # Renamed
    timeframe = timeframe_str if timeframe_str is not None else default_timeframe
    try:
        risk_tolerance = int(risk_tolerance_str) if risk_tolerance_str is not None else default_risk_tolerance
        risk_tolerance = max(1, min(20, risk_tolerance))
    except ValueError:
        print(f"Warning: Invalid risk_tolerance value '{risk_tolerance_str}'. Using default: {default_risk_tolerance}")
        risk_tolerance = default_risk_tolerance

    if not tickers_assess:
        print("No tickers specified for assessment.")
        return

    print(f"Assessing tickers: {', '.join(tickers_assess)}")
    print(f"Timeframe: {timeframe}")
    print(f"Risk Tolerance: {risk_tolerance}")

    assessment_results = []
    print("Calculating assessment scores...")
    for ticker_as in tickers_assess: # Renamed
        try:
            # Use the global session for yfinance
            stock = yf.Ticker(ticker_as, session=YFINANCE_SESSION)
            period_map = {"1d": "1y", "1h": "6mo", "1wk": "5y"}
            period = period_map.get(timeframe, "1y")
            hist = stock.history(period=period, interval=timeframe)
            if hist.empty or 'Close' not in hist.columns:
                print(f"Warning: Could not fetch data for {ticker_as} with timeframe {timeframe}. Skipping.")
                assessment_results.append({'ticker': ticker_as, 'error': 'Data fetch failed'})
                continue

            latest_price = safe_score(hist['Close'].iloc[-1])
            assess_score = 50.0
            ema50 = None
            if len(hist) >= 50:
                 ema50 = safe_score(hist['Close'].ewm(span=50, adjust=False).mean().iloc[-1])
                 if ema50 > 0 and latest_price > 0:
                      price_vs_ema50_pct = ((latest_price - ema50) / ema50) * 100
                      assess_score += price_vs_ema50_pct * 0.5
            rsi = None
            if len(hist) >= 14:
                 delta = hist['Close'].diff()
                 gain = delta.where(delta > 0, 0)
                 loss = -delta.where(delta < 0, 0)
                 avg_gain = gain.ewm(span=14, adjust=False).mean()
                 avg_loss = loss.ewm(span=14, adjust=False).mean()
                 rs = avg_gain / avg_loss
                 rsi_series = 100 - (100 / (1 + rs)) # Keep as series first
                 rsi = safe_score(rsi_series.iloc[-1]) if not rsi_series.empty else None # Get latest RSI
                 if rsi is not None:
                      assess_score += (rsi - 50) * 0.3
            assess_score += (risk_tolerance - 10) * 0.5
            assess_score = max(0, min(100, assess_score))
            assessment_results.append({
                'ticker': ticker_as,
                'score': assess_score,
                'live_price': latest_price,
                'timeframe': timeframe,
                'risk_tolerance': risk_tolerance
            })
        except Exception as e:
            print(f"Error assessing ticker {ticker_as}: {e}")
            import traceback
            traceback.print_exc()
            assessment_results.append({'ticker': ticker_as, 'error': str(e)})
            continue

    assessment_results.sort(key=lambda x: safe_score(x.get('score', 0)), reverse=True)
    print("\n--- Assessment Results ---")
    if not assessment_results:
        print("No assessment results available.")
    else:
        table_data = []
        for result in assessment_results:
            if 'error' in result:
                table_data.append([result['ticker'], 'Error', result['error'], '-', '-'])
            else:
                table_data.append([
                    result.get('ticker', 'ERR'),
                    f"{safe_score(result.get('score', 0)):.2f}%",
                    f"${safe_score(result.get('live_price', 0)):,.2f}",
                    result.get('timeframe', 'N/A'),
                    result.get('risk_tolerance', 'N/A')
                ])
        print(tabulate(table_data, headers=["Ticker", "Score", "Price", "Timeframe", "Risk Tolerance"], tablefmt="pretty"))
    print(f"Assessment for code '{assess_code}' complete.")

async def handle_cultivate_command(args):
    """Handles the /cultivate command from the terminal."""
    print("\n--- /cultivate Command ---")
    if len(args) < 3:
        print("Usage: /cultivate <portfolio_value> <frac_shares=True/False> <cultivate_code=A/B> [save_code=3725]")
        return

    try:
        portfolio_value = float(args[0].strip())
        if portfolio_value <= 0:
            print("Error: portfolio_value must be positive.")
            return
    except ValueError:
        print("Error: Invalid portfolio_value format. Please enter a number.")
        return

    frac_shares_str = args[1].strip().lower()
    if frac_shares_str not in ['true', 'false']:
        print("Error: Invalid frac_shares value. Please enter 'True' or 'False'.")
        return
    frac_shares = frac_shares_str == 'true'
    cultivate_code = args[2].strip().upper()
    save_code = None
    if len(args) > 3:
        if args[3].lower().startswith("save_code="):
            save_code = args[3][len("save_code="):].strip()
        else:
            print(f"Warning: Unknown argument '{args[3]}'. Ignoring.")

    print(f"Running Cultivate for code: '{cultivate_code}' with value ${portfolio_value:,.2f}...")
    print(f"Fractional Shares: {frac_shares}")

    source_file = None
    min_score_threshold = 70
    num_top_tickers = 10
    score_column_name = None

    if cultivate_code == 'A':
        source_file = 'breakout_data.csv'
        min_score_threshold = 75
        num_top_tickers = 15
        score_column_name = 'SCORE'
    elif cultivate_code == 'B':
        source_file = 'market_data.csv' # This file structure might be different
        min_score_threshold = 60
        num_top_tickers = 20
        score_column_name = 'AVG_SCORE' # Assuming this is the score to use
    else:
        print(f"Error: Unknown cultivate_code '{cultivate_code}'. Cannot proceed.")
        return

    if not source_file or not os.path.exists(source_file):
        print(f"Error: Source data file '{source_file}' not found for cultivate code '{cultivate_code}'.")
        return
    if not score_column_name:
         print(f"Error: Score column name not defined for cultivate code '{cultivate_code}'. Cannot proceed.")
         return
    print(f"Reading data from: '{source_file}'")

    try:
        df = pd.read_csv(source_file)
        if df.empty:
            print(f"No data found in '{source_file}'.")
            return

        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df.dropna(subset=['DATE'], inplace=True)
            if df.empty:
                 print(f"No valid dates found in '{source_file}'.")
                 return
            latest_date = df['DATE'].max()
            df_latest = df[df['DATE'] == latest_date].copy()
            print(f"Using data from latest date: {latest_date.strftime('%m/%d/%Y')}")
        else:
            print("Warning: 'DATE' column not found in source file. Using all data.")
            df_latest = df.copy()

        if df_latest.empty:
            print(f"No data found for the latest date in '{source_file}' or file was empty.")
            return
        if score_column_name not in df_latest.columns:
             print(f"Error: Score column '{score_column_name}' not found in '{source_file}'.")
             return
        df_latest[score_column_name] = pd.to_numeric(df_latest[score_column_name], errors='coerce')
        df_latest.dropna(subset=[score_column_name], inplace=True)
        if df_latest.empty:
            print(f"No valid numeric scores found in '{source_file}'.")
            return

        df_filtered = df_latest[df_latest[score_column_name] >= min_score_threshold].copy()
        if df_filtered.empty:
            print(f"No tickers found with a score >= {min_score_threshold}% in '{source_file}'.")
            return
        df_sorted = df_filtered.sort_values(by=score_column_name, ascending=False).head(num_top_tickers).copy()
        if df_sorted.empty:
            print(f"No tickers remaining after selecting top {num_top_tickers}.")
            return

        print(f"Selected {len(df_sorted)} tickers with score >= {min_score_threshold}% (Top {num_top_tickers}):")
        # Ensure 'TICKER' column exists before trying to display it
        if 'TICKER' not in df_sorted.columns:
            print(f"Error: 'TICKER' column not found in the selected data from '{source_file}'. Cannot proceed with allocation.")
            return
        print(df_sorted[['TICKER', score_column_name]].to_string(index=False))

        print("\nFetching live prices and calculating allocations...")
        cultivate_allocations = []
        total_allocated_value = 0.0
        tickers_to_fetch = df_sorted['TICKER'].tolist()
        if not tickers_to_fetch:
             print("No tickers selected for allocation.")
             return

        live_prices = {}
        try:
            # Use yf.download with the global session
            live_prices_data = yf.download(tickers_to_fetch, period="1d", interval="1m", progress=False, session=YFINANCE_SESSION)
            if live_prices_data.empty or 'Close' not in live_prices_data.columns:
                 print("Warning: Could not fetch live prices for selected tickers.")
                 if 'PRICE' in df_sorted.columns: # Fallback to PRICE column
                      print("Using 'PRICE' column from source file as fallback.")
                      live_prices = df_sorted.set_index('TICKER')['PRICE'].apply(safe_score).to_dict()
                 else:
                      print("No 'PRICE' column in source file for fallback. Cannot calculate allocations.")
                      return # Critical if no prices
            else:
                 if isinstance(live_prices_data.columns, pd.MultiIndex) and 'Close' in live_prices_data.columns.levels[0]:
                     live_prices = live_prices_data['Close'].iloc[-1].apply(safe_score).to_dict()
                 elif not isinstance(live_prices_data.columns, pd.MultiIndex) and 'Close' in live_prices_data.columns : # Single ticker
                     live_prices = {tickers_to_fetch[0]: safe_score(live_prices_data['Close'].iloc[-1])}
                 else: # Fallback if structure is unexpected
                     print("Warning: Unexpected structure for live price data. Trying fallback.")
                     if 'PRICE' in df_sorted.columns:
                         live_prices = df_sorted.set_index('TICKER')['PRICE'].apply(safe_score).to_dict()
                     else:
                         print("No 'PRICE' column for fallback. Cannot proceed.")
                         return
        except Exception as e:
            print(f"Error fetching live prices: {e}. Attempting fallback to 'PRICE' column if available.")
            import traceback
            traceback.print_exc()
            if 'PRICE' in df_sorted.columns:
                print("Using 'PRICE' column from source file as fallback due to live price fetch error.")
                live_prices = df_sorted.set_index('TICKER')['PRICE'].apply(safe_score).to_dict()
            else:
                print("No 'PRICE' column for fallback after live price fetch error. Cannot calculate allocations.")
                return


        total_score_of_selected = safe_score(df_sorted[score_column_name].sum())
        if total_score_of_selected <= 0:
            print("Error: Total score of selected tickers is zero or negative. Cannot calculate weighted allocation.")
            return

        for index, row in df_sorted.iterrows():
            ticker_cult = row['TICKER'] # Renamed
            score = safe_score(row[score_column_name])
            live_price = safe_score(live_prices.get(ticker_cult, 0.0))
            if live_price <= 0:
                print(f"Warning: Live price for {ticker_cult} is zero or not available ({live_price}). Skipping allocation.")
                continue

            weighted_percent_allocation = (score / total_score_of_selected) * 100.0
            target_allocation_value = portfolio_value * (weighted_percent_allocation / 100.0)
            shares = 0.0
            try:
                exact_shares = target_allocation_value / live_price
                if frac_shares:
                    shares = round(exact_shares, 4)
                else:
                    shares = float(math.floor(exact_shares))
            except ZeroDivisionError:
                shares = 0.0
            except Exception:
                shares = 0.0
            shares = max(0.0, shares)
            actual_money_allocation = shares * live_price
            share_threshold = 0.0001 if frac_shares else 1.0

            if shares >= share_threshold:
                cultivate_allocations.append({
                    'ticker': ticker_cult,
                    'score': score,
                    'live_price': live_price,
                    'shares': shares,
                    'actual_money_allocation': actual_money_allocation,
                    'weighted_percent_allocation': weighted_percent_allocation,
                    'actual_percent_allocation': (actual_money_allocation / portfolio_value) * 100.0 if portfolio_value > 0 else 0.0
                })
                total_allocated_value += actual_money_allocation

        remaining_cash = portfolio_value - total_allocated_value
        remaining_cash_percent = (remaining_cash / portfolio_value) * 100.0 if portfolio_value > 0 else 0.0
        cultivate_allocations.sort(key=lambda x: safe_score(x.get('score', 0)), reverse=True)

        print("\n--- Cultivate Allocation Results ---")
        if not cultivate_allocations:
            print("No tickers met the criteria for allocation.")
        else:
            table_data = [
                [item['ticker'], f"{safe_score(item['score']):.2f}%", f"${safe_score(item['live_price']):,.2f}",
                 f"{item['shares']:.4f}" if frac_shares and item['shares'] > 0 else f"{int(item['shares'])}",
                 f"${safe_score(item['actual_money_allocation']):,.2f}",
                 f"{safe_score(item['actual_percent_allocation']):.2f}%"]
                for item in cultivate_allocations
            ]
            table_data.append(['Cash', '-', '-', '-', f"${safe_score(remaining_cash):,.2f}", f"{safe_score(remaining_cash_percent):.2f}%"])
            print(tabulate(table_data, headers=["Ticker", "Score", "Price", "Shares", "Actual $ Alloc", "Actual % Alloc"], tablefmt="pretty"))
        print(f"Cultivate analysis for code '{cultivate_code}' complete.")

        if save_code == "3725":
            print("Save code provided. Saving cultivate data...")
            save_file = f"cultivate_code_{cultivate_code}_data.csv"
            file_exists = os.path.isfile(save_file)
            save_date_str = datetime.now(est_timezone).strftime('%m/%d/%Y')
            try:
                with open(save_file, 'a', newline='', encoding='utf-8') as f:
                    headers = ['DATE', 'TICKER', 'SCORE', 'PRICE', 'SHARES', 'ACTUAL_DOLLAR_ALLOC', 'ACTUAL_PERCENT_ALLOC']
                    writer = csv.DictWriter(f, fieldnames=headers)
                    if not file_exists or os.path.getsize(save_file) == 0:
                        writer.writeheader()
                    for item in cultivate_allocations:
                        writer.writerow({
                            'DATE': save_date_str,
                            'TICKER': item.get('ticker', 'ERR'),
                            'SCORE': f"{safe_score(item.get('score', 0)):.2f}",
                            'PRICE': f"{safe_score(item.get('live_price', 0)):.2f}",
                            'SHARES': f"{item.get('shares', 0):.4f}" if frac_shares else f"{int(item.get('shares', 0))}",
                            'ACTUAL_DOLLAR_ALLOC': f"{safe_score(item.get('actual_money_allocation', 0)):.2f}",
                            'ACTUAL_PERCENT_ALLOC': f"{safe_score(item.get('actual_percent_allocation', 0)):.2f}"
                        })
                    writer.writerow({
                        'DATE': save_date_str,
                        'TICKER': 'Cash',
                        'SCORE': '-', 'PRICE': '-', 'SHARES': '-',
                        'ACTUAL_DOLLAR_ALLOC': f"{safe_score(remaining_cash):.2f}",
                        'ACTUAL_PERCENT_ALLOC': f"{safe_score(remaining_cash_percent):.2f}"
                    })
                print(f"Cultivate data for {save_date_str} saved to '{save_file}'.")
            except IOError as e:
                print(f"Error writing cultivate data to CSV {save_file}: {e}")
            except Exception as e:
                print(f"Unexpected error saving cultivate data: {e}")
                import traceback
                traceback.print_exc()
    except Exception as e:
        print(f"Error during cultivate process: {e}")
        import traceback
        traceback.print_exc()

# --- Main Terminal Loop ---
async def main():
    """Main asynchronous function to run the terminal application."""
    run_startup_sequence()

    while True:
        try:
            command_line = input("\nEnter command: ").strip()
            if not command_line:
                continue

            parts = command_line.split()
            if not parts:
                continue

            command = parts[0].lower()
            args = parts[1:]

            if command == '/exit':
                print("Exiting Market Insights Center Singularity. Goodbye!")
                break
            elif command == '/custom':
                await handle_custom_command(args)
            elif command == '/invest':
                await handle_invest_command(args)
            elif command == '/market':
                 await handle_market_command(args)
            elif command == '/breakout':
                 await handle_breakout_command(args)
            elif command == '/assess':
                 await handle_assess_command(args)
            elif command == '/cultivate':
                 await handle_cultivate_command(args)
            else:
                print(f"Unknown command: {command}. Type '/exit' to quit or see the list of available commands above.")
        except EOFError:
            print("\nExiting due to end of input.")
            break
        except KeyboardInterrupt:
            print("\nOperation cancelled. Enter '/exit' to quit.")
        except Exception as e:
            print(f"\nAn unexpected error occurred in the main loop: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())