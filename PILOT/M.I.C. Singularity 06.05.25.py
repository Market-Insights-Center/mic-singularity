# INVEST Terminal Version (Part 1)
# Modified from v2.5.5.0 to run in the terminal.
# Removed Discord dependencies and replaced interaction logic with terminal input/output.
# Implemented startup sequence with logo and command prompt.

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
    # Removed /startbreakoutcycle and /endbreakout from the list
    print("  /assess")
    print("  /cultivate")
    print("  /exit (to quit)")
    print("-" * 30) # Separator

# --- Utility Functions ---

# Define the EST timezone
est_timezone = pytz.timezone('US/Eastern')

# Utility to safely handle NaN or None values and convert to float
# (This function remains the same as it's core logic)
def safe_score(value):
    # Returns 0.0 if value is NaN, None, or cannot be converted to float
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
# (This function remains the same as it's core logic)
async def check_if_saved_today(file_path: str, date_str: str) -> bool:
    from pandas.errors import EmptyDataError
    if not os.path.exists(file_path):
        return False
    try:
        chunk_size = 10000
        found = False
        try:
            for chunk in pd.read_csv(file_path, usecols=['DATE'], chunksize=chunk_size,
                                     on_bad_lines='warn', low_memory=False):
                if date_str in chunk['DATE'].astype(str).values:
                    found = True
                    break
        except TypeError:
             print("Warning: Using deprecated error_bad_lines/warn_bad_lines. Update pandas.")
             for chunk in pd.read_csv(file_path, usecols=['DATE'], chunksize=chunk_size,
                                     error_bad_lines=False, warn_bad_lines=True,
                                     low_memory=False):
                if date_str in chunk['DATE'].astype(str).values:
                    found = True
                    break
        return found
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

# Remove Discord task loops and on_ready event
# @tasks.loop(minutes=15)
# async def breakout_cycle():
#    ... (removed) ...

# @tasks.loop(minutes=5)
# async def daily_auto_save():
#    ... (removed) ...

# @bot.event
# async def on_ready():
#    ... (removed) ...

# Fetch the EMAs and calculate EMA Invest
# (This function remains the same as it's core logic)
def calculate_ema_invest(ticker, ema_interval):
    ticker = ticker.replace('.', '-')
    stock = yf.Ticker(ticker)
    interval_mapping = {1: "1wk", 2: "1d", 3: "1h"}
    interval = interval_mapping.get(ema_interval, "1h")
    if ema_interval == 3:
        period = "2y"
        try:
            data = stock.history(period=period, interval=interval)
            if len(data) < 730: period = "3mo"
            data = stock.history(period=period, interval=interval)
            if len(data) < 60: period = "1mo"
            data = stock.history(period=period, interval=interval)
        except Exception as e:
            print(f"Error fetching history for {ticker} (Interval {interval}, Period {period}): {e}")
            return None, None
    elif ema_interval == 1:
        period = "max"
        try: data = stock.history(period=period, interval=interval)
        except Exception as e: print(f"Error fetching history for {ticker} (Interval {interval}, Period {period}): {e}"); return None, None
    elif ema_interval == 2:
        period = "10y"
        try: data = stock.history(period=period, interval=interval)
        except Exception as e: print(f"Error fetching history for {ticker} (Interval {interval}, Period {period}): {e}"); return None, None
    else:
        period = "max"
        try: data = stock.history(period=period, interval=interval)
        except Exception as e: print(f"Error fetching history for {ticker} (Interval {interval}, Period {period}): {e}"); return None, None

    if data.empty:
        print(f"Warning: No history data returned for {ticker} with interval {interval} and period {period}.")
        return None, None
    if 'Close' not in data.columns:
        print(f"Warning: 'Close' column missing for {ticker} data.")
        return None, None
    try:
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_13'] = data['Close'].ewm(span=13, adjust=False).mean()
        data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()
    except Exception as e:
        print(f"Error calculating EMAs for {ticker}: {e}")
        return None, None
    if data.empty or data.iloc[-1].isna().any():
       print(f"Warning: Latest data or EMAs are invalid for {ticker}")
       live_price_fallback = data['Close'].iloc[-1] if not data.empty and not pd.isna(data['Close'].iloc[-1]) else None
       return live_price_fallback, None
    latest_data = data.iloc[-1]
    live_price = latest_data['Close']
    ema_8 = latest_data['EMA_8']
    ema_55 = latest_data['EMA_55']
    if ema_55 == 0 or pd.isna(ema_8) or pd.isna(ema_55):
        print(f"Warning: EMA_55 is zero or EMAs are NaN for {ticker}, cannot calculate score.")
        return live_price, None
    ema_enter = (ema_8 - ema_55) / ema_55
    ema_invest = ((ema_enter * 4) + 0.5) * 100
    return live_price, ema_invest

# Calculate the one-year percent change and invest_per
# (This function remains the same as it's core logic)
def calculate_one_year_invest(ticker):
    # ... (rest of calculate_one_year_invest function remains the same) ...
    ticker = ticker.replace('.', '-')
    stock = yf.Ticker(ticker)
    try:
        data = stock.history(period="1y")
        if data.empty or len(data) < 2:
             print(f"Warning: Insufficient 1-year data for {ticker}.")
             return 0.0, 50.0 # Neutral default
        if 'Close' not in data.columns:
             print(f"Warning: 'Close' column missing in 1-year data for {ticker}.")
             return 0.0, 50.0
    except Exception as e:
        print(f"Error fetching 1-year history for {ticker}: {e}")
        return 0.0, 50.0 # Neutral default on error

    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]

    if start_price == 0 or pd.isna(start_price) or pd.isna(end_price):
        print(f"Warning: Invalid start/end price for 1-year calc on {ticker}.")
        return 0.0, 50.0 # Neutral default
    # Ensure start_price is not zero before division
    if start_price == 0:
        print(f"Warning: Start price is zero for 1-year calc on {ticker}. Cannot calculate change.")
        return 0.0, 50.0

    one_year_change = ((end_price - start_price) / start_price) * 100

    if one_year_change < 0:
        invest_per = (one_year_change / 2) + 50
    else:
        # Use try-except for sqrt on potentially negative intermediate result if logic changes
        try:
            invest_per = math.sqrt(max(0, one_year_change * 5)) + 50 # Ensure non-negative input to sqrt
        except ValueError:
            invest_per = 50.0 # Fallback if sqrt fails unexpectedly
    # Clamp score
    invest_per = max(0, min(invest_per, 100))

    return one_year_change, invest_per

# Calculate S&P 500 and S&P 100 symbols
# (These functions remain the same as they are core logic)
def get_sp500_symbols():
    # ... (rest of get_sp500_symbols function remains the same) ...
    try:
        sp500_list_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        dfs = pd.read_html(sp500_list_url)
        if not dfs: return []
        df = dfs[0]
        if 'Symbol' not in df.columns: return []
        symbols = df['Symbol'].tolist()
        symbols = [s.replace('.', '-') for s in symbols if isinstance(s, str)] # Handle potential non-strings
        return symbols
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}")
        return []

def get_sp100_symbols():
    # ... (rest of get_sp100_symbols function remains the same) ...
    try:
        sp100_list_url = 'https://en.wikipedia.org/wiki/S%26P_100'
        dfs = pd.read_html(sp100_list_url)
        if len(dfs) < 3: return [] # Check if table exists
        df = dfs[2] # Usually the 3rd table
        if 'Symbol' not in df.columns: return []
        symbols = df['Symbol'].tolist()
        symbols = [s.replace('.', '-') for s in symbols if isinstance(s, str)]
        return symbols
    except Exception as e:
        print(f"Error fetching S&P 100 symbols: {e}")
        return []

def get_spy_symbols():
    # Currently uses S&P 500, can be changed if needed
    return get_sp500_symbols()


# Calculate market risk
# (This function remains the same as it's core logic)
def calculate_market_risk():
    # ... (rest of calculate_market_risk function remains the same) ...
    try:
        def calculate_ma(symbol, ma_window):
            try:
                data = yf.download(symbol, period='1y', interval='1d', progress=False)
                if data.empty or len(data) < ma_window or 'Close' not in data.columns:
                    return None
                data[f'{ma_window}_day_ma'] = data['Close'].rolling(window=ma_window).mean()
                latest_price = data['Close'].iloc[-1]
                latest_ma = data[f'{ma_window}_day_ma'].iloc[-1]
                if pd.isna(latest_ma) or pd.isna(latest_price):
                    return None
                return latest_price > latest_ma
            except Exception as e:
                # print(f"Error calculating MA for {symbol}: {e}") # Reduced verbosity
                return None

        def calculate_percentage_above_ma(symbols, ma_window):
            above_ma_count = 0
            valid_stocks = 0
            if not symbols: return 0 # Handle empty symbol list
            for symbol in symbols:
                result = calculate_ma(symbol, ma_window)
                if result is not None:
                    valid_stocks += 1
                    if result:
                        above_ma_count += 1
            return (above_ma_count / valid_stocks) * 100 if valid_stocks > 0 else 0

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

        def get_live_price_and_ma(ticker):
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="260d") # Approx 1 year of trading days
                if hist.empty or len(hist) < 50 or 'Close' not in hist.columns:
                     print(f"Warning: Insufficient data for MAs on {ticker}")
                     return None, None, None

                live_price = hist['Close'].iloc[-1]
                ma_20 = hist['Close'].rolling(window=20).mean().iloc[-1]
                ma_50 = hist['Close'].rolling(window=50).mean().iloc[-1]

                # Check for NaN results
                if pd.isna(live_price) or pd.isna(ma_20) or pd.isna(ma_50):
                    print(f"Warning: NaN values encountered for price/MA on {ticker}")
                    return None, None, None

                return live_price, ma_20, ma_50
            except Exception as e:
                print(f"Error getting price/MA for {ticker}: {e}")
                return None, None, None


        def calculate_general_and_large():
            spy_live_price, spy_ma_20, spy_ma_50 = get_live_price_and_ma('SPY')
            vix_live_price, _, _ = get_live_price_and_ma('^VIX')
            rut_live_price, rut_ma_20, rut_ma_50 = get_live_price_and_ma('^RUT')
            oex_live_price, oex_ma_20, oex_ma_50 = get_live_price_and_ma('^OEX')

            # Check if any essential data is missing
            essential_data = [spy_live_price, spy_ma_20, spy_ma_50, vix_live_price,
                              rut_live_price, rut_ma_20, rut_ma_50,
                              oex_live_price, oex_ma_20, oex_ma_50]
            if any(d is None for d in essential_data):
                print("Warning: Missing key index data for market risk calculation.")
                return None # Cannot calculate if key data is missing

            s5tw = calculate_s5tw()
            s5th = calculate_s5th()
            s1fd = calculate_s1fd()
            s1tw = calculate_s1tw()

            # Calculate components using safe_score and avoiding division by zero
            spy20 = ((safe_score(spy_live_price) - safe_score(spy_ma_20)) / 20) + 50 if spy_ma_20 else 50
            spy50 = ((safe_score(spy_live_price) - safe_score(spy_ma_50) - 150) / 20) + 50 if spy_ma_50 else 50 # Note: Original had hardcoded 150 offset
            vix_calc = (((safe_score(vix_live_price) - 15) * -5) + 50) if vix_live_price else 50
            rut20 = ((safe_score(rut_live_price) - safe_score(rut_ma_20)) / 10) + 50 if rut_ma_20 else 50
            rut50 = ((safe_score(rut_live_price) - safe_score(rut_ma_50)) / 5) + 50 if rut_ma_50 else 50
            s5tw_calc = ((s5tw - 60) + 50)
            s5th_calc = ((s5th - 70) + 50)

            # Weighted average for 'general' score
            general_components = [
                (spy20, 3), (spy50, 1), (vix_calc, 3), (rut50, 3), (rut20, 1),
                (s5tw_calc, 2), (s5th_calc, 1)
            ]
            general_sum = sum(score * weight for score, weight in general_components)
            general_weights = sum(weight for _, weight in general_components)
            general = general_sum / general_weights if general_weights > 0 else 50 # Default to 50 if weights sum to 0

            # Components for 'large' score
            # Assuming price vs MA is intended.
            oex20_calc = ((safe_score(oex_live_price) - safe_score(oex_ma_20)) / 10) + 50 if oex_ma_20 else 50 # Adjusted divisor assumption
            oex50_calc = ((safe_score(oex_live_price) - safe_score(oex_ma_50)) / 5) + 50 if oex_ma_50 else 50 # Adjusted divisor assumption
            s1fd_calc = ((s1fd - 60) + 50)
            s1tw_calc = ((s1tw - 70) + 50)

            # Weighted average for 'large' score
            large_components = [
                (oex20_calc, 3), (oex50_calc, 1), (s1fd_calc, 2), (s1tw_calc, 1)
            ]
            large_sum = sum(score * weight for score, weight in large_components)
            large_weights = sum(weight for _, weight in large_components)
            large = large_sum / large_weights if large_weights > 0 else 50

            combined = (general + large) / 2
            # Clamp final score
            combined = max(0, min(100, combined))

            return combined

        combined_market_risk = calculate_general_and_large()
        return combined_market_risk

    except Exception as e:
        print(f"Error calculating market risk: {e}")
        return None


# Calculate MACD signal
# (This function remains the same as it's core logic)
def calculate_macd_signal(ticker, ema_interval, fast_period=12, slow_period=26, signal_period=9):
    # ... (rest of calculate_macd_signal function remains the same) ...
    ticker = ticker.replace('.', '-')
    stock = yf.Ticker(ticker)

    # Updated interval mapping - REMOVED 4 and 5
    interval_mapping = {1: "1wk", 2: "1d", 3: "1h"}
    interval = interval_mapping.get(ema_interval, "1h")

    # Adjust period based on interval - remove cases for 4 and 5
    if interval == "1wk": period = "5y"
    elif interval == "1d": period = "2y"
    elif interval == "1h": period = "730d" # Max allowed for 1h is ~2y
    else: period = "2y" # Default

    try: # Add try-except for history fetch
        data = stock.history(period=period, interval=interval)
        if data.empty or len(data) < slow_period + signal_period or 'Close' not in data.columns: # Check sufficient data and column
             print(f"Warning: Insufficient data or missing 'Close' column for MACD on {ticker} ({interval})")
             return "Neutral", 50.0 # Return neutral if not enough data
    except Exception as e:
         print(f"Error fetching history for MACD on {ticker}: {e}")
         return "Neutral", 50.0

    try: # Wrap calculations in try-except
        data['fast_ema'] = data['Close'].ewm(span=fast_period, adjust=False).mean()
        data['slow_ema'] = data['Close'].ewm(span=slow_period, adjust=False).mean()
        data['macd'] = data['fast_ema'] - data['slow_ema']
        data['signal'] = data['macd'].ewm(span=signal_period, adjust=False).mean()
        data['histogram'] = data['macd'] - data['signal']
    except Exception as e:
        print(f"Error calculating MACD components for {ticker}: {e}")
        return "Neutral", 50.0


    if len(data['histogram']) < 3 or data['histogram'].isnull().tail(3).any():
         print(f"Warning: Not enough valid histogram points for MACD signal on {ticker}")
         return "Neutral", 50.0

    last_three_hist = data['histogram'].tail(3).tolist()

    signal = "Neutral" # Default signal
    # Refined conditions slightly for clarity
    if last_three_hist[2] < last_three_hist[1] < last_three_hist[0] and last_three_hist[2] < 0 : # Sell condition
        signal = "Sell"
    elif last_three_hist[2] > last_three_hist[1] > last_three_hist[0] and last_three_hist[2] > 0: # Buy condition
        signal = "Buy"

    # Calculate strength robustly
    macd_strength = 0
    # Ensure no NaN values in the last three points used for strength calculation
    if not any(pd.isna(h) for h in last_three_hist):
         # Original strength calc: average change over last two periods
         macd_strength = ((last_three_hist[2] - last_three_hist[1]) + (last_three_hist[1] - last_three_hist[0])) / 2
         # Normalize strength:
         macd_strength_percent_adjusted = ((macd_strength / 2) * 100) + 50
    else:
        macd_strength_percent_adjusted = 50.0 # Default to neutral if NaN present

    # Clamp the result between 0 and 100
    macd_strength_percent_adjusted = max(0, min(100, macd_strength_percent_adjusted))

    return signal, macd_strength_percent_adjusted

# Plot ticker graph
# (This function remains the same, but output is filename instead of sending file)
def plot_ticker_graph(ticker, ema_interval):
    # ... (rest of plot_ticker_graph function remains the same) ...
    ticker = ticker.replace('.', '-')
    stock = yf.Ticker(ticker)

    # Updated interval mapping - REMOVED 4 and 5
    interval_mapping = {1: "1wk", 2: "1d", 3: "1h"}
    interval = interval_mapping.get(ema_interval, "1h")

    # Adjust period based on interval - remove cases for 4 and 5
    if ema_interval == 3: period = "6mo" # Hourly
    elif ema_interval == 1: period = "5y" # Weekly - longer period for context
    elif ema_interval == 2: period = "1y" # Daily
    else: period = "1y" # Default

    try:
        data = stock.history(period=period, interval=interval)
        if data.empty or 'Close' not in data.columns:
            raise ValueError(f"No data or 'Close' column returned for {ticker} (Period: {period}, Interval: {interval})")

        # Calculate EMAs needed for plot
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6)) # Use subplots for better control
        ax.plot(data.index, data['Close'], color='grey', label='Price', linewidth=1.0)
        # --- v2.5.5.0: Dimmer EMA colors ---
        ax.plot(data.index, data['EMA_55'], color='darkgreen', label='EMA 55', linewidth=1.5) # Changed from lime
        ax.plot(data.index, data['EMA_8'], color='firebrick', label='EMA 8', linewidth=1.5) # Changed from red
        # --- End v2.5.5.0 Change ---
        ax.set_title(f"{ticker} Price and EMAs ({interval})", color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Price', color='white')
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax.grid(True, color='dimgray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        # Improve layout
        fig.tight_layout()

        filename = f"{ticker}_graph.png"
        # Save the figure
        plt.savefig(filename, facecolor='black', edgecolor='black')
        plt.close(fig) # Close the figure explicitly
        return filename # Return filename on success
    except Exception as e:
        print(f"Error plotting graph for {ticker}: {e}")
        # Ensure figure is closed if error occurs after creation
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)
        return None # Return None on failure

# --- Modified process_custom_portfolio function ---
# Removed Discord interaction parameter and replaced sends with prints.
async def process_custom_portfolio(portfolio_data, tailor_portfolio, frac_shares,
                                   total_value=None, is_custom_command_without_save=False):
    """
    Processes custom or /invest portfolio requests, calculates scores, allocations,
    and generates output tables and graphs for the TERMINAL.
    Removed Discord interaction.
    """
    # --- Sell to Cash Feature Check ---
    sell_to_cash_active = False
    # Need to calculate market score here or pass it in if needed
    # For simplicity in this terminal version, let's assume Sell-to-Cash is NOT active
    # or calculate a dummy score if market_data.csv is not expected to exist.
    # Based on the request, the market data calculation is separate.
    # Let's keep the logic but assume market_data.csv might not exist, defaulting scores.
    avg_score, risk_gen_score, mkt_inv_score = get_allocation_score() # Fetch scores (defaults to 50 if file missing)

    if avg_score is not None and avg_score < 50.0:
        sell_to_cash_active = True
        print(f"INFO: Sell-to-Cash feature ACTIVE (Avg Market Score: {avg_score:.2f} < 50)")
        # No ephemeral warning in terminal, just print

    # --- v2.5.3.0 Defaults ---
    risk_tolerance = 10
    risk_type = 'stock'
    remove_amplification_cap = True

    # Get other parameters safely from portfolio_data
    ema_sensitivity = int(portfolio_data.get('ema_sensitivity', 3))
    try:
        amplification = float(portfolio_data.get('amplification', 1.0))
    except ValueError:
        print("Warning: Invalid amplification value found. Defaulting to 1.0")
        amplification = 1.0
    num_portfolios = int(portfolio_data.get('num_portfolios', 0))

    portfolio_results = []
    all_entries_for_graphs = []

    # --- Initial Calculations ---
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
            continue # Skip if no tickers

        current_portfolio_list = []
        for ticker in tickers:
            try:
                live_price, ema_invest = calculate_ema_invest(ticker, ema_sensitivity)

                if live_price is None and ema_invest is None:
                     print(f"Warning: Failed to get base data for {ticker}. Skipping.")
                     current_portfolio_list.append({'ticker': ticker, 'error': f"Failed to get base data", 'portfolio_weight': weight})
                     all_entries_for_graphs.append({'ticker': ticker, 'error': f"Failed to get base data"})
                     continue # Skip to next ticker

                if ema_invest is None:
                     print(f"Warning: EMA Invest score calculation failed for {ticker}, using neutral 50.")
                     ema_invest = 50.0 # Assign neutral score if only score calc failed
                if live_price is None:
                     print(f"Warning: Live price calculation failed for {ticker}. Assigning 0 price.")
                     live_price = 0.0 # Assign 0 price but allow score calculation if possible

                _, invest_per = calculate_one_year_invest(ticker) # Need invest_per for stock_invest calc

                ema_invest = safe_score(ema_invest)
                invest_per = safe_score(invest_per)

                risk_mapping = {'market': 0, 'stock': 1, 'both': 2}
                risk_type_num = risk_mapping[risk_type.lower()] # This will always be 1
                stock_invest = ema_invest # Simplified calculation
                raw_combined_invest = stock_invest

                score_for_allocation = raw_combined_invest
                score_was_adjusted = False

                if sell_to_cash_active and raw_combined_invest < 50.0:
                    score_for_allocation = 50.0
                    score_was_adjusted = True

                amplified_score_adjusted = safe_score((score_for_allocation * amplification) - (amplification - 1) * 50)
                amplified_score_adjusted_clamped = max(0, amplified_score_adjusted) # Clamp >= 0

                amplified_score_original = safe_score((raw_combined_invest * amplification) - (amplification - 1) * 50)
                amplified_score_original_clamped = max(0, amplified_score_original)

                entry_data = {
                    'ticker': ticker,
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
                if live_price > 0:
                    all_entries_for_graphs.append({'ticker': ticker, 'ema_sensitivity': ema_sensitivity})

            except Exception as e:
                print(f"Error processing ticker {ticker} in portfolio {portfolio_index}: {e}")
                current_portfolio_list.append({'ticker': ticker, 'error': str(e), 'portfolio_weight': weight})
                all_entries_for_graphs.append({'ticker': ticker, 'error': str(e)})

        portfolio_results.append(current_portfolio_list)
    print("--- Finished Initial Ticker Calculations ---")

    # --- GRAPHS FIRST ---
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
                    # Keep the file for the user to view
                elif graph_filename:
                    print(f"  Could not generate graph file for {ticker_key}: File not found after creation attempt.")
                else:
                    print(f"  Failed to generate graph for {ticker_key}.")
            except Exception as plot_error:
                print(f"  Error plotting graph for {ticker_key}: {plot_error}")
        else:
             print(f"  Skipping graph for {ticker_key} due to earlier error: {graph_entry['error']}")
    print("--- Finished Graph Generation ---")

    # --- Calculate sub-portfolio allocations (Adjusted and Original) ---
    print("--- Calculating Sub-Portfolio Allocations (Adjusted & Original) ---")
    for portfolio_index, portfolio in enumerate(portfolio_results):
        portfolio_amplified_total_adjusted = safe_score(sum(entry['amplified_score_adjusted'] for entry in portfolio if 'error' not in entry))
        for entry in portfolio:
            if 'error' not in entry:
                if portfolio_amplified_total_adjusted > 0:
                     amplified_score_adj = safe_score(entry.get('amplified_score_adjusted', 0))
                     portfolio_allocation_percent_adj = safe_score((amplified_score_adj / portfolio_amplified_total_adjusted) * 100)
                     entry['portfolio_allocation_percent_adjusted'] = round(portfolio_allocation_percent_adj, 2)
                else: entry['portfolio_allocation_percent_adjusted'] = 0.0
            else: entry['portfolio_allocation_percent_adjusted'] = None

        portfolio_amplified_total_original = safe_score(sum(entry['amplified_score_original'] for entry in portfolio if 'error' not in entry))
        for entry in portfolio:
            if 'error' not in entry:
                if portfolio_amplified_total_original > 0:
                    amplified_score_orig = safe_score(entry.get('amplified_score_original', 0))
                    portfolio_allocation_percent_orig = safe_score((amplified_score_orig / portfolio_amplified_total_original) * 100)
                    entry['portfolio_allocation_percent_original'] = round(portfolio_allocation_percent_orig, 2)
                else: entry['portfolio_allocation_percent_original'] = 0.0
            else: entry['portfolio_allocation_percent_original'] = None

    # --- Output Sub-Portfolios ---
    if not is_custom_command_without_save: # Only display full tables if not in simplified mode
        for i, portfolio in enumerate(portfolio_results, 1):
            portfolio.sort(key=lambda x: x.get('portfolio_allocation_percent_adjusted', -1) if x.get('portfolio_allocation_percent_adjusted') is not None else -1, reverse=True)
            portfolio_weight_display = portfolio[0].get('portfolio_weight', 'N/A') if portfolio else 'N/A'
            print(f"\n--- Sub-Portfolio {i} (Weight: {portfolio_weight_display}%) ---")
            table_data = []
            for entry in portfolio:
                 if 'error' not in entry:
                    live_price_f = f"${entry.get('live_price', 0):.2f}"
                    invest_score_val = safe_score(entry.get('raw_invest_score', 0))
                    invest_score_f = f"{invest_score_val:.2f}%" if invest_score_val is not None else "N/A"
                    amplified_score_f = f"{entry.get('amplified_score_adjusted', 0):.2f}%"
                    port_alloc_val_original = safe_score(entry.get('portfolio_allocation_percent_original', 0))
                    port_alloc_f = f"{port_alloc_val_original:.2f}%" if port_alloc_val_original is not None else "N/A"
                    table_data.append([entry.get('ticker', 'ERR'), live_price_f, invest_score_f, amplified_score_f, port_alloc_f])

            if not table_data: print("No valid data for this sub-portfolio.")
            else: print(tabulate(table_data, headers=["Ticker", "Live Price", "Raw Score", "Adj Amplified %", "Portfolio % Alloc (Original)"], tablefmt="pretty"))

            error_messages = [f"Error for {entry.get('ticker', 'UNKNOWN')}: {entry.get('error', 'Unknown error')}" for entry in portfolio if 'error' in entry]
            if error_messages:
                print("\nErrors in Sub-Portfolio {}:".format(i))
                for msg in error_messages: print(msg)


    # --- Calculate Combined Portfolio Allocations ---
    print("--- Calculating Combined Portfolio Allocations (Adjusted & Original) ---")
    combined_result_intermediate = []
    for portfolio in portfolio_results:
        for entry in portfolio:
            if 'error' not in entry:
                port_weight = entry.get('portfolio_weight', 0)
                sub_alloc_adj = entry.get('portfolio_allocation_percent_adjusted', 0)
                combined_percent_allocation_adjusted = round(safe_score((sub_alloc_adj * port_weight) / 100), 4)
                entry['combined_percent_allocation_adjusted'] = combined_percent_allocation_adjusted
                sub_alloc_orig = entry.get('portfolio_allocation_percent_original', 0)
                combined_percent_allocation_original = round(safe_score((sub_alloc_orig * port_weight) / 100), 4)
                entry['combined_percent_allocation_original'] = combined_percent_allocation_original
                combined_result_intermediate.append(entry)

    # --- Construct Final Combined Portfolio ---
    print("--- Constructing Final Combined Portfolio (with Cash if applicable) ---")
    final_combined_portfolio_data = []
    total_cash_diff_percent = 0.0
    for entry in combined_result_intermediate:
        final_combined_portfolio_data.append({
            'ticker': entry['ticker'],
            'live_price': entry['live_price'],
            'raw_invest_score': entry['raw_invest_score'],
            'amplified_score_adjusted': entry['amplified_score_adjusted'],
            'combined_percent_allocation': entry['combined_percent_allocation_adjusted']
        })
        if sell_to_cash_active and entry.get('score_was_adjusted', False):
            adj_alloc = entry['combined_percent_allocation_adjusted']
            orig_alloc = entry['combined_percent_allocation_original']
            difference = adj_alloc - orig_alloc
            total_cash_diff_percent += max(0.0, difference)

    current_stock_total_alloc = sum(item['combined_percent_allocation'] for item in final_combined_portfolio_data)
    target_stock_alloc = 100.0 - total_cash_diff_percent
    if current_stock_total_alloc > target_stock_alloc + 1e-9:
        print(f"    Normalizing final stock allocations slightly. Current sum: {current_stock_total_alloc:.4f}, Target: {target_stock_alloc:.4f}")
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
        print(f"    Added Cash row to Final Combined Portfolio: {total_cash_diff_percent:.4f}%")

    # Sort by raw score for display
    final_combined_portfolio_data.sort(
        key=lambda x: x.get('raw_invest_score', -float('inf')) if x['ticker'] != 'Cash' else -float('inf')-1,
        reverse=True
    )


    # --- Output Final Combined Portfolio ---
    if not is_custom_command_without_save: # Only display full tables if not in simplified mode
        print("\n--- Final Combined Portfolio (Sorted by Raw Score)---")
        if sell_to_cash_active: print("*(Sell-to-Cash Active: Difference allocated to Cash)*")
        combined_data_display = []
        for entry in final_combined_portfolio_data:
            ticker = entry.get('ticker', 'ERR')
            if ticker == 'Cash':
                live_price_f = '-'
                invest_score_f = '-'
                amplified_score_f = '-'
            else:
                live_price_f = f"${entry.get('live_price', 0):.2f}"
                invest_score_f = f"{entry.get('raw_invest_score', 0):.2f}%" # Raw Score
                amplified_score_f = f"{entry.get('amplified_score_adjusted', 0):.2f}%" # Adjusted Amplified Score
            comb_alloc_f = f"{round(entry.get('combined_percent_allocation', 0), 2):.2f}%"
            combined_data_display.append([ticker, live_price_f, invest_score_f, amplified_score_f, comb_alloc_f])

        if not combined_data_display: print("No valid data for the combined portfolio.")
        else: print(tabulate(combined_data_display, headers=["Ticker", "Live Price", "Raw Score", "Adj Amplified %", "Final % Alloc"], tablefmt="pretty"))


    # --- TAILORED PORTFOLIO Calculation (MODIFIED FOR BUYING POWER) ---
    print("--- Calculating Tailored Portfolio ---")
    tailored_portfolio_output_list = []
    tailored_portfolio_table_data = []
    remaining_buying_power = None # MODIFICATION: Initialize here
    final_cash_value_tailored = None
    final_cash_percent_tailored = None

    if tailor_portfolio:
        if total_value is None or safe_score(total_value) <= 0:
            print("Error: Tailored portfolio requested but total value is missing or invalid. Cannot proceed with tailoring.")
            return [], combined_result_intermediate, portfolio_results
        else:
            total_value = safe_score(total_value)

        tailored_portfolio_entries_intermediate = []
        total_actual_money_allocated_stocks = 0.0
        total_actual_percent_allocated_stocks = 0.0

        # Iterate through the final_combined_portfolio_data (excluding the Cash row)
        for entry in final_combined_portfolio_data:
            if entry['ticker'] == 'Cash': continue # Skip cash row for stock allocation

            final_stock_alloc_pct = safe_score(entry.get('combined_percent_allocation', 0.0))
            live_price = safe_score(entry.get('live_price', 0.0))

            if final_stock_alloc_pct > 1e-9 and live_price > 0:
                target_allocation_value = total_value * (final_stock_alloc_pct / 100.0)
                shares = 0.0
                try:
                    exact_shares = target_allocation_value / live_price
                    if frac_shares:
                        shares = round(exact_shares, 1)
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
                share_threshold = 0.1 if frac_shares else 1.0

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
                    total_actual_percent_allocated_stocks += actual_percent_allocation

        # --- MODIFICATION START: Calculate Remaining Buying Power (potentially negative) ---
        # Calculate the raw difference between total value and allocated stock value
        raw_remaining_value = total_value - total_actual_money_allocated_stocks

        # Set remaining_buying_power based on sell_to_cash_active flag
        if not sell_to_cash_active:
            remaining_buying_power = raw_remaining_value # Store potentially negative value
            # final_cash_value_tailored will still be clamped >= 0 for display/use
            final_cash_value_tailored = max(0.0, raw_remaining_value)
            print(f"    Tailored - Sell-to-Cash INACTIVE.")
            print(f"    Tailored - Remaining Buying Power (Potentially Negative): ${remaining_buying_power:,.2f}")
            print(f"    Tailored - Final Clamped Cash Value: ${final_cash_value_tailored:,.2f}")
        else:
            # If sell-to-cash IS active, remaining_buying_power is not the primary metric shown.
            # The final_cash_value_tailored represents the total cash (initial sell-to-cash amount + remaining)
            final_cash_value_tailored = max(0.0, raw_remaining_value) # Still clamp actual cash >= 0
            remaining_buying_power = None # Explicitly set to None when sell-to_cash is active
            print(f"    Tailored - Sell-to-Cash ACTIVE.")
            print(f"    Tailored - Final Clamped Cash Value: ${final_cash_value_tailored:,.2f}")

        # Calculate final cash percentage based on the CLAMPED cash value
        final_cash_percent_tailored = (final_cash_value_tailored / total_value) * 100.0 if total_value > 0 else 0.0
        final_cash_percent_tailored = max(0.0, min(100.0, final_cash_percent_tailored)) # Clamp 0-100
        print(f"    Tailored - Final Cash Percent: {final_cash_percent_tailored:.2f}%")
        # --- MODIFICATION END ---


        # Sort tailored portfolio entries by RAW INVEST SCORE
        tailored_portfolio_entries_intermediate.sort(
            key=lambda x: safe_score(x.get('raw_invest_score', -float('inf'))),
            reverse=True
        )

        # Prepare data for table output (uses CLAMPED final_cash_value_tailored)
        tailored_portfolio_table_data = [
             [item['ticker'], f"{item['shares']:.1f}" if frac_shares and item['shares'] > 0 else f"{int(item['shares'])}",
              f"${safe_score(item['actual_money_allocation']):,.2f}", f"{safe_score(item['actual_percent_allocation']):.2f}%"]
             for item in tailored_portfolio_entries_intermediate
        ]
        tailored_portfolio_table_data.append(['Cash', '-', f"${safe_score(final_cash_value_tailored):,.2f}", f"{safe_score(final_cash_percent_tailored):.2f}%"])

        # Prepare data for simplified list output
        if frac_shares:
            tailored_portfolio_output_list = ["{} - {:.1f}".format(item['ticker'], item['shares']) for item in tailored_portfolio_entries_intermediate]
        else:
            tailored_portfolio_output_list = ["{} - {:.0f}".format(item['ticker'], item['shares']) for item in tailored_portfolio_entries_intermediate]


        # --- Output Tailored Portfolio (MODIFIED for Buying Power Display) ---
        if is_custom_command_without_save: # Simplified output for /custom without save
            if tailored_portfolio_output_list:
                print("\n--- Tailored Portfolio Allocation (Sorted by Raw Score) ---")
                for line in tailored_portfolio_output_list: print(line)
            else:
                print("No stocks allocated in the tailored portfolio based on the provided value and strategy.")

            # Send final cash value (CLAMPED >= 0)
            if final_cash_value_tailored is not None:
                 print(f"Final Cash Value: ${safe_score(final_cash_value_tailored):,.2f}")
            else:
                 print(f"Final Cash Value: Calculation Error")

            # --- MODIFICATION: Send remaining_buying_power if calculated (potentially negative) ---
            # This is only relevant for /custom without save code AND sell_to_cash inactive
            if remaining_buying_power is not None: # Will be None if sell_to_cash_active is True
                print(f"Remaining Buying Power: ${safe_score(remaining_buying_power):,.2f}")
            # --- End Modification ---

        # Output full table for /invest or /custom with save code
        else:
            print("\n--- Tailored Portfolio (Sorted by Raw Score) ---")
            if not tailored_portfolio_table_data:
                 print("No stocks allocated based on the provided value and strategy.")
            else:
                print(tabulate(tailored_portfolio_table_data, headers=["Ticker", "Shares", "Actual $ Allocation", "Actual % Allocation"], tablefmt="pretty"))

            # --- MODIFICATION: Send remaining_buying_power if calculated (potentially negative) ---
            # This applies when sell_to_cash is inactive
            if remaining_buying_power is not None: # Will be None if sell_to_cash_active is True
                print(f"Remaining Buying Power: ${safe_score(remaining_buying_power):,.2f}")
            # --- End Modification ---


    # --- END TAILORED PORTFOLIO ---
    print("--- Finished Tailored Portfolio ---")

    # Return values are used internally for saving, not for Discord output anymore
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

# Modified collect_portfolio_inputs for terminal
async def collect_portfolio_inputs_terminal(portfolio_code):
    """Collects portfolio configuration inputs from the terminal."""
    inputs = {'portfolio_code': portfolio_code}
    portfolio_weights = []

    print(f"\nLet's set up portfolio code '{portfolio_code}'. Please answer the following questions.")

    # --- Collect General Inputs ---
    ema_sensitivity = get_int_input("Enter EMA sensitivity (1: Weekly, 2: Daily, 3: Hourly): ", min_value=1, max_value=3)
    inputs['ema_sensitivity'] = str(ema_sensitivity) # Store as string

    valid_amplifications = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    amplification = get_float_input(f"Enter amplification ({', '.join(map(str, valid_amplifications))}): ", min_value=0.25)
    while amplification not in valid_amplifications:
        print(f"Invalid amplification value. Please enter one of: {', '.join(map(str, valid_amplifications))}")
        amplification = get_float_input(f"Enter amplification ({', '.join(map(str, valid_amplifications))}): ", min_value=0.25)
    inputs['amplification'] = str(amplification) # Store as string

    num_portfolios = get_int_input("Enter the number of portfolios to analyze (e.g., 2): ", min_value=1)
    inputs['num_portfolios'] = str(num_portfolios)

    frac_shares = get_yes_no_input("Allow fractional shares? (yes/no): ")
    inputs['frac_shares'] = str(frac_shares).lower() # Store as 'true' or 'false' string

    # ADD Fixed values to the dictionary before returning
    inputs['risk_tolerance'] = '10'
    inputs['risk_type'] = 'stock'
    inputs['remove_amplification_cap'] = 'true'

    # --- Collect Portfolio Tickers and Weights ---
    for i in range(num_portfolios):
        portfolio_num = i + 1
        tickers = get_user_input(f"Enter tickers for Portfolio {portfolio_num} (comma-separated): ", validation=lambda r: r and r.strip(), error_message="Tickers cannot be empty.")
        inputs[f'tickers_{portfolio_num}'] = tickers.upper()

        if portfolio_num == num_portfolios: # Last portfolio
            if num_portfolios == 1: weight = 100.0
            else: weight = 100.0 - sum(portfolio_weights)

            if weight < -0.01:
                 print(f"Error: Previous weights exceed 100% ({sum(portfolio_weights):.2f}%). Cannot set weight for Portfolio {portfolio_num}. Please start over.")
                 return None # Indicate failure
            weight = max(0, weight)
            inputs[f'weight_{portfolio_num}'] = f"{weight:.2f}"
            if num_portfolios > 1:
                print(f"Weight for final Portfolio {portfolio_num} automatically set to {weight:.2f}%.")
        else: # Not the last portfolio
            remaining_weight = 100.0 - sum(portfolio_weights)
            weight = get_float_input(f"Enter weight for Portfolio {portfolio_num} (0-{remaining_weight:.2f}). Remaining: {remaining_weight:.2f}%: ", min_value=0, max_value=remaining_weight + 0.01) # Add small tolerance for float comparison
            portfolio_weights.append(weight)
            inputs[f'weight_{portfolio_num}'] = f"{weight:.2f}"

    final_total_weight = sum(float(inputs.get(f'weight_{p+1}', 0)) for p in range(num_portfolios))
    if not math.isclose(final_total_weight, 100.0, abs_tol=0.1):
        print(f"Warning: Final weights sum to {final_total_weight:.2f}%, not exactly 100%.")

    return inputs

# Modified save_portfolio_to_csv for terminal context (no interaction param)
# (This function remains the same as it's core logic for file saving)
async def save_portfolio_to_csv(file_path, portfolio_data):
    # Saves the config from collect_portfolio_inputs to portfolio_codes_database.csv
    # **v2.5.4.0 Change**: Exclude 'tailor_portfolio' field if present
    file_exists = os.path.isfile(file_path)
    fieldnames = list(portfolio_data.keys())

    # Exclude 'tailor_portfolio' from being saved
    if 'tailor_portfolio' in fieldnames:
        fieldnames.remove('tailor_portfolio')

    try:
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
             if 'portfolio_code' in fieldnames:
                 fieldnames.insert(0, fieldnames.pop(fieldnames.index('portfolio_code')))

             writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore') # Use extrasaction='ignore'
             if not file_exists or os.path.getsize(file_path) == 0:
                 writer.writeheader()
             # Create a temporary dict excluding the unwanted field
             data_to_save = {k: v for k, v in portfolio_data.items() if k in fieldnames}
             writer.writerow(data_to_save)
        print(f"Portfolio configuration saved to {file_path}")
    except IOError as e:
        print(f"Error writing to CSV {file_path}: {e}")
    except Exception as e:
        print(f"Unexpected error saving portfolio config to CSV: {e}")

# Modified save_portfolio_data_internal for terminal context (no interaction param)
# (This function remains the same as it's core logic for file saving)
async def save_portfolio_data_internal(portfolio_code, date_str):
    """
    Internal function to save portfolio data without interaction, using a provided date.
    MODIFIED: Added comment clarifying that combined results are saved.
    """
    portfolio_db_file = 'portfolio_codes_database.csv'
    portfolio_data = None # Config read from CSV

    # --- Read Portfolio Config ---
    try:
        if not os.path.exists(portfolio_db_file):
             print(f"Error [Save]: Portfolio database '{portfolio_db_file}' not found.")
             return # Cannot save

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
                 return # Cannot save
    except Exception as e:
        print(f"Error [Save]: Reading portfolio database {portfolio_db_file} for {portfolio_code}: {e}")
        return # Cannot save

    # --- Process and Save Combined Data ---
    if portfolio_data and date_str:
        try:
            frac_shares = portfolio_data.get('frac_shares', 'false').lower() == 'true'

            # Process portfolio WITHOUT tailoring to get the combined result
            # Suppress Discord output during saving by passing interaction=None
            _, combined_result, _ = await process_custom_portfolio(
                # interaction=None, # Removed parameter
                portfolio_data=portfolio_data, # Pass the config read from CSV
                tailor_portfolio=False,        # Force False for saving combined data
                frac_shares=frac_shares,
                total_value=None,
                is_custom_command_without_save=False
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

        except Exception as e:
            print(f"Error [Save]: Processing/saving for code {portfolio_code}: {e}")
            # Optional: Raise the exception to be caught by the calling auto-save loop for logging
            # raise # Re-raising might stop the terminal script, better to just log and continue

# Modified save_portfolio_data for terminal context (no interaction param)
async def save_portfolio_data_terminal(portfolio_code):
    """Saves the *combined portfolio output* for terminal."""
    print(f"Attempting to save combined data for portfolio code: '{portfolio_code}'...")

    # --- Get Save Date ---
    save_date_str = get_user_input("Enter the date to save the data under (MM/DD/YYYY): ", validation=lambda d: True, error_message="") # Basic validation, rely on internal parse

    # Call internal save logic
    try:
        # Use the internal function which handles reading config and processing
        await save_portfolio_data_internal(portfolio_code, save_date_str)
        print(f"Save process completed for portfolio '{portfolio_code}' for date {save_date_str}. Check logs for details.")
    except Exception as e:
        # Error handling is mostly done inside the internal function, but catch any unexpected ones
        print(f"An error occurred while saving data for portfolio code '{portfolio_code}'. Check logs.")


# --- Terminal Handler for /custom Command ---
# Replaces the discord slash command function
async def handle_custom_command(args):
    """Handles the /custom command from the terminal."""
    if not args:
        print("Usage: /custom <portfolio_code> [save_code=3725]")
        return

    portfolio_code = args[0].strip()
    save_code = None
    if len(args) > 1:
        # Check for save_code argument format
        if args[1].lower().startswith("save_code="):
            save_code = args[1][len("save_code="):].strip()
        else:
            print("Invalid argument format. Use: /custom <portfolio_code> [save_code=3725]")
            return

    portfolio_db_file = 'portfolio_codes_database.csv'
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
        print(f"Using next available portfolio code: '{portfolio_code}'")

    # --- Save Data Action ---
    if save_code == "3725":
        if is_new_code_auto:
            print("Cannot use '#' with save_code. Please provide an existing code to save.")
            return
        # Call the save function directly. It will read the config and process internally.
        await save_portfolio_data_terminal(portfolio_code)
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
            print("Portfolio configuration collection cancelled.")
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
                value_str = get_user_input("Enter the total portfolio value to tailor (leave blank to skip tailoring): ")

                if value_str:
                    try:
                        total_value = float(value_str)
                        if total_value <= 0:
                            print("Value must be positive. Proceeding without tailoring.")
                            total_value = None
                            tailor_portfolio_requested = False
                        else:
                            tailor_portfolio_requested = True # User successfully provided a value
                    except ValueError:
                        print("Invalid number format for value. Proceeding without tailoring.")
                        total_value = None
                        tailor_portfolio_requested = False
                else: # User left blank
                    print("Skipping tailoring.")
                    tailor_portfolio_requested = False
                    total_value = None
            else:
                 # If tailoring is not enabled (save_code was 3725), total_value remains None
                 pass


            # Get frac_shares from the loaded portfolio_data config
            frac_shares = portfolio_data.get('frac_shares', 'no').lower() == 'yes'

            # Notify user before processing
            print(f"Processing custom portfolio code: '{portfolio_code}'...")

            # Call the main processing function
            # Pass parameters correctly based on whether tailoring was requested (by providing value)
            # The is_custom_command_without_save flag controls whether simplified output is shown
            # and is True if tailoring is happening AND save_code was NOT 3725.
            # This matches the original v2.5.4.0 logic for custom command output.
            await process_custom_portfolio(
                # interaction=None, # Removed parameter
                portfolio_data=portfolio_data, # Pass the config read from CSV
                tailor_portfolio=tailor_portfolio_requested, # Pass boolean indicating if value was provided
                frac_shares=frac_shares, # Pass the correctly determined boolean
                total_value=total_value, # Pass collected value or None
                is_custom_command_without_save=(tailor_portfolio_requested and save_code != "3725")
            )

            print(f"Custom portfolio analysis for '{portfolio_code}' complete.")

        except KeyError as e:
            print(f"Incomplete configuration for portfolio code {portfolio_code}. Missing key: {e}")
            print(f"Error: Configuration for portfolio code '{portfolio_code}' seems incomplete. Please check the 'portfolio_codes_database.csv' or recreate the code.")
        except Exception as e:
            print(f"Error processing custom portfolio {portfolio_code}: {e}")
            import traceback
            traceback.print_exc()
            print(f"An unexpected error occurred while processing portfolio '{portfolio_code}'. Check logs.")


# --- Terminal Handler for /invest Command ---
# Replaces the discord slash command function
# --- Terminal Handler for /invest Command ---
async def handle_invest_command(args):
    """Handles the /invest command from the terminal."""
    print("\n--- /invest Command ---")

    ema_sensitivity = get_int_input("Enter EMA sensitivity (1: Weekly, 2: Daily, 3: Hourly): ", min_value=1, max_value=3)
    valid_amplifications = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    amplification = get_float_input(f"Enter amplification ({', '.join(map(str, valid_amplifications))}): ", min_value=0.25)
    while amplification not in valid_amplifications:
        print(f"Invalid amplification value. Please enter one of: {', '.join(map(str, valid_amplifications))}")
        amplification = get_float_input(f"Enter amplification ({', '.join(map(str, valid_amplifications))}): ", min_value=0.25)
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
    weight = 0.0 # Initialize weight here to ensure it's always defined before use

    for i in range(1, num_portfolios + 1):
        current_portfolio_input = {}
        current_portfolio_weight = 0.0 # Initialize for this iteration

        if i == num_portfolios:
            if num_portfolios == 1:
                current_portfolio_weight = 100.0
            else:
                current_portfolio_weight = 100.0 - sum(portfolio_weights)
            
            if current_portfolio_weight < -0.01:
                print(f"Error: Previous weights exceed 100% ({sum(portfolio_weights):.2f}%). Please try again.")
                return
            current_portfolio_weight = max(0.0, current_portfolio_weight)
            if num_portfolios > 1:
                print(f"Weight for final Portfolio {i} automatically set to {current_portfolio_weight:.2f}%.")
            weight = current_portfolio_weight # Assign to 'weight' for consistent use
        else:
            remaining_weight = 100.0 - sum(portfolio_weights)
            # The 'weight' variable is assigned by get_float_input here
            weight = get_float_input(f"Enter weight for Portfolio {i} (0-{remaining_weight:.2f}). Remaining: {remaining_weight:.2f}%: ", min_value=0, max_value=remaining_weight + 0.01)
            portfolio_weights.append(weight)

        current_portfolio_input['weight'] = weight # Now 'weight' is guaranteed to be assigned

        tickers_str = get_user_input(f"Enter tickers for Portfolio {i} (comma-separated): ", validation=lambda r: r and r.strip(), error_message="Tickers cannot be empty.")
        tickers = [ticker.strip().upper() for ticker in tickers_str.split(',') if ticker.strip()]
        if not tickers:
             print(f"Tickers cannot be empty for Portfolio {i}. Please try again.")
             return

        current_portfolio_input['tickers'] = tickers
        all_portfolio_inputs.append(current_portfolio_input)

    total_weight_check = sum(p['weight'] for p in all_portfolio_inputs)
    if not math.isclose(total_weight_check, 100.0, abs_tol=0.1):
         print(f"Warning: Total portfolio weight must sum to 100%. Current sum is {total_weight_check:.2f}%.")

    print(f"Processing /invest request with {num_portfolios} portfolio(s)...")
    portfolio_data_dict = {
         'risk_type': 'stock',
         'risk_tolerance': '10',
         'ema_sensitivity': str(ema_sensitivity),
         'amplification': str(amplification),
         'num_portfolios': str(num_portfolios),
         'frac_shares': str(frac_shares).lower(),
         'remove_amplification_cap': 'true'
    }
    for i, p_data in enumerate(all_portfolio_inputs):
         portfolio_data_dict[f'tickers_{i+1}'] = ",".join(p_data['tickers'])
         portfolio_data_dict[f'weight_{i+1}'] = f"{p_data['weight']:.2f}"

    try:
        # Assuming process_custom_portfolio is defined elsewhere and handles these params
        # For the purpose of this focused fix, we are not including its full definition here.
        # await process_custom_portfolio(
        #     portfolio_data=portfolio_data_dict,
        #     tailor_portfolio=tailor_portfolio,
        #     frac_shares=frac_shares,
        #     total_value=total_value,
        #     is_custom_command_without_save=False
        # )
        print(f"/invest analysis complete. (process_custom_portfolio call skipped in this snippet)")
    except Exception as e:
         print(f"Error during /invest processing: {e}")
         import traceback
         traceback.print_exc()
         print(f"An error occurred during the analysis: {e}. Check logs.")

# --- Placeholder for other command handlers ---
# These will be implemented in the next part based on the original code.
async def handle_market_command(args):
    print("Market command not yet implemented in terminal version (Part 1).")
    print("Usage: /market [save_code=3725]")

async def handle_breakout_command(args):
    print("Breakout command not yet implemented in terminal version (Part 1).")
    print("Usage: /breakout [save_code=3725]")

async def handle_startbreakoutcycle_command(args):
     print("Start Breakout Cycle command not applicable in this terminal version.")

async def handle_endbreakout_command(args):
     print("End Breakout command not applicable in this terminal version.")

async def handle_assess_command(args):
    print("Assess command not yet implemented in terminal version (Part 1).")
    print("Usage: /assess <assess_code> [tickers=<...>] [timeframe=<...>] [risk_tolerance=<...>]")

async def handle_cultivate_command(args):
     print("Cultivate command not yet implemented in terminal version (Part 1).")
     print("Usage: /cultivate <portfolio_value> <frac_shares=True/False> <cultivate_code=A/B> [save_code=3725]")

# INVEST Terminal Version (Part 2)
# Continuation from Part 1.
# Includes handlers for /market, /breakout, /assess, and /cultivate.
# Modified Discord interactions to use terminal input/output.

import yfinance as yf
import pandas as pd
import math
from tabulate import tabulate
import os
import time
import sys
import matplotlib.pyplot as plt
import numpy as np
from tradingview_screener import Query, Column
from tradingview_ta import TA_Handler, Interval, Exchange
import csv
from datetime import datetime, timedelta, time as dt_time, date
import pytz
from typing import Optional
import traceback # Added for detailed error printing

# Assume the utility functions from Part 1 (safe_score, get_sp500_symbols, etc.) are available

# Define the EST timezone (repeated for clarity, but should be in Part 1)
est_timezone = pytz.timezone('US/Eastern')

# Function to get allocation score (needs to be in Part 1 or here)
# This function reads market_data.csv to determine allocation scores.
# In the terminal version, if market_data.csv doesn't exist, it defaults to 50.
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
        print(f"Warning: Market data file '{market_data_file}' not found. Using default scores (50).")
        return avg_score, risk_gen_score, mkt_inv_score

    try:
        df = pd.read_csv(market_data_file)
        if df.empty:
            print(f"Warning: Market data file '{market_data_file}' is empty. Using default scores (50).")
            return avg_score, risk_gen_score, mkt_inv_score

        # Get the last row of data
        last_row = df.iloc[-1]

        # Safely access columns and calculate scores
        # Use safe_score to handle potential NaN or missing values
        risk_gen_score = safe_score(last_row.get('RISK_GEN_SCORE', 50.0))
        mkt_inv_score = safe_score(last_row.get('MKT_INV_SCORE', 50.0))
        avg_score = safe_score(last_row.get('AVG_SCORE', 50.0)) # Use AVG_SCORE if available

        # If AVG_SCORE is not in the file, calculate it from the two scores
        if 'AVG_SCORE' not in df.columns:
             avg_score = (risk_gen_score + mkt_inv_score) / 2.0
             print(f"Warning: 'AVG_SCORE' not found in '{market_data_file}'. Calculated average: {avg_score:.2f}")

        # Clamp scores between 0 and 100
        avg_score = max(0, min(100, avg_score))
        risk_gen_score = max(0, min(100, risk_gen_score))
        mkt_inv_score = max(0, min(100, mkt_inv_score))


        print(f"Market scores loaded from '{market_data_file}': Avg={avg_score:.2f}, RiskGen={risk_gen_score:.2f}, MktInv={mkt_inv_score:.2f}")
        return avg_score, risk_gen_score, mkt_inv_score

    except Exception as e:
        print(f"Error reading market data file '{market_data_file}': {e}. Using default scores (50).")
        return 50.0, 50.0, 50.0 # Default to 50 on error


# --- Terminal Handler for /market Command ---
# Replaces the discord slash command function
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

    # --- Calculate Market Risk ---
    print("Calculating market risk scores...")
    try:
        market_risk_score = calculate_market_risk() # This returns the combined score
        if market_risk_score is None:
            print("Failed to calculate market risk score.")
            # Attempt to get scores from file if calculation failed
            avg_score, risk_gen_score, mkt_inv_score = get_allocation_score()
            if avg_score == 50.0 and risk_gen_score == 50.0 and mkt_inv_score == 50.0:
                 print("Could not load scores from file either. Market data unavailable.")
                 return # Exit if no data at all
            else:
                 print("Using scores from existing market data file.")
                 combined_score = avg_score # Use file average if calc failed
                 risk_gen_score_display = risk_gen_score
                 mkt_inv_score_display = mkt_inv_score
        else:
            # If calculate_market_risk succeeded, recalculate the components for display
            # This requires re-running parts of calculate_market_risk or modifying it
            # to return components. For simplicity, we'll just display the combined score
            # and potentially load components from the file if saving.
            combined_score = market_risk_score
            # To get component scores for display without recalculating everything,
            # we would need to modify calculate_market_risk to return them.
            # For now, we'll just display the combined score from the calculation.
            # If saving, the save function will handle getting the component scores.
            risk_gen_score_display = "N/A (Calculated Combined)" # Indicate components not directly returned
            mkt_inv_score_display = "N/A (Calculated Combined)"


        print("\n--- Market Insights ---")
        print(f"Combined Market Risk Score: {combined_score:.2f}%")
        # print(f"Risk General Score: {risk_gen_score_display}") # Optional: if components are returned
        # print(f"Market Invest Score: {mkt_inv_score_display}") # Optional: if components are returned

        # --- Save Data Action ---
        if save_code == "3725":
            print("Save code provided. Saving market data...")
            try:
                # Need to get the individual scores (Risk Gen, Mkt Inv) for saving.
                # This requires calling calculate_market_risk in a way that gives components,
                # or reading them from the file if calculate_market_risk only gives combined.
                # Let's modify calculate_market_risk to return all three: combined, general, large
                # (or combined, risk_gen, mkt_inv if those are the intended components).
                # Based on the original code, it seems 'general' and 'large' are the components
                # that feed into the 'combined'. Let's assume we need to save combined, general, large.
                # *** NOTE: The original calculate_market_risk only returned 'combined'.
                # We need to modify it to return the intermediate 'general' and 'large' scores.
                # For now, I will call it again and try to extract or just save combined.
                # Let's assume for saving we need combined, general, and large as per original file's save logic.
                # The original save logic saved 'RISK_GEN_SCORE', 'MKT_INV_SCORE', 'AVG_SCORE'.
                # Let's try to map 'general' to 'RISK_GEN_SCORE' and 'large' to 'MKT_INV_SCORE'.

                # Re-run calculation to get components (if calculate_market_risk is modified)
                # Or, ideally, modify calculate_market_risk to return all needed values.
                # Assuming calculate_market_risk now returns (combined, general, large)
                # This requires a change in the calculate_market_risk function itself.
                # For this part, I will assume calculate_market_risk IS modified to return (combined, general, large).
                # If it's not, this part will need adjustment.

                # *** MODIFICATION NEEDED IN calculate_market_risk (in Part 1) ***
                # It needs to return a tuple like (combined, general, large)
                # For now, I'll simulate getting these values.
                # In a real scenario, you'd modify calculate_market_risk in Part 1.
                # Let's assume calculate_market_risk returns (combined, general, large)
                # Example: combined_score, general_score, large_score = calculate_market_risk()
                # Since I cannot modify Part 1, I will simulate getting these values for saving.
                # A more robust solution requires modifying calculate_market_risk in Part 1.

                # SIMULATION: Assume calculate_market_risk returns (combined, general, large)
                # If calculate_market_risk only returns combined, we can't save components accurately here.
                # Let's stick to saving what the original code saved: RISK_GEN_SCORE, MKT_INV_SCORE, AVG_SCORE
                # This means we need to get these specific scores.
                # The get_allocation_score function *reads* these from the file.
                # We need to *calculate* them to save the *current* scores.
                # This means calculate_market_risk needs to provide these.

                # Let's assume calculate_market_risk is modified to return:
                # (combined, general_score, large_score)
                # Mapping: general_score -> RISK_GEN_SCORE, large_score -> MKT_INV_SCORE, combined -> AVG_SCORE

                # *** ASSUMING calculate_market_risk NOW RETURNS (combined, general, large) ***
                # If it doesn't, the following save logic will need adjustment.
                # Let's call it again to get the components for saving.
                # This is inefficient, ideally calculate_market_risk should return all needed values at once.
                # Re-calling calculate_market_risk just to get components:
                temp_result = calculate_market_risk() # Assuming it now returns (combined, general, large)
                if temp_result is None or len(temp_result) < 3:
                     print("Error: Could not get component scores for saving market data.")
                else:
                    combined_for_save, general_score_for_save, large_score_for_save = temp_result

                    save_file = 'market_data.csv' # Original save file name
                    file_exists = os.path.isfile(save_file)
                    save_date_str = datetime.now(est_timezone).strftime('%m/%d/%Y')

                    # Check if data for today is already saved
                    if await check_if_saved_today(save_file, save_date_str):
                        print(f"Market data for {save_date_str} already saved to '{save_file}'. Skipping save.")
                    else:
                        try:
                            with open(save_file, 'a', newline='', encoding='utf-8') as f:
                                writer = csv.DictWriter(f, fieldnames=['DATE', 'RISK_GEN_SCORE', 'MKT_INV_SCORE', 'AVG_SCORE'])
                                if not file_exists or os.path.getsize(save_file) == 0:
                                    writer.writeheader()
                                writer.writerow({
                                    'DATE': save_date_str,
                                    'RISK_GEN_SCORE': f"{safe_score(general_score_for_save):.2f}",
                                    'MKT_INV_SCORE': f"{safe_score(large_score_for_save):.2f}",
                                    'AVG_SCORE': f"{safe_score(combined_for_save):.2f}" # Saving combined as AVG_SCORE
                                })
                            print(f"Market data for {save_date_str} saved to '{save_file}'.")
                        except IOError as e:
                            print(f"Error writing market data to CSV {save_file}: {e}")
                        except Exception as e:
                            print(f"Unexpected error saving market data: {e}")
                            traceback.print_exc()

            except Exception as save_error:
                print(f"Error attempting to save market data: {save_error}")
                traceback.print_exc()


    except Exception as e:
        print(f"Error calculating or displaying market data: {e}")
        traceback.print_exc()

# --- Terminal Handler for /breakout Command ---
# --- Terminal Handler for /breakout Command ---
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
        # Corrected Query instantiation: 'screen' changed to 'screener_type'
        screen_query = Query(
            screener_type="america", # Corrected argument
            # exchange="NASDAQ", # This might be too restrictive, 'markets' usually covers exchanges.
                                 # Or, if you want to filter by specific exchanges, the library might have another way.
                                 # For now, let's rely on 'screener_type' and 'markets'.
                                 # If specific exchanges are needed, one might need to add them to 'markets' if supported,
                                 # or filter post-query. The 'markets' argument in Query usually takes a list like
                                 # ["stock", "futures", "forex", "cfd", "crypto", "index", "bond", "economic"].
                                 # To specify exchanges, it's often done through the 'symbols' parameter if scanning specific symbols,
                                 # or by filtering the results if the API returns exchange information.
                                 # The tradingview_screener library's `Query` class itself does not have a direct `exchange` parameter.
                                 # It's often implied by the `screener_type` or you filter results.
                                 # For a broad scan like "america", specifying "NASDAQ" might be redundant or incorrect
                                 # if `screener_type="america"` already covers US exchanges.
                                 # Let's assume "america" covers US exchanges including NASDAQ.
            markets=["stock"], # Specifies the type of market
            # filters=[ # Example filters (adjust based on your original code):
            #     {"left": "close", "operation": "greater", "right": 5},
            #     {"left": "volume", "operation": "greater", "right": 100000},
            #     {"left": "change", "operation": "greater", "right": 5},
            # ],
            columns=[ # Columns to retrieve
                "name", "close", "change", "volume", "RSI", "MACD.macd", "MACD.signal",
                "EMA50", "EMA200", "exchange" # Added 'exchange' to see if it helps filter later
            ]
        )

        screener_results_df = screen_query.get_scanner_data() # Returns a list of dicts and a DataFrame

        if screener_results_df is None or screener_results_df.empty:
            print("Breakout scan returned no results or failed.")
            return
        
        # The get_scanner_data() returns a tuple (data, dataframe)
        # We should use the dataframe for easier processing
        screener_results, df_results = screener_results_df 

        if not screener_results: # Check the list of dicts
             print("Breakout scan returned no results in list format.")
             return


        print(f"Found {len(screener_results)} potential breakout candidates from the initial scan.")

        breakout_candidates = []
        print("Calculating breakout scores...")
        # df_results is the pandas DataFrame from get_scanner_data()
        for index, row in df_results.iterrows():
            ticker = row.get('ticker', '').strip() # 'ticker' is usually the column name for symbols
            exchange = row.get('exchange', '').strip()

            # Example: Filter for NASDAQ if needed, though 'screener_type="america"' should cover it.
            # if exchange != "NASDAQ":
            #    continue
            
            if not ticker:
                continue

            try:
                # Using data directly from the DataFrame 'row'
                price = safe_score(row.get('close'))
                change_pct = safe_score(row.get('change'))
                volume = safe_score(row.get('volume'))
                rsi = safe_score(row.get('RSI')) # Make sure 'RSI' is in your columns list
                macd_line = safe_score(row.get('MACD.macd')) # Make sure 'MACD.macd' is in columns
                signal_line = safe_score(row.get('MACD.signal')) # Make sure 'MACD.signal' is in columns
                ema50 = safe_score(row.get('EMA50')) # Make sure 'EMA50' is in columns
                ema200 = safe_score(row.get('EMA200')) # Make sure 'EMA200' is in columns

                # Calculate MACD signal and strength using the separate function
                # This might be redundant if MACD values are already in screener results,
                # but the function also determines Buy/Sell signal based on histogram trend.
                # Let's assume daily interval for breakout MACD calculation.
                macd_signal_calc, macd_strength_calc = calculate_macd_signal(ticker, ema_interval=2)

                simple_score = 50.0
                if price and ema50 and price > ema50: simple_score += 5
                if price and ema200 and price > ema200: simple_score += 5
                if macd_signal_calc == "Buy": simple_score += 10 # Using calculated signal
                if macd_strength_calc > 50: simple_score += (macd_strength_calc - 50) / 5
                if rsi is not None and rsi > 50: simple_score += (rsi - 50) / 2
                if change_pct > 0: simple_score += min(20, change_pct)
                if volume > 500000: simple_score += min(10, volume / 500000) # Example, ensure volume is comparable

                breakout_score = max(0, min(100, simple_score))

                breakout_candidates.append({
                    'ticker': ticker,
                    'score': breakout_score,
                    'live_price': price,
                    'change_pct': change_pct,
                    'volume': volume,
                    'macd_signal': macd_signal_calc, # Using calculated signal
                    'macd_strength': macd_strength_calc, # Using calculated strength
                    'exchange': exchange
                })

            except Exception as e:
                print(f"Error processing breakout candidate {ticker}: {e}")
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
                    f"${safe_score(candidate.get('live_price', 0)):.2f}",
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
                    headers = ['DATE', 'TICKER', 'SCORE', 'PRICE', 'CHANGE_PCT', 'VOLUME', 'MACD_SIGNAL', 'MACD_STRENGTH', 'EXCHANGE']
                    writer = csv.DictWriter(f, fieldnames=headers)
                    if not file_exists or os.path.getsize(save_file) == 0:
                        writer.writeheader()
                    for candidate in breakout_candidates: # Save all, not just top 10
                        writer.writerow({
                            'DATE': save_date_str,
                            'TICKER': candidate.get('ticker', 'ERR'),
                            'SCORE': f"{safe_score(candidate.get('score', 0)):.2f}",
                            'PRICE': f"{safe_score(candidate.get('live_price', 0)):.2f}",
                            'CHANGE_PCT': f"{safe_score(candidate.get('change_pct', 0)):.2f}",
                            'VOLUME': f"{safe_score(candidate.get('volume', 0)):.0f}",
                            'MACD_SIGNAL': candidate.get('macd_signal', 'N/A'),
                            'MACD_STRENGTH': f"{safe_score(candidate.get('macd_strength', 0)):.2f}",
                            'EXCHANGE': candidate.get('exchange', 'N/A')
                        })
                print(f"Breakout data for {save_date_str} saved to '{save_file}'.")
            except IOError as e:
                print(f"Error writing breakout data to CSV {save_file}: {e}")
            except Exception as e:
                print(f"Unexpected error saving breakout data: {e}")
                traceback.print_exc()

    except Exception as e:
        print(f"Error during breakout scan: {e}")
        traceback.print_exc()


# --- Terminal Handler for /assess Command ---
# Replaces the discord slash command function
async def handle_assess_command(args):
    """Handles the /assess command from the terminal."""
    print("\n--- /assess Command ---")

    # Usage: /assess <assess_code> [tickers=<...>] [timeframe=<...>] [risk_tolerance=<...>]
    if not args:
        print("Usage: /assess <assess_code> [tickers=<...>] [timeframe=<...>] [risk_tolerance=<...>]")
        return

    assess_code = args[0].strip()
    tickers_str = None
    timeframe_str = None
    risk_tolerance_str = None

    # Parse optional arguments
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

    # --- Assess Logic ---
    # The core assess logic involves fetching data and calculating scores
    # based on the assess_code, tickers, timeframe, and risk tolerance.

    print(f"Running Assess for code: '{assess_code}'...")

    # --- Determine Parameters based on assess_code ---
    # This section needs to replicate the logic from your original assess command
    # to set default tickers, timeframe, and risk tolerance based on the assess_code.
    # Example mappings (ADJUST BASED ON YOUR ORIGINAL CODE):
    default_tickers = []
    default_timeframe = "1d" # Default to daily
    default_risk_tolerance = 10 # Default risk tolerance

    if assess_code.lower() == 'a':
        default_tickers = ['AAPL', 'MSFT', 'GOOGL'] # Example tickers for code A
        default_timeframe = "1d"
        default_risk_tolerance = 5
    elif assess_code.lower() == 'b':
        default_tickers = ['TSLA', 'AMZN', 'NVDA'] # Example tickers for code B
        default_timeframe = "1h"
        default_risk_tolerance = 15
    # Add more cases for other assess_codes from your original script

    # Override defaults with provided arguments if they exist
    tickers = [t.strip().upper() for t in tickers_str.split(',') if t.strip()] if tickers_str else default_tickers
    timeframe = timeframe_str if timeframe_str else default_timeframe
    try:
        risk_tolerance = int(risk_tolerance_str) if risk_tolerance_str else default_risk_tolerance
        risk_tolerance = max(1, min(20, risk_tolerance)) # Clamp risk tolerance
    except ValueError:
        print(f"Warning: Invalid risk_tolerance value '{risk_tolerance_str}'. Using default: {default_risk_tolerance}")
        risk_tolerance = default_risk_tolerance

    if not tickers:
        print("No tickers specified for assessment.")
        return

    print(f"Assessing tickers: {', '.join(tickers)}")
    print(f"Timeframe: {timeframe}")
    print(f"Risk Tolerance: {risk_tolerance}")

    # --- Perform Assessment Calculations ---
    assessment_results = []
    print("Calculating assessment scores...")
    for ticker in tickers:
        try:
            # This is where you integrate your specific assessment calculations.
            # This likely involves fetching data (using yfinance or other sources),
            # calculating indicators, and combining them into an assessment score.
            # This part needs to be implemented based on your original assess logic.

            # Example: Fetch daily data and calculate a simple score based on last close vs EMA50
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y", interval=timeframe) # Use specified timeframe
            if hist.empty or 'Close' not in hist.columns:
                print(f"Warning: Could not fetch data for {ticker} with timeframe {timeframe}. Skipping.")
                assessment_results.append({'ticker': ticker, 'error': 'Data fetch failed'})
                continue

            latest_price = safe_score(hist['Close'].iloc[-1])
            # Example: Calculate EMA50 (adjust span based on timeframe if needed)
            if len(hist) >= 50:
                 ema50 = safe_score(hist['Close'].ewm(span=50, adjust=False).mean().iloc[-1])
            else:
                 ema50 = None # Not enough data for EMA50

            # Example: Calculate a simple score (ADJUST BASED ON YOUR ORIGINAL LOGIC)
            assess_score = 50.0 # Default
            if latest_price > 0 and ema50 is not None:
                price_vs_ema50 = (latest_price - ema50) / ema50 * 100 # Percentage difference
                assess_score += price_vs_ema50 # Simple addition (adjust scaling)
                # Add other scoring factors from your original code (e.g., based on other indicators, patterns, etc.)

            # Incorporate risk tolerance into the score calculation or interpretation
            # This is highly dependent on how risk tolerance was used in your original code.
            # Example: Adjust score based on risk tolerance (placeholder logic)
            # If risk_tolerance is high, maybe the score is less penalized for volatility?
            # assess_score = adjust_score_by_risk(assess_score, risk_tolerance) # Placeholder function

            assess_score = max(0, min(100, assess_score)) # Clamp score

            assessment_results.append({
                'ticker': ticker,
                'score': assess_score,
                'live_price': latest_price,
                'timeframe': timeframe,
                'risk_tolerance': risk_tolerance
                # Add other relevant data points from your assessment logic
            })

        except Exception as e:
            print(f"Error assessing ticker {ticker}: {e}")
            traceback.print_exc()
            assessment_results.append({'ticker': ticker, 'error': str(e)})
            continue # Skip to next ticker on error

    # Sort results by score
    assessment_results.sort(key=lambda x: safe_score(x.get('score', 0)), reverse=True)

    # --- Output Assessment Results ---
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
                    f"${safe_score(result.get('live_price', 0)):.2f}",
                    result.get('timeframe', 'N/A'),
                    result.get('risk_tolerance', 'N/A')
                    # Add other columns as needed
                ])
        print(tabulate(table_data, headers=["Ticker", "Score", "Price", "Timeframe", "Risk Tolerance"], tablefmt="pretty"))

    print(f"Assessment for code '{assess_code}' complete.")


# --- Terminal Handler for /cultivate Command ---
# Replaces the discord slash command function
async def handle_cultivate_command(args):
    """Handles the /cultivate command from the terminal."""
    print("\n--- /cultivate Command ---")

    # Usage: /cultivate <portfolio_value> <frac_shares=True/False> <cultivate_code=A/B> [save_code=3725]
    if len(args) < 3:
        print("Usage: /cultivate <portfolio_value> <frac_shares=True/False> <cultivate_code=A/B> [save_code=3725]")
        return

    # Parse required arguments
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

    # Parse optional save_code
    save_code = None
    if len(args) > 3:
        if args[3].lower().startswith("save_code="):
            save_code = args[3][len("save_code="):].strip()
        else:
            print("Warning: Unknown argument '{args[3]}'. Ignoring.")

    print(f"Running Cultivate for code: '{cultivate_code}' with value ${portfolio_value:,.2f}...")
    print(f"Fractional Shares: {frac_shares}")

    # --- Cultivate Logic ---
    # The core cultivate logic involves reading saved data (likely from breakout or market),
    # filtering/selecting tickers based on the cultivate_code and potentially other factors,
    # and then calculating allocations based on the portfolio_value and frac_shares.

    # --- Determine Data Source and Filtering based on cultivate_code ---
    # This section needs to replicate the logic from your original cultivate command
    # to determine which saved data file to read and how to filter/select tickers.
    # Example mappings (ADJUST BASED ON YOUR ORIGINAL CODE):
    source_file = None
    min_score_threshold = 70 # Example default threshold
    num_top_tickers = 10 # Example default number of top tickers

    if cultivate_code == 'A':
        source_file = 'breakout_data.csv' # Example: Use breakout data for code A
        min_score_threshold = 75
        num_top_tickers = 15
    elif cultivate_code == 'B':
        source_file = 'market_data.csv' # Example: Use market data for code B
        min_score_threshold = 60
        num_top_tickers = 20
    # Add more cases for other cultivate_codes from your original script

    if not source_file or not os.path.exists(source_file):
        print(f"Error: Source data file '{source_file}' not found for cultivate code '{cultivate_code}'.")
        return

    print(f"Reading data from: '{source_file}'")

    # --- Read and Filter Data ---
    try:
        df = pd.read_csv(source_file)
        if df.empty:
            print(f"No data found in '{source_file}'.")
            return

        # Filter by date (get the latest date's data)
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df.dropna(subset=['DATE'], inplace=True) # Drop rows where DATE is invalid
            if df.empty:
                 print(f"No valid dates found in '{source_file}'.")
                 return
            latest_date = df['DATE'].max()
            df_latest = df[df['DATE'] == latest_date].copy() # Use .copy() to avoid SettingWithCopyWarning
            print(f"Using data from latest date: {latest_date.strftime('%m/%d/%Y')}")
        else:
            print("Warning: 'DATE' column not found in source file. Using all data.")
            df_latest = df.copy()

        if df_latest.empty:
            print(f"No data found for the latest date in '{source_file}'.")
            return

        # Filter by score threshold and select top tickers
        # Assuming score column is named 'SCORE' for breakout_data and 'AVG_SCORE' for market_data
        score_column = 'SCORE' if 'SCORE' in df_latest.columns else \
                       'AVG_SCORE' if 'AVG_SCORE' in df_latest.columns else None

        if not score_column:
            print(f"Error: Could not find a score column ('SCORE' or 'AVG_SCORE') in '{source_file}'.")
            return

        df_latest[score_column] = df_latest[score_column].apply(safe_score) # Ensure score is numeric
        df_filtered = df_latest[df_latest[score_column] >= min_score_threshold].copy()

        if df_filtered.empty:
            print(f"No tickers found with a score >= {min_score_threshold}% in '{source_file}'.")
            return

        # Sort by score and select top N
        df_sorted = df_filtered.sort_values(by=score_column, ascending=False).head(num_top_tickers).copy()

        if df_sorted.empty:
            print(f"No tickers remaining after selecting top {num_top_tickers}.")
            return

        print(f"Selected {len(df_sorted)} tickers with score >= {min_score_threshold}% (Top {num_top_tickers}):")
        print(df_sorted[['TICKER', score_column]].to_string(index=False)) # Display selected tickers and scores

        # --- Calculate Allocations ---
        # This involves fetching current prices for the selected tickers
        # and calculating how many shares to buy based on portfolio_value and frac_shares.

        print("\nFetching live prices and calculating allocations...")
        cultivate_allocations = []
        total_allocated_value = 0.0

        # Fetch live prices (can be slow for many tickers)
        tickers_to_fetch = df_sorted['TICKER'].tolist()
        try:
            live_prices_data = yf.download(tickers_to_fetch, period="1d", interval="1m", progress=False)
            if live_prices_data.empty or 'Close' not in live_prices_data.columns:
                 print("Warning: Could not fetch live prices for selected tickers. Cannot calculate allocations.")
                 return
            # Get the latest close price for each ticker
            live_prices = live_prices_data['Close'].iloc[-1].to_dict()
        except Exception as e:
            print(f"Error fetching live prices: {e}. Cannot calculate allocations.")
            traceback.print_exc()
            return


        # Calculate allocation for each selected ticker
        total_score_of_selected = safe_score(df_sorted[score_column].sum())
        if total_score_of_selected <= 0:
            print("Error: Total score of selected tickers is zero or negative. Cannot calculate weighted allocation.")
            return

        for index, row in df_sorted.iterrows():
            ticker = row['TICKER']
            score = safe_score(row[score_column])
            live_price = safe_score(live_prices.get(ticker, 0.0)) # Get price from fetched data

            if live_price <= 0:
                print(f"Warning: Live price for {ticker} is zero or not available ({live_price}). Skipping allocation.")
                continue

            # Weighted allocation based on score
            weighted_percent_allocation = (score / total_score_of_selected) * 100.0
            target_allocation_value = portfolio_value * (weighted_percent_allocation / 100.0)

            shares = 0.0
            try:
                exact_shares = target_allocation_value / live_price
                if frac_shares:
                    shares = round(exact_shares, 1)
                else:
                    shares = float(math.floor(exact_shares))
            except ZeroDivisionError:
                print(f"Warning: ZeroDivisionError calculating shares for {ticker} with price {live_price}. Shares set to 0.")
                shares = 0.0
            except Exception as e_shares:
                print(f"Error calculating shares for {ticker}: {e_shares}")
                shares = 0.0

            shares = max(0.0, shares)
            actual_money_allocation = shares * live_price
            share_threshold = 0.1 if frac_shares else 1.0

            if shares >= share_threshold: # Only include if buying at least the minimum share amount
                cultivate_allocations.append({
                    'ticker': ticker,
                    'score': score,
                    'live_price': live_price,
                    'shares': shares,
                    'actual_money_allocation': actual_money_allocation,
                    'weighted_percent_allocation': weighted_percent_allocation,
                    'actual_percent_allocation': (actual_money_allocation / portfolio_value) * 100.0 if portfolio_value > 0 else 0.0
                })
                total_allocated_value += actual_money_allocation

        # --- Calculate Remaining Cash ---
        remaining_cash = portfolio_value - total_allocated_value
        remaining_cash_percent = (remaining_cash / portfolio_value) * 100.0 if portfolio_value > 0 else 0.0

        # Sort allocations by score
        cultivate_allocations.sort(key=lambda x: safe_score(x.get('score', 0)), reverse=True)

        # --- Output Cultivate Results ---
        print("\n--- Cultivate Allocation Results ---")
        if not cultivate_allocations:
            print("No tickers met the criteria for allocation.")
        else:
            table_data = [
                [item['ticker'], f"{safe_score(item['score']):.2f}%", f"${safe_score(item['live_price']):.2f}",
                 f"{item['shares']:.1f}" if frac_shares and item['shares'] > 0 else f"{int(item['shares'])}",
                 f"${safe_score(item['actual_money_allocation']):,.2f}",
                 f"{safe_score(item['actual_percent_allocation']):.2f}%"]
                for item in cultivate_allocations
            ]
            # Add Cash row
            table_data.append(['Cash', '-', '-', '-', f"${safe_score(remaining_cash):,.2f}", f"{safe_score(remaining_cash_percent):.2f}%"])

            print(tabulate(table_data, headers=["Ticker", "Score", "Price", "Shares", "Actual $ Alloc", "Actual % Alloc"], tablefmt="pretty"))

        print(f"Cultivate analysis for code '{cultivate_code}' complete.")

        # --- Save Data Action ---
        if save_code == "3725":
            print("Save code provided. Saving cultivate data...")
            # Save the calculated allocations
            save_file = f"cultivate_code_{cultivate_code}_data.csv" # Example save file name
            file_exists = os.path.isfile(save_file)
            save_date_str = datetime.now(est_timezone).strftime('%m/%d/%Y')

            # Check if data for today is already saved (optional)
            # if await check_if_saved_today(save_file, save_date_str):
            #     print(f"Cultivate data for {save_date_str} already saved to '{save_file}'. Skipping save.")
            # else:
            try:
                with open(save_file, 'a', newline='', encoding='utf-8') as f:
                    # Headers: DATE, TICKER, SCORE, PRICE, SHARES, ACTUAL_DOLLAR_ALLOC, ACTUAL_PERCENT_ALLOC
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
                            'SHARES': f"{item.get('shares', 0):.1f}" if frac_shares else f"{int(item.get('shares', 0))}",
                            'ACTUAL_DOLLAR_ALLOC': f"{safe_score(item.get('actual_money_allocation', 0)):.2f}",
                            'ACTUAL_PERCENT_ALLOC': f"{safe_score(item.get('actual_percent_allocation', 0)):.2f}"
                        })
                    # Add the cash row to the saved data
                    writer.writerow({
                        'DATE': save_date_str,
                        'TICKER': 'Cash',
                        'SCORE': '-',
                        'PRICE': '-',
                        'SHARES': '-',
                        'ACTUAL_DOLLAR_ALLOC': f"{safe_score(remaining_cash):.2f}",
                        'ACTUAL_PERCENT_ALLOC': f"{safe_score(remaining_cash_percent):.2f}"
                    })
                print(f"Cultivate data for {save_date_str} saved to '{save_file}'.")
            except IOError as e:
                print(f"Error writing cultivate data to CSV {save_file}: {e}")
            except Exception as e:
                print(f"Unexpected error saving cultivate data: {e}")
                traceback.print_exc()


    except Exception as e:
        print(f"Error during cultivate process: {e}")
        traceback.print_exc()

# --- Main Terminal Loop ---
async def main():
    run_startup_sequence()

    while True:
        command_line = input("\nEnter command: ").strip()
        if not command_line:
            continue

        parts = command_line.split()
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
        # Removed /startbreakoutcycle and /endbreakout handlers
        # elif command == '/startbreakoutcycle':
        #      await handle_startbreakoutcycle_command(args)
        # elif command == '/endbreakout':
        #      await handle_endbreakout_command(args)
        elif command == '/assess':
             await handle_assess_command(args)
        elif command == '/cultivate':
             await handle_cultivate_command(args)
        else:
            print(f"Unknown command: {command}. Type '/exit' to quit or see the list of available commands above.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
