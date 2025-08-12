import yfinance as yf
import pandas as pd
import math
from math import sqrt
from tabulate import tabulate
import os
import uuid
import matplotlib
matplotlib.use('Agg') # Use the 'Agg' backend which is non-interactive
import asyncio
import matplotlib.pyplot as plt
import numpy as np
from tradingview_screener import Query, Column
# from tradingview_ta import TA_Handler, Interval, Exchange
import csv
from datetime import datetime, timedelta # Keep standard datetime
import pytz
from typing import Optional, List, Dict, Any
import time as py_time
import traceback
import logging # For R.I.S.K. module's logging
import json
import google.generativeai as genai
# We will only import FunctionDeclaration and Tool, and define parameters using dicts
# At the top of your script with other google.generativeai imports:
from google.generativeai.types import FunctionDeclaration, Tool
# REMOVE: from google.generativeai.types import Content, Part # Not strictly needed if using dicts for parts
import fear_and_greed
import humanize
from nltk.tokenize import sent_tokenize
import nltk

# It's good practice to ensure nltk packages are available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("📂 Downloading 'punkt' for nltk...")
    nltk.download('punkt', quiet=True)

# --- Global Variables & Constants ---
PORTFOLIO_DB_FILE = 'portfolio_codes_database.csv'
PORTFOLIO_OUTPUT_DIR = 'portfolio_outputs' # New directory for custom portfolio run outputs
EST_TIMEZONE = pytz.timezone('US/Eastern') # R.I.S.K uses EST, Singularity uses it for consistency if needed
BREAKOUT_TICKERS_FILE = "breakout_tickers.csv"
BREAKOUT_HISTORICAL_DB_FILE = "breakout_historical_database.csv"
MARKET_FULL_SENS_DATA_FILE_PREFIX = "market_full_sens_"
MARKET_HEDGING_TICKERS = ['SPY', 'DIA', 'QQQ']
RESOURCE_HEDGING_TICKERS = ['GLD', 'SLV']
HEDGING_TICKERS = MARKET_HEDGING_TICKERS + RESOURCE_HEDGING_TICKERS
CULTIVATE_INITIAL_METRICS_FILE = "cultivate_initial_metrics.csv"
CULTIVATE_T1_FILE = "cultivate_ticker_list_one.csv"
CULTIVATE_T_MINUS_1_FILE = "cultivate_ticker_list_negative_one.csv"
CULTIVATE_TF_FINAL_FILE = "cultivate_ticker_list_final.csv"
CULTIVATE_COMBINED_DATA_FILE_PREFIX = "cultivate_combined_"
USERS_FAVORITES_FILE = 'users_favorites.txt' # For /briefing command

# --- Gemini API Configuration --- #
GEMINI_API_KEY = "AIzaSyCL8jTe5XWYMWTh7_fexKbNCoVNezzCc8Y"  # <--- REPLACE WITH YOUR ACTUAL API KEY
try:
    if GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        print("⚠️ Warning: Replace 'YOUR_GEMINI_API_KEY' with your actual Gemini API key.")
        gemini_model = None
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or another model like 'gemini-pro'
        print("✔ Gemini API configured successfully.")
except Exception as e:
    print(f"❌ Error configuring Gemini API: {e}")
    gemini_model = None # Set to None if configuration fails
# --- End Gemini API Configuration ---

# --- AI Chat Session State ---
# gemini_model: Optional[genai.GenerativeModel] = None # Must be initialized by API config
GEMINI_CHAT_SESSION = None # Will hold the active ChatSession object - deprecated if managing history manually
AI_CONVERSATION_HISTORY = [] # Initialize this list globally in your script
CURRENT_AI_SESSION_ORIGINAL_REQUEST = None
AI_INTERNAL_STEP_COUNT = 0 # For progress indication

# --- R.I.S.K. Module Specific Constants & Globals ---
RISK_CSV_FILE = "market_data.csv"  # Main data file for RISK module
RISK_EOD_CSV_FILE = "risk_eod_data.csv"  # EOD data file for RISK module
RISK_LOG_FILE = 'risk_calculations.log'

# R.I.S.K. global state for market signal (will be loaded/updated from CSV for Singularity)
risk_persistent_signal = "Hold" # Default
risk_signal_day = None # Default

# --- Logging Setup (for R.I.S.K. module parts) ---
risk_logger = logging.getLogger('RISK_MODULE')
risk_logger.setLevel(logging.INFO)
risk_logger.propagate = False
if not risk_logger.hasHandlers(): # This ensures it's set up once
    risk_file_handler = logging.FileHandler(RISK_LOG_FILE)
    risk_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
    risk_file_handler.setFormatter(risk_formatter)
    risk_logger.addHandler(risk_file_handler)

# --- Utility Functions ---
def ensure_portfolio_output_dir():
    """Ensures the directory for saving portfolio outputs exists."""
    if not os.path.exists(PORTFOLIO_OUTPUT_DIR):
        try:
            os.makedirs(PORTFOLIO_OUTPUT_DIR)
            print(f"✔ Created directory for portfolio outputs: {PORTFOLIO_OUTPUT_DIR}")
        except OSError as e:
            print(f"❌ Error creating directory {PORTFOLIO_OUTPUT_DIR}: {e}. Please create it manually.")
            # Potentially raise an error or exit if this is critical
ensure_portfolio_output_dir() # Call at startup

async def _continuous_spinner_animation(stop_event: asyncio.Event, message_prefix: str = "AI is processing..."):
    """Helper coroutine for a continuous spinning animation that ends with 'Done!'."""
    animation_chars = ["|", "/", "-", "\\"]
    idx = 0
    try:
        while not stop_event.is_set(): # Loop continues as long as stop_event is not set
            print(f"\r{message_prefix} {animation_chars[idx % len(animation_chars)]}  ", end="", flush=True)
            idx += 1
            await asyncio.sleep(0.1) # Control spin speed
    except asyncio.CancelledError:
        # If the task is cancelled, we'll just clear the line in the finally block
        # as 'Done!' would be inappropriate.
        pass
    finally:
        # This block runs regardless of how the try block was exited.
        if stop_event.is_set():
            # Normal completion: stop_event was set by the parent.
            # Overwrite the spinner line with the "Done!" message.
            # Add enough spaces to ensure the previous animation (e.g., "|  ") is overwritten.
            print(f"\r{message_prefix} Done!          ", end="", flush=True)
        else:
            # Abnormal exit or cancellation where stop_event wasn't set explicitly by the parent
            # prior to this finally block running (e.g. task cancelled externally).
            # In this case, just clear the line.
            print(f"\r{' ' * (len(message_prefix) + 20)}\r", end="", flush=True) # Clear more space

def make_hashable(obj):
    """ Recursively converts dicts and lists to hashable tuples. """
    if isinstance(obj, dict):
        return tuple((k, make_hashable(v)) for k, v in sorted(obj.items()))
    if isinstance(obj, list):
        return tuple(make_hashable(e) for e in obj)
    # This handles the specific 'RepeatedComposite' type from the error, converting it to a list first
    if "RepeatedComposite" in str(type(obj)):
         return tuple(make_hashable(e) for e in list(obj))
    return obj

def _get_custom_portfolio_run_csv_filepath(portfolio_code: str) -> str:
    """Generates the CSV filepath for a custom portfolio's run output."""
    # Ensure PORTFOLIO_OUTPUT_DIR is a globally defined constant
    return os.path.join(PORTFOLIO_OUTPUT_DIR, f"run_data_portfolio_{portfolio_code.lower().replace(' ','_')}.csv")

async def _save_custom_portfolio_run_to_csv(portfolio_code: str, 
                                        tailored_stock_holdings: List[Dict[str, Any]], 
                                        final_cash: float, 
                                        total_portfolio_value_for_percent_calc: Optional[float] = None,
                                        is_called_by_ai: bool = False):
    """
    Saves the detailed tailored output of a custom portfolio run to a CSV file.
    Overwrites the file if it already exists.
    """
    filepath = _get_custom_portfolio_run_csv_filepath(portfolio_code)
    timestamp_utc_str = datetime.now(pytz.UTC).isoformat()
    # ... (data_for_csv and fieldnames logic remains the same) ...
    data_for_csv = []
    for holding in tailored_stock_holdings: # Ensure this part is correct
        data_for_csv.append({
            'Ticker': holding.get('ticker'),
            'Shares': holding.get('shares'),
            'LivePriceAtEval': holding.get('live_price_at_eval'),
            'ActualMoneyAllocation': holding.get('actual_money_allocation'),
            'ActualPercentAllocation': holding.get('actual_percent_allocation'),
            'RawInvestScore': holding.get('raw_invest_score', 'N/A')
        })
    
    cash_percent_alloc_val = 'N/A'
    if total_portfolio_value_for_percent_calc is not None and total_portfolio_value_for_percent_calc > 0:
        cash_percent_alloc_val = (final_cash / total_portfolio_value_for_percent_calc) * 100.0
    
    data_for_csv.append({
        'Ticker': 'Cash', 'Shares': '-', 'LivePriceAtEval': 1.0,
        'ActualMoneyAllocation': final_cash,
        'ActualPercentAllocation': cash_percent_alloc_val if isinstance(cash_percent_alloc_val, str) else f"{cash_percent_alloc_val:.2f}",
        'RawInvestScore': 'N/A'
    })
    fieldnames = ['Ticker', 'Shares', 'LivePriceAtEval', 'ActualMoneyAllocation', 'ActualPercentAllocation', 'RawInvestScore']

    try:
        ensure_portfolio_output_dir() 
        # <<< START DEBUG PRINT (Unconditional) >>>
        # print(f"CONSOLE_DEBUG_SAVE: Attempting to save portfolio '{portfolio_code}' to CSV at: '{filepath}'")
        # <<< END DEBUG PRINT >>>
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            csvfile.write(f"# portfolio_code: {portfolio_code}\n")
            csvfile.write(f"# timestamp_utc: {timestamp_utc_str}\n")
            csvfile.write(f"# ---BEGIN_DATA---\n") 
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_for_csv)
        
        # <<< START DEBUG PRINT (Unconditional) >>>
        # print(f"CONSOLE_DEBUG_SAVE: Successfully wrote CSV for portfolio '{portfolio_code}' to: '{filepath}'. File exists: {os.path.exists(filepath)}")
        # <<< END DEBUG PRINT >>>
        
        if not is_called_by_ai: # Original conditional print
            print(f"📂 Custom portfolio run CSV for '{portfolio_code}' saved to: {filepath}")
    except IOError as e:
        # <<< START DEBUG PRINT (Unconditional) >>>
        # print(f"CONSOLE_DEBUG_SAVE: IOError for portfolio '{portfolio_code}' at '{filepath}': {e}")
        # <<< END DEBUG PRINT >>>
        if not is_called_by_ai:
            print(f"❌ Error saving custom portfolio run CSV for '{portfolio_code}' to {filepath}: {e}")
    except Exception as e_gen:
        # <<< START DEBUG PRINT (Unconditional) >>>
        # print(f"CONSOLE_DEBUG_SAVE: Unexpected error for portfolio '{portfolio_code}' at '{filepath}': {e_gen}")
        # <<< END DEBUG PRINT >>>
        if not is_called_by_ai:
             print(f"❌ Unexpected error saving custom portfolio run CSV for '{portfolio_code}': {e_gen}")

async def _load_custom_portfolio_run_from_csv(portfolio_code: str, is_called_by_ai: bool = False) -> Dict[str, Any]:
    """
    Loads the last saved detailed run output of a custom portfolio from its CSV file.
    Always returns a dictionary with 'status', 'data' (if successful), 'message', and 'error_details'/'warnings'.
    """
    filepath = _get_custom_portfolio_run_csv_filepath(portfolio_code)
    ensure_portfolio_output_dir() # Good to have here as well

    # <<< START DEBUG PRINT (Unconditional) >>>
    # print(f"CONSOLE_DEBUG_LOAD: Attempting to load portfolio '{portfolio_code}' from CSV at: '{filepath}'")
    # print(f"CONSOLE_DEBUG_LOAD: File '{filepath}' exists: {os.path.exists(filepath)}")
    # <<< END DEBUG PRINT >>>

    if not os.path.exists(filepath):
        msg = f"No saved run CSV found for portfolio '{portfolio_code}' at '{filepath}'."
        # CONSOLE_DEBUG_LOAD already printed existence, this is the formal return
        return {"status": "error_file_not_found", "data": None, "message": msg, "error_details": msg}
    
    # ... (rest of the function from the previous good version with detailed status returns) ...
    # Ensure the rest of the function (metadata parsing, row processing, error status returns) 
    # is as per the refined version from the prior response that returns a dict.
    # For brevity, not repeating the entire function body here, but it should include the 
    # detailed status/error dictionary returns as discussed previously.
    # Example of a return path if parsing fails later:
    # except (IOError, csv.Error) as e_csv_io:
    #    msg = f"Critical CSV I/O or parsing error for '{filepath}' ('{portfolio_code}'): {e_csv_io}"
    #    print(f"CONSOLE_DEBUG_LOAD: {msg}") # Unconditional debug print
    #    return {"status": "error_csv_read_critical", "data": None, "message": msg, "error_details": str(e_csv_io)}

    # Re-paste the full improved _load_custom_portfolio_run_from_csv from my previous detailed answer here,
    # just ensure the CONSOLE_DEBUG_LOAD prints are at the top as shown.
    metadata = {"portfolio_code_from_file": portfolio_code}
    tailored_holdings_from_csv = []
    final_cash_value_from_csv = 0.0
    problematic_rows_details = []
    found_begin_data_marker = False

    try:
        with open(filepath, 'r', encoding='utf-8') as csvfile:
            lines_read_for_meta = 0
            for line in csvfile:
                lines_read_for_meta += 1
                line = line.strip()
                if line.startswith("# portfolio_code:"):
                    metadata["portfolio_code_from_file"] = line.split(":", 1)[1].strip()
                elif line.startswith("# timestamp_utc:"):
                    metadata["timestamp_utc"] = line.split(":", 1)[1].strip()
                elif line.startswith("# ---BEGIN_DATA---"):
                    found_begin_data_marker = True
                    break
                if lines_read_for_meta > 10:
                    msg = f"Could not find '---BEGIN_DATA---' marker after {lines_read_for_meta} lines in '{filepath}' for '{portfolio_code}'."
                    # print(f"CONSOLE_DEBUG_LOAD: {msg}")
                    return {"status": "error_parsing_metadata", "data": None, "message": msg, "error_details": msg}
            
            if not found_begin_data_marker:
                msg = f"'---BEGIN_DATA---' marker not found in '{filepath}'. File might be malformed."
                # print(f"CONSOLE_DEBUG_LOAD: {msg}")
                return {"status": "error_missing_begin_data_marker", "data": None, "message": msg, "error_details": msg}

            reader = csv.DictReader(csvfile)
            row_num_in_data_section = 0
            for row in reader:
                row_num_in_data_section += 1
                try:
                    shares_raw = row.get('Shares')
                    shares_val = shares_raw
                    if shares_raw != '-':
                        try:
                            shares_val = float(shares_raw)
                            if shares_val.is_integer(): shares_val = int(shares_val)
                        except (ValueError, TypeError): pass

                    percent_alloc_raw = row.get('ActualPercentAllocation')
                    percent_alloc_val = percent_alloc_raw
                    if percent_alloc_raw and percent_alloc_raw.lower() != 'n/a':
                        try: percent_alloc_val = float(percent_alloc_raw)
                        except ValueError: pass

                    raw_score_raw = row.get('RawInvestScore')
                    raw_score_val = raw_score_raw
                    if raw_score_raw and raw_score_raw.lower() != 'n/a':
                        try: raw_score_val = float(raw_score_raw)
                        except ValueError: pass

                    holding_dict = {
                        'ticker': row.get('Ticker'), 'shares': shares_val,
                        'live_price_at_eval': safe_score(row.get('LivePriceAtEval')),
                        'actual_money_allocation': safe_score(row.get('ActualMoneyAllocation')),
                        'actual_percent_allocation': percent_alloc_val,
                        'raw_invest_score': raw_score_val
                    }
                    if str(row.get('Ticker', '')).upper() == 'CASH':
                        final_cash_value_from_csv = holding_dict['actual_money_allocation']
                    else:
                        tailored_holdings_from_csv.append(holding_dict)
                except Exception as e_row_parse:
                    detail = f"Row {row_num_in_data_section}: Data '{str(row)[:100]}...' Error: {e_row_parse}"
                    # print(f"CONSOLE_DEBUG_LOAD: Warning - Skipping problematic row in '{filepath}'. {detail}")
                    problematic_rows_details.append(detail)

        if row_num_in_data_section == 0 and not tailored_holdings_from_csv and math.isclose(final_cash_value_from_csv, 0.0):
            base_msg = f"CSV for '{portfolio_code}' read, but no data rows found or all rows were problematic."
            if problematic_rows_details:
                err_dtls = base_msg + f" First few errors: {'; '.join(problematic_rows_details[:2])}"
                # print(f"CONSOLE_DEBUG_LOAD: {err_dtls}")
                return {"status": "error_all_rows_problematic", "data": None, "message": err_dtls, "error_details": err_dtls}
            else:
                # print(f"CONSOLE_DEBUG_LOAD: {base_msg} (No specific row errors captured, but data section empty).")
                return {"status": "error_no_data_rows", "data": None, "message": base_msg, "error_details": base_msg}

        output_data = {
            "portfolio_code": metadata.get("portfolio_code_from_file", portfolio_code),
            "timestamp_utc": metadata.get("timestamp_utc", datetime.now(pytz.UTC).isoformat()),
            "tailored_holdings": tailored_holdings_from_csv,
            "final_cash_value": final_cash_value_from_csv
        }

        if problematic_rows_details:
            msg = f"Successfully loaded CSV for '{portfolio_code}' but {len(problematic_rows_details)} rows had issues."
            # print(f"CONSOLE_DEBUG_LOAD: {msg} First error: {problematic_rows_details[0]}")
            return {"status": "success_with_warnings", "data": output_data, "message": msg, "warnings": problematic_rows_details}
        
        # print(f"CONSOLE_DEBUG_LOAD: Successfully loaded CSV for '{portfolio_code}'.")
        return {"status": "success", "data": output_data, "message": f"Successfully loaded CSV for '{portfolio_code}'."}

    except (IOError, csv.Error) as e_csv_io:
        msg = f"Critical CSV I/O or parsing error for '{filepath}' ('{portfolio_code}'): {e_csv_io}"
        # print(f"CONSOLE_DEBUG_LOAD: {msg}")
        return {"status": "error_csv_read_critical", "data": None, "message": msg, "error_details": str(e_csv_io)}
    except Exception as e_gen_load_csv:
        msg = f"Unexpected error loading saved run CSV '{filepath}' for '{portfolio_code}': {e_gen_load_csv}"
        # print(f"CONSOLE_DEBUG_LOAD: {msg}") # Removed traceback print for conciseness here
        return {"status": "error_unexpected_load", "data": None, "message": msg, "error_details": str(e_gen_load_csv)}

def safe_score(value: Any) -> float:
    try:
        if pd.isna(value) or value is None: return 0.0
        if isinstance(value, str): value = value.replace('%', '').replace('$', '').strip()
        return float(value)
    except (ValueError, TypeError): return 0.0

async def calculate_ema_invest(ticker: str, ema_interval: int, is_called_by_ai: bool = False) -> tuple[Optional[float], Optional[float]]:
    ticker_yf_format = ticker.replace('.', '-')
    stock = yf.Ticker(ticker_yf_format)
    interval_map = {1: "1wk", 2: "1d", 3: "1h"}
    period_map = {1: "max", 2: "10y", 3: "2y"}
    interval_str = interval_map.get(ema_interval, "1h")
    period_str = period_map.get(ema_interval, "2y")

    try:
        data = await asyncio.to_thread(stock.history, period=period_str, interval=interval_str)
    except Exception as e:
        # if not is_called_by_ai: print(f"EMA Invest: Error fetching history for {ticker}: {e}")
        return None, None
    if data.empty or 'Close' not in data.columns: return None, None
    try:
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()
    except Exception as e:
        # if not is_called_by_ai: print(f"EMA Invest: Error calculating EMAs for {ticker}: {e}")
        return None, None
    if data.empty or data.iloc[-1][['Close', 'EMA_8', 'EMA_55']].isna().any():
        return (data['Close'].iloc[-1] if not data.empty and pd.notna(data['Close'].iloc[-1]) else None), None
    latest = data.iloc[-1]
    live_price, ema_8, ema_55 = latest['Close'], latest['EMA_8'], latest['EMA_55']
    if pd.isna(live_price) or pd.isna(ema_8) or pd.isna(ema_55) or ema_55 == 0: return live_price, None
    ema_enter = (ema_8 - ema_55) / ema_55
    ema_invest_score = ((ema_enter * 4) + 0.5) * 100
    return float(live_price), float(ema_invest_score)

async def calculate_one_year_invest(ticker: str, is_called_by_ai: bool = False) -> tuple[float, float]:
    ticker_yf_format = ticker.replace('.', '-')
    stock = yf.Ticker(ticker_yf_format)
    try:
        data = await asyncio.to_thread(stock.history, period="1y")
        if data.empty or len(data) < 2 or 'Close' not in data.columns: return 0.0, 50.0
    except Exception as e:
        # if not is_called_by_ai: print(f"1Y Invest: Error fetching history for {ticker}: {e}")
        return 0.0, 50.0
    start_price, end_price = data['Close'].iloc[0], data['Close'].iloc[-1]
    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0: return 0.0, 50.0
    one_year_change = ((end_price - start_price) / start_price) * 100
    invest_per = 50.0
    try:
        invest_per = (one_year_change / 2) + 50 if one_year_change < 0 else math.sqrt(max(0, one_year_change * 5)) + 50
    except ValueError: invest_per = 50.0
    return float(one_year_change), float(max(0, min(invest_per, 100)))

def plot_ticker_graph(ticker: str, ema_interval: int, is_called_by_ai: bool = False) -> Optional[str]:
    ticker_yf_format = ticker.replace('.', '-')
    stock = yf.Ticker(ticker_yf_format)
    interval_map = {1: "1wk", 2: "1d", 3: "1h"}
    period_map = {1: "5y", 2: "1y", 3: "6mo"}
    interval_str = interval_map.get(ema_interval, "1h")
    period_str = period_map.get(ema_interval, "1y")
    try:
        data = stock.history(period=period_str, interval=interval_str)
        if data.empty or 'Close' not in data.columns: raise ValueError("No data")
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(data.index, data['Close'], color='grey', label='Price', linewidth=1.0)
        ax.plot(data.index, data['EMA_55'], color='darkgreen', label='EMA 55', linewidth=1.5)
        ax.plot(data.index, data['EMA_8'], color='firebrick', label='EMA 8', linewidth=1.5)
        ax.set_title(f"{ticker} Price and EMAs ({interval_str})", color='white')
        ax.set_xlabel('Date', color='white'); ax.set_ylabel('Price', color='white')
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax.grid(True, color='dimgray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
        fig.tight_layout()
        filename = f"{ticker}_graph_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black', edgecolor='black')
        plt.close(fig)
        if not is_called_by_ai: print(f"📂 Graph saved: {filename}")
        return filename
    except Exception as e:
        if not is_called_by_ai: print(f"❌ Error plotting graph for {ticker}: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number): plt.close(fig)
        return None

def get_allocation_score(is_called_by_ai: bool = False) -> tuple[float, float, float]:
    avg_s, gen_s, mkt_inv_s = 50.0, 50.0, 50.0 # Defaults
    if not os.path.exists(RISK_CSV_FILE):
        if not is_called_by_ai:
            print(f"Warning: Market data file '{RISK_CSV_FILE}' not found. Using default scores (50.0). Run /risk.")
        return avg_s, gen_s, mkt_inv_s
    try:
        df = pd.read_csv(RISK_CSV_FILE, on_bad_lines='skip')
        if df.empty or not all(c in df.columns for c in ['General Market Score', 'Market Invest Score']):
            if not is_called_by_ai: print(f"Warning: '{RISK_CSV_FILE}' empty or missing columns. Using defaults (50.0).")
            return avg_s, gen_s, mkt_inv_s
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
            df.dropna(subset=['Timestamp'], inplace=True)
            if not df.empty: df = df.sort_values(by='Timestamp', ascending=True)
            # else: # if not is_called_by_ai: print(f"Warning: No valid timestamps in '{RISK_CSV_FILE}'.") # Fallback
        # else: # if not is_called_by_ai: print(f"Warning: 'Timestamp' column not found in '{RISK_CSV_FILE}'.") # Fallback
        if df.empty: return avg_s, gen_s, mkt_inv_s # After processing
        latest = df.iloc[-1]
        gs_val, mis_val = safe_score(latest.get('General Market Score')), safe_score(latest.get('Market Invest Score'))
        if pd.isna(gs_val) or pd.isna(mis_val): return avg_s, gen_s, mkt_inv_s # N/A after conversion
        avg_s_calc = (gs_val + (2 * mis_val)) / 3.0
        avg_s, gen_s, mkt_inv_s = max(0,min(100,avg_s_calc)), max(0,min(100,gs_val)), max(0,min(100,mis_val))
        if not is_called_by_ai: print(f"  get_allocation_score: Using scores: Avg(Sigma)={avg_s:.2f}, Gen={gen_s:.2f}, MktInv={mkt_inv_s:.2f}")
        return avg_s, gen_s, mkt_inv_s
    except Exception as e:
        if not is_called_by_ai: print(f"Error in get_allocation_score: {e}. Using defaults (50.0)."); traceback.print_exc()
        return avg_s, gen_s, mkt_inv_s

# --- NEW HELPER FUNCTIONS FOR /briefing ---

async def get_daily_change_for_tickers(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetches the live price and daily percentage change for a list of tickers.
    Uses yfinance download for efficiency with multiple tickers.
    """
    if not tickers:
        return {}
    
    results = {}
    try:
        # Fetch data for the last 5 days to ensure we have at least two trading days of data
        data = await asyncio.to_thread(
            yf.download,
            tickers=tickers,
            period="5d", # Changed from 2d to 5d for robustness
            interval="1d",
            progress=False,
            timeout=20
        )
        if data.empty:
            return {ticker: {'error': 'No data returned'} for ticker in tickers}

        # Handle single vs multiple ticker download format
        for ticker in tickers:
            ticker_data = None
            if len(tickers) == 1:
                ticker_data = data
            elif ticker in data.columns.get_level_values(1):
                ticker_data = data.xs(ticker, level=1, axis=1)
            
            if ticker_data is not None and not ticker_data.empty and len(ticker_data['Close'].dropna()) > 1:
                valid_closes = ticker_data['Close'].dropna()
                live_price = valid_closes.iloc[-1]
                prev_close = valid_closes.iloc[-2]
                if pd.notna(live_price) and pd.notna(prev_close) and prev_close != 0:
                    daily_change = ((live_price - prev_close) / prev_close) * 100
                    results[ticker] = {'live_price': live_price, 'change_pct': daily_change}
                else:
                    results[ticker] = {'error': 'Price data invalid'}
            else:
                results[ticker] = {'error': 'Insufficient historical data'}
    except Exception as e:
        return {ticker: {'error': str(e)} for ticker in tickers}
    
    return results

async def get_sp500_movers(is_called_by_ai: bool = False) -> Dict[str, List[Dict]]:
    """
    Fetches all S&P 500 stocks and identifies the top 3 and bottom 3 performers for the day.
    """
    if not is_called_by_ai: print("  Briefing: Fetching S&P 500 movers...")
    sp500_symbols = await asyncio.to_thread(get_sp500_symbols_singularity, is_called_by_ai=True)
    if not sp500_symbols:
        return {'top': [], 'bottom': [], 'error': 'Could not fetch S&P 500 symbol list.'}

    # Fetch daily changes for all S&P 500 stocks
    all_changes = await get_daily_change_for_tickers(sp500_symbols)
    
    valid_performers = []
    for ticker, data in all_changes.items():
        if 'change_pct' in data and pd.notna(data['change_pct']):
            valid_performers.append({'ticker': ticker, 'change_pct': data['change_pct']})
            
    if not valid_performers:
        return {'top': [], 'bottom': [], 'error': 'Could not calculate daily changes for S&P 500 stocks.'}

    # Sort by percentage change
    valid_performers.sort(key=lambda x: x['change_pct'], reverse=True)
    
    top_3 = valid_performers[:3]
    bottom_3 = valid_performers[-3:]
    bottom_3.reverse() # To show biggest loser first

    return {'top': top_3, 'bottom': bottom_3}

async def get_multi_period_change(tickers: List[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """
    Fetches 1-day, 1-week, and 1-month percentage change for a list of tickers.
    """
    results = {}
    if not tickers:
        return results

    end_date = datetime.now()
    # Fetch data for a bit more than 1 month to ensure we have data points
    start_date_1m = end_date - timedelta(days=40)
    
    try:
        data = await asyncio.to_thread(
            yf.download,
            tickers=tickers,
            start=start_date_1m.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            progress=False,
            timeout=20
        )
        if data.empty:
            return {ticker: {'1D': None, '1W': None, '1M': None} for ticker in tickers}

        # Isolate the 'Close' prices DataFrame, which is more robust
        close_data = data.get('Close')

        for ticker in tickers:
            # Get the 'Close' price series for the current ticker
            close_prices = None
            if len(tickers) == 1 and isinstance(close_data, pd.Series):
                close_prices = close_data
            elif isinstance(close_data, pd.DataFrame) and ticker in close_data.columns:
                close_prices = close_data[ticker]

            if close_prices is not None and not close_prices.empty:
                close_prices = close_prices.dropna()
                if len(close_prices) > 1:
                    # Add checks for division by zero
                    change_1d = ((close_prices.iloc[-1] - close_prices.iloc[-2]) / close_prices.iloc[-2]) * 100 if len(close_prices) >= 2 and close_prices.iloc[-2] != 0 else None
                    change_1w = ((close_prices.iloc[-1] - close_prices.iloc[-6]) / close_prices.iloc[-6]) * 100 if len(close_prices) >= 6 and close_prices.iloc[-6] != 0 else None
                    change_1m = ((close_prices.iloc[-1] - close_prices.iloc[-22]) / close_prices.iloc[-22]) * 100 if len(close_prices) >= 22 and close_prices.iloc[-22] != 0 else None
                    results[ticker] = {'1D': change_1d, '1W': change_1w, '1M': change_1m}
                else:
                    results[ticker] = {'1D': None, '1W': None, '1M': None}
            else:
                results[ticker] = {'1D': None, '1W': None, '1M': None}
    except Exception:
        results = {ticker: {'1D': None, '1W': None, '1M': None} for ticker in tickers}
        
    return results

async def process_custom_portfolio(
    portfolio_data_config: Dict[str, Any],
    tailor_portfolio_requested: bool,
    frac_shares_singularity: bool,
    total_value_singularity: Optional[float] = None, 
    is_custom_command_simplified_output: bool = False,
    is_called_by_ai: bool = False
) -> tuple[List[str], List[Dict[str, Any]], float, List[Dict[str, Any]]]:
    """
    Processes custom or /invest portfolio requests.
    Calculates scores, allocations, generates output tables, graphs, and a pie chart.
    Returns:
        - tailored_portfolio_output_list_final (List[str]): Simplified list of tailored holdings for CLI string output.
        - final_combined_portfolio_data_calc (List[Dict]): Percentage allocations data before tailoring.
        - final_cash_value_tailored (float): Cash remaining after tailoring.
        - tailored_portfolio_structured_data (List[Dict]): Detailed structured list of tailored holdings (stocks/hedges).
    """
    suppress_prints = is_custom_command_simplified_output or is_called_by_ai
    sell_to_cash_active = False
    
    avg_score, _, _ = get_allocation_score(is_called_by_ai=suppress_prints)

    if avg_score is not None and avg_score < 50.0:
        sell_to_cash_active = True
        if not suppress_prints:
            print(f"\n:warning: **Sell-to-Cash Feature Active!** (Avg Market Score: {avg_score:.2f} < 50).")

    if isinstance(portfolio_data_config, pd.Series): # Handle if config is a pandas Series from CSV
        portfolio_data_config = portfolio_data_config.to_dict()

    ema_sensitivity = int(safe_score(portfolio_data_config.get('ema_sensitivity', 3)))
    amplification = float(safe_score(portfolio_data_config.get('amplification', 1.0)))
    num_portfolios = int(safe_score(portfolio_data_config.get('num_portfolios', 0)))

    portfolio_results_list = []
    all_entries_for_graphs_plotting = []

    for i in range(num_portfolios):
        portfolio_index = i + 1
        tickers_str = portfolio_data_config.get(f'tickers_{portfolio_index}', '')
        weight = safe_score(portfolio_data_config.get(f'weight_{portfolio_index}', '0'))
        tickers = [ticker.strip().upper() for ticker in str(tickers_str).split(',') if ticker.strip()]
        if not tickers:
            continue

        current_portfolio_list_calc = []
        for ticker in tickers:
            try:
                live_price, ema_invest = await calculate_ema_invest(ticker, ema_sensitivity, is_called_by_ai=suppress_prints)
                if live_price is None and ema_invest is None: 
                    current_portfolio_list_calc.append({'ticker': ticker, 'error': "Failed to fetch critical data", 'portfolio_weight': weight})
                    all_entries_for_graphs_plotting.append({'ticker': ticker, 'error': "Failed to fetch critical data"})
                    continue
                
                ema_invest_score = 50.0 if ema_invest is None else ema_invest
                live_price_val = 0.0 if live_price is None else live_price 

                _, _ = await calculate_one_year_invest(ticker, is_called_by_ai=suppress_prints)

                raw_combined_invest_score = safe_score(ema_invest_score)
                score_for_allocation_logic = raw_combined_invest_score
                score_was_adjusted_flag = False

                if sell_to_cash_active and raw_combined_invest_score < 50.0:
                    score_for_allocation_logic = 50.0 
                    score_was_adjusted_flag = True
                
                amplified_score_adjusted_calc = safe_score((score_for_allocation_logic * amplification) - (amplification - 1) * 50)
                amplified_score_adjusted_clamped = max(0, amplified_score_adjusted_calc)
                
                amplified_score_original_calc = safe_score((raw_combined_invest_score * amplification) - (amplification - 1) * 50)
                amplified_score_original_clamped = max(0, amplified_score_original_calc)

                entry_data = {
                    'ticker': ticker, 'live_price': live_price_val, 'raw_invest_score': raw_combined_invest_score,
                    'amplified_score_adjusted': amplified_score_adjusted_clamped,
                    'amplified_score_original': amplified_score_original_clamped, 
                    'portfolio_weight': weight, 'score_was_adjusted': score_was_adjusted_flag,
                    'portfolio_allocation_percent_adjusted': None, 
                    'portfolio_allocation_percent_original': None, 
                    'combined_percent_allocation_adjusted': None, 
                    'combined_percent_allocation_original': None, 
                }
                current_portfolio_list_calc.append(entry_data)
                if live_price_val > 0: 
                    all_entries_for_graphs_plotting.append({'ticker': ticker, 'ema_sensitivity': ema_sensitivity})
            except Exception as e_ticker_loop:
                current_portfolio_list_calc.append({'ticker': ticker, 'error': str(e_ticker_loop), 'portfolio_weight': weight})
                all_entries_for_graphs_plotting.append({'ticker': ticker, 'error': str(e_ticker_loop)})
        portfolio_results_list.append(current_portfolio_list_calc)

    sent_graphs = set()
    if not suppress_prints:
        for graph_entry in all_entries_for_graphs_plotting:
            ticker_key_graph = graph_entry.get('ticker')
            if not ticker_key_graph or ticker_key_graph in sent_graphs: continue 
            if 'error' not in graph_entry: 
                # plot_ticker_graph is synchronous, run in thread
                await asyncio.to_thread(plot_ticker_graph, ticker_key_graph, graph_entry['ema_sensitivity'], is_called_by_ai=suppress_prints)
                sent_graphs.add(ticker_key_graph)

    for portfolio_list_item in portfolio_results_list:
        portfolio_amplified_total_adjusted = safe_score(sum(entry['amplified_score_adjusted'] for entry in portfolio_list_item if 'error' not in entry))
        for entry in portfolio_list_item:
            if 'error' not in entry:
                if portfolio_amplified_total_adjusted > 0:
                    entry['portfolio_allocation_percent_adjusted'] = round(safe_score((entry.get('amplified_score_adjusted', 0) / portfolio_amplified_total_adjusted) * 100), 2)
                else: entry['portfolio_allocation_percent_adjusted'] = 0.0
            else: entry['portfolio_allocation_percent_adjusted'] = None
        
        portfolio_amplified_total_original = safe_score(sum(entry['amplified_score_original'] for entry in portfolio_list_item if 'error' not in entry))
        for entry in portfolio_list_item:
            if 'error' not in entry:
                if portfolio_amplified_total_original > 0:
                    entry['portfolio_allocation_percent_original'] = round(safe_score((entry.get('amplified_score_original', 0) / portfolio_amplified_total_original) * 100), 2)
                else: entry['portfolio_allocation_percent_original'] = 0.0
            else: entry['portfolio_allocation_percent_original'] = None 
    
    if not suppress_prints:
        print("\n--- Sub-Portfolio Details ---")
        for i, portfolio_list_print in enumerate(portfolio_results_list, 1):
            portfolio_list_print.sort(key=lambda x: x.get('portfolio_allocation_percent_adjusted', -1) if x.get('portfolio_allocation_percent_adjusted') is not None else -1, reverse=True)
            portfolio_weight_display_val = portfolio_list_print[0].get('portfolio_weight', 'N/A') if portfolio_list_print and 'error' not in portfolio_list_print[0] else 'N/A'
            print(f"\n**--- Sub-Portfolio {i} (Weight: {portfolio_weight_display_val}%) ---**")
            table_data_sub_print = []
            for entry_print in portfolio_list_print:
                if 'error' not in entry_print:
                    live_price_f = f"${entry_print.get('live_price', 0):.2f}"
                    invest_score_val_f = safe_score(entry_print.get('raw_invest_score', 0)) 
                    invest_score_f = f"{invest_score_val_f:.2f}%" if invest_score_val_f is not None else "N/A"
                    amplified_score_f_sub = f"{entry_print.get('amplified_score_adjusted', 0):.2f}%"
                    port_alloc_val_original_f = safe_score(entry_print.get('portfolio_allocation_percent_original', 0))
                    port_alloc_f_sub = f"{port_alloc_val_original_f:.2f}%" if port_alloc_val_original_f is not None else "N/A"
                    table_data_sub_print.append([entry_print.get('ticker', 'ERR'), live_price_f, invest_score_f, amplified_score_f_sub, port_alloc_f_sub])
            
            if not table_data_sub_print: print("No valid data for this sub-portfolio.")
            else: print(tabulate(table_data_sub_print, headers=["Ticker", "Live Price", "Raw Score", "Adj Amplified %", "Portfolio % Alloc (Original)"], tablefmt="pretty"))
            
            error_messages_sub = [f"Error for {entry.get('ticker', 'UNKNOWN')}: {entry.get('error', 'Unknown error')}" for entry in portfolio_list_print if 'error' in entry]
            if error_messages_sub: print(f"Errors in Sub-Portfolio {i}:\n" + "\n".join(error_messages_sub))

    combined_result_intermediate_calc = []
    for portfolio_list_intermediate in portfolio_results_list:
        for entry_intermediate in portfolio_list_intermediate:
            if 'error' not in entry_intermediate:
                port_weight_intermediate = entry_intermediate.get('portfolio_weight', 0)
                sub_alloc_adj_intermediate = entry_intermediate.get('portfolio_allocation_percent_adjusted', 0)
                entry_intermediate['combined_percent_allocation_adjusted'] = round(safe_score((sub_alloc_adj_intermediate * port_weight_intermediate) / 100), 4)
                sub_alloc_orig_intermediate = entry_intermediate.get('portfolio_allocation_percent_original', 0)
                entry_intermediate['combined_percent_allocation_original'] = round(safe_score((sub_alloc_orig_intermediate * port_weight_intermediate) / 100), 4)
                combined_result_intermediate_calc.append(entry_intermediate)

    final_combined_portfolio_data_calc = []
    cash_allocation_from_sell_to_cash_feature_pct = 0.0

    for entry_comb in combined_result_intermediate_calc:
        final_combined_portfolio_data_calc.append({
            'ticker': entry_comb['ticker'],
            'live_price': entry_comb['live_price'],
            'raw_invest_score': entry_comb['raw_invest_score'],
            'amplified_score_adjusted': entry_comb['amplified_score_adjusted'],
            'combined_percent_allocation': entry_comb['combined_percent_allocation_adjusted'] 
        })
        if sell_to_cash_active and entry_comb.get('score_was_adjusted', False):
            difference_for_cash_calc = entry_comb['combined_percent_allocation_adjusted'] - entry_comb['combined_percent_allocation_original']
            cash_allocation_from_sell_to_cash_feature_pct += max(0.0, difference_for_cash_calc)

    if sell_to_cash_active and cash_allocation_from_sell_to_cash_feature_pct > 0:
        cash_allocation_from_sell_to_cash_feature_pct = min(cash_allocation_from_sell_to_cash_feature_pct, 100.0)
        target_total_stock_allocation_pct = 100.0 - cash_allocation_from_sell_to_cash_feature_pct
        current_total_stock_allocation_pct = sum(item['combined_percent_allocation'] for item in final_combined_portfolio_data_calc if item['ticker'] != 'Cash')

        if current_total_stock_allocation_pct > 1e-9: 
            normalization_factor_for_stocks = target_total_stock_allocation_pct / current_total_stock_allocation_pct
            for item in final_combined_portfolio_data_calc:
                if item['ticker'] != 'Cash':
                    item['combined_percent_allocation'] *= normalization_factor_for_stocks
        elif target_total_stock_allocation_pct == 0.0: 
            for item in final_combined_portfolio_data_calc:
                if item['ticker'] != 'Cash':
                    item['combined_percent_allocation'] = 0.0
        
        final_combined_portfolio_data_calc.append({
            'ticker': 'Cash', 'live_price': 1.0, 'raw_invest_score': None,
            'amplified_score_adjusted': None, 'combined_percent_allocation': cash_allocation_from_sell_to_cash_feature_pct
        })
    
    current_total_allocation_final_norm = sum(item.get('combined_percent_allocation', 0.0) for item in final_combined_portfolio_data_calc)
    if not math.isclose(current_total_allocation_final_norm, 100.0, abs_tol=0.01) and current_total_allocation_final_norm > 1e-9:
        norm_factor_final_val = 100.0 / current_total_allocation_final_norm
        cash_item_final = next((item for item in final_combined_portfolio_data_calc if item['ticker'] == 'Cash'), None)
        
        for item_norm in final_combined_portfolio_data_calc:
            if item_norm['ticker'] != 'Cash':
                item_norm['combined_percent_allocation'] = item_norm.get('combined_percent_allocation', 0.0) * norm_factor_final_val
        
        current_stock_sum_after_final_norm = sum(item.get('combined_percent_allocation', 0.0) for item in final_combined_portfolio_data_calc if item['ticker'] != 'Cash')
        
        if cash_item_final:
            cash_item_final['combined_percent_allocation'] = max(0.0, 100.0 - current_stock_sum_after_final_norm)
            cash_item_final['combined_percent_allocation'] = min(cash_item_final['combined_percent_allocation'], 100.0) 
        elif (100.0 - current_stock_sum_after_final_norm) > 1e-4: 
             final_combined_portfolio_data_calc.append({
                'ticker': 'Cash', 'live_price': 1.0, 'raw_invest_score': None,
                'amplified_score_adjusted': None, 'combined_percent_allocation': max(0.0, 100.0 - current_stock_sum_after_final_norm)
            })
    
    # Sort by raw_invest_score for the "Final Combined Portfolio" display, before tailoring
    final_combined_portfolio_data_calc.sort(
        key=lambda x: x.get('raw_invest_score', -float('inf')) if x.get('ticker') != 'Cash' else -float('inf')-1, 
        reverse=True
    )
    
    if not suppress_prints:
        print("\n**--- Final Combined Portfolio (Sorted by Raw Score)---**")
        cash_present_in_final = any(item['ticker'] == 'Cash' and item.get('combined_percent_allocation', 0) > 1e-4 for item in final_combined_portfolio_data_calc)
        if sell_to_cash_active and cash_present_in_final: print("*(Sell-to-Cash Active, resulting in cash allocation)*")
        elif sell_to_cash_active: print("*(Sell-to-Cash Active, but might not have resulted in explicit cash if all scores were high)*")

        combined_data_display_final = []
        for entry_disp_final in final_combined_portfolio_data_calc:
            ticker_disp = entry_disp_final.get('ticker', 'ERR')
            live_price_f_disp, invest_score_f_disp, amplified_score_f_disp = '-', '-', '-'
            if ticker_disp != 'Cash':
                live_price_f_disp = f"${entry_disp_final.get('live_price', 0):.2f}"
                raw_score_val = entry_disp_final.get('raw_invest_score')
                invest_score_f_disp = f"{safe_score(raw_score_val):.2f}%" if raw_score_val is not None else "N/A"
                amplified_score_val = entry_disp_final.get('amplified_score_adjusted')
                amplified_score_f_disp = f"{safe_score(amplified_score_val):.2f}%" if amplified_score_val is not None else "N/A"
            
            comb_alloc_val = entry_disp_final.get('combined_percent_allocation', 0)
            comb_alloc_f_disp = f"{round(comb_alloc_val, 2):.2f}%" if comb_alloc_val is not None else "N/A"
            combined_data_display_final.append([ticker_disp, live_price_f_disp, invest_score_f_disp, amplified_score_f_disp, comb_alloc_f_disp])
        
        if not combined_data_display_final: print("No valid data for the combined portfolio.")
        else: print(tabulate(combined_data_display_final, headers=["Ticker", "Live Price", "Raw Score", "Basis Amplified %", "Final % Alloc"], tablefmt="pretty"))

    # --- Tailoring Section ---
    tailored_portfolio_output_list_final = []
    tailored_portfolio_structured_data = []
    final_cash_value_tailored = 0.0 # Default if not tailoring

    if tailor_portfolio_requested:
        if total_value_singularity is None: # Should not happen if tailor_portfolio_requested is True based on typical calling logic
             if not suppress_prints:
                print("Info: Tailoring not performed as total_value_singularity is not specified (this should ideally be caught earlier).")
             return [], final_combined_portfolio_data_calc, 0.0, []

        total_value_float_for_tailor = safe_score(total_value_singularity)
        if total_value_float_for_tailor <= 0: # For AI, allow 0 if it's an intentional test; for CLI, error out.
             if not is_called_by_ai and not suppress_prints:
                print("Error: Tailored portfolio requested but total value is zero or negative. Cannot tailor.")
             # If AI sent 0, it might be testing an edge case; let it proceed but result will be all cash.
             # For CLI, this is an error.
             if not is_called_by_ai:
                 return [], final_combined_portfolio_data_calc, total_value_float_for_tailor, [] # Return error state for CLI
             # For AI, if value is 0, all will be cash.
             final_cash_value_tailored = total_value_float_for_tailor # Which is 0 or negative
             # Fall through to print tailored output (which will be just cash)

        current_tailored_entries_for_calc = []
        total_actual_money_spent_on_stocks = 0.0
        remaining_portfolio_value_for_allocation = total_value_float_for_tailor

        # Sort stocks by their target combined_percent_allocation to prioritize when capital is constrained.
        # The `final_combined_portfolio_data_calc` was sorted by raw_invest_score for display. Re-sort for allocation.
        alloc_priority_sorted_list = sorted(
            [item for item in final_combined_portfolio_data_calc if item['ticker'] != 'Cash'], # Exclude cash from sorting for allocation
            key=lambda x: x.get('combined_percent_allocation', 0.0), 
            reverse=True
        )

        for entry_tailoring in alloc_priority_sorted_list:
            # No need to check for 'Cash' here as it's already filtered out

            final_stock_alloc_pct_tailor = safe_score(entry_tailoring.get('combined_percent_allocation', 0.0))
            live_price_for_tailor = safe_score(entry_tailoring.get('live_price', 0.0))

            if final_stock_alloc_pct_tailor > 1e-9 and live_price_for_tailor > 0 and remaining_portfolio_value_for_allocation > 1e-7 : # Only try if cash left
                # Ideal dollar allocation based on its percentage of the *original total portfolio value*
                ideal_dollar_allocation_for_ticker = total_value_float_for_tailor * (final_stock_alloc_pct_tailor / 100.0)
                
                shares_to_buy_ideal_calc = 0.0
                if frac_shares_singularity:
                    # For fractional, calculate shares as precisely as possible initially based on ideal dollar amount
                    shares_to_buy_ideal_calc = ideal_dollar_allocation_for_ticker / live_price_for_tailor
                    # Round to a practical number of decimal places, e.g., 1 as per prior logic
                    shares_to_buy_ideal_calc = round(shares_to_buy_ideal_calc, 1) 
                else: # Whole shares
                    shares_to_buy_ideal_calc = float(math.floor(ideal_dollar_allocation_for_ticker / live_price_for_tailor))
                
                shares_to_buy_ideal_calc = max(0.0, shares_to_buy_ideal_calc)
                cost_of_ideal_shares = shares_to_buy_ideal_calc * live_price_for_tailor

                shares_to_buy_final_for_this_stock = 0.0
                actual_money_allocated_this_ticker = 0.0

                if cost_of_ideal_shares <= remaining_portfolio_value_for_allocation + 1e-7: # Check with tolerance
                    shares_to_buy_final_for_this_stock = shares_to_buy_ideal_calc
                    actual_money_allocated_this_ticker = cost_of_ideal_shares
                else: # Cannot afford ideal, buy what's possible with remaining cash
                    if frac_shares_singularity:
                        affordable_shares_frac = remaining_portfolio_value_for_allocation / live_price_for_tailor
                        shares_to_buy_final_for_this_stock = round(affordable_shares_frac, 1) 
                        # Crucial: ensure that rounding doesn't make it exceed remaining cash
                        if shares_to_buy_final_for_this_stock * live_price_for_tailor > remaining_portfolio_value_for_allocation:
                            shares_to_buy_final_for_this_stock = math.floor(affordable_shares_frac * 10.0) / 10.0 # round down to 1 decimal
                    else: # Whole shares
                        shares_to_buy_final_for_this_stock = float(math.floor(remaining_portfolio_value_for_allocation / live_price_for_tailor))
                    
                    shares_to_buy_final_for_this_stock = max(0.0, shares_to_buy_final_for_this_stock) # Ensure non-negative
                    actual_money_allocated_this_ticker = shares_to_buy_final_for_this_stock * live_price_for_tailor
                
                min_share_purchase_threshold = 0.1 if frac_shares_singularity else 1.0
                
                if shares_to_buy_final_for_this_stock >= min_share_purchase_threshold:
                    actual_percent_of_total_value = (actual_money_allocated_this_ticker / total_value_float_for_tailor) * 100.0 if total_value_float_for_tailor > 0 else 0.0
                    current_tailored_entries_for_calc.append({
                        'ticker': entry_tailoring.get('ticker','ERR'),
                        'raw_invest_score': entry_tailoring.get('raw_invest_score', -float('inf')),
                        'shares': shares_to_buy_final_for_this_stock,
                        'live_price_at_eval': live_price_for_tailor,
                        'actual_money_allocation': actual_money_allocated_this_ticker,
                        'actual_percent_allocation': actual_percent_of_total_value 
                    })
                    total_actual_money_spent_on_stocks += actual_money_allocated_this_ticker
                    remaining_portfolio_value_for_allocation -= actual_money_allocated_this_ticker
                    remaining_portfolio_value_for_allocation = max(0.0, remaining_portfolio_value_for_allocation) # Ensure remaining doesn't dip from float math
        
        final_cash_value_tailored = total_value_float_for_tailor - total_actual_money_spent_on_stocks
        # Ensure final cash is not negative due to tiny floating point residuals after all calculations
        final_cash_value_tailored = max(0.0, final_cash_value_tailored) 
        
        # Sort the actual holdings by raw_invest_score for display/output consistency with previous logic
        current_tailored_entries_for_calc.sort(key=lambda x: safe_score(x.get('raw_invest_score', -float('inf'))), reverse=True)
        tailored_portfolio_structured_data = current_tailored_entries_for_calc

        # Output for CLI / AI simplified response
        if not suppress_prints or is_custom_command_simplified_output:
            print("\n--- Tailored Portfolio (Shares) ---")
            if current_tailored_entries_for_calc:
                share_format_string_cli = "{:.1f}" if frac_shares_singularity else "{:.0f}"
                tailored_portfolio_output_list_final = [f"{item['ticker']} - {share_format_string_cli.format(item['shares'])} shares" for item in current_tailored_entries_for_calc]
                print("\n".join(tailored_portfolio_output_list_final))
            else:
                print("No stocks allocated in the tailored portfolio based on the provided value and strategy.")
            print(f"Final Cash Value: ${safe_score(final_cash_value_tailored):,.2f}") # This will now be non-negative

        # Full details table for CLI
        if not suppress_prints and not is_custom_command_simplified_output:
            print("\n--- Tailored Portfolio (Full Details) ---")
            tailored_portfolio_table_data_display_full = []
            share_format_string_table = "{:.1f}" if frac_shares_singularity else "{:.0f}"
            for item_table in current_tailored_entries_for_calc: # Already sorted by raw score
                shares_display_table = share_format_string_table.format(item_table['shares'])
                tailored_portfolio_table_data_display_full.append([
                    item_table['ticker'],
                    shares_display_table,
                    f"${safe_score(item_table['actual_money_allocation']):,.2f}",
                    f"{safe_score(item_table['actual_percent_allocation']):.2f}%"
                ])
            
            final_cash_percent_display_val = (final_cash_value_tailored / total_value_float_for_tailor) * 100.0 if total_value_float_for_tailor > 0 else (100.0 if math.isclose(total_value_float_for_tailor,0) and math.isclose(final_cash_value_tailored,0) else 0.0)
            tailored_portfolio_table_data_display_full.append(['Cash', '-', f"${safe_score(final_cash_value_tailored):,.2f}", f"{safe_score(final_cash_percent_display_val):.2f}%"])
            
            if not tailored_portfolio_structured_data and math.isclose(final_cash_value_tailored, total_value_float_for_tailor):
                 print("No stocks allocated. All value remains as cash.")
            elif not tailored_portfolio_table_data_display_full : 
                print("Error: No data for tailored portfolio display (this should not happen).")
            else:
                print(tabulate(tailored_portfolio_table_data_display_full, headers=["Ticker", "Shares", "Actual $ Allocation", "Actual % Allocation"], tablefmt="pretty"))

        # Pie chart generation
        pie_chart_data_for_gen = []
        if tailored_portfolio_structured_data: # Use the final structured data
            for item_pie in tailored_portfolio_structured_data:
                # Only include positive allocations in pie chart
                if item_pie.get('actual_money_allocation', 0) > 1e-9:
                    pie_chart_data_for_gen.append({'ticker': item_pie['ticker'], 'value': item_pie['actual_money_allocation']})
        
        if final_cash_value_tailored > 1e-9: # Only include cash in pie chart if it's positive
            pie_chart_data_for_gen.append({'ticker': 'Cash', 'value': final_cash_value_tailored})

        if pie_chart_data_for_gen and (not suppress_prints or not is_custom_command_simplified_output):
            chart_title_base_str = "Invest Portfolio" # Default
            # Handle if portfolio_data_config is dict or Series (from CSV)
            if isinstance(portfolio_data_config, pd.Series): 
                 chart_title_base_str = portfolio_data_config.get('portfolio_code', "Invest Portfolio")
            elif isinstance(portfolio_data_config, dict):
                 chart_title_base_str = portfolio_data_config.get('portfolio_code', "Invest Portfolio")

            chart_title_final_str = f"{chart_title_base_str} Allocation (Value: ${safe_score(total_value_singularity if total_value_singularity is not None else 0):,.0f})"
            # generate_portfolio_pie_chart is async, run in thread if its internals are blocking (like plt.savefig)
            await asyncio.to_thread(generate_portfolio_pie_chart, pie_chart_data_for_gen, chart_title_final_str, "singularity_portfolio_pie", is_called_by_ai=suppress_prints)
        
        if not suppress_prints: # Final confirmation print
             print(f"Remaining Buying Power (Final Cash in Tailored Portfolio): ${safe_score(final_cash_value_tailored):,.2f}")
    
    # If not tailoring, final_cash_value_tailored remains its default (0.0)
    # tailored_portfolio_structured_data remains empty
    # tailored_portfolio_output_list_final remains empty
    # The key is that final_combined_portfolio_data_calc (percentage-based) is always returned.
    return tailored_portfolio_output_list_final, final_combined_portfolio_data_calc, final_cash_value_tailored, tailored_portfolio_structured_data
async def process_custom_portfolio(
    portfolio_data_config: Dict[str, Any],
    tailor_portfolio_requested: bool,
    frac_shares_singularity: bool,
    total_value_singularity: Optional[float] = None,
    is_custom_command_simplified_output: bool = False,
    is_called_by_ai: bool = False
) -> tuple[List[str], List[Dict[str, Any]], float, List[Dict[str, Any]]]:
    suppress_prints = is_custom_command_simplified_output or is_called_by_ai
    sell_to_cash_active = False
    
    # Ensure get_allocation_score and other helpers are correctly defined and await if async
    avg_score, _, _ = get_allocation_score(is_called_by_ai=suppress_prints) 

    if avg_score is not None and avg_score < 50.0:
        sell_to_cash_active = True
        if not suppress_prints:
            print(f"\n:warning: **Sell-to-Cash Feature Active!** (Avg Market Score: {avg_score:.2f} < 50).")

    if isinstance(portfolio_data_config, pd.Series):
        portfolio_data_config = portfolio_data_config.to_dict()

    ema_sensitivity = int(safe_score(portfolio_data_config.get('ema_sensitivity', 3)))
    amplification = float(safe_score(portfolio_data_config.get('amplification', 1.0)))
    num_portfolios = int(safe_score(portfolio_data_config.get('num_portfolios', 0)))

    portfolio_results_list = []
    all_entries_for_graphs_plotting = []

    for i in range(num_portfolios):
        portfolio_index = i + 1
        tickers_str = portfolio_data_config.get(f'tickers_{portfolio_index}', '')
        weight = safe_score(portfolio_data_config.get(f'weight_{portfolio_index}', '0'))
        tickers = [ticker.strip().upper() for ticker in str(tickers_str).split(',') if ticker.strip()]
        if not tickers: continue

        current_portfolio_list_calc = []
        for ticker in tickers:
            try:
                live_price, ema_invest = await calculate_ema_invest(ticker, ema_sensitivity, is_called_by_ai=suppress_prints)
                if live_price is None and ema_invest is None: 
                    current_portfolio_list_calc.append({'ticker': ticker, 'error': "Failed to fetch critical data", 'portfolio_weight': weight})
                    all_entries_for_graphs_plotting.append({'ticker': ticker, 'error': "Failed to fetch critical data"})
                    continue
                
                ema_invest_score = 50.0 if ema_invest is None else ema_invest
                live_price_val = 0.0 if live_price is None else live_price 
                _, _ = await calculate_one_year_invest(ticker, is_called_by_ai=suppress_prints)
                raw_combined_invest_score = safe_score(ema_invest_score)
                score_for_allocation_logic = raw_combined_invest_score
                score_was_adjusted_flag = False

                if sell_to_cash_active and raw_combined_invest_score < 50.0:
                    score_for_allocation_logic = 50.0 
                    score_was_adjusted_flag = True
                
                amplified_score_adjusted_calc = safe_score((score_for_allocation_logic * amplification) - (amplification - 1) * 50)
                amplified_score_adjusted_clamped = max(0, amplified_score_adjusted_calc)
                amplified_score_original_calc = safe_score((raw_combined_invest_score * amplification) - (amplification - 1) * 50)
                amplified_score_original_clamped = max(0, amplified_score_original_calc)

                entry_data = {
                    'ticker': ticker, 'live_price': live_price_val, 'raw_invest_score': raw_combined_invest_score,
                    'amplified_score_adjusted': amplified_score_adjusted_clamped,
                    'amplified_score_original': amplified_score_original_clamped, 
                    'portfolio_weight': weight, 'score_was_adjusted': score_was_adjusted_flag,
                    'portfolio_allocation_percent_adjusted': None, 'portfolio_allocation_percent_original': None, 
                    'combined_percent_allocation_adjusted': None, 'combined_percent_allocation_original': None, 
                }
                current_portfolio_list_calc.append(entry_data)
                if live_price_val > 0: 
                    all_entries_for_graphs_plotting.append({'ticker': ticker, 'ema_sensitivity': ema_sensitivity})
            except Exception as e_ticker_loop:
                current_portfolio_list_calc.append({'ticker': ticker, 'error': str(e_ticker_loop), 'portfolio_weight': weight})
                all_entries_for_graphs_plotting.append({'ticker': ticker, 'error': str(e_ticker_loop)})
        portfolio_results_list.append(current_portfolio_list_calc)

    sent_graphs = set()
    if not suppress_prints:
        for graph_entry in all_entries_for_graphs_plotting:
            ticker_key_graph = graph_entry.get('ticker')
            if not ticker_key_graph or ticker_key_graph in sent_graphs: continue 
            if 'error' not in graph_entry: 
                await asyncio.to_thread(plot_ticker_graph, ticker_key_graph, graph_entry['ema_sensitivity'], is_called_by_ai=suppress_prints)
                sent_graphs.add(ticker_key_graph)

    for portfolio_list_item in portfolio_results_list:
        portfolio_amplified_total_adjusted = safe_score(sum(entry['amplified_score_adjusted'] for entry in portfolio_list_item if 'error' not in entry))
        for entry in portfolio_list_item:
            if 'error' not in entry:
                entry['portfolio_allocation_percent_adjusted'] = (safe_score(entry.get('amplified_score_adjusted', 0) / portfolio_amplified_total_adjusted) * 100) if portfolio_amplified_total_adjusted > 0 else 0.0
        portfolio_amplified_total_original = safe_score(sum(entry['amplified_score_original'] for entry in portfolio_list_item if 'error' not in entry))
        for entry in portfolio_list_item:
            if 'error' not in entry:
                entry['portfolio_allocation_percent_original'] = (safe_score(entry.get('amplified_score_original', 0) / portfolio_amplified_total_original) * 100) if portfolio_amplified_total_original > 0 else 0.0
    
    # ... (Sub-portfolio printing logic - unchanged) ...

    combined_result_intermediate_calc = []
    for plist_item in portfolio_results_list:
        for entry_inter in plist_item:
            if 'error' not in entry_inter:
                port_weight_inter = entry_inter.get('portfolio_weight', 0)
                sub_alloc_adj_inter = entry_inter.get('portfolio_allocation_percent_adjusted', 0)
                entry_inter['combined_percent_allocation_adjusted'] = round(safe_score((sub_alloc_adj_inter * port_weight_inter) / 100), 4)
                sub_alloc_orig_inter = entry_inter.get('portfolio_allocation_percent_original', 0)
                entry_inter['combined_percent_allocation_original'] = round(safe_score((sub_alloc_orig_inter * port_weight_inter) / 100), 4)
                combined_result_intermediate_calc.append(entry_inter)

    final_combined_portfolio_data_calc = []
    cash_allocation_from_sell_to_cash_feature_pct = 0.0
    for entry_comb in combined_result_intermediate_calc:
        final_combined_portfolio_data_calc.append({
            'ticker': entry_comb['ticker'], 'live_price': entry_comb['live_price'],
            'raw_invest_score': entry_comb['raw_invest_score'],
            'amplified_score_adjusted': entry_comb['amplified_score_adjusted'],
            'combined_percent_allocation': entry_comb['combined_percent_allocation_adjusted'] 
        })
        if sell_to_cash_active and entry_comb.get('score_was_adjusted', False):
            cash_allocation_from_sell_to_cash_feature_pct += max(0.0, entry_comb['combined_percent_allocation_adjusted'] - entry_comb['combined_percent_allocation_original'])

    # ... (Sell-to-cash adjustment and final normalization logic - unchanged) ...
    # Ensure final_combined_portfolio_data_calc includes 'Cash' row if sell_to_cash is active or due to normalization.
    # Final sort by raw_invest_score for display before tailoring.
    final_combined_portfolio_data_calc.sort(
        key=lambda x: x.get('raw_invest_score', -float('inf')) if x.get('ticker', '').upper() != 'CASH' else -float('inf')-1, 
        reverse=True
    )
    # ... (Printing of "Final Combined Portfolio" table - unchanged) ...

    tailored_portfolio_output_list_final = []
    tailored_portfolio_structured_data = []
    final_cash_value_tailored = 0.0

    if tailor_portfolio_requested:
        if total_value_singularity is None:
             if not suppress_prints: print("Info: Tailoring not performed as total_value_singularity is not specified.")
             return [], final_combined_portfolio_data_calc, 0.0, []

        total_value_float_for_tailor = safe_score(total_value_singularity)
        if total_value_float_for_tailor <= 0:
             if not is_called_by_ai and not suppress_prints: print("Error: Tailored portfolio requested but total value is zero or negative.")
             if not is_called_by_ai: return [], final_combined_portfolio_data_calc, total_value_float_for_tailor, []
             final_cash_value_tailored = total_value_float_for_tailor # Will be <=0, results in all cash
             # Fall through to print tailored output which will be just cash.

        current_tailored_entries_for_calc = []
        total_actual_money_spent_on_stocks = 0.0
        remaining_portfolio_value_for_allocation = total_value_float_for_tailor

        # Sort for allocation priority (e.g., by target percentage)
        alloc_priority_sorted_list = sorted(
            [item for item in final_combined_portfolio_data_calc if item['ticker'].upper() != 'CASH'], 
            key=lambda x: x.get('combined_percent_allocation', 0.0), 
            reverse=True
        )

        for entry_tailoring in alloc_priority_sorted_list:
            final_stock_alloc_pct_tailor = safe_score(entry_tailoring.get('combined_percent_allocation', 0.0))
            live_price_for_tailor = safe_score(entry_tailoring.get('live_price', 0.0))

            if final_stock_alloc_pct_tailor > 1e-9 and live_price_for_tailor > 0 and remaining_portfolio_value_for_allocation > 1e-7 :
                ideal_dollar_allocation_for_ticker = total_value_float_for_tailor * (final_stock_alloc_pct_tailor / 100.0)
                shares_to_buy_ideal_calc = 0.0
                if frac_shares_singularity:
                    shares_to_buy_ideal_calc = round(ideal_dollar_allocation_for_ticker / live_price_for_tailor, 1) 
                else:
                    shares_to_buy_ideal_calc = float(math.floor(ideal_dollar_allocation_for_ticker / live_price_for_tailor))
                
                shares_to_buy_ideal_calc = max(0.0, shares_to_buy_ideal_calc)
                cost_of_ideal_shares = shares_to_buy_ideal_calc * live_price_for_tailor
                shares_to_buy_final_for_this_stock = 0.0
                actual_money_allocated_this_ticker = 0.0

                if cost_of_ideal_shares <= remaining_portfolio_value_for_allocation + 1e-7:
                    shares_to_buy_final_for_this_stock = shares_to_buy_ideal_calc
                    actual_money_allocated_this_ticker = cost_of_ideal_shares
                elif remaining_portfolio_value_for_allocation > 1e-7: 
                    if frac_shares_singularity:
                        affordable_shares_frac = remaining_portfolio_value_for_allocation / live_price_for_tailor
                        shares_to_buy_final_for_this_stock = round(affordable_shares_frac, 1) 
                        if shares_to_buy_final_for_this_stock * live_price_for_tailor > remaining_portfolio_value_for_allocation:
                            shares_to_buy_final_for_this_stock = math.floor(affordable_shares_frac * 10.0) / 10.0
                    else: 
                        shares_to_buy_final_for_this_stock = float(math.floor(remaining_portfolio_value_for_allocation / live_price_for_tailor))
                    shares_to_buy_final_for_this_stock = max(0.0, shares_to_buy_final_for_this_stock)
                    actual_money_allocated_this_ticker = shares_to_buy_final_for_this_stock * live_price_for_tailor
                
                min_share_purchase_threshold = 0.1 if frac_shares_singularity else 1.0
                if shares_to_buy_final_for_this_stock >= min_share_purchase_threshold:
                    # Add to list
                    current_tailored_entries_for_calc.append({
                        'ticker': entry_tailoring.get('ticker','ERR'),
                        'raw_invest_score': entry_tailoring.get('raw_invest_score', -float('inf')),
                        'shares': shares_to_buy_final_for_this_stock,
                        'live_price_at_eval': live_price_for_tailor,
                        'actual_money_allocation': actual_money_allocated_this_ticker,
                        'actual_percent_allocation': (actual_money_allocated_this_ticker / total_value_float_for_tailor) * 100.0 if total_value_float_for_tailor > 0 else 0.0
                    })
                    total_actual_money_spent_on_stocks += actual_money_allocated_this_ticker
                    remaining_portfolio_value_for_allocation -= actual_money_allocated_this_ticker
                    remaining_portfolio_value_for_allocation = max(0.0, remaining_portfolio_value_for_allocation)
        
        final_cash_value_tailored = total_value_float_for_tailor - total_actual_money_spent_on_stocks
        final_cash_value_tailored = max(0.0, final_cash_value_tailored) 

        # ** Second Pass for Whole Shares to Utilize Residual Cash **
        if not frac_shares_singularity and final_cash_value_tailored > 0 and total_value_float_for_tailor > 0:
            if not suppress_prints: 
                print(f"  Attempting second pass to allocate remaining cash: ${final_cash_value_tailored:,.2f} (whole shares mode)")
            
            # Consider stocks already in the portfolio for adding one more share
            potential_top_ups_pass2 = []
            for holding_entry in current_tailored_entries_for_calc: # Iterate over already selected stocks
                price_pass2 = safe_score(holding_entry.get('live_price_at_eval'))
                if price_pass2 > 0 and final_cash_value_tailored >= price_pass2: # Can afford at least one more
                    potential_top_ups_pass2.append({
                        'ticker': holding_entry['ticker'], 
                        'price': price_pass2, 
                        'raw_invest_score': holding_entry.get('raw_invest_score', -float('inf')),
                        # Add 'closeness_to_next_share_value' or similar metric if desired for sorting
                        # For simplicity, we'll sort by raw_invest_score.
                    })
            
            potential_top_ups_pass2.sort(key=lambda x: x['raw_invest_score'], reverse=True)

            for candidate_pass2 in potential_top_ups_pass2:
                if final_cash_value_tailored >= candidate_pass2['price']:
                    for existing_holding_idx, existing_holding_val in enumerate(current_tailored_entries_for_calc):
                        if existing_holding_val['ticker'] == candidate_pass2['ticker']:
                            current_tailored_entries_for_calc[existing_holding_idx]['shares'] += 1.0
                            cost_of_one_share_pass2 = candidate_pass2['price']
                            current_tailored_entries_for_calc[existing_holding_idx]['actual_money_allocation'] += cost_of_one_share_pass2
                            current_tailored_entries_for_calc[existing_holding_idx]['actual_percent_allocation'] = \
                                (current_tailored_entries_for_calc[existing_holding_idx]['actual_money_allocation'] / total_value_float_for_tailor) * 100.0 if total_value_float_for_tailor > 0 else 0.0
                            
                            total_actual_money_spent_on_stocks += cost_of_one_share_pass2
                            final_cash_value_tailored -= cost_of_one_share_pass2
                            if not suppress_prints: 
                                print(f"    Second pass: Bought 1 more share of {candidate_pass2['ticker']}. Cash left: ${final_cash_value_tailored:,.2f}")
                            break # Updated this holding
                else: # Cannot afford this candidate, try next (already sorted by score)
                    continue 
            final_cash_value_tailored = max(0.0, final_cash_value_tailored) # Ensure non-negative after second pass

        current_tailored_entries_for_calc.sort(key=lambda x: safe_score(x.get('raw_invest_score', -float('inf'))), reverse=True)
        tailored_portfolio_structured_data = current_tailored_entries_for_calc

        # ... (Rest of the printing and pie chart logic - unchanged from previous correct version) ...
        # Ensure all print/display sections correctly use the potentially updated current_tailored_entries_for_calc and final_cash_value_tailored
        if not suppress_prints or is_custom_command_simplified_output:
            print("\n--- Tailored Portfolio (Shares) ---") # This will reflect the second pass if it happened
            if current_tailored_entries_for_calc:
                share_format_string_cli = "{:.1f}" if frac_shares_singularity else "{:.0f}"
                tailored_portfolio_output_list_final = [f"{item['ticker']} - {share_format_string_cli.format(item['shares'])} shares" for item in current_tailored_entries_for_calc]
                print("\n".join(tailored_portfolio_output_list_final))
            else:
                print("No stocks allocated in the tailored portfolio based on the provided value and strategy.")
            print(f"Final Cash Value: ${safe_score(final_cash_value_tailored):,.2f}")


        if not suppress_prints and not is_custom_command_simplified_output:
            print("\n--- Tailored Portfolio (Full Details) ---")
            tailored_portfolio_table_data_display_full = []
            share_format_string_table = "{:.1f}" if frac_shares_singularity else "{:.0f}"
            for item_table in current_tailored_entries_for_calc:
                shares_display_table = share_format_string_table.format(item_table['shares'])
                tailored_portfolio_table_data_display_full.append([
                    item_table['ticker'], shares_display_table,
                    f"${safe_score(item_table['actual_money_allocation']):,.2f}",
                    f"{safe_score(item_table['actual_percent_allocation']):.2f}%"
                ])
            final_cash_percent_display_val = (final_cash_value_tailored / total_value_float_for_tailor) * 100.0 if total_value_float_for_tailor > 0 else (100.0 if math.isclose(total_value_float_for_tailor,0) and math.isclose(final_cash_value_tailored,0) else 0.0)
            tailored_portfolio_table_data_display_full.append(['Cash', '-', f"${safe_score(final_cash_value_tailored):,.2f}", f"{safe_score(final_cash_percent_display_val):.2f}%"])
            if not tailored_portfolio_structured_data and math.isclose(final_cash_value_tailored, total_value_float_for_tailor): print("No stocks allocated. All value remains as cash.")
            elif not tailored_portfolio_table_data_display_full : print("Error: No data for tailored portfolio display.")
            else: print(tabulate(tailored_portfolio_table_data_display_full, headers=["Ticker", "Shares", "Actual $ Allocation", "Actual % Allocation"], tablefmt="pretty"))

        pie_chart_data_for_gen = []
        if tailored_portfolio_structured_data:
            for item_pie in tailored_portfolio_structured_data:
                if item_pie.get('actual_money_allocation', 0) > 1e-9: pie_chart_data_for_gen.append({'ticker': item_pie['ticker'], 'value': item_pie['actual_money_allocation']})
        if final_cash_value_tailored > 1e-9: pie_chart_data_for_gen.append({'ticker': 'Cash', 'value': final_cash_value_tailored})
        if pie_chart_data_for_gen and (not suppress_prints or not is_custom_command_simplified_output):
            chart_title_base_str = "Invest Portfolio"
            if isinstance(portfolio_data_config, pd.Series): chart_title_base_str = portfolio_data_config.get('portfolio_code', "Invest Portfolio")
            elif isinstance(portfolio_data_config, dict): chart_title_base_str = portfolio_data_config.get('portfolio_code', "Invest Portfolio")
            chart_title_final_str = f"{chart_title_base_str} Allocation (Value: ${safe_score(total_value_singularity if total_value_singularity is not None else 0):,.0f})"
            await asyncio.to_thread(generate_portfolio_pie_chart, pie_chart_data_for_gen, chart_title_final_str, "singularity_portfolio_pie", is_called_by_ai=suppress_prints)
        if not suppress_prints: print(f"Remaining Buying Power (Final Cash in Tailored Portfolio): ${safe_score(final_cash_value_tailored):,.2f}")
    
    return tailored_portfolio_output_list_final, final_combined_portfolio_data_calc, final_cash_value_tailored, tailored_portfolio_structured_data

def generate_portfolio_pie_chart(portfolio_allocations: List[Dict[str, Any]], chart_title: str, filename_prefix: str = "portfolio_pie", is_called_by_ai: bool = False) -> Optional[str]:
    if not portfolio_allocations:
        if not is_called_by_ai: print("Pie Chart Error: No allocation data.")
        return None
    valid_data = [{'ticker': item['ticker'], 'value': item['value']} for item in portfolio_allocations if item.get('value', 0) > 1e-9]
    if not valid_data:
        if not is_called_by_ai: print("Pie Chart Error: No positive allocations.")
        return None

    labels = [item['ticker'] for item in valid_data]
    sizes = [item['value'] for item in valid_data]
    total_value_chart = sum(sizes)

    threshold_percentage = 1.5
    max_individual_slices = 14
    if len(labels) > max_individual_slices + 1:
        sorted_allocs = sorted(zip(sizes, labels), reverse=True)
        display_labels, display_sizes, other_value = [], [], 0.0
        for i, (size, label) in enumerate(sorted_allocs):
            if i < max_individual_slices: display_labels.append(label); display_sizes.append(size)
            else: other_value += size
        if other_value > 1e-9: display_labels.append("Others"); display_sizes.append(other_value)
        labels, sizes = display_labels, display_sizes
        if not labels: # if not is_called_by_ai: print("Pie Chart Error: All slices grouped.");
             return None

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8))
    if not labels: plt.close(fig); return None # Should be caught earlier

    custom_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC', '#A0CBE8', '#FFBE7D', '#F4ADA8', '#B5D9D0', '#8CD17D']
    colors_to_use = custom_colors[:len(labels)] if len(labels) <= len(custom_colors) else [plt.cm.get_cmap('viridis', len(labels))(i) for i in range(len(labels))]
    explode_values = [0.05 if i == 0 and len(labels) > 0 else 0 for i in range(len(labels))]

    wedges, _, autotexts = ax.pie(
        sizes, explode=explode_values, labels=None,
        autopct=lambda pct: f"{pct:.1f}%" if pct > threshold_percentage else '',
        startangle=90, colors=colors_to_use, pctdistance=0.80,
        wedgeprops={'edgecolor': '#2c2f33', 'linewidth': 1}
    )
    for autotext in autotexts: autotext.set_color('white'); autotext.set_fontsize(9); autotext.set_fontweight('bold')
    ax.set_title(chart_title, fontsize=18, color='white', pad=25, fontweight='bold')
    ax.axis('equal')
    legend_labels = [f'{l} ({s/total_value_chart*100:.1f}%)' for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title="Holdings", loc="center left", bbox_to_anchor=(1.05, 0, 0.5, 1),
              fontsize='medium', labelcolor='lightgrey', title_fontsize='large', facecolor='#36393f', edgecolor='grey')
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.png"
    try:
        plt.savefig(filename, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        if not is_called_by_ai: print(f"Pie chart saved: {filename}")
    except Exception as e:
        if not is_called_by_ai: print(f"Error saving pie chart: {e}")
        plt.close(fig); return None
    plt.close(fig)
    return filename

def fetch_yahoo_finance_data_spear(ticker):
    ticker = ticker.replace(".", "-")
    stock = yf.Ticker(ticker)
    info = stock.info
    today = datetime.today()

    # 1 Day Change Percentage
    one_day_ago = today - pd.Timedelta(days=2)
    start_date_one = one_day_ago.strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start_date_one, progress=False, timeout=10)
    one_day_change_percent = (data['Close'].iloc[-1] / data['Open'].iloc[0] - 1).item() if not data.empty and len(data) > 0 else 0.0

    # 1 Month Change Percentage
    thirty_days_ago = today - pd.Timedelta(days=31)
    start_date_two = thirty_days_ago.strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start_date_two, progress=False, timeout=10)
    one_month_change_percent = (data['Close'].iloc[-1] / data['Open'].iloc[0] - 1).item() if not data.empty and len(data) > 0 else 0.0

    # 6 Month Change Percentage
    hundred_eighty_days_ago = today - pd.Timedelta(days=181)
    start_date_six = hundred_eighty_days_ago.strftime("%Y-%m-%d")
    data = yf.download(ticker, start=start_date_six, progress=False, timeout=10)
    six_month_change_percent = (data['Close'].iloc[-1] / data['Open'].iloc[0] - 1).item() if not data.empty and len(data) > 0 else 0.0

    one_year_change_percent = info.get('52WeekChange', 0.0)

    # SPY 1D% Change
    spy_data = yf.download("SPY", start=start_date_one, progress=False, timeout=10)
    spy_one_day_change_percent = (spy_data['Close'].iloc[-1] / spy_data['Open'].iloc[0] - 1).item() if not spy_data.empty and len(spy_data) > 0 else 0.0

    market_cap = info.get('marketCap', 0)

    return {
        '1D% Change': one_day_change_percent,
        '1M% Change': one_month_change_percent,
        '3M% Change': six_month_change_percent,
        '1Y% Change': float(one_year_change_percent or 0.0),
        'SPY 1D% Change': spy_one_day_change_percent,
        'Market Cap In Billions of USD': float(market_cap or 0.0)
    }

def plot_spear_graph(ticker, recommended_price, is_called_by_ai: bool = False):
    today = datetime.today()
    one_year_ago = today - pd.Timedelta(days=365)
    data = yf.download(ticker, start=one_year_ago.strftime("%Y-%m-%d"), end=today.strftime("%Y-%m-%d"), progress=False, timeout=15)
    
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price', color='grey')
    
    if recommended_price is not None:
        ax.axhline(y=recommended_price, color='r', linestyle='--', label=f'Predicted Price: ${recommended_price:.2f}')
    
    ax.set_title(f'{ticker.upper()} Stock Price - Past Year', color='white')
    ax.set_xlabel('Date', color='white'); ax.set_ylabel('Price', color='white')
    ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
    ax.grid(True, color='dimgray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.tick_params(axis='x', colors='white'); ax.tick_params(axis='y', colors='white')
    fig.tight_layout()

    filename = f"spear_graph_{ticker.replace('.','-')}_{uuid.uuid4().hex[:6]}.png"
    plt.savefig(filename, facecolor='black', edgecolor='black')
    plt.close(fig)
    if not is_called_by_ai: print(f"SPEAR graph saved: {filename}")
    return filename

def business_summary_spear(ticker, business_summary_length_sentences):
    company = yf.Ticker(ticker)
    info = company.info
    summary_raw = info.get('longBusinessSummary', 'Business Summary not available')
    industry = info.get('industry', 'Industry not available')
    sector = info.get('sector', 'Sector not available')
    market_cap_raw = info.get('marketCap')

    market_cap_formatted = humanize.intword(market_cap_raw) if market_cap_raw else 'N/A'
    
    summary_trim = ' '.join(sent_tokenize(summary_raw)[:business_summary_length_sentences])
    return summary_trim, industry, sector, market_cap_formatted

def calculate_spear_prediction(ticker, sector, relevance, hype, fear, earnings_date_str, earnings_time, trend, reversal, meme_stock, trade, actual_price):
    finance_data = fetch_yahoo_finance_data_spear(ticker)
    one_month_chg_per = finance_data['1M% Change']
    six_month_chg_per = finance_data['3M% Change']
    one_year_chg_per = finance_data['1Y% Change']
    spy_one_day_chg_per = finance_data['SPY 1D% Change']
    market_cap = finance_data['Market Cap In Billions of USD']

    def fetch_options_data(ticker_opt: str):
        stock = yf.Ticker(ticker_opt)
        today = datetime.today().date()
        try:
            exp_dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in stock.options]
            future_exp_dates = [d for d in exp_dates if d > today]
            if not future_exp_dates: raise ValueError("No future expiration dates.")
            next_exp_date = min(future_exp_dates)
            opt_chain = stock.option_chain(next_exp_date.strftime('%Y-%m-%d'))
            pcr = opt_chain.puts['openInterest'].sum() / opt_chain.calls['openInterest'].sum() if opt_chain.calls['openInterest'].sum() != 0 else 0
            avg_iv = (opt_chain.puts['impliedVolatility'].mean() + opt_chain.calls['impliedVolatility'].mean()) / 2 * 100
            return {"put_call_ratio": pcr, "average_implied_volatility": avg_iv}
        except Exception:
            return {"put_call_ratio": 1, "average_implied_volatility": 0}

    c1 = "1" if one_year_chg_per < 0.5 else "2" if one_year_chg_per < 0.75 else "3" if one_year_chg_per < 1 else "4" if one_year_chg_per < 1.5 else "5"
    c2 = "5" if market_cap < 5e9 else "3" if market_cap < 2.5e10 else "2" if market_cap < 5e10 else "3" if market_cap < 1e11 else "4" if market_cap < 2e11 else "5"
    c3 = sector
    c4 = relevance
    c5 = ((float(c1) + float(c2) + float(c3) + float(c4)) + hype)/4
    c6 = math.sqrt(abs((c5**2) - 1)) if c5 < 3 else math.sqrt(abs((c5**2) + 1)) if c5 > 3 else 3.25
    c7 = c6 * 1.25 if c4 == "5" else c6
    c8 = c7 - 3
    c9 = c8 * -c8 if c8 < 0 else c8**2
    c10 = c9 / 3 * 0.1 if market_cap > 1e11 else c9 * 0.1
    c11 = one_month_chg_per*100/30
    days_difference = (datetime.strptime(earnings_date_str, "%Y-%m-%d") - datetime.today()).days + (1 if earnings_time == 'a' else 0)
    c12 = int(days_difference) * c11
    c13 = ((((spy_one_day_chg_per * 100) - 0.12)) * c10) + c10
    vix_data = yf.download("^VIX", start=(datetime.today() - pd.Timedelta(days=2)).strftime("%Y-%m-%d"), progress=False)
    vix_live = vix_data['Open'].values[-1] if not vix_data.empty else 17.0
    c14 = "-1" if vix_live < 13.5 else "-0.5" if vix_live < 15.25 else "0" if vix_live < 17.0 else "0.5" if vix_live < 18.75 else "1"
    c15 = ((0.5 * float(c14) * float(c13)) + float(c13))
    uncertain = 1 if trend in ["No Trend", "n"] else 0
    c16 = "3" if fear == 50 else "2" if trend in ["Upwards","u"] else "1" if trend in ["Downwards","d"] else "0" if trend == "Use Stock" else "-1" if fear > 55 else "-2" if market_cap >= 1e11 else "-3" if uncertain == 1 else "-4"
    c17 = float(-1*c15) if c16=="3" and trend in ["Upwards","u"] else float(c15) if c16=="3" else float(-1*(((((fear-50)/50))*c15)-c15)) if c16=="2" else float(-1*(((((fear-50)/50))*c15)+c15)) if c16=="1" else float(((((fear-50)/50))*c15)+c15) if c16=="0" and (one_month_chg_per>=0.1) else float(((((fear-50)/50))*c15)-c15) if c16=="0" and (one_month_chg_per<=0.1) else float(((((fear-50)/50))*c15)+c15) if c16=="-1" else float(((((fear-50)/50))*c15)-c15) if c16=="-2" else float(c15) if c16=="-3" else float(((((fear-50)/50))*c15)+c15)
    c18 = ((c17+c15)/4) if uncertain==1 and fear==50 and (one_month_chg_per>=0.1) else ((c17-c15)/4) if uncertain==1 and fear==50 and (one_month_chg_per<=0.1) else (((c17+(float(-1*(((fear-50)/50))))*(3*c15))+c15)/4) if uncertain==1 and (one_month_chg_per>=0.1) else (((c17+(float(-1*(((fear-50)/50))))*(3*c15))-c15)/4) if uncertain==1 and (one_month_chg_per<=0.1) else float(c17)
    c19 = c18 * 3 if uncertain == 1 and -0.05 < c18 < 0.05 else float(c18)
    c20 = c19 * 2 if market_cap < 5e9 and -0.05 < c19 < 0.05 else float(c19)
    c21 = sqrt(abs(c20))*-2 if reversal in ["Yes","y"] and c20>0 else sqrt(abs(c20*-1))*2 if reversal in ["Yes","y"] and c20<0 else float(c20)
    price_in_value = 1 if six_month_chg_per > 0.3 or six_month_chg_per < 0.1 else 0
    price_in_hype = 1 if abs(hype) > 0.56 else 0
    price_in_affirmation = 0 if price_in_value == 1 and price_in_hype == 1 else 1
    c22 = -1 if price_in_affirmation == 1 else 1
    c23 = c21 * float(c22) if price_in_affirmation == 1 else float(c21)
    c24 = (10*c23)+c23 if price_in_affirmation==1 and 0<c23<=0.005 else (10*c23)-c23 if price_in_affirmation==1 and -0.005<=c23<0 else (5*c23)+c23 if price_in_affirmation==1 and 0<c23<=0.01 else (5*c23)-c23 if price_in_affirmation==1 and -0.01<=c23<0 else (2*c23)+c23 if price_in_affirmation==1 and 0<c23<=0.5 else (2*c23)-c23 if price_in_affirmation==1 and -0.5<=c23<0 else (1.5*c23)+c23 if price_in_affirmation==1 and 0<c23<=0.1 else (1.5*c23)-c23 if price_in_affirmation==1 and -0.1<=c23<0 else float(c23)
    c25 = (c24*2) if abs(hype)>0.5 and -0.05<c23<0.05 and price_in_affirmation==1 else (c24*1.5) if abs(hype)>0.5 and -0.1<c23<0.1 and price_in_affirmation==1 else float(c24)
    c26 = c25*20 if -0.01<=c25<=0.01 and meme_stock in ["Yes","y"] else c25*10 if -0.025<=c25<=0.025 and meme_stock in ["Yes","y"] else c25*5 if -0.05<=c25<=0.05 and meme_stock in ["Yes","y"] else float(c25)
    c27 = c26 * 7.5 if -0.025 <= c26 <= 0.025 else float(c26)
    c28 = c27/5 if abs(c27)>=0.3 and meme_stock in ["No","n"] else float(c27)
    c29 = c28/8 if abs(c28)>=1 else c28/5 if abs(c28)>=0.75 else c28/2 if abs(c28)>=0.4 else float(c28)
    c30 = c29 * (((50-fear)*8/100)+1) if fear<=55 else c29 * (((50-fear)*4/100)+1) if fear<=60 else c29*(((50-fear)*2/100)+1)
    result_options = fetch_options_data(ticker)
    put_call, iv = result_options['put_call_ratio'], result_options['average_implied_volatility']
    c31 = ((iv-300)/-600) if iv>=300 and market_cap>=2e10 else c30
    c32 = (((iv-150)/-600)-0.1) if iv>=150 and c30<0 else (((iv-150)/600)+0.1) if iv>=150 and c30>=0 else c30
    c33 = (iv/-1200) if c30<0 else (iv/1200)
    c34 = ((put_call-2)/-25) if put_call>2 and c30<0 else ((put_call-2)/25) if put_call>2 and c30>=0 else c30
    c35 = abs(one_month_chg_per-(one_year_chg_per/12)) if meme_stock not in ["Yes","y"] and c30>=0 else (-1*abs(one_month_chg_per-(one_year_chg_per/12))) if meme_stock not in ["Yes","y"] and c30<0 else c30
    c36 = (c31+c32+c34)/3
    c37 = (c30+c33+c36)/3
    c38 = (c35+c37)/2 if -0.4 < c35 < 0.4 else c37
    c39 = 0.15 if c38>0.35 and market_cap>1e11 else -0.15 if c38<-0.35 and market_cap>1e11 else 0.35 if c38>0.35 else -0.35 if c38<-0.35 else c38

    return {
        'price_in': price_in_affirmation, 'prediction': float(c39),
        'prediction_in_time': float(c12 / 100), 'finance_data': finance_data
    }

async def save_portfolio_to_csv(file_path: str, portfolio_data_to_save: Dict[str, Any], is_called_by_ai: bool = False):
    """Saves portfolio configuration to the CSV database."""
    file_exists = os.path.isfile(file_path)
    fieldnames = ['portfolio_code', 'ema_sensitivity', 'amplification', 'num_portfolios', 'frac_shares', 'risk_tolerance', 'risk_type', 'remove_amplification_cap']
    num_portfolios_val = int(portfolio_data_to_save.get('num_portfolios', 0))
    for i in range(1, num_portfolios_val + 1):
        fieldnames.extend([f'tickers_{i}', f'weight_{i}'])
    for key in portfolio_data_to_save.keys():
        if key not in fieldnames: fieldnames.append(key)

    try:
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists or os.path.getsize(file_path) == 0: writer.writeheader()
            data_to_write = {key: portfolio_data_to_save.get(key) for key in fieldnames if key in portfolio_data_to_save}
            writer.writerow(data_to_write)
        if not is_called_by_ai:
            print(f"Portfolio configuration '{portfolio_data_to_save.get('portfolio_code')}' saved to {file_path}")
    except Exception as e:
        if not is_called_by_ai:
            print(f"Error saving portfolio config to CSV {file_path}: {e}")

async def save_portfolio_data_singularity(portfolio_code_to_save: str, date_str_to_save: str, is_called_by_ai: bool = False):
    """
    Saves *combined percentage data* for a custom portfolio (original '3725' functionality).
    This is different from saving the full tailored run output.
    """
    if not os.path.exists(PORTFOLIO_DB_FILE):
        if not is_called_by_ai: print(f"Error: Portfolio DB '{PORTFOLIO_DB_FILE}' not found.")
        return
    portfolio_config_data = None
    try:
        with open(PORTFOLIO_DB_FILE, 'r', encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('portfolio_code', '').strip().lower() == portfolio_code_to_save.lower():
                    portfolio_config_data = row; break
        if not portfolio_config_data:
            if not is_called_by_ai: print(f"Error: Portfolio code '{portfolio_code_to_save}' not found in DB.")
            return
    except Exception as e:
        if not is_called_by_ai: print(f"Error reading DB for code {portfolio_code_to_save}: {e}")
        return

    if portfolio_config_data and date_str_to_save:
        try:
            frac_s = portfolio_config_data.get('frac_shares', 'false').lower() == 'true'
            if not is_called_by_ai: print(f"Running analysis for '{portfolio_code_to_save}' to generate data for saving...")
            # Suppress prints from process_custom_portfolio for this internal call
            _, combined_result_for_save, _, _ = await process_custom_portfolio(
                portfolio_data_config=portfolio_config_data, tailor_portfolio_requested=False,
                frac_shares_singularity=frac_s, total_value_singularity=None,
                is_custom_command_simplified_output=True, is_called_by_ai=True # Fully suppress
            )

            if combined_result_for_save:
                data_to_write_csv = []
                for item in combined_result_for_save:
                    if item.get('ticker') != 'Cash' and safe_score(item.get('combined_percent_allocation_adjusted', 0)) > 1e-4:
                        data_to_write_csv.append({
                            'DATE': date_str_to_save, 'TICKER': item.get('ticker', 'ERR'),
                            'PRICE': f"{safe_score(item.get('live_price')):.2f}" if item.get('live_price') is not None else "N/A",
                            'COMBINED_ALLOCATION_PERCENT': f"{safe_score(item.get('combined_percent_allocation_adjusted')):.2f}" if item.get('combined_percent_allocation_adjusted') is not None else "N/A"
                        })
                if not data_to_write_csv:
                    if not is_called_by_ai: print(f"No stock data with allocation > 0 for '{portfolio_code_to_save}' to save.")
                    return
                sorted_data = sorted(data_to_write_csv, key=lambda x: float(x['COMBINED_ALLOCATION_PERCENT'].rstrip('%')) if x['COMBINED_ALLOCATION_PERCENT'] not in ["N/A","ERR"] else -1, reverse=True)
                save_filename = f"portfolio_code_{portfolio_code_to_save}_data.csv" # Original naming for this specific save type
                file_exists = os.path.isfile(save_filename)
                headers = ['DATE', 'TICKER', 'PRICE', 'COMBINED_ALLOCATION_PERCENT']
                with open(save_filename, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    if not file_exists or os.path.getsize(f.name) == 0: writer.writeheader()
                    writer.writerows(sorted_data)
                if not is_called_by_ai:
                    print(f"Saved {len(sorted_data)} rows of combined % data for '{portfolio_code_to_save}' to '{save_filename}' for {date_str_to_save}.")
            else:
                if not is_called_by_ai: print(f"No combined data for '{portfolio_code_to_save}' to save.")
        except Exception as e:
            if not is_called_by_ai: print(f"Error processing/saving data for {portfolio_code_to_save}: {e}"); traceback.print_exc()

# --- Singularity Specific Functions ---
def display_welcome_message():
    """Displays the welcome message, loading animation, and ASCII art."""
    print("Initializing Market Insights Center Singularity...")
    # Loading animation
    animation_chars = ["|", "/", "-", "\\"]
    for _ in range(10): # Number of cycles for the animation
        for char in animation_chars:
            print(f"\rLoading... {char}", end="", flush=True)
            py_time.sleep(0.1) # Speed of animation
    print("\rLoading... Done!          ") # Clear animation line and add padding

    print("\n" + "="*50)
    print("THE MARKET INSIGHTS CENTER SINGULARITY IS NOW ACTIVE!")
    print("="*50 + "\n")

    # ASCII Art defined using a single triple-quoted raw string
    # This allows for easier copy-pasting of multi-line ASCII art.
    # The r""" ensures backslashes in your art are treated literally.
    # Ensure your ASCII art is properly formatted within the triple quotes.
    # The leading newline after r""" will be part of the string unless the art starts on the same line.
    # Any indentation within the triple quotes will also be part of the string.
    full_ascii_message = r"""                                                                              

                                                                         08008                                                                        
                                                                        80000000                                                                      
                                                                       8000000000                                                                     
                                                                      000000000009                                                                    
                                                                    900000000000000                                                                   
                                                                   800000000000000080                                                                 
                                                                  80000000000000000009                                                                
                                                                68000000000000000000008                                                               
                                                               80000000000000000000000080                                                             
                                                              6000000000000000000000000089                                                            
                                                            0800000000000000000000000000009                                                           
                                                           200000000000000000000000000000000                                                          
                                                          80000000000000000000000000000000000                                                         
                                                         00000000000000000000000000000000000004                                                       
                                                       40000000000000000000000000000000000000008                                                      
                                                      80000000000000000000000000000000000000000080                                                    
                                                     8000000000000000000000000000000000000000000006                                                   
                                                   0800000000000000000000080000000000000000000000009                                                  
                                                  9000000000000000000000000000000000000000000000000000                                                
                                                 8000000000000000000000096   06800000000000000000000009                                               
                                                9000000000000000000090           09800000000000000000009                                              
                                              00000000000000000090                    9000000000000000008                                             
                                             80000000000000090                            60000000000000000                                           
                                            90000000000096         098000008000088990         90000000000008                                          
                                          000000000084        9980000000000000000000000880       090000000008                                         
                                         900000000       390000000000089       060000000000888       0980000080                                       
                                        0000880       60000000000008               090000000000085       6800008                                      
                                      58090       6800000000000008   8084            0900000000000006        9909                                     
                                      0        680000000880800009   8000009            800086980000000040        0                                    
                                            60000000840   000009   80000000             80000    60000000000                                          
                                         4800000005       90000     600009              000004      50000000089                                       
                                      6800000003          00006                          00000          9800000009                                    
                                     800000080            00000                          90000            900000080                                   
                                      98000000099         80000                          00004        098000000080                                    
                                         080000000090     000000                        80000      09880000008                                        
                                            088000000088  000000                       600008  068000000088                                           
                                      0         980000000082800080                    800008400000000800         0                                    
                                     200880        0900000000000009                 2800000000000000         88008                                    
                                       4000006         090000000088880            90800000000086         69880004                                     
                                        08000000980        390000000000006604980000000000084         9900000009                                       
                                          6000000000996         94000000000000000000006          9688808000088                                        
                                           4000000000000990            7890590607            998880000880009                                          
                                             800000000000000890                          968800000000000009                                           
                                              600000000000000000860                  69800000000000000088                                             
                                                8000000000000000000084           29988000000000000000089                                              
                                                 40000000000000000000008890  6900000000000000000000008                                                
                                                   80000000000000000000088800000000000000000000000006                                                 
                                                    60000000000000000000000000000000000000000000008                                                   
                                                      80000000000000000000000000000000000000000000                                                    
                                                       60000000000000000000000000000000000000008                                                      
                                                         80000000000000000000000000000000000008                                                       
                                                          880000000000000000000000000000000000                                                        
                                                            00000000000000000000000000000008                                                          
                                                             600000000000000000000000000000                                                           
                                                               80000000000000000000000008                                                             
                                                                900000000000000000000006                                                              
                                                                 080000000000000000000                                                                
                                                                   800000000000000009                                                                 
                                                                    680000000000000                                                                   
                                                                      800000000009                                                                    
                                                                       6000000080                                                                     
                                                                         800008                                                                       
                                                                          569                                                                       

                                                                          
                                                                                      
                                                                                                                                                      
                                                                                                                                                     
                    11111                   1111                                                                                                      
                    133333                313331                                                                       133                            
                   1333331                333331                              3111311111113131                    11111111111                         
                   1333331              13333331                   111111111111333331111111111111111111          133331   311                         
                   3333331             13333331                     7111113    13331               311111       13331       1                         
                  33333333            133313331                                3331                           113331                                  
                  133313337          1331 33337                                1331                           13331                                   
                 3333333333         1333  1313                                 1331                          33331                                    
                 1333  1331        1331   1 31                                33337                         13331                                     
                 1333  1331       3331    1311                                1331                          1333                                      
                 1331  3333      1331     331                                 1331                         1331                                       
                 133    1337    1331      331                                13317                        3331                                        
                 111    1333   1331      7331                                1331                         1331                                        
                1331    1333 11331       1333                                1331                        7331                                         
                1331     33313333         133                                1333                        1331                                         
                1333     1333331          131                                131                         133                                          
                3331      11111           31                                1331                        1331                                          
               33137                      11                                1331                        1331                  11                      
               3333                      111                                1331                        111                   11                      
               1133                       31                                1331                        331                  1317                     
               1331                      131                                1331                        131                 1331                      
               1331                     7337                                1331                        331                 3337                      
               1131                     331           113      11111117     1331      11      113       131                1331    1131               
               1331                     311          11331   11331   1111   3331  11111      11331      131               1333    11331               
               1331                     131           113   1333       3313333311113          111       1331             1331      111                
               1331                     131                   111113111111113331                         331           11331                          
               1331                     331                       131313    1111                          311        11111                            
               1111                     131                                                                11113  111111                              
                                        131                                                                  311111111                                
                                       3131                                                                                                           
                                        11                                                                                                                                                                                                                       
"""

    print("Presenting Visual Identification Matrix:")
    # The typing animation iterates character by character, including newlines.
    for char_art in full_ascii_message:
        print(char_art, end="", flush=True)
        py_time.sleep(0.0001) # Typing animation speed for each character
    print("\n\n") # Add extra newlines after the ASCII art

def display_commands():
    """Displays the list of available commands and their general usage with a typing animation."""

    command_lines = [] # We'll build the text here

    command_lines.append("\nAvailable Commands:")
    command_lines.append("-------------------")
    
    command_lines.append("\nGENERAL Commands")
    command_lines.append("-------------------")
    command_lines.append("/briefing - Generate a comprehensive daily market summary.")
    command_lines.append("  Description: Provides a snapshot of market prices, risk scores, top/bottom movers, breakout activity, and watchlist performance.")
    command_lines.append("  CLI Usage:")
    command_lines.append("    /briefing   (The script will run all necessary analyses and display the report)")

    command_lines.append("\nSPEAR Commands")
    command_lines.append("-------------------")
    command_lines.append("/spear - Predict a stock's performance around its upcoming earnings report.")
    command_lines.append("  Description: Uses the SPEAR model to generate an earnings prediction based on financial and market sentiment inputs.")
    command_lines.append("  CLI Usage:")
    command_lines.append("    /spear   (The script will then prompt you for all necessary inputs step-by-step)")

    command_lines.append("\nINVEST Commands")
    command_lines.append("-------------------")
    command_lines.append("/invest - Analyze multiple stocks based on EMA sensitivity and amplification.")
    command_lines.append("  Description: Prompts for EMA sensitivity, amplification, number of sub-portfolios,")
    command_lines.append("               tickers for each, and their weights. Can optionally tailor to a portfolio value.")
    command_lines.append("  CLI Usage:")
    command_lines.append("    /invest   (The script will then prompt you for all necessary inputs step-by-step)")

    command_lines.append("\n/custom - Run portfolio analysis using a saved code, create/save a new one, or save legacy data.")
    command_lines.append("  Description: Manages custom portfolio configurations. Running a portfolio automatically saves/overwrites")
    command_lines.append("               its detailed tailored output. The '3725' option is for a legacy combined percentage save.")
    command_lines.append("  CLI Usage:")
    command_lines.append("    /custom MYPORTFOLIO             (Runs portfolio 'MYPORTFOLIO', creates if new, saves detailed run output)")
    command_lines.append("    /custom #                       (Creates a new portfolio with the next available numeric code, saves detailed run output)")
    command_lines.append("    /custom MYPORTFOLIO 3725        (Saves legacy combined percentage data for 'MYPORTFOLIO' after prompting for date)")

    command_lines.append("\n/quickscore - Get quick scores and graphs for a single ticker.")
    command_lines.append("  Description: Provides EMA-based investment scores (Weekly, Daily, Hourly) and generates price/EMA graphs for one stock.")
    command_lines.append("  CLI Usage:")
    command_lines.append("    /quickscore AAPL                (Get scores and graphs for Apple Inc.)")
    command_lines.append("    /quickscore MSFT                (Get scores and graphs for Microsoft Corp.)")

    command_lines.append("\n/breakout - Run breakout analysis or save current breakout data.")
    command_lines.append("  Description: Identifies stocks with strong breakout potential or saves the current list of breakout stocks historically.")
    command_lines.append("  CLI Usage:")
    command_lines.append("    /breakout                       (Runs a new breakout analysis and saves the current findings)")
    command_lines.append("    /breakout 3725                  (Saves the current breakout_tickers.csv data to the historical database, prompts for date)")

    command_lines.append("\n/market - Display S&P 500 market scores or save full S&P 500 market data.")
    command_lines.append("  Description: Provides an overview of S&P 500 stock scores or saves detailed data for a chosen EMA sensitivity.")
    command_lines.append("  CLI Usage:")
    command_lines.append("    /market                         (Prompts for EMA sensitivity to display S&P 500 scores)")
    command_lines.append("    /market 2                       (Directly displays S&P 500 scores using Daily EMA sensitivity)")
    command_lines.append("    /market 3725                    (Prompts for sensitivity and date to save full S&P 500 market data)")

    command_lines.append("\n/cultivate - Craft a Cultivate portfolio or save its data.")
    command_lines.append("  Description: Generates a diversified portfolio based on 'Cultivate' strategy codes A or B, portfolio value, and share preference.")
    command_lines.append("  CLI Usage:")
    command_lines.append("    /cultivate A 10000 yes          (Run Cultivate Code A for $10,000 value with fractional shares)")
    command_lines.append("    /cultivate B 50000 no           (Run Cultivate Code B for $50,000 value without fractional shares)")
    command_lines.append("    /cultivate A 25000 yes 3725     (Generate data for Cultivate Code A, $25k, frac. shares, then prompts for date to save)")

    command_lines.append("\n/assess - Assess stock volatility, portfolio risk, etc., based on different codes.")
    command_lines.append("  Description: Performs various financial assessments.")
    command_lines.append("    A (Stock Volatility): Analyzes individual stock volatility against user's risk tolerance.")
    command_lines.append("      CLI Usage: /assess A AAPL,GOOG 1Y 3  (Assess Apple and Google over 1 year with risk tolerance 3)")
    command_lines.append("                 /assess A TSLA 3M 5       (Assess Tesla over 3 months with risk tolerance 5)")
    command_lines.append("    B (Manual Portfolio Risk): Calculates Beta/Correlation for a manually entered portfolio.")
    command_lines.append("      CLI Usage: /assess B 1y              (Script will prompt for tickers/shares/cash for a 1-year backtest)")
    command_lines.append("                 /assess B 5y              (Prompt for holdings for a 5-year backtest)")
    command_lines.append("    C (Custom Portfolio Risk): Calculates Beta/Correlation for a saved custom portfolio configuration.")
    command_lines.append("      CLI Usage: /assess C MYPORTFOLIO 25000 3y (Assess 'MYPORTFOLIO' tailored to $25,000 for a 3-year backtest)")
    command_lines.append("                 /assess C AlphaGrowth 100000 5y (Assess 'AlphaGrowth' at $100,000 for 5-year backtest)")
    command_lines.append("    D (Cultivate Portfolio Risk): Calculates Beta/Correlation for a generated Cultivate portfolio.")
    command_lines.append("      CLI Usage: /assess D A 50000 yes 5y    (Assess Cultivate Code A, $50k, frac. shares, 5-year backtest)")
    command_lines.append("                 /assess D B 10000 no 1y     (Assess Cultivate Code B, $10k, no frac. shares, 1-year backtest)")

    command_lines.append("\nRISK Commands")
    command_lines.append("-------------------")
    command_lines.append("/risk - Perform R.I.S.K. module calculations, display results, and save data.")
    command_lines.append("  Description: Calculates a suite of market risk indicators and determines a market signal.")
    command_lines.append("  CLI Usage:")
    command_lines.append("    /risk                           (Performs standard R.I.S.K. calculation and saves data)")
    command_lines.append("    /risk eod                       (Performs End-of-Day R.I.S.K. calculation and saves EOD specific data)")

    command_lines.append("\n/history - Generate and save historical R.I.S.K. module graphs.")
    command_lines.append("  Description: Creates visual charts of historical R.I.S.K. indicators.")
    command_lines.append("  CLI Usage:")
    command_lines.append("    /history                        (Generates and saves all R.I.S.K. history graphs)")

    command_lines.append("\nAI Command")
    command_lines.append("-------------------")
    command_lines.append("/ai - Interact with an AI to perform tasks using natural language.")
    command_lines.append("  CLI Usage Examples:")
    command_lines.append("    /ai show me the breakout stocks then quickscore the top one")
    command_lines.append("    /ai run the daily market analysis scores with hourly sensitivity")
    command_lines.append("    /ai perform the end-of-day risk assessment")
    command_lines.append("    /ai assess stock volatility for NVDA and AMD over 3 months with risk tolerance 4")
    command_lines.append("    /ai run my custom portfolio 'AlphaPicks' and tailor it to $75000 with fractional shares")
    command_lines.append("    /ai save the data for custom portfolio 'BetaGrowth' for today (legacy combined % save)")
    command_lines.append("    /ai I want to run an invest analysis with hourly sensitivity and 1.5x amplification. Portfolio one is QQQ, SPY with 70% weight. Portfolio two is GLD with 30% weight. Tailor to $100k.")
    command_lines.append("    /ai execute cultivate analysis code A for $20000 value without fractional shares, then save the results for today.")
    command_lines.append("    /ai compare my custom portfolio 'TechWinners' with its last saved run, using a $50000 value for the fresh run.")
    command_lines.append("    /ai give me the daily briefing")


    command_lines.append("\nUtility Commands")
    command_lines.append("-------------------")
    command_lines.append("/help - Display this list of commands.")
    command_lines.append("/exit - Close the Market Insights Center Singularity.")
    command_lines.append("-------------------\n")

    full_command_text = "\n".join(command_lines)

    typing_speed = 0.001 # Adjust for desired speed (e.g., 0.002 is slightly slower)
    for char_cmd in full_command_text:
        print(char_cmd, end="", flush=True)
        py_time.sleep(typing_speed)
    print() # Ensure a final newline

async def get_comparison_for_custom_portfolio(
    ai_params: Optional[Dict] = None,
    is_called_by_ai: bool = False, # This flag still useful for other conditional logic if needed
    args: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Loads saved output, runs fresh, returns comparison status, prints comparison table to console,
    AND THEN SAVES THE FRESH RUN OUTPUT, OVERWRITING THE PREVIOUS CSV.
    """
    # Default response for critical early failures
    if not ai_params:
        return {
            "portfolio_code": "N/A",
            "status": "error_dev_params_missing",
            "saved_run_data_load_details": {"loaded_successfully": False, "message": "Internal error: Parameters missing for tool."},
            "fresh_run_generation_details": {"generated_successfully": False, "message": "Not attempted due to missing parameters."},
            "comparison_outcome_summary": "Internal error: Parameters missing for comparison tool.",
            "save_status_of_fresh_run": "Not attempted due to missing parameters."
        }

    portfolio_code = ai_params.get("portfolio_code")
    if not portfolio_code:
        return {
            "portfolio_code": "N/A",
            "status": "error_portfolio_code_missing",
            "saved_run_data_load_details": {"loaded_successfully": False, "message": "Portfolio code not provided to tool."},
            "fresh_run_generation_details": {"generated_successfully": False, "message": "Not attempted due to missing portfolio code."},
            "comparison_outcome_summary": "Portfolio code missing for comparison.",
            "save_status_of_fresh_run": "Not attempted due to missing portfolio code."
        }

    value_for_assessment_raw = ai_params.get("value_for_assessment")
    use_fractional_shares_override = ai_params.get("use_fractional_shares_override", None)

    # Initialize the detailed response structure
    response = {
        "portfolio_code": portfolio_code,
        "status": "pending_all_stages",
        "saved_run_data_load_details": {
            "loaded_successfully": False, "status_from_loader": "init", "message": "Load not attempted yet.",
            "timestamp_utc": "N/A", "holdings_count": 0, "final_cash": 0.0,
            "error_details_from_loader": None, "warnings_from_loader": None,
            "holdings_list_detail_for_diff": []
        },
        "fresh_run_generation_details": {
            "generated_successfully": False, "status_from_generator": "init", "message": "Generation not attempted yet.",
            "timestamp_utc": "N/A", "parameters_used": {}, "holdings_count": 0, "final_cash": 0.0,
            "error_details_from_generator": None, "holdings_list_detail_for_diff": []
        },
        "comparison_outcome_summary": "Comparison pending all stages.",
        "save_status_of_fresh_run": "Save pending fresh run completion."
    }

    value_for_assessment_float: Optional[float] = None
    if value_for_assessment_raw is not None:
        try:
            value_for_assessment_float = float(value_for_assessment_raw)
            if value_for_assessment_float <= 0:
                 if not is_called_by_ai: print(f"Info (get_comparison): 'value_for_assessment' ({value_for_assessment_raw}) for portfolio '{portfolio_code}' is not positive. Fresh run may be untailored or use portfolio's default if tailoring value is invalid.")
        except ValueError:
            if not is_called_by_ai: print(f"Info (get_comparison): Non-numeric 'value_for_assessment' ('{value_for_assessment_raw}') for portfolio '{portfolio_code}'. Fresh run may be untailored.")
            value_for_assessment_float = None

    # --- 1. Load Previously Saved Output FROM CSV (Represents "Old Saved One") ---
    load_result_dict = await _load_custom_portfolio_run_from_csv(portfolio_code, is_called_by_ai=True)

    response["saved_run_data_load_details"]["status_from_loader"] = load_result_dict["status"]
    response["saved_run_data_load_details"]["message"] = load_result_dict["message"]
    
    saved_holdings_for_diff = []

    if load_result_dict["status"].startswith("success"):
        saved_output_data_full = load_result_dict.get("data", {})
        saved_holdings = saved_output_data_full.get("tailored_holdings", [])
        saved_cash = saved_output_data_full.get("final_cash_value", 0.0)
        saved_ts = saved_output_data_full.get("timestamp_utc", "N/A")
        saved_holdings_for_diff = saved_holdings
        response["saved_run_data_load_details"].update({
            "loaded_successfully": True, "timestamp_utc": saved_ts,
            "holdings_count": len(saved_holdings), "final_cash": safe_score(saved_cash),
            "warnings_from_loader": load_result_dict.get("warnings"),
            "holdings_list_detail_for_diff": saved_holdings
        })
    else:
        response["saved_run_data_load_details"]["loaded_successfully"] = False
        response["saved_run_data_load_details"]["error_details_from_loader"] = load_result_dict.get("error_details", load_result_dict["message"])

    # --- 2. Calculate New Output (Fresh Run) ---
    portfolio_config_for_fresh_run = None
    fresh_run_holdings_list_from_execution = []
    final_cash_fresh_run_execution: float = 0.0
    total_value_for_fresh_run_param_execution: Optional[float] = None

    if not os.path.exists(PORTFOLIO_DB_FILE):
        err_msg_db = f"Portfolio configuration database '{PORTFOLIO_DB_FILE}' not found."
        response["fresh_run_generation_details"].update({
            "message": err_msg_db, "error_details_from_generator": err_msg_db,
            "status_from_generator": "error_db_not_found" })
    else:
        try:
            with open(PORTFOLIO_DB_FILE, 'r', encoding='utf-8', newline='') as f_db_fresh:
                reader_fresh = csv.DictReader(f_db_fresh)
                for row_fresh in reader_fresh:
                    if row_fresh.get('portfolio_code','').strip().lower() == portfolio_code.lower():
                        portfolio_config_for_fresh_run = dict(row_fresh)
                        break
            if not portfolio_config_for_fresh_run:
                err_msg_code = f"Portfolio code '{portfolio_code}' not found in database."
                response["fresh_run_generation_details"].update({
                    "message": err_msg_code, "error_details_from_generator": err_msg_code,
                    "status_from_generator": "error_code_not_found_in_db" })
            else:
                frac_shares_for_fresh_run: bool
                if use_fractional_shares_override is not None:
                    frac_shares_for_fresh_run = use_fractional_shares_override
                else:
                    csv_frac_shares_str = portfolio_config_for_fresh_run.get('frac_shares','false').strip().lower()
                    frac_shares_for_fresh_run = csv_frac_shares_str in ['true', 'yes']

                tailor_fresh_run_flag = isinstance(value_for_assessment_float, (int, float)) and value_for_assessment_float > 0
                total_value_for_fresh_run_param_execution = value_for_assessment_float if tailor_fresh_run_flag else None

                fresh_run_parameters_used = {
                    "target_value_for_tailoring": total_value_for_fresh_run_param_execution,
                    "fractional_shares_used_in_run": frac_shares_for_fresh_run,
                    "tailoring_was_requested": tailor_fresh_run_flag,
                    "config_ema_sensitivity": portfolio_config_for_fresh_run.get('ema_sensitivity'),
                    "config_amplification": portfolio_config_for_fresh_run.get('amplification')
                }
                response["fresh_run_generation_details"]["parameters_used"] = fresh_run_parameters_used

                _, _, temp_final_cash_fresh, temp_fresh_holdings_list = await process_custom_portfolio(
                    portfolio_data_config=portfolio_config_for_fresh_run,
                    tailor_portfolio_requested=tailor_fresh_run_flag,
                    frac_shares_singularity=frac_shares_for_fresh_run,
                    total_value_singularity=total_value_for_fresh_run_param_execution,
                    is_custom_command_simplified_output=True, # Output of this internal run is not directly shown
                    is_called_by_ai=True # This is an internal call by this tool
                )
                fresh_run_holdings_list_from_execution = temp_fresh_holdings_list
                final_cash_fresh_run_execution = temp_final_cash_fresh

                response["fresh_run_generation_details"].update({
                    "generated_successfully": True, "status_from_generator": "success",
                    "message": "Fresh run generated successfully.", "timestamp_utc": datetime.now(pytz.UTC).isoformat(),
                    "holdings_count": len(fresh_run_holdings_list_from_execution),
                    "final_cash": safe_score(final_cash_fresh_run_execution),
                    "holdings_list_detail_for_diff": fresh_run_holdings_list_from_execution
                })
        except Exception as e_fresh_run:
            tb_str_fresh = traceback.format_exc()
            error_detail_fresh = f"Exception: {str(e_fresh_run)} (Traceback: {tb_str_fresh[:300]}...)"
            response["fresh_run_generation_details"].update({
                "message": f"Error during fresh run of '{portfolio_code}': {error_detail_fresh}",
                "error_details_from_generator": error_detail_fresh,
                "status_from_generator": "error_fresh_run_exception" })

    # --- 3. Compare New Output with Old Saved One & Construct Summary ---
    saved_ok = response["saved_run_data_load_details"]["loaded_successfully"]
    fresh_ok = response["fresh_run_generation_details"]["generated_successfully"]
    s_details = response["saved_run_data_load_details"]
    f_details = response["fresh_run_generation_details"]

    summary_parts = [f"Comparison report for portfolio '{portfolio_code}':"]

    if saved_ok:
        summary_parts.append(
            f"Previously saved run (Timestamp: {s_details['timestamp_utc']}): Loaded successfully with {s_details['holdings_count']} holdings and cash ${s_details['final_cash']:,.2f}."
            f"{' Warnings during load: ' + '; '.join(s_details.get('warnings_from_loader', [])[:1]) + ('...' if len(s_details.get('warnings_from_loader',[])) > 1 else '') if s_details.get('warnings_from_loader') else ''}"
        )
    else:
        summary_parts.append(f"Previously saved run data: Failed to load. Reason: {s_details['message']}")

    if fresh_ok:
        fresh_val_disp = 'N/A'; frac_s_disp = "N/A"
        if f_details.get("parameters_used"):
            if f_details['parameters_used'].get('tailoring_was_requested'):
                val = f_details['parameters_used'].get('target_value_for_tailoring')
                fresh_val_disp = f"${val:,.2f}" if val is not None else "$0.00 (Value not specified)"
            else: fresh_val_disp = 'Untailored'
            frac_s_disp = 'Yes' if f_details['parameters_used'].get('fractional_shares_used_in_run') else 'No'
        else: fresh_val_disp = "Params N/A"
        summary_parts.append(
            f"Fresh run (Target Value: {fresh_val_disp}, FracShares: {frac_s_disp}): Generated successfully with {f_details['holdings_count']} holdings and final cash ${f_details['final_cash']:,.2f}."
        )
    else:
        summary_parts.append(f"Fresh run: Failed to generate. Reason: {f_details['message']}")

    table_data_comparison_for_print = []

    if saved_ok and fresh_ok:
        response["status"] = "success_comparison_available"
        
        s_holdings_list_for_calc = s_details["holdings_list_detail_for_diff"]
        f_holdings_list_for_calc = f_details["holdings_list_detail_for_diff"]

        s_tickers_map = {str(h['ticker']).upper(): h for h in s_holdings_list_for_calc if h.get('ticker')}
        f_tickers_map = {str(h['ticker']).upper(): h for h in f_holdings_list_for_calc if h.get('ticker')}

        s_ticker_set = set(s_tickers_map.keys())
        f_ticker_set = set(f_tickers_map.keys())
        all_unique_tickers_for_table = sorted(list(s_ticker_set.union(f_ticker_set)))

        added_tickers = sorted(list(f_ticker_set - s_ticker_set))
        removed_tickers = sorted(list(s_ticker_set - f_ticker_set))
        common_tickers = sorted(list(s_ticker_set.intersection(f_ticker_set)))

        summary_parts.append(f"Holdings Count: Saved run had {s_details['holdings_count']}, Fresh run has {f_details['holdings_count']}.")
        summary_parts.append(f"Final Cash: Saved run was ${s_details['final_cash']:,.2f}, Fresh run is ${f_details['final_cash']:,.2f}.")

        if added_tickers:
            summary_parts.append(f"Tickers added in fresh run ({len(added_tickers)}): {', '.join(added_tickers[:5])}{'...' if len(added_tickers) > 5 else ''}.")
        if removed_tickers:
            summary_parts.append(f"Tickers removed from saved run ({len(removed_tickers)}): {', '.join(removed_tickers[:5])}{'...' if len(removed_tickers) > 5 else ''}.")
        
        changed_common_details_for_summary = []
        MAX_COMMON_TO_DETAIL_SUMMARY = 3
        common_detailed_count_summary = 0
        
        def get_numeric_shares_for_diff(shares_val):
            if isinstance(shares_val, (int, float)): return shares_val
            try: return float(shares_val)
            except (ValueError, TypeError): return 0.0
        
        frac_shares_in_fresh_run_for_table = f_details.get("parameters_used", {}).get("fractional_shares_used_in_run", False)

        for ticker_key_upper in all_unique_tickers_for_table:
            if ticker_key_upper == "CASH": continue

            s_item = s_tickers_map.get(ticker_key_upper)
            f_item = f_tickers_map.get(ticker_key_upper)

            saved_shares_raw = s_item.get('shares', 0) if s_item else 0
            fresh_shares_raw = f_item.get('shares', 0) if f_item else 0
            
            saved_shares_numeric = get_numeric_shares_for_diff(saved_shares_raw)
            fresh_shares_numeric = get_numeric_shares_for_diff(fresh_shares_raw)
            share_change = fresh_shares_numeric - saved_shares_numeric

            s_display_str_table = str(saved_shares_raw)
            if isinstance(saved_shares_raw, float) and not saved_shares_raw.is_integer(): s_display_str_table = f"{saved_shares_raw:.1f}"
            elif isinstance(saved_shares_raw, (int, float)): s_display_str_table = f"{saved_shares_raw:.0f}"
            
            f_display_str_table = str(fresh_shares_raw)
            if isinstance(fresh_shares_raw, (int,float)):
                if frac_shares_in_fresh_run_for_table and fresh_shares_raw != 0:
                    f_display_str_table = f"{float(fresh_shares_raw):.1f}"
                else:
                    f_display_str_table = f"{float(fresh_shares_raw):.0f}"
            
            change_display_table = "0"
            if not math.isclose(share_change, 0):
                is_frac_change = False
                if (isinstance(saved_shares_numeric, float) and not saved_shares_numeric.is_integer()) or \
                   (isinstance(fresh_shares_numeric, float) and not fresh_shares_numeric.is_integer()) or \
                   (not float(share_change).is_integer()):
                    is_frac_change = True
                change_display_table = f"{share_change:+.1f}" if is_frac_change else f"{share_change:+.0f}"
            
            table_data_comparison_for_print.append([ticker_key_upper, s_display_str_table, f_display_str_table, change_display_table])
            
            if ticker_key_upper in common_tickers:
                s_alloc = s_item.get('actual_money_allocation', 0.0) if s_item else 0.0
                f_alloc = f_item.get('actual_money_allocation', 0.0) if f_item else 0.0
                share_changed_summary = str(saved_shares_raw) != str(fresh_shares_raw)
                alloc_changed_significantly_summary = not math.isclose(safe_score(s_alloc), safe_score(f_alloc), abs_tol=0.01)
                if share_changed_summary or alloc_changed_significantly_summary:
                    if common_detailed_count_summary < MAX_COMMON_TO_DETAIL_SUMMARY:
                        change_desc_summary = f"{ticker_key_upper}: "
                        if share_changed_summary: change_desc_summary += f"Shares {s_display_str_table} -> {f_display_str_table}"
                        if alloc_changed_significantly_summary:
                            if share_changed_summary: change_desc_summary += ", "
                            change_desc_summary += f"Value ${safe_score(s_alloc):.2f} -> ${safe_score(f_alloc):.2f}"
                        changed_common_details_for_summary.append(change_desc_summary)
                        common_detailed_count_summary +=1
        
        if changed_common_details_for_summary:
            summary_parts.append(f"Key changes in common tickers: {'; '.join(changed_common_details_for_summary)}{'...' if common_detailed_count_summary == MAX_COMMON_TO_DETAIL_SUMMARY and len(common_tickers) > MAX_COMMON_TO_DETAIL_SUMMARY else ''}.")
        elif not added_tickers and not removed_tickers and common_tickers:
             summary_parts.append(f"No major share/value changes noted for the {len(common_tickers)} common tickers (summary limited to top changes). See table for full details.")
        elif not common_tickers and not added_tickers and not removed_tickers and (s_details['holdings_count'] > 0 or f_details['holdings_count'] > 0) :
            summary_parts.append("Ticker sets were different or one/both were empty resulting in no common tickers to compare in detail.")
        elif not added_tickers and not removed_tickers and not changed_common_details_for_summary and not common_tickers and s_details['holdings_count'] == 0 and f_details['holdings_count'] == 0:
            summary_parts.append("Both saved and fresh runs resulted in no stock holdings.")
        elif not changed_common_details_for_summary and not added_tickers and not removed_tickers and len(common_tickers) > 0 :
            summary_parts.append(f"No major share/value changes noted for the {len(common_tickers)} common tickers (summary limited to top changes). See table for full details.")

    elif fresh_ok and not saved_ok:
        response["status"] = "partial_success_fresh_run_only"
        summary_parts.append(f"Comparison with previous run is not possible: {s_details['message']}")
    elif saved_ok and not fresh_ok:
        response["status"] = "partial_success_saved_data_only"
        summary_parts.append(f"Comparison with a fresh run is not possible: {f_details['message']}")
    else:
        response["status"] = "error_all_stages_failed"
        summary_parts.append("Full comparison impossible as both loading saved data and generating fresh run failed.")

    response["comparison_outcome_summary"] = " ".join(summary_parts)

    # --- 4. Output Table to Terminal (ALWAYS if comparison was possible) ---
    # MODIFIED CONDITION: Print table if saved_ok and fresh_ok, regardless of is_called_by_ai
    if saved_ok and fresh_ok:
        print("\n--- Portfolio Holdings Comparison (Share Changes) ---") # This now prints to console even if AI called.
        if table_data_comparison_for_print:
            print(tabulate(table_data_comparison_for_print,
                           headers=["Ticker", "Shares (Saved)", "Shares (Fresh)", "Change (Fresh - Saved)"],
                           tablefmt="pretty"))
        else:
            print("No stock holdings data available in one or both portfolios to compare share changes (excluding cash).")
        print("---------------------------------------------------\n")
    elif not is_called_by_ai: # If not called by AI, and comparison failed, print the notice.
        print(f"\n--- Portfolio Comparison Notice (CLI) ---")
        print(response["comparison_outcome_summary"]) # Print the summary which includes reasons for no table
        print(f"-----------------------------------\n")


    # --- 5. Overwrite the CSV with the New Custom Output (Fresh Run) ---
    if fresh_ok:
        if not is_called_by_ai: # Inform CLI user about the save action by this tool
            print(f"Info (get_comparison): Now saving/overwriting the fresh run output for portfolio '{portfolio_code}' to its CSV file...")
        try:
            await _save_custom_portfolio_run_to_csv(
                portfolio_code=portfolio_code,
                tailored_stock_holdings=fresh_run_holdings_list_from_execution, # from step 2
                final_cash=final_cash_fresh_run_execution, # from step 2
                total_portfolio_value_for_percent_calc=total_value_for_fresh_run_param_execution, # from step 2
                is_called_by_ai=True # This is an internal save action by the tool
            )
            response["save_status_of_fresh_run"] = f"Successfully saved fresh run output for portfolio '{portfolio_code}' to CSV."
            if not is_called_by_ai:
                 print(f"Info (get_comparison): Fresh run output for '{portfolio_code}' has been saved/overwritten by the comparison tool.")
        except Exception as e_save_fresh:
            response["save_status_of_fresh_run"] = f"Error saving fresh run output for portfolio '{portfolio_code}': {e_save_fresh}"
            if not is_called_by_ai:
                 print(f"Error (get_comparison): Failed to save fresh run output for '{portfolio_code}': {e_save_fresh}")
    else:
        response["save_status_of_fresh_run"] = "Fresh run was not successful, so it was not saved by the comparison tool."
        if not is_called_by_ai:
            print(f"Info (get_comparison): Fresh run for '{portfolio_code}' was not successful, so no output was saved by the comparison tool.")

    return response

async def handle_invest_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    # suppress_prints = is_called_by_ai # Use this flag for process_custom_portfolio
    if not is_called_by_ai: print("\n--- /invest Command ---")
    portfolio_data_config_invest = {'risk_type': 'stock', 'risk_tolerance': '10', 'remove_amplification_cap': 'true'}
    tailor_run, total_val_run, frac_s_run = False, None, False

    if ai_params: # AI Call
        # Parameter validation as before...
        try:
            ema_sens = int(ai_params.get("ema_sensitivity"))
            if ema_sens not in [1,2,3]: return "Error (AI /invest): Invalid EMA sensitivity."
            portfolio_data_config_invest['ema_sensitivity'] = str(ema_sens)
            portfolio_data_config_invest['amplification'] = str(float(ai_params.get("amplification")))
            sub_portfolios = ai_params.get("sub_portfolios")
            if not sub_portfolios: return "Error (AI /invest): 'sub_portfolios' required."
            portfolio_data_config_invest['num_portfolios'] = str(len(sub_portfolios))
            total_weight_ai = 0
            for i, sub_p in enumerate(sub_portfolios, 1):
                tickers, weight = sub_p.get("tickers"), sub_p.get("weight")
                if not tickers or weight is None: return f"Error (AI /invest): Sub-portfolio {i} missing tickers/weight."
                weight_val = float(weight)
                if not (0 <= weight_val <= 100): return f"Error (AI /invest): Weight for sub-portfolio {i} out of range."
                total_weight_ai += weight_val
                portfolio_data_config_invest[f'tickers_{i}'] = str(tickers).upper()
                portfolio_data_config_invest[f'weight_{i}'] = f"{weight_val:.2f}"
            if not math.isclose(total_weight_ai, 100.0, abs_tol=1.0): return f"Error (AI /invest): Weights must sum to ~100. Got {total_weight_ai:.2f}."

            tailor_run = ai_params.get("tailor_to_value", False)
            if tailor_run:
                total_val_run = ai_params.get("total_value")
                if total_val_run is None or float(total_val_run) <= 0: return "Error (AI /invest): Positive 'total_value' required for tailoring."
                total_val_run = float(total_val_run)
            frac_s_run = ai_params.get("use_fractional_shares", False)
            portfolio_data_config_invest['frac_shares'] = str(frac_s_run).lower()
        except (KeyError, ValueError) as e: return f"Error (AI /invest): Parameter issue: {e}"
        # Common processing
        # Pass is_called_by_ai=True to process_custom_portfolio
        tailored_list_str, combined_data, final_cash, _ = await process_custom_portfolio(
            portfolio_data_config=portfolio_data_config_invest, tailor_portfolio_requested=tailor_run,
            frac_shares_singularity=frac_s_run, total_value_singularity=total_val_run,
            is_custom_command_simplified_output=tailor_run, is_called_by_ai=True # True here
        )
        summary = f"/invest analysis completed (EMA Sens: {portfolio_data_config_invest['ema_sensitivity']}, Amp: {portfolio_data_config_invest['amplification']}). "
        if tailor_run:
            summary += f"Tailored to ${total_val_run:,.2f} (FracShares: {frac_s_run}). Final Cash: ${final_cash:,.2f}. "
            summary += "Top holdings: " + (", ".join(tailored_list_str[:3]) + "..." if len(tailored_list_str)>3 else ", ".join(tailored_list_str)) if tailored_list_str else "No stock holdings."
        else:
            summary += "Top combined allocations: " + (", ".join([f"{d['ticker']}({d.get('combined_percent_allocation_adjusted',0):.1f}%)" for d in combined_data[:3] if 'ticker' in d])) if combined_data else "No combined allocation data."
        return summary
    else: # --- CLI Path ---
        # This is the original CLI logic from M.I.C. Singularity 29.05.25.py
        try:
            ema_sens_str_cli = input("Enter EMA sensitivity (1: Weekly, 2: Daily, 3: Hourly): ")
            ema_sensitivity_cli = int(ema_sens_str_cli)
            if ema_sensitivity_cli not in [1, 2, 3]:
                print("Invalid EMA sensitivity. Must be 1, 2, or 3.")
                return None 

            amp_str_cli = input("Enter amplification factor (e.g., 0.25, 0.5, 1, 2, 3, 4, 5): ")
            amplification_cli = float(amp_str_cli)
            
            num_port_str_cli = input("How many portfolios would you like to calculate? (e.g., 2): ")
            num_portfolios_cli = int(num_port_str_cli)
            if num_portfolios_cli <= 0:
                print("Number of portfolios must be greater than 0.")
                return None

            portfolio_data_config_invest['ema_sensitivity'] = str(ema_sensitivity_cli)
            portfolio_data_config_invest['amplification'] = str(amplification_cli)
            portfolio_data_config_invest['num_portfolios'] = str(num_portfolios_cli)

            current_total_weight_cli = 0.0
            for i in range(1, num_portfolios_cli + 1):
                print(f"\n--- Portfolio {i} ---")
                tickers_input_cli = input(f"Enter tickers for Portfolio {i} (comma-separated, e.g., AAPL,MSFT): ").upper()
                if not tickers_input_cli.strip():
                    print("Tickers cannot be empty. Please start over or provide valid tickers.") # Matched original
                    return None 
                portfolio_data_config_invest[f'tickers_{i}'] = tickers_input_cli

                weight_val_cli = 0.0
                if i == num_portfolios_cli: 
                    weight_val_cli = 100.0 - current_total_weight_cli
                    if weight_val_cli < -0.01: 
                        print(f"Error: Previous weights ({current_total_weight_cli}%) exceed 100%. Cannot set weight for final portfolio.")
                        return None
                    weight_val_cli = max(0, weight_val_cli) 
                    print(f"Weight for Portfolio {i} automatically set to: {weight_val_cli:.2f}%")
                else:
                    remaining_weight_cli = 100.0 - current_total_weight_cli
                    weight_str_cli = input(f"Enter weight for Portfolio {i} (0-{remaining_weight_cli:.2f}%): ")
                    weight_val_cli = float(weight_str_cli)
                    if not (-0.01 < weight_val_cli < remaining_weight_cli + 0.01): # Minor tolerance from original
                        print(f"Invalid weight. Must be between 0 and {remaining_weight_cli:.2f}%.")
                        return None
                portfolio_data_config_invest[f'weight_{i}'] = f"{weight_val_cli:.2f}"
                current_total_weight_cli += weight_val_cli
            
            if not math.isclose(current_total_weight_cli, 100.0, abs_tol=0.1): # Check from original
                print(f"Warning: Total weights sum to {current_total_weight_cli:.2f}%, not 100%. Results might be skewed.")

            tailor_str_cli = input("Tailor the table to your portfolio value? (yes/no): ").lower()
            tailor_portfolio_for_run = tailor_str_cli == 'yes'
            if tailor_portfolio_for_run:
                val_str_cli = input("Enter the total value for the combined portfolio (e.g., 10000): ")
                total_value_for_tailoring_run = float(val_str_cli)
                if total_value_for_tailoring_run <= 0: 
                    print("Portfolio value must be positive.")
                    return None 
            
            frac_s_str_cli = input("Tailor using fractional shares? (yes/no): ").lower()
            frac_shares_for_tailoring_run = frac_s_str_cli == 'yes'
            portfolio_data_config_invest['frac_shares'] = str(frac_shares_for_tailoring_run).lower()

            print("\nCLI: Processing /invest request...")
            await process_custom_portfolio(
                portfolio_data_config=portfolio_data_config_invest,
                tailor_portfolio_requested=tailor_portfolio_for_run,
                frac_shares_singularity=frac_shares_for_tailoring_run,
                total_value_singularity=total_value_for_tailoring_run,
                is_custom_command_simplified_output=False # /invest CLI always shows full output
            )
            print("\n/invest analysis complete.") # From original
            return None # CLI path prints directly, no summary string for AI

        except ValueError: 
            print("CLI Error: Invalid input. Please enter numbers where expected (e.g., for sensitivity, amplification, count, value, weight).")
            return None
        except Exception as e_invest_cli: 
            print(f"CLI Error occurred during /invest: {e_invest_cli}")
            traceback.print_exc()
            return None

async def handle_spear_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """Handles the /spear command for CLI and AI."""
    if not is_called_by_ai:
        print("\n--- /spear Command ---")

    params = {}
    try:
        if is_called_by_ai and ai_params:
            params = ai_params.copy()

            # --- MODIFICATION START ---
            if not params.get('ticker'):
                return "Error: Ticker is a required parameter for SPEAR analysis."
            
            params['ticker'] = params.get('ticker', '').replace(".", "-")

            # Intelligently load from databases if info is not already provided by AI
            try:
                spear_bank_df = pd.read_csv('spear_bank.csv')
                stock_data = spear_bank_df[spear_bank_df['Ticker'].str.lower() == params['ticker'].lower()]
                if not stock_data.empty:
                    params.setdefault('sector_relevance', float(stock_data['Sector to Market'].iloc[0]))
                    params.setdefault('stock_relevance', float(stock_data['Stock to Sector'].iloc[0]))
                    params.setdefault('is_meme_stock', stock_data['Meme Stock'].iloc[0])
            except (FileNotFoundError, KeyError):
                pass  # Ignore if file or columns don't exist

            try:
                spear_trend_df = pd.read_csv('spear_trend.csv')
                if not spear_trend_df.empty:
                    params.setdefault('market_trend', spear_trend_df['Market Trend'].iloc[0])
                    params.setdefault('market_reversal_likely', spear_trend_df['Reversal Likely'].iloc[0])
            except (FileNotFoundError, KeyError):
                pass # Ignore if file or columns don't exist

            # Handle natural language variations for AI
            if 'earnings_date' in params and params['earnings_date'].lower() == 'tomorrow':
                params['earnings_date'] = (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            if 'earnings_time' in params:
                time_str = params['earnings_time'].lower()
                if 'after' in time_str: params['earnings_time'] = 'a'
                elif 'pre' in time_str or 'before' in time_str: params['earnings_time'] = 'p'
            
            if 'market_trend' in params:
                trend_str = params['market_trend'].lower()
                if 'upward' in trend_str: params['market_trend'] = 'Upwards'
                elif 'downward' in trend_str: params['market_trend'] = 'Downwards'
                elif 'no' in trend_str: params['market_trend'] = 'No Trend'

            # Final check for any remaining missing required parameters
            required_keys = ['ticker', 'sector_relevance', 'stock_relevance', 'hype', 'earnings_date', 'earnings_time']
            missing_keys = [key for key in required_keys if key not in params]
            if missing_keys:
                return f"Error: Missing required parameters for SPEAR analysis: {', '.join(missing_keys)}. Please provide them."
            # --- MODIFICATION END ---

        else: # CLI path (remains the same)
            ticker_in = ask_singularity_input('Enter the ticker', validation_fn=lambda x: x.strip(), is_called_by_ai=is_called_by_ai)
            if not ticker_in: return
            params['ticker'] = ticker_in.replace(".", "-")
            
            stock = yf.Ticker(params['ticker'])
            market_cap = stock.info.get('marketCap', 0)

            try:
                spear_bank_df = pd.read_csv('spear_bank.csv')
                stock_data = spear_bank_df[spear_bank_df['Ticker'].str.lower() == params['ticker'].lower()]
                if not stock_data.empty:
                    params['sector_relevance'] = float(stock_data['Sector to Market'].iloc[0])
                    params['stock_relevance'] = float(stock_data['Stock to Sector'].iloc[0])
                    params['is_meme_stock'] = stock_data['Meme Stock'].iloc[0]
                    print("Info: Found database entry in 'spear_bank.csv'. Auto-filling some inputs.")
            except FileNotFoundError:
                pass 

            try:
                spear_trend_df = pd.read_csv('spear_trend.csv')
                if not spear_trend_df.empty:
                    params['market_trend'] = spear_trend_df['Market Trend'].iloc[0]
                    params['market_reversal_likely'] = spear_trend_df['Reversal Likely'].iloc[0]
                    print("Info: Found database entry in 'spear_trend.csv'. Auto-filling trend inputs.")
            except FileNotFoundError:
                pass

            if 'sector_relevance' not in params:
                params['sector_relevance'] = float(ask_singularity_input('Enter The Sector To Market Relevance Number (1 to 5)', lambda x: 1<=float(x)<=5, is_called_by_ai=is_called_by_ai))
            if 'stock_relevance' not in params:
                params['stock_relevance'] = float(ask_singularity_input('Enter The Stock To Sector Relevance Number (1 to 5)', lambda x: 1<=float(x)<=5, is_called_by_ai=is_called_by_ai))
            if 'hype' not in params:
                params['hype'] = float(ask_singularity_input('Enter The Hype Value (-1 to 1)', lambda x: -1<=float(x)<=1, is_called_by_ai=is_called_by_ai))
            
            if 'is_meme_stock' not in params:
                 params['is_meme_stock'] = ask_singularity_input('Is it a meme stock? (Yes/y or No/n)', default_val="No", is_called_by_ai=is_called_by_ai) if market_cap < 5e10 else "No"
            
            if 'market_trend' not in params:
                params['market_trend'] = ask_singularity_input('What Is The Market Trend (Upwards/u, Downwards/d, No Trend/n)', default_val="No Trend", is_called_by_ai=is_called_by_ai) if params['is_meme_stock'].lower() in ["no", "n"] else "No Trend"

            if 'market_reversal_likely' not in params:
                 params['market_reversal_likely'] = ask_singularity_input('Is A Market Reversal Likely (Yes/y or No/n)', default_val="No", is_called_by_ai=is_called_by_ai) if params['market_trend'].lower() in ["no trend", "n"] and params['is_meme_stock'].lower() in ["no", "n"] else "No"

            if 'earnings_date' not in params:
                params['earnings_date'] = ask_singularity_input('Enter The Earnings Date (YYYY-MM-DD)', lambda x: datetime.strptime(x, "%Y-%m-%d"), is_called_by_ai=is_called_by_ai)
            if 'earnings_time' not in params:
                params['earnings_time'] = ask_singularity_input('Enter The Earnings Time (p for Pre-Market, a for After Hours)', lambda x: x.lower() in ['p', 'a'], is_called_by_ai=is_called_by_ai)

    except (ValueError, TypeError, AttributeError) as e:
        err_msg = f"Error: Invalid input provided. {e}"
        if not is_called_by_ai: print(err_msg)
        return err_msg if is_called_by_ai else None

    # --- Core Logic ---
    tckr = yf.Ticker(params['ticker'])
    # Use auto_adjust=False to get 'Close' price, which is needed for SPEAR logic
    # fast_info might be adjusted, let's use history for consistency
    hist_price = tckr.history(period="1d", auto_adjust=False)
    actual_price = hist_price['Close'].iloc[-1] if not hist_price.empty else None

    if not actual_price:
        err_msg = f"Error: Could not fetch live price for {params['ticker']}."
        if not is_called_by_ai: print(err_msg)
        return err_msg if is_called_by_ai else None
        
    fear_and_greed_data = fear_and_greed.get()
    fear_value = round(fear_and_greed_data[0])
    
    prediction_data = calculate_spear_prediction(
        params['ticker'], params['sector_relevance'], params['stock_relevance'], params['hype'],
        fear_value, params['earnings_date'], params['earnings_time'],
        params.get('market_trend', "No Trend"), params.get('market_reversal_likely', "No"), 
        params.get('is_meme_stock', "No"), 1, actual_price
    )
    
    # --- Output Formatting ---
    prediction = prediction_data['prediction']
    d1 = (actual_price * ((prediction / 2) + 1))
    
    trade_recommendation = "No trade recommendation."
    if -0.025 <= prediction <= 0.025:
        trade_recommendation = "Do Not Trade Options As The Expected Change Is Too Small. Place a Buy Stop Order At ±1% On Shares If Desired."
    elif 0.025 < prediction <= 0.05:
        trade_recommendation = "Do Not Trade Options. Place A Buy Stop Order For A Long Position At +2%."
    elif prediction > 0.05:
        strike_price = round(d1, -1) if actual_price > 100 else round(d1, 0) if actual_price > 10 else round(d1, 1)
        trade_recommendation = f"Buy Calls (${strike_price:.2f} Strike Price) Or Place An Order For A Long Position. Prepare With Proper Risk Management Such As Taking A Smaller Position."
    elif -0.05 <= prediction < -0.025:
        trade_recommendation = "Do Not Trade Options. Place A Buy Stop Order For A Short Position At -2%."
    elif prediction < -0.05:
        strike_price = round(d1, -1) if actual_price > 100 else round(d1, 0) if actual_price > 10 else round(d1, 1)
        trade_recommendation = f"Buy Puts (${strike_price:.2f} Strike Price) Or Place An Order For A Short Position. Prepare With Proper Risk Management Such As Taking A Smaller Position."

    if is_called_by_ai:
        summary = (f"SPEAR analysis for {params['ticker'].upper()} predicts a {prediction:.2%} change at earnings. "
                   f"The model's recommendation is to '{trade_recommendation}'. "
                   f"A graph showing the predicted price has been saved.")
        plot_spear_graph(params['ticker'], actual_price * (1 + prediction), is_called_by_ai=True)
        return summary
    else:
        # CLI prints full details
        print("\n" + "="*50)
        summary_trim, industry, sector, market_cap_summary = business_summary_spear(params['ticker'], 3)
        print(f"**Business Summary for {params['ticker'].upper()}:**\n{summary_trim}")
        print(f"\nIndustry: {industry}\nSector: {sector}\nMarket Cap: {market_cap_summary}")
        
        print("\n**--- SPEAR Analysis Results ---**")
        print(f"# Ticker: {params['ticker'].upper()}")
        
        print(f"\n**Trade Recommendation:**\n{trade_recommendation}\n")
        
        data_table = {
            'Prediction At Earnings': f"{prediction:.2%}",
            'Growth From Now To Earnings': f"{prediction_data['prediction_in_time']:.2%}",
            'Live Price': f"${actual_price:,.2f}",
            'Earnings Date': params['earnings_date'],
            'Earnings Time': "After Hours" if params['earnings_time'] == 'a' else "Pre-Market",
            '1D% Change': f"{prediction_data['finance_data']['1D% Change']:.2%}",
            '1M% Change': f"{prediction_data['finance_data']['1M% Change']:.2%}",
            '1Y% Change': f"{prediction_data['finance_data']['1Y% Change']:.2%}",
        }
        print(tabulate(data_table.items(), headers=["Metric", "Value"], tablefmt="grid"))
        
        plot_spear_graph(params['ticker'], actual_price * (1 + prediction), is_called_by_ai=False)
        print("="*50 + "\n")
        return None
                 
async def collect_portfolio_inputs_singularity(portfolio_code_singularity: str, is_called_by_ai: bool = False) -> Optional[Dict[str, Any]]:
    """Collects inputs for a new custom portfolio config. (CLI focused)"""
    if not is_called_by_ai: print(f"\n--- Creating New Portfolio Configuration: '{portfolio_code_singularity}' ---")
    inputs = {'portfolio_code': portfolio_code_singularity}
    try:
        # EMA Sensitivity
        while True:
            ema_sens_str = input("Enter EMA sensitivity (1: Weekly, 2: Daily, 3: Hourly): ")
            try: ema_sens_val = int(ema_sens_str); assert ema_sens_val in [1,2,3]; inputs['ema_sensitivity'] = str(ema_sens_val); break
            except: print("Invalid input.")
        # Amplification
        valid_amps = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        while True:
            amp_str = input(f"Enter amplification ({', '.join(map(str, valid_amps))}): ")
            try: amp_val = float(amp_str); assert amp_val in valid_amps; inputs['amplification'] = str(amp_val); break
            except: print("Invalid input.")
        # Num Portfolios
        while True:
            num_port_str = input("Enter number of sub-portfolios (e.g., 2): ")
            try: num_p = int(num_port_str); assert num_p > 0; inputs['num_portfolios'] = str(num_p); break
            except: print("Must be > 0.")
        # Fractional Shares
        while True:
            frac_s_str = input("Allow fractional shares for tailoring? (yes/no): ").lower()
            if frac_s_str in ['yes', 'no']: inputs['frac_shares'] = 'true' if frac_s_str == 'yes' else 'false'; break
            else: print("Invalid input.")

        inputs['risk_tolerance'] = '10'; inputs['risk_type'] = 'stock'; inputs['remove_amplification_cap'] = 'true'
        current_total_weight = 0.0
        for i in range(1, int(inputs['num_portfolios']) + 1):
            print(f"\n--- Sub-Portfolio {i} ---")
            while True:
                tickers_in = input(f"Enter tickers for Sub-Portfolio {i} (comma-separated): ").upper()
                if tickers_in.strip(): inputs[f'tickers_{i}'] = tickers_in; break
                else: print("Tickers cannot be empty.")
            if i == int(inputs['num_portfolios']):
                weight_val = 100.0 - current_total_weight
                if weight_val < -0.01: print(f"Error: Previous weights sum to {current_total_weight}%."); return None
                weight_val = max(0, weight_val)
                print(f"Weight for Sub-Portfolio {i} auto-set to: {weight_val:.2f}%")
            else:
                remaining_w = 100.0 - current_total_weight
                while True:
                    weight_str = input(f"Enter weight for Sub-Portfolio {i} (0-{remaining_w:.2f}%): ")
                    try: weight_val = float(weight_str); assert -0.01 < weight_val < remaining_w + 0.01; break
                    except: print(f"Invalid weight. Must be 0 to {remaining_w:.2f}%.")
            inputs[f'weight_{i}'] = f"{weight_val:.2f}"; current_total_weight += weight_val
        if not math.isclose(current_total_weight, 100.0, abs_tol=0.1) and not is_called_by_ai:
            print(f"Warning: Sum of weights is {current_total_weight:.2f}%, not 100%.")
        return inputs
    except ValueError as ve:
        if not is_called_by_ai: print(f"Invalid input: {ve}. Config not saved.")
        return None
    except Exception as e:
        if not is_called_by_ai: print(f"Unexpected error during input: {e}"); traceback.print_exc()
        return None

async def handle_custom_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    if not is_called_by_ai:
        print("\n--- /custom Command ---")
    summary_for_ai = "Custom command initiated."

    if ai_params: # AI Call
        action = ai_params.get("action")
        portfolio_code_input = ai_params.get("portfolio_code")

        if not portfolio_code_input:
            return "Error for AI (/custom): 'portfolio_code' is required."

        if action == "run_existing_portfolio":
            tailor_run_ai = ai_params.get("tailor_to_value", False)
            total_value_ai_float: Optional[float] = None
            # frac_shares_override_ai: Optional[bool] = ai_params.get("use_fractional_shares") if "use_fractional_shares" in ai_params else None
            # The frac_shares_final_run_ai logic below correctly handles the override or config value

            if tailor_run_ai:
                total_value_ai_raw = ai_params.get("total_value")
                if total_value_ai_raw is None:
                    return "Error for AI (/custom run): 'total_value' required when 'tailor_to_value' is true."
                try:
                    total_value_ai_float = float(total_value_ai_raw)
                    if total_value_ai_float <= 0:
                        return "Error for AI (/custom run): 'total_value' must be a positive number."
                except ValueError:
                    return "Error for AI (/custom run): 'total_value' is not a valid number."

            portfolio_config_from_db = None
            if not os.path.exists(PORTFOLIO_DB_FILE):
                return f"Error for AI (/custom run): Portfolio database '{PORTFOLIO_DB_FILE}' not found."
            try:
                with open(PORTFOLIO_DB_FILE, 'r', encoding='utf-8', newline='') as f_db:
                    reader = csv.DictReader(f_db)
                    for row in reader:
                        if row.get('portfolio_code', '').strip().lower() == portfolio_code_input.lower():
                            portfolio_config_from_db = row
                            break
                if not portfolio_config_from_db:
                    return f"Error for AI (/custom run): Portfolio code '{portfolio_code_input}' not found in database."

                frac_shares_override_ai: Optional[bool] = ai_params.get("use_fractional_shares") if "use_fractional_shares" in ai_params else None # Get override
                frac_shares_final_run_ai: bool
                if frac_shares_override_ai is not None:
                    frac_shares_final_run_ai = frac_shares_override_ai
                else:
                    csv_frac_shares_str = portfolio_config_from_db.get('frac_shares', 'false').strip().lower()
                    frac_shares_final_run_ai = csv_frac_shares_str in ['true', 'yes']

                # Process the portfolio
                _, _, final_cash_value_run, tailored_data_run = await process_custom_portfolio(
                    portfolio_data_config=portfolio_config_from_db,
                    tailor_portfolio_requested=tailor_run_ai,
                    frac_shares_singularity=frac_shares_final_run_ai,
                    total_value_singularity=total_value_ai_float,
                    is_custom_command_simplified_output=True, # True for AI calls if tailored
                    is_called_by_ai=True
                )

                # <<< MODIFIED SAVE CALL FOR AI PATH >>>
                await _save_custom_portfolio_run_to_csv(
                    portfolio_code=portfolio_code_input,
                    tailored_stock_holdings=tailored_data_run, # This is tailored_portfolio_structured_data
                    final_cash=final_cash_value_run,
                    total_portfolio_value_for_percent_calc=total_value_ai_float if tailor_run_ai else None,
                    is_called_by_ai=True
                )
                # <<< END MODIFIED SAVE CALL FOR AI PATH >>>

                summary_for_ai = f"Analysis for custom portfolio '{portfolio_code_input}' completed. Detailed run output saved/overwritten to CSV. "
                if tailor_run_ai:
                    summary_for_ai += f"Tailored to ${total_value_ai_float:,.2f} (Fractional Shares: {frac_shares_final_run_ai}). Final cash: ${final_cash_value_run:,.2f}."
                return summary_for_ai
            except Exception as e_ai_run:
                return f"Error processing AI request for /custom run '{portfolio_code_input}': {str(e_ai_run)}"

        elif action == "save_portfolio_data":
            date_to_save_legacy = ai_params.get("date_to_save")
            if not date_to_save_legacy:
                return "Error for AI (/custom save_portfolio_data): 'date_to_save' is required."
            try:
                datetime.strptime(date_to_save_legacy, '%m/%d/%Y')
            except ValueError:
                return f"Error for AI (/custom save_portfolio_data): Invalid date format '{date_to_save_legacy}'. Use MM/DD/YYYY."
            await save_portfolio_data_singularity(portfolio_code_input, date_to_save_legacy, is_called_by_ai=True)
            return f"Legacy combined percentage data for portfolio '{portfolio_code_input}' requested for save on {date_to_save_legacy}."
        else:
            return f"Error for AI (/custom): Unknown or unsupported action '{action}'."

    else: # CLI Path
        # ... (CLI argument parsing and new portfolio creation logic remains the same) ...
        if not args:
            print("Usage: /custom <portfolio_code_or_#> [save_data_code 3725 (for legacy combined % save)]")
            print("Note: Running a portfolio (e.g. /custom MYPORT) now automatically saves/overwrites its detailed run output to CSV.") # Updated note
            return None

        portfolio_code_cli = args[0].strip()
        legacy_save_code_cli = args[1].strip() if len(args) > 1 else None
        is_new_code_auto_cli = False

        if portfolio_code_cli == '#':
            next_code_num = 1
            if os.path.exists(PORTFOLIO_DB_FILE):
                max_code = 0
                try:
                    df_codes_cli = pd.read_csv(PORTFOLIO_DB_FILE)
                    numeric_codes_cli = pd.to_numeric(df_codes_cli['portfolio_code'], errors='coerce').dropna()
                    if not numeric_codes_cli.empty: max_code = int(numeric_codes_cli.max())
                except Exception: pass
                next_code_num = max_code + 1
            portfolio_code_cli = str(next_code_num)
            is_new_code_auto_cli = True
            print(f"CLI: Using next available portfolio code: `{portfolio_code_cli}`")

        if legacy_save_code_cli == "3725":
            # ... (legacy save logic unchanged) ...
            if is_new_code_auto_cli:
                print("CLI Error: Cannot use '#' (auto-generated code) directly with the legacy '3725' save_data_code.")
                return None
            date_to_save_str_cli = input(f"CLI: Enter date (MM/DD/YYYY) to save legacy combined % data for portfolio '{portfolio_code_cli}': ")
            try:
                datetime.strptime(date_to_save_str_cli, '%m/%d/%Y')
                await save_portfolio_data_singularity(portfolio_code_cli, date_to_save_str_cli, is_called_by_ai=False)
            except ValueError: print("CLI: Invalid date format for legacy save. Save operation cancelled.")
            return None


        portfolio_config_from_db_cli = None
        if os.path.exists(PORTFOLIO_DB_FILE) and not is_new_code_auto_cli:
            try:
                with open(PORTFOLIO_DB_FILE, 'r', encoding='utf-8', newline='') as file_cli_db:
                    reader_cli_db = csv.DictReader(file_cli_db)
                    for row_cli_db in reader_cli_db:
                        if row_cli_db.get('portfolio_code', '').strip().lower() == portfolio_code_cli.lower():
                            portfolio_config_from_db_cli = row_cli_db; break
            except Exception as e_read_db_cli: print(f"CLI: Error reading portfolio DB: {e_read_db_cli}")

        if portfolio_config_from_db_cli is None:
            print(f"CLI: Portfolio code '{portfolio_code_cli}' not found or creating new. Starting interactive setup...")
            new_portfolio_config_cli = await collect_portfolio_inputs_singularity(portfolio_code_cli, is_called_by_ai=False)
            if new_portfolio_config_cli:
                await save_portfolio_to_csv(PORTFOLIO_DB_FILE, new_portfolio_config_cli, is_called_by_ai=False)
                portfolio_config_from_db_cli = new_portfolio_config_cli
                print(f"CLI: New portfolio configuration '{portfolio_code_cli}' saved.")
                run_now_cli_str = input(f"Run portfolio '{portfolio_code_cli}' now with this new configuration? (yes/no, default: yes): ").lower().strip()
                if run_now_cli_str == 'no': return None
            else: print("CLI: Portfolio configuration cancelled or incomplete."); return None
        
        # This is the part for running an existing or newly created portfolio via CLI
        if portfolio_config_from_db_cli:
            try:
                csv_frac_shares_str_cli = portfolio_config_from_db_cli.get('frac_shares', 'false').strip().lower()
                frac_shares_setting_from_config_cli = csv_frac_shares_str_cli in ['true', 'yes']
                frac_shares_for_this_run_cli = frac_shares_setting_from_config_cli

                print(f"--- Running Custom Portfolio: {portfolio_code_cli} ---")
                print(f"  Configuration default for fractional shares: {frac_shares_setting_from_config_cli}")

                tailor_this_run_cli = False
                total_value_for_this_run_cli: Optional[float] = None

                tailor_prompt_cli = input(f"CLI: Tailor portfolio '{portfolio_code_cli}' to a value for this run? (yes/no, default: no): ").lower().strip()
                if tailor_prompt_cli == 'yes':
                    tailor_this_run_cli = True
                    val_input_cli = input("CLI: Enter total portfolio value for tailoring: ").strip()
                    try:
                        total_value_for_this_run_cli = float(val_input_cli)
                        if total_value_for_this_run_cli <= 0:
                            print("CLI: Portfolio value must be positive. Proceeding without tailoring.")
                            tailor_this_run_cli = False
                        else: # Value is positive, ask about fractional shares for this run
                            override_frac_s_cli = input(f"CLI: Override fractional shares for this run? (current config: {frac_shares_setting_from_config_cli}) (yes/no/config, default: config): ").lower().strip()
                            if override_frac_s_cli == 'yes': frac_shares_for_this_run_cli = True
                            elif override_frac_s_cli == 'no': frac_shares_for_this_run_cli = False
                            # If 'config' or empty, frac_shares_for_this_run_cli remains as frac_shares_setting_from_config_cli
                    except ValueError:
                        print("CLI: Invalid portfolio value. Proceeding without tailoring.")
                        tailor_this_run_cli = False
                
                print(f"  For this run, using fractional shares: {frac_shares_for_this_run_cli}")
                if tailor_this_run_cli: print(f"  Tailoring to value: ${total_value_for_this_run_cli:,.2f}")
                else: print("  Not tailoring to a specific value (will show percentages if not tailored).")

                # Process the portfolio
                _, _, final_cash_cli_run, tailored_data_cli_run = await process_custom_portfolio(
                    portfolio_data_config=portfolio_config_from_db_cli,
                    tailor_portfolio_requested=tailor_this_run_cli,
                    frac_shares_singularity=frac_shares_for_this_run_cli,
                    total_value_singularity=total_value_for_this_run_cli,
                    is_custom_command_simplified_output=tailor_this_run_cli,
                    is_called_by_ai=False
                )

                # <<< MODIFIED SAVE CALL FOR CLI PATH >>>
                await _save_custom_portfolio_run_to_csv(
                    portfolio_code=portfolio_code_cli,
                    tailored_stock_holdings=tailored_data_cli_run, # This is tailored_portfolio_structured_data
                    final_cash=final_cash_cli_run,
                    total_portfolio_value_for_percent_calc=total_value_for_this_run_cli if tailor_this_run_cli else None,
                    is_called_by_ai=False
                )
                print(f"\nCLI: Custom portfolio analysis for `{portfolio_code_cli}` complete. Detailed run output saved/overwritten to CSV.")
                # <<< END MODIFIED SAVE CALL FOR CLI PATH >>>

            except Exception as e_custom_cli_run:
                print(f"CLI Error processing portfolio '{portfolio_code_cli}': {e_custom_cli_run}")
                traceback.print_exc()
        return None
    
# --- How to call the new save function ---
# You would modify your `handle_custom_command` function (CLI path).
# When a portfolio is run and results are obtained:
#
# async def handle_custom_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
#     # ... (existing logic to get portfolio_config_from_db_cli, tailor_this_run_cli, etc.) ...
#     if portfolio_config_from_db_cli: # And after running it
#         try:
#             # ... (logic to get frac_shares_for_this_run_cli, total_value_for_this_run_cli) ...
#
#             # Call process_custom_portfolio
#             _, _, final_cash_cli_run, tailored_data_cli_run = await process_custom_portfolio(
#                 portfolio_data_config=portfolio_config_from_db_cli,
#                 tailor_portfolio_requested=tailor_this_run_cli,
#                 frac_shares_singularity=frac_shares_for_this_run_cli,
#                 total_value_singularity=total_value_for_this_run_cli,
#                 is_custom_command_simplified_output=tailor_this_run_cli, 
#                 is_called_by_ai=False # For CLI call
#             )
#
#             # *** REPLACE THE OLD JSON SAVE WITH THE NEW CSV SAVE ***
#             # Old: await _save_custom_portfolio_run_output(portfolio_code_cli, tailored_data_cli_run, final_cash_cli_run, is_called_by_ai=False)
#             await _save_custom_portfolio_run_to_csv(
#                 portfolio_code=portfolio_code_cli, 
#                 tailored_stock_holdings=tailored_data_cli_run, # This is the list of stock dicts
#                 final_cash=final_cash_cli_run,
#                 total_portfolio_value_for_percent_calc=total_value_for_this_run_cli if tailor_this_run_cli else None, # Pass total value for cash % calc
#                 is_called_by_ai=False
#             )
#             print(f"\nCLI: Custom portfolio analysis for `{portfolio_code_cli}` complete. Detailed run output saved to CSV.")
#
#         except Exception as e_custom_cli_run:
#             # ... (error handling) ...
#     return None

# Similarly, if the AI calls `handle_custom_command` with `action: "run_existing_portfolio"`,
# that function should also be updated internally to use `_save_custom_portfolio_run_to_csv`.

# --- Breakout Command Functions --- (Assumed mostly unchanged, add is_called_by_ai flags)
async def run_breakout_analysis_singularity(is_called_by_ai: bool = False) -> dict:
    """
    Performs breakout analysis, prints results, saves data, and returns a dictionary
    with current, new, and removed breakout stocks.
    """
    if not is_called_by_ai:
        print("\n--- Running Breakout Analysis ---")

    invest_score_threshold = 100.0
    fraction_threshold = 3.0 / 4.0
    
    existing_tickers_data = {}
    if os.path.exists(BREAKOUT_TICKERS_FILE):
        try:
            df_existing = pd.read_csv(BREAKOUT_TICKERS_FILE)
            if not df_existing.empty:
                for col in ["Highest Invest Score", "Lowest Invest Score", "Live Price", "1Y% Change", "Invest Score"]:
                    if col in df_existing.columns:
                        if df_existing[col].dtype == 'object':
                            df_existing[col] = df_existing[col].astype(str).str.replace('%', '', regex=False).str.replace('$', '', regex=False).str.strip()
                        df_existing[col] = pd.to_numeric(df_existing[col], errors='coerce')
                existing_tickers_data = df_existing.set_index('Ticker').to_dict('index')
        except Exception as read_err:
            if not is_called_by_ai:
                print(f"Warning: Error reading existing breakout file '{BREAKOUT_TICKERS_FILE}': {read_err}. Proceeding without it.")

    existing_tickers_set = set(existing_tickers_data.keys())

    if not is_called_by_ai:
        print("Running TradingView Screening for new breakout candidates...")
    new_tickers_from_screener = []
    try:
        query = Query().select('name', 'close', 'change', 'volume', 'market_cap_basic', 'change|1W', 'average_volume_90d_calc') \
            .where(
                Column('market_cap_basic') >= 1_000_000_000,
                Column('volume') >= 1_000_000,
                Column('change|1W') >= 20,
                Column('close') >= 1,
                Column('average_volume_90d_calc') >= 1_000_000
            ).order_by('change', ascending=False).limit(100)
        scanner_results_tuple = await asyncio.to_thread(query.get_scanner_data, timeout=60)
        if scanner_results_tuple and isinstance(scanner_results_tuple, tuple) and len(scanner_results_tuple) > 0 and isinstance(scanner_results_tuple[1], pd.DataFrame):
            new_tickers_df = scanner_results_tuple[1]
            if 'name' in new_tickers_df.columns:
                new_tickers_from_screener = [str(t).split(':')[-1].replace('.', '-') for t in new_tickers_df['name'].tolist() if pd.notna(t)]
                new_tickers_from_screener = sorted(list(set(new_tickers_from_screener)))
                if not is_called_by_ai:
                    print(f"TradingView Screener found {len(new_tickers_from_screener)} potential new tickers.")
        else:
            if not is_called_by_ai:
                print("Warning: TradingView screening returned no data or unexpected format.")
    except Exception as screen_err:
        if not is_called_by_ai:
            print(f"Error during TradingView screening: {screen_err}")

    all_tickers_to_process = sorted(list(set(list(existing_tickers_data.keys()) + new_tickers_from_screener)))
    if not is_called_by_ai:
        print(f"Processing {len(all_tickers_to_process)} unique tickers (existing + new)...")

    temp_updated_data_for_df_build = []
    processed_count = 0
    for ticker_b in all_tickers_to_process:
        processed_count += 1
        if not is_called_by_ai and processed_count % 20 == 0 and len(all_tickers_to_process) > 20:
            print(f"  ...breakout processing {processed_count}/{len(all_tickers_to_process)}")
        try:
            live_price_raw, current_invest_score_raw = await calculate_ema_invest(ticker_b, 2, is_called_by_ai=True)
            one_year_change_raw, _ = await calculate_one_year_invest(ticker_b, is_called_by_ai=True)
            current_invest_score = safe_score(current_invest_score_raw) if current_invest_score_raw is not None else None
            live_price = safe_score(live_price_raw) if live_price_raw is not None else None
            one_year_change = safe_score(one_year_change_raw) if one_year_change_raw is not None else None
            existing_entry = existing_tickers_data.get(ticker_b, {})
            highest_score_prev = safe_score(existing_entry.get("Highest Invest Score")) if pd.notna(existing_entry.get("Highest Invest Score")) else -float('inf')
            lowest_score_prev = safe_score(existing_entry.get("Lowest Invest Score")) if pd.notna(existing_entry.get("Lowest Invest Score")) else float('inf')
            current_highest_score = highest_score_prev
            current_lowest_score = lowest_score_prev
            if current_invest_score is not None:
                if current_highest_score == -float('inf') or current_invest_score > current_highest_score:
                    current_highest_score = current_invest_score
                if current_lowest_score == float('inf') or current_invest_score < current_lowest_score:
                    current_lowest_score = current_invest_score
            remove_ticker = False
            if current_invest_score is None:
                remove_ticker = True
            elif current_highest_score > -float('inf') and current_highest_score > 0:
                if (current_invest_score > 600 or
                    current_invest_score < invest_score_threshold or
                    current_invest_score < fraction_threshold * current_highest_score):
                    remove_ticker = True
            elif current_invest_score < invest_score_threshold:
                remove_ticker = True
            if not remove_ticker:
                status = "Repeat" if ticker_b in existing_tickers_data else "New"
                stock_data_dict = {
                    "Ticker": ticker_b,
                    "Live Price": f"{live_price:.2f}" if live_price is not None else "N/A",
                    "Invest Score": f"{current_invest_score:.2f}%" if current_invest_score is not None else "N/A",
                    "Highest Invest Score": f"{current_highest_score:.2f}%" if current_highest_score > -float('inf') else "N/A",
                    "Lowest Invest Score": f"{current_lowest_score:.2f}%" if current_lowest_score < float('inf') else "N/A",
                    "1Y% Change": f"{one_year_change:.2f}%" if one_year_change is not None else "N/A",
                    "Status": status,
                    "_sort_score_internal": current_invest_score if current_invest_score is not None else -float('inf')
                }
                temp_updated_data_for_df_build.append(stock_data_dict)
        except Exception as e_ticker_b:
            if not is_called_by_ai:
                print(f"Error processing breakout logic for ticker {ticker_b}: {e_ticker_b}")

    temp_updated_data_for_df_build.sort(key=lambda x: x['_sort_score_internal'], reverse=True)
    df_data_for_csv_and_return = []
    for item_dict in temp_updated_data_for_df_build:
        clean_item = {k: v for k, v in item_dict.items() if k != '_sort_score_internal'}
        df_data_for_csv_and_return.append(clean_item)

    final_tickers_set = {item['Ticker'] for item in df_data_for_csv_and_return}
    newly_added_tickers = [item for item in df_data_for_csv_and_return if item.get("Status") == "New"]
    removed_tickers = list(existing_tickers_set - final_tickers_set)

    if df_data_for_csv_and_return:
        final_df_to_save = pd.DataFrame(df_data_for_csv_and_return)
        try:
            final_df_to_save.to_csv(BREAKOUT_TICKERS_FILE, index=False)
            if not is_called_by_ai:
                print(f"Successfully saved current breakout data to '{BREAKOUT_TICKERS_FILE}'.")
        except IOError as e_io_b:
            if not is_called_by_ai:
                print(f"Error writing breakout data to '{BREAKOUT_TICKERS_FILE}': {e_io_b}")
    else:
        if not is_called_by_ai:
            print("No tickers met the breakout criteria to update the file or display.")
            try:
                open(BREAKOUT_TICKERS_FILE, 'w').close()
                print(f"'{BREAKOUT_TICKERS_FILE}' has been cleared as no tickers met criteria.")
            except Exception as e_clear:
                print(f"Could not clear '{BREAKOUT_TICKERS_FILE}': {e_clear}")

    if not is_called_by_ai:
        print("\n--- Breakout Analysis Results (Top 15 for CLI) ---")
        if df_data_for_csv_and_return:
            display_cols_cli = ["Ticker", "Live Price", "Invest Score", "Status"]
            actual_display_cols = [col for col in display_cols_cli if col in pd.DataFrame(df_data_for_csv_and_return).columns]
            if actual_display_cols:
                 print(tabulate(pd.DataFrame(df_data_for_csv_and_return)[actual_display_cols].head(15), headers="keys", tablefmt="pretty", showindex=False))
            else:
                print("No columns configured for breakout display, or DataFrame is missing them.")
        else:
            print("No breakout stocks found.")
        
        if newly_added_tickers:
            print(f"\nNewly Added Tickers: {', '.join([t['Ticker'] for t in newly_added_tickers])}")
        if removed_tickers:
            print(f"\nRemoved Tickers: {', '.join(removed_tickers)}")
        print("\n--- Breakout Analysis Complete ---")
        
    return {
        "current_breakout_stocks": df_data_for_csv_and_return,
        "newly_added_stocks": newly_added_tickers,
        "removed_stocks": removed_tickers
    }

async def handle_breakout_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """
    Handles the /breakout command for CLI and AI.
    For AI, expects ai_params={"action": "run" or "save", "date_to_save": "MM/DD/YYYY" (if saving)}.
    For CLI, /breakout runs analysis, /breakout 3725 prompts to save.
    """
    if not is_called_by_ai: print("\n--- /breakout Command ---")

    action_to_perform = None
    date_for_save = None

    if ai_params:
        action_to_perform = ai_params.get("action")
        if not action_to_perform or action_to_perform not in ["run", "save"]:
            return "Error for AI (/breakout): 'action' ('run' or 'save') is required."

        if action_to_perform == "save":
            date_for_save = ai_params.get("date_to_save")
            if not date_for_save:
                return "Error for AI (/breakout save): 'date_to_save' is required."
            try:
                datetime.strptime(date_for_save, '%m/%d/%Y')
            except ValueError:
                return f"AI Error: Invalid date format '{date_for_save}'. Use MM/DD/YYYY."
    else:
        if args and args[0] == "3725":
            action_to_perform = "save"
            date_for_save = input("CLI: Enter date (MM/DD/YYYY) to save breakout data: ")
            try:
                datetime.strptime(date_for_save, '%m/%d/%Y')
            except ValueError:
                print("CLI: Invalid date format. Save operation cancelled.")
                return None
        else:
            action_to_perform = "run"

    if action_to_perform == "run":
        breakout_results_dict = await run_breakout_analysis_singularity(is_called_by_ai=is_called_by_ai)
        if is_called_by_ai:
            current_stocks = breakout_results_dict.get("current_breakout_stocks", [])
            new_stocks = breakout_results_dict.get("newly_added_stocks", [])
            removed_stocks = breakout_results_dict.get("removed_stocks", [])

            if not current_stocks and not new_stocks and not removed_stocks:
                return {"summary": "Breakout analysis ran successfully but found no stocks matching the criteria and no changes from the previous list."}
            
            # This structured dictionary is much more useful for the AI.
            return {
                "summary": f"Breakout analysis complete. Found {len(current_stocks)} current stocks, {len(new_stocks)} new additions, and {len(removed_stocks)} removals.",
                "current_stocks": current_stocks[:10], # Return top 10 for brevity
                "newly_added_tickers": [t['Ticker'] for t in new_stocks],
                "removed_tickers": removed_stocks
            }
        else:
            return None

    elif action_to_perform == "save":
        if date_for_save:
            return await save_breakout_data_singularity(date_for_save, is_called_by_ai=is_called_by_ai)
        else:
            return "Error: Save action was chosen, but no date was provided."

    else:
        msg = f"Unknown or unhandled action '{action_to_perform}' for breakout command."
        if not is_called_by_ai:
            print(msg)
        return f"Error: {msg}" if is_called_by_ai else None

breakout_command_tool = FunctionDeclaration(
    name="handle_breakout_command",
    description="Handles breakout stock analysis. Action 'run' performs a new analysis, compares it to the previous run, and returns a detailed breakdown of current, newly added, and removed stocks. Action 'save' archives the most recent run's data for a specified date.",
    parameters={"type": "object", "properties": {
        "action": {"type": "string", "description": "Choose 'run' to find new breakout stocks and see changes, or 'save' to archive the last run's data.", "enum": ["run", "save"]},
        "date_to_save": {"type": "string", "description": "Required for 'save' action: Date in MM/DD/YYYY format. If user says 'today', use the current date."}
    }, "required": ["action"]}
)

async def save_breakout_data_singularity(date_str: str, is_called_by_ai: bool = False) -> str:
    """
    Saves the current breakout data from BREAKOUT_TICKERS_FILE to
    BREAKOUT_HISTORICAL_DB_FILE for a given date.
    Returns a summary string.
    """
    if not is_called_by_ai:
        print(f"\n--- Saving Breakout Data for Date: {date_str} ---")

    if not os.path.exists(BREAKOUT_TICKERS_FILE):
        msg = f"Error: Current breakout data file '{BREAKOUT_TICKERS_FILE}' not found. Cannot save historical data."
        if not is_called_by_ai:
            print(msg)
        return msg

    save_count = 0
    try:
        df_current_breakout = pd.read_csv(BREAKOUT_TICKERS_FILE)
        if df_current_breakout.empty:
            msg = f"Info: Current breakout file '{BREAKOUT_TICKERS_FILE}' is empty. Nothing to save to historical DB."
            if not is_called_by_ai:
                print(msg)
            return msg

        historical_data_to_save = []
        for _, row in df_current_breakout.iterrows():
            # Ensure data is clean before saving
            price_str = str(row.get('Live Price', 'N/A')).replace('$', '').strip()
            score_str = str(row.get('Invest Score', 'N/A')).replace('%', '').strip()

            # Use safe_score to handle potential "N/A" or errors during conversion
            price_val = safe_score(price_str)
            score_val = safe_score(score_str) # Assuming Invest Score can be numeric after stripping '%'

            historical_data_to_save.append({
                'DATE': date_str,
                'TICKER': row.get('Ticker', 'ERR'),
                'PRICE': f"{price_val:.2f}" if price_val is not None and not pd.isna(price_val) else "N/A",
                'INVEST_SCORE': f"{score_val:.2f}" if score_val is not None and not pd.isna(score_val) else "N/A" # Ensure consistent formatting
            })

        if not historical_data_to_save:
            msg = "No valid breakout data rows to save after processing."
            if not is_called_by_ai: print(msg)
            return msg

        file_exists_hist = os.path.isfile(BREAKOUT_HISTORICAL_DB_FILE)
        headers_hist = ['DATE', 'TICKER', 'PRICE', 'INVEST_SCORE']

        with open(BREAKOUT_HISTORICAL_DB_FILE, 'a', newline='', encoding='utf-8') as f_hist:
            writer_hist = csv.DictWriter(f_hist, fieldnames=headers_hist)
            if not file_exists_hist or os.path.getsize(f_hist.name) == 0:
                writer_hist.writeheader()
            for data_row_hist in historical_data_to_save:
                writer_hist.writerow(data_row_hist)
                save_count += 1
        msg = f"Successfully saved {save_count} breakout records to '{BREAKOUT_HISTORICAL_DB_FILE}' for date {date_str}."
        if not is_called_by_ai:
            print(msg)
        return msg

    except pd.errors.EmptyDataError:
        msg = f"Warning: Breakout source file '{BREAKOUT_TICKERS_FILE}' is empty (pandas error). Nothing saved."
        if not is_called_by_ai: print(msg)
        return msg
    except KeyError as e_key:
        msg = f"Warning: Missing expected column in '{BREAKOUT_TICKERS_FILE}': {e_key}. Cannot save historical data."
        if not is_called_by_ai: print(msg)
        return msg
    except IOError as e_io_hist:
        msg = f"Error writing to historical breakout save file '{BREAKOUT_HISTORICAL_DB_FILE}': {e_io_hist}"
        if not is_called_by_ai: print(msg)
        return msg
    except Exception as e_save_hist:
        msg = f"An unexpected error occurred processing/saving historical breakout data: {e_save_hist}"
        if not is_called_by_ai:
            print(msg)
            traceback.print_exc()
        return msg
    
# --- Market Command Functions --- (Assumed mostly unchanged, add is_called_by_ai flags)
def get_sp500_symbols_singularity(is_called_by_ai: bool = False) -> List[str]:
    """Fetches S&P 500 symbols from Wikipedia for Singularity use."""
    try:
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        dfs = pd.read_html(sp500_url) # This is a blocking call
        if not dfs:
            if not is_called_by_ai: print("Error: Could not parse S&P 500 table from Wikipedia.")
            return []
        sp500_df = dfs[0] # First table is usually the components
        if 'Symbol' not in sp500_df.columns:
            if not is_called_by_ai: print("Error: 'Symbol' column not found in S&P 500 data.")
            return []

        symbols = [str(s).replace('.', '-') for s in sp500_df['Symbol'].tolist() if isinstance(s, str)]
        symbols = sorted(list(set(s for s in symbols if s))) # Unique, non-empty, sorted
        if not is_called_by_ai:
            print(f"Fetched {len(symbols)} S&P 500 symbols.")
        return symbols
    except Exception as e:
        if not is_called_by_ai:
            print(f"Error fetching S&P 500 symbols: {e}")
            # traceback.print_exc() # Optional: print traceback for CLI debugging
        return []

async def calculate_market_invest_scores_singularity(tickers: List[str], ema_sens: int, is_called_by_ai: bool = False) -> List[Dict[str, Any]]:
    """
    Calculates EMA Invest scores for a list of tickers with a given sensitivity.
    Returns a list of dictionaries: {'ticker': str, 'live_price': Optional[float], 'score': Optional[float]}.
    Prints progress if not called by AI.
    """
    result_data_market = []
    total_tickers = len(tickers)
    if not is_called_by_ai:
        print(f"\nCalculating Invest scores for {total_tickers} market tickers (Sensitivity: {ema_sens})...")

    chunk_size = 25 # Process tickers in chunks
    processed_count_market = 0

    for i in range(0, total_tickers, chunk_size):
        chunk = tickers[i:i + chunk_size]
        # calculate_ema_invest is async, gather results for the chunk
        tasks = [calculate_ema_invest(ticker, ema_sens, is_called_by_ai=True) for ticker in chunk] # True to suppress its prints
        results_chunk = await asyncio.gather(*tasks, return_exceptions=True)

        for idx, res_item in enumerate(results_chunk):
            ticker_processed = chunk[idx]
            if isinstance(res_item, Exception):
                # For AI calls, detailed errors might be too verbose in the summary.
                # The AI will get a summary of how many succeeded/failed.
                # if not is_called_by_ai:
                # print(f"  Error processing {ticker_processed} for market score: {res_item}")
                result_data_market.append({'ticker': ticker_processed, 'live_price': None, 'score': None, 'error': str(res_item)})
            elif res_item is not None:
                live_price_market, ema_invest_score_market = res_item
                result_data_market.append({
                    'ticker': ticker_processed,
                    'live_price': live_price_market,
                    'score': ema_invest_score_market # Store raw score (can be None or float)
                })
            else: # Should ideally not happen if calculate_ema_invest always returns tuple
                 result_data_market.append({'ticker': ticker_processed, 'live_price': None, 'score': None, 'error': 'Unknown error from calculate_ema_invest'})

            processed_count_market += 1
            if not is_called_by_ai and (processed_count_market % 50 == 0 or processed_count_market == total_tickers):
                print(f"  ...market scores calculated for {processed_count_market}/{total_tickers} tickers.")

    # Sort by score (descending), handling None scores
    result_data_market.sort(key=lambda x: safe_score(x.get('score', -float('inf'))), reverse=True)
    if not is_called_by_ai:
        print("Finished calculating all market scores.")
    return result_data_market

async def save_market_data_singularity(sensitivity: int, date_str: str, is_called_by_ai: bool = False):
    """
    Saves full market data (Ticker, Price, Score) for S&P500 for a given sensitivity and date.
    """
    if not is_called_by_ai:
        print(f"\n--- Saving Full S&P500 Market Data (Sensitivity: {sensitivity}) for Date: {date_str} ---")

    # Run get_sp500_symbols_singularity in a thread as it's blocking (pd.read_html)
    # Pass is_called_by_ai=True to suppress its prints
    sp500_symbols = await asyncio.to_thread(get_sp500_symbols_singularity, is_called_by_ai=True)
    if not sp500_symbols:
        if not is_called_by_ai: print("Error: Could not retrieve S&P 500 symbols. Cannot save market data.")
        return # Or return an error status/message

    # calculate_market_invest_scores_singularity is async
    # Pass is_called_by_ai=True to suppress its prints
    all_scores_data = await calculate_market_invest_scores_singularity(sp500_symbols, sensitivity, is_called_by_ai=True)

    if not all_scores_data:
        if not is_called_by_ai: print(f"Error: No valid market data calculated for Sensitivity {sensitivity}. Nothing saved.")
        return

    data_to_save = []
    for item in all_scores_data:
        if item.get('score') is not None: # Only save if score calculation was successful
            price_val = safe_score(item.get('live_price'))
            score_val = safe_score(item.get('score'))
            data_to_save.append({
                'DATE': date_str,
                'TICKER': item.get('ticker', 'ERR'),
                'PRICE': f"{price_val:.2f}" if price_val is not None and not pd.isna(price_val) else "N/A",
                'SCORE': f"{score_val:.2f}" if score_val is not None and not pd.isna(score_val) else "N/A" # Save formatted score
            })

    if not data_to_save:
        if not is_called_by_ai: print(f"No S&P500 tickers with valid scores found for Sensitivity {sensitivity}. Nothing saved.")
        return

    save_filename = f"{MARKET_FULL_SENS_DATA_FILE_PREFIX}{sensitivity}_data.csv"
    file_exists = os.path.isfile(save_filename)
    headers_save = ['DATE', 'TICKER', 'PRICE', 'SCORE']

    try:
        with open(save_filename, 'a', newline='', encoding='utf-8') as f_market:
            writer_market = csv.DictWriter(f_market, fieldnames=headers_save)
            if not file_exists or os.path.getsize(f_market.name) == 0:
                writer_market.writeheader()
            writer_market.writerows(data_to_save) # Use writerows for list of dicts
        if not is_called_by_ai:
            print(f"Successfully saved {len(data_to_save)} S&P500 records to '{save_filename}'.")
    except IOError as e_io_market:
        if not is_called_by_ai: print(f"Error writing market data to '{save_filename}': {e_io_market}")
    except Exception as e_save_mkt:
        if not is_called_by_ai:
            print(f"Unexpected error saving market data: {e_save_mkt}")
            traceback.print_exc()


async def handle_market_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """
    Handles the /market command for S&P500 overview or data saving.
    """
    if not is_called_by_ai:
        print("\n--- /market Command (S&P 500) ---")

    action, sensitivity_for_action, date_for_save = None, None, None

    if ai_params: # AI Call
        action_ai = ai_params.get("action", "").lower()
        sensitivity_ai_raw = ai_params.get("sensitivity")
        date_ai_raw = ai_params.get("date_str")

        if not action_ai or sensitivity_ai_raw is None:
            return "Error for AI (/market): 'action' and 'sensitivity' are required."
        try:
            sensitivity_for_action = int(float(sensitivity_ai_raw)) # Robust conversion
            if sensitivity_for_action not in [1, 2, 3]:
                return f"Error for AI (/market): Sensitivity '{sensitivity_for_action}' out of range [1,2,3]."
            action = action_ai
            if action == "save":
                if not date_ai_raw:
                    return "Error for AI (/market save): 'date_str' (MM/DD/YYYY) is required."
                datetime.strptime(date_ai_raw, '%m/%d/%Y') # Validate
                date_for_save = date_ai_raw
        except ValueError:
            return f"Error for AI (/market): Invalid sensitivity '{sensitivity_ai_raw}' or date '{date_ai_raw}'. Sens:1,2,3. Date:MM/DD/YYYY."
    else: # CLI Path
        if not args: # Interactive display if no args
            action = "display_interactive"
        elif args[0] == "3725": # CLI save trigger
            action = "save_interactive"
        else: # Assume CLI display with provided sensitivity if format matches
            if len(args) >= 1:
                try:
                    sens_cli = int(args[0])
                    if sens_cli in [1,2,3]:
                        action = "display" # Direct display with sensitivity
                        sensitivity_for_action = sens_cli
                    else: action = "display_interactive" # Invalid sens, prompt
                except ValueError: action = "display_interactive" # Not a number, prompt
            else: action = "display_interactive"


    # --- Perform Action ---
    summary_for_ai_parts = []

    if action == "save" or action == "save_interactive":
        try:
            if action == "save_interactive": # Prompt for CLI
                sens_str_cli = input("Enter S&P500 Market Sensitivity (1, 2, or 3) to save: ")
                sensitivity_for_action = int(sens_str_cli)
                if sensitivity_for_action not in [1, 2, 3]:
                    print("Invalid sensitivity. Must be 1, 2, or 3."); return None if not is_called_by_ai else "Invalid sensitivity."
                date_str_cli = input(f"Enter date (MM/DD/YYYY) to save full S&P500 market data for Sensitivity {sensitivity_for_action}: ")
                datetime.strptime(date_str_cli, '%m/%d/%Y')
                date_for_save = date_str_cli

            await save_market_data_singularity(sensitivity_for_action, date_for_save, is_called_by_ai=is_called_by_ai)
            summary_for_ai_parts.append(f"S&P500 market data for sensitivity {sensitivity_for_action} save process initiated for date {date_for_save}.")
            if not is_called_by_ai: print(summary_for_ai_parts[0]) # Also print for CLI if initiated this way

        except ValueError as ve_cli_save:
            msg = f"Invalid input: {ve_cli_save}. Use numbers for sensitivity and MM/DD/YYYY for date."
            if not is_called_by_ai: print(msg)
            return msg if is_called_by_ai else None
        except Exception as e_save_op:
            msg = f"An error occurred during S&P500 market data save: {e_save_op}"
            if not is_called_by_ai: print(msg); traceback.print_exc()
            return msg if is_called_by_ai else None

    elif action == "display" or action == "display_interactive":
        try:
            if action == "display_interactive": # Prompt for CLI
                sens_str_disp_cli = input("Enter S&P500 Market Sensitivity (1:Weekly, 2:Daily, 3:Hourly) to display scores: ")
                sensitivity_for_action = int(sens_str_disp_cli)
                if sensitivity_for_action not in [1, 2, 3]:
                    print("Invalid sensitivity."); return None if not is_called_by_ai else "Invalid sensitivity."

            # Run get_sp500_symbols_singularity in a thread
            sp500_symbols_disp = await asyncio.to_thread(get_sp500_symbols_singularity, is_called_by_ai=is_called_by_ai)
            if not sp500_symbols_disp:
                msg = "Error: Could not retrieve S&P 500 symbols for display."
                if not is_called_by_ai: print(msg)
                return msg if is_called_by_ai else None

            if not is_called_by_ai:
                print(f"\nCalculating S&P 500 market scores (Sensitivity: {sensitivity_for_action}). This may take some time...")
            all_scores_data_disp = await calculate_market_invest_scores_singularity(sp500_symbols_disp, sensitivity_for_action, is_called_by_ai=is_called_by_ai)

            if not all_scores_data_disp:
                msg = "Error calculating S&P500 market scores or no data returned."
                if not is_called_by_ai: print(msg)
                return msg if is_called_by_ai else None

            valid_scores_market = [item for item in all_scores_data_disp if item.get('score') is not None]
            if not valid_scores_market:
                msg = "No valid S&P500 scores could be calculated for display."
                if not is_called_by_ai: print(msg)
                return msg if is_called_by_ai else None

            top_10 = valid_scores_market[:10]
            bottom_10_raw = valid_scores_market[-10:]
            bottom_10 = sorted(bottom_10_raw, key=lambda x: safe_score(x.get('score', float('inf'))))
            spy_score_item = next((item for item in all_scores_data_disp if item['ticker'] == 'SPY'), None)

            def format_row(item_dict: Dict[str, Any]) -> List[str]:
                price = safe_score(item_dict.get('live_price'))
                score = safe_score(item_dict.get('score'))
                return [
                    item_dict.get('ticker', 'ERR'),
                    f"${price:.2f}" if price is not None and not pd.isna(price) else "N/A",
                    f"{score:.2f}%" if score is not None and not pd.isna(score) else "N/A"
                ]

            if not is_called_by_ai: # Print tables for CLI
                print(f"\n**Top 10 S&P 500 Stocks (Sensitivity: {sensitivity_for_action})**")
                if top_10: print(tabulate([format_row(r) for r in top_10], headers=["Ticker", "Price", "Score"], tablefmt="pretty"))
                else: print("No top scores data.")
                print(f"\n**Bottom 10 S&P 500 Stocks (Sensitivity: {sensitivity_for_action})**")
                if bottom_10: print(tabulate([format_row(r) for r in bottom_10], headers=["Ticker", "Price", "Score"], tablefmt="pretty"))
                else: print("No bottom scores data.")
                print(f"\n**SPY Score (Sensitivity: {sensitivity_for_action})**")
                if spy_score_item and spy_score_item.get('score') is not None:
                    print(tabulate([format_row(spy_score_item)], headers=["Ticker", "Price", "Score"], tablefmt="pretty"))
                else: print("SPY score not available or could not be calculated.")
                print("\n--- S&P500 Market Display Complete ---")

            summary_for_ai_parts.append(f"S&P 500 market analysis scores (Sensitivity: {sensitivity_for_action}) generated.")
            if top_10: summary_for_ai_parts.append("Top tickers include: " + ", ".join([f"{r['ticker']}({safe_score(r.get('score')):.1f}%)" for r in top_10[:3]]))
            if spy_score_item and spy_score_item.get('score') is not None: summary_for_ai_parts.append(f"SPY score: {safe_score(spy_score_item.get('score')):.2f}%.")

        except ValueError as ve_cli_disp: # Handles int conversion for CLI sensitivity
            msg = f"Invalid input for sensitivity: {ve_cli_disp}. Must be a number (1, 2, or 3)."
            if not is_called_by_ai: print(msg)
            return msg if is_called_by_ai else None
        except Exception as e_disp_market:
            msg = f"An error occurred during S&P500 market display: {e_disp_market}"
            if not is_called_by_ai: print(msg); traceback.print_exc()
            return msg if is_called_by_ai else None
    else:
        msg = "Market command action not recognized or insufficient arguments."
        if not is_called_by_ai: print(msg) # Potentially show help or usage here for CLI
        return msg if is_called_by_ai else None

    return " ".join(summary_for_ai_parts) if is_called_by_ai else None

# --- NEW /briefing Command Handler ---
async def handle_briefing_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """
    Generates a daily market briefing by running multiple analyses.
    """
    if not is_called_by_ai:
        print("\n--- Generating Daily Market Briefing ---")

    # Step 1: Check for and handle the user's favorites/watchlist (this is fast)
    watchlist_tickers = []
    watchlist_message = None  # To hold messages for the final report
    if not os.path.exists(USERS_FAVORITES_FILE):
        if is_called_by_ai:
            # For AI, don't fail. Just note that the watchlist is missing.
            watchlist_message = f"Watchlist section skipped: '{USERS_FAVORITES_FILE}' not found. The user can create this by running the /briefing command from the terminal."
        else: # CLI path remains the same
            print(f"Watchlist file '{USERS_FAVORITES_FILE}' not found.")
            user_favorites_input = input("Please enter your comma-separated watchlist tickers (e.g., AAPL,MSFT,NVDA): ").strip().upper()
            if user_favorites_input:
                watchlist_tickers = [t.strip() for t in user_favorites_input.split(',') if t.strip()]
                try:
                    with open(USERS_FAVORITES_FILE, 'w', encoding='utf-8') as f:
                        f.write(user_favorites_input)
                    print(f"Watchlist saved to '{USERS_FAVORITES_FILE}'.")
                except IOError as e:
                    print(f"Error saving watchlist: {e}")
            else:
                print("No watchlist provided. Skipping favorites section in briefing.")
    else:
        try:
            with open(USERS_FAVORITES_FILE, 'r', encoding='utf-8') as f:
                content = f.read().strip().upper()
                watchlist_tickers = [t.strip() for t in content.split(',') if t.strip()]
        except IOError as e:
            print(f"Error reading watchlist file: {e}")
            # Also set a message if reading an existing file fails
            watchlist_message = "Watchlist section skipped: Could not read the favorites file due to an error."

    # --- Start of Sequential Progress Reporting ---
    results_dict = {}

    # Task Group 1 (RISK, SPY/VIX) - Relatively fast with optimization
    if not is_called_by_ai:
        print("➪ Briefing Step 1/4: Calculating R.I.S.K. Scores and Market Indexes...")
    
    tasks1 = {
        "spy_vix": get_daily_change_for_tickers(['SPY', '^VIX']),
        "risk": perform_risk_calculations_singularity(is_called_by_ai=True),
    }
    results1 = await asyncio.gather(*tasks1.values(), return_exceptions=True)
    results_dict.update(dict(zip(tasks1.keys(), results1)))

    # Task Group 2 (S&P 500 Movers) - Can be slow
    if not is_called_by_ai:
        print("➪ Briefing Step 2/4: Analyzing S&P 500 Movers (this may take a moment)...")
    
    tasks2 = {"sp500_movers": get_sp500_movers(is_called_by_ai=True)}
    results2 = await asyncio.gather(*tasks2.values(), return_exceptions=True)
    results_dict.update(dict(zip(tasks2.keys(), results2)))

    # Task Group 3 (Breakouts & Watchlist) - Can be slow
    if not is_called_by_ai:
        print("➪ Briefing Step 3/4: Running Breakout Analysis and Checking Watchlist...")

    tasks3 = {"breakouts": run_breakout_analysis_singularity(is_called_by_ai=True)}
    if watchlist_tickers:
        tasks3["watchlist"] = get_daily_change_for_tickers(watchlist_tickers)
    
    results3 = await asyncio.gather(*tasks3.values(), return_exceptions=True)
    results_dict.update(dict(zip(tasks3.keys(), results3)))
    
    # Step 4: Final Processing (fast)
    if not is_called_by_ai:
        print("➪ Briefing Step 4/4: Compiling Final Report...")

    # Check for major errors in gathered data
    for key, result in results_dict.items():
        if isinstance(result, Exception):
            err_msg = f"Error gathering data for '{key}': {result}"
            if not is_called_by_ai: print(err_msg)
            return {"status": "error", "message": err_msg}

    # Process and format the results
    briefing_data = {}
    
    # Intro
    spy_data = results_dict.get('spy_vix', {}).get('SPY', {})
    vix_data = results_dict.get('spy_vix', {}).get('^VIX', {})
    briefing_data['intro'] = {
        'spy_price': spy_data.get('live_price'),
        'spy_change': spy_data.get('change_pct'),
        'vix_price': vix_data.get('live_price'),
        'vix_change': vix_data.get('change_pct'),
    }

    # RISK Scores
    risk_results = results_dict.get('risk', {})
    briefing_data['risk'] = {
        'general_score': risk_results.get('general_score'),
        'market_invest_score': risk_results.get('market_invest_score'),
    }

    # S&P 500 Movers
    briefing_data['sp500_movers'] = results_dict.get('sp500_movers', {'top': [], 'bottom': []})

    # Breakouts
    breakout_results = results_dict.get('breakouts', {})
    top_3_breakout_tickers = [item['Ticker'] for item in breakout_results.get('current_breakout_stocks', [])[:3]]
    top_3_perf_data = await get_multi_period_change(top_3_breakout_tickers)
    
    briefing_data['breakouts'] = {
        'newly_added': [item['Ticker'] for item in breakout_results.get('newly_added_stocks', [])],
        'removed': breakout_results.get('removed_stocks', []),
        'top_3': breakout_results.get('current_breakout_stocks', [])[:3],
        'top_3_performance': top_3_perf_data
    }

    # Watchlist Movers
    if 'watchlist' in results_dict and results_dict.get('watchlist'):
        watchlist_changes = results_dict['watchlist']
        valid_watchlist = [{'ticker': t, **d} for t, d in watchlist_changes.items() if 'change_pct' in d and pd.notna(d['change_pct'])]
        if valid_watchlist:
            valid_watchlist.sort(key=lambda x: x['change_pct'], reverse=True)
            briefing_data['watchlist'] = {
                'top': valid_watchlist[:3],
                'bottom': list(reversed(valid_watchlist[-3:]))
            }
    elif watchlist_message:
        # If the watchlist wasn't processed but we have a message, add it.
        briefing_data['watchlist'] = {
            'error_message': watchlist_message
        }

    # Output the briefing
    if is_called_by_ai:
        return {"status": "success", "data": briefing_data}

    # CLI Output Formatting
    print("\n" + "="*50)
    print("DAILY MARKET BRIEFING")
    print(f"{datetime.now(EST_TIMEZONE).strftime('%B %d, %Y - %I:%M %p EST')}")
    print("="*50)

    # Intro
    intro = briefing_data['intro']
    spy_p, spy_c = (f"${intro['spy_price']:.2f}", f"{intro['spy_change']:.2f}%") if intro.get('spy_price') is not None else ("N/A", "N/A")
    vix_p, vix_c = (f"{intro['vix_price']:.2f}", f"{intro['vix_change']:.2f}%") if intro.get('vix_price') is not None else ("N/A", "N/A")
    print(f"SPY: {spy_p} ({spy_c}) | VIX: {vix_p} ({vix_c})")

    # RISK
    risk_info = briefing_data['risk']
    print(f"RISK Scores -> General: {risk_info.get('general_score', 'N/A')} | Market Invest: {risk_info.get('market_invest_score', 'N/A')}")
    
    # S&P 500 Movers
    print("\n--- S&P 500 Movers ---")
    sp_movers = briefing_data['sp500_movers']
    top_sp = ", ".join([f"{t['ticker']} ({t['change_pct']:.2f}%)" for t in sp_movers.get('top', [])])
    bottom_sp = ", ".join([f"{t['ticker']} ({t['change_pct']:.2f}%)" for t in sp_movers.get('bottom', [])])
    print(f"  Top 3: {top_sp if top_sp else 'N/A'}")
    print(f"  Bottom 3: {bottom_sp if bottom_sp else 'N/A'}")

    # Breakouts
    print("\n--- Breakout Analysis ---")
    b_info = briefing_data['breakouts']
    print(f"  New: {', '.join(b_info['newly_added']) if b_info['newly_added'] else 'None'}")
    print(f"  Removed: {', '.join(b_info['removed']) if b_info['removed'] else 'None'}")
    print("  Top 3 by Invest Score:")
    if b_info['top_3']:
        for stock in b_info['top_3']:
            ticker = stock['Ticker']
            perf = b_info['top_3_performance'].get(ticker, {})
            score = stock.get('Invest Score', 'N/A')
            d1 = f"{perf.get('1D'):.2f}%" if perf.get('1D') is not None else 'N/A'
            w1 = f"{perf.get('1W'):.2f}%" if perf.get('1W') is not None else 'N/A'
            m1 = f"{perf.get('1M'):.2f}%" if perf.get('1M') is not None else 'N/A'
            print(f"    - {ticker} (Score: {score}) | 1D: {d1}, 1W: {w1}, 1M: {m1}")
    else:
        print("No current breakout stocks.")

    # Watchlist
    if 'watchlist' in briefing_data:
        w_movers = briefing_data['watchlist']
        # Check if we have movers or an error message to display
        if 'error_message' in w_movers:
             print(f"\n--- Your Watchlist Movers ---\n  {w_movers['error_message']}")
        else:
            print("\n--- Your Watchlist Movers ---")
            top_w = ", ".join([f"{t['ticker']} ({t['change_pct']:.2f}%)" for t in w_movers.get('top', [])])
            bottom_w = ", ".join([f"{t['ticker']} ({t['change_pct']:.2f}%)" for t in w_movers.get('bottom', [])])
            print(f"  Top 3: {top_w if top_w else 'N/A'}")
            print(f"  Bottom 3: {bottom_w if bottom_w else 'N/A'}")
    
    print("\n" + "="*50)
    return None # For CLI, direct printing is the output
    
# --- Cultivate Command Functions --- (Add is_called_by_ai flags)
def get_yf_data_singularity(tickers: List[str], period: str = "10y", interval: str = "1d", is_called_by_ai: bool = False) -> pd.DataFrame:
    """
    Downloads historical closing price data for multiple tickers using yfinance.
    Optimized to build DataFrame from a list of Series to avoid fragmentation.
    """
    if not tickers:
        return pd.DataFrame()

    tickers_list = list(set(tickers)) # Ensure unique tickers

    # This function is blocking (yf.download). For a fully async app, consider asyncio.to_thread
    # However, since many yfinance calls are made, keeping it sync here might be acceptable
    # depending on overall script structure. If this is called from an async path many times
    # for single tickers, an async version of this fetching would be better.
    # For now, keeping it synchronous as per original implied structure for this part.

    try:
        if not is_called_by_ai and len(tickers_list) > 5: # Basic progress for long CLI calls
             print(f"      Fetching yfinance data for {len(tickers_list)} tickers (period: {period}, interval: {interval})...")

        data = yf.download(tickers_list, period=period, interval=interval, progress=False, auto_adjust=False, group_by='ticker', timeout=30)

        if data.empty:
            # if not is_called_by_ai: print(f"    get_yf_data_singularity: yfinance.download returned EMPTY DataFrame for: {tickers_list}")
            return pd.DataFrame()

        all_series = []
        if isinstance(data.columns, pd.MultiIndex):
            for ticker_name in tickers_list:
                close_series = None
                if (ticker_name, 'Close') in data.columns: close_series = data[(ticker_name, 'Close')]
                elif ('Close', ticker_name) in data.columns: close_series = data[('Close', ticker_name)]

                if close_series is not None and not close_series.empty and not close_series.isnull().all():
                    series_numeric = pd.to_numeric(close_series, errors='coerce')
                    if not series_numeric.isnull().all(): # Check again after numeric conversion
                        series_numeric.name = ticker_name
                        all_series.append(series_numeric)
        elif len(tickers_list) == 1 and 'Close' in data.columns: # Single ticker
            ticker_name = tickers_list[0]
            if not data['Close'].isnull().all():
                series_numeric = pd.to_numeric(data['Close'], errors='coerce')
                if not series_numeric.isnull().all():
                    series_numeric.name = ticker_name
                    all_series.append(series_numeric)
        # Add other fallbacks if needed for different yf.download structures

        if not all_series:
            # if not is_called_by_ai: print(f"    get_yf_data_singularity: No valid data series collected for: {tickers_list}")
            return pd.DataFrame()

        df_out = pd.concat(all_series, axis=1)
        if df_out.empty: return pd.DataFrame()

        df_out.index = pd.to_datetime(df_out.index)
        df_out = df_out.dropna(axis=0, how='all').dropna(axis=1, how='all')
        return df_out

    except Exception as e:
        # if not is_called_by_ai: # Be less verbose for AI calls for common errors
            # if "Failed to get ticker" in str(e) or "No data found" in str(e) or "DNSError" in str(e):
                # print(f"    Warning in get_yf_data_singularity for {tickers_list}: {type(e).__name__}")
            # else:
                # print(f"    Error in get_yf_data_singularity for {tickers_list}: {type(e).__name__} - {e}")
                # traceback.print_exc()
        return pd.DataFrame()


def screen_stocks_singularity(is_called_by_ai: bool = False) -> List[str]:
    """ Screens for stocks using TradingView. Singularity version."""
    if not is_called_by_ai:
        print("    Starting Step: Stock Screening (TradingView - Cultivate Code A)...")
    try:
        # Ensure Query and Column are imported from tradingview_screener
        query = Query().select(
            'name',
            'market_cap_basic',
            'average_volume_90d_calc',
        ).where(
            Column('market_cap_basic') >= 50_000_000_000,  # 50B Market Cap
            Column('average_volume_90d_calc') >= 1_000_000 # 1M Avg Volume
        ).limit(500) # Limit to a reasonable number for further processing

        # if not is_called_by_ai: print("      Executing screener query...")
        # get_scanner_data is blocking, should be run in a thread if called from async often
        scanner_results_tuple = query.get_scanner_data(timeout=60)
        # if not is_called_by_ai: print("      Screener query finished.")

        if scanner_results_tuple and isinstance(scanner_results_tuple, tuple) and len(scanner_results_tuple) > 0 and isinstance(scanner_results_tuple[1], pd.DataFrame):
            df = scanner_results_tuple[1]
            if not df.empty and 'name' in df.columns:
                tickers = [str(t).split(':')[-1].replace('.', '-') for t in df['name'].tolist() if pd.notna(t)]
                cleaned_tickers = sorted(list(set(tickers))) # Unique and sorted
                if not is_called_by_ai:
                    print(f"    Screening complete. Found {len(cleaned_tickers)} potential tickers for Cultivate Code A.")
                return cleaned_tickers
            else:
                # if not is_called_by_ai: print("    Warning: 'name' column not found or empty in screening results.")
                return []
        else:
            # if not is_called_by_ai: print("    Warning: Stock screener returned no data or unexpected format.")
            return []
    except Exception as e:
        # if not is_called_by_ai:
        #     print(f"    Error during stock screening: {type(e).__name__} - {e}")
        #     if "Max retries exceeded" in str(e) or "ConnectTimeoutError" in str(e) or "Failed to resolve" in str(e):
        #         print("    This is likely a network issue or TradingView service problem.")
            # else:
            #     traceback.print_exc()
        return []


def calculate_metrics_singularity(tickers_list: List[str], spy_data_10y: pd.DataFrame, is_called_by_ai: bool = False) -> Dict[str, Dict[str, float]]:
    """ Calculates 1y Beta, 1y Correlation, and 1y Avg Leverages relative to SPY."""
    if not is_called_by_ai:
        print(f"    Starting Step: Calculating Metrics (Beta/Corr/Leverage) for {len(tickers_list)} tickers...")
    metrics = {}

    if spy_data_10y.empty or 'SPY' not in spy_data_10y.columns:
        # if not is_called_by_ai: print("    Error: Valid 10y SPY historical data ('SPY' column) is required for metrics calculation.")
        return {}
    try:
        spy_data_10y.index = pd.to_datetime(spy_data_10y.index)
        # SPY daily returns from the pre-fetched 10y data
        spy_daily_returns_full_10y = spy_data_10y['SPY'].pct_change().dropna()
        if spy_daily_returns_full_10y.empty:
            # if not is_called_by_ai: print("    Error: SPY daily returns from 10y data are all NaN after calculation.")
            return {}
    except Exception as e:
        # if not is_called_by_ai: print(f"    Error preparing SPY daily returns for metrics: {e}")
        return {}

    # Calculate historical SPY Invest Score (Daily, using the 10y SPY data)
    spy_invest_scores_hist_10y = None
    try:
        spy_close_series_10y = pd.to_numeric(spy_data_10y['SPY'], errors='coerce').dropna()
        if len(spy_close_series_10y) >= 55: # Min period for EMA 55
            spy_ema_8_hist = spy_close_series_10y.ewm(span=8, adjust=False).mean()
            spy_ema_55_hist = spy_close_series_10y.ewm(span=55, adjust=False).mean()
            spy_invest_score_series_hist = pd.Series(np.nan, index=spy_close_series_10y.index) # Initialize with NaNs
            # Calculate score only where EMA55 is valid and not zero
            valid_indices_hist = spy_ema_55_hist.index[(spy_ema_55_hist.notna()) & (spy_ema_55_hist != 0)]
            if not valid_indices_hist.empty:
                 ema_enter_hist = (spy_ema_8_hist.loc[valid_indices_hist] - spy_ema_55_hist.loc[valid_indices_hist]) / spy_ema_55_hist.loc[valid_indices_hist]
                 spy_invest_score_series_hist.loc[valid_indices_hist] = ((ema_enter_hist * 4) + 0.5) * 100 # Original Cultivate EMA Invest score formula
            spy_invest_scores_hist_10y = spy_invest_score_series_hist.dropna()
        # else: # if not is_called_by_ai: print(f"      Warning: Insufficient SPY data ({len(spy_close_series_10y)} pts) for historical Invest Score.")
    except Exception as e_spy_score:
        # if not is_called_by_ai: print(f"      Error calculating historical SPY Invest Score: {e_spy_score}")
        pass # Continue without historical SPY scores if calculation fails

    # Fetch 10y history for all tickers involved in metrics calculation (including SPY again to ensure alignment if needed, though we have it)
    # This might be redundant if all_tickers_data_metrics already fetched everything.
    # For robustness, let's use a combined fetch for the specific period needed for metrics.
    all_tickers_for_metrics_fetch = list(set(tickers_list + ['SPY'])) # Ensure SPY is there
    all_tickers_hist_data_metrics = get_yf_data_singularity(all_tickers_for_metrics_fetch, period="10y", interval="1d", is_called_by_ai=is_called_by_ai)

    if all_tickers_hist_data_metrics.empty:
        # if not is_called_by_ai: print("    Error: Failed to fetch historical data for tickers in calculate_metrics_singularity.")
        return {}

    daily_returns_all_metrics = all_tickers_hist_data_metrics.pct_change().iloc[1:].dropna(how='all') # Drop first NaN row and rows where all are NaN
    if 'SPY' not in daily_returns_all_metrics.columns:
        # if not is_called_by_ai: print("      Error: SPY column missing in combined daily returns for metrics calculation.")
        return {}

    processed_count_metrics = 0
    successful_count_metrics = 0
    for ticker_m in tickers_list:
        processed_count_metrics += 1
        # if not is_called_by_ai and processed_count_metrics % 20 == 0 and len(tickers_list) > 20:
            # print(f"        Metrics calculation progress: {processed_count_metrics}/{len(tickers_list)} (Successful: {successful_count_metrics})")

        if ticker_m == 'SPY' or ticker_m not in daily_returns_all_metrics.columns:
            continue # Skip SPY itself or if its data is missing

        ticker_returns_m = daily_returns_all_metrics[ticker_m]
        # Use SPY returns from the same combined fetch for perfect alignment
        spy_aligned_returns_m = daily_returns_all_metrics['SPY']

        # Combine ticker and SPY returns, join with historical SPY Invest scores
        combined_df_m = pd.concat([ticker_returns_m, spy_aligned_returns_m], axis=1, keys=[ticker_m, 'SPY']).dropna() # Drop rows where either is NaN initially

        if spy_invest_scores_hist_10y is not None and not spy_invest_scores_hist_10y.empty:
            # Ensure spy_invest_scores_hist_10y index is datetime and matches
            spy_invest_scores_hist_10y.index = pd.to_datetime(spy_invest_scores_hist_10y.index)
            combined_df_m = combined_df_m.join(spy_invest_scores_hist_10y.rename('SPY_Score'), how='inner') # Inner join to align dates
        else:
            combined_df_m['SPY_Score'] = np.nan # Add a NaN column if scores not available

        if len(combined_df_m) < 252: # Need at least ~1 year of aligned data for 1y metrics
            continue

        # Take the last 1 year (252 trading days) of aligned data
        data_1y_m = combined_df_m.tail(252)
        ticker_returns_1y_m = data_1y_m[ticker_m]
        spy_returns_1y_m = data_1y_m['SPY']
        spy_scores_1y_m = data_1y_m['SPY_Score'] # This will be NaNs if historical scores weren't available

        beta_1y, correlation_1y = np.nan, np.nan
        avg_leverage_uptrend_1y, avg_leverage_downtrend_1y, avg_leverage_general_1y = np.nan, np.nan, np.nan

        try:
            if ticker_returns_1y_m.nunique() > 1 and spy_returns_1y_m.nunique() > 1: # Ensure variance
                spy_variance_1y = np.var(spy_returns_1y_m)
                if not pd.isna(spy_variance_1y) and spy_variance_1y > 1e-12: # Avoid division by zero
                    covariance_matrix_1y = np.cov(ticker_returns_1y_m, spy_returns_1y_m)
                    if covariance_matrix_1y.shape == (2,2): beta_1y = covariance_matrix_1y[0,1] / spy_variance_1y

                correlation_matrix_1y = np.corrcoef(ticker_returns_1y_m, spy_returns_1y_m)
                if correlation_matrix_1y.shape == (2,2):
                    correlation_1y = correlation_matrix_1y[0,1]
                    if pd.isna(correlation_1y): correlation_1y = 0.0 # Default to 0 if NaN after calculation
            else: # No variance or insufficient unique points
                beta_1y, correlation_1y = 0.0, 0.0 # Or handle as error/skip
        except Exception: pass # Silently pass calculation errors for beta/corr

        try:
            # Calculate leverage carefully, avoiding division by zero or tiny SPY returns
            # Replace zero SPY returns with NaN before division to avoid inf/-inf leverage values
            spy_returns_1y_for_leverage = spy_returns_1y_m.replace(0, np.nan)
            with np.errstate(divide='ignore', invalid='ignore'): # Suppress warnings for NaN division
                leverage_raw_1y = ticker_returns_1y_m / spy_returns_1y_for_leverage
            leverage_raw_1y.replace([np.inf, -np.inf], np.nan, inplace=True) # Clean up any remaining infs

            if leverage_raw_1y.notna().any():
                avg_leverage_general_1y = np.nanmean(leverage_raw_1y)

            # Leverage based on SPY Invest Score trend (if available)
            if spy_scores_1y_m.notna().any():
                uptrend_mask = (spy_scores_1y_m > 60) & leverage_raw_1y.notna() # Using >60 for uptrend based on typical score ranges
                if uptrend_mask.any(): avg_leverage_uptrend_1y = np.nanmean(leverage_raw_1y[uptrend_mask])

                downtrend_mask = (spy_scores_1y_m < 40) & leverage_raw_1y.notna() # Using <40 for downtrend
                if downtrend_mask.any(): avg_leverage_downtrend_1y = np.nanmean(leverage_raw_1y[downtrend_mask])
        except Exception: pass # Silently pass leverage calculation errors

        # Only add to metrics if key values are valid
        has_valid_leverage = not pd.isna(avg_leverage_general_1y) or \
                             not pd.isna(avg_leverage_uptrend_1y) or \
                             not pd.isna(avg_leverage_downtrend_1y)

        if not pd.isna(beta_1y) and not pd.isna(correlation_1y) and has_valid_leverage:
            metrics[ticker_m] = {
                'beta_1y': beta_1y, 'correlation_1y': correlation_1y,
                'avg_leverage_uptrend_1y': avg_leverage_uptrend_1y,
                'avg_leverage_downtrend_1y': avg_leverage_downtrend_1y,
                'avg_leverage_general_1y': avg_leverage_general_1y
            }
            successful_count_metrics +=1

    # if not is_called_by_ai:
    #     print(f"    Finished Step: Calculating Metrics. Successful calculations: {successful_count_metrics}/{len(tickers_list)}")
    return metrics


def save_initial_metrics_singularity(metrics: Dict[str, Dict[str, float]], tickers_processed: List[str], is_called_by_ai: bool = False):
    """ Saves calculated initial metrics to CULTIVATE_INITIAL_METRICS_FILE."""
    # if not is_called_by_ai: print("    Starting Step: Saving Initial Metrics...")
    if not metrics:
        # if not is_called_by_ai: print("      Skipping initial metrics save: metrics dictionary is empty.")
        return

    initial_metrics_list_to_save = []
    for ticker_s in tickers_processed: # Iterate over the order of tickers that were meant to be processed
        if ticker_s in metrics: # Check if metrics were successfully calculated for it
             metric_data_s = metrics[ticker_s]
             initial_metrics_list_to_save.append({
                 'Ticker': ticker_s,
                 'Beta (1y)': metric_data_s.get('beta_1y'),
                 'Correlation (1y)': metric_data_s.get('correlation_1y'),
                 'Avg Leverage (Uptrend >50)': metric_data_s.get('avg_leverage_uptrend_1y'), # Note: condition was >60 in calc
                 'Avg Leverage (Downtrend <50)': metric_data_s.get('avg_leverage_downtrend_1y'), # Note: condition was <40 in calc
                 'Avg Leverage (General)': metric_data_s.get('avg_leverage_general_1y')
             })

    if initial_metrics_list_to_save:
        try:
            df_initial_metrics = pd.DataFrame(initial_metrics_list_to_save)
            # Define column order for consistency
            cols_ordered = ['Ticker', 'Beta (1y)', 'Correlation (1y)',
                            'Avg Leverage (Uptrend >50)', 'Avg Leverage (Downtrend <50)', 'Avg Leverage (General)']
            df_initial_metrics = df_initial_metrics.reindex(columns=cols_ordered)

            # Round numeric columns for cleaner CSV output
            numeric_cols_to_round = df_initial_metrics.select_dtypes(include=np.number).columns.tolist()
            for col_to_round in numeric_cols_to_round:
                 df_initial_metrics[col_to_round] = df_initial_metrics[col_to_round].apply(lambda x: round(safe_score(x), 4) if pd.notna(x) else np.nan)

            df_initial_metrics.to_csv(CULTIVATE_INITIAL_METRICS_FILE, index=False, float_format='%.4f')
            # if not is_called_by_ai:
            #     print(f"      Successfully saved initial metrics for {len(df_initial_metrics)} tickers to {CULTIVATE_INITIAL_METRICS_FILE}")
        except Exception as e:
            # if not is_called_by_ai: print(f"      Error saving initial metrics CSV ({CULTIVATE_INITIAL_METRICS_FILE}): {e}")
            pass # Silently pass for AI calls if saving fails

def calculate_cultivate_formulas_singularity(allocation_score: float, is_called_by_ai: bool = False) -> Optional[Dict[str, Any]]:
    """ Calculates Lambda, Omega, Alpha, Beta_alloc, Mu, Rho, Omega_target, Delta, Eta, Kappa."""
    # if not is_called_by_ai: print("    Starting Step: Calculating Cultivate Formula Variables...")
    sigma = safe_score(allocation_score) # Allocation_score is Sigma
    # Clamp sigma to avoid math errors with log/exp at exact 0 or 100
    sigma_safe = max(0.0001, min(99.9999, sigma))
    # Ratio term for some formulas, handle division by zero if sigma_safe is 100
    sigma_ratio_term = sigma_safe / (100.0 - sigma_safe) if (100.0 - sigma_safe) > 1e-9 else np.inf

    results = {}
    try:
        # Omega and Lambda (Cash vs Stock/Hedge allocation of Epsilon)
        log_term_omega = np.log(7.0/6.0) / 50.0
        exp_term_omega = np.exp(-log_term_omega * sigma_safe)
        inner_omega_calc = (49.0/60.0 * sigma_safe * exp_term_omega + 40.0)
        results['omega'] = max(0.0, min(100.0, 100.0 - inner_omega_calc)) # % Cash
        results['lambda'] = max(0.0, min(100.0, 100.0 - results['omega']))  # % Stock/Hedge

        # Lambda_hedge (Overall Hedging % within Lambda part)
        results['lambda_hedge'] = max(0.0, min(100.0, 100 - ((1/1000) * sigma_safe**2 + (7/20) * sigma_safe + 40)))

        # Alpha and Beta_alloc (Common Stock vs Hedging % within Lambda part - theoretical based on lambda_hedge)
        # The description implies Alpha and Beta are parts of Lambda, and Beta_alloc relates to lambda_hedge
        # Original: Alpha (common stock % of Lambda), Beta (hedging % of Lambda)
        # Alpha might be (100 - lambda_hedge) % of Lambda, and Beta_alloc is lambda_hedge % of Lambda
        # Recalculating alpha and beta_alloc based on their proportion of Lambda, where (lambda_hedge) is the proportion for hedging
        alpha_proportion_of_lambda = 100.0 - results['lambda_hedge'] # This is % of Lambda that is NOT hedge, so common stock
        results['alpha'] = max(0.0, min(100.0, alpha_proportion_of_lambda)) # % Common Stock within (Stock/Hedge part)
        results['beta_alloc'] = max(0.0, min(100.0, results['lambda_hedge'])) # % Hedging within (Stock/Hedge part)

        # Mu (Target Beta range for common stocks)
        exp_term_mu = np.exp(-np.log(11.0/4.0) * sigma_ratio_term) if not np.isinf(sigma_ratio_term) else (0.0 if sigma_safe > 50 else 1.0)
        mu_center_val = -1/4 + (11/4) * (1 - exp_term_mu)
        results['mu_center'] = mu_center_val
        results['mu_range'] = (mu_center_val - 2/3, mu_center_val + 2/3)

        # Rho (Target Correlation range for common stocks)
        exp_term_rho = np.exp(-np.log(4.0) * sigma_ratio_term) if not np.isinf(sigma_ratio_term) else (0.0 if sigma_safe > 50 else 1.0)
        rho_center_val = 3/4 - exp_term_rho
        results['rho_center'] = rho_center_val
        results['rho_range'] = (rho_center_val - 1/8, rho_center_val + 1/8)

        # Omega_target (Target Leverage range for common stocks) - name clash with cash omega, this is for leverage
        exp_term_omega_t = np.exp(-np.log(7.0/3.0) * sigma_ratio_term) if not np.isinf(sigma_ratio_term) else (0.0 if sigma_safe > 50 else 1.0)
        omega_target_center_val = -1/2 + (7/2) * (1 - exp_term_omega_t)
        results['omega_target_center'] = omega_target_center_val
        results['omega_target_range'] = (omega_target_center_val - 1/2, omega_target_center_val + 1/2)

        # Delta (Amplification factor for scores)
        exp_term_delta = np.exp(-np.log(11.0/8.0) * sigma_ratio_term) if not np.isinf(sigma_ratio_term) else (0.0 if sigma_safe > 50 else 1.0)
        results['delta'] = max(0.25, min(5.0, 1/4 + (11/4) * (1 - exp_term_delta)))

        # Eta and Kappa (Resource vs Market Hedge allocation within Beta_alloc/lambda_hedge part)
        results['eta'] = max(0.0, min(100.0, -sigma_safe**2 / 500.0 - 3.0*sigma_safe / 10.0 + 60.0)) # % Resource Hedge
        results['kappa'] = max(0.0, min(100.0, 100.0 - results['eta'])) # % Market Hedge

        # if not is_called_by_ai: print("      Cultivate formula variable calculations complete.")
        return results
    except Exception as e:
        # if not is_called_by_ai: print(f"      Error calculating cultivate formulas: {e}"); traceback.print_exc()
        return None

async def select_tickers_singularity(
    tickers_to_filter: list,
    metrics: dict,
    invest_scores_all: dict, # Expects {'ticker': {'score': float, 'live_price': float}}
    formula_results: dict,
    portfolio_value: float,
    is_called_by_ai: bool = False
) -> tuple[list, str | None, dict, int]:
    """
    Selects final tickers based on metrics, scores, and formulas. Async for calculate_ema_invest call.
    Returns: (final_tickers_list, warning_msg, invest_scores_all, num_target_common_stocks_calculated)
    """
    # if not is_called_by_ai:
    #     print("    Starting Step: Selecting Final Tickers (Using Beta/Corr/Leverage, Score > 0)...")

    mu_range = formula_results.get('mu_range', (-np.inf, np.inf))
    rho_range = formula_results.get('rho_range', (-np.inf, np.inf))
    omega_target_range = formula_results.get('omega_target_range', (-np.inf, np.inf)) # Leverage target range
    epsilon_val = safe_score(portfolio_value)

    if epsilon_val <= 0:
        return [], "Error: Invalid portfolio value for ticker selection.", invest_scores_all, 0

    # Target number of common stock tickers for the Sigma portfolio component
    num_tickers_sigma_target_calculated = max(0, max(1, math.ceil(0.3 * math.sqrt(epsilon_val))) - len(HEDGING_TICKERS))
    # if not is_called_by_ai:
    #     print(f"      Target number of common stock tickers (Sigma_count): {num_tickers_sigma_target_calculated}")

    _, spy_invest_latest_raw = await calculate_ema_invest('SPY', 2, is_called_by_ai=True)
    spy_invest_latest = safe_score(spy_invest_latest_raw)

    leverage_key_to_use = 'avg_leverage_general_1y' # Default
    if spy_invest_latest is not None and not pd.isna(spy_invest_latest):
        if spy_invest_latest >= 60: leverage_key_to_use = 'avg_leverage_uptrend_1y'
        elif spy_invest_latest <= 40: leverage_key_to_use = 'avg_leverage_downtrend_1y'
    # if not is_called_by_ai and not is_called_by_ai: # Control printing here
        # print(f"        Using '{leverage_key_to_use}' for leverage filtering based on SPY Invest Score: {spy_invest_latest:.2f}%")

    T1_candidates, T_temp_candidates = [], []
    valid_tickers_for_filtering = [
        t for t in tickers_to_filter
        if t in metrics and t in invest_scores_all and invest_scores_all[t].get('score') is not None
    ]

    for ticker_sel in valid_tickers_for_filtering:
        metric_sel = metrics[ticker_sel]
        score_info_sel = invest_scores_all[ticker_sel]
        current_score_sel = safe_score(score_info_sel.get('score'))

        if current_score_sel <= 0: continue

        beta_s = safe_score(metric_sel.get('beta_1y'))
        corr_s = safe_score(metric_sel.get('correlation_1y'))
        leverage_s = safe_score(metric_sel.get(leverage_key_to_use))

        if pd.isna(beta_s) or pd.isna(corr_s) or pd.isna(leverage_s): continue

        in_mu_range = mu_range[0] <= beta_s <= mu_range[1]
        in_rho_range = rho_range[0] <= corr_s <= rho_range[1]
        in_omega_target_range = omega_target_range[0] <= leverage_s <= omega_target_range[1]

        if in_mu_range and in_rho_range and in_omega_target_range:
            T1_candidates.append({'ticker': ticker_sel, 'score': current_score_sel})
        else:
            mu_center = safe_score(formula_results.get('mu_center', 0))
            mu_half_width = safe_score((mu_range[1] - mu_range[0]) / 2.0)
            rho_center = safe_score(formula_results.get('rho_center', 0))
            rho_half_width = safe_score((rho_range[1] - rho_range[0]) / 2.0)
            omega_t_center = safe_score(formula_results.get('omega_target_center', 0))
            omega_t_half_width = safe_score((omega_target_range[1] - omega_target_range[0]) / 2.0)

            if (mu_center - mu_half_width * 1.5 <= beta_s <= mu_center + mu_half_width * 1.5 and
                rho_center - rho_half_width * 1.5 <= corr_s <= rho_center + rho_half_width * 1.5 and
                omega_t_center - omega_t_half_width * 1.5 <= leverage_s <= omega_t_center + omega_t_half_width * 1.5):
                T_temp_candidates.append({'ticker': ticker_sel, 'score': current_score_sel})

    T1_candidates.sort(key=lambda x: x['score'], reverse=True)
    T_temp_candidates.sort(key=lambda x: x['score'], reverse=True)

    T1_tickers_set_sel = {item['ticker'] for item in T1_candidates}
    T_minus_1_candidates = [item for item in T_temp_candidates if item['ticker'] not in T1_tickers_set_sel]

    try:
        if T1_candidates: pd.DataFrame(T1_candidates).to_csv(CULTIVATE_T1_FILE, index=False)
        else: pd.DataFrame([]).to_csv(CULTIVATE_T1_FILE, index=False) # Save empty if none
        if T_minus_1_candidates: pd.DataFrame(T_minus_1_candidates).to_csv(CULTIVATE_T_MINUS_1_FILE, index=False)
        else: pd.DataFrame([]).to_csv(CULTIVATE_T_MINUS_1_FILE, index=False)
    except Exception as e_csv_interim:
        # if not is_called_by_ai: print(f"      Warning: Error saving T1/T_minus_1 CSVs: {e_csv_interim}")
        pass

    Tf_list_final_selection = T1_candidates[:num_tickers_sigma_target_calculated]
    remaining_needed_sel = num_tickers_sigma_target_calculated - len(Tf_list_final_selection)
    if remaining_needed_sel > 0 and T_minus_1_candidates:
        Tf_list_final_selection.extend(T_minus_1_candidates[:remaining_needed_sel])

    selection_warning_msg = None
    if not Tf_list_final_selection:
        selection_warning_msg = "Warning: No tickers selected for Common Stock portfolio (Tf list is empty)."
    elif len(Tf_list_final_selection) < num_tickers_sigma_target_calculated:
        selection_warning_msg = (f"Warning: Target common stock tickers ({num_tickers_sigma_target_calculated}) "
                                 f"not reached. Selected {len(Tf_list_final_selection)}.")

    final_selected_tickers_only = [item['ticker'] for item in Tf_list_final_selection]
    try:
        if Tf_list_final_selection: pd.DataFrame(Tf_list_final_selection).to_csv(CULTIVATE_TF_FINAL_FILE, index=False)
        else: pd.DataFrame([]).to_csv(CULTIVATE_TF_FINAL_FILE, index=False) # Save empty if none
        # if not is_called_by_ai:
        #     print(f"        Selected {len(final_selected_tickers_only)} final Common Stock tickers (Tf). Saved to {CULTIVATE_TF_FINAL_FILE}")
    except Exception as e_csv_tf_final:
        # if not is_called_by_ai: print(f"      Warning: Error saving Tf (final selection) CSV: {e_csv_tf_final}")
        pass

    # if not is_called_by_ai: print("    Finished Step: Selecting Final Tickers.")
    return final_selected_tickers_only, selection_warning_msg, invest_scores_all, num_tickers_sigma_target_calculated

def build_and_process_portfolios_singularity(
    common_stock_tickers: list, # Final selected common stocks (Tf)
    formula_results: dict, # Contains delta (amplification), omega (cash %), lambda_hedge, eta, kappa, alpha
    total_portfolio_value: float,
    frac_shares: bool,
    invest_scores_all: dict, # {'ticker': {'score': float, 'live_price': float}}
    is_called_by_ai: bool = False
) -> tuple:
    """ Builds and processes Cultivate portfolios based on final selections and formulas.
        Returns combined data, tailored holdings, cash values, and TARGET segment dollar values.
    """
    # if not is_called_by_ai:
    #     print("    Starting Step: Building & Processing Final Cultivate Portfolios...")

    epsilon_val = safe_score(total_portfolio_value)
    cash_allocation_omega_pct = safe_score(formula_results.get('omega', 0.0))
    lambda_val_pct = safe_score(formula_results.get('lambda', 100.0 - cash_allocation_omega_pct))

    # alpha_allocation_pct is % of (Lambda part) for Common Stock
    alpha_allocation_pct = safe_score(formula_results.get('alpha', 0.0)) # This is % of Lambda for Common Stock
    # lambda_hedge_allocation_pct is % of (Lambda part) for Overall Hedging
    lambda_hedge_allocation_pct = safe_score(formula_results.get('lambda_hedge', 0.0)) # This is % of Lambda for Hedging

    eta_pct = safe_score(formula_results.get('eta', 0.0))
    kappa_pct = safe_score(formula_results.get('kappa', 100.0 - eta_pct))
    amplification_delta = safe_score(formula_results.get('delta', 1.0))

    # --- Calculate TARGET dollar values for each major segment ---
    initial_omega_cash_dollar_value = epsilon_val * (cash_allocation_omega_pct / 100.0)
    value_for_lambda_part_stocks_hedges = epsilon_val * (lambda_val_pct / 100.0)

    # Target value for Common Stock (Alpha part of Lambda)
    # formula_results['alpha'] is the percentage of the LAMBDA part that goes to common stocks.
    target_value_for_common_stock_alpha = value_for_lambda_part_stocks_hedges * (alpha_allocation_pct / 100.0)

    # Target value for Overall Hedging (Beta_alloc part of Lambda, which is lambda_hedge_allocation_pct of Lambda)
    # formula_results['lambda_hedge'] is the percentage of the LAMBDA part that goes to HEDGING.
    target_value_for_overall_hedging_beta = value_for_lambda_part_stocks_hedges * (lambda_hedge_allocation_pct / 100.0)


    # Allocate within the Overall Hedging part
    target_value_for_market_hedging_kappa = target_value_for_overall_hedging_beta * (kappa_pct / 100.0)
    target_value_for_resource_hedging_eta = target_value_for_overall_hedging_beta * (eta_pct / 100.0)


    all_ticker_data_for_build = {}
    tickers_needed_for_build = list(set(common_stock_tickers + MARKET_HEDGING_TICKERS + RESOURCE_HEDGING_TICKERS))

    for ticker_build in tickers_needed_for_build:
        if ticker_build in invest_scores_all:
            score_info = invest_scores_all[ticker_build]
            live_price_build = safe_score(score_info.get('live_price'))
            raw_score_build = safe_score(score_info.get('score'))
            if live_price_build > 0 and not pd.isna(raw_score_build):
                all_ticker_data_for_build[ticker_build] = {'live_price': live_price_build, 'raw_invest_score': raw_score_build}
            else:
                all_ticker_data_for_build[ticker_build] = {'live_price': 0.0, 'raw_invest_score': -float('inf'), 'error': 'Invalid price or score'}
        else:
            all_ticker_data_for_build[ticker_build] = {'live_price': 0.0, 'raw_invest_score': -float('inf'), 'error': 'Data missing'}

    combined_portfolio_list_for_saving = []

    def process_sub_portfolio(tickers: List[str], portfolio_label: str):
        temp_list, total_amplified_score = [], 0.0
        for t in [tk for tk in tickers if tk in all_ticker_data_for_build and 'error' not in all_ticker_data_for_build[tk]]:
            data = all_ticker_data_for_build[t]
            amplified_score = max(0.0, (data['raw_invest_score'] * amplification_delta) - (amplification_delta - 1.0) * 50.0)
            temp_list.append({'ticker': t, **data, 'amplified_score': amplified_score, 'portfolio': portfolio_label})
            total_amplified_score += amplified_score
        for entry in temp_list:
            sub_alloc_pct = (entry['amplified_score'] / total_amplified_score) * 100.0 if total_amplified_score > 1e-9 else 0.0
            entry['sub_portfolio_allocation_percent'] = sub_alloc_pct
        return temp_list

    common_stock_portfolio_processed = process_sub_portfolio(common_stock_tickers, "Common Stock")
    for entry_cs in common_stock_portfolio_processed:
        alloc_of_lambda_part = entry_cs['sub_portfolio_allocation_percent'] * (alpha_allocation_pct / 100.0)
        entry_cs['combined_percent_allocation_of_lambda'] = alloc_of_lambda_part
        combined_portfolio_list_for_saving.append(entry_cs)

    market_hedge_portfolio_processed = process_sub_portfolio(MARKET_HEDGING_TICKERS, "Market Hedging")
    for entry_mh in market_hedge_portfolio_processed:
        alloc_of_overall_hedging = entry_mh['sub_portfolio_allocation_percent'] * (kappa_pct / 100.0)
        alloc_of_lambda_part = alloc_of_overall_hedging * (lambda_hedge_allocation_pct / 100.0)
        entry_mh['combined_percent_allocation_of_lambda'] = alloc_of_lambda_part
        combined_portfolio_list_for_saving.append(entry_mh)

    resource_hedge_portfolio_processed = process_sub_portfolio(RESOURCE_HEDGING_TICKERS, "Resource Hedging")
    for entry_rh in resource_hedge_portfolio_processed:
        alloc_of_overall_hedging = entry_rh['sub_portfolio_allocation_percent'] * (eta_pct / 100.0)
        alloc_of_lambda_part = alloc_of_overall_hedging * (lambda_hedge_allocation_pct / 100.0)
        entry_rh['combined_percent_allocation_of_lambda'] = alloc_of_lambda_part
        combined_portfolio_list_for_saving.append(entry_rh)

    total_combined_alloc_lambda_sum = sum(e.get('combined_percent_allocation_of_lambda', 0.0) for e in combined_portfolio_list_for_saving)
    if total_combined_alloc_lambda_sum > 1e-9:
        norm_factor_lambda = 100.0 / total_combined_alloc_lambda_sum
        for e_norm in combined_portfolio_list_for_saving:
            e_norm['combined_percent_allocation_of_lambda'] *= norm_factor_lambda

    tailored_portfolio_entries_final_build = []
    total_actual_money_allocated_to_stocks_hedges = 0.0

    for entry_tailor in combined_portfolio_list_for_saving:
        alloc_pct_within_lambda = safe_score(entry_tailor.get('combined_percent_allocation_of_lambda', 0.0))
        live_price_tailor = safe_score(entry_tailor.get('live_price', 0.0))
        if alloc_pct_within_lambda > 1e-9 and live_price_tailor > 1e-9:
            target_dollar_allocation_for_ticker = value_for_lambda_part_stocks_hedges * (alloc_pct_within_lambda / 100.0)
            shares_to_buy = 0.0
            try:
                exact_shares = target_dollar_allocation_for_ticker / live_price_tailor
                if frac_shares: shares_to_buy = round(exact_shares, 1)
                else: shares_to_buy = float(math.floor(exact_shares))
            except ZeroDivisionError: shares_to_buy = 0.0
            shares_to_buy = max(0.0, shares_to_buy)
            actual_money_spent_on_ticker = shares_to_buy * live_price_tailor
            share_purchase_threshold = 0.1 if frac_shares else 1.0
            if shares_to_buy >= share_purchase_threshold:
                actual_percent_of_total_epsilon = (actual_money_spent_on_ticker / epsilon_val) * 100.0 if epsilon_val > 0 else 0.0
                tailored_portfolio_entries_final_build.append({
                    'ticker': entry_tailor['ticker'],
                    'portfolio_group': entry_tailor.get('portfolio', '?'),
                    'shares': shares_to_buy,
                    'live_price_at_eval': live_price_tailor,
                    'actual_money_allocation': actual_money_spent_on_ticker,
                    'actual_percent_allocation_total_epsilon': actual_percent_of_total_epsilon,
                    'raw_invest_score': entry_tailor.get('raw_invest_score', -float('inf'))
                })
                total_actual_money_allocated_to_stocks_hedges += actual_money_spent_on_ticker

    unspent_from_lambda_part = value_for_lambda_part_stocks_hedges - total_actual_money_allocated_to_stocks_hedges
    final_cash_value_cultivate = max(0.0, initial_omega_cash_dollar_value + unspent_from_lambda_part)
    final_cash_percent_of_epsilon = (final_cash_value_cultivate / epsilon_val) * 100.0 if epsilon_val > 0 else 0.0
    final_cash_percent_of_epsilon = max(0.0, min(100.0, final_cash_percent_of_epsilon))
    tailored_portfolio_entries_final_build.sort(key=lambda x: x.get('raw_invest_score', -float('inf')), reverse=True)
    actual_common_stock_value = sum(e['actual_money_allocation'] for e in tailored_portfolio_entries_final_build if e['portfolio_group'] == 'Common Stock')
    actual_market_hedge_value = sum(e['actual_money_allocation'] for e in tailored_portfolio_entries_final_build if e['portfolio_group'] == 'Market Hedging')
    actual_resource_hedge_value = sum(e['actual_money_allocation'] for e in tailored_portfolio_entries_final_build if e['portfolio_group'] == 'Resource Hedging')

    # if not is_called_by_ai:
    #     print("    Finished Step: Building & Processing Final Cultivate Portfolios.")

    return (combined_portfolio_list_for_saving,
            tailored_portfolio_entries_final_build,
            final_cash_value_cultivate,
            final_cash_percent_of_epsilon,
            value_for_lambda_part_stocks_hedges, # Target non-cash value
            actual_common_stock_value,
            actual_market_hedge_value,
            actual_resource_hedge_value,
            initial_omega_cash_dollar_value, # Target initial cash
            # --- Added return values for target segment values ---
            target_value_for_common_stock_alpha,
            target_value_for_overall_hedging_beta,
            target_value_for_market_hedging_kappa,
            target_value_for_resource_hedging_eta
           )


async def run_cultivate_analysis_singularity(
    portfolio_value: float,
    frac_shares: bool,
    cultivate_code_str: str,
    is_called_by_ai: bool = False,
    is_saving_run: bool = False
) -> tuple[list[dict], list[dict], float, str, float, bool, str | None]:
    """
    Orchestrates Cultivate portfolio analysis for Singularity.
    Returns: (combined_data_for_save, tailored_holdings_final, final_cash, code, epsilon, frac_shares_used, error_msg)
    """
    suppress_sub_prints = is_called_by_ai or is_saving_run

    if not suppress_sub_prints:
        print(f"\n--- Cultivate Analysis (Code: {cultivate_code_str.upper()}, Value: ${portfolio_value:,.0f}) ---")

    epsilon_val = safe_score(portfolio_value)
    if epsilon_val <= 0:
        if not suppress_sub_prints: print("Error: Portfolio value must be a positive number for Cultivate analysis.")
        return [], [], 0.0, cultivate_code_str, epsilon_val, frac_shares, "Error: Invalid portfolio value"

    if not suppress_sub_prints: print("Step 1/7: Getting Allocation Score (Sigma)...")
    allocation_score_cult, _, _ = get_allocation_score(is_called_by_ai=suppress_sub_prints)
    if allocation_score_cult is None:
        if not suppress_sub_prints: print("Error: Failed to retrieve Allocation Score (Sigma). Aborting Cultivate.")
        return [], [], 0.0, cultivate_code_str, epsilon_val, frac_shares, "Error: Failed to get Allocation Score"
    sigma_cult = allocation_score_cult

    if not suppress_sub_prints: print("Step 2/7: Calculating Cultivate Formula Variables...")
    formula_results_cult = calculate_cultivate_formulas_singularity(sigma_cult, is_called_by_ai=suppress_sub_prints)
    if formula_results_cult is None:
        if not suppress_sub_prints: print("Error: Failed to calculate portfolio structure variables. Aborting Cultivate.")
        return [], [], 0.0, cultivate_code_str, epsilon_val, frac_shares, "Error: Formula calculation failed"

    tickers_to_process_cult = []
    if not suppress_sub_prints: print(f"Step 3/7: Getting Initial Ticker List (Code {cultivate_code_str.upper()})...")
    if cultivate_code_str.upper() == 'A':
        tickers_to_process_cult = await asyncio.to_thread(screen_stocks_singularity, is_called_by_ai=suppress_sub_prints)
        if not tickers_to_process_cult and not suppress_sub_prints:
            print("Warning: No stocks passed initial screening for Code A. Proceeding with hedging/cash only.")
    elif cultivate_code_str.upper() == 'B':
        tickers_to_process_cult = await asyncio.to_thread(get_sp500_symbols_singularity, is_called_by_ai=suppress_sub_prints)
        if not tickers_to_process_cult:
            if not suppress_sub_prints: print("Error: Failed to retrieve S&P 500 symbols for Code B. Aborting Cultivate.")
            return [], [], 0.0, cultivate_code_str, epsilon_val, frac_shares, "Error: Failed to get S&P 500 symbols"

    if not suppress_sub_prints: print("Step 4/7: Calculating Metrics (Beta/Corr/Leverage)...")
    spy_hist_data_metrics_cult = await asyncio.to_thread(get_yf_data_singularity, ['SPY'], period="10y", interval="1d", is_called_by_ai=suppress_sub_prints)
    metrics_dict_cult = {}
    if not spy_hist_data_metrics_cult.empty:
        if tickers_to_process_cult:
            metrics_dict_cult = await asyncio.to_thread(calculate_metrics_singularity, tickers_to_process_cult, spy_hist_data_metrics_cult, is_called_by_ai=suppress_sub_prints)
            if not is_saving_run:
                await asyncio.to_thread(save_initial_metrics_singularity, metrics_dict_cult, tickers_to_process_cult, is_called_by_ai=suppress_sub_prints)
    elif not suppress_sub_prints and tickers_to_process_cult:
        print("Warning: Could not get SPY data for metrics; metrics for common stocks will be skipped or empty.")

    if not suppress_sub_prints: print("Step 5/7: Calculating Invest Scores for all relevant tickers...")
    invest_scores_all_cult = {}
    all_tickers_for_scoring_cult = list(set((tickers_to_process_cult if tickers_to_process_cult else []) + HEDGING_TICKERS))
    if all_tickers_for_scoring_cult:
        score_tasks = [calculate_ema_invest(ticker, 2, is_called_by_ai=True) for ticker in all_tickers_for_scoring_cult]
        score_results_tuples = await asyncio.gather(*score_tasks, return_exceptions=True)
        for i, ticker_sc in enumerate(all_tickers_for_scoring_cult):
            res_sc_tuple = score_results_tuples[i]
            if isinstance(res_sc_tuple, Exception) or res_sc_tuple is None or len(res_sc_tuple) < 2 or res_sc_tuple[0] is None or res_sc_tuple[1] is None:
                invest_scores_all_cult[ticker_sc] = {'score': -float('inf'), 'live_price': 0.0, 'error': 'Fetch/Score failed'}
            else:
                live_price_sc, score_val_sc = res_sc_tuple
                invest_scores_all_cult[ticker_sc] = {'score': safe_score(score_val_sc), 'live_price': safe_score(live_price_sc)}
    elif not suppress_sub_prints: print("Warning: No tickers (common or hedging) to score for Cultivate analysis.")

    if not suppress_sub_prints: print("Step 6/7: Selecting Final Common Stock Tickers (Tf)...")
    final_common_stock_tickers_cult, selection_warning_cult, _, num_target_common_stocks_final = await select_tickers_singularity(
        tickers_to_filter=tickers_to_process_cult, metrics=metrics_dict_cult,
        invest_scores_all=invest_scores_all_cult, formula_results=formula_results_cult,
        portfolio_value=epsilon_val, is_called_by_ai=suppress_sub_prints
    )
    if selection_warning_cult and not suppress_sub_prints: print(f"*** {selection_warning_cult} ***")

    if not suppress_sub_prints: print("Step 7/7: Building & Processing Final Cultivate Portfolios...")
    (combined_data_for_save, tailored_holdings_final, final_cash_value_cult,
     final_cash_percent_cult, value_for_lambda_part_target_cult,
     common_value_actual_cult, market_hedge_value_actual_cult, resource_hedge_value_actual_cult,
     initial_omega_cash_target_cult,
     target_common_stock_value_alpha_calc, target_overall_hedging_value_beta_calc,
     target_market_hedging_value_kappa_calc, target_resource_hedging_value_eta_calc
     ) = await asyncio.to_thread(
        build_and_process_portfolios_singularity,
        final_common_stock_tickers_cult,
        formula_results_cult,
        epsilon_val,
        frac_shares,
        invest_scores_all_cult,
        is_called_by_ai=suppress_sub_prints
    )

    if not suppress_sub_prints:
        print("\n--- Cultivate Analysis Results ---")
        print(f"Portfolio Value (Epsilon): ${epsilon_val:,.2f}, Cultivate Code: {cultivate_code_str.upper()}, Fractional Shares: {'Yes' if frac_shares else 'No'}")

        print("\n**Combined Portfolio Allocation (Percentages of Non-Cash Value)**")
        combined_table_data_cult_disp = []
        if combined_data_for_save:
            sorted_combined_for_disp = sorted(combined_data_for_save, key=lambda x: x.get('combined_percent_allocation_of_lambda',0.0), reverse=True)
            for entry_comb_disp in sorted_combined_for_disp:
                alloc_pct_disp = safe_score(entry_comb_disp.get('combined_percent_allocation_of_lambda', 0.0))
                if alloc_pct_disp >= 0.01:
                    price_f_disp = f"${safe_score(entry_comb_disp.get('live_price',0.0)):.2f}"
                    score_f_disp = f"{safe_score(entry_comb_disp.get('raw_invest_score',0.0)):.2f}%"
                    combined_table_data_cult_disp.append([
                        entry_comb_disp.get('ticker','ERR'), entry_comb_disp.get('portfolio','?'),
                        price_f_disp, score_f_disp, f"{alloc_pct_disp:.2f}%"
                    ])
            if combined_table_data_cult_disp:
                print(tabulate(combined_table_data_cult_disp, headers=["Ticker", "Portfolio Group", "Price", "Raw Score", "% of Non-Cash Part"], tablefmt="pretty"))
            else: print("No significant allocations in the combined (non-cash) portfolio.")
        else: print("No combined portfolio data generated (excluding cash).")

        print("\n**Tailored Portfolio (Actual Shares, $ Allocation, and % of Total Epsilon)**")
        tailored_table_data_cult_disp_actual = []
        if tailored_holdings_final:
            for item_tail_actual in tailored_holdings_final:
                shares_f_actual = f"{item_tail_actual['shares']:.1f}" if frac_shares and item_tail_actual['shares'] > 0 else f"{int(item_tail_actual['shares'])}"
                money_f_actual = f"${safe_score(item_tail_actual.get('actual_money_allocation',0.0)):,.2f}"
                percent_f_actual = f"{safe_score(item_tail_actual.get('actual_percent_allocation_total_epsilon',0.0)):.2f}%"
                tailored_table_data_cult_disp_actual.append([
                    item_tail_actual.get('ticker','ERR'), item_tail_actual.get('portfolio_group','?'),
                    shares_f_actual, money_f_actual, percent_f_actual
                ])
        tailored_table_data_cult_disp_actual.append(['Cash', 'Cash Reserve', '-', f"${safe_score(final_cash_value_cult):,.2f}", f"{safe_score(final_cash_percent_cult):.2f}%"])
        print(tabulate(tailored_table_data_cult_disp_actual, headers=["Ticker", "Portfolio Group", "Shares", "$ Allocation", "% of Total Epsilon"], tablefmt="pretty"))

        pie_data_for_chart_cult = [{'ticker': item['ticker'], 'value': item['actual_money_allocation']} for item in tailored_holdings_final if item.get('actual_money_allocation',0) > 1e-9]
        if final_cash_value_cult > 1e-9: pie_data_for_chart_cult.append({'ticker': 'Cash', 'value': final_cash_value_cult})
        if pie_data_for_chart_cult:
            pie_chart_title = f"Cultivate Portfolio (Code {cultivate_code_str.upper()}, Epsilon ${epsilon_val:,.0f})"
            await asyncio.to_thread(generate_portfolio_pie_chart, pie_data_for_chart_cult, pie_chart_title, "cultivate_pie", is_called_by_ai=suppress_sub_prints)

        print("\n**The Invest Greeks (Cultivate Portfolio Structure)**")
        num_tickers_overall_target_report = max(1, math.ceil(0.3 * math.sqrt(epsilon_val)))
        greek_data_cult_disp = [
            ["Sigma (Overall Allocation Score)", f"{safe_score(sigma_cult):.2f}"],
            ["Epsilon (Total Portfolio Value)", f"${safe_score(epsilon_val):,.2f}"],
            ["Omega (% Epsilon to Cash - Initial Target)", f"{safe_score(formula_results_cult.get('omega',0.0)):.2f}% -> ${initial_omega_cash_target_cult:,.2f}"],
            ["Lambda (% Epsilon to Stock/Hedge - Target)", f"{safe_score(formula_results_cult.get('lambda',0.0)):.2f}% -> ${value_for_lambda_part_target_cult:,.2f}"],
            ["  Alpha (% Lambda to Common Stock - Target)", f"{safe_score(formula_results_cult.get('alpha',0.0)):.2f}% -> Target ${target_common_stock_value_alpha_calc:,.2f}, Actual ${common_value_actual_cult:,.2f}"],
            ["  Beta_alloc (% Lambda to Hedging - Target)", f"{safe_score(formula_results_cult.get('beta_alloc',0.0)):.2f}% -> Target ${target_overall_hedging_value_beta_calc:,.2f}"],
            ["    Kappa (% Hedging to Market - Target)", f"{safe_score(formula_results_cult.get('kappa',0.0)):.2f}% -> Target ${target_market_hedging_value_kappa_calc:,.2f}, Actual ${market_hedge_value_actual_cult:,.2f}"],
            ["    Eta (% Hedging to Resource - Target)", f"{safe_score(formula_results_cult.get('eta',0.0)):.2f}% -> Target ${target_resource_hedging_value_eta_calc:,.2f}, Actual ${resource_hedge_value_actual_cult:,.2f}"],
            ["Delta (Amplification Factor)", f"{safe_score(formula_results_cult.get('delta',1.0)):.2f}x"],
            ["Target Common Stock Tickers (from Select)", f"{num_target_common_stocks_final:.0f}"],
            ["Selected Common Tickers (Actual Tf)", f"{len(final_common_stock_tickers_cult)}"],
            ["Target Overall Portfolio Tickers (Incl. Hedges)", f"{num_tickers_overall_target_report:.0f}"],
            ["Final Cash Value (Tailored)", f"${safe_score(final_cash_value_cult):,.2f} ({final_cash_percent_cult:.2f}%)"],
        ]
        print(tabulate(greek_data_cult_disp, headers=["Cultivate Variable", "Value / Breakdown"], tablefmt="grid"))
        print("\n--- Cultivate Analysis Complete ---")

    return (combined_data_for_save, tailored_holdings_final, final_cash_value_cult,
            cultivate_code_str, epsilon_val, frac_shares, None)

async def save_cultivate_data_internal_singularity(
    combined_portfolio_data_to_save: List[Dict], # Expects list from build_and_process_portfolios_singularity
    date_str_to_save: str,
    cultivate_code_for_save: str,
    epsilon_for_save: float,
    is_called_by_ai: bool = False # To control prints from this save function
):
    """Saves combined Cultivate portfolio data to CSV. is_called_by_ai controls its own prints."""
    if not combined_portfolio_data_to_save:
        # if not is_called_by_ai:
            # print(f"[Save Cultivate]: No valid combined portfolio data to save for Code {cultivate_code_for_save}, Epsilon {epsilon_for_save}.")
        return

    # combined_portfolio_data_to_save contains 'combined_percent_allocation_of_lambda'
    # Sort by this allocation before saving
    sorted_combined_for_file = sorted(
        combined_portfolio_data_to_save,
        key=lambda x: x.get('combined_percent_allocation_of_lambda', 0.0),
        reverse=True
    )

    epsilon_int_for_filename = int(epsilon_for_save) # For cleaner filename
    # Ensure CULTIVATE_COMBINED_DATA_FILE_PREFIX is defined
    save_file_cultivate = f"{CULTIVATE_COMBINED_DATA_FILE_PREFIX}{cultivate_code_for_save.upper()}_{epsilon_int_for_filename}.csv"
    file_exists_cult_save_check = os.path.isfile(save_file_cultivate)
    save_count_actual = 0
    headers_for_cult_save = ['DATE', 'TICKER', 'PORTFOLIO_GROUP', 'PRICE', 'RAW_INVEST_SCORE', 'COMBINED_ALLOCATION_PERCENT_OF_LAMBDA']

    try:
        with open(save_file_cultivate, 'a', newline='', encoding='utf-8') as f_cult_out:
            writer_cult_csv = csv.DictWriter(f_cult_out, fieldnames=headers_for_cult_save)
            if not file_exists_cult_save_check or os.path.getsize(f_cult_out.name) == 0:
                writer_cult_csv.writeheader()

            for item_to_write_cult in sorted_combined_for_file:
                alloc_pct_lambda_write = safe_score(item_to_write_cult.get('combined_percent_allocation_of_lambda', 0.0))
                # Only save if allocation (relative to non-cash part) is significant
                if alloc_pct_lambda_write > 1e-4 :
                    price_val_write = safe_score(item_to_write_cult.get('live_price'))
                    score_val_write = safe_score(item_to_write_cult.get('raw_invest_score'))
                    writer_cult_csv.writerow({
                        'DATE': date_str_to_save,
                        'TICKER': item_to_write_cult.get('ticker', 'ERR'),
                        'PORTFOLIO_GROUP': item_to_write_cult.get('portfolio', '?'), # From build_and_process...
                        'PRICE': f"{price_val_write:.2f}" if price_val_write is not None and not pd.isna(price_val_write) else "N/A",
                        'RAW_INVEST_SCORE': f"{score_val_write:.2f}%" if score_val_write is not None and not pd.isna(score_val_write) else "N/A",
                        'COMBINED_ALLOCATION_PERCENT_OF_LAMBDA': f"{alloc_pct_lambda_write:.2f}%"
                    })
                    save_count_actual +=1
        # if not is_called_by_ai:
            # print(f"[Save Cultivate]: Saved {save_count_actual} rows of combined data for Code '{cultivate_code_for_save.upper()}' (Epsilon: {epsilon_int_for_filename}) to '{save_file_cultivate}' for date {date_str_to_save}.")
    except IOError as e_io_cult_write:
        # if not is_called_by_ai: print(f"Error [Save Cultivate]: Writing to save file '{save_file_cultivate}': {e_io_cult_write}")
        pass
    except Exception as e_s_cult_write:
        # if not is_called_by_ai:
        #     print(f"Error [Save Cultivate]: Processing/saving data for Code '{cultivate_code_for_save.upper()}' (Epsilon: {epsilon_for_save}): {e_s_cult_write}")
        #     traceback.print_exc()
        pass


async def handle_cultivate_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """
    Handles the /cultivate command for CLI and AI.
    """
    # if not is_called_by_ai: print("\n--- /cultivate Command ---") # AI handler will announce
    summary_for_ai_response = "Cultivate command initiated."

    cult_code, portfolio_val, frac_s_bool, action_type, date_to_save_val = None, None, None, "run_analysis", None

    if ai_params: # Called by AI
        try:
            cult_code = ai_params.get("cultivate_code", "").upper()
            if cult_code not in ['A', 'B']: return "Error for AI (/cultivate): 'cultivate_code' must be 'A' or 'B'."
            raw_val = ai_params.get("portfolio_value")
            if raw_val is None: return "Error for AI (/cultivate): 'portfolio_value' is required."
            portfolio_val = float(raw_val)
            if portfolio_val <= 0: return "Error for AI (/cultivate): 'portfolio_value' must be positive."
            frac_s_bool = bool(ai_params.get("use_fractional_shares", False)) # Default to False if not provided

            action_type = ai_params.get("action", "run_analysis").lower()
            if action_type not in ["run_analysis", "save_data"]:
                return f"Error for AI (/cultivate): Invalid action '{action_type}'. Must be 'run_analysis' or 'save_data'."
            if action_type == "save_data":
                date_str_raw = ai_params.get("date_to_save")
                if not date_str_raw: return "Error for AI (/cultivate save): 'date_to_save' (MM/DD/YYYY) required."
                datetime.strptime(date_str_raw, '%m/%d/%Y'); date_to_save_val = date_str_raw # Validate and assign
        except (ValueError, KeyError) as e_ai_param: return f"Error for AI (/cultivate): Parameter issue: {e_ai_param}"
        except Exception as e_ai_gen: return f"Error for AI (/cultivate) processing params: {e_ai_gen}"
    else: # CLI Path
        try:
            if len(args) < 3:
                print("CLI Usage: /cultivate <Code A/B> <PortfolioValue> <FracShares yes/no> [save_code 3725]")
                return None
            cult_code = args[0].upper()
            if cult_code not in ['A', 'B']: print("CLI Error: Cultivate Code must be 'A' or 'B'."); return None
            portfolio_val = float(args[1])
            if portfolio_val <= 0: print("CLI Error: Portfolio value must be positive."); return None
            frac_s_str = args[2].lower()
            if frac_s_str not in ['yes', 'no']: print("CLI Error: Fractional shares must be 'yes' or 'no'."); return None
            frac_s_bool = frac_s_str == 'yes'
            if len(args) > 3 and args[3] == "3725": # CLI save action
                action_type = "save_data"
                date_str_cli = input(f"CLI: Enter date (MM/DD/YYYY) to save Cultivate (Code {cult_code}, Val ${portfolio_val:,.0f}): ")
                try: datetime.strptime(date_str_cli, '%m/%d/%Y'); date_to_save_val = date_str_cli
                except ValueError: print("CLI Error: Invalid date format. Save cancelled."); return None
        except (ValueError, IndexError) as e_cli_parse:
            print(f"CLI Error: Problem with Cultivate arguments: {e_cli_parse}"); return None
        except Exception as e_cli_gen:
            print(f"CLI Error handling Cultivate: {e_cli_gen}"); traceback.print_exc(); return None

    # --- Execute Cultivate Analysis ---
    # is_saving_run flag tells run_cultivate_analysis_singularity to suppress its own detailed prints if true
    # is_called_by_ai flag also suppresses prints in run_cultivate_analysis_singularity for AI calls
    is_for_saving_only = action_type == "save_data"

    # if not is_called_by_ai: # CLI announcements
        # if is_for_saving_only: print(f"CLI: Generating Cultivate data for saving (Code: {cult_code}, Val: {portfolio_val}, Date: {date_to_save_val})...")
        # else: print(f"CLI: Running Cultivate Analysis (Code: {cult_code}, Val: {portfolio_val}, FracShares: {frac_s_bool})...")

    combined_data, tailored_entries, final_cash, \
    code_used, eps_used, frac_s_used, err_msg = await run_cultivate_analysis_singularity(
        portfolio_value=portfolio_val, frac_shares=frac_s_bool,
        cultivate_code_str=cult_code,
        is_called_by_ai=is_called_by_ai, # Main flag for AI context
        is_saving_run=is_for_saving_only # Suppresses detailed prints if only for saving
    )

    if err_msg:
        summary_for_ai_response = f"Cultivate analysis (Code {cult_code}) failed: {err_msg}"
        # if not is_called_by_ai: print(summary_for_ai_response) # CLI gets direct print from run_cultivate or here
        return summary_for_ai_response if is_called_by_ai else None

    # --- Handle Saving or Summarize for AI ---
    if action_type == "save_data" and date_to_save_val:
        if combined_data:
            # if not is_called_by_ai: print(f"CLI: Saving Cultivate data for Code {code_used}, Val ${eps_used:,.0f} for date {date_to_save_val}...")
            # save_cultivate_data_internal_singularity is async
            await save_cultivate_data_internal_singularity(
                combined_portfolio_data_to_save=combined_data,
                date_str_to_save=date_to_save_val,
                cultivate_code_for_save=code_used,
                epsilon_for_save=eps_used,
                is_called_by_ai=is_called_by_ai # Control prints from save function
            )
            summary_for_ai_response = f"Cultivate analysis (Code {code_used}, Val ${eps_used:,.0f}) data generated and save process initiated for {date_to_save_val}."
        else:
            summary_for_ai_response = f"Cultivate 'save_data' action requested for Code {code_used}, but no combined data was generated to save."
            # if not is_called_by_ai: print(summary_for_ai_response)
    elif action_type == "run_analysis": # Primarily for display (CLI) or summary (AI)
        summary_for_ai_response = f"Cultivate analysis (Code {code_used}, Val ${eps_used:,.0f}, FracShares {frac_s_used}) completed. "
        if tailored_entries or final_cash > 0:
            summary_for_ai_response += f"Final cash: ${final_cash:,.2f}. "
            if tailored_entries:
                summary_for_ai_response += "Top holdings: " + ", ".join([f"{d['ticker']}({d.get('actual_percent_allocation_total_epsilon',0):.1f}%)" for d in tailored_entries[:3]])
            else: summary_for_ai_response += "No stock holdings in tailored portfolio."
        else: summary_for_ai_response += "No holdings or cash in the tailored portfolio."

    return summary_for_ai_response if is_called_by_ai else None # CLI already had prints from run_cultivate_analysis_singularity

# --- Assess Command Functions --- (Add is_called_by_ai flags)
def ask_singularity_input(prompt: str, validation_fn=None, error_msg: str = "Invalid input.", default_val=None, is_called_by_ai: bool = False) -> Optional[str]:
    """
    Helper function to ask for user input in Singularity CLI, with optional validation.
    Returns validated string or None if validation fails or user cancels.
    This function is CLI-specific and should ideally not be reached in an AI flow.
    """
    if is_called_by_ai:
        # This function is for CLI interaction. AI should provide data via parameters.
        # print("Warning: ask_singularity_input called in AI context. This indicates a logic flow issue.")
        return None # Or raise an error

    while True:
        full_prompt = f"{prompt}"
        if default_val is not None:
            full_prompt += f" (default: {default_val}, press Enter to use)"
        full_prompt += ": "

        user_response = input(full_prompt).strip()
        if not user_response and default_val is not None:
            return str(default_val)

        if not user_response and default_val is None: # Handle case where input is required
            print("Input is required.")
            continue


        if validation_fn:
            if validation_fn(user_response):
                return user_response
            else:
                print(error_msg)
                retry = input("Try again? (yes/no, default: yes): ").lower().strip()
                if retry == 'no':
                    return None
        else: # No validation function, accept any non-empty input (already handled by check above)
            return user_response


async def calculate_portfolio_beta_correlation_singularity(
    portfolio_holdings: List[Dict[str, Any]], # List of {'ticker': str, 'value': float}
    total_portfolio_value: float,
    backtest_period: str, # e.g., "1y", "3mo"
    is_called_by_ai: bool = False
) -> Optional[tuple[float, float]]:
    """
    Calculates weighted average Beta and Correlation for a given portfolio against SPY.
    `portfolio_holdings`: List of dicts, each with 'ticker' and 'value' (actual dollar allocation).
    Cash should be one of the tickers with its value if present.
    """
    # if not is_called_by_ai:
    #     print(f"  Calculating Beta/Correlation over {backtest_period} for portfolio value ${total_portfolio_value:,.2f}...")

    if not portfolio_holdings or total_portfolio_value <= 0:
        # if not is_called_by_ai: print("  Error: No holdings or invalid total value for Beta/Corr calculation.")
        return None

    # Filter out holdings with zero or invalid value, but keep track of Cash
    valid_holdings_for_calc = [h for h in portfolio_holdings if isinstance(h.get('value'), (int, float)) and h['value'] > 1e-9]
    if not valid_holdings_for_calc:
        # If the only holding was Cash and it got filtered, or no valid holdings at all.
        # Check if 'Cash' was the *only* thing initially or effectively became so.
        is_only_cash = all(h['ticker'].upper() == 'CASH' for h in portfolio_holdings) or \
                       (not any(h['ticker'].upper() != 'CASH' for h in valid_holdings_for_calc) and \
                        any(h['ticker'].upper() == 'CASH' for h in valid_holdings_for_calc))

        if is_only_cash:
            # if not is_called_by_ai: print("  Portfolio consists only of Cash. Beta: 0.0, Correlation: 0.0")
            return 0.0, 0.0 # Cash has 0 beta and 0 correlation
        # if not is_called_by_ai: print("  Error: No valid holdings with positive value for Beta/Corr calculation.")
        return None

    portfolio_stock_tickers_assess = [h['ticker'] for h in valid_holdings_for_calc if h['ticker'].upper() != 'CASH']

    if not portfolio_stock_tickers_assess: # Only cash holdings remain after filtering
        # if not is_called_by_ai: print("  Portfolio effectively consists only of Cash after filtering. Beta: 0.0, Correlation: 0.0")
        return 0.0, 0.0

    all_tickers_for_hist_fetch = list(set(portfolio_stock_tickers_assess + ['SPY']))
    # get_yf_data_singularity is now a synchronous wrapper around yf.download (can be slow)
    # For async, this should be: hist_data_assess = await asyncio.to_thread(get_yf_data_singularity, ...)
    hist_data_assess = await asyncio.to_thread(get_yf_data_singularity, all_tickers_for_hist_fetch, period=backtest_period, interval="1d", is_called_by_ai=True)


    if hist_data_assess.empty or 'SPY' not in hist_data_assess.columns:
        # if not is_called_by_ai: print(f"  Error: Could not fetch sufficient historical data for SPY or portfolio tickers for period {backtest_period}.")
        return None
    if hist_data_assess['SPY'].isnull().all() or len(hist_data_assess['SPY'].dropna()) < 20: # Min data points for meaningful calc
        # if not is_called_by_ai: print(f"  Error: Insufficient valid data points for SPY over period {backtest_period}.")
        return None

    daily_returns_assess_df = hist_data_assess.pct_change().iloc[1:] # Remove first row of NaNs
    if daily_returns_assess_df.empty or 'SPY' not in daily_returns_assess_df.columns:
        # if not is_called_by_ai: print("  Error calculating daily returns or SPY returns missing for assessment.")
        return None

    spy_returns_series = daily_returns_assess_df['SPY'].dropna()
    if spy_returns_series.empty or spy_returns_series.std() == 0: # Check for variance
        # if not is_called_by_ai: print("  Error: SPY returns are empty or have no variance. Cannot calculate Beta/Correlation.")
        return None

    stock_metrics_calculated = {} # Store calculated beta/corr for each stock
    for ticker_met in portfolio_stock_tickers_assess:
        beta_val, correlation_val = np.nan, np.nan # Default to NaN
        if ticker_met in daily_returns_assess_df.columns and not daily_returns_assess_df[ticker_met].isnull().all():
            ticker_returns_series = daily_returns_assess_df[ticker_met].dropna()
            # Align data by index (date) and drop any rows where either series has NaN for that date
            aligned_data = pd.concat([ticker_returns_series, spy_returns_series], axis=1, join='inner').dropna()

            if len(aligned_data) >= 20: # Need enough common data points
                aligned_ticker_returns = aligned_data.iloc[:, 0]
                aligned_spy_returns = aligned_data.iloc[:, 1]

                if aligned_ticker_returns.std() > 1e-9 and aligned_spy_returns.std() > 1e-9: # Check for variance in aligned series
                    try:
                        # Calculate Beta: Cov(stock, spy) / Var(spy)
                        covariance_matrix = np.cov(aligned_ticker_returns, aligned_spy_returns)
                        if covariance_matrix.shape == (2,2) and covariance_matrix[1, 1] != 0: # Var(spy) is denominator
                             beta_val = covariance_matrix[0, 1] / covariance_matrix[1, 1]

                        # Calculate Correlation
                        correlation_coef_matrix = np.corrcoef(aligned_ticker_returns, aligned_spy_returns)
                        if correlation_coef_matrix.shape == (2,2):
                            correlation_val = correlation_coef_matrix[0, 1]
                            if pd.isna(correlation_val): correlation_val = 0.0 # Default if somehow NaN after calc
                    except (ValueError, IndexError, TypeError, np.linalg.LinAlgError):
                        # if not is_called_by_ai: print(f"    Could not calculate beta/corr for {ticker_met} due to linalg/data issue.")
                        pass # Keep as NaN
                else: # No variance in one or both of the aligned series
                    beta_val, correlation_val = 0.0, 0.0 # Treat as no relationship if no variance
        stock_metrics_calculated[ticker_met] = {'beta': beta_val, 'correlation': correlation_val}

    # Add Cash metrics (0 beta, 0 correlation)
    stock_metrics_calculated['CASH'] = {'beta': 0.0, 'correlation': 0.0}

    # Calculate weighted average Beta and Correlation for the portfolio
    weighted_beta_sum = 0.0
    weighted_correlation_sum = 0.0
    total_weight_processed = 0.0 # For normalization if some holdings couldn't be processed

    for holding in valid_holdings_for_calc:
        ticker_h = holding['ticker'].upper() # Ensure consistent casing for dict lookup
        value_h = holding['value']
        weight_h = value_h / total_portfolio_value

        metrics_for_ticker = stock_metrics_calculated.get(ticker_h)
        if metrics_for_ticker:
            beta_for_calc = metrics_for_ticker.get('beta', 0.0) # Default to 0 if beta specific calc failed but ticker was processed
            corr_for_calc = metrics_for_ticker.get('correlation', 0.0)

            if not pd.isna(beta_for_calc):
                weighted_beta_sum += weight_h * beta_for_calc
            if not pd.isna(corr_for_calc):
                weighted_correlation_sum += weight_h * corr_for_calc
            total_weight_processed += weight_h
        # else: if not is_called_by_ai and ticker_h != 'CASH': print(f"    Warning: Metrics not found for holding {ticker_h}, excluded from weighted average.")


    # Normalize if total_weight_processed is not 1 (e.g., due to some stocks failing)
    # This is generally not needed if we default missing betas/corrs to 0, as weight is still accounted for.
    # If total_weight_processed is significantly less than 1, it means some stock values were non-zero but their metrics failed.
    # For now, we sum based on original weights assuming 0 for failed metrics.

    # if not is_called_by_ai:
    #     print(f"  Calculated Portfolio Beta: {weighted_beta_sum:.4f}, Correlation: {weighted_correlation_sum:.4f}")
    return weighted_beta_sum, weighted_correlation_sum


# --- Main Assess Command Handler ---
async def handle_assess_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """
    Handles the /assess command for CLI and AI.
    """
    summary_for_ai = "Assessment initiated."
    assess_code_input, specific_params_dict = None, {}

    if ai_params: 
        assess_code_input = ai_params.get("assess_code", "").upper()
        specific_params_dict = ai_params 
    elif args: 
        assess_code_input = args[0].upper()
    else: 
        msg = "Usage: /assess <AssessCode A/B/C/D> [additional_args...]. Type /help for details."
        if not is_called_by_ai: print(msg)
        return "Error: Assess code (A, B, C, or D) not specified." if is_called_by_ai else None

    # --- Code A: Stock Volatility Assessment ---
    if assess_code_input == 'A':
        # ... (Code A logic remains unchanged) ...
        if not is_called_by_ai: print("--- Assess Code A: Stock Volatility ---")
        tickers_str_a, timeframe_str_a, risk_tolerance_a_int = None, None, None
        try:
            if ai_params:
                tickers_str_a = specific_params_dict.get("tickers_str")
                timeframe_str_a = specific_params_dict.get("timeframe_str", "1Y") 
                risk_tolerance_a_int = int(specific_params_dict.get("risk_tolerance", 3)) 
                if not tickers_str_a:
                    return "Error for AI (Assess A): 'tickers_str' (comma-separated tickers) is required."
            elif len(args) >= 4: 
                tickers_str_a = args[1]
                timeframe_str_a = args[2].upper()
                risk_tolerance_a_int = int(args[3])
            else:
                msg_cli_a = "CLI Usage for Code A: /assess A <tickers_comma_sep> <timeframe 1Y/3M/1M> <risk_tolerance 1-5>"
                if not is_called_by_ai: print(msg_cli_a)
                return "Error: Insufficient arguments for Assess Code A." if is_called_by_ai else None

            tickers_list_a = [t.strip().upper() for t in tickers_str_a.split(',') if t.strip()]
            if not tickers_list_a:
                return f"Error (Assess A): No valid tickers found in '{tickers_str_a}'."

            timeframe_upper_a = timeframe_str_a.upper()
            timeframe_map_a = {'1Y': "1y", '3M': "3mo", '1M': "1mo"}
            plot_ema_map_a = {'1Y': 2, '3M': 3, '1M': 3} 

            if timeframe_upper_a not in timeframe_map_a:
                return f"Error (Assess A): Invalid timeframe '{timeframe_str_a}'. Use 1Y, 3M, or 1M."
            selected_yf_period_a = timeframe_map_a[timeframe_upper_a]
            plot_ema_sensitivity_a = plot_ema_map_a[timeframe_upper_a]

            if not (1 <= risk_tolerance_a_int <= 5):
                return f"Error (Assess A): Invalid risk tolerance '{risk_tolerance_a_int}'. Must be 1-5."

            results_for_table_a = []
            assessment_summaries_ai_list = []

            if not is_called_by_ai:
                for ticker_a_graph in tickers_list_a:
                    await asyncio.to_thread(plot_ticker_graph, ticker_a_graph, plot_ema_sensitivity_a, is_called_by_ai=is_called_by_ai)

            for ticker_a_item in tickers_list_a:
                try:
                    hist_df_a = await asyncio.to_thread(get_yf_data_singularity, [ticker_a_item], period=selected_yf_period_a, is_called_by_ai=True)
                    if hist_df_a.empty or ticker_a_item not in hist_df_a.columns or len(hist_df_a[ticker_a_item].dropna()) <= 1:
                        results_for_table_a.append([ticker_a_item, "N/A", "N/A", "N/A", "Data Error"])
                        assessment_summaries_ai_list.append(f"{ticker_a_item}: Data Error.")
                        continue

                    close_prices_series_a = hist_df_a[ticker_a_item].dropna()
                    abs_daily_pct_change_a = close_prices_series_a.pct_change().abs() * 100
                    aapc_val_a = abs_daily_pct_change_a.iloc[1:].mean() if len(abs_daily_pct_change_a.iloc[1:]) > 0 else 0.0
                    vol_score_map_a = [(1,0),(2,1),(3,2),(4,3),(5,4),(6,5),(7,6),(8,7),(9,8),(10,9)] 
                    volatility_score_a = 10 
                    for aapc_thresh, score_val in vol_score_map_a:
                        if aapc_val_a <= aapc_thresh: volatility_score_a = score_val; break
                    start_price_a = close_prices_series_a.iloc[0]
                    end_price_a = close_prices_series_a.iloc[-1]
                    period_change_pct_a = ((end_price_a - start_price_a) / start_price_a) * 100 if start_price_a != 0 else 0.0
                    risk_tolerance_ranges_map = {1:(0,1), 2:(2,3), 3:(4,5), 4:(6,7), 5:(8,10)} 
                    vol_score_min, vol_score_max = risk_tolerance_ranges_map[risk_tolerance_a_int]
                    correspondence_a = "Matches" if vol_score_min <= volatility_score_a <= vol_score_max else "No Match"
                    results_for_table_a.append([ticker_a_item, f"{period_change_pct_a:.2f}%", f"{aapc_val_a:.2f}%", volatility_score_a, correspondence_a])
                    assessment_summaries_ai_list.append(f"{ticker_a_item}({timeframe_upper_a},RT{risk_tolerance_a_int}):Chg {period_change_pct_a:.1f}%,AAPC {aapc_val_a:.1f}%,VolSc {volatility_score_a},Match:{correspondence_a}")
                except Exception as e_item_a:
                    results_for_table_a.append([ticker_a_item, "CalcErr", "CalcErr", "CalcErr", f"Error: {e_item_a}"])
                    assessment_summaries_ai_list.append(f"{ticker_a_item}: Calculation Error ({e_item_a}).")

            if results_for_table_a and not is_called_by_ai: 
                print("\n**Stock Volatility Assessment Results (Code A)**")
                results_for_table_a.sort(key=lambda x: x[3] if isinstance(x[3], (int,float)) else float('inf'))
                print(tabulate(results_for_table_a, headers=["Ticker", f"{timeframe_upper_a} Change", "AAPC (%)", "Vol Score (0-9)", "Risk Match"], tablefmt="pretty"))
            summary_for_ai = "Assess A (Stock Volatility) results: " + " | ".join(assessment_summaries_ai_list) if assessment_summaries_ai_list else "No results for Assess A."
        except ValueError as ve_a:
            summary_for_ai = f"Error (Assess A): Invalid parameter type (e.g., risk tolerance not a number). {ve_a}"
            if not is_called_by_ai: print(summary_for_ai)
        except Exception as e_assess_a:
            summary_for_ai = f"An unexpected error occurred in Assess Code A: {e_assess_a}"
            if not is_called_by_ai: print(summary_for_ai); traceback.print_exc()
        return summary_for_ai

    # --- Code B: Manual Portfolio Assessment ---
    elif assess_code_input == 'B':
        # ... (Code B logic remains unchanged) ...
        if not is_called_by_ai: print("--- Assess Code B: Manual Portfolio Risk (Beta/Correlation) ---")
        backtest_period_b_str, manual_portfolio_holdings_b_list = None, []
        try:
            if ai_params:
                backtest_period_b_str = specific_params_dict.get("backtest_period_str", "1y") 
                raw_holdings_ai = specific_params_dict.get("manual_portfolio_holdings")
                if not raw_holdings_ai or not isinstance(raw_holdings_ai, list):
                    return "Error for AI (Assess B): 'manual_portfolio_holdings' (list of dicts with ticker/shares or ticker/value) is required."
                for holding_item_ai in raw_holdings_ai:
                    ticker_b_item = str(holding_item_ai.get("ticker", "")).upper().strip()
                    shares_b_item_raw = holding_item_ai.get("shares")
                    value_b_item_raw = holding_item_ai.get("value") 
                    if not ticker_b_item: continue 
                    if ticker_b_item == "CASH":
                        if value_b_item_raw is not None:
                            manual_portfolio_holdings_b_list.append({'ticker': 'CASH', 'value': float(value_b_item_raw)})
                        else: return "Error for AI (Assess B): Cash holding needs a 'value'."
                    elif shares_b_item_raw is not None: 
                        shares_b_item = float(shares_b_item_raw)
                        live_price_b_item, _ = await calculate_ema_invest(ticker_b_item, 2, is_called_by_ai=True) 
                        if live_price_b_item is not None and live_price_b_item > 0:
                            holding_value_b = shares_b_item * live_price_b_item
                            manual_portfolio_holdings_b_list.append({'ticker': ticker_b_item, 'value': holding_value_b, 'shares': shares_b_item, 'price_at_eval': live_price_b_item})
                        else: return f"Error for AI (Assess B): Could not fetch live price for {ticker_b_item} to calculate value from shares."
                    elif value_b_item_raw is not None: 
                         manual_portfolio_holdings_b_list.append({'ticker': ticker_b_item, 'value': float(value_b_item_raw)})
                    else: 
                        return f"Error for AI (Assess B): Holding '{ticker_b_item}' needs 'shares' or 'value'."
            elif len(args) >= 2: 
                backtest_period_b_str = args[1].lower()
                if not is_called_by_ai: 
                    print(f"CLI Backtesting Period for Manual Portfolio: {backtest_period_b_str}")
                    print("Enter portfolio holdings (ticker and shares/value). Type 'cash' for cash, 'done' when finished.")
                    while True:
                        ticker_input_b = ask_singularity_input("Enter ticker (or 'cash', 'done')", is_called_by_ai=is_called_by_ai)
                        if ticker_input_b is None: break 
                        ticker_input_b = ticker_input_b.upper().strip()
                        if ticker_input_b == 'DONE': break
                        if ticker_input_b == 'CASH':
                            cash_value_str = ask_singularity_input("Enter cash value", lambda x: float(x) >= 0, "Cash value must be non-negative.", is_called_by_ai=is_called_by_ai)
                            if cash_value_str is None: continue 
                            manual_portfolio_holdings_b_list.append({'ticker': 'CASH', 'value': float(cash_value_str)})
                        else: 
                            shares_str_b = ask_singularity_input(f"Enter shares for {ticker_input_b}", lambda x: float(x) > 0, "Shares must be positive.", is_called_by_ai=is_called_by_ai)
                            if shares_str_b is None: continue
                            shares_b = float(shares_str_b)
                            live_price_b_cli, _ = await calculate_ema_invest(ticker_input_b, 2, is_called_by_ai=True) 
                            if live_price_b_cli is not None and live_price_b_cli > 0:
                                value_b_cli = shares_b * live_price_b_cli
                                manual_portfolio_holdings_b_list.append({'ticker': ticker_input_b, 'value': value_b_cli, 'shares': shares_b, 'price_at_eval': live_price_b_cli})
                                if not is_called_by_ai: print(f"Added {shares_b} shares of {ticker_input_b} at ${live_price_b_cli:.2f} (Value: ${value_b_cli:.2f})")
                            else:
                                if not is_called_by_ai: print(f"Could not fetch price for {ticker_input_b}. Not added.")
            else: 
                msg_cli_b = "CLI Usage for Code B: /assess B <backtest_period 1y/5y/10y> (then prompts for holdings)"
                if not is_called_by_ai: print(msg_cli_b)
                return "Error: Insufficient arguments for Assess Code B." if is_called_by_ai else None

            if backtest_period_b_str not in ['1y', '3y', '5y', '10y']: 
                return f"Error (Assess B): Invalid backtest period '{backtest_period_b_str}'. Use 1y, 3y, 5y, or 10y."
            if not manual_portfolio_holdings_b_list:
                return "Error (Assess B): No portfolio holdings were provided or derived."
            total_value_b_calc = sum(h.get('value', 0.0) for h in manual_portfolio_holdings_b_list)
            if total_value_b_calc <= 0:
                return "Error (Assess B): Total portfolio value is zero or negative based on inputs."

            if not is_called_by_ai:
                print(f"\n--- Manual Portfolio Summary (Assess B) ---")
                print(f"Total Calculated Portfolio Value: ${total_value_b_calc:,.2f}")
                print("Holdings for Beta/Correlation Calculation:")
                for h_disp in manual_portfolio_holdings_b_list:
                    share_price_info = ""
                    if 'shares' in h_disp and 'price_at_eval' in h_disp:
                        share_price_info = f" ({h_disp['shares']} shares @ ${h_disp['price_at_eval']:.2f})"
                    print(f"  - {h_disp['ticker']}: Value ${h_disp['value']:,.2f}{share_price_info}")

            beta_corr_tuple_b = await calculate_portfolio_beta_correlation_singularity(
                manual_portfolio_holdings_b_list, total_value_b_calc, backtest_period_b_str, is_called_by_ai=is_called_by_ai
            )
            if beta_corr_tuple_b:
                weighted_beta_b, weighted_corr_b = beta_corr_tuple_b
                result_text_b = (f"Manual Portfolio Assessment (Value: ${total_value_b_calc:,.2f}, Period: {backtest_period_b_str}): "
                                 f"Weighted Avg Beta vs SPY: {weighted_beta_b:.4f}, Weighted Avg Correlation to SPY: {weighted_corr_b:.4f}.")
                if not is_called_by_ai:
                    print("\n**Manual Portfolio Risk Assessment Results (Code B)**")
                    print(f"  Backtest Period: {backtest_period_b_str}")
                    print(f"  Weighted Average Beta vs SPY: {weighted_beta_b:.4f}")
                    print(f"  Weighted Average Correlation to SPY: {weighted_corr_b:.4f}")
                summary_for_ai = result_text_b
            else:
                summary_for_ai = f"Could not calculate Beta/Correlation for the manual portfolio (Period: {backtest_period_b_str}). Check holdings data or market conditions."
                if not is_called_by_ai: print(summary_for_ai)
        except ValueError as ve_b: 
            summary_for_ai = f"Error (Assess B): Invalid numerical input or data format. {ve_b}"
            if not is_called_by_ai: print(summary_for_ai)
        except Exception as e_assess_b:
            summary_for_ai = f"An unexpected error occurred in Assess Code B: {e_assess_b}"
            if not is_called_by_ai: print(summary_for_ai); traceback.print_exc()
        return summary_for_ai

    # --- Code C: Custom Saved Portfolio Risk Assessment ---
    elif assess_code_input == 'C':
        if not is_called_by_ai: print("--- Assess Code C: Custom Saved Portfolio Risk (Beta/Correlation) ---")
        custom_portfolio_code_c, value_for_assess_c_float, backtest_period_c_str = None, None, None
        try:
            if ai_params:
                custom_portfolio_code_c = specific_params_dict.get("custom_portfolio_code")
                value_raw_c = specific_params_dict.get("value_for_assessment")
                backtest_period_c_str = specific_params_dict.get("backtest_period_str", "3y") 

                if not custom_portfolio_code_c or value_raw_c is None:
                    return "Error for AI (Assess C): 'custom_portfolio_code' and 'value_for_assessment' are required."
                value_for_assess_c_float = float(value_raw_c)
                if value_for_assess_c_float <= 0: return "Error for AI (Assess C): 'value_for_assessment' must be positive."
            elif len(args) >= 5: 
                custom_portfolio_code_c = args[2]
                value_for_assess_c_float = float(args[3])
                if value_for_assess_c_float <= 0: print("CLI Error: Value for assessment must be positive."); return None
                backtest_period_c_str = args[4].lower()
            else:
                msg_cli_c = "CLI Usage for Code C: /assess C <custom_portfolio_code> <value_for_assessment> <backtest_period 1y/3y/5y/10y>"
                if not is_called_by_ai: print(msg_cli_c)
                return "Error: Insufficient arguments for Assess Code C." if is_called_by_ai else None

            if backtest_period_c_str not in ['1y', '3y', '5y', '10y']:
                return f"Error (Assess C): Invalid backtest period '{backtest_period_c_str}'. Use 1y, 3y, 5y, or 10y."

            custom_config_data_c = None
            if not os.path.exists(PORTFOLIO_DB_FILE): 
                return f"Error (Assess C): Portfolio database '{PORTFOLIO_DB_FILE}' not found."
            with open(PORTFOLIO_DB_FILE, 'r', encoding='utf-8', newline='') as file_c_db:
                reader_c_db = csv.DictReader(file_c_db)
                for row_c_db in reader_c_db:
                    if row_c_db.get('portfolio_code','').strip().lower() == custom_portfolio_code_c.lower():
                        custom_config_data_c = row_c_db; break
            if not custom_config_data_c:
                return f"Error (Assess C): Custom portfolio code '{custom_portfolio_code_c}' not found in database."
            
            csv_frac_shares_str_c = custom_config_data_c.get('frac_shares', 'false').strip().lower() #
            frac_shares_from_config_c = csv_frac_shares_str_c in ['true', 'yes'] # Updated logic

            if not is_called_by_ai: print(f"  Generating tailored holdings for '{custom_portfolio_code_c}' with value ${value_for_assess_c_float:,.2f}...")
            _, _, final_cash_val_c, structured_holdings_list_c = await process_custom_portfolio(
                portfolio_data_config=custom_config_data_c,
                tailor_portfolio_requested=True, 
                frac_shares_singularity=frac_shares_from_config_c,
                total_value_singularity=value_for_assess_c_float,
                is_custom_command_simplified_output=True, 
                is_called_by_ai=True 
            )

            portfolio_holdings_for_beta_c = [] 
            if structured_holdings_list_c: 
                for item_h_c in structured_holdings_list_c:
                    if item_h_c.get('actual_money_allocation', 0) > 1e-9 : 
                        portfolio_holdings_for_beta_c.append({'ticker': item_h_c['ticker'], 'value': float(item_h_c['actual_money_allocation'])})
            if final_cash_val_c > 1e-9: 
                portfolio_holdings_for_beta_c.append({'ticker': 'CASH', 'value': float(final_cash_val_c)})

            if not portfolio_holdings_for_beta_c :
                summary_for_ai = f"Assess C for '{custom_portfolio_code_c}' (Value ${value_for_assess_c_float:,.0f}): No holdings derived from portfolio configuration for beta/correlation analysis."
                if not is_called_by_ai: print(summary_for_ai)
                return summary_for_ai 

            beta_corr_tuple_c = await calculate_portfolio_beta_correlation_singularity(
                portfolio_holdings_for_beta_c, value_for_assess_c_float, backtest_period_c_str, is_called_by_ai=is_called_by_ai
            )

            if beta_corr_tuple_c:
                weighted_beta_c, weighted_corr_c = beta_corr_tuple_c
                summary_for_ai = (f"Custom Portfolio '{custom_portfolio_code_c}' (Assessed Value ${value_for_assess_c_float:,.2f}, Period {backtest_period_c_str}): "
                                  f"Weighted Avg Beta vs SPY: {weighted_beta_c:.4f}, Weighted Avg Correlation to SPY: {weighted_corr_c:.4f}.")
                if not is_called_by_ai:
                    print("\n**Custom Portfolio Risk Assessment Results (Code C)**")
                    print(f"  Portfolio Code: {custom_portfolio_code_c}, Assessed Value: ${value_for_assess_c_float:,.2f}")
                    print(f"  Backtest Period: {backtest_period_c_str}")
                    print(f"  Weighted Average Beta vs SPY: {weighted_beta_c:.4f}")
                    print(f"  Weighted Average Correlation to SPY: {weighted_corr_c:.4f}")
            else:
                summary_for_ai = f"Could not calculate Beta/Correlation for custom portfolio '{custom_portfolio_code_c}' (Period {backtest_period_c_str})."
                if not is_called_by_ai: print(summary_for_ai)
        except ValueError as ve_c: 
            summary_for_ai = f"Error (Assess C): Invalid numerical input for value_for_assessment. {ve_c}"
            if not is_called_by_ai: print(summary_for_ai)
        except Exception as e_assess_c:
            summary_for_ai = f"An unexpected error occurred in Assess Code C: {e_assess_c}"
            if not is_called_by_ai: print(summary_for_ai); traceback.print_exc()
        return summary_for_ai

    # --- Code D: Cultivate Portfolio Risk Assessment ---
    elif assess_code_input == 'D':
        # ... (Code D logic remains unchanged regarding its call to run_cultivate_analysis_singularity, which uses the frac_shares parameter directly) ...
        if not is_called_by_ai: print("--- Assess Code D: Cultivate Portfolio Risk (Beta/Correlation) ---")
        cultivate_code_d_str, value_epsilon_d_float, frac_s_d_bool, backtest_period_d_str = None, None, None, None
        try:
            if ai_params:
                cultivate_code_d_str = specific_params_dict.get("cultivate_portfolio_code", "").upper()
                value_eps_raw_d = specific_params_dict.get("value_for_assessment") 
                frac_s_d_bool = specific_params_dict.get("use_fractional_shares", False) 
                backtest_period_d_str = specific_params_dict.get("backtest_period_str", "5y") 

                if cultivate_code_d_str not in ['A', 'B'] or value_eps_raw_d is None:
                    return "Error for AI (Assess D): Valid 'cultivate_portfolio_code' (A/B) and 'value_for_assessment' required."
                value_epsilon_d_float = float(value_eps_raw_d)
                if value_epsilon_d_float <= 0: return "Error for AI (Assess D): 'value_for_assessment' must be positive."
            elif len(args) >= 6: 
                cultivate_code_d_str = args[2].upper()
                value_epsilon_d_float = float(args[3])
                if value_epsilon_d_float <= 0: print("CLI Error: Value for Epsilon must be positive."); return None
                frac_s_str_d = args[4].lower()
                if frac_s_str_d not in ['yes', 'no']: print("CLI Error: Fractional shares must be 'yes' or 'no'."); return None
                frac_s_d_bool = frac_s_str_d == 'yes'
                backtest_period_d_str = args[5].lower()
            else:
                msg_cli_d = "CLI Usage for Code D: /assess D <cultivate_code A/B> <value_epsilon> <frac_shares y/n> <backtest_period 1y/3y/5y/10y>"
                if not is_called_by_ai: print(msg_cli_d)
                return "Error: Insufficient arguments for Assess Code D." if is_called_by_ai else None

            if cultivate_code_d_str not in ['A', 'B']: return f"Error (Assess D): Invalid Cultivate Code '{cultivate_code_d_str}'. Use A or B."
            if backtest_period_d_str not in ['1y', '3y', '5y', '10y']:
                 return f"Error (Assess D): Invalid backtest period '{backtest_period_d_str}'. Use 1y, 3y, 5y, or 10y."

            if not is_called_by_ai: print(f"  Running Cultivate analysis (Code {cultivate_code_d_str}, Epsilon ${value_epsilon_d_float:,.0f}) to get holdings for assessment...")
            _, tailored_holdings_list_d, final_cash_val_d, _, _, _, err_msg_cult_d = await run_cultivate_analysis_singularity(
                portfolio_value=value_epsilon_d_float,
                frac_shares=frac_s_d_bool, # This boolean is passed directly to run_cultivate
                cultivate_code_str=cultivate_code_d_str,
                is_called_by_ai=True, 
                is_saving_run=True    
            )
            if err_msg_cult_d:
                return f"Error (Assess D) during underlying Cultivate run: {err_msg_cult_d}"

            portfolio_holdings_for_beta_d = [] 
            if tailored_holdings_list_d: 
                for item_h_d in tailored_holdings_list_d:
                    if item_h_d.get('actual_money_allocation', 0) > 1e-9:
                        portfolio_holdings_for_beta_d.append({'ticker': item_h_d['ticker'], 'value': float(item_h_d['actual_money_allocation'])})
            if final_cash_val_d > 1e-9:
                portfolio_holdings_for_beta_d.append({'ticker': 'CASH', 'value': float(final_cash_val_d)})

            if not portfolio_holdings_for_beta_d:
                summary_for_ai = f"Assess D for Cultivate Code '{cultivate_code_d_str}' (Epsilon ${value_epsilon_d_float:,.0f}): No holdings derived from Cultivate analysis."
                if not is_called_by_ai: print(summary_for_ai)
                return summary_for_ai

            beta_corr_tuple_d = await calculate_portfolio_beta_correlation_singularity(
                portfolio_holdings_for_beta_d, value_epsilon_d_float, backtest_period_d_str, is_called_by_ai=is_called_by_ai
            )
            if beta_corr_tuple_d:
                weighted_beta_d, weighted_corr_d = beta_corr_tuple_d
                summary_for_ai = (f"Cultivate Portfolio Risk (Code {cultivate_code_d_str}, Epsilon ${value_epsilon_d_float:,.2f}, Period {backtest_period_d_str}): "
                                  f"Weighted Avg Beta vs SPY: {weighted_beta_d:.4f}, Weighted Avg Correlation to SPY: {weighted_corr_d:.4f}.")
                if not is_called_by_ai:
                    print("\n**Cultivate Portfolio Risk Assessment Results (Code D)**")
                    print(f"  Cultivate Code: {cultivate_code_d_str}, Epsilon Value: ${value_epsilon_d_float:,.2f}, Fractional Shares: {'Yes' if frac_s_d_bool else 'No'}")
                    print(f"  Backtest Period: {backtest_period_d_str}")
                    print(f"  Weighted Average Beta vs SPY: {weighted_beta_d:.4f}")
                    print(f"  Weighted Average Correlation to SPY: {weighted_corr_d:.4f}")
            else:
                summary_for_ai = f"Could not calculate Beta/Correlation for Cultivate portfolio (Code {cultivate_code_d_str}, Period {backtest_period_d_str})."
                if not is_called_by_ai: print(summary_for_ai)
        except ValueError as ve_d: 
            summary_for_ai = f"Error (Assess D): Invalid numerical input for Epsilon value. {ve_d}"
            if not is_called_by_ai: print(summary_for_ai)
        except Exception as e_assess_d:
            summary_for_ai = f"An unexpected error occurred in Assess Code D: {e_assess_d}"
            if not is_called_by_ai: print(summary_for_ai); traceback.print_exc()
        return summary_for_ai
    else: 
        msg = f"Unknown or unsupported Assess Code: '{assess_code_input}'. Use A, B, C, or D."
        if not is_called_by_ai: print(msg)
        return msg
    
# --- R.I.S.K. Module Functions --- (Add is_called_by_ai flags)
def get_sp100_symbols_risk(is_called_by_ai: bool = False) -> list:
    """Fetches S&P 100 symbols for RISK module."""
    try:
        sp100_list_url = 'https://en.wikipedia.org/wiki/S%26P_100'
        # This is a blocking call, consider asyncio.to_thread if called from a highly async path frequently
        df_list = pd.read_html(sp100_list_url)
        # The S&P 100 components table is usually the 3rd table (index 2) on the Wikipedia page
        # This can change if Wikipedia page structure is updated.
        df = df_list[2]
        symbols = df['Symbol'].tolist()
        # Replace '.' with '-' for yfinance compatibility (e.g., BRK.B -> BRK-B)
        return [s.replace('.', '-') for s in symbols if isinstance(s, str)]
    except Exception as e:
        risk_logger.error(f"Error fetching S&P 100 symbols: {e}")
        # if not is_called_by_ai: print(f"Error fetching S&P 100 symbols: {e}")
        return []

def calculate_ma_risk(symbol: str, ma_window: int, is_called_by_ai: bool = False) -> Optional[bool]:
    """Calculates if price is above MA for a single symbol for RISK module."""
    try:
        symbol_yf = symbol.replace('.', '-')
        # Determine period based on MA window to optimize data download
        if ma_window >= 200: period_str = '2y'
        elif ma_window >= 50: period_str = '1y'
        elif ma_window >= 20: period_str = '6mo'
        else: period_str = '3mo'

        # yf.download is blocking
        data = yf.download(symbol_yf, period=period_str, interval='1d', progress=False, timeout=15) # Adjusted timeout

        if data.empty or len(data) < ma_window:
            # risk_logger.debug(f"Insufficient data for {symbol_yf} MA({ma_window}). Have {len(data)} points.")
            return None

        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        if data['Close'].isnull().all(): return None
        data['Close'] = data['Close'].ffill() # Forward fill to handle sparse NaNs
        if data['Close'].isnull().any(): return None # Still null after ffill

        ma_col_name = f'{ma_window}_day_ma'
        data[ma_col_name] = data['Close'].rolling(window=ma_window).mean()

        latest_close = data['Close'].dropna().iloc[-1] if not data['Close'].dropna().empty else None
        latest_ma = data[ma_col_name].dropna().iloc[-1] if not data[ma_col_name].dropna().empty else None

        if latest_close is None or latest_ma is None:
            # risk_logger.debug(f"Could not determine latest close/MA for {symbol_yf} MA({ma_window}).")
            return None
        return latest_close > latest_ma
    except Exception as e:
        # risk_logger.warning(f"Error processing {symbol} MA({ma_window}): {type(e).__name__}") # Less verbose for common issues
        return None

async def calculate_percentage_above_ma_risk(symbols: List[str], ma_window: int, is_called_by_ai: bool = False) -> float:
    """
    Calculates percentage of symbols above a given moving average.
    OPTIMIZED to use a single bulk download instead of one per symbol.
    """
    if not symbols:
        return 0.0

    # Determine period based on MA window to optimize data download
    if ma_window >= 200:
        period_str = '2y'
    elif ma_window >= 50:
        period_str = '1y'
    else:
        period_str = '6mo'

    try:
        # Perform one bulk download for all symbols. This is the key optimization.
        data = await asyncio.to_thread(
            yf.download,
            tickers=symbols,
            period=period_str,
            interval='1d',
            progress=False,
            auto_adjust=False, # Keep 'Close' column
            group_by='ticker', # Grouping by ticker is easier to work with
            timeout=90 # Increased timeout for large request
        )

        if data.empty:
            risk_logger.warning(f"MA({ma_window}): Bulk yfinance download returned no data.")
            return 0.0

        above_ma_count = 0
        valid_stocks_count = 0

        # Iterate through each symbol that we attempted to download
        for symbol in symbols:
            # Check if we got data for this specific symbol
            if symbol not in data.columns.levels[0]:
                continue

            # Extract the 'Close' prices for the symbol, drop NaNs
            close_prices = data[symbol]['Close'].dropna()

            # Ensure we have enough data points for the moving average calculation
            if len(close_prices) < ma_window:
                continue

            valid_stocks_count += 1
            # Calculate the moving average
            ma = close_prices.rolling(window=ma_window).mean()

            # Get the last valid price and the last valid MA value
            last_price = close_prices.iloc[-1]
            last_ma = ma.iloc[-1]

            # If the last price is above the last MA, increment the counter
            if pd.notna(last_price) and pd.notna(last_ma) and last_price > last_ma:
                above_ma_count += 1

        if valid_stocks_count == 0:
            risk_logger.warning(f"MA({ma_window}): No stocks with sufficient data found after bulk download.")
            return 0.0

        percentage = (above_ma_count / valid_stocks_count) * 100
        risk_logger.info(f"MA({ma_window}): {above_ma_count}/{valid_stocks_count} stocks above MA ({percentage:.2f}%)")
        return percentage

    except Exception as e:
        risk_logger.error(f"Error in OPTIMIZED calculate_percentage_above_ma_risk for MA({ma_window}): {e}")
        # Return a neutral value on error
        return 0.0
    
# These functions now become async due to calculate_percentage_above_ma_risk becoming async
async def calculate_s5tw_risk(is_called_by_ai: bool = False): # S&P500 20-day MA
    # get_sp500_symbols_singularity is blocking
    sp500_symbols = await asyncio.to_thread(get_sp500_symbols_singularity, is_called_by_ai=True)
    return await calculate_percentage_above_ma_risk(sp500_symbols, 20, is_called_by_ai)

async def calculate_s5th_risk(is_called_by_ai: bool = False): # S&P500 200-day MA
    sp500_symbols = await asyncio.to_thread(get_sp500_symbols_singularity, is_called_by_ai=True)
    return await calculate_percentage_above_ma_risk(sp500_symbols, 200, is_called_by_ai)

async def calculate_s1fd_risk(is_called_by_ai: bool = False): # S&P100 5-day MA
    sp100_symbols = await asyncio.to_thread(get_sp100_symbols_risk, is_called_by_ai=True)
    return await calculate_percentage_above_ma_risk(sp100_symbols, 5, is_called_by_ai)

async def calculate_s1tw_risk(is_called_by_ai: bool = False): # S&P100 20-day MA
    sp100_symbols = await asyncio.to_thread(get_sp100_symbols_risk, is_called_by_ai=True)
    return await calculate_percentage_above_ma_risk(sp100_symbols, 20, is_called_by_ai)


def get_live_price_and_ma_risk(ticker: str, ma_windows: Optional[List[int]] = None, is_called_by_ai: bool = False) -> tuple[Optional[float], Dict[int, Optional[float]]]:
    """Fetches live price and specified MAs for a ticker for RISK module. Synchronous."""
    if ma_windows is None: ma_windows = [20, 50] # Default MAs if none specified
    ma_values_result = {ma: None for ma in ma_windows} # Initialize with None

    try:
        stock = yf.Ticker(ticker)
        hist_period = '6mo' # Default, adjust based on max MA window
        if ma_windows:
            max_ma_needed = max(ma_windows, default=0)
            if max_ma_needed >= 200: hist_period = '2y'
            elif max_ma_needed >= 50: hist_period = '1y'
            # else 6mo is fine

        # yf.history is blocking
        hist_df = stock.history(period=hist_period, interval="1d")
        if hist_df.empty:
            risk_logger.warning(f"No history data for {ticker} (period {hist_period}) in get_live_price_and_ma_risk")
            return None, ma_values_result

        hist_df['Close'] = pd.to_numeric(hist_df['Close'], errors='coerce')
        live_price_val = hist_df['Close'].dropna().iloc[-1] if not hist_df['Close'].dropna().empty else None

        for window in ma_windows:
            if len(hist_df) >= window:
                ma_calc_val = hist_df['Close'].rolling(window=window).mean().iloc[-1]
                ma_values_result[window] = ma_calc_val if not pd.isna(ma_calc_val) else None
            # else: ma_values_result[window] remains None (insufficient data)

        return live_price_val, ma_values_result
    except Exception as e:
        risk_logger.error(f"Error in get_live_price_and_ma_risk for {ticker}: {e}")
        return None, ma_values_result # Return None for price and dict of Nones for MAs

def calculate_ema_score_risk(ticker:str ="SPY", is_called_by_ai: bool = False) -> Optional[float]:
    """Calculates specific EMA-based score for RISK module (0-100). Synchronous."""
    try:
        # yf.Ticker().history is blocking
        data = yf.Ticker(ticker).history(period="1y", interval="1d") # Fetch 1 year of daily data
        if data.empty or len(data) < 55: # Check if enough data for EMA 55
            # risk_logger.warning(f"Insufficient data for {ticker} EMA score (need >55 days, have {len(data)})")
            return None

        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        if data['Close'].isnull().all(): return None
        data['Close'] = data['Close'].ffill() # Fill NaNs before EMA calculation
        if data['Close'].isnull().any(): return None # Still NaN after ffill

        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()

        ema_8_val = data['EMA_8'].iloc[-1]
        ema_55_val = data['EMA_55'].iloc[-1]

        if pd.isna(ema_8_val) or pd.isna(ema_55_val) or ema_55_val == 0:
            # risk_logger.warning(f"NaN or zero EMA_55 for {ticker} in RISK EMA score. EMA_8: {ema_8_val}, EMA_55: {ema_55_val}")
            return None

        # Original R.I.S.K. formula for its EMA Score
        ema_invest_specific_risk = (((ema_8_val - ema_55_val) / ema_55_val * 5) + 0.5) * 100
        return float(np.clip(ema_invest_specific_risk, 0, 100)) # Clip score between 0 and 100
    except Exception as e:
        risk_logger.error(f"Error calculating EMA score for {ticker} in RISK module: {e}")
        return None

async def calculate_risk_scores_singularity(is_called_by_ai: bool = False) -> tuple:
    """Calculates all RISK scores for Singularity. Main components are sync, MA % are async."""
    risk_logger.info("Starting main RISK score component calculation.")

    # These are blocking yfinance calls, run them in threads
    spy_live_price, spy_mas = await asyncio.to_thread(get_live_price_and_ma_risk, 'SPY', [20, 50], is_called_by_ai=True)
    vix_live_price, _ = await asyncio.to_thread(get_live_price_and_ma_risk, '^VIX', [], is_called_by_ai=True)
    rut_live_price, rut_mas = await asyncio.to_thread(get_live_price_and_ma_risk, '^RUT', [20, 50], is_called_by_ai=True)
    oex_live_price, oex_mas = await asyncio.to_thread(get_live_price_and_ma_risk, '^OEX', [20, 50], is_called_by_ai=True)

    # These MA percentage calculations are already async
    s5tw_val = await calculate_s5tw_risk(is_called_by_ai=True)
    s5th_val = await calculate_s5th_risk(is_called_by_ai=True)
    s1fd_val = await calculate_s1fd_risk(is_called_by_ai=True)
    s1tw_val = await calculate_s1tw_risk(is_called_by_ai=True)

    critical_data_map = {
        'SPY Price': spy_live_price, 'SPY MA20': spy_mas.get(20), 'SPY MA50': spy_mas.get(50),
        'VIX Price': vix_live_price,
        'RUT Price': rut_live_price, 'RUT MA20': rut_mas.get(20), 'RUT MA50': rut_mas.get(50),
        'OEX Price': oex_live_price, 'OEX MA20': oex_mas.get(20), 'OEX MA50': oex_mas.get(50),
        'S5TW (%)': s5tw_val, 'S5TH (%)': s5th_val, 'S1FD (%)': s1fd_val, 'S1TW (%)': s1tw_val
    }

    all_critical_valid = True
    for name, value in critical_data_map.items():
        if value is None or (isinstance(value, (float, int)) and pd.isna(value)): # Also check int for pd.isna robustness
            risk_logger.error(f"RISK score calculation: Missing critical data for '{name}'. Value: {value}")
            all_critical_valid = False
            # If a MA value is None (e.g. spy_mas[20]), the formulas below will use 50.0 as neutral default.
            # For prices (SPY, VIX etc.), if None, the entire calculation might be compromised.

    if not all_critical_valid and any(p is None for p in [spy_live_price, vix_live_price, rut_live_price, oex_live_price]):
        risk_logger.error(f"Cannot calculate full RISK scores due to missing critical PRICE data components.")
        return None, None, None, None, spy_live_price, vix_live_price # Return what we have

    try:
        # Component score calculations (ensure inputs are numbers, use defaults for MAs if None)
        spy20 = np.clip(((spy_live_price - spy_mas.get(20, spy_live_price)) / 20) + 50, 0, 100) if spy_live_price is not None else 50.0
        spy50 = np.clip(((spy_live_price - spy_mas.get(50, spy_live_price) - 150) / 20) + 50, 0, 100) if spy_live_price is not None else 50.0
        vix_score = np.clip((((vix_live_price - 15) * -5) + 50), 0, 100) if vix_live_price is not None else 50.0
        rut20 = np.clip(((rut_live_price - rut_mas.get(20, rut_live_price)) / 10) + 50, 0, 100) if rut_live_price is not None else 50.0
        rut50 = np.clip(((rut_live_price - rut_mas.get(50, rut_live_price)) / 5) + 50, 0, 100) if rut_live_price is not None else 50.0
        s5tw_score = np.clip(((s5tw_val - 60) + 50), 0, 100) if s5tw_val is not None else 50.0
        s5th_score = np.clip(((s5th_val - 70) + 50), 0, 100) if s5th_val is not None else 50.0
        # OEX formula was division by 100, seems like a large divisor compared to RUT's /10 and /5.
        # This follows the original script. If it's a typo, it should be /10 or similar.
        oex20_score = np.clip(((oex_live_price - oex_mas.get(20, oex_live_price)) / 100) + 50, 0, 100) if oex_live_price is not None else 50.0
        oex50_score = np.clip(((oex_live_price - oex_mas.get(50, oex_live_price) - 25) / 100) + 50, 0, 100) if oex_live_price is not None else 50.0
        s1fd_score = np.clip(((s1fd_val - 60) + 50), 0, 100) if s1fd_val is not None else 50.0
        s1tw_score = np.clip(((s1tw_val - 70) + 50), 0, 100) if s1tw_val is not None else 50.0
    except TypeError as te: # Handles if any price component was None and not caught by initial check
        risk_logger.error(f"TypeError during RISK score component calculation. One of the inputs might be None or non-numeric: {te}")
        return None, None, None, None, spy_live_price, vix_live_price

    # R.I.S.K specific EMA score for SPY (sync call)
    ema_score_val_risk = await asyncio.to_thread(calculate_ema_score_risk, "SPY", is_called_by_ai=True)
    if ema_score_val_risk is None:
        risk_logger.error("RISK EMA score calculation for SPY failed. Using neutral 50.0 for combined scores.")
        ema_score_val_risk = 50.0 # Neutral default if it fails

    general_score = np.clip(((3*spy20)+spy50+(3*vix_score)+(3*rut50)+rut20+(2*s5tw_score)+s5th_score)/13.0, 0, 100)
    large_cap_score = np.clip(((3*oex20_score)+oex50_score+(2*s1fd_score)+s1tw_score)/7.0, 0, 100)
    combined_score = np.clip((general_score + large_cap_score + ema_score_val_risk) / 3.0, 0, 100)

    risk_logger.info(f"RISK Scores Calculated: General={general_score:.2f}, LargeCap={large_cap_score:.2f}, EMA(RISK)={ema_score_val_risk:.2f}, Combined={combined_score:.2f}")
    return general_score, large_cap_score, ema_score_val_risk, combined_score, spy_live_price, vix_live_price


def calculate_recession_likelihood_ema_risk(ticker:str ="SPY", interval:str ="1mo", period:str ="5y", is_called_by_ai: bool = False) -> Optional[float]:
    """Calculates Momentum Based Recession Likelihood for RISK module. Synchronous."""
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval) # Blocking
        if data.empty or len(data) < 55: # Need enough data for EMA 55
            return None
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()
        ema_8, ema_55 = data['EMA_8'].iloc[-1], data['EMA_55'].iloc[-1]

        if pd.isna(ema_8) or pd.isna(ema_55) or ema_55 == 0: return None

        # Original formula: (((ema8-ema55)/ema55)+0.5)*100
        x_value = (((ema_8 - ema_55) / ema_55) + 0.5) * 100
        # Original formula: 100*EXP(-((45.622216*X/2750)^4))
        likelihood = 100 * np.exp(-((45.622216 * x_value / 2750) ** 4))
        return float(np.clip(likelihood, 0, 100))
    except Exception as e:
        risk_logger.error(f"Error calculating momentum (EMA) recession likelihood for {ticker}: {e}")
        return None

def calculate_recession_likelihood_vix_risk(vix_price: Optional[float], is_called_by_ai: bool = False) -> Optional[float]:
    """Calculates VIX Based Recession Likelihood for RISK module."""
    if vix_price is not None and not pd.isna(vix_price):
        try:
            # Original formula: 0.01384083*(VIX Price^2)
            likelihood = 0.01384083 * (float(vix_price) ** 2)
            return float(np.clip(likelihood, 0, 100))
        except ValueError:
            risk_logger.error(f"Could not convert VIX price '{vix_price}' to float for recession calc.")
            return None
    return None

def calculate_market_invest_score_risk(vix_contraction_chance: Optional[float], ema_contraction_chance: Optional[float], is_called_by_ai: bool = False) -> tuple:
    """
    Calculates Market Invest Score (MIS) for RISK module.
    Returns: (raw_unapped_mis, capped_mis_for_signal_logic, rounded_capped_mis_for_display)
    """
    if vix_contraction_chance is None or ema_contraction_chance is None:
        risk_logger.warning("Cannot calculate Market Invest Score: VIX or EMA contraction chance is None.")
        return None, None, None

    uncapped_score_mis = None
    try:
        if ema_contraction_chance == 0: # Avoid division by zero
            # If EMA chance is 0 (strong market momentum), MIS depends on VIX chance.
            # If VIX is also very low (e.g., 0), market is extremely bullish (high MIS, e.g. 100).
            # If VIX is high while EMA chance is 0, it's a divergence, could be risky (low MIS, e.g. 0).
            # Original formula structure (50 - (ratio-1)*100) would lead to issues.
            # Let's interpret: if EMA chance is 0, means strong upside momentum.
            # If VIX is also 0, implies very low fear -> ratio is undefined 0/0.
            # A robust interpretation:
            if vix_contraction_chance == 0: # Both 0, extremely bullish
                uncapped_score_mis = 200.0 # Example: very high score before capping, effectively 100 after cap
            else: # EMA 0, VIX > 0: Strong momentum but some fear. Ratio -> infinity. Score -> -infinity
                uncapped_score_mis = -100.0 # Example: very low score, effectively 0 after cap
        else: # Normal case
            ratio = vix_contraction_chance / ema_contraction_chance
            # Original formula: 100.0 - (((ratio - 1.0) * 100.0) + 50.0) = 50.0 - (ratio - 1.0) * 100.0
            uncapped_score_mis = 50.0 - (ratio - 1.0) * 100.0
    except Exception as e:
        risk_logger.error(f"Error calculating Market Invest Score ratio component: {e}")
        return None, None, None

    if uncapped_score_mis is None: return None, None, None

    capped_score_for_signal_mis = float(np.clip(uncapped_score_mis, 0, 100))
    rounded_capped_score_for_display_mis = int(round(capped_score_for_signal_mis))
    return uncapped_score_mis, capped_score_for_signal_mis, rounded_capped_score_for_display_mis


def calculate_market_ivr_risk(new_raw_score: Optional[float], csv_file_path: str = RISK_CSV_FILE, is_called_by_ai: bool = False) -> Optional[float]:
    """Calculates Market Invest Score Rank (IVR) for RISK module."""
    if new_raw_score is None:
        risk_logger.warning("Cannot calculate Market IVR: new raw score is None.")
        return None

    historical_raw_scores = []
    if os.path.exists(csv_file_path):
        try:
            df = pd.read_csv(csv_file_path, on_bad_lines='skip')
            if 'Raw Market Invest Score' in df.columns:
                # Ensure only valid numeric scores are used from history
                historical_raw_scores = pd.to_numeric(df['Raw Market Invest Score'], errors='coerce').dropna().tolist()
        except Exception as e:
            risk_logger.error(f"Error reading historical raw scores for IVR from {csv_file_path}: {e}")
            # Continue, IVR will be based on current score vs empty list if read fails
    else:
        risk_logger.info(f"Historical data file {csv_file_path} not found for IVR calculation. IVR will be 0 or 100.")


    if not historical_raw_scores: # No history or history is all non-numeric
        # If new_raw_score is, for example, positive, it's the "highest" so far (rank 100). If negative, "lowest" (rank 0).
        # A common IVR interpretation when no history: if it's above a midpoint (e.g. 0 for raw MIS), 100, else 0.
        # Or, simply, if it's the only data point, it's at 100% of its own range (if positive) or 0% (if negative).
        # Let's assume if no history, it's ranked relative to itself being high or low.
        # For simplicity, if no comparable history, current score is considered "high" if positive, "low" if negative for ranking.
        # This is a bit arbitrary. A common IVR definition implies comparison against a population.
        # If the population is 1 (the current score), rank is often taken as 50%, or 0/100 based on its value.
        # Let's use: if no history, IVR is 0 if score is <=0, and 100 if score > 0.
        # However, original script might imply 0 if it's the lowest (which it is if no history and negative).
        return 0.0 if new_raw_score <= np.mean(historical_raw_scores if historical_raw_scores else [0]) else 100.0 # Simplified: if better than mean of (empty or [0])

    # IVR: Percentage of historical scores that are LOWER than the current score.
    lower_count = sum(1 for score in historical_raw_scores if new_raw_score > score) # Strict less than
    market_ivr = (lower_count / len(historical_raw_scores)) * 100.0
    return float(market_ivr)

def calculate_market_iv_risk(eod_csv_file_path: str = RISK_EOD_CSV_FILE, is_called_by_ai: bool = False) -> Optional[float]:
    """Calculates Market Implied Volatility (IV) based on EOD Raw Market Invest Scores."""
    if not os.path.exists(eod_csv_file_path):
        risk_logger.info(f"EOD data file {eod_csv_file_path} not found for Market IV calculation. Cannot calculate.")
        return None
    try:
        df_eod = pd.read_csv(eod_csv_file_path, on_bad_lines='skip')
        if df_eod.empty or 'Raw Market Invest Score (EOD)' not in df_eod.columns or 'Date' not in df_eod.columns:
            risk_logger.warning(f"EOD data file {eod_csv_file_path} is empty or missing required columns for Market IV.")
            return None

        df_eod['Date'] = pd.to_datetime(df_eod['Date'], errors='coerce')
        df_eod = df_eod.sort_values(by='Date', ascending=True).dropna(subset=['Date'])
        # Ensure EOD scores are numeric
        eod_scores_series = pd.to_numeric(df_eod['Raw Market Invest Score (EOD)'], errors='coerce').dropna()

        if len(eod_scores_series) < 21: # Need at least ~1 month (20-21 trading days) of EOD scores
            risk_logger.info(f"Insufficient EOD data points ({len(eod_scores_series)}) for Market IV calculation (need >= 21).")
            return None

        # Calculate daily changes in Raw MIS (EOD)
        # Use .diff() for change, then take absolute for magnitude
        daily_changes_eod_mis = eod_scores_series.diff().abs().dropna()

        if len(daily_changes_eod_mis) < 20 : # Need at least 20 changes for a 20-day average
            risk_logger.info(f"Insufficient daily changes ({len(daily_changes_eod_mis)}) for Market IV (need >= 20).")
            return None

        # Average magnitude of daily change over the last 20 trading days
        # This is analogous to an Average True Range (ATR) but on the MIS.
        average_daily_mis_change_magnitude = daily_changes_eod_mis.tail(20).mean()
        if pd.isna(average_daily_mis_change_magnitude):
            risk_logger.warning("Average daily MIS change magnitude is NaN. Cannot calculate Market IV.")
            return None

        # Original formula: ( ( (AvgChangeMagnitude / 10000) + 1) ^ 252 - 1) * 100
        # This formula annualizes the average daily change magnitude.
        # The division by 10000 seems to scale it down significantly before exponentiation.
        # This might be specific to the expected range/scale of Raw MIS.
        scaling_factor = 10000.0 # As per original formula logic
        expressed_percentage_daily_change = average_daily_mis_change_magnitude / scaling_factor
        growth_factor_annualized = (1 + expressed_percentage_daily_change) ** 252 # Annualize (252 trading days)
        market_iv_calculated = (growth_factor_annualized - 1) * 100 # Convert to percentage

        return float(market_iv_calculated)
    except Exception as e:
        risk_logger.exception(f"Error calculating Market IV from {eod_csv_file_path}:")
        return None

# --- Core R.I.S.K. Orchestration Function ---

async def perform_risk_calculations_singularity(is_eod_save: bool = False, is_called_by_ai: bool = False):
    """
    Performs one-time R.I.S.K. calculations, prints results (if not AI), saves data to CSV,
    and returns a summary dictionary.
    """
    global risk_persistent_signal, risk_signal_day # Manage global state for signal persistence
    # risk_logger should be globally defined and configured

    risk_logger.info(f"--- Singularity: Performing R.I.S.K. calculations cycle (EOD Save: {is_eod_save}) ---")

    results_summary = { # Initialize with N/A
        "general_score": 'N/A', "large_cap_score": 'N/A', "ema_risk_score": 'N/A',
        "combined_score": 'N/A', "spy_price": 'N/A', "vix_price": 'N/A',
        "momentum_recession_chance": 'N/A', "vix_recession_chance": 'N/A',
        "raw_market_invest_score": 'N/A', "market_invest_score": 'N/A', # Capped MIS
        "market_ivr": 'N/A', "market_iv_eod": 'N/A',
        "market_signal": risk_persistent_signal, # Start with current persistent signal
        "signal_date_info": f"(Since {risk_signal_day.strftime('%Y-%m-%d')})" if risk_signal_day and risk_persistent_signal != "Hold" else ""
    }

    general, large, ema_risk, combined, spy_p, vix_p = await calculate_risk_scores_singularity(is_called_by_ai=True)
    likelihood_ema_val = await asyncio.to_thread(calculate_recession_likelihood_ema_risk, is_called_by_ai=True)
    likelihood_vix_val = await asyncio.to_thread(calculate_recession_likelihood_vix_risk, vix_p, is_called_by_ai=True)

    uncapped_mis, capped_mis_signal, rounded_mis_display = None, None, None
    if likelihood_vix_val is not None and likelihood_ema_val is not None:
        uncapped_mis, capped_mis_signal, rounded_mis_display = await asyncio.to_thread(
            calculate_market_invest_score_risk, likelihood_vix_val, likelihood_ema_val, is_called_by_ai=True
        )
    else:
        risk_logger.warning("One or both recession likelihoods are None. Market Invest Score cannot be calculated.")

    market_ivr_val = None
    if uncapped_mis is not None: #calculate_market_ivr_risk uses RISK_CSV_FILE
        market_ivr_val = await asyncio.to_thread(calculate_market_ivr_risk, uncapped_mis, RISK_CSV_FILE, is_called_by_ai=True)

    # calculate_market_iv_risk uses RISK_EOD_CSV_FILE
    market_iv_val_eod = await asyncio.to_thread(calculate_market_iv_risk, RISK_EOD_CSV_FILE, is_called_by_ai=True)

    current_date_est = datetime.now(EST_TIMEZONE).date() # EST_TIMEZONE must be globally defined
    previous_capped_mis_from_csv = None
    if os.path.exists(RISK_CSV_FILE): # RISK_CSV_FILE must be globally defined
        try:
            df_hist_signal = pd.read_csv(RISK_CSV_FILE, on_bad_lines='skip')
            if not df_hist_signal.empty and 'Market Invest Score' in df_hist_signal.columns:
                valid_prev_mis_series = pd.to_numeric(df_hist_signal['Market Invest Score'], errors='coerce').dropna()
                if not valid_prev_mis_series.empty:
                    previous_capped_mis_from_csv = valid_prev_mis_series.iloc[-1]
        except Exception as e_read_sig:
            risk_logger.error(f"Error reading previous MIS from {RISK_CSV_FILE} for signal logic: {e_read_sig}")

    if capped_mis_signal is not None:
        if previous_capped_mis_from_csv is not None:
            if previous_capped_mis_from_csv < 50 and capped_mis_signal >= 50:
                risk_persistent_signal = "Buy"
                risk_signal_day = current_date_est
            elif previous_capped_mis_from_csv >= 50 and capped_mis_signal < 50:
                risk_persistent_signal = "Sell"
                risk_signal_day = current_date_est
        if risk_signal_day is None :
            risk_persistent_signal = "Buy" if capped_mis_signal >= 50 else "Sell"
            risk_signal_day = current_date_est
    
    signal_day_str_for_summary = f"(Since {risk_signal_day.strftime('%Y-%m-%d')})" if risk_signal_day and risk_persistent_signal != "Hold" else ""

    results_summary.update({
        "general_score": f"{general:.2f}" if general is not None else 'N/A',
        "large_cap_score": f"{large:.2f}" if large is not None else 'N/A',
        "ema_risk_score": f"{ema_risk:.2f}" if ema_risk is not None else 'N/A',
        "combined_score": f"{combined:.2f}" if combined is not None else 'N/A',
        "spy_price": f"{spy_p:.2f}" if spy_p is not None else 'N/A',
        "vix_price": f"{vix_p:.2f}" if vix_p is not None else 'N/A',
        "momentum_recession_chance": f"{likelihood_ema_val:.1f}%" if likelihood_ema_val is not None else 'N/A',
        "vix_recession_chance": f"{likelihood_vix_val:.1f}%" if likelihood_vix_val is not None else 'N/A',
        "raw_market_invest_score": f"{uncapped_mis:.2f}" if uncapped_mis is not None else 'N/A',
        "market_invest_score": f"{capped_mis_signal:.2f}" if capped_mis_signal is not None else 'N/A',
        "market_ivr": f"{market_ivr_val:.1f}%" if market_ivr_val is not None else 'N/A',
        "market_iv_eod": f"{market_iv_val_eod:.2f}%" if market_iv_val_eod is not None else 'N/A',
        "market_signal": risk_persistent_signal,
        "signal_date_info": signal_day_str_for_summary.strip()
    })

    timestamp_iso_utc = datetime.now(pytz.UTC).isoformat() # MODIFIED HERE
    main_csv_fieldnames = ['Timestamp', 'General Market Score', 'Large Market Cap Score', 'EMA Score', 'Combined Score',
                           'Live SPY Price', 'Live VIX Price', 'Momentum Based Recession Chance', 'VIX Based Recession Chance',
                           'Raw Market Invest Score', 'Market Invest Score', 'Market IVR', 'Market Signal', 'Signal Date']
    try:
        main_file_exists = os.path.exists(RISK_CSV_FILE) # RISK_CSV_FILE
        with open(RISK_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile_main:
            writer_main = csv.DictWriter(csvfile_main, fieldnames=main_csv_fieldnames)
            if not main_file_exists or os.path.getsize(RISK_CSV_FILE) == 0:
                writer_main.writeheader()
            row_to_write_main = {
                'Timestamp': timestamp_iso_utc,
                'General Market Score': results_summary["general_score"],
                'Large Market Cap Score': results_summary["large_cap_score"],
                'EMA Score': results_summary["ema_risk_score"],
                'Combined Score': results_summary["combined_score"],
                'Live SPY Price': results_summary["spy_price"],
                'Live VIX Price': results_summary["vix_price"],
                'Momentum Based Recession Chance': results_summary["momentum_recession_chance"],
                'VIX Based Recession Chance': results_summary["vix_recession_chance"],
                'Raw Market Invest Score': results_summary["raw_market_invest_score"],
                'Market Invest Score': results_summary["market_invest_score"],
                'Market IVR': results_summary["market_ivr"],
                'Market Signal': risk_persistent_signal,
                'Signal Date': risk_signal_day.strftime('%Y-%m-%d') if risk_signal_day else 'N/A'
            }
            writer_main.writerow(row_to_write_main)
        risk_logger.info(f"Data appended to main R.I.S.K. data file: {RISK_CSV_FILE}")
    except Exception as e_csv_main:
        risk_logger.exception(f"Error writing to main R.I.S.K. CSV file {RISK_CSV_FILE}:")

    if is_eod_save:
        if uncapped_mis is not None and market_iv_val_eod is not None:
            eod_csv_fieldnames = ['Date', 'Raw Market Invest Score (EOD)', 'Market IV (EOD)']
            try: # RISK_EOD_CSV_FILE must be globally defined
                eod_file_exists_check = os.path.exists(RISK_EOD_CSV_FILE)
                with open(RISK_EOD_CSV_FILE, 'a', newline='', encoding='utf-8') as eod_csvfile_out:
                    eod_writer_csv = csv.DictWriter(eod_csvfile_out, fieldnames=eod_csv_fieldnames)
                    if not eod_file_exists_check or os.path.getsize(RISK_EOD_CSV_FILE) == 0:
                        eod_writer_csv.writeheader()
                    eod_row_to_write = {
                        'Date': datetime.now(EST_TIMEZONE).strftime('%Y-%m-%d'), # EST_TIMEZONE
                        'Raw Market Invest Score (EOD)': f"{uncapped_mis:.2f}",
                        'Market IV (EOD)': f"{market_iv_val_eod:.2f}"
                    }
                    eod_writer_csv.writerow(eod_row_to_write)
                risk_logger.info(f"EOD data appended to EOD R.I.S.K. data file: {RISK_EOD_CSV_FILE}")
            except Exception as e_eod_csv_write:
                risk_logger.exception(f"Error writing to EOD R.I.S.K. CSV file {RISK_EOD_CSV_FILE}:")
        else:
            risk_logger.warning("Cannot save EOD data: Raw Market Invest Score or Market IV (EOD) is missing.")

    if not is_called_by_ai:
        print("\n--- R.I.S.K. Analysis Results (Console Output) ---")
        print(f"  General Market Score: {results_summary['general_score']}")
        print(f"  Large Market Cap Score: {results_summary['large_cap_score']}")
        print(f"  EMA Score (R.I.S.K. Specific): {results_summary['ema_risk_score']}")
        print(f"  Combined Score: {results_summary['combined_score']}")
        print("-" * 20 + " Contraction & Volatility " + "-" * 20)
        print(f"  Momentum Based Recession Chance (EMA): {results_summary['momentum_recession_chance']}")
        print(f"  VIX Based Recession Chance: {results_summary['vix_recession_chance']}")
        print(f"  Market Invest Score (for display): {rounded_mis_display if rounded_mis_display is not None else 'N/A'}")
        print(f"  Market IVR: {results_summary['market_ivr']}")
        print(f"  Market IV (based on EOD data): {results_summary['market_iv_eod']}")
        print(f"  Market Signal: {results_summary['market_signal']} {results_summary['signal_date_info']}")
        print("--- End of R.I.S.K. Analysis Console Output ---")

    return results_summary

# --- R.I.S.K. Graph Generation Function ---

async def generate_risk_graphs_singularity(is_called_by_ai: bool = False) -> dict:
    """Generates and saves historical RISK graphs for Singularity. Returns status and file list."""
    if not is_called_by_ai: print("\n--- Generating R.I.S.K. Historical Graphs ---")
    graph_files_generated_list = []
    status_message = "Graph generation initiated."

    if not os.path.exists(RISK_CSV_FILE):
        message = f"Error: Main data file '{RISK_CSV_FILE}' not found. Cannot generate graphs."
        if not is_called_by_ai: print(message)
        return {"status": "error", "message": message, "files": []}
    try:
        # This is a blocking call
        df_main_risk_hist = pd.read_csv(RISK_CSV_FILE, on_bad_lines='skip')
    except Exception as e_read_main_hist:
        message = f"Error reading '{RISK_CSV_FILE}': {e_read_main_hist}"
        if not is_called_by_ai: print(message)
        return {"status": "error", "message": message, "files": []}

    if df_main_risk_hist.empty:
        message = f"'{RISK_CSV_FILE}' is empty. No data to graph."
        if not is_called_by_ai: print(message)
        return {"status": "nodata", "message": message, "files": []}

    # Ensure Timestamp is datetime and sort
    df_main_risk_hist['Timestamp'] = pd.to_datetime(df_main_risk_hist['Timestamp'], errors='coerce').dt.tz_localize(None) # Remove tz for plotting if present
    df_main_risk_hist = df_main_risk_hist.sort_values(by='Timestamp').dropna(subset=['Timestamp'])

    # Common plot settings
    plt.style.use('dark_background')
    key_color = 'white'; grid_color = 'gray'; fig_size_tuple = (14, 7) # Slightly wider

    # Graph 1: Market Scores
    try:
        plt.figure(figsize=fig_size_tuple)
        score_cols_to_plot = ['General Market Score', 'Large Market Cap Score', 'EMA Score', 'Combined Score', 'Market Invest Score']
        for col_name in score_cols_to_plot:
            if col_name in df_main_risk_hist.columns:
                # Ensure data is numeric for plotting, coercing errors
                plt.plot(df_main_risk_hist['Timestamp'], pd.to_numeric(df_main_risk_hist[col_name], errors='coerce'), label=col_name.replace(' Score','S').replace(' Market','M'), linewidth=1.5) # Shorter labels
        plt.title('Historical Market Scores (R.I.S.K.)', color=key_color, fontsize=16)
        plt.xlabel('Timestamp', color=key_color); plt.ylabel('Score (0-100)', color=key_color)
        plt.legend(labelcolor=key_color, fontsize='small'); plt.grid(True, color=grid_color, linestyle=':')
        plt.tick_params(axis='x', colors=key_color, rotation=25); plt.tick_params(axis='y', colors=key_color)
        plt.tight_layout()
        filename1 = f"risk_market_scores_hist_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename1, facecolor='black'); graph_files_generated_list.append(filename1); plt.close()
    except Exception as e_g1_plot:
        if not is_called_by_ai: print(f"Error generating scores graph: {e_g1_plot}")

    # Graph 2: SPY & VIX Prices
    try:
        fig, ax1_spy = plt.subplots(figsize=fig_size_tuple)
        if 'Live SPY Price' in df_main_risk_hist.columns:
            ax1_spy.plot(df_main_risk_hist['Timestamp'], pd.to_numeric(df_main_risk_hist['Live SPY Price'], errors='coerce'), label='SPY Price', color='lime', linewidth=1.5)
        ax1_spy.set_ylabel('SPY Price ($)', color='lime'); ax1_spy.tick_params(axis='y', labelcolor='lime')
        ax1_spy.tick_params(axis='x', colors=key_color, rotation=25) # X-axis for primary
        ax2_vix = ax1_spy.twinx() # Share X-axis
        if 'Live VIX Price' in df_main_risk_hist.columns:
            ax2_vix.plot(df_main_risk_hist['Timestamp'], pd.to_numeric(df_main_risk_hist['Live VIX Price'], errors='coerce'), label='VIX Price', color='red', linewidth=1.5)
        ax2_vix.set_ylabel('VIX Price', color='red'); ax2_vix.tick_params(axis='y', labelcolor='red')
        plt.title('Historical SPY & VIX Prices (R.I.S.K.)', color=key_color, fontsize=16)
        # Combine legends
        lines1, labels1 = ax1_spy.get_legend_handles_labels()
        lines2, labels2 = ax2_vix.get_legend_handles_labels()
        if lines1 or lines2: ax2_vix.legend(lines1 + lines2, labels1 + labels2, loc='upper left', labelcolor=key_color, fontsize='small')
        ax1_spy.grid(True, color=grid_color, linestyle=':'); plt.tight_layout()
        filename2 = f"risk_spy_vix_prices_hist_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename2, facecolor='black'); graph_files_generated_list.append(filename2); plt.close()
    except Exception as e_g2_plot:
        if not is_called_by_ai: print(f"Error generating SPY/VIX graph: {e_g2_plot}")

    # Graph 3: Recession Chances
    try:
        plt.figure(figsize=fig_size_tuple)
        recession_cols_to_plot = ['Momentum Based Recession Chance', 'VIX Based Recession Chance']
        for col_name_rec in recession_cols_to_plot:
            if col_name_rec in df_main_risk_hist.columns:
                # Values might have '%' suffix, remove it before converting to numeric
                numeric_series_rec = pd.to_numeric(df_main_risk_hist[col_name_rec].astype(str).str.rstrip('%'), errors='coerce')
                plt.plot(df_main_risk_hist['Timestamp'], numeric_series_rec, label=col_name_rec.replace(' Chance','').replace(' Based',''), linewidth=1.5) # Shorter labels
        plt.title('Historical Recession Chances (R.I.S.K.)', color=key_color, fontsize=16)
        plt.xlabel('Timestamp', color=key_color); plt.ylabel('Recession Chance (%)', color=key_color)
        plt.legend(labelcolor=key_color, fontsize='small'); plt.grid(True, color=grid_color, linestyle=':')
        plt.tick_params(axis='x', colors=key_color, rotation=25); plt.tick_params(axis='y', colors=key_color)
        plt.ylim(0, 105); plt.tight_layout() # Y-axis limit 0-100 (with padding)
        filename3 = f"risk_recession_chances_hist_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename3, facecolor='black'); graph_files_generated_list.append(filename3); plt.close()
    except Exception as e_g3_plot:
        if not is_called_by_ai: print(f"Error generating recession chances graph: {e_g3_plot}")

    # Graph 4: Market IVR (from main RISK_CSV_FILE)
    try:
        if 'Market IVR' in df_main_risk_hist.columns:
            plt.figure(figsize=fig_size_tuple)
            numeric_ivr_series = pd.to_numeric(df_main_risk_hist['Market IVR'].astype(str).str.rstrip('%'), errors='coerce')
            plt.plot(df_main_risk_hist['Timestamp'], numeric_ivr_series, label='Market IVR', color='cyan', linewidth=2)
            plt.title('Historical Market IVR (R.I.S.K.)', color=key_color, fontsize=16)
            plt.xlabel('Timestamp', color=key_color); plt.ylabel('Market IVR (%)', color=key_color)
            plt.legend(labelcolor=key_color, fontsize='small'); plt.grid(True, color=grid_color, linestyle=':')
            plt.tick_params(axis='x', colors=key_color, rotation=25); plt.tick_params(axis='y', colors=key_color)
            plt.tight_layout(); filename4 = f"risk_market_ivr_hist_{uuid.uuid4().hex[:6]}.png"; plt.savefig(filename4, facecolor='black'); graph_files_generated_list.append(filename4); plt.close()
    except Exception as e_g4_plot:
        if not is_called_by_ai: print(f"Error generating IVR graph: {e_g4_plot}")

    # Graph 5: Market IV (from EOD RISK_EOD_CSV_FILE)
    df_eod_risk_hist = None
    if os.path.exists(RISK_EOD_CSV_FILE):
        try:
            df_eod_risk_hist = pd.read_csv(RISK_EOD_CSV_FILE, on_bad_lines='skip') # Blocking
            if not df_eod_risk_hist.empty and 'Market IV (EOD)' in df_eod_risk_hist.columns and 'Date' in df_eod_risk_hist.columns:
                df_eod_risk_hist['Date'] = pd.to_datetime(df_eod_risk_hist['Date'], errors='coerce')
                df_eod_risk_hist = df_eod_risk_hist.sort_values(by='Date').dropna(subset=['Date'])
                plt.figure(figsize=fig_size_tuple)
                numeric_eod_iv_series = pd.to_numeric(df_eod_risk_hist['Market IV (EOD)'].astype(str).str.rstrip('%'), errors='coerce') # Remove % if present
                plt.plot(df_eod_risk_hist['Date'], numeric_eod_iv_series, label='Market IV (EOD)', color='fuchsia', linewidth=2)
                plt.title('Historical Market IV (EOD - R.I.S.K.)', color=key_color, fontsize=16)
                plt.xlabel('Date', color=key_color); plt.ylabel('Market IV (%)', color=key_color) # Assume it's a percentage
                plt.legend(labelcolor=key_color, fontsize='small'); plt.grid(True, color=grid_color, linestyle=':')
                plt.tick_params(axis='x', colors=key_color, rotation=25); plt.tick_params(axis='y', colors=key_color)
                plt.tight_layout(); filename5 = f"risk_market_iv_eod_hist_{uuid.uuid4().hex[:6]}.png"; plt.savefig(filename5, facecolor='black'); graph_files_generated_list.append(filename5); plt.close()
        except Exception as e_g5_plot:
            if not is_called_by_ai: print(f"Error generating EOD Market IV graph: {e_g5_plot}")
    elif not is_called_by_ai:
        print(f"Note: EOD data file '{RISK_EOD_CSV_FILE}' not found. Skipping Market IV (EOD) graph.")


    if graph_files_generated_list:
        status_message = f"Successfully generated {len(graph_files_generated_list)} R.I.S.K. history graphs."
        if not is_called_by_ai:
            print("\nGenerated R.I.S.K. graph files (check local directory):")
            for fname in graph_files_generated_list: print(f"  - {fname}")
        return {"status": "success", "message": status_message, "files": graph_files_generated_list}
    else:
        status_message = "No R.I.S.K. history graphs were generated, possibly due to missing data or errors during plotting."
        if not is_called_by_ai: print(status_message)
        return {"status": "no_graphs", "message": status_message, "files": []}


# --- R.I.S.K. Command Handlers ---

async def handle_risk_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """
    Handles the /risk command for both CLI and AI.
    AI call expects ai_params={"assessment_type": "standard" or "eod"}.
    """
    # if not is_called_by_ai: print("\n--- /risk Command ---") # AI handler will announce

    is_eod_run_flag = False
    if ai_params: # AI Call
        assessment_type_ai = ai_params.get("assessment_type", "standard").lower()
        if assessment_type_ai == "eod":
            is_eod_run_flag = True
            # if not is_called_by_ai: print("AI: Performing End-of-Day R.I.S.K. calculation and save.") # AI handler makes announcements
        # else: # Standard run
            # if not is_called_by_ai: print("AI: Performing standard R.I.S.K. calculation and save.")
    else: # CLI Path
        if args and args[0].lower() == 'eod':
            is_eod_run_flag = True
            if not is_called_by_ai: print("CLI: Performing End-of-Day R.I.S.K. calculation and save.")
        # else: # Standard CLI run (no args or non-'eod' arg)
            # if not is_called_by_ai: print("CLI: Performing standard R.I.S.K. calculation and save.")

    # perform_risk_calculations_singularity is async
    risk_results_dict = await perform_risk_calculations_singularity(
        is_eod_save=is_eod_run_flag,
        is_called_by_ai=is_called_by_ai # This controls prints within perform_risk_calculations
    )

    if is_called_by_ai: # Prepare summary for AI
        if risk_results_dict:
            summary_parts_ai = [f"R.I.S.K. Analysis Complete (EOD Run: {is_eod_run_flag})."]
            summary_parts_ai.append(f"Signal: {risk_results_dict.get('market_signal', 'N/A')} {risk_results_dict.get('signal_date_info', '').strip()}.")
            summary_parts_ai.append(f"Market Invest Score: {risk_results_dict.get('market_invest_score', 'N/A')}.")
            summary_parts_ai.append(f"Combined Score: {risk_results_dict.get('combined_score', 'N/A')}.")
            summary_parts_ai.append(f"Market IVR: {risk_results_dict.get('market_ivr', 'N/A')}.")
            # Add more key scores if desired by AI
            return " ".join(summary_parts_ai)
        else:
            return "Error: R.I.S.K. analysis performed but no summary data was returned to AI."
    else:
        return None # For CLI, output is handled by perform_risk_calculations_singularity directly


async def handle_history_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """
    Handles the /history command for Singularity to generate R.I.S.K. graphs.
    Returns a summary string for AI. args and ai_params are not used currently but kept for consistency.
    """
    # if not is_called_by_ai: print("\n--- /history Command (R.I.S.K. Graphs) ---") # AI Handler announces

    # generate_risk_graphs_singularity is async
    result_dict_graphs = await generate_risk_graphs_singularity(is_called_by_ai=is_called_by_ai)

    if is_called_by_ai: # Prepare summary for AI
        status = result_dict_graphs.get("status", "unknown")
        message = result_dict_graphs.get("message", "No specific message.")
        files = result_dict_graphs.get("files", [])

        if status == "success" and files:
            return f"Successfully generated {len(files)} R.I.S.K. history graphs: {', '.join(files)}."
        elif status == "success": # Success but no files reported (should not happen if files are key)
            return "R.I.S.K. history graph generation reported success, but no filenames were returned."
        else: # Error, nodata, or no_graphs
            return f"R.I.S.K. history graph generation: {message}"
    else:
        return None # For CLI, output is handled by generate_risk_graphs_singularity directly

# --- Quickscore Command ---
async def handle_quickscore_command(args: List[str], ai_params: Optional[Dict]=None, is_called_by_ai: bool = False):
    if not is_called_by_ai: print("\n--- /quickscore Command ---")
    ticker_qs = None
    if ai_params: ticker_qs = ai_params.get("ticker")
    elif args: ticker_qs = args[0].upper()

    if not ticker_qs:
        msg = "Usage: /quickscore <ticker> or AI must provide ticker."
        if not is_called_by_ai: print(msg)
        return f"Error: {msg}" if is_called_by_ai else None

    if not is_called_by_ai: print(f"Processing /quickscore for {ticker_qs}...")
    scores_qs, graphs_qs_files, live_price_qs_display = {}, [], "N/A"
    sensitivity_map = {1: 'Weekly (5Y)', 2: 'Daily (1Y)', 3: 'Hourly (6M)'}

    for sens_key, sens_name in sensitivity_map.items():
        # if not is_called_by_ai: print(f"  Calculating for {sens_name}...")
        live_p, ema_inv = await calculate_ema_invest(ticker_qs, sens_key, is_called_by_ai=is_called_by_ai)
        scores_qs[sens_key] = f"{ema_inv:.2f}%" if ema_inv is not None else "N/A"
        if live_p is not None and sens_key == 2: live_price_qs_display = f"${live_p:.2f}"
        # if not is_called_by_ai: print(f"  Generating graph for {sens_name}...")
        graph_file = await asyncio.to_thread(plot_ticker_graph, ticker_qs, sens_key, is_called_by_ai=is_called_by_ai)
        graphs_qs_files.append(f"{sens_name}: {graph_file if graph_file else 'Failed'}")

    if not is_called_by_ai: # Print results for CLI
        print("\n--- /quickscore Results ---")
        print(f"Ticker: {ticker_qs}\nLive Price (Daily): {live_price_qs_display}\nInvest Scores:")
        for sk, sn in sensitivity_map.items(): print(f"  {sn}: {scores_qs.get(sk, 'N/A')}")
        print("\nGenerated Graphs:"); [print(f"  {g}") for g in graphs_qs_files]
        print("\n/quickscore analysis complete.")

    summary = f"Quickscore for {ticker_qs}: Price {live_price_qs_display}. Scores: "
    summary += ", ".join([f"{sensitivity_map[sk]} {scores_qs.get(sk,'N/A')}" for sk in sensitivity_map]) + ". "
    summary += "Graphs: " + ", ".join([g for g in graphs_qs_files if "Failed" not in g]) + "."
    return summary


# --- AI Tool Definitions (FunctionDeclarations) ---
# These must match the functions you want the AI to be able to call.
# Ensure the names here match the keys in AVAILABLE_PYTHON_FUNCTIONS.

# In handle_ai_prompt, define the new tool near the others

briefing_tool = FunctionDeclaration(
    name="handle_briefing_command",
    description="Generates and returns a comprehensive daily market briefing. It summarizes key market indicators (SPY, VIX), risk scores, top/bottom movers in the S&P 500, breakout stock activity, and performance of a user's watchlist. This is a one-stop tool for a full market snapshot.",
    parameters=None # No parameters needed from the AI
)

spear_analysis_tool = FunctionDeclaration(
    name="handle_spear_command",
    description="Runs the SPEAR (Stock Performance Expectation and Analysis Report) model to predict a stock's percentage change around its upcoming earnings report. Requires multiple financial and market sentiment inputs.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "The stock ticker symbol to analyze, e.g., 'NVDA'."},
            "sector_relevance": {"type": "number", "description": "The relevance of the stock's sector to the overall market (1-5)."},
            "stock_relevance": {"type": "number", "description": "The relevance of the stock to its own sector (1-5)."},
            "hype": {"type": "number", "description": "The hype value surrounding the stock, from -1 (very negative) to 1 (very positive)."},
            "earnings_date": {"type": "string", "description": "The upcoming earnings date in YYYY-MM-DD format."},
            "earnings_time": {"type": "string", "description": "The time of the earnings report: 'p' for Pre-Market or 'a' for After Hours.", "enum": ["p", "a"]},
        },
        "required": ["ticker", "sector_relevance", "stock_relevance", "hype", "earnings_date", "earnings_time"]
    }
)

breakout_command_tool = FunctionDeclaration(
    name="handle_breakout_command",
    description="Handles breakout stock analysis. Action 'run' performs a new analysis, compares it to the previous run, and returns a detailed breakdown of current, newly added, and removed stocks. Action 'save' archives the most recent run's data for a specified date.",
    parameters={"type": "object", "properties": {
        "action": {"type": "string", "description": "Choose 'run' to find new breakout stocks and see changes, or 'save' to archive the last run's data.", "enum": ["run", "save"]},
        "date_to_save": {"type": "string", "description": "Required for 'save' action: Date in MM/DD/YYYY format. If user says 'today', use the current date."}
    }, "required": ["action"]}
)

quickscore_tool = FunctionDeclaration(
    name="handle_quickscore_command",
    description="Performs a quick analysis for a SINGLE, SPECIFIC stock ticker. Calculates various EMA-based investment scores and generates price/EMA graphs. Returns a string summary of scores and graph locations.",
    parameters={"type": "object", "properties": {
        "ticker": {"type": "string", "description": "The specific stock ticker symbol to analyze, e.g., 'AAPL', 'MSFT'."}
    }, "required": ["ticker"]}
)

market_command_tool = FunctionDeclaration(
    name="handle_market_command",
    description="Provides a general overview of the S&P 500 market or saves its full data. Action 'display' shows top/bottom S&P 500 stocks and SPY score based on EMA sensitivity (1:Weekly, 2:Daily, 3:Hourly). Action 'save' saves comprehensive S&P 500 market data for the chosen sensitivity and date. Returns a summary string.",
    parameters={"type": "object", "properties": {
        "action": {"type": "string", "description": "Choose 'display' to show S&P 500 market scores, or 'save' to save detailed S&P 500 market data.", "enum": ["display", "save"]},
        "sensitivity": {"type": "integer", "description": "EMA Sensitivity: 1 (Weekly), 2 (Daily), 3 (Hourly). If user requests display and is vague, default to 2 (Daily). Required for both actions."},
        "date_str": {"type": "string", "description": "Required for 'save' action: Date in MM/DD/YYYY format. If user says 'today', use current date."}
    }, "required": ["action", "sensitivity"]}
)

risk_assessment_tool = FunctionDeclaration(
    name="handle_risk_command",
    description="Performs a comprehensive R.I.S.K. module assessment of the overall market. Calculates various scores (General, Large Cap, EMA, Combined, Market Invest Score, IVR, IV), recession likelihoods, and determines a market signal (Buy/Sell/Hold). Use 'assessment_type':'eod' for end-of-day specific calculations and EOD data saving. Returns a dictionary of key scores and the signal.",
    parameters={"type": "object", "properties": {
        "assessment_type": {"type": "string", "description": "Specify 'standard' for a regular risk assessment, or 'eod' for an end-of-day assessment and EOD-specific data save. Default to 'standard' if user is vague.", "enum": ["standard", "eod"]}
    }, "required": ["assessment_type"]}
)

generate_history_graphs_tool = FunctionDeclaration(
    name="handle_history_command",
    description="Generates and saves a series of historical graphs for the R.I.S.K. module, including scores, SPY/VIX prices, recession chances, IVR, and IV. The function reports on the success and filenames of generated graphs.",
    parameters=None
)

custom_command_tool = FunctionDeclaration(
    name="handle_custom_command",
    description=(
        "Manages custom portfolios. Action 'run_existing_portfolio' runs an existing portfolio configuration "
        "(can be tailored with 'total_value', 'use_fractional_shares') AND AUTOMATICALLY SAVES/OVERWRITES its detailed "
        "run output to its CSV file, making it the new baseline. Action 'save_portfolio_data' saves *legacy combined "
        "percentage data*. "
        "IMPORTANT FOR 'RUN AND COMPARE' REQUESTS: If a user asks to 'run portfolio X and then compare it to its previous state', "
        "you should first use 'get_comparison_for_custom_portfolio' to get the comparison against the true prior state on disk. "
        "Then, AFTER that comparison is done, if the user wants to persist the new run, call this tool ('handle_custom_command') "
        "with action 'run_existing_portfolio' and the new parameters to actually save the new run as the current baseline."
    ),
    parameters={"type": "object", "properties": {
        "action": {"type": "string", "description": "Specify 'run_existing_portfolio' to analyze it and save/overwrite its run output CSV, or 'save_portfolio_data' for legacy combined % save.", "enum": ["run_existing_portfolio", "save_portfolio_data"]},
        "portfolio_code": {"type": "string", "description": "The code/name of the custom portfolio."},
        "tailor_to_value": {"type": "boolean", "description": "For 'run_existing_portfolio': Optional. Set to true to tailor results to a specific 'total_value'."},
        "total_value": {"type": "number", "description": "For 'run_existing_portfolio' with tailoring: The monetary value to tailor the portfolio analysis to."},
        "use_fractional_shares": {"type": "boolean", "description": "For 'run_existing_portfolio': Optional. If true, fractional shares will be used. If not specified, respects portfolio's saved configuration."},
        "date_to_save": {"type": "string", "description": "For 'save_portfolio_data' (legacy save): Date MM/DD/YYYY. If user says 'today', use current date."}
    }, "required": ["action", "portfolio_code"]}
)

cultivate_analysis_tool = FunctionDeclaration(
    name="handle_cultivate_command",
    description="Runs the complex Cultivate portfolio analysis (Code 'A' for Screener-based, 'B' for S&P 500-based) for a given portfolio value and fractional shares preference. Can also save the generated combined data if 'action' is 'save_data' for a 'date_to_save'. Returns a summary of the analysis or save action.",
    parameters={"type": "object", "properties": {
        "cultivate_code": {"type": "string", "description": "The Cultivate strategy code: 'A' (Screener) or 'B' (S&P 500).", "enum": ["A", "B"]},
        "portfolio_value": {"type": "number", "description": "The total portfolio value for the Cultivate analysis (e.g., 10000, 50000)."},
        "use_fractional_shares": {"type": "boolean", "description": "Whether to use fractional shares (true/false). Defaults to false if not specified."},
        "action": {"type": "string", "description": "Optional. 'run_analysis' (default) to perform and display analysis, or 'save_data' to run and then save the combined data.", "enum": ["run_analysis", "save_data"]},
        "date_to_save": {"type": "string", "description": "Optional. Date (MM/DD/YYYY) to save data for. Required if 'action' is 'save_data'. If 'today', use current date."}
    }, "required": ["cultivate_code", "portfolio_value", "use_fractional_shares"]}
)

invest_analysis_tool = FunctionDeclaration(
    name="handle_invest_command",
    description="Analyzes multiple user-defined stock groups (sub-portfolios) with specified tickers and percentage weights. Calculates investment scores based on EMA sensitivity and amplification. Can optionally tailor the final combined portfolio to a total value. Sum of weights for all sub-portfolios must be 100%. Returns a summary.",
    parameters={"type": "object", "properties": {
        "ema_sensitivity": {"type": "integer", "description": "EMA sensitivity: 1 (Weekly), 2 (Daily), or 3 (Hourly)."},
        "amplification": {"type": "number", "description": "Amplification factor (e.g., 0.5, 1.0, 2.0)."},
        "sub_portfolios": {"type": "array", "description": "A list of sub-portfolios. Each is an object with 'tickers' (comma-separated string like 'AAPL,MSFT') and 'weight' (number, e.g., 60 for 60%). Sum of all weights must be 100.", "items": {"type": "object", "properties": {"tickers": {"type": "string"}, "weight": {"type": "number"}}, "required": ["tickers", "weight"]}},
        "tailor_to_value": {"type": "boolean", "description": "Optional. True to tailor. Default false."},
        "total_value": {"type": "number", "description": "Optional. Total value for tailoring. Required if 'tailor_to_value' is true."},
        "use_fractional_shares": {"type": "boolean", "description": "Optional. For tailoring. Default false."}
    }, "required": ["ema_sensitivity", "amplification", "sub_portfolios"]}
)

handle_assess_tool = FunctionDeclaration(
    name="handle_assess_command",
    description="Performs specific financial assessments based on an 'assess_code'. Code 'A' for Stock Volatility (needs tickers, timeframe, risk_tolerance). Code 'B' for Manual Portfolio risk (needs holdings, backtest_period). Code 'C' for Custom Saved Portfolio risk (needs portfolio_code, value, backtest_period). Code 'D' for Cultivate Portfolio risk (needs cultivate_code, value, frac_shares, backtest_period). Returns a summary.",
    parameters={"type": "object", "properties": {
        "assess_code": {"type": "string", "description": "Assessment type: 'A' (Stock Volatility), 'B' (Manual Portfolio), 'C' (Custom Portfolio Risk), 'D' (Cultivate Portfolio Risk).", "enum": ["A", "B", "C", "D"]},
        "tickers_str": {"type": "string", "description": "For assess_code 'A': Comma-separated tickers e.g., 'AAPL,MSFT'."},
        "timeframe_str": {"type": "string", "description": "For assess_code 'A': Analysis timeframe ('1Y','3M','1M'). Defaults to '1Y'.", "enum": ["1Y", "3M", "1M"]},
        "risk_tolerance": {"type": "integer", "description": "For assess_code 'A': User's risk tolerance (1-5). Defaults to 3."},
        "backtest_period_str": {"type": "string", "description": "For assess_codes 'B','C','D': Backtesting period (e.g.,'1y','3y','5y','10y'). Defaults vary per code if not specified by user (e.g., B:1y, C:3y, D:5y)."},
        "manual_portfolio_holdings": {"type": "array", "description": "For assess_code 'B': List of portfolio holdings. Stocks: {'ticker':'SYMBOL','shares':X.Y}. Cash: {'ticker':'Cash','value':Z.Z}", "items": {"type": "object", "properties": {"ticker": {"type": "string"}, "shares": {"type": "number"}, "value": {"type": "number"}}, "required": ["ticker"]}},
        "custom_portfolio_code": {"type": "string", "description": "For assess_code 'C': Code/name of a saved custom portfolio."},
        "value_for_assessment": {"type": "number", "description": "For assess_codes 'B' (optional, if providing values not shares), 'C', and 'D': Total portfolio value for assessment (e.g., 10000)."},
        "cultivate_portfolio_code": {"type": "string", "description": "For assess_code 'D': Cultivate strategy code ('A' or 'B').", "enum": ["A", "B"]},
        "use_fractional_shares": {"type": "boolean", "description": "For assess_code 'D': Whether to use fractional shares (true/false). Defaults to false."}
    }, "required": ["assess_code"]}
)

get_comparison_for_custom_portfolio_tool = FunctionDeclaration(
    name="get_comparison_for_custom_portfolio",
    description=(
        "Performs a full comparison cycle for a custom portfolio. It loads the previously saved run, runs the portfolio fresh with new parameters, provides a detailed comparison, and then saves the new run, overwriting the old one. Use this when the user wants to compare a portfolio to its last saved state."
    ),
    parameters={"type":"object", "properties":{
        "portfolio_code":{"type":"string","description":"The code/name of the custom portfolio to compare and then update."},
        "value_for_assessment":{"type":"number","description":"Optional. The monetary value to use for the 'fresh run' part of the comparison if tailoring is desired. If not provided, the fresh run might be untailored."},
        "use_fractional_shares_override":{"type":"boolean","description":"Optional. For the 'fresh run' part: true/false to override the portfolio's default fractional share setting for the comparison. If null/not provided, respects the portfolio's saved configuration."}
    },"required":["portfolio_code"]}
)

# --- AVAILABLE_PYTHON_FUNCTIONS Dictionary ---
# This dictionary maps the AI tool names to your actual Python functions.
# Ensure all functions listed in FunctionDeclarations are present here.
AVAILABLE_PYTHON_FUNCTIONS: dict[str, callable] = {
    "handle_briefing_command": handle_briefing_command,
    "handle_breakout_command": handle_breakout_command,
    "handle_quickscore_command": handle_quickscore_command,
    "handle_market_command": handle_market_command,
    "handle_risk_command": handle_risk_command,
    "handle_history_command": handle_history_command,
    "handle_custom_command": handle_custom_command,
    "handle_cultivate_command": handle_cultivate_command,
    "handle_invest_command": handle_invest_command,
    "handle_assess_command": handle_assess_command,
    "get_comparison_for_custom_portfolio": get_comparison_for_custom_portfolio,
    "handle_spear_command": handle_spear_command,
}

async def handle_ai_prompt(user_new_message: str, is_new_session: bool = False, original_session_request: Optional[str] = None):
    global AI_CONVERSATION_HISTORY, gemini_model, AVAILABLE_PYTHON_FUNCTIONS, AI_INTERNAL_STEP_COUNT, CURRENT_AI_SESSION_ORIGINAL_REQUEST

    if not gemini_model:
        print("Error: Gemini model is not configured. Cannot proceed with AI chat.")
        return "Error: Gemini model not configured for AI prompt handling."
    if 'AVAILABLE_PYTHON_FUNCTIONS' not in globals() or not AVAILABLE_PYTHON_FUNCTIONS:
        print("Error: AVAILABLE_PYTHON_FUNCTIONS dictionary is not defined or empty.")
        return "Error: System configuration issue (AVAILABLE_PYTHON_FUNCTIONS)."

    # MODIFICATION: Update the list of tools for the AI
    try:
        all_gemini_tools = Tool(function_declarations=[
            briefing_tool, # NEW
            breakout_command_tool,
            quickscore_tool,
            market_command_tool,
            risk_assessment_tool,
            generate_history_graphs_tool,
            custom_command_tool,
            cultivate_analysis_tool,
            invest_analysis_tool,
            handle_assess_tool,
            get_comparison_for_custom_portfolio_tool,
            spear_analysis_tool
        ])
    except NameError as e:
        print(f"Error: A FunctionDeclaration object for a tool is not defined globally: {e}")
        return f"Error: System configuration issue (Tool FunctionDeclaration missing: {e})."
    except Exception as e_tool_creation:
        print(f"Error creating Tool object: {e_tool_creation}")
        return "Error: System configuration issue (Tool creation failed)."
    
    # [ The rest of the handle_ai_prompt function remains unchanged ]
    # ...
    # This function's logic for session management, tool calling, and response synthesis is complex but does not need to be modified
    # to add a new tool, as long as the tool is added to the `all_gemini_tools` list and `AVAILABLE_PYTHON_FUNCTIONS` dictionary.
    # For brevity, the rest of this long function is omitted.
    if is_new_session:
        AI_CONVERSATION_HISTORY.clear()
        AI_INTERNAL_STEP_COUNT = 0
        CURRENT_AI_SESSION_ORIGINAL_REQUEST = original_session_request if original_session_request else user_new_message
        
        base_system_prompt_template = load_system_prompt()
        current_date_val = datetime.now().strftime('%B %d, %Y')
        current_date_mmddyyyy_val = datetime.now().strftime('%m/%d/%Y')
        try:
            # MODIFICATION: Add a more robust format call to prevent ValueErrors
            class SafeDict(dict):
                def __missing__(self, key):
                    return f"{{{key}}}" # Keep missing keys as placeholders, not for formatting
            
            formatting_dict = SafeDict(
                current_date_for_ai_prompt=current_date_val,
                current_date_mmddyyyy_for_ai_prompt=current_date_mmddyyyy_val
            )
            formatted_system_prompt_text = base_system_prompt_template.format_map(formatting_dict)

        except Exception as e_format_sys:
            # This block should now be much harder to trigger with format_map
            print(f"Error during system prompt formatting: {e_format_sys}. Check system_prompt.txt or default prompt string.")
            formatted_system_prompt_text = "You are a helpful AI assistant. Today's date is {current_date_for_ai_prompt}.".format(current_date_for_ai_prompt=current_date_val)

        AI_CONVERSATION_HISTORY.append({"role": "user", "parts": [{"text": formatted_system_prompt_text}]})
        AI_CONVERSATION_HISTORY.append({"role": "model", "parts": [{"text": "Okay, I understand my role, guidelines, and today's date. How can I assist you today?"}]})

    AI_CONVERSATION_HISTORY.append({"role": "user", "parts": [{"text": user_new_message}]})

    max_internal_turns = 10
    executed_tool_calls_in_current_interaction = set()
    functions_were_called_in_this_interaction = False
    final_text_response_from_ai = ""
    repeat_detected_in_loop = False
    tool_name_that_repeated = ""


    stop_main_spinner_event = asyncio.Event()
    main_spinner_task = asyncio.create_task(_continuous_spinner_animation(stop_main_spinner_event, "AI is processing request..."))

    try:
        for turn_num in range(max_internal_turns):
            AI_INTERNAL_STEP_COUNT = turn_num
            try:
                gemini_response = await asyncio.to_thread(
                    gemini_model.generate_content,
                    contents=AI_CONVERSATION_HISTORY,
                    tools=[all_gemini_tools],
                    tool_config={"function_calling_config": {"mode": "ANY"}}
                )
            except Exception as e_gen_content_api:
                final_text_response_from_ai = f"Gemini API Error during content generation: {e_gen_content_api}"
                break

            gemini_candidate = gemini_response.candidates[0] if gemini_response.candidates else None
            if not gemini_candidate:
                final_text_response_from_ai = "AI returned no response candidate from API."
                break

            model_turn_history_parts_to_add = []
            function_call_to_run_this_turn = None
            text_from_model_in_turn = ""

            if gemini_candidate.content and gemini_candidate.content.parts:
                for part_object in gemini_candidate.content.parts:
                    if part_object.function_call and part_object.function_call.name:
                        fc_obj = part_object.function_call
                        function_call_to_run_this_turn = fc_obj
                        serializable_args = {k: (v if isinstance(v, (str, int, float, bool, list, dict, type(None))) else str(v)) for k, v in fc_obj.args.items()}
                        model_turn_history_parts_to_add.append({
                            "function_call": {"name": fc_obj.name, "args": serializable_args }
                        })
                        break
                    elif hasattr(part_object, 'text') and part_object.text:
                        text_from_model_in_turn += part_object.text

            if model_turn_history_parts_to_add:
                AI_CONVERSATION_HISTORY.append({"role": "model", "parts": model_turn_history_parts_to_add})
            elif text_from_model_in_turn.strip():
                 AI_CONVERSATION_HISTORY.append({"role": "model", "parts": [{"text": text_from_model_in_turn.strip()}]})

            if function_call_to_run_this_turn:
                functions_were_called_in_this_interaction = True
                tool_name_current = function_call_to_run_this_turn.name
                tool_args_dict_current = dict(function_call_to_run_this_turn.args)
                
                current_call_signature_tuple = (tool_name_current, make_hashable(tool_args_dict_current))
                if current_call_signature_tuple in executed_tool_calls_in_current_interaction:
                    tool_name_that_repeated = tool_name_current
                    error_msg_for_history = (f"System detected an attempt by AI to re-run tool '{tool_name_current}' "
                                             f"with identical parameters within the same user request processing. "
                                             f"This specific tool call was aborted by the system to prevent a loop. AI should try a different approach or summarize.")
                    AI_CONVERSATION_HISTORY.append({"role": "tool", "parts": [{"function_response": {"name": tool_name_current, "response": {"error": "Repetitive tool call proposed by AI.", "status": "aborted_by_system_due_to_repetition", "details": error_msg_for_history}}}]})
                    repeat_detected_in_loop = True
                    break

                executed_tool_calls_in_current_interaction.add(current_call_signature_tuple)

                if tool_name_current in AVAILABLE_PYTHON_FUNCTIONS:
                    python_function_to_execute = AVAILABLE_PYTHON_FUNCTIONS[tool_name_current]
                    tool_result_from_execution = f"Error: Tool '{tool_name_current}' did not return a recognizable value."
                    try:
                        if asyncio.iscoroutinefunction(python_function_to_execute):
                            tool_result_from_execution = await python_function_to_execute(args=[], ai_params=tool_args_dict_current, is_called_by_ai=True)
                        else:
                            tool_result_from_execution = await asyncio.to_thread(python_function_to_execute, args=[], ai_params=tool_args_dict_current, is_called_by_ai=True)
                    except Exception as e_tool_execution_error:
                        risk_logger.error(f"Tool execution error for {tool_name_current} with params {tool_args_dict_current}: {e_tool_execution_error}", exc_info=True)
                        tool_result_from_execution = f"Error during execution of Python function for tool '{tool_name_current}': {str(e_tool_execution_error)}"
                    
                    response_payload_for_history = {}
                    if isinstance(tool_result_from_execution, (dict, list)): response_payload_for_history = {"result": tool_result_from_execution, "status": "success_structured_response"}
                    elif isinstance(tool_result_from_execution, str): response_payload_for_history = {"summary_from_tool": tool_result_from_execution, "status": "success_text_response"}
                    elif tool_result_from_execution is None: response_payload_for_history = {"status": "success_no_explicit_output", "message": f"Tool '{tool_name_current}' executed and returned None."}
                    else: response_payload_for_history = {"result_str": str(tool_result_from_execution), "status": "success_unknown_return_type"}
                    
                    try:
                        MAX_TOOL_RESPONSE_SIZE_BYTES = 30000
                        json_str_payload = json.dumps(response_payload_for_history)
                        if len(json_str_payload.encode('utf-8')) > MAX_TOOL_RESPONSE_SIZE_BYTES:
                            keys_to_try_summarize = ["comparison_outcome_summary", "summary_from_tool", "message", "error_details", "error"]
                            found_summary_for_trunc = None
                            for key_s in keys_to_try_summarize:
                                if isinstance(response_payload_for_history.get(key_s), str):
                                    found_summary_for_trunc = response_payload_for_history[key_s]
                                    break
                            if found_summary_for_trunc:
                                 response_payload_for_history = {"status": "truncated_due_to_size", "summary_from_tool": found_summary_for_trunc[:1500]+"..."}
                            else:
                                 response_payload_for_history = {"status": "truncated_due_to_size", "error": "Original tool output was too large and a brief summary field was not found or suitable for truncation."}
                    except TypeError as e_json_dump:
                        response_payload_for_history = {"status": "error_serialization_tool_response", "error": f"Tool output for {tool_name_current} was not JSON serializable: {e_json_dump}. Original type: {type(tool_result_from_execution)}"}
                    
                    AI_CONVERSATION_HISTORY.append({"role": "tool", "parts": [{"function_response": {"name": tool_name_current, "response": response_payload_for_history}}]})
                else:
                    error_unknown_tool_msg = f"Error: AI proposed an unknown function '{tool_name_current}' which is not in AVAILABLE_PYTHON_FUNCTIONS."
                    AI_CONVERSATION_HISTORY.append({"role": "tool", "parts": [{"function_response": {"name": tool_name_current, "response": {"error": error_unknown_tool_msg}}}]})
                    final_text_response_from_ai = "AI proposed an unknown tool. Cannot proceed with this path."
                    break
            elif text_from_model_in_turn.strip():
                final_text_response_from_ai = text_from_model_in_turn.strip()
                break
            else:
                if not functions_were_called_in_this_interaction:
                    final_text_response_from_ai = "AI did not provide a response or call a function."
                break

        else: # Loop completed max_internal_turns
            if not final_text_response_from_ai.strip():
                final_text_response_from_ai = f"Max processing turns ({max_internal_turns}) reached. AI will now attempt to summarize based on actions taken."

    finally:
        stop_main_spinner_event.set()
        if 'main_spinner_task' in locals() and main_spinner_task:
            try: await main_spinner_task
            except asyncio.CancelledError: pass
            
    # Determine if explicit summarization by AI is needed
    needs_explicit_summary_check = (functions_were_called_in_this_interaction and not final_text_response_from_ai.strip()) or \
                                   (repeat_detected_in_loop and not final_text_response_from_ai.strip())

    if needs_explicit_summary_check:
        true_original_request_for_summary_ctx = CURRENT_AI_SESSION_ORIGINAL_REQUEST if CURRENT_AI_SESSION_ORIGINAL_REQUEST else user_new_message
        
        last_comparison_tool_name = "get_comparison_for_custom_portfolio"
        relevant_comparison_summary_from_tool = None
        # Search backwards in history for the last call to the comparison tool
        try:
            for i in range(len(AI_CONVERSATION_HISTORY) - 1, -1, -1):
                entry = AI_CONVERSATION_HISTORY[i]
                if entry.get("role") == "tool" and entry.get("parts") and \
                   entry["parts"][0].get("function_response") and \
                   entry["parts"][0]["function_response"].get("name") == last_comparison_tool_name:
                    
                    tool_response_content = entry["parts"][0]["function_response"].get("response", {})
                    if "comparison_outcome_summary" in tool_response_content: # Check if the key summary field exists
                        relevant_comparison_summary_from_tool = tool_response_content.get("comparison_outcome_summary")
                        break # Found the most recent (or any) comparison summary
        except Exception as e_hist_parse_summary:
            print(f"CONSOLE_DEBUG_AI_HANDLER: Error parsing history for specific comparison summary: {e_hist_parse_summary}")

        # Construct the summarization prompt
        summarization_prompt_text_for_ai = (
            f"The user's original request for this session was: '{true_original_request_for_summary_ctx}'. "
            f"The latest user message that initiated this sequence of tool calls was: '{user_new_message}'. "
            f"Review the entire conversation history, including all tool calls and their results. "
        )

        if relevant_comparison_summary_from_tool:
            summarization_prompt_text_for_ai += (
                f"One of the tools, '{last_comparison_tool_name}', was called and it provided the following summary of its findings: '{relevant_comparison_summary_from_tool}'. "
                f"Your final answer regarding any portfolio comparison MUST be directly based on this specific summary from the tool. Do NOT state that the comparison could not be completed if the tool itself provided this summary. "
                f"Incorporate this comparison information accurately along with results from any other tools that were called in response to the user's overall request (like assessments or other portfolio runs). "
            )
            if repeat_detected_in_loop: # Also inform about the repetition if it occurred
                 summarization_prompt_text_for_ai += f"Note: A repetitive call to tool '{tool_name_that_repeated}' was detected and aborted by the system. Please formulate your response based on information gathered *before* this aborted step, and acknowledge if the repetition prevented full completion of a sub-task. "
        else: # No specific comparison summary found to highlight, or it was empty.
            summarization_prompt_text_for_ai += (
                f"Please review our entire conversation history. "
                f"Provide a concise, final, textual answer based on all gathered information. Do not call any new functions. "
            )
            if repeat_detected_in_loop:
                 summarization_prompt_text_for_ai += f"A repetitive call to tool '{tool_name_that_repeated}' was detected and aborted by the system. Please summarize based on information prior to this, and acknowledge if this impacted the overall request. "
            else: # General instruction if no comparison summary or repetition to highlight
                 summarization_prompt_text_for_ai += f"If a part of the request could not be completed (e.g., a comparison was attempted but the tool did not return a usable 'comparison_outcome_summary' or reported a failure), acknowledge that specific part clearly based on any available tool messages. "
        
        summarization_prompt_text_for_ai += (
            f"Provide a comprehensive answer for all parts of the user's request. Adhere strictly to your general synthesis guidelines."
        )
        
        AI_CONVERSATION_HISTORY.append({"role": "user", "parts": [{"text": summarization_prompt_text_for_ai}]})
        # print(f"DEBUG_AI_HANDLER: Summarization prompt sent to AI: {summarization_prompt_text_for_ai}") # For debugging the prompt
        final_text_response_from_ai = "" # Reset for summarization output

        stop_summary_spinner_event = asyncio.Event()
        summary_spinner_task_obj = asyncio.create_task(_continuous_spinner_animation(stop_summary_spinner_event, "AI is synthesizing final answer..."))
        try:
            summary_gemini_response_obj = await asyncio.to_thread(
                gemini_model.generate_content,
                contents=AI_CONVERSATION_HISTORY, tools=None, 
                tool_config={"function_calling_config": {"mode": "NONE"}} # Crucial: No tools for summary
            )
            summary_candidate_obj = summary_gemini_response_obj.candidates[0] if summary_gemini_response_obj.candidates else None
            synthesized_summary_text_from_ai = ""
            if summary_candidate_obj and summary_candidate_obj.content and summary_candidate_obj.content.parts:
                for summary_part_obj in summary_candidate_obj.content.parts:
                    if hasattr(summary_part_obj, 'text') and summary_part_obj.text: 
                        synthesized_summary_text_from_ai += summary_part_obj.text
            
            if synthesized_summary_text_from_ai.strip():
                final_text_response_from_ai = synthesized_summary_text_from_ai.strip()
                AI_CONVERSATION_HISTORY.append({"role": "model", "parts": [{"text": final_text_response_from_ai}]})
            else: 
                final_text_response_from_ai = "AI completed processing but did not provide a final textual summary when explicitly asked during the summarization phase."
                AI_CONVERSATION_HISTORY.append({"role": "model", "parts": [{"text": final_text_response_from_ai}]})
        except Exception as e_summary_api_call:
            final_text_response_from_ai = f"An error occurred during the AI summarization API call: {e_summary_api_call}"
            AI_CONVERSATION_HISTORY.append({"role": "model", "parts": [{"text": final_text_response_from_ai}]})
            risk_logger.error(f"Summarization API call failed: {e_summary_api_call}", exc_info=True)
        finally:
            stop_summary_spinner_event.set() 
            if 'summary_spinner_task_obj' in locals() and summary_spinner_task_obj:
                try: await summary_spinner_task_obj
                except asyncio.CancelledError: pass

    # Final output to user
    print("\n--- AI's Final Synthesized Answer ---")
    print(final_text_response_from_ai if final_text_response_from_ai.strip() else "No final textual response was generated by the AI for this interaction.")
    print("-------------------------------------\n")

    return "AI processing complete." # Or could return final_text_response_from_ai if needed by the caller

# Example of how load_system_prompt might look (ensure it's defined in your actual script)
def load_system_prompt(file_path="system_prompt.txt") -> str:
    default_prompt = """You are an AI assistant for the 'Market Insights Center Singularity' script.
Your goal is to help the user by determining which of the available script functions (tools) can fulfill their request, potentially in multiple steps.
Today's date is {current_date_for_ai_prompt}.

Based on the user's request and any previous function results, decide which function to call next or if the task is complete.
If the request needs multiple steps, call one function at a time. I will execute it and give you its result. Then you can decide the next step.
If the user's request clearly maps to one of the available tools, respond with the function call.
If the request is ambiguous or doesn't map to available tools, ask for clarification.
Strictly adhere to parameter types and descriptions.
For dates (like 'date_to_save', 'date_str'), if user says 'today', use today's date: {current_date_mmddyyyy_for_ai_prompt}.
For 'handle_invest_command': 'sub_portfolios' is an array of objects, each with 'tickers' (string) and 'weight' (number). Sum of weights should be 100.
For 'handle_assess_command' code 'B', construct 'manual_portfolio_holdings' array: e.g., [{{'ticker':'AAPL','shares':10}}, {{'ticker':'Cash','value':500}}].
For 'handle_custom_command': Action 'run_existing_portfolio' runs a saved portfolio and automatically saves/overwrites its detailed run output. Action 'save_portfolio_data' saves *legacy combined percentage data* for one (requires 'date_to_save').
For 'get_comparison_for_custom_portfolio': This tool fetches a previously saved detailed run output, runs the portfolio fresh, and provides a comparison. It DOES NOT overwrite the saved run data by itself. To overwrite, explicitly ask to run and save the portfolio again (which uses 'handle_custom_command' with 'run_existing_portfolio' action).
For 'handle_cultivate_command': Requires 'cultivate_code' (A/B), 'portfolio_value', 'use_fractional_shares' (true/false). Action 'save_data' also needs 'date_to_save'.

**IMPORTANT: After all necessary functions have been executed and you have the information required to answer the user's original request,
synthesize a final, concise answer that directly addresses that original request.
Do not just state the output of the last function called. Analyze all gathered data from the function results in the conversation history.
If you believe the original request has been fully addressed with the information you have, clearly state the final answer.**
"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        # print(f"Warning: System prompt file '{file_path}' not found. Using default prompt.") # Already handled if called from handle_ai_prompt init
        try:
            with open(file_path, 'w', encoding='utf-8') as f_create:
                f_create.write(default_prompt)
            # print(f"Created default system prompt file at '{file_path}'. Please review and customize it.")
        except Exception: pass # Ignore if cannot create
        return default_prompt
    except Exception as e_load:
        # print(f"Error loading system prompt from '{file_path}': {e_load}. Using default prompt.")
        return default_prompt

# --- Main Singularity Loop ---
async def main_singularity():
    global AI_CONVERSATION_HISTORY, risk_persistent_signal, risk_signal_day, gemini_model
    global CURRENT_AI_SESSION_ORIGINAL_REQUEST, AI_INTERNAL_STEP_COUNT
    # No 'global risk_logger' here, as we are not reassigning the global 'risk_logger' variable itself.
    # We are just using the one already configured in the global scope.

    # --- Initialize Gemini API model (ensure this is done robustly) ---
    if 'gemini_model' not in globals() or gemini_model is None:
        if 'GEMINI_API_KEY' not in globals() or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY" or not GEMINI_API_KEY:
            print("FATAL: Gemini API Key not set or is a placeholder. Please set the GEMINI_API_KEY variable.")
            return
        try:
            if gemini_model is None: # If initial configuration failed
                 print("FATAL: Gemini model was not configured successfully at startup.")
                 return
        except Exception as e_cfg_main: # This try-except might be redundant if gemini_model check is enough
            print(f"FATAL: Error related to Gemini API in main_singularity: {e_cfg_main}")
            return

    # --- Check if risk_logger is available (it should be from global setup) ---
    risk_logger.info("Singularity Application Main Loop Started.")

    display_welcome_message() 
    display_commands()

    while True:
        try:
            prompt_text = "[AI Chat Active] > " if AI_CONVERSATION_HISTORY else "Enter command: "
            user_input_full_line = input(prompt_text).strip()

            if not user_input_full_line:
                continue

            if user_input_full_line.lower() == "end chat":
                if AI_CONVERSATION_HISTORY:
                    print("AI chat session ended by user. History cleared.")
                    AI_CONVERSATION_HISTORY.clear()
                    CURRENT_AI_SESSION_ORIGINAL_REQUEST = None
                    AI_INTERNAL_STEP_COUNT = 0
                else:
                    print("No active AI chat session to end.")
                continue

            command_parts = user_input_full_line.split()
            command = command_parts[0].lower()
            args = command_parts[1:]

            if command == "/ai":
                user_natural_prompt = " ".join(args)
                if not user_natural_prompt:
                    print("Usage: /ai <your request or question for the AI assistant>")
                elif not gemini_model:
                    print("Gemini API model is not available. Cannot process AI request.")
                else:
                    is_new_ai_session = not AI_CONVERSATION_HISTORY
                    if is_new_ai_session:
                        print("No active AI chat. Starting a new session with this /ai prompt...")
                        CURRENT_AI_SESSION_ORIGINAL_REQUEST = user_natural_prompt
                        AI_INTERNAL_STEP_COUNT = 0
                    await handle_ai_prompt(
                        user_natural_prompt,
                        is_new_session=is_new_ai_session,
                        original_session_request=CURRENT_AI_SESSION_ORIGINAL_REQUEST if CURRENT_AI_SESSION_ORIGINAL_REQUEST else user_natural_prompt
                    )
            
            # --- MODIFIED CLI COMMAND ROUTING ---
            elif command == "/briefing": await handle_briefing_command(args, ai_params=None, is_called_by_ai=False)
            elif command == "/spear": await handle_spear_command(args, ai_params=None, is_called_by_ai=False)
            elif command == "/invest": await handle_invest_command(args, ai_params=None, is_called_by_ai=False)
            elif command == "/custom": await handle_custom_command(args, ai_params=None, is_called_by_ai=False)
            elif command == "/breakout": await handle_breakout_command(args, ai_params=None, is_called_by_ai=False)
            elif command == "/market": await handle_market_command(args, ai_params=None, is_called_by_ai=False)
            elif command == "/cultivate": await handle_cultivate_command(args, ai_params=None, is_called_by_ai=False)
            elif command == "/assess": await handle_assess_command(args, ai_params=None, is_called_by_ai=False)
            elif command == "/quickscore": await handle_quickscore_command(args, ai_params=None, is_called_by_ai=False)
            elif command == "/risk": await handle_risk_command(args, ai_params=None, is_called_by_ai=False)
            elif command == "/history": await handle_history_command(args, ai_params=None, is_called_by_ai=False)
            elif command == "/help": display_commands()
            elif command == "/exit":
                print("Exiting Market Insights Center Singularity. Goodbye!")
                risk_logger.info("Singularity Application Exited by user command.")
                break
            else:
                if AI_CONVERSATION_HISTORY and not user_input_full_line.startswith("/"):
                    await handle_ai_prompt(
                        user_input_full_line,
                        is_new_session=False,
                        original_session_request=CURRENT_AI_SESSION_ORIGINAL_REQUEST
                    )
                else:
                    print(f"Unknown command: {command}. Type /help for available commands.")

        except KeyboardInterrupt:
            print("\nExiting Market Insights Center Singularity (KeyboardInterrupt). Goodbye!")
            if AI_CONVERSATION_HISTORY :
                risk_logger.info(f"Singularity App Interrupted. AI chat history had {len(AI_CONVERSATION_HISTORY)} turns.")
            else:
                risk_logger.info("Singularity App Interrupted by User (KeyboardInterrupt).")
            CURRENT_AI_SESSION_ORIGINAL_REQUEST = None
            AI_INTERNAL_STEP_COUNT = 0
            break
        except Exception as e_main_loop:
            print(f"An unexpected error occurred in the main loop: {e_main_loop}")
            risk_logger.exception("Unexpected error in main_singularity loop:")
            traceback.print_exc()

if __name__ == "__main__":
    # --- One-time Setup (API Keys, Loggers, Global Models) ---
    # This section should ideally be at the very top of your script, outside any function.
    # For this example, simulating it here.
    # Ensure GEMINI_API_KEY is correctly set before this point.
    # Example: GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_ACTUAL") or "YOUR_FALLBACK_KEY"

    # Configure Gemini API (should happen once)
    if 'GEMINI_API_KEY' in globals() and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY" and GEMINI_API_KEY:
        try:
            if 'genai' not in globals(): import google.generativeai as genai # Ensure import
            genai.configure(api_key=GEMINI_API_KEY)
            if 'gemini_model' not in globals() or gemini_model is None:
                 gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            print("Gemini API configured at script startup.")
        except Exception as e_startup_gemini:
            print(f"FATAL: Error configuring Gemini API at startup: {e_startup_gemini}")
            gemini_model = None # Ensure model is None if config fails
    else:
        print("Warning: GEMINI_API_KEY not found or is a placeholder. AI features will not work.")
        gemini_model = None


    # Configure risk_logger (should happen once)
    if 'risk_logger' in globals() and isinstance(risk_logger, logging.Logger) and not risk_logger.hasHandlers():
        if 'RISK_LOG_FILE' not in globals(): RISK_LOG_FILE = 'risk_calculations_default.log'
        try:
            risk_file_handler_startup = logging.FileHandler(RISK_LOG_FILE)
            risk_formatter_startup = logging.Formatter('%(asctime)s - %(levelname)s - Module:%(module)s - Func:%(funcName)s - %(message)s')
            risk_file_handler_startup.setFormatter(risk_formatter_startup)
            risk_logger.addHandler(risk_file_handler_startup)
            risk_logger.info("Risk logger configured at script startup.")
        except Exception as e_logger_setup:
            print(f"Error setting up risk_logger: {e_logger_setup}")


    # Initialize AI Conversation History and other state variables (should be global)
    if 'AI_CONVERSATION_HISTORY' not in globals(): AI_CONVERSATION_HISTORY = []
    if 'CURRENT_AI_SESSION_ORIGINAL_REQUEST' not in globals(): CURRENT_AI_SESSION_ORIGINAL_REQUEST = None
    if 'AI_INTERNAL_STEP_COUNT' not in globals(): AI_INTERNAL_STEP_COUNT = 0


    try:
        asyncio.run(main_singularity())
    except KeyboardInterrupt:
        print("\nApplication terminated by user (main execution scope).")
        if 'risk_logger' in globals() and risk_logger: risk_logger.info("Application terminated by user from main execution scope.")
    except Exception as e_run_critical:
        print(f"Critical error running application: {e_run_critical}")
        traceback.print_exc()
        if 'risk_logger' in globals() and risk_logger:
            risk_logger.critical(f"Application crashed with an unexpected error: {e_run_critical}", exc_info=True)
        else: # Fallback basic logging if risk_logger itself failed or wasn't set up
            if 'RISK_LOG_FILE' not in globals(): RISK_LOG_FILE = 'risk_calculations_critical_error.log'
            logging.basicConfig(filename=RISK_LOG_FILE, level=logging.ERROR) # Basic config
            logging.critical(f"Application CRASH (logger might have failed): {e_run_critical}", exc_info=True)