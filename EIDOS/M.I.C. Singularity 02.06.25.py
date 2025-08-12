import yfinance as yf
import pandas as pd
import math
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
from datetime import datetime # Keep standard datetime
from datetime import time as dt_time, datetime as dt_datetime # For specific time objects if needed by R.I.S.K logic
import pytz
from typing import Optional, List, Dict, Any
import time as py_time
import traceback
import logging # For R.I.S.K. module's logging
import json
import google.generativeai as genai
# We will only import FunctionDeclaration and Tool, and define parameters using dicts
import google.generativeai as genai
# At the top of your script with other google.generativeai imports:
from google.generativeai.types import FunctionDeclaration, Tool
# REMOVE: from google.generativeai.types import Content, Part

# --- Global Variables & Constants ---
portfolio_db_file = 'portfolio_codes_database.csv'
est_timezone = pytz.timezone('US/Eastern') # R.I.S.K uses EST, Singularity uses it for consistency if needed
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

# --- Gemini API Configuration --- # <--- ADD THIS SECTION
GEMINI_API_KEY = "AIzaSyDrtdvnjkMipwqOn-dboX2mpFJ_x05ulkI"  # <--- REPLACE WITH YOUR ACTUAL API KEY
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or another model like 'gemini-pro'
    print("Gemini API configured successfully.")
except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    gemini_model = None # Set to None if configuration fails
# --- End Gemini API Configuration ---

# --- AI Chat Session State ---
GEMINI_CHAT_SESSION = None # Will hold the active ChatSession object
CONVERSATION_HISTORY_FOR_LOGGING = [] # For logging purposes
AI_CONVERSATION_HISTORY = [] # Initialize this list globally in your script

# --- R.I.S.K. Module Specific Constants & Globals ---
RISK_CSV_FILE = "market_data.csv"  # Main data file for RISK module
RISK_EOD_CSV_FILE = "risk_eod_data.csv"  # EOD data file for RISK module
RISK_LOG_FILE = 'risk_calculations.log'

# R.I.S.K. global state for market signal (will be loaded/updated from CSV for Singularity)
risk_persistent_signal = "Hold" # Default
risk_signal_day = None # Default

# --- Logging Setup (for R.I.S.K. module parts) ---
# Configure a separate logger for RISK module functions to avoid conflicts if main Singularity has other logging
risk_logger = logging.getLogger('RISK_MODULE')
risk_logger.setLevel(logging.INFO)
# Prevent risk_logger from propagating to the root logger if main Singularity has one
risk_logger.propagate = False 
if not risk_logger.hasHandlers(): # Add handler only if it doesn't exist
    risk_file_handler = logging.FileHandler(RISK_LOG_FILE)
    risk_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
    risk_file_handler.setFormatter(risk_formatter)
    risk_logger.addHandler(risk_file_handler)

# --- Utility Functions (Adapted from original) ---
# safe_score, calculate_ema_invest, calculate_one_year_invest, plot_ticker_graph
# get_allocation_score, process_custom_portfolio, generate_portfolio_pie_chart
# save_portfolio_to_csv, save_portfolio_data_singularity
# get_sp500_symbols_singularity, get_spy_symbols_singularity, calculate_market_invest_scores_singularity, save_market_data_singularity
# run_breakout_analysis_singularity, save_breakout_data_singularity
# (These functions are assumed to be present from previous steps and are not repeated here for brevity)

def load_system_prompt(file_path="system_prompt.txt") -> str:
    """Loads the system prompt from a text file."""
    # Ensure all literal curly braces in this default_prompt are doubled
    default_prompt = """You are an AI assistant for the 'Market Insights Center Singularity' script.
Your goal is to help the user by determining which of the available script functions (tools) can fulfill their request, potentially in multiple steps.
Today's date is {current_date_for_ai_prompt}.

Based on the user's request and any previous function results, decide which function to call next or if the task is complete.
If the request needs multiple steps, call one function at a time. I will execute it and give you its result. Then you can decide the next step.
If the user's request clearly maps to one of the available tools, respond with the function call.
If the request is ambiguous or doesn't map to available tools, ask for clarification.
If you believe the user's request has been fully addressed, respond with a final summary message.
Strictly adhere to parameter types and descriptions.
For dates (like 'date_to_save', 'date_str'), if user says 'today', use today's date: {current_date_mmddyyyy_for_ai_prompt}.
For 'handle_invest_command': 'sub_portfolios' is an array of objects, each with 'tickers' (string) and 'weight' (number). Sum of weights should be 100.
For 'handle_assess_command' code 'B', construct 'manual_portfolio_holdings' array: e.g., [{{'ticker':'AAPL','shares':10}}, {{'ticker':'Cash','value':500}}].
For 'handle_custom_command': Action 'run_existing_portfolio' runs a saved portfolio. Action 'save_portfolio_data' saves data for one.
For 'handle_cultivate_command': Requires 'cultivate_code' (A/B), 'portfolio_value', 'use_fractional_shares' (true/false). Action 'save_data' also needs 'date_to_save'.
"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # When reading from file, we expect the placeholders to be filled by handle_ai_prompt
            return f.read() 
    except FileNotFoundError:
        print(f"Warning: System prompt file '{file_path}' not found. Using default prompt.")
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                # Write the raw default_prompt. The .format() call will happen later.
                # The default_prompt string itself already contains the placeholders.
                f.write(default_prompt) # Write it with its existing placeholders {current_date...}
            print(f"Created default system prompt file at '{file_path}'. Please review and customize it.")
            # Return the raw default prompt, formatting will happen in handle_ai_prompt
            return default_prompt 
        except Exception as e_create:
            # The error was in f.write(default_prompt.format(...))
            # Now we write default_prompt directly. The error message below should change if an issue persists here.
            print(f"Error creating default system prompt file with content: {e_create}")
            return default_prompt # Return the raw default prompt (with unescaped example braces if not fixed)
    except Exception as e:
        print(f"Error loading system prompt from '{file_path}': {e}. Using default prompt.")
        return default_prompt

# Then, in handle_ai_prompt:
# ...
# base_system_prompt_text = load_system_prompt()
# formatted_system_prompt = base_system_prompt_text.format(
#     current_date_for_ai_prompt=datetime.now().strftime('%B %d, %Y'),
#     current_date_mmddyyyy_for_ai_prompt=datetime.now().strftime('%m/%d/%Y')
# )
# initial_chat_history = [
#     {"role": "user", "parts": [{"text": formatted_system_prompt}]},
# ...
    
def safe_score(value: Any) -> float:
    """
    Safely converts a value to a float, returning 0.0 for NaN, None, or conversion errors.
    Handles strings with '%' or '$'.
    """
    try:
        if pd.isna(value) or value is None:
            return 0.0
        if isinstance(value, str):
            value = value.replace('%', '').replace('$', '').strip()
        return float(value)
    except (ValueError, TypeError):
        return 0.0

async def calculate_ema_invest(ticker: str, ema_interval: int) -> tuple[Optional[float], Optional[float]]:
    """
    Fetches historical data and calculates the EMA Invest score. Async.
    """
    ticker_yf_format = ticker.replace('.', '-')
    stock = yf.Ticker(ticker_yf_format)

    interval_mapping = {1: "1wk", 2: "1d", 3: "1h"}
    interval_str = interval_mapping.get(ema_interval, "1h")

    period_str = ""
    if ema_interval == 3: period_str = "2y"
    elif ema_interval == 1: period_str = "max"
    elif ema_interval == 2: period_str = "10y"
    else: period_str = "2y"

    try:
        # yfinance calls can sometimes be slow or fail; running in a thread.
        data = await asyncio.to_thread(stock.history, period=period_str, interval=interval_str)
    except Exception as e:
        # Reduced verbosity for common ticker fetch errors unless debugging.
        # print(f"[{datetime.now(est_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')}] calculate_ema_invest: Error fetching history for {ticker} (Interval {interval_str}, Period {period_str}): {e}")
        return None, None

    if data.empty:
        # print(f"[{datetime.now(est_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')}] calculate_ema_invest: No history data for {ticker} (Interval {interval_str}, Period {period_str}).")
        return None, None
    if 'Close' not in data.columns:
        # print(f"[{datetime.now(est_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')}] calculate_ema_invest: 'Close' column missing for {ticker}.")
        return None, None

    try:
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_13'] = data['Close'].ewm(span=13, adjust=False).mean() # Not used in score but often useful
        data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean() # Not used in score
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()
    except Exception as e:
        print(f"[{datetime.now(est_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')}] calculate_ema_invest: Error calculating EMAs for {ticker}: {e}")
        return None, None # Return None if EMA calculation fails

    # Check if the latest row has NaN values for critical columns
    if data.empty or data.iloc[-1][['Close', 'EMA_8', 'EMA_55']].isna().any():
       live_price_fallback = data['Close'].iloc[-1] if not data.empty and pd.notna(data['Close'].iloc[-1]) else None
       # print(f"Warning: NaN values in latest data for {ticker}. Price: {live_price_fallback}, Score: None")
       return live_price_fallback, None

    latest_data = data.iloc[-1]
    live_price = latest_data['Close']
    ema_8 = latest_data['EMA_8']
    ema_55 = latest_data['EMA_55']

    if pd.isna(live_price) or pd.isna(ema_8) or pd.isna(ema_55) or ema_55 == 0:
        # print(f"Warning: Critical EMA values are NaN or EMA55 is zero for {ticker}. Price: {live_price}, Score: None")
        return live_price, None

    ema_enter = (ema_8 - ema_55) / ema_55
    ema_invest_score = ((ema_enter * 4) + 0.5) * 100
    return float(live_price), float(ema_invest_score)

async def calculate_one_year_invest(ticker: str) -> tuple[float, float]:
    """
    Calculates the one-year percentage change and invest_per score. Async.
    """
    ticker_yf_format = ticker.replace('.', '-')
    stock = yf.Ticker(ticker_yf_format)
    try:
        data = await asyncio.to_thread(stock.history, period="1y")
        if data.empty or len(data) < 2: return 0.0, 50.0 # Neutral default
        if 'Close' not in data.columns: return 0.0, 50.0
    except Exception as e:
        # print(f"[{datetime.now(est_timezone).strftime('%Y-%m-%d %H:%M:%S %Z')}] calculate_one_year_invest: Error fetching 1-year history for {ticker}: {e}")
        return 0.0, 50.0 # Neutral default on error

    start_price = data['Close'].iloc[0]
    end_price = data['Close'].iloc[-1]

    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return 0.0, 50.0 # Neutral default

    one_year_change = ((end_price - start_price) / start_price) * 100
    invest_per = 50.0 # Default
    try:
        if one_year_change < 0:
            invest_per = (one_year_change / 2) + 50
        else:
            invest_per = math.sqrt(max(0, one_year_change * 5)) + 50
    except ValueError: # math.sqrt domain error
        invest_per = 50.0
        
    invest_per = max(0, min(invest_per, 100)) # Clamp score
    return float(one_year_change), float(invest_per)

def plot_ticker_graph(ticker: str, ema_interval: int) -> Optional[str]:
    """
    Plots ticker price and EMAs, saves the graph to a file.
    """
    ticker_yf_format = ticker.replace('.', '-')
    stock = yf.Ticker(ticker_yf_format)

    interval_mapping = {1: "1wk", 2: "1d", 3: "1h"}
    interval_str = interval_mapping.get(ema_interval, "1h")

    period_str = ""
    if ema_interval == 3: period_str = "6mo" # Hourly
    elif ema_interval == 1: period_str = "5y" # Weekly - longer period for context
    elif ema_interval == 2: period_str = "1y" # Daily
    else: period_str = "1y" # Default

    try:
        data = stock.history(period=period_str, interval=interval_str)
        if data.empty or 'Close' not in data.columns:
            raise ValueError(f"No data or 'Close' column returned for {ticker} (Period: {period_str}, Interval: {interval_str})")

        # Calculate EMAs needed for plot
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 6)) # Use subplots for better control
        ax.plot(data.index, data['Close'], color='grey', label='Price', linewidth=1.0)
        ax.plot(data.index, data['EMA_55'], color='darkgreen', label='EMA 55', linewidth=1.5) 
        ax.plot(data.index, data['EMA_8'], color='firebrick', label='EMA 8', linewidth=1.5) 
        ax.set_title(f"{ticker} Price and EMAs ({interval_str})", color='white')
        ax.set_xlabel('Date', color='white')
        ax.set_ylabel('Price', color='white')
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax.grid(True, color='dimgray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')

        fig.tight_layout()

        # Generate a unique filename to avoid conflicts if multiple graphs are made for the same ticker
        filename = f"{ticker}_graph_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black', edgecolor='black')
        plt.close(fig) # Close the figure explicitly
        print(f"Graph saved: {filename}")
        return filename # Return filename on success
    except Exception as e:
        print(f"Error plotting graph for {ticker}: {e}")
        # Ensure figure is closed if error occurs after creation
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)
        return None # Return None on failure

def get_allocation_score() -> tuple[float, float, float]:
    """
    Calculates the overall market allocation score (Sigma for Cultivate), 
    General Market Score, and Market Invest Score by reading RISK_CSV_FILE.
    Returns (avg_score, general_score, market_invest_score).
    Returns (50.0, 50.0, 50.0) as defaults if data cannot be read or processed.
    """
    avg_score, general_score, market_invest_score = 50.0, 50.0, 50.0 # Default values

    if not os.path.exists(RISK_CSV_FILE):
        print(f"Warning: Market data file '{RISK_CSV_FILE}' not found for get_allocation_score(). Using default scores (50.0).")
        print("         Run /risk command to generate this file.")
        return avg_score, general_score, market_invest_score

    try:
        df = pd.read_csv(RISK_CSV_FILE, on_bad_lines='skip') 
        if df.empty:
            print(f"Warning: Market data file '{RISK_CSV_FILE}' is empty. Using default scores (50.0).")
            return avg_score, general_score, market_invest_score

        required_cols = ['General Market Score', 'Market Invest Score'] 
        if not all(col in df.columns for col in required_cols):
            print(f"Warning: '{RISK_CSV_FILE}' is missing required columns ('General Market Score', 'Market Invest Score'). Using default scores (50.0).")
            return avg_score, general_score, market_invest_score
        
        # --- MODIFICATION: Sort by Timestamp before selecting the last row ---
        if 'Timestamp' in df.columns:
            try:
                # Convert to datetime objects, coercing errors for robustness
                df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
                # Drop rows where Timestamp conversion failed, if any
                df.dropna(subset=['Timestamp'], inplace=True)
                if not df.empty:
                    df = df.sort_values(by='Timestamp', ascending=True)
                else:
                    print(f"Warning: No valid timestamps found in '{RISK_CSV_FILE}' after conversion. Cannot reliably get latest data.")
                    return avg_score, general_score, market_invest_score # Fallback to defaults
            except Exception as e_sort:
                print(f"Warning: Could not sort market data by Timestamp in get_allocation_score: {e_sort}. Proceeding without sorting.")
        else:
            print(f"Warning: 'Timestamp' column not found in '{RISK_CSV_FILE}'. Cannot guarantee the last row is the latest data.")
        # --- END OF MODIFICATION ---

        if df.empty: # Check again if all rows were dropped due to bad timestamps
            print(f"Warning: Market data file '{RISK_CSV_FILE}' became empty after timestamp processing. Using default scores (50.0).")
            return avg_score, general_score, market_invest_score

        latest_data = df.iloc[-1] # Now this should be the latest chronological entry if sorting was successful
        
        gs_val = latest_data.get('General Market Score')
        mis_val = latest_data.get('Market Invest Score') 

        gs_numeric = safe_score(gs_val) # safe_score should be defined in your script
        mis_numeric = safe_score(mis_val)

        if pd.isna(gs_numeric) or pd.isna(mis_numeric): 
            print(f"Warning: Latest row in '{RISK_CSV_FILE}' contains N/A for required scores after conversion. Using default scores (50.0).")
            return avg_score, general_score, market_invest_score

        general_score_calc = gs_numeric
        market_invest_score_calc = mis_numeric 

        avg_score_calc = (general_score_calc + (2 * market_invest_score_calc)) / 3.0
        
        avg_score = max(0.0, min(100.0, avg_score_calc))
        general_score = max(0.0, min(100.0, general_score_calc))
        market_invest_score = max(0.0, min(100.0, market_invest_score_calc))

        print(f"  get_allocation_score: Using scores from '{RISK_CSV_FILE}': Avg (Sigma)={avg_score:.2f}, General={general_score:.2f}, MktInv={market_invest_score:.2f}")
        return avg_score, general_score, market_invest_score

    except pd.errors.EmptyDataError:
        print(f"Warning: Market data file '{RISK_CSV_FILE}' is empty (caught by EmptyDataError). Using default scores (50.0).")
        return avg_score, general_score, market_invest_score
    except Exception as e:
        print(f"Error reading or processing '{RISK_CSV_FILE}' in get_allocation_score: {e}. Using default scores (50.0).")
        traceback.print_exc() # It's often good to see the traceback for unexpected errors
        return avg_score, general_score, market_invest_score
    
async def process_custom_portfolio(
    portfolio_data_config: Dict[str, Any],
    tailor_portfolio_requested: bool,
    frac_shares_singularity: bool,
    total_value_singularity: Optional[float] = None,
    is_custom_command_simplified_output: bool = False
) -> tuple[List[str], List[Dict[str, Any]], float, List[Dict[str, Any]]]:
    """
    Processes custom or /invest portfolio requests for Singularity.
    Calculates scores, allocations, generates output tables, graphs, and a pie chart.
    MODIFIED: Returns structured tailored holdings as the 4th element.
    """
    sell_to_cash_active = False
    avg_score, _, _ = get_allocation_score() 

    if avg_score is not None and avg_score < 50.0:
        sell_to_cash_active = True
        print(f"\n:warning: **Sell-to-Cash Feature Active!** (Avg Market Score: {avg_score:.2f} < 50).")

    ema_sensitivity = int(portfolio_data_config.get('ema_sensitivity', 3))
    amplification = float(portfolio_data_config.get('amplification', 1.0))
    num_portfolios = int(portfolio_data_config.get('num_portfolios', 0))

    portfolio_results_list = [] 
    all_entries_for_graphs_plotting = [] 

    for i in range(num_portfolios):
        portfolio_index = i + 1
        tickers_str = portfolio_data_config.get(f'tickers_{portfolio_index}', '')
        weight_str = portfolio_data_config.get(f'weight_{portfolio_index}', '0')
        weight = safe_score(weight_str)
        tickers = [ticker.strip().upper() for ticker in tickers_str.split(',') if ticker.strip()]
        if not tickers: continue

        current_portfolio_list_calc = []
        for ticker in tickers:
            try:
                live_price, ema_invest = await calculate_ema_invest(ticker, ema_sensitivity)
                if live_price is None and ema_invest is None: 
                    current_portfolio_list_calc.append({'ticker': ticker, 'error': "Failed to fetch critical data", 'portfolio_weight': weight})
                    all_entries_for_graphs_plotting.append({'ticker': ticker, 'error': "Failed to fetch critical data"})
                    continue 
                
                ema_invest = 50.0 if ema_invest is None else ema_invest 
                live_price = 0.0 if live_price is None else live_price 

                _, invest_per = await calculate_one_year_invest(ticker) 
                
                raw_combined_invest = safe_score(ema_invest)
                score_for_allocation = raw_combined_invest
                score_was_adjusted = False

                if sell_to_cash_active and raw_combined_invest < 50.0:
                    score_for_allocation = 50.0 
                    score_was_adjusted = True
                
                amplified_score_adjusted = safe_score((score_for_allocation * amplification) - (amplification - 1) * 50)
                amplified_score_adjusted_clamped = max(0, amplified_score_adjusted) 
                
                amplified_score_original = safe_score((raw_combined_invest * amplification) - (amplification - 1) * 50)
                amplified_score_original_clamped = max(0, amplified_score_original) 

                entry_data = {
                    'ticker': ticker, 'live_price': live_price, 'raw_invest_score': raw_combined_invest,
                    'amplified_score_adjusted': amplified_score_adjusted_clamped, 
                    'amplified_score_original': amplified_score_original_clamped, 
                    'portfolio_weight': weight, 'score_was_adjusted': score_was_adjusted,
                    'portfolio_allocation_percent_adjusted': None, 
                    'portfolio_allocation_percent_original': None, 
                    'combined_percent_allocation_adjusted': None, 
                    'combined_percent_allocation_original': None, 
                }
                current_portfolio_list_calc.append(entry_data)
                if live_price > 0: 
                    all_entries_for_graphs_plotting.append({'ticker': ticker, 'ema_sensitivity': ema_sensitivity})
            except Exception as e:
                current_portfolio_list_calc.append({'ticker': ticker, 'error': str(e), 'portfolio_weight': weight})
                all_entries_for_graphs_plotting.append({'ticker': ticker, 'error': str(e)})
        portfolio_results_list.append(current_portfolio_list_calc)

    sent_graphs = set()
    if not is_custom_command_simplified_output or not tailor_portfolio_requested: # Avoid graph prints if simplified output for tailored assess
        # print("\nGenerating ticker graphs (if any)...") # Reduced verbosity
        for graph_entry in all_entries_for_graphs_plotting:
            ticker_key = graph_entry.get('ticker')
            if not ticker_key or ticker_key in sent_graphs: continue
            if 'error' not in graph_entry:
                plot_ticker_graph(ticker_key, graph_entry['ema_sensitivity']) 
                sent_graphs.add(ticker_key)

    for portfolio_list_calc in portfolio_results_list:
        portfolio_amplified_total_adjusted = safe_score(sum(entry['amplified_score_adjusted'] for entry in portfolio_list_calc if 'error' not in entry))
        for entry in portfolio_list_calc:
            if 'error' not in entry:
                if portfolio_amplified_total_adjusted > 0:
                    amplified_score_adj = safe_score(entry.get('amplified_score_adjusted', 0))
                    portfolio_allocation_percent_adj = safe_score((amplified_score_adj / portfolio_amplified_total_adjusted) * 100)
                    entry['portfolio_allocation_percent_adjusted'] = round(portfolio_allocation_percent_adj, 2)
                else: entry['portfolio_allocation_percent_adjusted'] = 0.0
            else: entry['portfolio_allocation_percent_adjusted'] = None 
        
        portfolio_amplified_total_original = safe_score(sum(entry['amplified_score_original'] for entry in portfolio_list_calc if 'error' not in entry))
        for entry in portfolio_list_calc:
            if 'error' not in entry:
                if portfolio_amplified_total_original > 0:
                    amplified_score_orig = safe_score(entry.get('amplified_score_original', 0))
                    portfolio_allocation_percent_orig = safe_score((amplified_score_orig / portfolio_amplified_total_original) * 100)
                    entry['portfolio_allocation_percent_original'] = round(portfolio_allocation_percent_orig, 2)
                else: entry['portfolio_allocation_percent_original'] = 0.0
            else: entry['portfolio_allocation_percent_original'] = None 
    
    if not is_custom_command_simplified_output:
        print("\n--- Sub-Portfolio Details ---")
        for i, portfolio_list_calc in enumerate(portfolio_results_list, 1):
            portfolio_list_calc.sort(key=lambda x: x.get('portfolio_allocation_percent_adjusted', -1) if x.get('portfolio_allocation_percent_adjusted') is not None else -1, reverse=True)
            portfolio_weight_display = portfolio_list_calc[0].get('portfolio_weight', 'N/A') if portfolio_list_calc and 'error' not in portfolio_list_calc[0] else 'N/A'
            print(f"\n**--- Sub-Portfolio {i} (Weight: {portfolio_weight_display}%) ---**")
            table_data_sub = []
            for entry in portfolio_list_calc:
                if 'error' not in entry:
                    live_price_f = f"${entry.get('live_price', 0):.2f}"
                    invest_score_val = safe_score(entry.get('raw_invest_score', 0))
                    invest_score_f = f"{invest_score_val:.2f}%" if invest_score_val is not None else "N/A"
                    amplified_score_f = f"{entry.get('amplified_score_adjusted', 0):.2f}%" 
                    port_alloc_val_original = safe_score(entry.get('portfolio_allocation_percent_original', 0)) 
                    port_alloc_f = f"{port_alloc_val_original:.2f}%" if port_alloc_val_original is not None else "N/A"
                    table_data_sub.append([entry.get('ticker', 'ERR'), live_price_f, invest_score_f, amplified_score_f, port_alloc_f])
            
            if not table_data_sub: print("No valid data for this sub-portfolio.")
            else: print(tabulate(table_data_sub, headers=["Ticker", "Live Price", "Raw Score", "Adj Amplified %", "Portfolio % Alloc (Original)"], tablefmt="pretty"))
            
            error_messages = [f"Error for {entry.get('ticker', 'UNKNOWN')}: {entry.get('error', 'Unknown error')}" for entry in portfolio_list_calc if 'error' in entry]
            if error_messages: print(f"Errors in Sub-Portfolio {i}:\n" + "\n".join(error_messages))

    combined_result_intermediate_calc = []
    for portfolio_list_calc in portfolio_results_list:
        for entry in portfolio_list_calc:
            if 'error' not in entry:
                port_weight = entry.get('portfolio_weight', 0)
                sub_alloc_adj = entry.get('portfolio_allocation_percent_adjusted', 0)
                entry['combined_percent_allocation_adjusted'] = round(safe_score((sub_alloc_adj * port_weight) / 100), 4)
                
                sub_alloc_orig = entry.get('portfolio_allocation_percent_original', 0)
                entry['combined_percent_allocation_original'] = round(safe_score((sub_alloc_orig * port_weight) / 100), 4)
                combined_result_intermediate_calc.append(entry) 

    final_combined_portfolio_data_calc = []
    total_cash_diff_percent = 0.0 

    for entry in combined_result_intermediate_calc: 
        final_combined_portfolio_data_calc.append({
            'ticker': entry['ticker'], 
            'live_price': entry['live_price'],
            'raw_invest_score': entry['raw_invest_score'],
            'amplified_score_adjusted': entry['amplified_score_adjusted'], 
            'combined_percent_allocation': entry['combined_percent_allocation_adjusted'] 
        })

        if sell_to_cash_active and entry.get('score_was_adjusted', False):
            difference_for_cash = entry['combined_percent_allocation_adjusted'] - entry['combined_percent_allocation_original']
            total_cash_diff_percent += max(0.0, difference_for_cash) 

    if sell_to_cash_active and total_cash_diff_percent > 0:
        current_stock_total_alloc_percent = sum(item['combined_percent_allocation'] for item in final_combined_portfolio_data_calc if item['ticker'] != 'Cash')
        target_stock_alloc_percent = 100.0 - total_cash_diff_percent 

        if current_stock_total_alloc_percent > 1e-9 and target_stock_alloc_percent >= 0:
            norm_factor = target_stock_alloc_percent / current_stock_total_alloc_percent
            for item in final_combined_portfolio_data_calc:
                if item['ticker'] != 'Cash': 
                    item['combined_percent_allocation'] *= norm_factor
        elif target_stock_alloc_percent < 0: 
            for item in final_combined_portfolio_data_calc:
                if item['ticker'] != 'Cash':
                    item['combined_percent_allocation'] = 0.0
            total_cash_diff_percent = 100.0 

    if total_cash_diff_percent > 1e-4 : 
        total_cash_diff_percent = min(total_cash_diff_percent, 100.0) 
        final_combined_portfolio_data_calc.append({
            'ticker': 'Cash', 
            'live_price': 1.0, 
            'raw_invest_score': None,
            'amplified_score_adjusted': None, 
            'combined_percent_allocation': total_cash_diff_percent
        })

    final_combined_portfolio_data_calc.sort(
        key=lambda x: x.get('raw_invest_score', -float('inf')) if x.get('ticker') != 'Cash' else -float('inf')-1, 
        reverse=True 
    )
    
    current_total_allocation_final = sum(item['combined_percent_allocation'] for item in final_combined_portfolio_data_calc)
    if not math.isclose(current_total_allocation_final, 100.0, abs_tol=0.1) and current_total_allocation_final > 1e-9:
        norm_factor_final = 100.0 / current_total_allocation_final
        for item in final_combined_portfolio_data_calc:
            item['combined_percent_allocation'] *= norm_factor_final
            if item['ticker'] == 'Cash':
                item['combined_percent_allocation'] = min(item['combined_percent_allocation'], 100.0)

    if not is_custom_command_simplified_output:
        print("\n**--- Final Combined Portfolio (Sorted by Raw Score)---**")
        if sell_to_cash_active: print("*(Sell-to-Cash Active)*") 
        
        combined_data_display = []
        for entry in final_combined_portfolio_data_calc:
            ticker = entry.get('ticker', 'ERR')
            if ticker == 'Cash': 
                live_price_f, invest_score_f, amplified_score_f = '-', '-', '-'
            else:
                live_price_f = f"${entry.get('live_price', 0):.2f}"
                invest_score_f = f"{entry.get('raw_invest_score', 0):.2f}%"
                amplified_score_f = f"{entry.get('amplified_score_adjusted', 0):.2f}%" 
            comb_alloc_f = f"{round(entry.get('combined_percent_allocation', 0), 2):.2f}%"
            combined_data_display.append([ticker, live_price_f, invest_score_f, amplified_score_f, comb_alloc_f])

        if not combined_data_display: print("No valid data for the combined portfolio.")
        else: print(tabulate(combined_data_display, headers=["Ticker", "Live Price", "Raw Score", "Basis Amplified %", "Final % Alloc"], tablefmt="pretty"))

    tailored_portfolio_output_list_final = [] 
    tailored_portfolio_structured_data = [] 
    final_cash_value_tailored = 0.0 

    if tailor_portfolio_requested:
        if total_value_singularity is None or safe_score(total_value_singularity) <= 0:
            print("Error: Tailored portfolio requested but total value is missing or invalid.")
            return [], combined_result_intermediate_calc, 0.0, [] 
        
        total_value_float = safe_score(total_value_singularity)
        current_tailored_entries_calc = [] 
        total_actual_money_allocated_stocks = 0.0

        for entry in final_combined_portfolio_data_calc:
            if entry['ticker'] == 'Cash': continue 

            final_stock_alloc_pct = safe_score(entry.get('combined_percent_allocation', 0.0))
            live_price = safe_score(entry.get('live_price', 0.0))

            if final_stock_alloc_pct > 1e-9 and live_price > 0: 
                target_allocation_value_for_ticker = total_value_float * (final_stock_alloc_pct / 100.0)
                shares = 0.0
                try:
                    exact_shares = target_allocation_value_for_ticker / live_price
                    if frac_shares_singularity: shares = round(exact_shares, 1) 
                    else: shares = float(math.floor(exact_shares))
                except ZeroDivisionError: shares = 0.0
                shares = max(0.0, shares) 
                actual_money_allocation_for_ticker = shares * live_price
                
                share_threshold = 0.1 if frac_shares_singularity else 1.0 
                if shares >= share_threshold: 
                    actual_percent_of_total_value = (actual_money_allocation_for_ticker / total_value_float) * 100.0 if total_value_float > 0 else 0.0
                    current_tailored_entries_calc.append({
                        'ticker': entry.get('ticker','ERR'), 
                        'raw_invest_score': entry.get('raw_invest_score', -float('inf')), 
                        'shares': shares, 
                        'actual_money_allocation': actual_money_allocation_for_ticker, 
                        'actual_percent_allocation': actual_percent_of_total_value 
                    })
                    total_actual_money_allocated_stocks += actual_money_allocation_for_ticker
        
        raw_remaining_value_after_stocks = total_value_float - total_actual_money_allocated_stocks
        final_cash_value_tailored = max(0.0, raw_remaining_value_after_stocks) 
        
        current_tailored_entries_calc.sort(key=lambda x: safe_score(x.get('raw_invest_score', -float('inf'))), reverse=True)
        tailored_portfolio_structured_data = current_tailored_entries_calc 

        if is_custom_command_simplified_output: 
            print("\n--- Tailored Portfolio (Shares) ---") 
            if current_tailored_entries_calc:
                if frac_shares_singularity:
                    tailored_portfolio_output_list_final = [f"{item['ticker']} - {item['shares']:.1f} shares" for item in current_tailored_entries_calc]
                else:
                    tailored_portfolio_output_list_final = [f"{item['ticker']} - {int(item['shares'])} shares" for item in current_tailored_entries_calc]
                print("\n".join(tailored_portfolio_output_list_final))
            else:
                print("No stocks allocated in the tailored portfolio based on the provided value and strategy.")
            print(f"Final Cash Value: ${safe_score(final_cash_value_tailored):,.2f}")
        else: 
            print("\n--- Tailored Portfolio (Full Details) ---") 
            tailored_portfolio_table_data_display = []
            for item in current_tailored_entries_calc: 
                tailored_portfolio_table_data_display.append([
                    item['ticker'], 
                    f"{item['shares']:.1f}" if frac_shares_singularity and item['shares'] > 0 else f"{int(item['shares'])}", 
                    f"${safe_score(item['actual_money_allocation']):,.2f}", 
                    f"{safe_score(item['actual_percent_allocation']):.2f}%"
                ])
            final_cash_percent_display = (final_cash_value_tailored / total_value_float) * 100.0 if total_value_float > 0 else 0.0
            final_cash_percent_display = max(0.0, min(100.0, final_cash_percent_display)) 
            tailored_portfolio_table_data_display.append(['Cash', '-', f"${safe_score(final_cash_value_tailored):,.2f}", f"{safe_score(final_cash_percent_display):.2f}%"])
            
            if not tailored_portfolio_table_data_display: print("No stocks allocated.") 
            else: print(tabulate(tailored_portfolio_table_data_display, headers=["Ticker", "Shares", "Actual $ Allocation", "Actual % Allocation"], tablefmt="pretty"))

        pie_chart_allocations_data = []
        if tailored_portfolio_structured_data: 
            for item in tailored_portfolio_structured_data:
                if item.get('actual_money_allocation', 0) > 1e-9: 
                    pie_chart_allocations_data.append({'ticker': item['ticker'], 'value': item['actual_money_allocation']})
        if final_cash_value_tailored > 1e-9: 
            pie_chart_allocations_data.append({'ticker': 'Cash', 'value': final_cash_value_tailored})

        if pie_chart_allocations_data:
            chart_title_custom = "Portfolio Allocation"
            portfolio_code_val = portfolio_data_config.get('portfolio_code') 
            if portfolio_code_val: 
                chart_title_custom = f"Custom Portfolio '{portfolio_code_val}' Allocation"
                if total_value_singularity: chart_title_custom += f" (Value: ${safe_score(total_value_singularity):,.0f})"
            elif 'ema_sensitivity' in portfolio_data_config: 
                chart_title_custom = f"Invest Portfolio Allocation (Sens: {portfolio_data_config.get('ema_sensitivity', 'N/A')})"
                if total_value_singularity: chart_title_custom += f" (Value: ${safe_score(total_value_singularity):,.0f})"
            elif total_value_singularity: 
                 chart_title_custom = f"Tailored Portfolio Allocation (Value: ${safe_score(total_value_singularity):,.0f})"
            
            generate_portfolio_pie_chart( 
                portfolio_allocations=pie_chart_allocations_data,
                chart_title=chart_title_custom,
                filename_prefix="singularity_portfolio_pie" 
            )
        print(f"Remaining Buying Power (Final Cash in Tailored Portfolio): ${safe_score(final_cash_value_tailored):,.2f}")

    return tailored_portfolio_output_list_final, combined_result_intermediate_calc, final_cash_value_tailored, tailored_portfolio_structured_data

def generate_portfolio_pie_chart(portfolio_allocations: List[Dict[str, Any]], chart_title: str, filename_prefix: str = "portfolio_pie") -> Optional[str]:
    """
    Generates a pie chart from portfolio allocation data and saves it.
    """
    if not portfolio_allocations:
        print("Pie Chart Error: No allocation data provided.")
        return None

    labels_orig = [item['ticker'] for item in portfolio_allocations]
    sizes_orig = [item['value'] for item in portfolio_allocations]
    # Filter out zero or negative values as they can't be plotted meaningfully
    valid_data = [{'ticker': l, 'value': s} for l, s in zip(labels_orig, sizes_orig) if s > 1e-9] # Use a small threshold

    if not valid_data:
        print("Pie Chart Error: No positive allocations to plot.")
        return None

    labels = [item['ticker'] for item in valid_data]
    sizes = [item['value'] for item in valid_data]
    total_value_chart = sum(sizes)

    # Group small slices if there are too many, to keep the chart readable
    threshold_percentage = 1.5  # Slices smaller than this percentage of total value might be grouped
    max_individual_slices = 14 # Show up to this many individual slices, plus one "Others" slice if needed

    if len(labels) > max_individual_slices + 1: 
        sorted_allocations = sorted(zip(sizes, labels), reverse=True) # Largest first
        display_labels, display_sizes, other_value = [], [], 0.0
        for i, (size, label) in enumerate(sorted_allocations):
            if i < max_individual_slices:
                display_labels.append(label)
                display_sizes.append(size)
            else: other_value += size
        if other_value > 1e-9: # If there's a non-negligible "Others" slice
            display_labels.append("Others")
            display_sizes.append(other_value)
        labels, sizes = display_labels, display_sizes # Update labels and sizes with grouped data
        if not labels: # Should not happen if valid_data was not empty
            print("Pie Chart Error: All slices were grouped or too small after attempting to group.")
            return None

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 8)) 
    
    custom_colors_list = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC', '#A0CBE8', '#FFBE7D', '#F4ADA8', '#B5D9D0', '#8CD17D']
    num_actual_slices = len(labels)
    if num_actual_slices == 0 : # Prevent error if all slices became 0 after filtering
        print("Pie Chart Error: No slices to plot after filtering small ones.")
        plt.close(fig)
        return None

    colors_to_use_list = custom_colors_list[:num_actual_slices] if num_actual_slices <= len(custom_colors_list) else [plt.cm.get_cmap('viridis', num_actual_slices)(i) for i in range(num_actual_slices)]
    
    # Explode the largest slice slightly if there are slices
    explode_values_list = [0.05 if i == 0 and num_actual_slices > 0 else 0 for i in range(num_actual_slices)]

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode_values_list, labels=None, # Labels will be in the legend
        autopct=lambda pct: f"{pct:.1f}%" if pct > threshold_percentage else '', # Show percentage for slices > threshold
        startangle=90, colors=colors_to_use_list, pctdistance=0.80,
        wedgeprops={'edgecolor': '#2c2f33', 'linewidth': 1}
    )
    for autotext in autotexts: # Style autopct text (percentages)
        autotext.set_color('white'); autotext.set_fontsize(9); autotext.set_fontweight('bold')
    
    ax.set_title(chart_title, fontsize=18, color='white', pad=25, fontweight='bold')
    ax.axis('equal')  # Ensures the pie chart is circular

    # Add a legend with percentages
    legend_labels_list = [f'{l} ({s/total_value_chart*100:.1f}%)' for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels_list,
              title="Holdings", loc="center left", bbox_to_anchor=(1.05, 0, 0.5, 1), # Position legend outside
              fontsize='medium', labelcolor='lightgrey', title_fontsize='large',
              facecolor='#36393f', edgecolor='grey')
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust rect to make space for legend
    
    unique_id_str = uuid.uuid4().hex[:8]
    filename = f"{filename_prefix}_{unique_id_str}.png"
    try:
        plt.savefig(filename, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        print(f"Pie chart saved: {filename}")
    except Exception as e:
        print(f"Error saving pie chart: {e}")
        plt.close(fig); return None # Ensure figure is closed on error
    plt.close(fig) # Close the figure to free up memory
    return filename

async def save_portfolio_to_csv(file_path: str, portfolio_data_to_save: Dict[str, Any]):
    """Saves portfolio configuration to the CSV database."""
    file_exists = os.path.isfile(file_path)
    # Define fieldnames based on typical structure, ensure portfolio_code is first
    fieldnames = ['portfolio_code', 'ema_sensitivity', 'amplification', 'num_portfolios', 'frac_shares', 'risk_tolerance', 'risk_type', 'remove_amplification_cap']
    # Add ticker and weight fields dynamically based on num_portfolios in the data
    num_portfolios_val = int(portfolio_data_to_save.get('num_portfolios', 0))
    for i in range(1, num_portfolios_val + 1):
        fieldnames.append(f'tickers_{i}')
        fieldnames.append(f'weight_{i}')
    
    # Ensure all keys from the data to save are included in fieldnames if not already present.
    # This makes the saving more robust if new fields are added to portfolio_data_to_save.
    for key in portfolio_data_to_save.keys():
        if key not in fieldnames:
            fieldnames.append(key) # Add any missing keys to the end

    try:
        with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
            # Use extrasaction='ignore' so if portfolio_data_to_save has fewer fields than fieldnames, it's okay.
            # If it has more, they will be ignored unless added to fieldnames above.
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction='ignore')
            if not file_exists or os.path.getsize(file_path) == 0:
                writer.writeheader()
            
            # Create a dictionary with only the keys present in fieldnames to ensure correct writing order
            # and to avoid writing extra fields not defined in the header (though extrasaction='ignore' helps).
            data_to_write_ordered = {key: portfolio_data_to_save.get(key) for key in fieldnames if key in portfolio_data_to_save}
            writer.writerow(data_to_write_ordered)
        print(f"Portfolio configuration '{portfolio_data_to_save.get('portfolio_code')}' saved to {file_path}")
    except IOError as e:
        print(f"Error writing to CSV {file_path}: {e}")
    except Exception as e:
        print(f"Unexpected error saving portfolio config to CSV: {e}")

async def save_portfolio_data_singularity(portfolio_code_to_save: str, date_str_to_save: str):
    """
    Internal function to save combined portfolio data for a given code and date (Singularity version).
    This saves the *output* of a portfolio analysis, not its configuration.
    """
    portfolio_config_data = None # This will hold the configuration for the portfolio code
    if not os.path.exists(portfolio_db_file): # Check for the database of configurations
        print(f"Error: Portfolio configuration database '{portfolio_db_file}' not found.")
        return

    # 1. Read the configuration for the given portfolio_code_to_save
    try:
        with open(portfolio_db_file, 'r', encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('portfolio_code', '').strip().lower() == portfolio_code_to_save.lower():
                    portfolio_config_data = row # Found the configuration
                    break
            if not portfolio_config_data: # If loop finishes and no config found
                print(f"Error: Portfolio code '{portfolio_code_to_save}' not found in configuration database '{portfolio_db_file}'.")
                return
    except Exception as e:
        print(f"Error reading portfolio configuration database {portfolio_db_file} for code {portfolio_code_to_save}: {e}")
        return

    # 2. If configuration is found, run the analysis to get the combined_result_for_save
    if portfolio_config_data and date_str_to_save:
        try:
            # Get frac_shares from the loaded config for the analysis run
            frac_shares_from_config = portfolio_config_data.get('frac_shares', 'false').lower() == 'true'
            
            print(f"Running analysis for portfolio '{portfolio_code_to_save}' to generate data for saving...")
            # Run process_custom_portfolio to get the combined results.
            # We are interested in the 'combined_result_intermediate_calc' part of its return.
            # No tailoring is needed for saving the *combined output percentages*.
            # is_custom_command_simplified_output=True will suppress printouts from process_custom_portfolio.
            _, combined_result_for_save, _ = await process_custom_portfolio(
                portfolio_data_config=portfolio_config_data, # Pass the loaded config
                tailor_portfolio_requested=False,      # Not tailoring for this type of save
                frac_shares_singularity=frac_shares_from_config, # Use config's frac_shares
                total_value_singularity=None,                  # No total value needed for combined % save
                is_custom_command_simplified_output=True # Suppress detailed table outputs
            )

            # 3. Save the relevant parts of combined_result_for_save
            if combined_result_for_save:
                # combined_result_for_save is a list of dicts. Each dict has:
                # 'ticker', 'live_price', 'raw_invest_score', 
                # 'amplified_score_adjusted', 'amplified_score_original',
                # 'portfolio_weight', 'score_was_adjusted',
                # 'portfolio_allocation_percent_adjusted', 'portfolio_allocation_percent_original',
                # 'combined_percent_allocation_adjusted', 'combined_percent_allocation_original'
                
                # We need to save: DATE, TICKER, PRICE, COMBINED_ALLOCATION_PERCENT
                # COMBINED_ALLOCATION_PERCENT should be 'combined_percent_allocation_adjusted'
                
                data_to_write_to_csv = []
                for item_to_save in combined_result_for_save:
                    # Filter out non-stock entries or entries with no significant allocation
                    if item_to_save.get('ticker') != 'Cash' and safe_score(item_to_save.get('combined_percent_allocation_adjusted', 0)) > 1e-4:
                        data_to_write_to_csv.append({
                            'DATE': date_str_to_save,
                            'TICKER': item_to_save.get('ticker', 'ERR'),
                            'PRICE': f"{safe_score(item_to_save.get('live_price')):.2f}" if item_to_save.get('live_price') is not None else "N/A",
                            'COMBINED_ALLOCATION_PERCENT': f"{safe_score(item_to_save.get('combined_percent_allocation_adjusted')):.2f}" if item_to_save.get('combined_percent_allocation_adjusted') is not None else "N/A"
                        })
                
                if not data_to_write_to_csv:
                    print(f"No stock data with allocation > 0 found for portfolio '{portfolio_code_to_save}' to save.")
                    return

                # Sort by COMBINED_ALLOCATION_PERCENT descending before saving
                sorted_data_to_write = sorted(data_to_write_to_csv, key=lambda x: float(x['COMBINED_ALLOCATION_PERCENT'].rstrip('%')) if x['COMBINED_ALLOCATION_PERCENT'] not in ["N/A", "ERR"] else -1, reverse=True)
                
                save_filename = f"portfolio_code_{portfolio_code_to_save}_data.csv" # Standard output file
                file_exists_check = os.path.isfile(save_filename)
                save_count = 0
                headers_for_save = ['DATE', 'TICKER', 'PRICE', 'COMBINED_ALLOCATION_PERCENT']
                
                with open(save_filename, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=headers_for_save)
                    if not file_exists_check or os.path.getsize(f.name) == 0: # Check size of opened file
                        writer.writeheader()
                    for row_to_write in sorted_data_to_write:
                        writer.writerow(row_to_write)
                        save_count += 1
                print(f"Saved {save_count} rows of combined portfolio data for code '{portfolio_code_to_save}' to '{save_filename}' for date {date_str_to_save}.")
            else:
                print(f"No valid combined portfolio data generated for code '{portfolio_code_to_save}' to save (analysis returned empty).")
        except Exception as e:
            print(f"Error processing or saving data for portfolio code {portfolio_code_to_save}: {e}")
            traceback.print_exc()


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
     # You can replace this with your custom ASCII art.
     # Make sure each line of your art starts at the beginning of the line 
     # within the triple quotes if you don't want leading spaces from the Python code indentation.
     # Or, you can use textwrap.dedent() if you prefer to indent the string in your code:
     # import textwrap
     # full_ascii_message = textwrap.dedent(r"""
     #     .--''''''--.    
     #   .'            '.  
     #  /   O      O   \ 
     # |    \  ^^  /    |
     #  \     `----'     / 
     #   '.            .'  
     #     .--''''''--.    
     #    ( MIC Singularity )
     #     `------------'    
     # """)


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

    command_lines.append("\nINVEST Commands")
    command_lines.append("-------------------")
    command_lines.append("/invest - Analyze multiple stocks based on EMA sensitivity and amplification.")
    command_lines.append("  Required inputs will be prompted if not called by AI.")
    command_lines.append("\n/custom - Run portfolio analysis using a saved code, create/save a new one, or save data.")
    command_lines.append("  Usage: /custom <portfolio_code_or_#> [save_data_with_code_3725]")
    command_lines.append("  Example 1 (run/create): /custom MYPORTFOLIO")
    command_lines.append("  Example 2 (next auto code): /custom #")
    command_lines.append("  Example 3 (save data for existing code): /custom MYPORTFOLIO 3725")
    command_lines.append("\n/quickscore - Get quick scores and graphs for a single ticker.")
    command_lines.append("  Usage: /quickscore <ticker>")
    command_lines.append("  Example: /quickscore AAPL")
    command_lines.append("\n/breakout - Run breakout analysis or save current breakout data.")
    command_lines.append("  Usage: /breakout [save_data_with_code_3725]")
    command_lines.append("  Example 1 (run analysis): /breakout")
    command_lines.append("  Example 2 (save data): /breakout 3725")
    command_lines.append("\n/market - Display market scores or save full market data.")
    command_lines.append("  Usage: /market [save_data_with_code_3725 or AI params for action/sensitivity/date]")
    command_lines.append("  Example 1 (display scores CLI): /market")
    command_lines.append("  Example 2 (save data CLI): /market 3725")
    command_lines.append("\n/cultivate - Craft a Cultivate portfolio or save its data.")
    command_lines.append("  Usage: /cultivate <Code A/B> <PortfolioValue> <FracShares yes/no> [save_code 3725 or AI params]")
    command_lines.append("  Example (run CLI): /cultivate A 10000 yes")
    command_lines.append("  Example (save CLI): /cultivate B 50000 no 3725")
    command_lines.append("\n/assess - Assess stock volatility, portfolio risk, etc.")
    command_lines.append("  Usage: /assess <AssessCode A/B/C/D> [additional_args... or AI params]")
    command_lines.append("    A (Stock): /assess A <tickers_comma_sep> <timeframe 1Y/3M/1M> <risk_tolerance 1-5>")
    command_lines.append("       Example CLI: /assess A AAPL,MSFT 1Y 3")
    command_lines.append("    B (Manual Portfolio): /assess B <backtest_period 1y/5y/10y> (Prompts for holdings if CLI)")
    command_lines.append("       Example CLI: /assess B 1y")
    command_lines.append("    C (Custom Portfolio Risk): /assess C <custom_portfolio_code> <value_for_assessment> <backtest_period 1y/3y/5y/10y>")
    command_lines.append("       Example CLI: /assess C MYPORTFOLIO 25000 3y")
    command_lines.append("    D (Cultivate Portfolio Risk): /assess D <cultivate_code A/B> <value_epsilon> <frac_shares y/n> <backtest_period 1y/3y/5y/10y>")
    command_lines.append("       Example CLI: /assess D A 50000 yes 5y")

    # --- RISK Commands Section (Moved Before AI) ---
    command_lines.append("\nRISK Commands")
    command_lines.append("-------------------")
    command_lines.append("/risk - Perform one-time RISK module calculations, display results, and save data. Can be AI-driven via /ai.")
    command_lines.append("  CLI Usage: /risk [eod]")
    command_lines.append("  CLI Example (standard run): /risk")
    command_lines.append("  CLI Example (EOD specific save): /risk eod")
    command_lines.append("\n/history - Generate and save historical RISK module graphs. Can be AI-driven via /ai.")
    command_lines.append("  CLI Usage: /history")
    
    # --- AI Command Section ---
    command_lines.append("\nAI Command")
    command_lines.append("-------------------")
    command_lines.append("/ai - Interact with an AI to perform tasks using natural language.")
    command_lines.append("  Usage: /ai <your natural language prompt>")
    command_lines.append("  Examples:")
    command_lines.append("    /ai show me the breakout stocks then quickscore the top one")
    command_lines.append("    /ai run the daily market analysis scores")
    command_lines.append("    /ai perform the end-of-day risk assessment")
    command_lines.append("    /ai assess stock volatility for NVDA and AMD over 3 months with risk tolerance 4")
    command_lines.append("    /ai generate the risk history graphs")
    command_lines.append("    /ai run my custom portfolio 'AlphaPicks' and tailor it to $75000 with fractional shares")
    command_lines.append("    /ai save the data for custom portfolio 'BetaGrowth' for today")
    command_lines.append("    /ai I want to run an invest analysis with hourly sensitivity and 1.5x amplification. Portfolio one is QQQ, SPY with 70% weight. Portfolio two is GLD with 30% weight. Tailor to $100k.")
    command_lines.append("    /ai execute cultivate analysis code A for $20000 value without fractional shares, then save the results for today.")

    command_lines.append("\nUtility Commands")
    command_lines.append("-------------------")
    command_lines.append("/help - Display this list of commands.")
    command_lines.append("/exit - Close the Market Insights Center Singularity.")
    command_lines.append("-------------------\n")

    # Join all lines into a single string
    full_command_text = "\n".join(command_lines)
    
    # Typing animation
    typing_speed = 0.002  # Adjust this value for faster/slower typing (e.g., 0.01 is slower, 0.001 is faster)
    for char_cmd in full_command_text:
        print(char_cmd, end="", flush=True)
        py_time.sleep(typing_speed)
    print() # Ensure a final newline after the animation

async def handle_invest_command(args: List[str], ai_params: Optional[Dict] = None): # args is for CLI compatibility, not used by AI path
    """
    Handles the /invest command for CLI and AI.
    AI call expects ai_params with: ema_sensitivity, amplification, sub_portfolios (list of dicts),
    and optional tailoring info (tailor_to_value, total_value, use_fractional_shares).
    Returns a summary string for AI. CLI path prints directly and returns None.
    """
    print("\n--- /invest Command ---")
    
    # This dictionary will be populated either by AI params or CLI input
    portfolio_data_config_invest = {
        'risk_type': 'stock', 
        'risk_tolerance': '10', # Default from original script
        'remove_amplification_cap': 'true' # Default from original script
    }
    tailor_portfolio_for_run = False
    total_value_for_tailoring_run = None
    frac_shares_for_tailoring_run = False

    if ai_params: # Called by AI
        print(f"AI Call to /invest with params: {json.dumps(ai_params, indent=2)}") # Ensure json is imported
        try:
            ema_sensitivity = int(ai_params.get("ema_sensitivity"))
            if ema_sensitivity not in [1, 2, 3]:
                return "Error (AI /invest): Invalid 'ema_sensitivity'. Must be 1, 2, or 3."
            portfolio_data_config_invest['ema_sensitivity'] = str(ema_sensitivity)

            amplification = float(ai_params.get("amplification"))
            portfolio_data_config_invest['amplification'] = str(amplification)

            sub_portfolios_data = ai_params.get("sub_portfolios")
            if not sub_portfolios_data or not isinstance(sub_portfolios_data, list) or not sub_portfolios_data:
                return "Error (AI /invest): 'sub_portfolios' list is required and must not be empty."
            
            portfolio_data_config_invest['num_portfolios'] = str(len(sub_portfolios_data))
            
            current_total_weight_ai = 0.0
            for i, sub_p_data in enumerate(sub_portfolios_data, 1):
                tickers_input = sub_p_data.get("tickers")
                weight_val_num = sub_p_data.get("weight") 
                if not tickers_input or weight_val_num is None:
                    return f"Error (AI /invest): Sub-portfolio {i} is missing 'tickers' or 'weight'."
                try:
                    weight_val = float(weight_val_num)
                    if not (0 <= weight_val <= 100):
                         return f"Error (AI /invest): Weight for sub-portfolio {i} ({weight_val}%) must be between 0 and 100."
                    current_total_weight_ai += weight_val
                except ValueError:
                    return f"Error (AI /invest): Weight for sub-portfolio {i} ('{weight_val_num}') must be a number."

                portfolio_data_config_invest[f'tickers_{i}'] = str(tickers_input).upper()
                portfolio_data_config_invest[f'weight_{i}'] = f"{weight_val:.2f}"

            if not math.isclose(current_total_weight_ai, 100.0, abs_tol=1.0): # math needs to be imported
                return f"Error (AI /invest): Sum of 'weight' for all 'sub_portfolios' must be close to 100. Got {current_total_weight_ai:.2f}."

            tailor_portfolio_for_run = ai_params.get("tailor_to_value", False)
            if tailor_portfolio_for_run:
                total_value_for_tailoring_run = ai_params.get("total_value")
                if total_value_for_tailoring_run is None or float(total_value_for_tailoring_run) <= 0:
                    return "Error (AI /invest): If 'tailor_to_value' is true, a positive 'total_value' is required."
                total_value_for_tailoring_run = float(total_value_for_tailoring_run)
            
            frac_shares_for_tailoring_run = ai_params.get("use_fractional_shares", False)
            portfolio_data_config_invest['frac_shares'] = str(frac_shares_for_tailoring_run).lower()
            
            # Common processing logic for AI path
            print("\nAI: Processing /invest request configured by AI...")
            # Ensure process_custom_portfolio is defined and async
            tailored_list_str_output, combined_calc_data, final_cash_val, _ = await process_custom_portfolio(
                portfolio_data_config=portfolio_data_config_invest,
                tailor_portfolio_requested=tailor_portfolio_for_run,
                frac_shares_singularity=frac_shares_for_tailoring_run,
                total_value_singularity=total_value_for_tailoring_run,
                is_custom_command_simplified_output=tailor_portfolio_for_run 
            )
            
            summary_for_ai = f"/invest analysis completed (EMA Sens: {portfolio_data_config_invest['ema_sensitivity']}, Amplification: {portfolio_data_config_invest['amplification']}). "
            if tailor_portfolio_for_run:
                summary_for_ai += f"Tailored to ${total_value_for_tailoring_run:,.2f} (FracShares: {frac_shares_for_tailoring_run}). Final Cash: ${final_cash_val:,.2f}. "
                if tailored_list_str_output: 
                    summary_for_ai += "Top holdings: " + ", ".join(tailored_list_str_output[:3]) + ("..." if len(tailored_list_str_output) > 3 else "")
                else: summary_for_ai += "No stock holdings after tailoring."
            else: 
                if combined_calc_data: # combined_calc_data is List[Dict]
                     summary_for_ai += "Top combined allocations: " + ", ".join([f"{d['ticker']}({d.get('combined_percent_allocation_adjusted', d.get('combined_percent_allocation',0)):.1f}%)" for d in combined_calc_data[:3] if 'ticker' in d])
                else: summary_for_ai += "No combined allocation data to summarize."
            return summary_for_ai

        except KeyError as ke: return f"Error (AI /invest): Missing expected parameter from AI: {ke}"
        except ValueError as ve: return f"Error (AI /invest): Invalid data type for a parameter: {ve}"
        except Exception as e_invest_ai:
            traceback.print_exc(); return f"Error (AI /invest) processing request: {e_invest_ai}"

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
        
async def collect_portfolio_inputs_singularity(portfolio_code_singularity: str) -> Optional[Dict[str, Any]]:
    """Collects all inputs for a new custom portfolio configuration via Singularity."""
    print(f"\n--- Creating New Portfolio Configuration: '{portfolio_code_singularity}' ---")
    inputs_singularity = {'portfolio_code': portfolio_code_singularity}
    # portfolio_weights_collected = [] # Not strictly needed here as weights are summed directly

    try:
        ema_sens_str = input("Enter EMA sensitivity (1: Weekly, 2: Daily, 3: Hourly): ")
        ema_sens_val = int(ema_sens_str)
        if ema_sens_val not in [1,2,3]: raise ValueError("Invalid EMA sensitivity. Must be 1, 2, or 3.")
        inputs_singularity['ema_sensitivity'] = str(ema_sens_val)

        valid_amplifications = [0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
        amp_str = input(f"Enter amplification ({', '.join(map(str, valid_amplifications))}): ")
        amp_val = float(amp_str)
        if amp_val not in valid_amplifications: raise ValueError(f"Invalid amplification. Choose from {valid_amplifications}.")
        inputs_singularity['amplification'] = str(amp_val)

        num_port_str = input("Enter the number of sub-portfolios (e.g., 2): ")
        num_portfolios_singularity = int(num_port_str)
        if num_portfolios_singularity <= 0: raise ValueError("Number of sub-portfolios must be greater than 0.")
        inputs_singularity['num_portfolios'] = str(num_portfolios_singularity)

        frac_s_str = input("Allow fractional shares for tailoring this portfolio configuration? (yes/no): ").lower()
        if frac_s_str not in ['yes', 'no']: raise ValueError("Invalid input for fractional shares. Enter 'yes' or 'no'.")
        inputs_singularity['frac_shares'] = 'true' if frac_s_str == 'yes' else 'false'
        
        # Add fixed values from original collect_portfolio_inputs structure
        inputs_singularity['risk_tolerance'] = '10' # Default, not typically user-set for /custom config
        inputs_singularity['risk_type'] = 'stock'   # Default
        inputs_singularity['remove_amplification_cap'] = 'true' # Default

        current_total_weight_singularity = 0.0
        for i in range(1, num_portfolios_singularity + 1):
            print(f"\n--- Sub-Portfolio {i} ---")
            tickers_input_singularity = input(f"Enter tickers for Sub-Portfolio {i} (comma-separated, e.g., GOOG,AMZN): ").upper()
            if not tickers_input_singularity.strip(): raise ValueError("Tickers cannot be empty for a sub-portfolio.")
            inputs_singularity[f'tickers_{i}'] = tickers_input_singularity

            if i == num_portfolios_singularity: # Last portfolio, weight is auto-calculated
                weight_val_singularity = 100.0 - current_total_weight_singularity
                if weight_val_singularity < -0.01 : # Allow small tolerance for floating point arithmetic
                    print(f"Error: Previous weights sum to {current_total_weight_singularity}%, which exceeds 100%. Cannot set weight for the final portfolio.")
                    return None
                weight_val_singularity = max(0, weight_val_singularity) # Ensure weight is not negative
                print(f"Weight for Sub-Portfolio {i} automatically set to: {weight_val_singularity:.2f}%")
            else: # For portfolios before the last one
                remaining_weight_singularity = 100.0 - current_total_weight_singularity
                weight_str_singularity = input(f"Enter weight for Sub-Portfolio {i} (0-{remaining_weight_singularity:.2f}%): ")
                weight_val_singularity = float(weight_str_singularity)
                # Check if weight is within valid range (0 to remaining, allowing for float tolerance)
                if not (-0.01 < weight_val_singularity < remaining_weight_singularity + 0.01):
                    print(f"Invalid weight. Must be a positive number up to the remaining {remaining_weight_singularity:.2f}%.")
                    return None
            inputs_singularity[f'weight_{i}'] = f"{weight_val_singularity:.2f}" # Store weight as string, formatted
            current_total_weight_singularity += weight_val_singularity
        
        if not math.isclose(current_total_weight_singularity, 100.0, abs_tol=0.1): # Final check on total weight
            print(f"Warning: The sum of all sub-portfolio weights is {current_total_weight_singularity:.2f}%, which is not 100%. The configuration will be saved, but analysis results might be affected.")

        return inputs_singularity
    except ValueError as ve:
        print(f"Invalid input: {ve}. Portfolio configuration not saved. Please try again.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during input collection: {e}")
        traceback.print_exc()
        return None


async def handle_custom_command(args: List[str], ai_params: Optional[Dict] = None):
    """
    Handles the /custom command for CLI and AI.
    CLI: /custom <portfolio_code_or_#> [save_data_code]
    AI: Called with ai_params={"portfolio_code": "...", "tailor_to_value": bool, ...}
        AI mode currently only supports running an EXISTING portfolio.
    Returns a summary string for AI, prints to console for CLI/AI.
    """
    print("\n--- /custom Command ---")
    summary_for_ai = "Custom command initiated."

    if ai_params: # Called by AI - for running an existing portfolio
        print(f"AI Call to /custom with params: {ai_params}")
        portfolio_code_input = ai_params.get("portfolio_code")
        if not portfolio_code_input:
            return "Error for AI (/custom): 'portfolio_code' is required to run an existing portfolio."

        tailor_portfolio_custom_run = ai_params.get("tailor_to_value", False)
        total_value_custom_run = None
        frac_shares_custom_run = None # Will use portfolio's default unless AI overrides

        if tailor_portfolio_custom_run:
            total_value_custom_run = ai_params.get("total_value")
            if total_value_custom_run is None or float(total_value_custom_run) <= 0:
                return "Error for AI (/custom): If 'tailor_to_value' is true, a positive 'total_value' is required."
            total_value_custom_run = float(total_value_custom_run)
            
            # AI can optionally override portfolio's frac_shares setting for this run
            if "use_fractional_shares" in ai_params:
                frac_shares_custom_run = bool(ai_params.get("use_fractional_shares"))
        
        portfolio_config_from_csv = None
        if not os.path.exists(portfolio_db_file): # portfolio_db_file needs to be global
            return f"Error for AI (/custom): Portfolio database '{portfolio_db_file}' not found."
        
        try:
            with open(portfolio_db_file, 'r', encoding='utf-8', newline='') as file_custom:
                reader_custom = csv.DictReader(file_custom)
                for row_csv in reader_custom:
                    if row_csv.get('portfolio_code', '').strip().lower() == portfolio_code_input.lower():
                        portfolio_config_from_csv = row_csv
                        break
            if not portfolio_config_from_csv:
                return f"Error for AI (/custom): Portfolio code '{portfolio_code_input}' not found in database."

            # Determine fractional shares: AI override > portfolio config > default (false)
            if frac_shares_custom_run is None: # AI did not specify, use portfolio's setting
                frac_shares_final_run = portfolio_config_from_csv.get('frac_shares', 'false').lower() == 'true'
            else: # AI specified an override
                frac_shares_final_run = frac_shares_custom_run

            print(f"\nAI: Processing custom portfolio code: `{portfolio_code_input}`...")
            # process_custom_portfolio needs to be defined and async
            # It returns: tailored_output_list_final, combined_result_intermediate_calc, final_cash_value_tailored, tailored_portfolio_structured_data
            tailored_list, combined_calc, final_cash, structured_data = await process_custom_portfolio(
                portfolio_data_config=portfolio_config_from_csv,
                tailor_portfolio_requested=tailor_portfolio_custom_run,
                frac_shares_singularity=frac_shares_final_run,
                total_value_singularity=total_value_custom_run,
                is_custom_command_simplified_output=tailor_portfolio_custom_run # True if tailoring to simplify console output for AI
            )
            
            summary_for_ai = f"Analysis for custom portfolio '{portfolio_code_input}' completed. "
            if tailor_portfolio_custom_run:
                summary_for_ai += f"Tailored to ${total_value_custom_run:,.2f} with fractional shares: {frac_shares_final_run}. "
                summary_for_ai += f"Final cash: ${final_cash:,.2f}. Holdings: "
                if tailored_list:
                    summary_for_ai += ", ".join(tailored_list[:3]) + ("..." if len(tailored_list) > 3 else "")
                else:
                    summary_for_ai += "No stock holdings after tailoring."
            else:
                # Summarize combined portfolio if not tailoring (combined_calc might be more relevant)
                # This part might need more details based on what 'combined_calc' contains
                if combined_calc:
                     summary_for_ai += "Top combined allocations: " + ", ".join([f"{d['ticker']}({d.get('combined_percent_allocation', 0):.1f}%)" for d in combined_calc[:3]])
                else:
                    summary_for_ai += "No combined allocation data to summarize."
            return summary_for_ai

        except FileNotFoundError:
            return f"Error for AI (/custom): Portfolio database '{portfolio_db_file}' not found."
        except KeyError as e_key:
            return f"Error for AI (/custom): Configuration for '{portfolio_code_input}' seems incomplete. Missing key: {e_key}"
        except ValueError as e_val:
            return f"Error for AI (/custom): Invalid numerical input for portfolio value or other parameters. {e_val}"
        except Exception as e_custom_ai:
            traceback.print_exc()
            return f"Error for AI (/custom) processing '{portfolio_code_input}': {e_custom_ai}"

    else: # --- CLI Path ---
        if not args:
            # This matches your original script's help display for /custom if no args
            print("Usage: /custom <portfolio_code_or_#> [save_data_code]")
            print("Example (run/create existing or new): /custom MYPORTFOLIO")
            print("Example (auto-generate next numeric code): /custom #")
            print("Example (save combined data for existing code MYPORTFOLIO): /custom MYPORTFOLIO 3725")
            return None # No summary for AI if CLI path

        portfolio_code_input_cli = args[0].strip()
        save_data_code_singularity_cli = args[1].strip() if len(args) > 1 else None
        is_new_code_auto_singularity_cli = False

        if portfolio_code_input_cli == '#':
            # ... (your existing logic for auto-generating next numeric code for CLI) ...
            # This part is complex and involves reading portfolio_db_file
            # For brevity in this refactor, assuming this logic is preserved from your original script
            next_code_num = 1 
            if os.path.exists(portfolio_db_file):
                max_code = 0
                try:
                    # (Your logic to find max_code) ...
                    df_codes = pd.read_csv(portfolio_db_file)
                    numeric_codes = pd.to_numeric(df_codes['portfolio_code'], errors='coerce').dropna()
                    if not numeric_codes.empty: max_code = int(numeric_codes.max())
                except Exception: pass # Ignore errors in finding max code, will default to 1
                next_code_num = max_code + 1
            portfolio_code_input_cli = str(next_code_num)
            is_new_code_auto_singularity_cli = True
            print(f"CLI: Using next available portfolio code: `{portfolio_code_input_cli}`")


        if save_data_code_singularity_cli == "3725":
            # ... (your existing logic for saving data for CLI, calls save_portfolio_data_singularity) ...
            # This part requires interactive input for date
            if is_new_code_auto_singularity_cli:
                print("CLI: Cannot use '#' (auto-generated code) directly with save_data_code.")
                return None
            date_to_save_str_cli = input(f"CLI: Enter date (MM/DD/YYYY) to save combined data for portfolio '{portfolio_code_input_cli}': ")
            try:
                datetime.strptime(date_to_save_str_cli, '%m/%d/%Y')
                print(f"CLI: Attempting to save combined portfolio output data for code: `{portfolio_code_input_cli}` for date {date_to_save_str_cli}...")
                await save_portfolio_data_singularity(portfolio_code_input_cli, date_to_save_str_cli) # Needs to be defined
            except ValueError:
                print("CLI: Invalid date format. Save operation cancelled.")
            return None # CLI action complete

        # --- CLI: Run Analysis or Create New Configuration ---
        portfolio_config_from_csv_cli = None
        if os.path.exists(portfolio_db_file) and not is_new_code_auto_singularity_cli:
            # ... (your existing logic to load portfolio_config_from_csv_cli) ...
            try:
                with open(portfolio_db_file, 'r', encoding='utf-8', newline='') as file_cli:
                    reader_cli = csv.DictReader(file_cli)
                    for row_cli in reader_cli:
                        if row_cli.get('portfolio_code', '').strip().lower() == portfolio_code_input_cli.lower():
                            portfolio_config_from_csv_cli = row_cli; break
            except Exception as e_read_db: print(f"CLI: Error reading portfolio DB: {e_read_db}")

        if portfolio_config_from_csv_cli is None: # Not found or auto-generating new
            # ... (your existing logic for CLI: calls collect_portfolio_inputs_singularity, then save_portfolio_to_csv) ...
            # This is highly interactive.
            print(f"CLI: Portfolio code '{portfolio_code_input_cli}' not found or creating new. Starting interactive setup...")
            # collect_portfolio_inputs_singularity needs to be defined and async
            new_portfolio_config_cli = await collect_portfolio_inputs_singularity(portfolio_code_input_cli)
            if new_portfolio_config_cli:
                await save_portfolio_to_csv(portfolio_db_file, new_portfolio_config_cli) # Needs to be defined
                portfolio_config_from_csv_cli = new_portfolio_config_cli
            else:
                print("CLI: Portfolio configuration cancelled or incomplete.")
                return None
        
        if portfolio_config_from_csv_cli:
            try:
                # CLI: Interactive tailoring prompt
                tailor_cli_str = input(f"CLI: Tailor portfolio '{portfolio_code_input_cli}' to a specific value for this run? (yes/no): ").lower()
                tailor_portfolio_cli_run = tailor_cli_str == 'yes'
                total_value_cli_run = None
                frac_shares_cli_run = portfolio_config_from_csv_cli.get('frac_shares', 'false').lower() == 'true'

                if tailor_portfolio_cli_run:
                    val_cli_str = input("CLI: Enter total portfolio value: ")
                    total_value_cli_run = float(val_cli_str)
                    if total_value_cli_run <=0:
                        print("CLI: Portfolio value must be positive. Proceeding without tailoring.")
                        tailor_portfolio_cli_run = False
                    override_frac_s_cli_str = input(f"CLI: Use fractional shares (config is '{frac_shares_cli_run}')? (yes/no/config): ").lower()
                    if override_frac_s_cli_str == 'yes': frac_shares_cli_run = True
                    elif override_frac_s_cli_str == 'no': frac_shares_cli_run = False
                
                print(f"\nCLI: Processing custom portfolio code: `{portfolio_code_input_cli}`...")
                await process_custom_portfolio(
                    portfolio_data_config=portfolio_config_from_csv_cli,
                    tailor_portfolio_requested=tailor_portfolio_cli_run,
                    frac_shares_singularity=frac_shares_cli_run,
                    total_value_singularity=total_value_cli_run,
                    is_custom_command_simplified_output=tailor_portfolio_cli_run
                )
                print(f"\nCLI: Custom portfolio analysis for `{portfolio_code_input_cli}` complete.")
            except Exception as e_custom_cli:
                print(f"CLI Error processing portfolio '{portfolio_code_input_cli}': {e_custom_cli}")
                traceback.print_exc()
        return None # CLI path doesn't return a summary for AI

# --- Breakout Command Functions ---
async def run_breakout_analysis_singularity():
    """
    Performs breakout analysis, prints results to console, saves data to CSV,
    and returns a list of dictionaries, each containing details of top breakout stocks.
    """
    print("\n--- Running Breakout Analysis (AI Callable) ---")
    invest_score_threshold = 100.0  # From your original script
    fraction_threshold = 3.0 / 4.0  # From your original script
    
    updated_data_breakout_for_return = [] # This will be returned
    
    # --- Start of logic from your original run_breakout_analysis_singularity ---
    existing_tickers_data_breakout = {}
    if os.path.exists(BREAKOUT_TICKERS_FILE):
        try:
            df_existing = pd.read_csv(BREAKOUT_TICKERS_FILE)
            if not df_existing.empty:
                for col in ["Highest Invest Score", "Lowest Invest Score", "Live Price", "1Y% Change", "Invest Score"]:
                    if col in df_existing.columns:
                        if df_existing[col].dtype == 'object':
                            df_existing[col] = df_existing[col].astype(str).str.replace('%', '', regex=False).str.replace('$', '', regex=False).str.strip()
                        df_existing[col] = pd.to_numeric(df_existing[col], errors='coerce')
                existing_tickers_data_breakout = df_existing.set_index('Ticker').to_dict('index')
        except Exception as read_err:
            print(f"Error reading '{BREAKOUT_TICKERS_FILE}': {read_err}. Proceeding without existing data.")

    print("Running TradingView Screening for new breakout candidates...")
    new_tickers_breakout_tv = []
    try:
        query = Query().select('name', 'close', 'change', 'volume', 'market_cap_basic', 'change|1W', 'average_volume_90d_calc') \
            .where(
                Column('market_cap_basic') >= 1_000_000_000, Column('volume') >= 1_000_000,
                Column('change|1W') >= 20, Column('close') >= 1,
                Column('average_volume_90d_calc') >= 1_000_000
            ).order_by('change', ascending=False).limit(100)
        scanner_results = await asyncio.to_thread(query.get_scanner_data, timeout=60) # Ensure query is defined

        if scanner_results and isinstance(scanner_results, tuple) and len(scanner_results) > 0 and isinstance(scanner_results[1], pd.DataFrame):
            new_tickers_df_breakout = scanner_results[1]
            if 'name' in new_tickers_df_breakout.columns:
                new_tickers_breakout_tv = [str(t).split(':')[-1].replace('.', '-') for t in new_tickers_df_breakout['name'].tolist() if pd.notna(t)]
                new_tickers_breakout_tv = sorted(list(set(new_tickers_breakout_tv)))
                print(f"Screening found {len(new_tickers_breakout_tv)} potential new tickers.")
        else:
            print("Warning: TradingView screening returned no data or unexpected format.")
    except Exception as screen_err:
        print(f"Error during TradingView screening: {screen_err}")

    all_tickers_to_process_breakout = sorted(list(set(list(existing_tickers_data_breakout.keys()) + new_tickers_breakout_tv)))
    print(f"Processing {len(all_tickers_to_process_breakout)} unique tickers (existing + new)...")
    
    temp_updated_data_for_df = [] # Used to build the DataFrame for CSV and console

    processed_count = 0
    for ticker_b in all_tickers_to_process_breakout:
        processed_count += 1
        if processed_count % 20 == 0:
            print(f"  ...breakout processing {processed_count}/{len(all_tickers_to_process_breakout)}")
        try:
            # Ensure calculate_ema_invest and calculate_one_year_invest are defined and async
            live_price_b, current_invest_score_raw_b = await calculate_ema_invest(ticker_b, 2) 
            one_year_change_raw_b, _ = await calculate_one_year_invest(ticker_b)

            current_invest_score_b = safe_score(current_invest_score_raw_b) if current_invest_score_raw_b is not None else None
            live_price_val_b = safe_score(live_price_b) if live_price_b is not None else None
            one_year_change_val_b = safe_score(one_year_change_raw_b) if one_year_change_raw_b is not None else None

            existing_entry_b = existing_tickers_data_breakout.get(ticker_b, {})
            highest_score_prev_b = safe_score(existing_entry_b.get("Highest Invest Score")) if pd.notna(existing_entry_b.get("Highest Invest Score")) else -float('inf')
            lowest_score_prev_b = safe_score(existing_entry_b.get("Lowest Invest Score")) if pd.notna(existing_entry_b.get("Lowest Invest Score")) else float('inf')

            highest_invest_score_b = highest_score_prev_b
            lowest_invest_score_b = lowest_score_prev_b

            if current_invest_score_b is not None:
                if highest_invest_score_b == -float('inf') or current_invest_score_b > highest_invest_score_b:
                    highest_invest_score_b = current_invest_score_b
                if lowest_invest_score_b == float('inf') or current_invest_score_b < lowest_invest_score_b:
                    lowest_invest_score_b = current_invest_score_b
            
            remove_ticker_b = False
            if current_invest_score_b is None: remove_ticker_b = True
            elif highest_invest_score_b > -float('inf') and highest_invest_score_b > 0:
                if (current_invest_score_b > 600 or current_invest_score_b < invest_score_threshold or \
                    current_invest_score_b < fraction_threshold * highest_invest_score_b):
                    remove_ticker_b = True
            elif current_invest_score_b < invest_score_threshold: remove_ticker_b = True
            
            if not remove_ticker_b:
                status_b = "Repeat" if ticker_b in existing_tickers_data_breakout else "New"
                stock_data = {
                    "Ticker": ticker_b,
                    "Live Price": f"{live_price_val_b:.2f}" if live_price_val_b is not None else "N/A",
                    "Invest Score": f"{current_invest_score_b:.2f}%" if current_invest_score_b is not None else "N/A",
                    "Highest Invest Score": f"{highest_invest_score_b:.2f}%" if highest_invest_score_b > -float('inf') else "N/A",
                    "Lowest Invest Score": f"{lowest_invest_score_b:.2f}%" if lowest_invest_score_b < float('inf') else "N/A",
                    "1Y% Change": f"{one_year_change_val_b:.2f}%" if one_year_change_val_b is not None else "N/A",
                    "Status": status_b,
                    "_sort_score_internal": current_invest_score_b if current_invest_score_b is not None else -float('inf') # For sorting
                }
                temp_updated_data_for_df.append(stock_data)
        except Exception as e_ticker_b:
            print(f"Error processing breakout logic for ticker {ticker_b}: {e_ticker_b}")
    
    # Sort data before saving and for AI return
    temp_updated_data_for_df.sort(key=lambda x: x['_sort_score_internal'], reverse=True)
    
    # Prepare data for DataFrame (remove internal sort key)
    df_data = [{k: v for k, v in item.items() if k != '_sort_score_internal'} for item in temp_updated_data_for_df]
    updated_data_breakout_for_return = df_data # This is what will be returned to AI (potentially trimmed)

    if df_data:
        final_df_breakout = pd.DataFrame(df_data)
        try:
            final_df_breakout.to_csv(BREAKOUT_TICKERS_FILE, index=False) # BREAKOUT_TICKERS_FILE needs to be global
            print(f"Successfully saved current breakout data to '{BREAKOUT_TICKERS_FILE}'.")
        except IOError as e_io_b:
            print(f"Error writing breakout data to '{BREAKOUT_TICKERS_FILE}': {e_io_b}")

        # Console output
        print("\n--- Breakout Analysis Results (for User Console) ---")
        display_cols_console = ["Ticker", "Live Price", "Invest Score", "Status"]
        # Ensure the columns for display exist in the DataFrame
        actual_display_cols = [col for col in display_cols_console if col in final_df_breakout.columns]
        if actual_display_cols:
             print(tabulate(final_df_breakout[actual_display_cols].head(15), headers="keys", tablefmt="pretty", showindex=False)) # Show top 15
        else:
            print("No columns available for breakout display.")

    else:
        print("No tickers met the breakout criteria to update the file or display.")

    print("--- Breakout Analysis Complete ---")

    if not updated_data_breakout_for_return:
        print("Breakout analysis did not yield any tickers to return to AI.")
        return [] 
        
    return updated_data_breakout_for_return[:10] # Return top 10 detailed results for AI

async def save_breakout_data_singularity(date_str: str):
    """Saves the current breakout data from BREAKOUT_TICKERS_FILE to BREAKOUT_HISTORICAL_DB_FILE for a given date.
    Returns a summary string."""
    print(f"\n--- Saving Breakout Data for Date: {date_str} ---")
    # BREAKOUT_TICKERS_FILE and BREAKOUT_HISTORICAL_DB_FILE need to be global
    if not os.path.exists(BREAKOUT_TICKERS_FILE):
        msg = f"Error: Current breakout data file '{BREAKOUT_TICKERS_FILE}' not found. Cannot save historical data."
        print(msg)
        return msg

    save_count = 0
    try:
        df_current_breakout = pd.read_csv(BREAKOUT_TICKERS_FILE) # pd needs to be imported
        if df_current_breakout.empty:
            msg = f"Info: Current breakout file '{BREAKOUT_TICKERS_FILE}' is empty. Nothing to save to historical DB."
            print(msg)
            return msg

        historical_data_to_save = []
        for _, row in df_current_breakout.iterrows():
            price_str = str(row.get('Live Price', 'N/A')).replace('$', '')
            score_str = str(row.get('Invest Score', 'N/A')).replace('%', '')
            
            historical_data_to_save.append({
                'DATE': date_str,
                'TICKER': row.get('Ticker', 'ERR'),
                'PRICE': f"{safe_score(price_str):.2f}" if safe_score(price_str) is not None and not pd.isna(safe_score(price_str)) else "N/A", # safe_score needs to be defined
                'INVEST_SCORE': f"{safe_score(score_str):.2f}" if safe_score(score_str) is not None and not pd.isna(safe_score(score_str)) else "N/A"
            })
        
        file_exists_hist = os.path.isfile(BREAKOUT_HISTORICAL_DB_FILE)
        headers_hist = ['DATE', 'TICKER', 'PRICE', 'INVEST_SCORE']
        with open(BREAKOUT_HISTORICAL_DB_FILE, 'a', newline='', encoding='utf-8') as f_hist: # csv needs to be imported
            writer_hist = csv.DictWriter(f_hist, fieldnames=headers_hist)
            if not file_exists_hist or os.path.getsize(f_hist.name) == 0:
                writer_hist.writeheader()
            for data_row_hist in historical_data_to_save:
                writer_hist.writerow(data_row_hist)
                save_count += 1
        msg = f"Successfully saved {save_count} breakout records to '{BREAKOUT_HISTORICAL_DB_FILE}' for date {date_str}."
        print(msg)
        return msg

    except pd.errors.EmptyDataError:
        msg = f"Warning: Breakout source file '{BREAKOUT_TICKERS_FILE}' is empty. Nothing saved."
        print(msg)
        return msg
    except KeyError as e_key:
        msg = f"Warning: Missing column in '{BREAKOUT_TICKERS_FILE}': {e_key}. Cannot save historical data."
        print(msg)
        return msg
    except IOError as e_io_hist:
        msg = f"Error writing to historical breakout save file '{BREAKOUT_HISTORICAL_DB_FILE}': {e_io_hist}"
        print(msg)
        return msg
    except Exception as e_save_hist:
        msg = f"Error processing/saving historical breakout data: {e_save_hist}"
        print(msg)
        traceback.print_exc() # traceback needs to be imported
        return msg
    
async def handle_breakout_command(args: List[str], ai_params: Optional[Dict] = None):
    """
    Handles the /breakout command.
    CLI:
        /breakout : Runs breakout analysis.
        /breakout 3725 : Prompts for date to save current breakout data.
    AI:
        Called with ai_params={"date_to_save": "MM/DD/YYYY"} to save breakout data.
        (Note: AI uses 'run_breakout_analysis_singularity' tool directly to run analysis)
    Returns a summary string if called by AI for saving.
    """
    print("\n--- /breakout Command ---")

    if ai_params: # Called by AI - specifically for saving breakout data
        print(f"AI Call to /breakout (save action) with params: {ai_params}")
        date_to_save_ai = ai_params.get("date_to_save")
        if not date_to_save_ai:
            return "Error for AI (/breakout save): 'date_to_save' (MM/DD/YYYY) is required."
        try:
            datetime.strptime(date_to_save_ai, '%m/%d/%Y') # Validate date
            # save_breakout_data_singularity is now async and returns a summary string
            return await save_breakout_data_singularity(date_to_save_ai)
        except ValueError:
            msg = f"AI Error: Invalid date format '{date_to_save_ai}'. Expected MM/DD/YYYY."
            print(msg)
            return msg
        except Exception as e_save_op_ai:
            msg = f"AI Error during breakout data save operation: {e_save_op_ai}"
            print(msg)
            traceback.print_exc()
            return msg

    else: # --- CLI Path ---
        save_code_breakout_cli = args[0] if args else None

        if save_code_breakout_cli == "3725":
            date_to_save_breakout_cli = input(f"CLI: Enter date (MM/DD/YYYY) to save current breakout data under: ")
            try:
                datetime.strptime(date_to_save_breakout_cli, '%m/%d/%Y') # Validate date
                await save_breakout_data_singularity(date_to_save_breakout_cli) # Prints its own success/failure
            except ValueError:
                print("CLI: Invalid date format. Please use MM/DD/YYYY. Save operation cancelled.")
            except Exception as e_save_op_cli:
                print(f"CLI Error occurred during breakout data save operation: {e_save_op_cli}")
                traceback.print_exc()
            return None # CLI save action, no summary string for AI
        else:
            # CLI: Run on-demand breakout analysis
            # run_breakout_analysis_singularity is already async and prints its own results
            await run_breakout_analysis_singularity()
            return None # CLI analysis action, no summary string for AI
        
# --- Market Command Functions ---
def get_sp500_symbols_singularity() -> List[str]:
    """Fetches S&P 500 symbols from Wikipedia for Singularity use."""
    try:
        # Using a known reliable source for S&P 500 list
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # pandas read_html returns a list of DataFrames
        dfs = pd.read_html(sp500_url)
        if not dfs:
            print("Error: Could not parse S&P 500 table from Wikipedia.")
            return []
        # The first table on the page usually contains the S&P 500 components
        sp500_df = dfs[0]
        if 'Symbol' not in sp500_df.columns:
            print("Error: 'Symbol' column not found in S&P 500 data.")
            return []
        
        # Extract symbols, replace '.' with '-' for yfinance compatibility (e.g., BRK.B -> BRK-B)
        symbols = [str(s).replace('.', '-') for s in sp500_df['Symbol'].tolist() if isinstance(s, str)]
        symbols = sorted(list(set(s for s in symbols if s))) # Ensure unique and non-empty
        print(f"Fetched {len(symbols)} S&P 500 symbols.")
        return symbols
    except Exception as e:
        print(f"Error fetching S&P 500 symbols: {e}")
        traceback.print_exc()
        return []

def get_spy_symbols_singularity() -> List[str]:
    """Returns S&P 500 symbols, as SPY tracks the S&P 500."""
    return get_sp500_symbols_singularity()

async def calculate_market_invest_scores_singularity(tickers: List[str], ema_sens: int) -> List[Dict[str, Any]]:
    """
    Calculates EMA Invest scores for a list of tickers with a given sensitivity.
    Returns a list of dictionaries: {'ticker': str, 'live_price': Optional[float], 'score': Optional[float]}.
    """
    result_data_market = []
    total_tickers = len(tickers)
    print(f"\nCalculating Invest scores for {total_tickers} market tickers (Sensitivity: {ema_sens})...")
    
    # Process tickers in chunks to show progress and manage many API calls
    chunk_size = 25 # Process 25 tickers at a time for yfinance calls
    processed_count_market = 0

    for i in range(0, total_tickers, chunk_size):
        chunk = tickers[i:i + chunk_size]
        # Create a list of tasks for the current chunk
        tasks = [calculate_ema_invest(ticker, ema_sens) for ticker in chunk]
        # Run tasks concurrently for the current chunk
        results_chunk = await asyncio.gather(*tasks, return_exceptions=True)
        
        for idx, res_item in enumerate(results_chunk):
            ticker_processed = chunk[idx]
            if isinstance(res_item, Exception):
                # print(f"  Error processing {ticker_processed} for market score: {res_item}")
                result_data_market.append({'ticker': ticker_processed, 'live_price': None, 'score': None, 'error': str(res_item)})
            elif res_item is not None:
                live_price_market, ema_invest_score_market = res_item
                result_data_market.append({
                    'ticker': ticker_processed, 
                    'live_price': live_price_market, 
                    'score': ema_invest_score_market # Store raw score (can be None or float)
                })
            else: # Should not happen if calculate_ema_invest always returns a tuple
                result_data_market.append({'ticker': ticker_processed, 'live_price': None, 'score': None, 'error': 'Unknown error from calculate_ema_invest'})
            
            processed_count_market += 1
            if processed_count_market % 50 == 0 or processed_count_market == total_tickers : # Update every 50 or at the end
                print(f"  ...market scores calculated for {processed_count_market}/{total_tickers} tickers.")

    # Sort by score (descending), handling None scores by placing them at the end (or treating as -infinity)
    result_data_market.sort(key=lambda x: safe_score(x.get('score', -float('inf'))), reverse=True)
    print("Finished calculating all market scores.")
    return result_data_market

async def save_market_data_singularity(sensitivity: int, date_str: str):
    """
    Saves full market data (Ticker, Price, Score) for a given sensitivity and date.
    """
    print(f"\n--- Saving Full Market Data (Sensitivity: {sensitivity}) for Date: {date_str} ---")
    
    spy_symbols_market = get_spy_symbols_singularity()
    if not spy_symbols_market:
        print("Error: Could not retrieve S&P 500 symbols. Cannot save market data.")
        return

    print(f"Calculating scores for {len(spy_symbols_market)} S&P 500 tickers (Sens: {sensitivity})...")
    all_scores_data_market = await calculate_market_invest_scores_singularity(spy_symbols_market, sensitivity)

    if not all_scores_data_market:
        print(f"Error: No valid market data calculated for Sensitivity {sensitivity}. Nothing saved.")
        return

    data_to_save_market = []
    for item_market in all_scores_data_market:
        # Only save if score is not None (i.e., calculation was successful)
        if item_market.get('score') is not None:
            data_to_save_market.append({
                'DATE': date_str,
                'TICKER': item_market.get('ticker', 'ERR'),
                'PRICE': f"{safe_score(item_market.get('live_price')):.2f}" if item_market.get('live_price') is not None else "N/A",
                'SCORE': f"{safe_score(item_market.get('score')):.2f}" # Save raw score, formatted
            })
    
    if not data_to_save_market:
        print(f"No tickers with valid scores found for Sensitivity {sensitivity}. Nothing saved.")
        return

    save_filename_market = f"{MARKET_FULL_SENS_DATA_FILE_PREFIX}{sensitivity}_data.csv"
    file_exists_market = os.path.isfile(save_filename_market)
    headers_market_save = ['DATE', 'TICKER', 'PRICE', 'SCORE']
    
    try:
        with open(save_filename_market, 'a', newline='', encoding='utf-8') as f_market:
            writer_market = csv.DictWriter(f_market, fieldnames=headers_market_save)
            if not file_exists_market or os.path.getsize(f_market.name) == 0:
                writer_market.writeheader()
            writer_market.writerows(data_to_save_market) # Use writerows for list of dicts
        print(f"Successfully saved {len(data_to_save_market)} records to '{save_filename_market}'.")
    except IOError as e_io_market:
        print(f"Error writing market data to '{save_filename_market}': {e_io_market}")
    except Exception as e_save_mkt:
        print(f"Unexpected error saving market data: {e_save_mkt}")
        traceback.print_exc()


async def handle_market_command(args: List[str]):
    """
    Handles the /market command for both CLI and AI interaction.
    - CLI display: /market (prompts for sensitivity)
    - CLI save: /market 3725 (prompts for sensitivity and date)
    - AI display: Expects args=["display", <sensitivity_str_convertible_to_int_1_2_or_3>]
    - AI save: Expects args=["save", <sensitivity_str_convertible_to_int_1_2_or_3>, <date_str_MM/DD/YYYY>]
    Returns a summary string if called by AI, otherwise None or prints directly.
    """
    print("\n--- /market Command ---")
    
    action = None
    sensitivity_for_action = None
    date_for_save = None
    called_by_ai = False # Flag to indicate if AI initiated this call with proper args

    # Determine action based on args, prioritizing AI-specific arg structure
    if len(args) > 0:
        ai_action_candidate = args[0].lower()
        if ai_action_candidate == "display" and len(args) >= 2:
            try:
                # Robust conversion: float first, then int to handle "2.0" or 2.0
                sensitivity_val = int(float(args[1]))
                if sensitivity_val not in [1, 2, 3]:
                    msg = f"AI Error: Sensitivity '{args[1]}' resolved to '{sensitivity_val}', which is out of range [1,2,3]."
                    print(msg)
                    return msg # Return error for AI
                sensitivity_for_action = sensitivity_val
                action = "display"
                called_by_ai = True
            except ValueError:
                msg = f"AI Error: Sensitivity '{args[1]}' provided by AI is not a valid number for display."
                print(msg)
                return msg # Return error for AI
        
        elif ai_action_candidate == "save" and len(args) >= 3:
            try:
                sensitivity_val = int(float(args[1])) # Robust conversion
                if sensitivity_val not in [1, 2, 3]:
                    msg = f"AI Error: Sensitivity '{args[1]}' resolved to '{sensitivity_val}' for save, which is out of range [1,2,3]."
                    print(msg)
                    return msg
                
                date_str_input = args[2]
                datetime.strptime(date_str_input, '%m/%d/%Y') # Validate date format

                sensitivity_for_action = sensitivity_val
                date_for_save = date_str_input
                action = "save"
                called_by_ai = True
            except ValueError:
                msg = f"AI Error: Sensitivity '{args[1]}' or date '{args[2]}' format invalid for save. Sens: 1,2,3. Date: MM/DD/YYYY."
                print(msg)
                return msg
        
        elif args[0] == "3725": # CLI save trigger
            action = "save_interactive"
        # If no AI specific args matched, and not "3725", it's interactive display or invalid CLI
            
    if not action: # If no action determined yet, it's CLI interactive display
        action = "display_interactive"

    # --- Perform Action ---
    summary_for_ai = ""

    if action == "save" or action == "save_interactive":
        try:
            if action == "save_interactive": # Prompt for CLI
                sens_to_save_str = input("Enter Market Sensitivity (1, 2, or 3) to save: ")
                sensitivity_for_action = int(sens_to_save_str) # CLI input expected to be clean int
                if sensitivity_for_action not in [1, 2, 3]:
                    print("Invalid sensitivity. Must be 1, 2, or 3.")
                    return None 
                
                date_str_input_cli = input(f"Enter date (MM/DD/YYYY) to save full market data for Sensitivity {sensitivity_for_action}: ")
                datetime.strptime(date_str_input_cli, '%m/%d/%Y') 
                date_for_save = date_str_input_cli
            
            # save_market_data_singularity needs to be defined and async
            await save_market_data_singularity(sensitivity_for_action, date_for_save)
            summary_for_ai = f"Market data for sensitivity {sensitivity_for_action} successfully saved for date {date_for_save}."
            if not called_by_ai: print(summary_for_ai)
            return summary_for_ai

        except ValueError as ve: # Handles int conversion or date format errors for CLI
            msg = f"Invalid input: {ve}. Please use numbers for sensitivity and MM/DD/YYYY for date."
            if not called_by_ai: print(msg)
            return msg if called_by_ai else None # Return error to AI if it caused this
        except Exception as e_save_market_op:
            msg = f"An error occurred during market data save operation: {e_save_market_op}"
            if not called_by_ai: print(msg)
            traceback.print_exc()
            return msg if called_by_ai else None

    elif action == "display" or action == "display_interactive":
        try:
            if action == "display_interactive": # Prompt for CLI
                sens_to_display_str = input("Enter Market Sensitivity (1, 2, or 3) to display: ")
                sensitivity_for_action = int(sens_to_display_str) # CLI input expected to be clean int
                if sensitivity_for_action not in [1, 2, 3]:
                    print("Invalid sensitivity. Must be 1, 2, or 3.")
                    return None

            # get_spy_symbols_singularity and calculate_market_invest_scores_singularity need to be defined
            spy_symbols_market_disp = get_spy_symbols_singularity()
            if not spy_symbols_market_disp:
                msg = "Error: Could not retrieve S&P 500 symbols for display."
                if not called_by_ai: print(msg)
                return msg if called_by_ai else None

            print(f"\nCalculating market scores for S&P 500 (Sensitivity: {sensitivity_for_action}). This may take some time...")
            all_scores_data_market_disp = await calculate_market_invest_scores_singularity(spy_symbols_market_disp, sensitivity_for_action)

            if not all_scores_data_market_disp:
                msg = "Error calculating market scores or no data returned."
                if not called_by_ai: print(msg)
                return msg if called_by_ai else None
            
            valid_scores_market = [item for item in all_scores_data_market_disp if item.get('score') is not None]
            if not valid_scores_market:
                msg = "No valid scores could be calculated for display."
                if not called_by_ai: print(msg)
                return msg if called_by_ai else None

            top_10_scores = valid_scores_market[:10]
            bottom_10_scores_raw = valid_scores_market[-10:]
            bottom_10_scores = sorted(bottom_10_scores_raw, key=lambda x: safe_score(x.get('score', float('inf')))) # safe_score needs to be defined
            spy_score_data = next((item for item in all_scores_data_market_disp if item['ticker'] == 'SPY'), None)

            def format_market_row_for_display(item_dict_mkt: Dict[str, Any]) -> List[str]:
                ticker_m = item_dict_mkt.get('ticker', 'ERR')
                price_val_m = item_dict_mkt.get('live_price')
                score_val_m = item_dict_mkt.get('score')
                price_m_f = f"${safe_score(price_val_m):.2f}" if price_val_m is not None else "N/A"
                score_m_f = f"{safe_score(score_val_m):.2f}%" if score_val_m is not None else "N/A"
                return [ticker_m, price_m_f, score_m_f]

            print(f"\n**Top 10 S&P 500 Stocks (Sensitivity: {sensitivity_for_action})**")
            if top_10_scores: print(tabulate([format_market_row_for_display(r) for r in top_10_scores], headers=["Ticker", "Price", "Score"], tablefmt="pretty")) # tabulate needs to be imported
            else: print("No top scores data available.")
            print(f"\n**Bottom 10 S&P 500 Stocks (Sensitivity: {sensitivity_for_action})**")
            if bottom_10_scores: print(tabulate([format_market_row_for_display(r) for r in bottom_10_scores], headers=["Ticker", "Price", "Score"], tablefmt="pretty"))
            else: print("No bottom scores data available.")
            print(f"\n**SPY Score (Sensitivity: {sensitivity_for_action})**")
            if spy_score_data and spy_score_data.get('score') is not None: print(tabulate([format_market_row_for_display(spy_score_data)], headers=["Ticker", "Price", "Score"], tablefmt="pretty"))
            else: print("SPY score could not be calculated or is unavailable.")
            
            print("\n--- Market Display Complete ---")

            summary_lines = [f"Market analysis scores for S&P 500 (Sensitivity: {sensitivity_for_action}) displayed."]
            if top_10_scores: summary_lines.append("Top tickers include: " + ", ".join([f"{r['ticker']} ({safe_score(r.get('score')):.1f}%)" for r in top_10_scores[:3]]))
            if spy_score_data and spy_score_data.get('score') is not None: summary_lines.append(f"SPY score: {safe_score(spy_score_data.get('score')):.2f}%.")
            summary_for_ai = " ".join(summary_lines)
            return summary_for_ai

        except ValueError as ve: # Handles int conversion for CLI sensitivity
            msg = f"Invalid input for sensitivity: {ve}. Must be a number (1, 2, or 3)."
            if not called_by_ai: print(msg)
            return msg if called_by_ai else None
        except Exception as e_disp_market:
            msg = f"An error occurred during market display: {e_disp_market}"
            if not called_by_ai: print(msg)
            traceback.print_exc()
            return msg if called_by_ai else None
    else:
        # This case should ideally not be hit if logic for action is correct
        msg = "Market command action not recognized or insufficient arguments."
        if not called_by_ai: display_commands() 
        return msg # Return error for AI

def get_yf_data_singularity(tickers: List[str], period: str = "10y", interval: str = "1d") -> pd.DataFrame:
    """ 
    Downloads historical closing price data for multiple tickers using yfinance.
    Optimized to build DataFrame from a list of Series to avoid fragmentation.
    """
    if not tickers:
        return pd.DataFrame()
    
    tickers_list = list(set(tickers)) # Ensure unique tickers
    
    try:
        # yfinance download can be tricky with multiple tickers and column naming.
        # Using group_by='ticker' can help, but sometimes it still returns a flat structure for few tickers.
        data = yf.download(tickers_list, period=period, interval=interval, progress=False, auto_adjust=False, group_by='ticker', timeout=20)
        
        if data.empty:
            # print(f"    get_yf_data_singularity: yfinance.download returned EMPTY DataFrame for tickers: {tickers_list}")
            return pd.DataFrame()

        all_series = [] # List to hold individual ticker Series

        if isinstance(data.columns, pd.MultiIndex):
            # If MultiIndex, columns are typically ('Price Metric', 'Ticker') e.g., ('Close', 'AAPL')
            # or sometimes ('Ticker', 'Price Metric') e.g., ('AAPL', 'Close')
            for ticker_name in tickers_list:
                close_series = None
                # Try common MultiIndex structures
                if (ticker_name, 'Close') in data.columns:
                    close_series = data[(ticker_name, 'Close')]
                elif ('Close', ticker_name) in data.columns:
                    close_series = data[('Close', ticker_name)]
                
                if close_series is not None and not close_series.empty and not close_series.isnull().all():
                    series_numeric = pd.to_numeric(close_series, errors='coerce')
                    series_numeric.name = ticker_name # Name the Series correctly
                    if not series_numeric.isnull().all(): # Check again after numeric conversion
                        all_series.append(series_numeric)
        elif len(tickers_list) == 1 and 'Close' in data.columns : # Single ticker, simple DataFrame
            ticker_name = tickers_list[0]
            if not data['Close'].isnull().all():
                series_numeric = pd.to_numeric(data['Close'], errors='coerce')
                series_numeric.name = ticker_name
                if not series_numeric.isnull().all():
                    all_series.append(series_numeric)
        elif not data.empty: # Fallback for non-MultiIndex with multiple tickers (less common with group_by)
            # Try to get 'Close' data for each ticker if columns are just ticker symbols
            for ticker_name in tickers_list:
                if ticker_name in data.columns: # Assuming column is directly named after ticker and contains Close prices
                    close_series = data[ticker_name]
                    if not close_series.empty and not close_series.isnull().all():
                        series_numeric = pd.to_numeric(close_series, errors='coerce')
                        series_numeric.name = ticker_name
                        if not series_numeric.isnull().all():
                            all_series.append(series_numeric)
        
        if not all_series:
            # print(f"    get_yf_data_singularity: No valid data series collected for tickers: {tickers_list}")
            return pd.DataFrame()

        # Concatenate all collected Series at once
        df_out = pd.concat(all_series, axis=1)

        if df_out.empty:
            return pd.DataFrame()

        df_out.index = pd.to_datetime(df_out.index)
        # Columns should be named correctly from Series names
        
        df_out = df_out.dropna(axis=0, how='all') 
        df_out = df_out.dropna(axis=1, how='all') 

        if df_out.empty:
            return pd.DataFrame()
        
        return df_out

    except Exception as e:
        # print(f"    Error in get_yf_data_singularity for {tickers_list}: {type(e).__name__} - {e}")
        if "Failed to get ticker" in str(e) or "No data found" in str(e) or "DNSError" in str(e):
            pass # Common yfinance errors, can be less verbose if many occur
        else:
            traceback.print_exc() # Print traceback for unexpected errors
        return pd.DataFrame()

def screen_stocks_singularity() -> List[str]:
    """ Screens for stocks using TradingView. Singularity version."""
    print("    Starting Step: Stock Screening (TradingView)...")
    try:
        query = Query().select(
            'name', 
            'market_cap_basic',
            'average_volume_90d_calc',
        ).where(
            Column('market_cap_basic') >= 50_000_000_000, 
            Column('average_volume_90d_calc') >= 1_000_000
        ).limit(500) 

        print("      Executing screener query...")
        # Note: query.get_scanner_data() is a blocking call.
        # If this script were fully async, it should be run in a thread.
        # For a Singularity script that awaits other async yfinance calls, this sync call is acceptable here.
        scanner_results = query.get_scanner_data(timeout=60) 
        print("      Screener query finished.")

        if scanner_results and isinstance(scanner_results, tuple) and len(scanner_results) > 0 and isinstance(scanner_results[1], pd.DataFrame):
            df = scanner_results[1]
            if not df.empty and 'name' in df.columns:
                tickers = [str(t).split(':')[-1].replace('.', '-') for t in df['name'].tolist() if pd.notna(t)]
                cleaned_tickers = sorted(list(set(tickers)))
                print(f"    Screening complete. Found {len(cleaned_tickers)} potential tickers.")
                return cleaned_tickers
            else:
                print("    Warning: 'name' column not found or empty in screening results DataFrame.")
                return []
        else:
            print("    Warning: Stock screener returned no data or unexpected format.")
            return []
    except Exception as e: 
        print(f"    Error during stock screening: {type(e).__name__} - {e}")
        if "Max retries exceeded" in str(e) or "ConnectTimeoutError" in str(e) or "Failed to resolve" in str(e):
            print("    This is likely a network issue or TradingView service problem. Check your internet connection.")
        else:
            traceback.print_exc() 
        return []
    
def calculate_metrics_singularity(tickers_list: List[str], spy_data_10y: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """ Calculates 1y Beta, 1y Correlation, and 1y Avg Leverages relative to SPY. Singularity version."""
    print(f"    Starting Step: Calculating Metrics (Beta/Corr/Leverage) for {len(tickers_list)} tickers...")
    metrics = {}

    if spy_data_10y.empty or 'SPY' not in spy_data_10y.columns:
         print("    Error: Valid 10y SPY historical data ('SPY' column) is required for metrics calculation by calculate_metrics_singularity.")
         return {} # Return empty if SPY data is insufficient
    try:
        spy_data_10y.index = pd.to_datetime(spy_data_10y.index)
        spy_daily_returns_full = spy_data_10y['SPY'].pct_change().dropna() # Use .dropna() here
        if spy_daily_returns_full.empty:
             print("    Error: SPY daily returns are all NaN after calculation in calculate_metrics_singularity.")
             return {}
    except Exception as e:
         print(f"    Error preparing SPY daily returns for metrics in calculate_metrics_singularity: {e}")
         return {}

    # print("      Calculating historical SPY Invest Score (Daily, 10y)...") # Reduced verbosity
    spy_invest_scores_hist = None
    try:
        spy_close_series = pd.to_numeric(spy_data_10y['SPY'], errors='coerce').dropna()
        if len(spy_close_series) >= 55: # Min period for EMA 55
            spy_ema_8 = spy_close_series.ewm(span=8, adjust=False).mean()
            spy_ema_55 = spy_close_series.ewm(span=55, adjust=False).mean()
            spy_invest_score_series = pd.Series(np.nan, index=spy_close_series.index)
            valid_indices = spy_ema_55.index[(spy_ema_55.notna()) & (spy_ema_55 != 0)]
            if not valid_indices.empty:
                 ema_enter = (spy_ema_8.loc[valid_indices] - spy_ema_55.loc[valid_indices]) / spy_ema_55.loc[valid_indices]
                 spy_invest_score_series.loc[valid_indices] = ((ema_enter * 4) + 0.5) * 100
            spy_invest_scores_hist = spy_invest_score_series.dropna()
            # if spy_invest_scores_hist.empty: print("      Warning: Historical SPY Invest Score is empty after calculation.")
        # else: print(f"      Warning: Insufficient SPY data ({len(spy_close_series)} pts) for historical Invest Score EMA.")
    except Exception as e:
        print(f"      Error calculating historical SPY Invest Score: {e}")

    # print("      Fetching 10y history for tickers to calculate metrics...") # Reduced verbosity
    tickers_to_fetch_metrics = list(set(tickers_list + ['SPY'])) # Ensure SPY is included
    all_tickers_data_metrics = get_yf_data_singularity(tickers_to_fetch_metrics, period="10y", interval="1d")

    if all_tickers_data_metrics.empty:
        print("    Error: Failed to fetch historical data for the main list of tickers in calculate_metrics_singularity.")
        return {}

    daily_returns_all = all_tickers_data_metrics.pct_change().iloc[1:].dropna(how='all')
    if 'SPY' not in daily_returns_all.columns: # Check again after fetching all tickers
        print("      Error: SPY column missing in combined daily returns for metrics (after fetching all).")
        return {}
    
    processed_count_metrics = 0
    successful_count_metrics = 0
    for ticker_m in tickers_list:
        processed_count_metrics += 1
        if processed_count_metrics % 20 == 0 and len(tickers_list) > 20 : # Avoid printing for very small lists
            print(f"        Metrics calculation progress: {processed_count_metrics}/{len(tickers_list)} (Successful: {successful_count_metrics})")

        if ticker_m == 'SPY' or ticker_m not in daily_returns_all.columns:
            continue # Skip SPY itself or if its data is missing from the combined fetch

        ticker_returns_m = daily_returns_all[ticker_m]
        spy_aligned_returns_m = daily_returns_all['SPY'] # SPY returns from the combined fetch
        
        combined_df_m = pd.concat([ticker_returns_m, spy_aligned_returns_m], axis=1, keys=[ticker_m, 'SPY']).dropna()
        
        if spy_invest_scores_hist is not None and not spy_invest_scores_hist.empty:
            combined_df_m = combined_df_m.join(spy_invest_scores_hist.rename('SPY_Score'), how='inner')
        else:
            combined_df_m['SPY_Score'] = np.nan 

        if len(combined_df_m) < 252: # Need at least ~1 year of aligned data
            continue

        data_1y_m = combined_df_m.tail(252)
        ticker_returns_1y_m = data_1y_m[ticker_m]
        spy_returns_1y_m = data_1y_m['SPY']
        spy_scores_1y_m = data_1y_m['SPY_Score']

        beta_1y, correlation_1y = np.nan, np.nan
        avg_leverage_uptrend_1y, avg_leverage_downtrend_1y, avg_leverage_general_1y = np.nan, np.nan, np.nan

        try: 
            if ticker_returns_1y_m.nunique() > 1 and spy_returns_1y_m.nunique() > 1:
                spy_variance_1y = np.var(spy_returns_1y_m)
                if not pd.isna(spy_variance_1y) and spy_variance_1y > 1e-12: # Avoid division by zero
                    covariance_matrix_1y = np.cov(ticker_returns_1y_m, spy_returns_1y_m)
                    if covariance_matrix_1y.shape == (2,2): beta_1y = covariance_matrix_1y[0,1] / spy_variance_1y
                
                correlation_matrix_1y = np.corrcoef(ticker_returns_1y_m, spy_returns_1y_m)
                if correlation_matrix_1y.shape == (2,2):
                    correlation_1y = correlation_matrix_1y[0,1]
                    if pd.isna(correlation_1y): correlation_1y = 0.0 
            else: correlation_1y = 0.0 
        except Exception: pass

        try: 
            with np.errstate(divide='ignore', invalid='ignore'):
                leverage_raw_1y = ticker_returns_1y_m / spy_returns_1y_m
            leverage_raw_1y.replace([np.inf, -np.inf], np.nan, inplace=True)
            if leverage_raw_1y.notna().any(): avg_leverage_general_1y = np.nanmean(leverage_raw_1y)
            
            if spy_scores_1y_m.notna().any(): # Only if historical SPY scores are available
                uptrend_mask = (spy_scores_1y_m > 50) & leverage_raw_1y.notna()
                if uptrend_mask.any(): avg_leverage_uptrend_1y = np.nanmean(leverage_raw_1y[uptrend_mask])
                
                downtrend_mask = (spy_scores_1y_m < 50) & leverage_raw_1y.notna()
                if downtrend_mask.any(): avg_leverage_downtrend_1y = np.nanmean(leverage_raw_1y[downtrend_mask])
        except Exception: pass
        
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
            
    print(f"    Finished Step: Calculating Metrics. Successful calculations: {successful_count_metrics}/{len(tickers_list)}")
    return metrics

def save_initial_metrics_singularity(metrics: Dict[str, Dict[str, float]], tickers_processed: List[str]):
    """ Saves calculated initial metrics to CULTIVATE_INITIAL_METRICS_FILE. Singularity version."""
    print("    Starting Step: Saving Initial Metrics...")
    if not metrics:
        print("      Skipping initial metrics save as metrics calculation failed or yielded no results.")
        return

    initial_metrics_list = []
    for ticker_s in tickers_processed:
        if ticker_s in metrics:
             metric_data_s = metrics[ticker_s]
             initial_metrics_list.append({
                 'Ticker': ticker_s,
                 'Beta (1y)': metric_data_s.get('beta_1y'),
                 'Correlation (1y)': metric_data_s.get('correlation_1y'),
                 'Avg Leverage (Uptrend >50)': metric_data_s.get('avg_leverage_uptrend_1y'),
                 'Avg Leverage (Downtrend <50)': metric_data_s.get('avg_leverage_downtrend_1y'),
                 'Avg Leverage (General)': metric_data_s.get('avg_leverage_general_1y')
             })

    if initial_metrics_list:
        try:
            df_initial_metrics = pd.DataFrame(initial_metrics_list)
            cols_ordered = ['Ticker', 'Beta (1y)', 'Correlation (1y)', 'Avg Leverage (Uptrend >50)', 'Avg Leverage (Downtrend <50)', 'Avg Leverage (General)']
            df_initial_metrics = df_initial_metrics.reindex(columns=cols_ordered)
            num_cols = df_initial_metrics.select_dtypes(include=np.number).columns.tolist()
            for col in num_cols:
                 df_initial_metrics[col] = df_initial_metrics[col].apply(lambda x: round(safe_score(x), 4) if pd.notna(x) else np.nan)
            df_initial_metrics.to_csv(CULTIVATE_INITIAL_METRICS_FILE, index=False, float_format='%.4f')
            print(f"      Successfully saved initial metrics for {len(df_initial_metrics)} tickers to {CULTIVATE_INITIAL_METRICS_FILE}")
        except Exception as e:
            print(f"      Error saving initial metrics CSV ({CULTIVATE_INITIAL_METRICS_FILE}): {e}")

def calculate_cultivate_formulas_singularity(allocation_score: float) -> Optional[Dict[str, Any]]:
    """ Calculates Lambda, Omega, Alpha, Beta_alloc, Mu, Rho, Omega_target, Delta, Eta, Kappa. Singularity version."""
    print("    Starting Step: Calculating Cultivate Formula Variables...")
    sigma = safe_score(allocation_score)
    sigma_safe = max(0.0001, min(99.9999, sigma))
    sigma_ratio_term = sigma_safe / (100.0 - sigma_safe) if (100.0 - sigma_safe) > 1e-9 else np.inf

    results = {}
    try:
        log_term_O = np.log(7.0/6.0) / 50.0; exp_term_O = np.exp(-log_term_O * sigma_safe)
        inner_O = (49.0/60.0 * sigma_safe * exp_term_O + 40.0)
        results['omega'] = max(0.0, min(100.0, 100.0 - inner_O))
        results['lambda'] = max(0.0, min(100.0, 100.0 - results['omega']))
        
        results['lambda_hedge'] = max(0.0, min(100.0, 100 - ((1/1000) * sigma_safe**2 + (7/20) * sigma_safe + 40)))
        
        temp_alpha_factor = (1/1000) * sigma_safe**2 + (7/20) * sigma_safe + 40
        results['alpha'] = max(0.0, min(results['lambda'], temp_alpha_factor * results['lambda'] / 100.0))
        results['beta_alloc'] = max(0.0, results['lambda'] - results['alpha'])

        if np.isinf(sigma_ratio_term) or sigma_ratio_term < 1e-9:
            exp_term_mu, exp_term_rho, exp_term_omega_t, exp_term_delta = (0.0,0.0,0.0,0.0) if sigma_safe > 99.999 else (1.0,1.0,1.0,1.0)
        else:
            exp_term_mu = np.exp(-np.log(11.0/4.0) * sigma_ratio_term)
            exp_term_rho = np.exp(-np.log(4.0) * sigma_ratio_term)
            exp_term_omega_t = np.exp(-np.log(7.0/3.0) * sigma_ratio_term)
            exp_term_delta = np.exp(-np.log(11.0/8.0) * sigma_ratio_term)

        mu_val = -1/4 + (11/4) * (1 - exp_term_mu)
        results['mu_center'] = mu_val; results['mu_range'] = (mu_val - 2/3, mu_val + 2/3)
        rho_val = 3/4 - exp_term_rho
        results['rho_center'] = rho_val; results['rho_range'] = (rho_val - 1/8, rho_val + 1/8)
        omega_target_val = -1/2 + (7/2) * (1 - exp_term_omega_t)
        results['omega_target_center'] = omega_target_val; results['omega_target_range'] = (omega_target_val - 1/2, omega_target_val + 1/2)
        results['delta'] = max(0.25, min(5.0, 1/4 + (11/4) * (1 - exp_term_delta)))
        
        results['eta'] = max(0.0, min(100.0, -sigma_safe**2 / 500.0 - 3.0*sigma_safe / 10.0 + 60.0))
        results['kappa'] = max(0.0, min(100.0, 100.0 - results['eta']))
        
        print("      Cultivate formula variable calculations complete.")
        return results
    except Exception as e:
        print(f"      Error calculating cultivate formulas: {e}"); traceback.print_exc(); return None

async def select_tickers_singularity(tickers_to_filter: List[str], metrics: Dict, invest_scores_all: Dict, 
                       formula_results: Dict, portfolio_value: float) -> tuple[List[str], Optional[str], Dict]: # Made async
    """ Selects final tickers based on metrics, scores, and formulas. Singularity version. Now async."""
    print("    Starting Step: Selecting Final Tickers (Using Beta/Corr/Leverage, Score > 0)...")
    mu_range = formula_results.get('mu_range', (-np.inf, np.inf))
    rho_range = formula_results.get('rho_range', (-np.inf, np.inf))
    omega_target_range = formula_results.get('omega_target_range', (-np.inf, np.inf))
    epsilon = safe_score(portfolio_value)

    if epsilon <= 0: return [], "Error: Invalid portfolio value for ticker selection", invest_scores_all

    num_tickers_sigma_target = max(0, max(1, math.ceil(0.3 * math.sqrt(epsilon))) - len(HEDGING_TICKERS))
    print(f"      Target number of common stock tickers (Sigma_count): {num_tickers_sigma_target}")

    # Call calculate_ema_invest with await since select_tickers_singularity is now async
    _, spy_invest_latest = await calculate_ema_invest('SPY', 2) 
    leverage_key = 'avg_leverage_general_1y' # Default
    if spy_invest_latest is not None:
        if safe_score(spy_invest_latest) >= 60: leverage_key = 'avg_leverage_uptrend_1y'
        elif safe_score(spy_invest_latest) <= 40: leverage_key = 'avg_leverage_downtrend_1y'
    # print(f"        Using '{leverage_key}' for filtering based on SPY Invest Score: {safe_score(spy_invest_latest):.2f}%")

    T1, T_temp = [], []
    tickers_available_for_filtering = [t for t in tickers_to_filter if t in metrics and t in invest_scores_all]

    for ticker_sel in tickers_available_for_filtering:
        metric_sel = metrics.get(ticker_sel, {})
        score_info_sel = invest_scores_all.get(ticker_sel, {})
        beta_s = safe_score(metric_sel.get('beta_1y')); corr_s = safe_score(metric_sel.get('correlation_1y'))
        leverage_s = safe_score(metric_sel.get(leverage_key)); score_s = safe_score(score_info_sel.get('score'))

        if score_s <= 0 or pd.isna(beta_s) or pd.isna(corr_s) or pd.isna(leverage_s): continue

        in_mu = mu_range[0] <= beta_s <= mu_range[1]
        in_rho = rho_range[0] <= corr_s <= rho_range[1]
        in_omega_target = omega_target_range[0] <= leverage_s <= omega_target_range[1]

        if in_mu and in_rho and in_omega_target:
            T1.append({'ticker': ticker_sel, 'score': score_s})
        else:
            mu_center = safe_score(formula_results.get('mu_center', 0)); mu_width_half = safe_score((mu_range[1] - mu_range[0]) / 2)
            rho_center = safe_score(formula_results.get('rho_center', 0)); rho_width_half = safe_score((rho_range[1] - rho_range[0]) / 2)
            omega_t_center = safe_score(formula_results.get('omega_target_center', 0)); omega_t_width_half = safe_score((omega_target_range[1] - omega_target_range[0]) / 2)
            
            if (mu_center - mu_width_half * 1.5 <= beta_s <= mu_center + mu_width_half * 1.5 and
                rho_center - rho_width_half * 1.5 <= corr_s <= rho_center + rho_width_half * 1.5 and
                omega_t_center - omega_t_width_half * 1.5 <= leverage_s <= omega_t_center + omega_t_width_half * 1.5):
                T_temp.append({'ticker': ticker_sel, 'score': score_s})
    
    T1.sort(key=lambda x: x['score'], reverse=True); T_temp.sort(key=lambda x: x['score'], reverse=True)
    T1_tickers_set = {item['ticker'] for item in T1}
    T_minus_1 = [item for item in T_temp if item['ticker'] not in T1_tickers_set]
    
    try:
        pd.DataFrame(T1 if T1 else []).to_csv(CULTIVATE_T1_FILE, index=False)
        pd.DataFrame(T_minus_1 if T_minus_1 else []).to_csv(CULTIVATE_T_MINUS_1_FILE, index=False)
    except Exception as e_csv: print(f"      Error saving T1/T_minus_1 CSVs: {e_csv}")

    Tf_list_final_sel = T1[:num_tickers_sigma_target]
    remaining_needed = num_tickers_sigma_target - len(Tf_list_final_sel)
    if remaining_needed > 0 and T_minus_1: Tf_list_final_sel.extend(T_minus_1[:remaining_needed])
    
    warning_msg_sel = None
    if not Tf_list_final_sel: warning_msg_sel = "Warning: No tickers selected for Common Stock portfolio."
    elif len(Tf_list_final_sel) < num_tickers_sigma_target:
        warning_msg_sel = f"Warning: Target common stock tickers ({num_tickers_sigma_target}) not reached. Selected {len(Tf_list_final_sel)}."
    
    Tf_tickers_only_sel = [item['ticker'] for item in Tf_list_final_sel]
    try:
        pd.DataFrame(Tf_list_final_sel if Tf_list_final_sel else []).to_csv(CULTIVATE_TF_FINAL_FILE, index=False)
        print(f"        Selected {len(Tf_tickers_only_sel)} final Common Stock tickers (Tf). Saved to {CULTIVATE_TF_FINAL_FILE}")
    except Exception as e_csv_tf: print(f"      Error saving Tf CSV: {e_csv_tf}")
        
    print("    Finished Step: Selecting Final Tickers.")
    return Tf_tickers_only_sel, warning_msg_sel, invest_scores_all

def build_and_process_portfolios_singularity(common_stock_tickers: List[str], amplification: float, 
                                     total_portfolio_value: float, cash_allocation_omega: float, 
                                     frac_shares: bool, invest_scores_all: Dict, 
                                     eta: float, kappa: float, lambda_hedge: float) -> tuple:
    """ Builds and processes Cultivate portfolios. Singularity version. """
    print("    Starting Step: Building & Processing Final Cultivate Portfolios...")
    epsilon = safe_score(total_portfolio_value)
    cash_alloc_omega_pct = safe_score(cash_allocation_omega)
    amplification_val = safe_score(amplification)
    eta_pct = safe_score(eta); kappa_pct = safe_score(kappa); lambda_hedge_pct = safe_score(lambda_hedge)

    initial_omega_cash_val = epsilon * (cash_alloc_omega_pct / 100.0)
    value_for_stocks_hedges = max(0.0, epsilon - initial_omega_cash_val)
    value_for_overall_hedging = value_for_stocks_hedges * (lambda_hedge_pct / 100.0)
    value_for_common_stock = max(0.0, value_for_stocks_hedges - value_for_overall_hedging)
    value_for_market_hedging = value_for_overall_hedging * (kappa_pct / 100.0)
    value_for_resource_hedging = value_for_overall_hedging * (eta_pct / 100.0)

    all_ticker_data_build = {}
    for ticker_build in list(set(common_stock_tickers + HEDGING_TICKERS)):
        if ticker_build in invest_scores_all and 'live_price' in invest_scores_all[ticker_build] and 'score' in invest_scores_all[ticker_build]:
            live_price_build = invest_scores_all[ticker_build]['live_price']
            invest_score_build = invest_scores_all[ticker_build]['score']
            if safe_score(live_price_build) > 0 and invest_score_build is not None and not pd.isna(invest_score_build):
                all_ticker_data_build[ticker_build] = {'live_price': safe_score(live_price_build), 'raw_invest_score': safe_score(invest_score_build)}
            else: all_ticker_data_build[ticker_build] = {'live_price': 0.0, 'raw_invest_score': -float('inf'), 'error': 'Invalid data'}
        else: all_ticker_data_build[ticker_build] = {'live_price': 0.0, 'raw_invest_score': -float('inf'), 'error': 'Data missing'}
    
    combined_portfolio_list_build = []
    # Process Common Stock
    temp_common_list_b, common_total_amp_score_b = [], 0.0
    for ticker_c in [t for t in common_stock_tickers if t in all_ticker_data_build and 'error' not in all_ticker_data_build[t]]:
        data_c = all_ticker_data_build[ticker_c]
        amp_score_c = max(0.0, safe_score((data_c['raw_invest_score'] * amplification_val) - (amplification_val - 1.0) * 50.0))
        temp_common_list_b.append({'ticker': ticker_c, **data_c, 'amplified_score': amp_score_c, 'portfolio': 'Common Stock'})
        common_total_amp_score_b += amp_score_c
    for entry_c in temp_common_list_b:
        sub_alloc_c = (entry_c['amplified_score'] / common_total_amp_score_b) * 100.0 if common_total_amp_score_b > 1e-9 else 0.0
        entry_c['combined_percent_allocation'] = safe_score(sub_alloc_c * (100.0 - lambda_hedge_pct) / 100.0) # Relative to value_for_stocks_hedges
        combined_portfolio_list_build.append(entry_c)

    # Process Market Hedging
    temp_market_h_list, market_h_total_amp_score = [], 0.0
    for ticker_mh in [t for t in MARKET_HEDGING_TICKERS if t in all_ticker_data_build and 'error' not in all_ticker_data_build[t]]:
        data_mh = all_ticker_data_build[ticker_mh]
        amp_score_mh = max(0.0, safe_score((data_mh['raw_invest_score'] * amplification_val) - (amplification_val - 1.0) * 50.0))
        temp_market_h_list.append({'ticker': ticker_mh, **data_mh, 'amplified_score': amp_score_mh, 'portfolio': 'Market Hedging'})
        market_h_total_amp_score += amp_score_mh
    for entry_mh in temp_market_h_list:
        sub_alloc_mh = (entry_mh['amplified_score'] / market_h_total_amp_score) * 100.0 if market_h_total_amp_score > 1e-9 else 0.0
        entry_mh['combined_percent_allocation'] = safe_score(sub_alloc_mh * kappa_pct / 100.0 * lambda_hedge_pct / 100.0)
        combined_portfolio_list_build.append(entry_mh)

    # Process Resource Hedging
    temp_resource_h_list, resource_h_total_amp_score = [], 0.0
    for ticker_rh in [t for t in RESOURCE_HEDGING_TICKERS if t in all_ticker_data_build and 'error' not in all_ticker_data_build[t]]:
        data_rh = all_ticker_data_build[ticker_rh]
        amp_score_rh = max(0.0, safe_score((data_rh['raw_invest_score'] * amplification_val) - (amplification_val - 1.0) * 50.0))
        temp_resource_h_list.append({'ticker': ticker_rh, **data_rh, 'amplified_score': amp_score_rh, 'portfolio': 'Resource Hedging'})
        resource_h_total_amp_score += amp_score_rh
    for entry_rh in temp_resource_h_list:
        sub_alloc_rh = (entry_rh['amplified_score'] / resource_h_total_amp_score) * 100.0 if resource_h_total_amp_score > 1e-9 else 0.0
        entry_rh['combined_percent_allocation'] = safe_score(sub_alloc_rh * eta_pct / 100.0 * lambda_hedge_pct / 100.0)
        combined_portfolio_list_build.append(entry_rh)
        
    # Normalize combined_percent_allocation to sum to 100% (of the non-cash portion)
    total_combined_alloc_sum = sum(e.get('combined_percent_allocation', 0.0) for e in combined_portfolio_list_build)
    if total_combined_alloc_sum > 1e-9:
        norm_factor_comb = 100.0 / total_combined_alloc_sum
        for e in combined_portfolio_list_build:
            e['combined_percent_allocation'] *= norm_factor_comb
            
    # Tailored Portfolio Calculation
    tailored_portfolio_entries_build = []
    total_actual_money_allocated_stocks_hedges = 0.0
    for entry_t in combined_portfolio_list_build:
        alloc_pct_t = safe_score(entry_t.get('combined_percent_allocation', 0.0))
        live_price_t = safe_score(entry_t.get('live_price', 0.0))
        if alloc_pct_t > 1e-9 and live_price_t > 1e-9:
            target_alloc_val_t = value_for_stocks_hedges * (alloc_pct_t / 100.0)
            shares_t = round(target_alloc_val_t / live_price_t, 1) if frac_shares else float(math.floor(target_alloc_val_t / live_price_t))
            shares_t = max(0.0, shares_t)
            actual_money_alloc_t = shares_t * live_price_t
            share_thresh_t = 0.1 if frac_shares else 1.0
            if shares_t >= share_thresh_t:
                actual_pct_total_t = (actual_money_alloc_t / epsilon) * 100.0 if epsilon > 0 else 0.0
                tailored_portfolio_entries_build.append({
                    'ticker': entry_t['ticker'], 'portfolio': entry_t['portfolio'], 'shares': shares_t,
                    'actual_money_allocation': actual_money_alloc_t,
                    'actual_percent_allocation_total': actual_pct_total_t,
                    'raw_invest_score': entry_t['raw_invest_score'] # For sorting tailored output
                })
                total_actual_money_allocated_stocks_hedges += actual_money_alloc_t
                
    remaining_from_stocks_hedges = value_for_stocks_hedges - total_actual_money_allocated_stocks_hedges
    final_cash_value_build = max(0.0, initial_omega_cash_val + remaining_from_stocks_hedges)
    final_cash_percent_build = (final_cash_value_build / epsilon) * 100.0 if epsilon > 0 else 0.0
    final_cash_percent_build = max(0.0, min(100.0, final_cash_percent_build))

    tailored_portfolio_entries_build.sort(key=lambda x: x.get('raw_invest_score', -float('inf')), reverse=True)
    common_val_actual = sum(e['actual_money_allocation'] for e in tailored_portfolio_entries_build if e['portfolio'] == 'Common Stock')
    market_h_val_actual = sum(e['actual_money_allocation'] for e in tailored_portfolio_entries_build if e['portfolio'] == 'Market Hedging')
    resource_h_val_actual = sum(e['actual_money_allocation'] for e in tailored_portfolio_entries_build if e['portfolio'] == 'Resource Hedging')
    
    print("    Finished Step: Building & Processing Final Cultivate Portfolios.")
    return (combined_portfolio_list_build, tailored_portfolio_entries_build, final_cash_value_build, final_cash_percent_build,
            value_for_stocks_hedges, common_val_actual, market_h_val_actual, resource_h_val_actual, initial_omega_cash_val)

async def run_cultivate_analysis_singularity(
    portfolio_value: float, 
    frac_shares: bool, 
    cultivate_code_str: str, 
    is_saving_run: bool = False
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], float, str, float, bool, Optional[str]]:
    """ 
    Orchestrates Cultivate portfolio analysis for Singularity. 
    Returns:
        1. combined_portfolio_list_cult (List[Dict]): For saving combined data.
        2. tailored_portfolio_entries_cult (List[Dict]): Structured tailored holdings (stocks/hedges).
        3. final_cash_value_cult (float): Final cash in tailored portfolio.
        4. cultivate_code_str (str): The cultivate code used ('A' or 'B').
        5. epsilon_val (float): The portfolio value (epsilon) used.
        6. frac_shares (bool): The fractional shares preference used.
        7. error_msg (Optional[str]): None on success, or an error message string.
    """
    # print(f"\n--- Cultivate Analysis (Code: {cultivate_code_str.upper()}) ---") # Reduced verbosity for internal calls
    epsilon_val = safe_score(portfolio_value)
    if epsilon_val <= 0:
        # print("Error: Portfolio value must be a positive number for Cultivate analysis.") # Redundant if called by handler
        return [], [], 0.0, cultivate_code_str, epsilon_val, frac_shares, "Error: Invalid portfolio value"

    # Step 1: Get Allocation Score (Sigma)
    if not is_saving_run: print("Step 1/7: Getting Allocation Score (Sigma)...")
    allocation_score_cult, _, _ = get_allocation_score() 
    if allocation_score_cult is None:
        # print("Error: Failed to retrieve Allocation Score (Sigma). Aborting Cultivate.")
        return [], [], 0.0, cultivate_code_str, epsilon_val, frac_shares, "Error: Failed to get Allocation Score"
    sigma_cult = allocation_score_cult

    # Step 2: Calculate Formula Variables
    if not is_saving_run: print("Step 2/7: Calculating Cultivate Formula Variables...")
    formula_results_cult = calculate_cultivate_formulas_singularity(sigma_cult) 
    if formula_results_cult is None:
        # print("Error: Failed to calculate portfolio structure variables. Aborting Cultivate.")
        return [], [], 0.0, cultivate_code_str, epsilon_val, frac_shares, "Error: Formula calculation failed"
    
    tickers_to_process_cult = []
    if cultivate_code_str.upper() == 'A':
        if not is_saving_run: print("Step 3/7: Screening stocks (Cultivate Code A)...")
        tickers_to_process_cult = screen_stocks_singularity() 
        if not tickers_to_process_cult and not is_saving_run:
            print("Warning: No stocks passed initial screening for Code A. Proceeding with hedging/cash only.")
    elif cultivate_code_str.upper() == 'B':
        if not is_saving_run: print("Step 3/7: Getting S&P 500 tickers (Cultivate Code B)...")
        tickers_to_process_cult = get_spy_symbols_singularity() 
        if not tickers_to_process_cult:
            # print("Error: Failed to retrieve S&P 500 symbols for Code B. Aborting Cultivate.")
            return [], [], 0.0, cultivate_code_str, epsilon_val, frac_shares, "Error: Failed to get S&P 500 symbols"

    if not is_saving_run: print("Step 4/7: Calculating Metrics (Beta/Corr/Leverage)...")
    spy_hist_data_metrics_cult = get_yf_data_singularity(['SPY'], period="10y", interval="1d") 
    
    metrics_dict_cult = {}
    if not spy_hist_data_metrics_cult.empty: 
        if tickers_to_process_cult: 
            metrics_dict_cult = calculate_metrics_singularity(tickers_to_process_cult, spy_hist_data_metrics_cult) 
            if not is_saving_run: save_initial_metrics_singularity(metrics_dict_cult, tickers_to_process_cult) 
    elif not is_saving_run and tickers_to_process_cult : 
        print("Warning: Could not get SPY data for metrics; metrics calculation for common stocks will be skipped.")
        
    if not is_saving_run: print("Step 5/7: Calculating Invest Scores for all relevant tickers...")
    invest_scores_all_cult = {}
    all_tickers_for_scoring_cult = list(set((tickers_to_process_cult if tickers_to_process_cult else []) + HEDGING_TICKERS))
    if all_tickers_for_scoring_cult:
        score_tasks = [calculate_ema_invest(ticker, 2) for ticker in all_tickers_for_scoring_cult] 
        score_results = await asyncio.gather(*score_tasks, return_exceptions=True) 
        for i, ticker_sc in enumerate(all_tickers_for_scoring_cult):
            res_sc = score_results[i]
            if isinstance(res_sc, Exception) or res_sc is None or res_sc[0] is None or res_sc[1] is None:
                invest_scores_all_cult[ticker_sc] = {'score': -float('inf'), 'live_price': 0.0, 'error': 'Fetch/Score failed'}
            else:
                invest_scores_all_cult[ticker_sc] = {'score': safe_score(res_sc[1]), 'live_price': safe_score(res_sc[0])}
    # else: print("Warning: No tickers (common or hedging) to score for Cultivate analysis.") # Already handled if list is empty

    if not is_saving_run: print("Step 6/7: Selecting Final Common Stock Tickers (Tf)...")
    final_common_stock_tickers_cult, selection_warning_cult, _ = await select_tickers_singularity(
        tickers_to_filter=tickers_to_process_cult, metrics=metrics_dict_cult, 
        invest_scores_all=invest_scores_all_cult, formula_results=formula_results_cult,
        portfolio_value=epsilon_val
    )
    if selection_warning_cult and not is_saving_run: print(f"*** {selection_warning_cult} ***")

    if not is_saving_run: print("Step 7/7: Building & Processing Final Cultivate Portfolios...")
    (combined_portfolio_list_cult, tailored_portfolio_entries_cult, final_cash_value_cult, 
     final_cash_percent_cult, value_for_stocks_hedges_cult, common_value_actual_cult, 
     market_hedge_value_actual_cult, resource_hedge_value_actual_cult, 
     initial_omega_cash_cult) = build_and_process_portfolios_singularity(
        common_stock_tickers=final_common_stock_tickers_cult, amplification=formula_results_cult.get('delta', 1.0),
        total_portfolio_value=epsilon_val, cash_allocation_omega=formula_results_cult.get('omega', 0.0),
        frac_shares=frac_shares, invest_scores_all=invest_scores_all_cult,
        eta=formula_results_cult.get('eta', 0.0), kappa=formula_results_cult.get('kappa', 0.0),
        lambda_hedge=formula_results_cult.get('lambda_hedge', 0.0)
    )

    if not is_saving_run:
        # ... (Display logic for combined, tailored, pie chart, greeks - remains the same) ...
        print("\n--- Cultivate Analysis Results ---")
        print("\n**Combined Portfolio Allocation (Relative to Non-Cash Portion)**")
        combined_table_data_cult = []
        if combined_portfolio_list_cult:
            sorted_combined_cult = sorted(combined_portfolio_list_cult, key=lambda x: x.get('combined_percent_allocation',0.0), reverse=True)
            for entry_comb_c in sorted_combined_cult:
                alloc_pct_c = safe_score(entry_comb_c.get('combined_percent_allocation', 0.0))
                if alloc_pct_c >= 0.01: 
                    price_f_c = f"${safe_score(entry_comb_c.get('live_price',0.0)):.2f}"
                    score_f_c = f"{safe_score(entry_comb_c.get('raw_invest_score',0.0)):.2f}%"
                    combined_table_data_cult.append([
                        entry_comb_c.get('ticker','ERR'), entry_comb_c.get('portfolio','?'), price_f_c, score_f_c, f"{alloc_pct_c:.2f}%"
                    ])
            if combined_table_data_cult:
                print(tabulate(combined_table_data_cult, headers=["Ticker", "Portfolio", "Price", "Raw Score", "Combined % Alloc (Non-Cash)"], tablefmt="pretty"))
            else: print("No significant allocations in the combined (non-cash) portfolio.")
        else: print("No combined portfolio data generated (excluding cash).")

        print("\n**Tailored Portfolio (Actual $ and % of Total Epsilon)**")
        tailored_table_data_cult_disp = []
        if tailored_portfolio_entries_cult:
            for item_tail_c in tailored_portfolio_entries_cult:
                shares_f_c = f"{item_tail_c['shares']:.1f}" if frac_shares and item_tail_c['shares'] > 0 else f"{int(item_tail_c['shares'])}"
                money_f_c = f"${safe_score(item_tail_c.get('actual_money_allocation',0.0)):,.2f}"
                percent_f_c = f"{safe_score(item_tail_c.get('actual_percent_allocation_total',0.0)):.2f}%"
                tailored_table_data_cult_disp.append([
                    item_tail_c.get('ticker','ERR'), item_tail_c.get('portfolio','?'), shares_f_c, money_f_c, percent_f_c
                ])
        tailored_table_data_cult_disp.append(['Cash', 'Cash', '-', f"${safe_score(final_cash_value_cult):,.2f}", f"{safe_score(final_cash_percent_cult):.2f}%"])
        print(tabulate(tailored_table_data_cult_disp, headers=["Ticker", "Portfolio", "Shares", "$ Allocation", "% of Total Epsilon"], tablefmt="pretty"))

        pie_data_cult = [{'ticker': item['ticker'], 'value': item['actual_money_allocation']} for item in tailored_portfolio_entries_cult if item.get('actual_money_allocation',0) > 1e-9]
        if final_cash_value_cult > 1e-9: pie_data_cult.append({'ticker': 'Cash', 'value': final_cash_value_cult})
        if pie_data_cult:
            generate_portfolio_pie_chart(pie_data_cult, f"Cultivate Portfolio (Code {cultivate_code_str.upper()}, Epsilon ${epsilon_val:,.0f})", "cultivate_pie")

        print("\n**The Invest Greeks (Cultivate Portfolio Structure)**")
        num_tickers_sigma_report_cult = max(1, math.ceil(0.3 * math.sqrt(epsilon_val)))
        greek_data_cult = [
            ["Sigma (Allocation Score)", f"{safe_score(sigma_cult):.2f}"],
            ["Lambda (Stock/Hedge % of Epsilon)", f"{safe_score(formula_results_cult.get('lambda',0.0)):.2f}%"],
            ["Lambda Hedge (Hedge % of Non-Cash)", f"{safe_score(formula_results_cult.get('lambda_hedge',0.0)):.2f}%"],
            ["Kappa (Market Hedge % of Lambda Hedge)", f"{safe_score(formula_results_cult.get('kappa',0.0)):.2f}%"],
            ["Eta (Resource Hedge % of Lambda Hedge)", f"{safe_score(formula_results_cult.get('eta',0.0)):.2f}%"],
            ["Omega (Cash % of Epsilon - Initial)", f"{safe_score(formula_results_cult.get('omega',0.0)):.2f}%"],
            ["Alpha (Common Stock % of Lambda)", f"{safe_score(formula_results_cult.get('alpha',0.0)):.2f}%"],
            ["Beta (Hedging % of Lambda - Theoretical)", f"{safe_score(formula_results_cult.get('beta_alloc',0.0)):.2f}%"],
            ["Delta (Amplification)", f"{safe_score(formula_results_cult.get('delta',0.0)):.2f}x"],
            ["Epsilon (Total Portfolio Value)", f"${safe_score(epsilon_val):,.2f}"],
            ["Sigma Portfolio (Target Common Tickers)", f"{num_tickers_sigma_report_cult:.0f}"],
            ["Selected Common Tickers (Tf)", f"{len(final_common_stock_tickers_cult)}"],
            ["Value for Stocks/Hedges (Non-Cash)", f"${safe_score(value_for_stocks_hedges_cult):,.2f}"],
            ["  Actual Common Stock Value", f"${safe_score(common_value_actual_cult):,.2f}"],
            ["  Actual Market Hedge Value", f"${safe_score(market_hedge_value_actual_cult):,.2f}"],
            ["  Actual Resource Hedge Value", f"${safe_score(resource_hedge_value_actual_cult):,.2f}"],
            ["Final Cash Value (Tailored)", f"${safe_score(final_cash_value_cult):,.2f} ({final_cash_percent_cult:.2f}%)"],
        ]
        print(tabulate(greek_data_cult, headers=["Variable", "Value"], tablefmt="grid"))
        print("\n--- Cultivate Analysis Complete ---")

    return (combined_portfolio_list_cult, tailored_portfolio_entries_cult, final_cash_value_cult, 
            cultivate_code_str, epsilon_val, frac_shares, None)

async def save_cultivate_data_internal_singularity(combined_portfolio_data: List[Dict], date_str: str, cultivate_code: str, epsilon: float):
    """Saves combined Cultivate portfolio data to CSV. Singularity version."""
    if not combined_portfolio_data:
        print(f"[Save Cultivate]: No valid combined portfolio data to save for Code {cultivate_code}, Epsilon {epsilon}.")
        return

    # Sort by combined_percent_allocation (which is relative to non-cash part)
    sorted_combined_save = sorted(combined_portfolio_data, key=lambda x: x.get('combined_percent_allocation', 0.0), reverse=True)
    
    epsilon_int_save = int(epsilon)
    save_file_cult = f"{CULTIVATE_COMBINED_DATA_FILE_PREFIX}{cultivate_code.upper()}_{epsilon_int_save}.csv"
    file_exists_cult_save = os.path.isfile(save_file_cult)
    save_count_cult = 0
    headers_cult_save = ['DATE', 'TICKER', 'PORTFOLIO_TYPE', 'PRICE', 'RAW_INVEST_SCORE', 'COMBINED_ALLOCATION_PERCENT_NON_CASH']

    try:
        with open(save_file_cult, 'a', newline='', encoding='utf-8') as f_cult_save:
            writer_cult = csv.DictWriter(f_cult_save, fieldnames=headers_cult_save)
            if not file_exists_cult_save or os.path.getsize(f_cult_save.name) == 0:
                writer_cult.writeheader()
            
            for item_s_c in sorted_combined_save:
                alloc_pct_s_c = safe_score(item_s_c.get('combined_percent_allocation', 0.0))
                # Only save if allocation (relative to non-cash) is significant
                if alloc_pct_s_c > 1e-4 : 
                    writer_cult.writerow({
                        'DATE': date_str,
                        'TICKER': item_s_c.get('ticker', 'ERR'),
                        'PORTFOLIO_TYPE': item_s_c.get('portfolio', '?'), # Common Stock, Market Hedging, Resource Hedging
                        'PRICE': f"{safe_score(item_s_c.get('live_price')):.2f}" if item_s_c.get('live_price') is not None else "N/A",
                        'RAW_INVEST_SCORE': f"{safe_score(item_s_c.get('raw_invest_score')):.2f}%" if item_s_c.get('raw_invest_score') is not None else "N/A",
                        'COMBINED_ALLOCATION_PERCENT_NON_CASH': f"{alloc_pct_s_c:.2f}%"
                    })
                    save_count_cult +=1
        print(f"[Save Cultivate]: Saved {save_count_cult} rows of combined data for Code '{cultivate_code.upper()}' (Epsilon: {epsilon_int_save}) to '{save_file_cult}' for date {date_str}.")
    except IOError as e_io_cult_save:
        print(f"Error [Save Cultivate]: Writing to save file '{save_file_cult}': {e_io_cult_save}")
    except Exception as e_s_cult:
        print(f"Error [Save Cultivate]: Processing/saving data for Code '{cultivate_code.upper()}' (Epsilon: {epsilon}): {e_s_cult}")
        traceback.print_exc()

async def handle_cultivate_command(args: List[str], ai_params: Optional[Dict] = None):
    """
    Handles the /cultivate command for CLI and AI.
    CLI: /cultivate <Code A/B> <PortfolioValue> <FracShares yes/no> [save_code 3725]
    AI: Expects ai_params with: cultivate_code, portfolio_value, use_fractional_shares,
        and optional action ("run_analysis" or "save_data") and date_to_save.
    Returns a summary string for AI.
    """
    print("\n--- /cultivate Command ---")
    summary_for_ai = "Cultivate command initiated."

    cult_code_input = None
    portfolio_value_input = None
    frac_shares_input_bool = None
    action = "run_analysis" # Default action for AI if not specified
    date_to_save_cult = None

    if ai_params: # Called by AI
        print(f"AI Call to /cultivate with params: {ai_params}")
        try:
            cult_code_input = ai_params.get("cultivate_code", "").upper()
            if cult_code_input not in ['A', 'B']:
                return "Error for AI (/cultivate): 'cultivate_code' must be 'A' or 'B'."

            portfolio_value_input = float(ai_params.get("portfolio_value", 0))
            if portfolio_value_input <= 0:
                return "Error for AI (/cultivate): 'portfolio_value' must be a positive number."

            frac_shares_input_bool = bool(ai_params.get("use_fractional_shares", False)) # Default to False

            action = ai_params.get("action", "run_analysis").lower()
            if action not in ["run_analysis", "save_data"]:
                return f"Error for AI (/cultivate): Invalid action '{action}'. Must be 'run_analysis' or 'save_data'."

            if action == "save_data":
                date_to_save_cult_str = ai_params.get("date_to_save")
                if not date_to_save_cult_str:
                    return "Error for AI (/cultivate): 'date_to_save' (MM/DD/YYYY) is required for 'save_data' action."
                datetime.strptime(date_to_save_cult_str, '%m/%d/%Y') # Validate
                date_to_save_cult = date_to_save_cult_str
            
            # AI path directly calls run_cultivate_analysis_singularity and potentially save
            print(f"AI: Running Cultivate Analysis (Code: {cult_code_input}, Value: {portfolio_value_input}, FracShares: {frac_shares_input_bool}, Action: {action})...")
            # run_cultivate_analysis_singularity returns:
            # (combined_list, tailored_entries, final_cash, code_str, eps_val, frac_s_val, error_msg)
            combined_data, tailored_entries, final_cash, \
            code_used, eps_used, frac_s_used, err_msg = await run_cultivate_analysis_singularity(
                portfolio_value=portfolio_value_input,
                frac_shares=frac_shares_input_bool,
                cultivate_code_str=cult_code_input,
                is_saving_run=(action == "save_data") # Suppress display if primary goal is saving
            )

            if err_msg:
                summary_for_ai = f"Cultivate analysis (Code {cult_code_input}) failed: {err_msg}"
                print(summary_for_ai)
                return summary_for_ai

            summary_for_ai = f"Cultivate analysis (Code {code_used}, Value ${eps_used:,.0f}, FracShares {frac_s_used}) completed. "
            if tailored_entries or final_cash > 0: # Summarize results if any
                summary_for_ai += f"Final cash: ${final_cash:,.2f}. "
                if tailored_entries:
                    summary_for_ai += "Top holdings: " + ", ".join([f"{d['ticker']}({d.get('actual_percent_allocation_total',0):.1f}%)" for d in tailored_entries[:3]])
                else:
                    summary_for_ai += "No stock holdings in tailored portfolio."
            
            if action == "save_data":
                if combined_data and date_to_save_cult:
                    print(f"AI: Saving Cultivate data for Code {code_used}, Value ${eps_used:,.0f} for date {date_to_save_cult}...")
                    # save_cultivate_data_internal_singularity needs to be defined and async
                    await save_cultivate_data_internal_singularity(combined_data, date_to_save_cult, code_used, eps_used)
                    summary_for_ai += f" Data saved for {date_to_save_cult}."
                else:
                    save_err = "Save action requested by AI, but no combined data generated or date missing."
                    print(save_err)
                    summary_for_ai += f" {save_err}"
            
            return summary_for_ai

        except ValueError as ve: # For float conversion or date format
            return f"Error for AI (/cultivate): Invalid data type for a parameter. {ve}"
        except KeyError as ke:
            return f"Error for AI (/cultivate): Missing expected parameter from AI: {ke}"
        except Exception as e_cult_ai:
            traceback.print_exc()
            return f"Error for AI (/cultivate) processing request: {e_cult_ai}"

    else: # --- CLI Path ---
        try:
            if len(args) < 3:
                print("CLI Usage: /cultivate <Code A/B> <PortfolioValue> <FracShares yes/no> [save_code 3725]")
                return None

            cult_code_input_cli = args[0].upper()
            if cult_code_input_cli not in ['A', 'B']:
                print("CLI Error: Invalid Cultivate Code. Must be 'A' or 'B'.")
                return None
            
            portfolio_value_input_cli = float(args[1])
            if portfolio_value_input_cli <= 0:
                print("CLI Error: Portfolio value must be a positive number.")
                return None
                
            frac_shares_input_str_cli = args[2].lower()
            if frac_shares_input_str_cli not in ['yes', 'no']:
                print("CLI Error: Fractional shares input must be 'yes' or 'no'.")
                return None
            frac_shares_input_bool_cli = frac_shares_input_str_cli == 'yes'

            save_action_code_cli = args[3] if len(args) > 3 else None

            if save_action_code_cli == "3725":
                date_to_save_cult_cli = input(f"CLI: Enter date (MM/DD/YYYY) to save combined Cultivate data for Code {cult_code_input_cli}, Value ${portfolio_value_input_cli:,.0f}: ")
                try:
                    datetime.strptime(date_to_save_cult_cli, '%m/%d/%Y') # Validate date
                    print(f"CLI: Generating Cultivate data for saving...")
                    combined_data_s, _, _, code_s, eps_s, _, err_s = await run_cultivate_analysis_singularity(
                        portfolio_value=portfolio_value_input_cli, frac_shares=frac_shares_input_bool_cli,
                        cultivate_code_str=cult_code_input_cli, is_saving_run=True
                    )
                    if err_s: print(f"CLI Error during data generation for save: {err_s}"); return None
                    await save_cultivate_data_internal_singularity(combined_data_s, date_to_save_cult_cli, code_s, eps_s)
                except ValueError:
                    print("CLI Error: Invalid date format. Save operation cancelled.")
                except Exception as e_save_cult_op_cli:
                    print(f"CLI Error occurred during Cultivate data save operation: {e_save_cult_op_cli}")
                    traceback.print_exc()
            else:
                # Run analysis for display (CLI)
                await run_cultivate_analysis_singularity(
                    portfolio_value=portfolio_value_input_cli, frac_shares=frac_shares_input_bool_cli,
                    cultivate_code_str=cult_code_input_cli, is_saving_run=False
                )
            return None # CLI path prints directly

        except ValueError:
            print("CLI Error: Invalid input. Portfolio value must be a number. FracShares must be 'yes' or 'no'.")
            return None
        except IndexError:
             print("CLI Error: Insufficient arguments for /cultivate.")
             return None
        except Exception as e_cult_handle_cli:
            print(f"CLI Error occurred handling /cultivate command: {e_cult_handle_cli}")
            traceback.print_exc()
            return None

# --- Assess Command Functions ---

def ask_singularity_input(prompt: str, validation_fn=None, error_msg: str = "Invalid input.", default_val=None) -> Optional[str]:
    """
    Helper function to ask for user input in Singularity, with optional validation.
    Returns validated string or None if validation fails or user cancels.
    """
    while True:
        full_prompt = f"{prompt}"
        if default_val is not None:
            full_prompt += f" (default: {default_val}, press Enter to use)"
        full_prompt += ": "
        
        user_response = input(full_prompt).strip()
        if not user_response and default_val is not None:
            return str(default_val) # Return default if user just presses Enter

        if validation_fn:
            if validation_fn(user_response):
                return user_response
            else:
                print(error_msg)
                retry = input("Try again? (yes/no, default: yes): ").lower()
                if retry == 'no':
                    return None
        else: # No validation function, accept any non-empty input
            return user_response


# --- Assess Command Functions ---

async def calculate_portfolio_beta_correlation_singularity(
    portfolio_holdings: List[Dict[str, Any]], 
    total_portfolio_value: float,
    backtest_period: str
) -> Optional[tuple[float, float]]:
    """
    Calculates weighted average Beta and Correlation for a given portfolio against SPY.
    `portfolio_holdings`: List of dicts, each with 'ticker' and 'value' (actual dollar allocation).
    Cash should be one of the tickers with value if present.
    """
    # print(f"  Calculating Beta/Correlation over {backtest_period}...") # Reduced verbosity
    if not portfolio_holdings or total_portfolio_value <= 0:
        return None

    valid_holdings_for_calc = [h for h in portfolio_holdings if isinstance(h.get('value'), (int, float)) and h['value'] > 1e-9]
    if not valid_holdings_for_calc:
        if any(h['ticker'] == 'Cash' for h in portfolio_holdings): return 0.0, 0.0 
        return None

    portfolio_tickers_assess = [h['ticker'] for h in valid_holdings_for_calc if h['ticker'] != 'Cash']
    
    if not portfolio_tickers_assess and any(h['ticker'] == 'Cash' for h in valid_holdings_for_calc):
        return 0.0, 0.0
    if not portfolio_tickers_assess: 
        return None

    all_tickers_for_hist_assess = list(set(portfolio_tickers_assess + ['SPY']))
    hist_data_assess = get_yf_data_singularity(all_tickers_for_hist_assess, period=backtest_period, interval="1d")

    if hist_data_assess.empty or 'SPY' not in hist_data_assess.columns:
        # print(f"  Error: Could not fetch sufficient historical data for SPY or portfolio tickers for period {backtest_period}.")
        return None
    
    if hist_data_assess['SPY'].isnull().all() or len(hist_data_assess['SPY'].dropna()) < 20: 
        # print(f"  Error: Insufficient valid data points for SPY over period {backtest_period}.")
        return None

    daily_returns_assess = hist_data_assess.pct_change().iloc[1:] 
    if daily_returns_assess.empty or 'SPY' not in daily_returns_assess.columns:
        # print("  Error calculating daily returns or SPY returns missing for assessment.")
        return None

    spy_returns_assess = daily_returns_assess['SPY'].dropna()
    if spy_returns_assess.empty or spy_returns_assess.std() == 0:
        # print("  Error: SPY returns are empty or have no variance. Cannot calculate Beta/Correlation.")
        return None

    stock_metrics_assess = {}
    for ticker_met_assess in portfolio_tickers_assess:
        beta_val_assess, correlation_val_assess = np.nan, np.nan 
        if ticker_met_assess in daily_returns_assess.columns and not daily_returns_assess[ticker_met_assess].isnull().all():
            ticker_returns_assess = daily_returns_assess[ticker_met_assess].dropna()
            
            aligned_data = pd.concat([ticker_returns_assess, spy_returns_assess], axis=1, join='inner').dropna()
            if len(aligned_data) >= 20: 
                aligned_ticker_returns = aligned_data.iloc[:, 0]
                aligned_spy_returns = aligned_data.iloc[:, 1]

                if aligned_ticker_returns.std() > 1e-9 and aligned_spy_returns.std() > 1e-9: 
                    try:
                        cov_matrix_assess = np.cov(aligned_ticker_returns, aligned_spy_returns)
                        if cov_matrix_assess.shape == (2,2) and cov_matrix_assess[1, 1] != 0:
                             beta_val_assess = cov_matrix_assess[0, 1] / cov_matrix_assess[1, 1]
                        
                        corr_coef_matrix = np.corrcoef(aligned_ticker_returns, aligned_spy_returns)
                        if corr_coef_matrix.shape == (2,2):
                            correlation_val_assess = corr_coef_matrix[0, 1]
                            if pd.isna(correlation_val_assess): correlation_val_assess = 0.0 
                    except (ValueError, IndexError, TypeError, np.linalg.LinAlgError): 
                        pass 
                else: 
                    beta_val_assess, correlation_val_assess = 0.0, 0.0
        stock_metrics_assess[ticker_met_assess] = {'beta': beta_val_assess, 'correlation': correlation_val_assess}
    
    stock_metrics_assess['Cash'] = {'beta': 0.0, 'correlation': 0.0}

    weighted_beta_sum_assess, weighted_correlation_sum_assess = 0.0, 0.0
    for holding_assess in valid_holdings_for_calc: 
        ticker_h_assess = holding_assess['ticker']
        value_h_assess = holding_assess['value'] 
        weight_h_assess = value_h_assess / total_portfolio_value 
        
        metrics_for_ticker = stock_metrics_assess.get(ticker_h_assess, {'beta': 0.0, 'correlation': 0.0}) 
        beta_for_calc = metrics_for_ticker.get('beta', 0.0) 
        corr_for_calc = metrics_for_ticker.get('correlation', 0.0)

        if not pd.isna(beta_for_calc): 
            weighted_beta_sum_assess += weight_h_assess * beta_for_calc
        if not pd.isna(corr_for_calc):
            weighted_correlation_sum_assess += weight_h_assess * corr_for_calc
            
    return weighted_beta_sum_assess, weighted_correlation_sum_assess

async def handle_assess_command(args: List[str], ai_params: Optional[Dict] = None):
    """
    Handles the /assess command for CLI and AI.
    CLI: /assess <A/B/C/D> [specific_args...]
    AI: Called with ai_params={"assess_code": "A/B/C/D", ...other_params...}
    Returns a summary string for AI, prints to console for CLI/AI.
    """
    print("\n--- /assess Command ---")
    summary_for_ai = "Assessment initiated." # Default summary

    assess_code_cli = args[0].upper() if args else None
    
    # Determine assess_code and parameters based on call type
    assess_code = None
    specific_params = {} # For AI parameters

    if ai_params: # Called by AI
        print(f"AI Call with params: {ai_params}")
        assess_code = ai_params.get("assess_code", "").upper()
        specific_params = ai_params
    elif assess_code_cli: # Called from CLI
        assess_code = assess_code_cli
        # CLI specific args will be handled within each case using original 'args' list
    else: # No assess code provided from CLI
        print("Usage: /assess <AssessCode A/B/C/D> [additional_args...]. Type /help for more details.")
        return "Error: Assess code (A, B, C, or D) not specified." if ai_params else None

    # --- Code A: Stock Volatility Assessment ---
    if assess_code == 'A':
        print("--- Assess Code A: Stock Volatility ---")
        tickers_str_a, timeframe_str_a, risk_tolerance_a = None, None, None
        try:
            if ai_params:
                tickers_str_a = specific_params.get("tickers_str")
                timeframe_str_a = specific_params.get("timeframe_str", "1Y") # Default for AI
                risk_tolerance_a = int(specific_params.get("risk_tolerance", 3)) # Default for AI
                if not tickers_str_a:
                    return "Error for AI (Assess A): 'tickers_str' is required."
            elif len(args) >= 4: # CLI: /assess A <tickers> <timeframe> <risk_tolerance>
                tickers_str_a = args[1]
                timeframe_str_a = args[2].upper()
                risk_tolerance_a = int(args[3])
            else:
                print("CLI Usage for Code A: /assess A <tickers_comma_sep> <timeframe 1Y/3M/1M> <risk_tolerance 1-5>")
                return "Error: Insufficient arguments for Assess Code A." if ai_params else None

            # Call the dedicated AI function we made earlier, or integrate logic here
            # For this refactor, we integrate the logic directly:
            # (This logic is similar to the `ai_assess_stock_volatility` previously created)
            
            tickers_list_a = [t.strip().upper() for t in tickers_str_a.split(',') if t.strip()]
            if not tickers_list_a:
                return f"Error (Assess A): No valid tickers in '{tickers_str_a}'."

            timeframe_upper = timeframe_str_a.upper()
            timeframe_mapping_a = {'1Y': "1y", '3M': "3mo", '1M': "1mo"}
            plot_ema_map_a = {'1Y': 2, '3M': 3, '1M': 3}

            if timeframe_upper not in timeframe_mapping_a:
                return f"Error (Assess A): Invalid timeframe '{timeframe_str_a}'. Use 1Y, 3M, or 1M."
            selected_period_a = timeframe_mapping_a[timeframe_upper]
            plot_ema_sens_a = plot_ema_map_a[timeframe_upper]

            if not (1 <= risk_tolerance_a <= 5):
                return f"Error (Assess A): Invalid risk tolerance '{risk_tolerance_a}'. Must be 1-5."

            results_data_a_for_table = []
            assessment_summaries_for_ai_parts = []

            for ticker_a_item in tickers_list_a: # Generate graphs first
                print(f"  Generating graph for {ticker_a_item} (EMA Sens: {plot_ema_sens_a})...")
                await asyncio.to_thread(plot_ticker_graph, ticker_a_item, plot_ema_sens_a) # plot_ticker_graph needs to be defined

            for ticker_a_item in tickers_list_a:
                # (Calculations from your original assess A: hist_a_df, close_prices_a, aabc_a, vol_score_a, etc.)
                # For brevity, this detailed calculation logic is assumed to be here
                # Ensure get_yf_data_singularity, safe_score, etc. are defined and imported
                try:
                    hist_a_df = await asyncio.to_thread(get_yf_data_singularity, [ticker_a_item], period=selected_period_a)
                    if hist_a_df.empty or ticker_a_item not in hist_a_df.columns or len(hist_a_df[ticker_a_item].dropna()) <= 1:
                        # ... (handle data error) ...
                        results_data_a_for_table.append([ticker_a_item, "N/A", "N/A", "N/A", "Data Error"])
                        assessment_summaries_for_ai_parts.append(f"{ticker_a_item}: Data Error.")
                        continue
                    # ... (perform AAPC, VolScore, PeriodChange, RiskMatch calculations)
                    # This is a placeholder for the detailed calculations from your original script's /assess A
                    close_prices_a = hist_a_df[ticker_a_item].dropna()
                    abs_pct_change_a = close_prices_a.pct_change().abs() * 100
                    aabc_a = abs_pct_change_a.iloc[1:].mean() if len(abs_pct_change_a.iloc[1:]) > 0 else 0.0
                    score_map_a = [(1,0),(2,1),(3,2),(4,3),(5,4),(6,5),(7,6),(8,7),(9,8),(10,9)]; vol_score_a = 10
                    for th, sc in score_map_a:
                        if aabc_a <= th: vol_score_a = sc; break
                    start_p = close_prices_a.iloc[0]; end_p = close_prices_a.iloc[-1]
                    period_change = ((end_p - start_p) / start_p) * 100 if start_p != 0 else 0.0
                    risk_map_ranges = {1:(0,1), 2:(2,3), 3:(4,5), 4:(6,7), 5:(8,10)}
                    correspondence = "Matches" if risk_map_ranges[risk_tolerance_a][0] <= vol_score_a <= risk_map_ranges[risk_tolerance_a][1] else "No Match"
                    
                    results_data_a_for_table.append([ticker_a_item, f"{period_change:.2f}%", f"{aabc_a:.2f}%", vol_score_a, correspondence])
                    assessment_summaries_for_ai_parts.append(f"{ticker_a_item}({timeframe_upper},RT{risk_tolerance_a}):Chg {period_change:.1f}%,AAPC {aabc_a:.1f}%,VolSc {vol_score_a},Match {correspondence}")
                except Exception as e:
                    results_data_a_for_table.append([ticker_a_item, "CalcErr", "CalcErr", "CalcErr", "CalcErr"])
                    assessment_summaries_for_ai_parts.append(f"{ticker_a_item}: Calculation Error ({e}).")

            if results_data_a_for_table: # Print for CLI/User
                print("\n**Stock Volatility Assessment Results (Code A)**")
                results_data_a_for_table.sort(key=lambda x: x[3] if isinstance(x[3], (int,float)) else float('inf'))
                print(tabulate(results_data_a_for_table, headers=["Ticker", f"{timeframe_upper} Change", "AAPC", "Vol Score", "Risk Match"], tablefmt="pretty"))
            
            summary_for_ai = "Assess A (Stock Volatility) results: " + " | ".join(assessment_summaries_for_ai_parts) if assessment_summaries_for_ai_parts else "No results for Assess A."

        except ValueError as ve: # For int conversion errors primarily
            summary_for_ai = f"Error (Assess A): Invalid parameter type. {ve}"
            print(summary_for_ai)
        except Exception as e_a:
            summary_for_ai = f"Error in Assess Code A: {e_a}"
            print(summary_for_ai)
            traceback.print_exc()
        return summary_for_ai

    # --- Code B: Manual Portfolio Assessment ---
    elif assess_code == 'B':
        print("--- Assess Code B: Manual Portfolio ---")
        backtest_period_b = None
        portfolio_holdings_b = [] # List of dicts: {'ticker': str, 'value': float}
        
        try:
            if ai_params:
                backtest_period_b = specific_params.get("backtest_period_str", "1y") # Default for AI
                # AI must provide holdings. `manual_portfolio_holdings` is a list of dicts.
                # The tool definition will ask AI for [{'ticker': 'X', 'shares': Y}, {'ticker': 'Cash', 'value': Z}]
                raw_holdings = specific_params.get("manual_portfolio_holdings")
                if not raw_holdings or not isinstance(raw_holdings, list):
                    return "Error for AI (Assess B): 'manual_portfolio_holdings' (list of dicts with ticker/shares/value) is required."
                
                print(f"AI provided holdings for Assess B: {raw_holdings}")
                temp_holdings_value_calc = []
                for holding_item in raw_holdings:
                    ticker_b_add = str(holding_item.get("ticker", "")).upper()
                    shares_b_val = holding_item.get("shares")
                    value_b_val = holding_item.get("value") # Direct value, e.g. for cash or if price is unknown

                    if not ticker_b_add: continue
                    if ticker_b_add == "CASH":
                        if value_b_val is not None:
                            temp_holdings_value_calc.append({'ticker': 'Cash', 'value': float(value_b_val)})
                        else: return "Error for AI (Assess B): Cash holding needs a 'value'."
                    elif shares_b_val is not None: # Ticker with shares
                        live_price_b_val, _ = await calculate_ema_invest(ticker_b_add, 2) # calculate_ema_invest needs to be defined
                        if live_price_b_val is not None and live_price_b_val > 0:
                            holding_value_b = float(shares_b_val) * live_price_b_val
                            temp_holdings_value_calc.append({'ticker': ticker_b_add, 'value': holding_value_b, 'shares': float(shares_b_val), 'price': live_price_b_val})
                        else: return f"Error for AI (Assess B): Could not fetch live price for {ticker_b_add} to calculate value from shares."
                    elif value_b_val is not None: # Ticker with direct value (less common for stocks, but possible)
                         temp_holdings_value_calc.append({'ticker': ticker_b_add, 'value': float(value_b_val)})
                    else:
                        return f"Error for AI (Assess B): Holding {ticker_b_add} needs 'shares' or 'value'."
                portfolio_holdings_b = temp_holdings_value_calc

            elif len(args) >= 2: # CLI: /assess B <backtest_period>
                backtest_period_b = args[1].lower()
                # CLI path continues with original interactive input for holdings
                print(f"CLI Backtesting Period: {backtest_period_b}")
                # Original logic for ask_singularity_input for tickers/shares/cash
                # This part is complex to fully replicate here without the ask_singularity_input function
                # For CLI, it will use the original script's input loop.
                # For AI, portfolio_holdings_b is already populated.
                # For brevity, the CLI interactive part for B is assumed to be handled by your original logic
                # if ai_params is None. This refactor focuses on AI path.
                if not ai_params: # Fallback to original CLI interactive input logic
                    print("CLI Assess B: Interactive input for holdings needed (original script logic).")
                    # This is where your original loop with ask_singularity_input would go.
                    # For AI, this section is skipped.
                    # Example of what would need to be here for CLI:
                    # while True:
                    #     ticker_b_add = input("Enter ticker (or 'done', 'cash'):").upper()
                    #     if ticker_b_add == 'DONE': break
                    #     # ... (logic to get shares/value and calculate holding value) ...
                    #     portfolio_holdings_b.append({'ticker': ticker_b_add, 'value': calculated_value})
                    return "Assess B CLI path for manual input not fully expanded in this refactor stub."


            if backtest_period_b not in ['1y', '5y', '10y']:
                return f"Error (Assess B): Invalid backtest period '{backtest_period_b}'. Use 1y, 5y, or 10y."
            if not portfolio_holdings_b:
                return "Error (Assess B): No portfolio holdings provided or derived."

            total_value_b = sum(h['value'] for h in portfolio_holdings_b)
            if total_value_b <= 0:
                return "Error (Assess B): Total portfolio value is zero or negative."

            print(f"Total Portfolio Value for Assess B: ${total_value_b:,.2f}")
            # calculate_portfolio_beta_correlation_singularity needs to be defined
            beta_corr_results_b = await calculate_portfolio_beta_correlation_singularity(portfolio_holdings_b, total_value_b, backtest_period_b)
            
            if beta_corr_results_b:
                w_beta_b, w_corr_b = beta_corr_results_b
                result_text = f"Manual Portfolio (Value: ${total_value_b:,.2f}, Period: {backtest_period_b}): Avg Beta {w_beta_b:.4f}, Avg Correlation {w_corr_b:.4f}."
                print("\n**Manual Portfolio Assessment Results (Code B)**")
                print(f"  Weighted Average Beta vs SPY ({backtest_period_b}): {w_beta_b:.4f}")
                print(f"  Weighted Average Correlation to SPY ({backtest_period_b}): {w_corr_b:.4f}")
                summary_for_ai = result_text
            else:
                summary_for_ai = f"Could not calculate Beta/Correlation for manual portfolio (Period: {backtest_period_b}). Check holdings data."
                print(summary_for_ai)

        except ValueError as ve:
            summary_for_ai = f"Error (Assess B): Invalid numerical input or data. {ve}"
            print(summary_for_ai)
        except Exception as e_b:
            summary_for_ai = f"Error in Assess Code B: {e_b}"
            print(summary_for_ai)
            traceback.print_exc()
        return summary_for_ai

    # --- Code C: Custom Portfolio Risk Assessment ---
    elif assess_code == 'C':
        print("--- Assess Code C: Custom Portfolio Risk ---")
        custom_code_c, assess_value_c, backtest_period_c = None, None, None
        try:
            if ai_params:
                custom_code_c = specific_params.get("custom_portfolio_code")
                assess_value_c = float(specific_params.get("value_for_assessment"))
                backtest_period_c = specific_params.get("backtest_period_str", "3y") # Default for AI
                if not custom_code_c or assess_value_c <= 0:
                    return "Error for AI (Assess C): 'custom_portfolio_code' and positive 'value_for_assessment' required."
            elif len(args) >= 5: # CLI: /assess C <custom_code> <value> <backtest_period>
                custom_code_c = args[1]
                assess_value_c = float(args[2])
                backtest_period_c = args[3].lower() # Original script took 4 args for assess C: /assess C MYPORTFOLIO 25000 3y. args[0] is assess, args[1] is C
                                                    # So this should be args[1] for C, args[2] for code, args[3] for value, args[4] for period.
                                                    # Correcting based on original script structure indicated in display_commands
                custom_code_c = args[2] # args[0] is 'assess', args[1] is 'C'
                assess_value_c = float(args[3])
                backtest_period_c = args[4].lower()

            else:
                print("CLI Usage for Code C: /assess C <custom_portfolio_code> <value_for_assessment> <backtest_period 1y/3y/5y/10y>")
                return "Error: Insufficient arguments for Assess Code C." if ai_params else None

            if backtest_period_c not in ['1y', '3y', '5y', '10y']:
                return f"Error (Assess C): Invalid backtest period '{backtest_period_c}'. Use 1y, 3y, 5y, or 10y."

            # (Original logic from assess C: load custom_config_c, run process_custom_portfolio, get holdings)
            # This requires portfolio_db_file, process_custom_portfolio to be defined
            custom_config_c_data = None
            if os.path.exists(portfolio_db_file): # portfolio_db_file needs to be global
                with open(portfolio_db_file, 'r', encoding='utf-8', newline='') as file_c:
                    reader_c = csv.DictReader(file_c)
                    for row_c in reader_c:
                        if row_c.get('portfolio_code','').strip().lower() == custom_code_c.lower():
                            custom_config_c_data = row_c; break
            if not custom_config_c_data: 
                return f"Error (Assess C): Custom portfolio code '{custom_code_c}' not found."

            frac_s_c = custom_config_c_data.get('frac_shares', 'false').lower() == 'true'
            # process_custom_portfolio returns (tailored_list_final, combined_intermediate, final_cash, structured_tailored_holdings)
            _, _, final_cash_c_val, structured_holdings_c = await process_custom_portfolio(
                portfolio_data_config=custom_config_c_data, tailor_portfolio_requested=True, 
                frac_shares_singularity=frac_s_c, total_value_singularity=assess_value_c, 
                is_custom_command_simplified_output=True 
            )
            
            portfolio_holdings_for_assess_c = []
            if structured_holdings_c:
                for item_c_h in structured_holdings_c:
                    if item_c_h.get('actual_money_allocation', 0) > 1e-9 :
                        portfolio_holdings_for_assess_c.append({'ticker': item_c_h['ticker'], 'value': float(item_c_h['actual_money_allocation'])})
            if final_cash_c_val > 1e-9:
                portfolio_holdings_for_assess_c.append({'ticker': 'Cash', 'value': float(final_cash_c_val)})

            if not portfolio_holdings_for_assess_c or not any(h['ticker'] != 'Cash' for h in portfolio_holdings_for_assess_c):
                summary_for_ai = f"Assess C for '{custom_code_c}' (Value ${assess_value_c:,.0f}): No stock holdings derived. Beta/Corr will be 0 if only cash."
                print(summary_for_ai)
                if any(h['ticker'] == 'Cash' for h in portfolio_holdings_for_assess_c):
                     print("  Portfolio consists only of cash. Weighted Beta: 0.0000, Weighted Correlation: 0.0000")
                return summary_for_ai
            
            beta_corr_results_c = await calculate_portfolio_beta_correlation_singularity(portfolio_holdings_for_assess_c, assess_value_c, backtest_period_c)
            if beta_corr_results_c:
                w_beta_c, w_corr_c = beta_corr_results_c
                summary_for_ai = f"Custom Portfolio '{custom_code_c}' (Value ${assess_value_c:,.0f}, Period {backtest_period_c}): Avg Beta {w_beta_c:.4f}, Avg Corr {w_corr_c:.4f}."
                print("\n**Custom Portfolio Risk Assessment Results (Code C)**")
                print(f"  Portfolio Code: {custom_code_c}, Assessed Value: ${assess_value_c:,.2f}")
                print(f"  Weighted Average Beta vs SPY ({backtest_period_c}): {w_beta_c:.4f}")
                print(f"  Weighted Average Correlation to SPY ({backtest_period_c}): {w_corr_c:.4f}")
            else:
                summary_for_ai = f"Could not calculate Beta/Correlation for custom portfolio '{custom_code_c}' (Period {backtest_period_c})."
                print(summary_for_ai)
        
        except ValueError as ve: # For float conversion
            summary_for_ai = f"Error (Assess C): Invalid numerical input for value. {ve}"
            print(summary_for_ai)
        except Exception as e_c:
            summary_for_ai = f"Error in Assess Code C: {e_c}"
            print(summary_for_ai)
            traceback.print_exc()
        return summary_for_ai

    # --- Code D: Cultivate Portfolio Risk Assessment ---
    elif assess_code == 'D':
        print("--- Assess Code D: Cultivate Portfolio Risk ---")
        cult_code_d, epsilon_d, frac_s_d, backtest_period_d = None, None, None, None
        try:
            if ai_params:
                cult_code_d = specific_params.get("cultivate_portfolio_code", "").upper()
                epsilon_d = float(specific_params.get("value_for_assessment")) # value_epsilon from original
                frac_s_d_bool = specific_params.get("use_fractional_shares", False) # Default for AI
                frac_s_d = 'yes' if frac_s_d_bool else 'no' # Convert bool to y/n if original script used that
                backtest_period_d = specific_params.get("backtest_period_str", "5y") # Default for AI

                if cult_code_d not in ['A', 'B'] or epsilon_d <= 0:
                    return "Error for AI (Assess D): Valid 'cultivate_portfolio_code' (A/B) and positive 'value_for_assessment' required."
            elif len(args) >= 6: # CLI: /assess D <cult_code> <value> <frac_shares y/n> <backtest_period>
                                 # args[0]=assess, args[1]=D, args[2]=cult_code, args[3]=value, args[4]=frac, args[5]=period
                cult_code_d = args[2].upper()
                epsilon_d = float(args[3])
                frac_s_d = args[4].lower()
                backtest_period_d = args[5].lower()
            else:
                print("CLI Usage for Code D: /assess D <cultivate_code A/B> <value_epsilon> <frac_shares y/n> <backtest_period 1y/3y/5y/10y>")
                return "Error: Insufficient arguments for Assess Code D." if ai_params else None

            if cult_code_d not in ['A', 'B']: return f"Error (Assess D): Invalid Cultivate Code '{cult_code_d}'. Use A or B."
            if frac_s_d not in ['yes', 'no']: return f"Error (Assess D): Fractional shares input '{frac_s_d}' must be 'yes' or 'no'."
            frac_s_d_bool_for_func = frac_s_d == 'yes'
            if backtest_period_d not in ['1y', '3y', '5y', '10y']:
                 return f"Error (Assess D): Invalid backtest period '{backtest_period_d}'. Use 1y, 3y, 5y, or 10y."

            # (Original logic from assess D: run_cultivate_analysis_singularity, get holdings)
            # This requires run_cultivate_analysis_singularity to be defined
            print("  Running Cultivate analysis for Assess D to get tailored holdings (this may take some time)...")
            _, tailored_holdings_cult_d, final_cash_cult_d, _, _, _, err_msg_cult_run = await run_cultivate_analysis_singularity(
                portfolio_value=epsilon_d, frac_shares=frac_s_d_bool_for_func, 
                cultivate_code_str=cult_code_d, is_saving_run=True 
            )
            if err_msg_cult_run:
                return f"Error (Assess D) during Cultivate run: {err_msg_cult_run}"

            portfolio_holdings_for_assess_d = []
            if tailored_holdings_cult_d:
                for item_cult_h in tailored_holdings_cult_d:
                    if item_cult_h.get('actual_money_allocation', 0) > 1e-9:
                        portfolio_holdings_for_assess_d.append({'ticker': item_cult_h['ticker'], 'value': float(item_cult_h['actual_money_allocation'])})
            if final_cash_cult_d > 1e-9:
                portfolio_holdings_for_assess_d.append({'ticker': 'Cash', 'value': float(final_cash_cult_d)})

            if not portfolio_holdings_for_assess_d or not any(h['ticker'] != 'Cash' for h in portfolio_holdings_for_assess_d):
                summary_for_ai = f"Assess D for Cultivate Code '{cult_code_d}' (Epsilon ${epsilon_d:,.0f}): No stock holdings. Beta/Corr will be 0 if only cash."
                print(summary_for_ai)
                if any(h['ticker'] == 'Cash' for h in portfolio_holdings_for_assess_d):
                     print("  Portfolio consists only of cash. Weighted Beta: 0.0000, Weighted Correlation: 0.0000")
                return summary_for_ai

            beta_corr_results_d = await calculate_portfolio_beta_correlation_singularity(portfolio_holdings_for_assess_d, epsilon_d, backtest_period_d)
            if beta_corr_results_d:
                w_beta_d, w_corr_d = beta_corr_results_d
                summary_for_ai = f"Cultivate Portfolio (Code {cult_code_d}, Epsilon ${epsilon_d:,.0f}, Period {backtest_period_d}): Avg Beta {w_beta_d:.4f}, Avg Corr {w_corr_d:.4f}."
                print("\n**Cultivate Portfolio Risk Assessment Results (Code D)**")
                print(f"  Cultivate Code: {cult_code_d}, Epsilon: ${epsilon_d:,.2f}")
                print(f"  Weighted Average Beta vs SPY ({backtest_period_d}): {w_beta_d:.4f}")
                print(f"  Weighted Average Correlation to SPY ({backtest_period_d}): {w_corr_d:.4f}")
            else:
                summary_for_ai = f"Could not calculate Beta/Correlation for Cultivate portfolio (Code {cult_code_d}, Period {backtest_period_d})."
                print(summary_for_ai)
        
        except ValueError as ve: # For float conversion
            summary_for_ai = f"Error (Assess D): Invalid numerical input for value. {ve}"
            print(summary_for_ai)
        except Exception as e_d:
            summary_for_ai = f"Error in Assess Code D: {e_d}"
            print(summary_for_ai)
            traceback.print_exc()
        return summary_for_ai

    else:
        msg = f"Unknown or unsupported Assess Code: '{assess_code}'. Use A, B, C, or D."
        print(msg)
        return msg if ai_params else None
    
# --- R.I.S.K. Module Functions (Adapted for Singularity) ---

def get_sp100_symbols_risk() -> List[str]:
    """Fetches S&P 100 symbols for RISK module."""
    try:
        sp100_list_url = 'https://en.wikipedia.org/wiki/S%26P_100'
        df = pd.read_html(sp100_list_url)[2] 
        symbols = df['Symbol'].tolist()
        return [s.replace('.', '-') for s in symbols if isinstance(s, str)]
    except Exception as e:
        risk_logger.error(f"Error fetching S&P 100 symbols: {e}")
        return []

def calculate_ma_risk(symbol: str, ma_window: int) -> Optional[bool]:
    """Calculates if price is above MA for RISK module."""
    try:
        symbol_yf = symbol.replace('.', '-')
        # Determine period based on MA window to optimize data download
        if ma_window >= 200: period = '2y'
        elif ma_window >= 50: period = '1y'
        elif ma_window >= 20: period = '6mo'
        else: period = '3mo'
        
        data = yf.download(symbol_yf, period=period, interval='1d', progress=False, timeout=10) # Shorter timeout
        if data.empty or len(data) < ma_window:
            # risk_logger.debug(f"Insufficient data for {symbol_yf} MA({ma_window}). Have {len(data)} points.")
            return None
        
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        if data['Close'].isnull().all(): return None # All close prices are NaN
        data['Close'].ffill(inplace=True) # Forward fill to handle sparse NaNs
        if data['Close'].isnull().any(): return None # Still null after ffill

        ma_col_name = f'{ma_window}_day_ma'
        data[ma_col_name] = data['Close'].rolling(window=ma_window).mean()
        
        latest_close = data['Close'].dropna().iloc[-1] if not data['Close'].dropna().empty else None
        latest_ma = data[ma_col_name].dropna().iloc[-1] if not data[ma_col_name].dropna().empty else None

        if latest_close is None or latest_ma is None: return None
        return latest_close > latest_ma
    except Exception as e:
        # risk_logger.warning(f"Error processing {symbol} MA({ma_window}): {type(e).__name__}") # Less verbose
        return None

def calculate_percentage_above_ma_risk(symbols: List[str], ma_window: int) -> float:
    """Calculates percentage of symbols above MA for RISK module."""
    if not symbols: return 0.0
    above_ma_count = 0
    valid_stocks_count = 0
    for i, symbol in enumerate(symbols):
        # if (i + 1) % 50 == 0: # Optional progress update for large lists
            # print(f"    MA({ma_window}) check progress: {i+1}/{len(symbols)}")
        result = calculate_ma_risk(symbol, ma_window)
        if result is not None:
            valid_stocks_count += 1
            if result: above_ma_count += 1
    if valid_stocks_count == 0: return 0.0
    percentage = (above_ma_count / valid_stocks_count) * 100
    risk_logger.info(f"MA({ma_window}): {above_ma_count}/{valid_stocks_count} stocks above MA ({percentage:.2f}%)")
    return percentage

def calculate_s5tw_risk(): return calculate_percentage_above_ma_risk(get_sp500_symbols_singularity(), 20)
def calculate_s5th_risk(): return calculate_percentage_above_ma_risk(get_spy_symbols_singularity(), 200)
def calculate_s1fd_risk(): return calculate_percentage_above_ma_risk(get_sp100_symbols_risk(), 5)
def calculate_s1tw_risk(): return calculate_percentage_above_ma_risk(get_sp100_symbols_risk(), 20)

def get_live_price_and_ma_risk(ticker: str, ma_windows: List[int] = None) -> tuple[Optional[float], Dict[int, Optional[float]]]:
    """Fetches live price and specified MAs for RISK module."""
    if ma_windows is None: ma_windows = [20, 50] # Default MAs if none specified
    try:
        stock = yf.Ticker(ticker)
        hist_period = '2y' # Sufficient for up to 200-day MA
        if ma_windows:
            max_ma = max(ma_windows, default=0)
            if max_ma >= 200: hist_period = '2y'
            elif max_ma >= 50: hist_period = '1y'
            else: hist_period = '6mo'
        
        hist = stock.history(period=hist_period, interval="1d") # Fetch once
        if hist.empty: 
            risk_logger.warning(f"No history data for {ticker} (period {hist_period}) in get_live_price_and_ma_risk")
            return None, {ma: None for ma in ma_windows}
        
        hist['Close'] = pd.to_numeric(hist['Close'], errors='coerce')
        live_price = hist['Close'].dropna().iloc[-1] if not hist['Close'].dropna().empty else None

        ma_values = {}
        for window in ma_windows:
            if len(hist) >= window:
                ma_val = hist['Close'].rolling(window=window).mean().iloc[-1]
                ma_values[window] = ma_val if not pd.isna(ma_val) else None
            else: 
                ma_values[window] = None
        return live_price, ma_values
    except Exception as e:
        risk_logger.error(f"Error in get_live_price_and_ma_risk for {ticker}: {e}")
        return None, {ma: None for ma in ma_windows}

def calculate_ema_score_risk(ticker:str ="SPY") -> Optional[float]:
    """Calculates specific EMA-based score for RISK module."""
    try:
        data = yf.Ticker(ticker).history(period="1y", interval="1d") # Fetch 1 year of daily data
        if data.empty or len(data) < 55: # Check if enough data for EMA 55
            # risk_logger.warning(f"Insufficient data for {ticker} EMA score (need >55 days, have {len(data)})")
            return None
        
        data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
        if data['Close'].isnull().all(): return None
        data['Close'].ffill(inplace=True)
        if data['Close'].isnull().any(): return None


        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()
        
        ema_8_val = data['EMA_8'].iloc[-1]
        ema_55_val = data['EMA_55'].iloc[-1]

        if pd.isna(ema_8_val) or pd.isna(ema_55_val) or ema_55_val == 0: 
            # risk_logger.warning(f"NaN or zero EMA_55 for {ticker} in EMA score. EMA_8: {ema_8_val}, EMA_55: {ema_55_val}")
            return None
        
        # Original R.I.S.K. formula for its EMA Score
        ema_invest_specific = (((ema_8_val - ema_55_val) / ema_55_val * 5) + 0.5) * 100
        return float(np.clip(ema_invest_specific, 0, 100))
    except Exception as e:
        risk_logger.error(f"Error calculating EMA score for {ticker} in RISK module: {e}")
        return None

def calculate_risk_scores_singularity() -> tuple:
    """Calculates all RISK scores for Singularity."""
    risk_logger.info("Starting RISK score component calculation.")
    spy_live_price, spy_mas = get_live_price_and_ma_risk('SPY', [20, 50])
    vix_live_price, _ = get_live_price_and_ma_risk('^VIX', []) 
    rut_live_price, rut_mas = get_live_price_and_ma_risk('^RUT', [20, 50])
    oex_live_price, oex_mas = get_live_price_and_ma_risk('^OEX', [20, 50])
    
    s5tw_val = calculate_s5tw_risk() 
    s5th_val = calculate_s5th_risk()
    s1fd_val = calculate_s1fd_risk()
    s1tw_val = calculate_s1tw_risk()

    critical_data_map = {
        'SPY Price': spy_live_price, 'SPY MA20': spy_mas.get(20), 'SPY MA50': spy_mas.get(50),
        'VIX Price': vix_live_price, 
        'RUT Price': rut_live_price, 'RUT MA20': rut_mas.get(20), 'RUT MA50': rut_mas.get(50),
        'OEX Price': oex_live_price, 'OEX MA20': oex_mas.get(20), 'OEX MA50': oex_mas.get(50),
        'S5TW': s5tw_val, 'S5TH': s5th_val, 'S1FD': s1fd_val, 'S1TW': s1tw_val
    }
    
    all_critical_valid = True
    for name, value in critical_data_map.items():
        if value is None or (isinstance(value, float) and pd.isna(value)):
            risk_logger.error(f"RISK score calculation: Missing critical data for '{name}'. Value: {value}")
            all_critical_valid = False
            # Set to a neutral default (e.g., 0 or a typical value) if a component is missing,
            # or decide to abort calculation. For now, let's use 0 for missing percentages
            # and typical values for prices/MAs if absolutely necessary, though None is better.
            # The original script formula implies all these must be present.
    
    if not all_critical_valid:
        risk_logger.error(f"Cannot calculate full RISK scores due to missing critical data components.")
        return None, None, None, None, spy_live_price, vix_live_price # Return what we have

    try:
        # Ensure all components are numbers before calculations
        # Using .get(window, some_neutral_default_if_none_is_fatal) if MA might be None
        spy20 = np.clip(((spy_live_price - spy_mas[20]) / 20) + 50, 0, 100) if spy_mas.get(20) is not None else 50.0
        spy50 = np.clip(((spy_live_price - spy_mas[50] - 150) / 20) + 50, 0, 100) if spy_mas.get(50) is not None else 50.0
        vix_score = np.clip((((vix_live_price - 15) * -5) + 50), 0, 100)
        rut20 = np.clip(((rut_live_price - rut_mas[20]) / 10) + 50, 0, 100) if rut_mas.get(20) is not None else 50.0
        rut50 = np.clip(((rut_live_price - rut_mas[50]) / 5) + 50, 0, 100) if rut_mas.get(50) is not None else 50.0
        s5tw_score = np.clip(((s5tw_val - 60) + 50), 0, 100)
        s5th_score = np.clip(((s5th_val - 70) + 50), 0, 100)
        oex20_score = np.clip(((oex_live_price - oex_mas[20]) / 100) + 50, 0, 100) if oex_mas.get(20) is not None else 50.0 # Original formula had /100, seems like a typo vs other index MAs, check R.I.S.K. intent
        oex50_score = np.clip(((oex_live_price - oex_mas[50] - 25) / 100) + 50, 0, 100) if oex_mas.get(50) is not None else 50.0 # Same here, /100
        s1fd_score = np.clip(((s1fd_val - 60) + 50), 0, 100)
        s1tw_score = np.clip(((s1tw_val - 70) + 50), 0, 100)
    except TypeError as te:
        risk_logger.error(f"TypeError during RISK score component calculation. One of the inputs might be None or non-numeric: {te}")
        return None, None, None, None, spy_live_price, vix_live_price

    ema_score_val_risk = calculate_ema_score_risk("SPY") # Use R.I.S.K. specific EMA score
    if ema_score_val_risk is None:
        risk_logger.error("RISK EMA score calculation failed. Cannot complete full RISK scores.")
        return None, None, None, None, spy_live_price, vix_live_price

    general_score = np.clip(((3*spy20)+spy50+(3*vix_score)+(3*rut50)+rut20+(2*s5tw_score)+s5th_score)/13.0, 0, 100)
    large_cap_score = np.clip(((3*oex20_score)+oex50_score+(2*s1fd_score)+s1tw_score)/7.0, 0, 100)
    combined_score = np.clip((general_score + large_cap_score + ema_score_val_risk) / 3.0, 0, 100)
    risk_logger.info(f"RISK Scores Calculated: General={general_score:.2f}, LargeCap={large_cap_score:.2f}, EMA(RISK)={ema_score_val_risk:.2f}, Combined={combined_score:.2f}")
    return general_score, large_cap_score, ema_score_val_risk, combined_score, spy_live_price, vix_live_price

def calculate_recession_likelihood_ema_risk(ticker:str ="SPY", interval:str ="1mo", period:str ="5y") -> Optional[float]:
    """Calculates Momentum Based Recession Likelihood for RISK module."""
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        if data.empty or len(data) < 55:
            return None
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()
        ema_8, ema_55 = data['EMA_8'].iloc[-1], data['EMA_55'].iloc[-1]
        if pd.isna(ema_8) or pd.isna(ema_55) or ema_55 == 0: return None
        x = (((ema_8 - ema_55) / ema_55) + 0.5) * 100 
        return float(np.clip(100 * np.exp(-((45.622216 * x / 2750) ** 4)), 0, 100))
    except Exception as e:
        risk_logger.error(f"Error calculating momentum (EMA) recession likelihood for {ticker}: {e}")
        return None

def calculate_recession_likelihood_vix_risk(vix_price: Optional[float]) -> Optional[float]:
    """Calculates VIX Based Recession Likelihood for RISK module."""
    if vix_price is not None and not pd.isna(vix_price):
        try: return float(np.clip(0.01384083 * (float(vix_price) ** 2), 0, 100))
        except ValueError: risk_logger.error(f"Could not convert VIX price '{vix_price}' to float."); return None
    return None

def calculate_market_invest_score_risk(vix_contraction_chance: Optional[float], ema_contraction_chance: Optional[float]) -> tuple:
    """Calculates Market Invest Score for RISK module."""
    if vix_contraction_chance is None or ema_contraction_chance is None:
        return None, None, None

    uncapped_score_mis = None
    if ema_contraction_chance == 0: # Avoid division by zero
        # If EMA chance is 0 (strong market), VIX chance leads to high ratio, should result in low MIS (e.g., 0)
        # Or, if VIX is also 0, it's undefined. Let's treat 0/0 as neutral (50 MIS) or a specific value.
        # Original formula would lead to issues. A robust interpretation:
        uncapped_score_mis = 0.0 if vix_contraction_chance > 0 else 50.0 # Highly bullish if both 0, or very risky if VIX>0 & EMA=0
    else:
        try:
            ratio = vix_contraction_chance / ema_contraction_chance
            # Original formula: 100.0 - (((ratio - 1.0) * 100.0) + 50.0) = 50.0 - (ratio - 1.0) * 100.0
            uncapped_score_mis = 50.0 - (ratio - 1.0) * 100.0
        except Exception as e:
            risk_logger.error(f"Error calculating Market Invest Score ratio: {e}")
            return None, None, None
    
    if uncapped_score_mis is None: return None, None, None

    capped_score_for_signal_mis = float(np.clip(uncapped_score_mis, 0, 100))
    rounded_capped_score_for_display_mis = int(round(capped_score_for_signal_mis))
    return uncapped_score_mis, capped_score_for_signal_mis, rounded_capped_score_for_display_mis


def calculate_market_ivr_risk(new_raw_score: Optional[float], csv_file_path: str = RISK_CSV_FILE) -> Optional[float]:
    """Calculates Market IVR for RISK module."""
    if new_raw_score is None: return None
    historical_raw_scores = []
    if os.path.exists(csv_file_path):
        try:
            df = pd.read_csv(csv_file_path, on_bad_lines='skip') 
            if 'Raw Market Invest Score' in df.columns:
                historical_raw_scores = pd.to_numeric(df['Raw Market Invest Score'], errors='coerce').dropna().tolist()
        except Exception as e:
            risk_logger.error(f"Error reading historical raw scores for IVR from {csv_file_path}: {e}")
            return None 
    
    if not historical_raw_scores: return 0.0 # IVR is 0 if no history or current is lowest
    lower_count = sum(1 for score in historical_raw_scores if new_raw_score < score) 
    market_ivr = (lower_count / len(historical_raw_scores)) * 100
    return float(market_ivr)

def calculate_market_iv_risk(eod_csv_file_path: str = RISK_EOD_CSV_FILE) -> Optional[float]:
    """Calculates Market IV for RISK module."""
    if not os.path.exists(eod_csv_file_path): return None
    try:
        df_eod = pd.read_csv(eod_csv_file_path, on_bad_lines='skip') 
        if df_eod.empty or 'Raw Market Invest Score (EOD)' not in df_eod.columns: return None
        df_eod['Date'] = pd.to_datetime(df_eod['Date'], errors='coerce')
        df_eod = df_eod.sort_values(by='Date', ascending=True).dropna(subset=['Date'])
        eod_scores = pd.to_numeric(df_eod['Raw Market Invest Score (EOD)'], errors='coerce').dropna()
        if len(eod_scores) < 21: return None # Need at least ~1 month of EOD scores for meaningful IV
        
        # Consider using a rolling window for changes, e.g., last 20 days for a monthly IV like measure
        changes = np.abs(eod_scores.rolling(window=2).apply(lambda x: x.iloc[1] - x.iloc[0] if len(x)==2 else np.nan, raw=True).dropna())
        if len(changes) < 20 : return None # Need enough changes

        average_market_invest_magnitude_score = changes.tail(20).mean() # Use recent changes
        if pd.isna(average_market_invest_magnitude_score): return None

        expressed_percentage_val = average_market_invest_magnitude_score / 10000.0 
        multiplier = 1 + expressed_percentage_val
        powered_value = multiplier ** 252 
        market_iv = (powered_value - 1) * 100 
        return float(market_iv)
    except Exception as e:
        risk_logger.exception(f"Error calculating Market IV from {eod_csv_file_path}:") 
        return None

async def perform_risk_calculations_singularity(is_eod_save: bool = False):
    """
    Performs one-time R.I.S.K. calculations, prints results, saves data to CSV files,
    and returns a summary dictionary.
    """
    global risk_persistent_signal, risk_signal_day # As in your original script

    risk_logger.info(f"--- Singularity: Performing R.I.S.K. calculations cycle (EOD Save: {is_eod_save}) ---")
    
    # Initialize results dictionary with defaults
    results_summary = {
        "general_score": 'N/A', "large_cap_score": 'N/A', "ema_risk_score": 'N/A',
        "combined_score": 'N/A', "spy_price": 'N/A', "vix_price": 'N/A',
        "momentum_recession_chance": 'N/A', "vix_recession_chance": 'N/A',
        "raw_market_invest_score": 'N/A', "market_invest_score": 'N/A',
        "market_ivr": 'N/A', "market_iv_eod": 'N/A',
        "market_signal": "Hold", "signal_date_info": ""
    }

    # --- 1. Calculate all scores and metrics ---
    general, large, ema_risk_score, combined, spy_price, vix_price = calculate_risk_scores_singularity()
    likelihood_ema, likelihood_vix = None, None
    uncapped_mis, capped_mis_for_signal, rounded_mis_for_display = None, None, None 
    market_ivr_val = None
    market_iv_val = None 

    if vix_price is not None:
        likelihood_vix = calculate_recession_likelihood_vix_risk(vix_price)
    likelihood_ema = calculate_recession_likelihood_ema_risk() 
    
    if likelihood_vix is not None and likelihood_ema is not None:
        uncapped_mis, capped_mis_for_signal, rounded_mis_for_display = calculate_market_invest_score_risk(likelihood_vix, likelihood_ema)
    else:
        risk_logger.warning("One or both recession likelihoods are None. Market Invest Score cannot be calculated.")

    if uncapped_mis is not None:
        market_ivr_val = calculate_market_ivr_risk(uncapped_mis, RISK_CSV_FILE)
    
    market_iv_val = calculate_market_iv_risk(RISK_EOD_CSV_FILE) 

    # Determine Market Signal (using global risk_persistent_signal, risk_signal_day)
    current_day_obj = dt_datetime.now(est_timezone).date()
    previous_capped_mis_for_signal_from_csv = None
    if os.path.exists(RISK_CSV_FILE):
        try:
            df_hist = pd.read_csv(RISK_CSV_FILE, on_bad_lines='skip')
            if not df_hist.empty and 'Market Invest Score' in df_hist.columns:
                valid_prev_scores = pd.to_numeric(df_hist['Market Invest Score'], errors='coerce').dropna()
                if not valid_prev_scores.empty:
                    previous_capped_mis_for_signal_from_csv = valid_prev_scores.iloc[-1]
        except Exception as e:
            risk_logger.error(f"Error reading previous MIS from {RISK_CSV_FILE} for signal logic: {e}")
    
    if capped_mis_for_signal is not None:
        if previous_capped_mis_for_signal_from_csv is not None:
            if previous_capped_mis_for_signal_from_csv < 50 and capped_mis_for_signal >= 50:
                risk_persistent_signal = "Buy"
                risk_signal_day = current_day_obj
            elif previous_capped_mis_for_signal_from_csv >= 50 and capped_mis_for_signal < 50:
                risk_persistent_signal = "Sell"
                risk_signal_day = current_day_obj
            elif risk_signal_day is None: # If no prior signal day, initialize based on current MIS
                 risk_persistent_signal = "Buy" if capped_mis_for_signal >= 50 else "Sell"
                 risk_signal_day = current_day_obj
            # else: signal and day persist from previous state if no threshold crossed
        else: # No previous score history, set signal based on current
            risk_persistent_signal = "Buy" if capped_mis_for_signal >= 50 else "Sell"
            risk_signal_day = current_day_obj
    # If capped_mis_for_signal is None, risk_persistent_signal retains its last value or default "Hold"

    signal_day_str_local = f" (Since {risk_signal_day.strftime('%Y-%m-%d')})" if risk_signal_day and risk_persistent_signal != "Hold" else ""

    # --- 2. Populate results_summary dictionary for return value ---
    if general is not None: results_summary["general_score"] = f"{general:.2f}"
    if large is not None: results_summary["large_cap_score"] = f"{large:.2f}"
    if ema_risk_score is not None: results_summary["ema_risk_score"] = f"{ema_risk_score:.2f}"
    if combined is not None: results_summary["combined_score"] = f"{combined:.2f}"
    if spy_price is not None: results_summary["spy_price"] = f"{spy_price:.2f}"
    if vix_price is not None: results_summary["vix_price"] = f"{vix_price:.2f}"
    if likelihood_ema is not None: results_summary["momentum_recession_chance"] = f"{likelihood_ema:.1f}%"
    if likelihood_vix is not None: results_summary["vix_recession_chance"] = f"{likelihood_vix:.1f}%"
    if uncapped_mis is not None: results_summary["raw_market_invest_score"] = f"{uncapped_mis:.2f}"
    if capped_mis_for_signal is not None: results_summary["market_invest_score"] = f"{capped_mis_for_signal:.2f}"
    if market_ivr_val is not None: results_summary["market_ivr"] = f"{market_ivr_val:.1f}%"
    if market_iv_val is not None: results_summary["market_iv_eod"] = f"{market_iv_val:.2f}%" # Added % for consistency
    results_summary["market_signal"] = risk_persistent_signal
    results_summary["signal_date_info"] = signal_day_str_local.strip()

    # --- 3. Save data to RISK_CSV_FILE ---
    timestamp_iso = dt_datetime.now(pytz.utc).isoformat() 
    csv_fieldnames = ['Timestamp', 'General Market Score', 'Large Market Cap Score', 'EMA Score', 'Combined Score',
                      'Live SPY Price', 'Live VIX Price', 'Momentum Based Recession Chance', 'VIX Based Recession Chance',
                      'Raw Market Invest Score', 'Market Invest Score', 'Market IVR', 'Market Signal', 'Signal Date']
    try:
        file_exists = os.path.exists(RISK_CSV_FILE)
        with open(RISK_CSV_FILE, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
            if not file_exists or os.path.getsize(RISK_CSV_FILE) == 0:
                writer.writeheader()
            
            row_to_write = {
                'Timestamp': timestamp_iso,
                'General Market Score': results_summary["general_score"],
                'Large Market Cap Score': results_summary["large_cap_score"],
                'EMA Score': results_summary["ema_risk_score"], # This is R.I.S.K specific EMA score
                'Combined Score': results_summary["combined_score"],
                'Live SPY Price': results_summary["spy_price"],
                'Live VIX Price': results_summary["vix_price"],
                'Momentum Based Recession Chance': results_summary["momentum_recession_chance"],
                'VIX Based Recession Chance': results_summary["vix_recession_chance"],
                'Raw Market Invest Score': results_summary["raw_market_invest_score"],
                'Market Invest Score': results_summary["market_invest_score"], # Capped score
                'Market IVR': results_summary["market_ivr"],
                'Market Signal': risk_persistent_signal,
                'Signal Date': risk_signal_day.strftime('%Y-%m-%d') if risk_signal_day else 'N/A'
            }
            writer.writerow(row_to_write)
        risk_logger.info(f"Data appended to main R.I.S.K. data file: {RISK_CSV_FILE}")
    except Exception as e_csv:
        risk_logger.exception(f"Error writing to main R.I.S.K. CSV file {RISK_CSV_FILE}:")

    # --- 4. Save data to RISK_EOD_CSV_FILE (if is_eod_save) ---
    if is_eod_save:
        # Ensure uncapped_mis and market_iv_val are available for EOD save
        if uncapped_mis is not None and market_iv_val is not None:
            eod_fieldnames = ['Date', 'Raw Market Invest Score (EOD)', 'Market IV (EOD)']
            try:
                eod_file_exists = os.path.exists(RISK_EOD_CSV_FILE)
                with open(RISK_EOD_CSV_FILE, 'a', newline='', encoding='utf-8') as eod_csvfile:
                    eod_writer = csv.DictWriter(eod_csvfile, fieldnames=eod_fieldnames)
                    if not eod_file_exists or os.path.getsize(RISK_EOD_CSV_FILE) == 0:
                        eod_writer.writeheader()
                    
                    eod_row = {
                        'Date': dt_datetime.now(est_timezone).strftime('%Y-%m-%d'), 
                        'Raw Market Invest Score (EOD)': f"{uncapped_mis:.2f}", # results_summary["raw_market_invest_score"]
                        'Market IV (EOD)': f"{market_iv_val:.2f}" # Using f-string for consistency if results_summary["market_iv_eod"] has '%'
                    }
                    eod_writer.writerow(eod_row)
                risk_logger.info(f"EOD data appended to EOD R.I.S.K. data file: {RISK_EOD_CSV_FILE}")
            except Exception as e_eod_csv:
                risk_logger.exception(f"Error writing to EOD R.I.S.K. CSV file {RISK_EOD_CSV_FILE}:")
        else:
            risk_logger.warning("Cannot save EOD data because Raw Market Invest Score or Market IV is missing.")

    # --- 5. Print analysis results to console ---
    print("\n--- R.I.S.K. Analysis Results (Console Output) ---") # Differentiate from AI summary if needed
    print(f"  General Market Score: {round(general) if general is not None else 'N/A'}")
    print(f"  Large Market Cap Score: {round(large) if large is not None else 'N/A'}")
    print(f"  EMA Score (R.I.S.K. Specific): {round(ema_risk_score) if ema_risk_score is not None else 'N/A'}")
    print(f"  Combined Score: {round(combined) if combined is not None else 'N/A'}")
    print("-" * 20 + " Contraction & Volatility " + "-" * 20)
    print(f"  Momentum Based Recession Chance (EMA): {likelihood_ema:.1f}%" if likelihood_ema is not None else "N/A")
    print(f"  VIX Based Recession Chance: {likelihood_vix:.1f}%" if likelihood_vix is not None else "N/A")
    # Use rounded_mis_for_display for the console print as per original script, but results_summary has the .2f version
    print(f"  Market Invest Score (for display): {rounded_mis_for_display if rounded_mis_for_display is not None else 'N/A'}")
    print(f"  Market IVR: {market_ivr_val:.1f}%" if market_ivr_val is not None else "N/A")
    print(f"  Market IV (based on EOD data): {results_summary['market_iv_eod']}") # Use value from summary for consistency
    print(f"  Market Signal: {risk_persistent_signal}{signal_day_str_local}")
    print("--- End of R.I.S.K. Analysis Console Output ---")

    # --- 6. Return the summary dictionary ---
    return results_summary

async def generate_risk_graphs_singularity():
    """Generates and saves historical RISK graphs for Singularity.
    Returns a list of generated graph filenames."""
    print("\n--- Generating R.I.S.K. Historical Graphs ---")
    graph_files_generated = [] # To store filenames

    if not os.path.exists(RISK_CSV_FILE): # RISK_CSV_FILE needs to be global
        message = f"Error: Main data file '{RISK_CSV_FILE}' not found. Cannot generate graphs."
        print(message)
        return {"status": "error", "message": message, "files": graph_files_generated}
    try:
        df = pd.read_csv(RISK_CSV_FILE, on_bad_lines='skip') # pd needs to be imported
    except Exception as e:
        message = f"Error reading '{RISK_CSV_FILE}': {e}"
        print(message)
        return {"status": "error", "message": message, "files": graph_files_generated}

    if df.empty:
        message = f"'{RISK_CSV_FILE}' is empty. No data to graph."
        print(message)
        return {"status": "nodata", "message": message, "files": graph_files_generated}

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.tz_localize(None)
    df = df.sort_values(by='Timestamp').dropna(subset=['Timestamp'])

    plt.style.use('dark_background') # matplotlib.pyplot as plt
    key_color = 'white'; grid_color = 'gray'; fig_size = (12, 7)
    
    # Graph 1: Scores
    try:
        plt.figure(figsize=fig_size)
        score_cols = ['General Market Score', 'Large Market Cap Score', 'EMA Score', 'Combined Score', 'Market Invest Score']
        for col in score_cols:
            if col in df.columns:
                plt.plot(df['Timestamp'], pd.to_numeric(df[col], errors='coerce'), label=col.replace(' Score',''), linewidth=1.5)
        plt.title('Historical Market Scores (RISK)', color=key_color); plt.xlabel('Timestamp', color=key_color); plt.ylabel('Score (0-100)', color=key_color)
        plt.legend(labelcolor=key_color); plt.grid(True, color=grid_color, linestyle=':'); plt.tick_params(axis='x', colors=key_color, rotation=20); plt.tick_params(axis='y', colors=key_color)
        plt.tight_layout(); filename1 = f"risk_market_scores_{uuid.uuid4().hex[:6]}.png"; plt.savefig(filename1, facecolor='black'); graph_files_generated.append(filename1); plt.close() # uuid needs to be imported
    except Exception as e_g1: print(f"Error generating scores graph: {e_g1}")

    # Graph 2: SPY & VIX Prices
    try:
        fig, ax1 = plt.subplots(figsize=fig_size)
        if 'Live SPY Price' in df.columns: ax1.plot(df['Timestamp'], pd.to_numeric(df['Live SPY Price'], errors='coerce'), label='SPY Price', color='lime')
        ax1.set_ylabel('SPY Price ($)', color='lime'); ax1.tick_params(axis='y', labelcolor='lime'); ax1.tick_params(axis='x', colors=key_color, rotation=20)
        ax2 = ax1.twinx()
        if 'Live VIX Price' in df.columns: ax2.plot(df['Timestamp'], pd.to_numeric(df['Live VIX Price'], errors='coerce'), label='VIX Price', color='red')
        ax2.set_ylabel('VIX Price', color='red'); ax2.tick_params(axis='y', labelcolor='red')
        plt.title('Historical SPY & VIX Prices (RISK)', color=key_color)
        lines, labels = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels()
        if lines or lines2: ax2.legend(lines + lines2, labels + labels2, loc='upper right', labelcolor=key_color)
        ax1.grid(True, color=grid_color, linestyle=':'); plt.tight_layout(); filename2 = f"risk_spy_vix_prices_{uuid.uuid4().hex[:6]}.png"; plt.savefig(filename2, facecolor='black'); graph_files_generated.append(filename2); plt.close()
    except Exception as e_g2: print(f"Error generating SPY/VIX graph: {e_g2}")
    
    # Graph 3: Recession Chances
    try:
        plt.figure(figsize=fig_size)
        recession_cols = ['Momentum Based Recession Chance', 'VIX Based Recession Chance']
        for col in recession_cols:
            if col in df.columns: plt.plot(df['Timestamp'], pd.to_numeric(df[col].astype(str).str.rstrip('%'), errors='coerce'), label=col.replace(' Chance',''), linewidth=1.5)
        plt.title('Historical Recession Chances (RISK)', color=key_color); plt.xlabel('Timestamp', color=key_color); plt.ylabel('Recession Chance (%)', color=key_color)
        plt.legend(labelcolor=key_color); plt.grid(True, color=grid_color, linestyle=':'); plt.tick_params(axis='x', colors=key_color, rotation=20); plt.tick_params(axis='y', colors=key_color)
        plt.ylim(0, 105); plt.tight_layout(); filename3 = f"risk_recession_chances_{uuid.uuid4().hex[:6]}.png"; plt.savefig(filename3, facecolor='black'); graph_files_generated.append(filename3); plt.close()
    except Exception as e_g3: print(f"Error generating recession chances graph: {e_g3}")

    # Graph 4: Market IVR (from main CSV)
    try:
        if 'Market IVR' in df.columns:
            plt.figure(figsize=fig_size)
            plt.plot(df['Timestamp'], pd.to_numeric(df['Market IVR'].astype(str).str.rstrip('%'), errors='coerce'), label='Market IVR', color='aqua', linewidth=2)
            plt.title('Historical Market IVR (RISK)', color=key_color); plt.xlabel('Timestamp', color=key_color); plt.ylabel('Market IVR (%)', color=key_color)
            plt.legend(labelcolor=key_color); plt.grid(True, color=grid_color, linestyle=':'); plt.tick_params(axis='x', colors=key_color, rotation=20); plt.tick_params(axis='y', colors=key_color)
            plt.tight_layout(); filename4 = f"risk_market_ivr_{uuid.uuid4().hex[:6]}.png"; plt.savefig(filename4, facecolor='black'); graph_files_generated.append(filename4); plt.close()
    except Exception as e_g4: print(f"Error generating IVR graph: {e_g4}")

    # Graph 5: Market IV (from EOD CSV)
    df_eod_graph = None # RISK_EOD_CSV_FILE needs to be global
    if os.path.exists(RISK_EOD_CSV_FILE):
        try:
            df_eod_graph = pd.read_csv(RISK_EOD_CSV_FILE, on_bad_lines='skip')
            if not df_eod_graph.empty and 'Market IV (EOD)' in df_eod_graph.columns and 'Date' in df_eod_graph.columns:
                df_eod_graph['Date'] = pd.to_datetime(df_eod_graph['Date'], errors='coerce')
                df_eod_graph = df_eod_graph.sort_values(by='Date').dropna(subset=['Date'])
                plt.figure(figsize=fig_size)
                plt.plot(df_eod_graph['Date'], pd.to_numeric(df_eod_graph['Market IV (EOD)'], errors='coerce'), label='Market IV (EOD)', color='fuchsia', linewidth=2)
                plt.title('Historical Market IV (EOD - RISK)', color=key_color); plt.xlabel('Date', color=key_color); plt.ylabel('Market IV', color=key_color)
                plt.legend(labelcolor=key_color); plt.grid(True, color=grid_color, linestyle=':'); plt.tick_params(axis='x', colors=key_color, rotation=20); plt.tick_params(axis='y', colors=key_color)
                plt.tight_layout(); filename5 = f"risk_market_iv_eod_{uuid.uuid4().hex[:6]}.png"; plt.savefig(filename5, facecolor='black'); graph_files_generated.append(filename5); plt.close()
        except Exception as e_g5: print(f"Error generating EOD Market IV graph: {e_g5}")
    
    # Graph 6: Combined IVR & IV
    try:
        if 'Market IVR' in df.columns and df_eod_graph is not None and 'Market IV (EOD)' in df_eod_graph.columns :
            df_main_ivr = df[['Timestamp', 'Market IVR']].copy()
            df_main_ivr['Date'] = pd.to_datetime(df_main_ivr['Timestamp'], errors='coerce').dt.normalize()
            df_ivr_daily = df_main_ivr.groupby('Date')['Market IVR'].last().reset_index() 
            df_ivr_daily['Market IVR (Daily)'] = pd.to_numeric(df_ivr_daily['Market IVR'].astype(str).str.rstrip('%'), errors='coerce')

            df_eod_iv = df_eod_graph[['Date', 'Market IV (EOD)']].copy() 
            df_eod_iv['Market IV (EOD)'] = pd.to_numeric(df_eod_iv['Market IV (EOD)'], errors='coerce')

            df_combined_plot = pd.merge(df_ivr_daily, df_eod_iv, on='Date', how='outer').sort_values(by='Date')
            
            if not df_combined_plot.empty:
                fig, ax_c1 = plt.subplots(figsize=fig_size)
                ax_c1.plot(df_combined_plot['Date'], df_combined_plot['Market IVR (Daily)'], label='Market IVR (Daily %)', color='cyan', linestyle='-', marker='.')
                ax_c1.set_xlabel('Date', color=key_color); ax_c1.set_ylabel('Market IVR (Daily %)', color='cyan'); ax_c1.tick_params(axis='y', labelcolor='cyan'); ax_c1.tick_params(axis='x', colors=key_color, rotation=20)
                ax_c2 = ax_c1.twinx()
                ax_c2.plot(df_combined_plot['Date'], df_combined_plot['Market IV (EOD)'], label='Market IV (EOD)', color='fuchsia', linestyle='--', marker='x')
                ax_c2.set_ylabel('Market IV (EOD)', color='fuchsia'); ax_c2.tick_params(axis='y', labelcolor='fuchsia')
                plt.title('Combined Market IVR & Market IV (RISK)', color=key_color)
                lines_c1, labels_c1 = ax_c1.get_legend_handles_labels(); lines_c2, labels_c2 = ax_c2.get_legend_handles_labels()
                if lines_c1 or lines_c2: ax_c2.legend(lines_c1 + lines_c2, labels_c1 + labels_c2, loc='upper right', labelcolor=key_color)
                ax_c1.grid(True, color=grid_color, linestyle=':'); plt.tight_layout(); filename6 = f"risk_combined_ivr_iv_{uuid.uuid4().hex[:6]}.png"; plt.savefig(filename6, facecolor='black'); graph_files_generated.append(filename6); plt.close()
    except Exception as e_g6: print(f"Error generating combined IVR/IV graph: {e_g6}")

    if graph_files_generated:
        print("\nGenerated graph files (for user console):")
        for fname in graph_files_generated: print(f"  - {fname}")
        return {"status": "success", "message": f"Successfully generated {len(graph_files_generated)} R.I.S.K. history graphs.", "files": graph_files_generated}
    else:
        message = "No R.I.S.K. history graphs were generated, possibly due to missing data or errors during plotting."
        print(message)
        return {"status": "no_graphs", "message": message, "files": graph_files_generated}

async def handle_risk_command(args: List[str]):
    """
    Handles the /risk command for both CLI and AI.
    AI call expects args=["standard"] or args=["eod"].
    Returns a summary string if called by AI.
    """
    print("\n--- /risk Command ---")
    
    is_eod_run = False
    called_by_ai = False
    ai_arg_processed = False

    if len(args) > 0:
        # Check for AI specific invocation first
        if args[0].lower() == "eod_ai": # AI specific argument
            is_eod_run = True
            called_by_ai = True
            ai_arg_processed = True
            print("AI: Performing End-of-Day RISK calculation and save.")
        elif args[0].lower() == "standard_ai": # AI specific argument
            is_eod_run = False
            called_by_ai = True
            ai_arg_processed = True
            print("AI: Performing standard RISK calculation and save.")
        elif args[0].lower() == 'eod' and not ai_arg_processed: # CLI specific 'eod'
            is_eod_run = True
            print("CLI: Performing End-of-Day RISK calculation and save.")
        # If args[0] is something else, or no args, it's standard CLI run
            
    if not called_by_ai and not ai_arg_processed and not (len(args) > 0 and args[0].lower() == 'eod'):
        print("CLI: Performing standard RISK calculation and save.")
        # is_eod_run remains False

    # Call the core logic function, which now returns a dictionary
    # perform_risk_calculations_singularity needs to be defined and async
    risk_results_dict = await perform_risk_calculations_singularity(is_eod_save=is_eod_run)

    if called_by_ai:
        if risk_results_dict:
            # Format a summary string from the dictionary for the AI
            summary_parts = [f"R.I.S.K. Analysis Complete (EOD Run: {is_eod_run})."]
            summary_parts.append(f"Market Signal: {risk_results_dict.get('market_signal', 'N/A')} {risk_results_dict.get('signal_date_info', '').strip()}.")
            summary_parts.append(f"Market Invest Score: {risk_results_dict.get('market_invest_score', 'N/A')}.")
            summary_parts.append(f"Combined Score: {risk_results_dict.get('combined_score', 'N/A')}.")
            summary_parts.append(f"Market IVR: {risk_results_dict.get('market_ivr', 'N/A')}.")
            # Add other key scores if desired for AI summary
            return " ".join(summary_parts)
        else:
            return "Error: R.I.S.K. analysis performed but no summary data was returned."
    
    return None # For CLI, output is handled by perform_risk_calculations_singularity

async def handle_history_command(args: List[str]): # args not currently used but kept for consistency
    """
    Handles the /history command for Singularity to generate R.I.S.K. graphs.
    Returns a summary string for AI.
    """
    # generate_risk_graphs_singularity is now async and returns a dict
    result_dict = await generate_risk_graphs_singularity() 
    
    if result_dict and result_dict.get("status") == "success":
        files_generated = result_dict.get("files", [])
        if files_generated:
            return f"Successfully generated {len(files_generated)} R.I.S.K. history graphs: {', '.join(files_generated)}."
        else:
            return "R.I.S.K. history graph generation reported success but no filenames were returned."
    elif result_dict:
        return f"R.I.S.K. history graph generation failed or produced no graphs. Status: {result_dict.get('status')}, Message: {result_dict.get('message')}"
    else:
        return "An unexpected issue occurred during R.I.S.K. history graph generation."

async def handle_quickscore_command(args: List[str]):
    """
    Handles the /quickscore command, prints results, and
    returns a summary string of the analysis.
    Args:
        args (List[str]): A list containing the ticker symbol as the first element.
    """
    print("\n--- /quickscore Command (for AI & User) ---")
    if not args:
        print("Usage: /quickscore <ticker>")
        return "Error: No ticker provided for quickscore."

    ticker_qs = args[0].upper()
    print(f"Processing /quickscore for {ticker_qs}...")

    scores_qs = {}
    graphs_qs_files = [] # To store filenames of graphs
    live_price_qs_display = "N/A"

    sensitivity_map_qs = {
        1: {'name': 'Weekly (5Y)', 'ema_sens': 1},
        2: {'name': 'Daily (1Y)', 'ema_sens': 2},
        3: {'name': 'Hourly (6M)', 'ema_sens': 3},
    }

    for sens_key, sens_info in sensitivity_map_qs.items():
        ema_sens = sens_info['ema_sens']
        print(f"  Calculating for Sensitivity {sens_key} ({sens_info['name']})...")
        
        live_price_val, ema_invest_val = await calculate_ema_invest(ticker_qs, ema_sens)
        scores_qs[sens_key] = f"{ema_invest_val:.2f}%" if ema_invest_val is not None else "N/A"
        if live_price_val is not None and sens_key == 2: # Use daily price for display
             live_price_qs_display = f"${live_price_val:.2f}"


        print(f"  Generating graph for Sensitivity {sens_key} ({sens_info['name']})...")
        graph_file_qs = await asyncio.to_thread(plot_ticker_graph, ticker_qs, ema_sens) # plot_ticker_graph is sync
        if graph_file_qs:
            graphs_qs_files.append(f"{sens_info['name']}: {graph_file_qs}")
        else:
            graphs_qs_files.append(f"{sens_info['name']}: Failed to generate graph.")

    # Print results for the user (as original function did)
    print("\n--- /quickscore Results (for User) ---")
    print(f"Ticker: {ticker_qs}")
    print(f"Live Price (Daily): {live_price_qs_display}")
    print("\nInvest Scores (Uncapped):")
    for sens_key, sens_info in sensitivity_map_qs.items():
        print(f"  Sensitivity {sens_key} ({sens_info['name']}): {scores_qs.get(sens_key, 'N/A')}")
    
    print("\nGenerated Graphs (check local directory):")
    for g_info in graphs_qs_files:
        print(f"  {g_info}")
    print("\n/quickscore analysis complete.")

    # Prepare summary for AI
    summary_for_ai = f"Quickscore analysis for {ticker_qs} completed. "
    summary_for_ai += f"Live Price (Daily): {live_price_qs_display}. "
    scores_summary_parts = []
    for sens_key, sens_info in sensitivity_map_qs.items():
        scores_summary_parts.append(f"Sens {sens_key} ({sens_info['name']}): {scores_qs.get(sens_key, 'N/A')}")
    summary_for_ai += "Scores: " + ", ".join(scores_summary_parts) + ". "
    graph_summary_parts = []
    for g_info in graphs_qs_files:
        if "Failed" not in g_info: # Only include successfully generated graph info
            graph_summary_parts.append(g_info.split(': ')[1]) # Just the filename part
    if graph_summary_parts:
        summary_for_ai += "Graphs generated: " + ", ".join(graph_summary_parts) + "."
    else:
        summary_for_ai += "No graphs were successfully generated."
        
    return summary_for_ai

async def handle_ai_prompt(user_new_message: str, is_new_session: bool = False):
    """
    Manages an AI chat session using a global AI_CONVERSATION_HISTORY list.
    Initializes history if is_new_session is True.
    Sends user messages, handles tool calls, and sends tool responses back to Gemini.
    The AI can also indicate when a conversation topic is concluded.
    """
    global AI_CONVERSATION_HISTORY, gemini_model, AVAILABLE_PYTHON_FUNCTIONS 

    print(f"\nUser AI Message: '{user_new_message}'")

    if not gemini_model:
        print("Gemini model is not configured. Cannot proceed with AI chat.")
        return "Error: Gemini model not configured."
    if 'AVAILABLE_PYTHON_FUNCTIONS' not in globals() or not AVAILABLE_PYTHON_FUNCTIONS:
        print("Error: AVAILABLE_PYTHON_FUNCTIONS dictionary is not defined or empty.")
        return "Error: System configuration issue (AVAILABLE_PYTHON_FUNCTIONS)."

    # --- Define All Tools Gemini Can Use (with Enhanced Descriptions) ---
    # These names MUST match the keys in your AVAILABLE_PYTHON_FUNCTIONS dictionary
    run_breakout_tool = FunctionDeclaration(
        name="run_breakout_analysis_singularity", 
        description="Identifies and returns a list of stocks currently showing strong breakout potential based on technical criteria. Use this if the user wants to find new, trending, or breakout stocks. Returns detailed data for up to 10 top breakout stocks (list of dictionaries, each with Ticker, Live Price, Invest Score, Status, etc.). Call this before attempting to save breakout data if fresh data is needed.", 
        parameters=None
    )
    quickscore_tool = FunctionDeclaration(
        name="handle_quickscore_command", 
        description="Performs a quick analysis for a SINGLE, SPECIFIC stock ticker provided by the user. It calculates various EMA-based investment scores and generates price/EMA graphs. Returns a string summary of the scores and graph locations. Use this for a focused look at one stock.", 
        parameters={"type":"object", "properties":{"ticker":{"type":"string", "description":"The specific stock ticker symbol to analyze, e.g., 'AAPL', 'MSFT'."}}, "required":["ticker"]}
    )
    market_command_tool = FunctionDeclaration(
        name="handle_market_command", 
        description="Provides a general overview of the S&P 500 market or saves its full data. Action 'display' shows top/bottom S&P 500 stocks and SPY score based on EMA sensitivity (1:Weekly, 2:Daily, 3:Hourly). Action 'save' saves comprehensive market data for the chosen sensitivity and date. Returns a summary string.", 
        parameters={"type":"object", "properties":{
            "action":{"type":"string","description":"Choose 'display' to show market scores, or 'save' to save detailed market data.","enum":["display","save"]},
            "sensitivity":{"type":"integer","description":"EMA Sensitivity: 1 (Weekly), 2 (Daily), 3 (Hourly). If user requests display and is vague, default to 2 (Daily)."}, 
            "date_str":{"type":"string","description":"Required for 'save' action: Date in MM/DD/YYYY format. If user says 'today', use current date: " + datetime.now().strftime('%m/%d/%Y') + "."}
        }, "required":["action","sensitivity"]}
    )
    risk_assessment_tool = FunctionDeclaration(
        name="handle_risk_command", 
        description="Performs a comprehensive R.I.S.K. module assessment of the overall market. Calculates General, Large Cap, EMA, and Combined scores, Market Invest Score, IVR, IV, recession likelihoods, and determines an overall market signal (Buy/Sell/Hold). Use 'assessment_type':'eod' for end-of-day specific calculations. Returns a dictionary of these key scores and the signal.", 
        parameters={"type":"object", "properties":{
            "assessment_type":{"type":"string","description":"Specify 'standard' for a regular risk assessment, or 'eod' for an end-of-day assessment. Default to 'standard' if user is vague.","enum":["standard","eod"]}
        }, "required":["assessment_type"]}
    )
    handle_assess_tool = FunctionDeclaration(
        name="handle_assess_command", 
        description="Performs specific financial assessments based on an 'assess_code'. Code 'A' for Stock Volatility analysis (needs tickers, timeframe, risk tolerance). Code 'B' for Manual Portfolio risk (needs holdings, backtest period). Code 'C' for Custom Saved Portfolio risk (needs portfolio code, value, backtest period). Code 'D' for Cultivate Portfolio risk (needs cultivate code, value, frac_shares, backtest period). Returns a summary string.", 
        parameters={"type":"object", "properties":{
            "assess_code":{"type":"string","description":"The specific assessment type: 'A' (Stock Volatility), 'B' (Manual Portfolio), 'C' (Custom Portfolio Risk), 'D' (Cultivate Portfolio Risk).","enum":["A","B","C","D"]},
            "tickers_str":{"type":"string","description":"For assess_code 'A': Comma-separated tickers e.g., 'AAPL,MSFT'."},
            "timeframe_str":{"type":"string","description":"For assess_code 'A': Analysis timeframe ('1Y','3M','1M'). Defaults to '1Y'.","enum":["1Y","3M","1M"]},
            "risk_tolerance":{"type":"integer","description":"For assess_code 'A': User's risk tolerance (1-5). Defaults to 3."},
            "backtest_period_str":{"type":"string","description":"For assess_codes 'B','C','D': Backtesting period (e.g.,'1y','3y','5y','10y')."},
            "manual_portfolio_holdings":{"type":"array","description":"For assess_code 'B': A list of portfolio holdings. Each item is a dictionary. Stocks: {{'ticker':'SYMBOL','shares':X.Y}}. Cash: {{'ticker':'Cash','value':Z.Z}}","items":{"type":"object","properties":{"ticker":{"type":"string"},"shares":{"type":"number"},"value":{"type":"number"}},"required":["ticker"]}},
            "custom_portfolio_code":{"type":"string","description":"For assess_code 'C': The code/name of a previously saved custom portfolio."},
            "value_for_assessment":{"type":"number","description":"For assess_codes 'C' and 'D': The total portfolio value for the assessment (e.g., 10000)."},
            "cultivate_portfolio_code":{"type":"string","description":"For assess_code 'D': The Cultivate portfolio strategy code ('A' or 'B').","enum":["A","B"]},
            "use_fractional_shares":{"type":"boolean","description":"For assess_code 'D': Whether to use fractional shares (true/false). Defaults to false."}
        },"required":["assess_code"]}
    )
    generate_history_graphs_tool = FunctionDeclaration(
        name="handle_history_command", 
        description="Generates and saves a series of historical graphs for the R.I.S.K. module, including scores, SPY/VIX prices, recession chances, IVR, and IV. The function reports on the success and filenames of generated graphs.", 
        parameters=None
    )
    custom_command_tool = FunctionDeclaration(
        name="handle_custom_command", 
        description="Runs analysis for an EXISTING saved custom portfolio OR saves data for an existing custom portfolio. Action 'run_existing_portfolio' runs an existing portfolio (can tailor with 'total_value', 'use_fractional_shares'). Action 'save_portfolio_data' saves data for an existing portfolio (needs 'date_to_save'). Requires 'portfolio_code' for both actions. Returns a summary.", 
        parameters={"type":"object", "properties":{
            "action":{"type":"string","description":"Specify 'run_existing_portfolio' to analyze it, or 'save_portfolio_data' to save its historical data.","enum":["run_existing_portfolio","save_portfolio_data"]},
            "portfolio_code":{"type":"string","description":"The code/name of the pre-existing custom portfolio."},
            "tailor_to_value":{"type":"boolean","description":"For 'run_existing_portfolio': Optional. Set to true to tailor results to a specific 'total_value'."},
            "total_value":{"type":"number","description":"For 'run_existing_portfolio' with tailoring: The monetary value to tailor the portfolio analysis to."},
            "use_fractional_shares":{"type":"boolean","description":"For 'run_existing_portfolio' with tailoring: Optional. If true, fractional shares will be used in tailoring."},
            "date_to_save":{"type":"string","description":"For 'save_portfolio_data': Date MM/DD/YYYY. If user says 'today', use current date: "+datetime.now().strftime('%m/%d/%Y') + "."}
        },"required":["action","portfolio_code"]}
    )
    save_breakout_data_tool = FunctionDeclaration(
        name="save_breakout_stock_data", # Maps to handle_breakout_command for AI save action
        description="Saves the data from the most recent breakout analysis run to a historical database for a specified date. Run breakout analysis (`run_breakout_analysis_singularity`) first for current data. Returns a success/failure message.",
        parameters={"type": "object","properties": {
            "date_to_save": {"type": "string","description": "Date MM/DD/YYYY. If user says 'today', use current date: " + datetime.now().strftime('%m/%d/%Y') + "."}
        },"required": ["date_to_save"]}
    )
    cultivate_analysis_tool = FunctionDeclaration(
        name="handle_cultivate_command", 
        description="Runs the complex Cultivate portfolio analysis (Code 'A' for Screener-based, 'B' for S&P 500-based) for a given portfolio value and fractional shares preference. Can also save the generated combined data if 'action' is 'save_data' for a 'date_to_save'. Returns a summary of the analysis or save action.", 
        parameters={"type":"object", "properties":{
            "cultivate_code":{"type":"string","description":"The Cultivate strategy code: 'A' (Screener) or 'B' (S&P 500).","enum":["A","B"]},
            "portfolio_value":{"type":"number","description":"The total portfolio value for the Cultivate analysis (e.g., 10000, 50000)."},
            "use_fractional_shares":{"type":"boolean","description":"Whether to use fractional shares (true/false). Defaults to false if not specified."},
            "action":{"type":"string","description":"Optional. 'run_analysis' (default) to perform and display analysis, or 'save_data' to run and then save the combined data.","enum":["run_analysis","save_data"]},
            "date_to_save":{"type":"string","description":"Optional. Date (MM/DD/YYYY) to save data for. Required if 'action' is 'save_data'. If 'today', use: "+datetime.now().strftime('%m/%d/%Y') + "."}
        },"required":["cultivate_code","portfolio_value","use_fractional_shares"]}
    )
    invest_analysis_tool = FunctionDeclaration(
        name="handle_invest_command", 
        description="Analyzes multiple user-defined stock groups (sub-portfolios) with specified tickers and percentage weights. Calculates investment scores based on EMA sensitivity and amplification. Can optionally tailor the final combined portfolio to a total value. Sum of weights for all sub-portfolios must be 100%. Returns a summary.", 
        parameters={"type":"object", "properties":{
            "ema_sensitivity":{"type":"integer","description":"EMA sensitivity: 1 (Weekly), 2 (Daily), or 3 (Hourly)."},
            "amplification":{"type":"number","description":"Amplification factor (e.g., 0.5, 1.0, 2.0)."},
            "sub_portfolios":{"type":"array","description":"A list of sub-portfolios. Each is an object with 'tickers' (comma-separated string like 'AAPL,MSFT') and 'weight' (number, e.g., 60 for 60%). Sum of all weights must be 100.","items":{"type":"object","properties":{"tickers":{"type":"string"},"weight":{"type":"number"}},"required":["tickers","weight"]}},
            "tailor_to_value":{"type":"boolean","description":"Optional. True to tailor. Default false."},
            "total_value":{"type":"number","description":"Optional. Total value for tailoring. Required if 'tailor_to_value' is true."},
            "use_fractional_shares":{"type":"boolean","description":"Optional. For tailoring. Default false."}
        },"required":["ema_sensitivity","amplification","sub_portfolios"]}
    )
    
    available_tools_for_gemini = Tool(function_declarations=[
        run_breakout_tool, quickscore_tool, market_command_tool, risk_assessment_tool,
        handle_assess_tool, generate_history_graphs_tool, custom_command_tool,
        invest_analysis_tool, cultivate_analysis_tool, save_breakout_data_tool
    ])
    tool_config_for_gemini = {"function_calling_config": {"mode": "ANY"}}

    if is_new_session: 
        print("Initializing new AI conversation history for session...")
        AI_CONVERSATION_HISTORY.clear() 
        base_system_prompt_template = load_system_prompt() # load_system_prompt defined elsewhere
        
        current_date_val = datetime.now().strftime('%B %d, %Y')
        current_date_mmddyyyy_val = datetime.now().strftime('%m/%d/%Y')
        
        try:
            formatted_system_user_prompt = base_system_prompt_template.format(
                user_prompt=user_new_message, 
                current_date_for_ai_prompt=current_date_val,
                current_date_mmddyyyy_for_ai_prompt=current_date_mmddyyyy_val
            )
        except KeyError as e_format:
            print(f"KeyError during system prompt formatting: {e_format}. Check system_prompt.txt (ensure examples like {{{{key}}}}:{{{{value}}}} use double curly braces).")
            traceback.print_exc(); return "Error: System prompt formatting failed."

        AI_CONVERSATION_HISTORY.append({"role": "user", "parts": [{"text": formatted_system_user_prompt}]})
        AI_CONVERSATION_HISTORY.append({"role": "model", "parts": [{"text": "Okay, I've received your initial request and understand my role based on the system guidelines. I will now process it using the available tools and context."}]})
    else: 
        if not AI_CONVERSATION_HISTORY:
            # This case should ideally be prevented by main_singularity setting is_new_session=True if history is empty
            print("Warning: Continuing AI chat, but history is empty. This might lead to loss of context. Forcing new session setup.")
            return await handle_ai_prompt(user_new_message, is_new_session=True) # Retry as new session
        AI_CONVERSATION_HISTORY.append({"role": "user", "parts": [{"text": user_new_message}]})
    
    max_internal_turns = 5 
    last_executed_function_call_details = None # Track within the processing of a single user_new_message

    for turn in range(max_internal_turns):
        print(f"\n--- AI Internal Turn {turn + 1}/{max_internal_turns} for user message: '{user_new_message[:50]}...' ---")
        if not AI_CONVERSATION_HISTORY: print("Critical Error: AI Conversation history became empty mid-processing."); break
        
        try:
            print(f"Sending to Gemini (total history items: {len(AI_CONVERSATION_HISTORY)})...")
            
            response = await asyncio.to_thread(
                gemini_model.generate_content,
                contents=AI_CONVERSATION_HISTORY, 
                tools=[available_tools_for_gemini],
                tool_config=tool_config_for_gemini
            )

            candidate = response.candidates[0] if response.candidates else None
            if not candidate: print("No candidate in Gemini's response."); break 
            
            current_finish_reason = candidate.finish_reason.name if hasattr(candidate.finish_reason, 'name') else str(candidate.finish_reason)
            print(f"AI Finish Reason: {current_finish_reason}")

            if current_finish_reason == "MALFORMED_FUNCTION_CALL":
                print("Error: Gemini attempted a function call but it was malformed.")
                malformed_call_info = "Malformed call details not available."
                if hasattr(candidate, 'content') and candidate.content and candidate.content.parts:
                     for p_proto_err in candidate.content.parts:
                        if p_proto_err.function_call.name: 
                            malformed_call_info = f"Malformed Call: Name: {p_proto_err.function_call.name}, Args: {dict(p_proto_err.function_call.args)}"
                            print(malformed_call_info)
                # Add structured error to history for AI to see its mistake
                AI_CONVERSATION_HISTORY.append({"role": "model", "parts": [{"text": f"My previous attempt to call a function resulted in a malformed structure. Details: {malformed_call_info}. I will try to re-evaluate based on the user's request or await further instructions."}]})
                # We don't break the main loop here; instead, this message goes back to AI in the next internal turn if any.
                # However, if AI keeps making malformed calls, max_internal_turns will stop it.
                # For now, let's just let this turn end, and the next interaction will have this error in history.
                # Or, to force AI to re-evaluate immediately *without* trying to call again, we can break the internal loop.
                break # Break internal loop, AI will respond to the error on next user input or if loop was to continue

            model_response_content = candidate.content 
            function_call_to_execute = None
            text_response_from_gemini = ""
            model_turn_parts_for_history = []

            if model_response_content and model_response_content.parts:
                for p_proto in model_response_content.parts:
                    if p_proto.function_call.name: 
                        function_call_to_execute = p_proto.function_call
                        model_turn_parts_for_history.append(
                            {"function_call": {"name": p_proto.function_call.name, "args": dict(p_proto.function_call.args)}})
                        break 
                    elif p_proto.text:
                        text_response_from_gemini += p_proto.text 
                if not function_call_to_execute and text_response_from_gemini:
                     model_turn_parts_for_history = [{"text": text_response_from_gemini}]
            
            if model_turn_parts_for_history:
                AI_CONVERSATION_HISTORY.append({"role": "model", "parts": model_turn_parts_for_history})
            elif not function_call_to_execute and not text_response_from_gemini.strip():
                 print("Warning: Model responded with an empty content part for its turn.")

            if function_call_to_execute:
                fc = function_call_to_execute; function_name = fc.name
                args_dict_from_ai = {key: val for key, val in fc.args.items()} 
                print(f"Gemini wants to call Function: {function_name}")
                print(f"With Arguments from AI: {json.dumps(args_dict_from_ai, indent=2)}")

                current_call_details = {'name': function_name, 'args_tuple': tuple(sorted(args_dict_from_ai.items()))} # Make args hashable for comparison
                
                if last_executed_function_call_details and \
                   last_executed_function_call_details['name'] == function_name and \
                   last_executed_function_call_details['args_tuple'] == current_call_details['args_tuple']:
                    
                    repeated_call_error_msg = f"Error: I am trying to run the function '{function_name}' with the exact same arguments as the last step, which may indicate a loop or no progress. I should try a different approach or provide a text response to the user."
                    print(f"Warning: AI attempting to re-run identical function call: {function_name}.")
                    AI_CONVERSATION_HISTORY.append({"role": "tool", "parts": [{"function_response": {"name": function_name, "response": {"error": repeated_call_error_msg}}}]})
                    last_executed_function_call_details = None # Reset to allow a *different* call next time
                    continue # Skip execution, go to next internal turn to send this error to AI

                if function_name in AVAILABLE_PYTHON_FUNCTIONS:
                    python_function = AVAILABLE_PYTHON_FUNCTIONS[function_name]
                    print(f"Executing Python function: {function_name}...")
                    function_result_value = f"Error: Function '{function_name}' did not return a value as expected."
                    try:
                        prepared_python_function_args = [] 
                        ai_params_for_function_call = {} 
                        call_with_ai_params_dict = False                     
                        
                        # --- Argument Preparation Dispatcher (MUST BE COMPLETE FOR ALL TOOLS) ---
                        if function_name == "handle_quickscore_command":
                            ticker_arg = args_dict_from_ai.get("ticker");
                            if ticker_arg: prepared_python_function_args = [[str(ticker_arg)]] 
                            else: raise ValueError("Ticker not provided for quickscore.")
                        elif function_name == "handle_market_command":
                            action_arg = args_dict_from_ai.get("action"); s_arg = args_dict_from_ai.get("sensitivity"); d_arg = args_dict_from_ai.get("date_str")
                            if action_arg and s_arg is not None:
                                s_int = int(float(s_arg)); # Robust conversion
                                if s_int not in [1,2,3]: raise ValueError(f"Sensitivity '{s_int}' out of range [1,2,3].")
                                c_args_list = [str(action_arg), str(s_int)]
                                if str(action_arg).lower() == "save":
                                    if d_arg: 
                                        try: datetime.strptime(str(d_arg), '%m/%d/%Y'); c_args_list.append(str(d_arg))
                                        except ValueError: raise ValueError(f"Invalid date format '{d_arg}' from AI. Expected MM/DD/YYYY.")
                                    else: raise ValueError("Date missing for market save action.")
                                prepared_python_function_args = [c_args_list]
                            else: raise ValueError("Action or sensitivity missing for market command.")
                        elif function_name == "handle_risk_command":
                            a_type = args_dict_from_ai.get("assessment_type", "standard").lower()
                            ai_arg = "standard_ai" if a_type == "standard" else "eod_ai"
                            prepared_python_function_args = [[ai_arg]]
                        elif function_name == "run_breakout_analysis_singularity":
                            prepared_python_function_args = [] 
                        elif function_name in ["handle_assess_command", "handle_custom_command", 
                                               "handle_invest_command", "handle_cultivate_command",
                                               "save_breakout_stock_data"]: 
                            call_with_ai_params_dict = True
                            ai_params_for_function_call = args_dict_from_ai
                        elif function_name == "handle_history_command":
                            prepared_python_function_args = [[]]
                        else: 
                            raise NotImplementedError(f"Argument preparation for function '{function_name}' is not defined.")

                        last_executed_function_call_details = current_call_details # Update after successful arg prep, before call

                        if asyncio.iscoroutinefunction(python_function):
                            if call_with_ai_params_dict: function_result_value = await python_function([], ai_params=ai_params_for_function_call)
                            elif function_name == "run_breakout_analysis_singularity": function_result_value = await python_function()
                            else: function_result_value = await python_function(*prepared_python_function_args)
                        else: 
                            if call_with_ai_params_dict: function_result_value = await asyncio.to_thread(python_function, [], ai_params=ai_params_for_function_call)
                            else: function_result_value = await asyncio.to_thread(python_function, *prepared_python_function_args)
                        
                        print(f"Result of Python function {function_name}: (Type: {type(function_result_value)})")
                        # (Optional detailed print of result for debugging)

                        tool_response_payload = {}
                        if isinstance(function_result_value, list): tool_response_payload = {"data_list": function_result_value} 
                        elif isinstance(function_result_value, dict):
                            try: json.dumps(function_result_value); tool_response_payload = function_result_value
                            except TypeError: tool_response_payload = {"summary": str(function_result_value)}
                        elif isinstance(function_result_value, str): tool_response_payload = {"summary": function_result_value}
                        elif function_result_value is None: tool_response_payload = {"message": f"Function '{function_name}' executed successfully but returned no explicit data (None)."}
                        else: tool_response_payload = {"content": str(function_result_value)}
                        
                        AI_CONVERSATION_HISTORY.append({"role": "tool", "parts": [{"function_response": {"name": function_name, "response": tool_response_payload}}]})
                        # Loop continues if not max_internal_turns

                    except ValueError as e_val_dispatch: 
                        error_message = f"Argument error for {function_name}: {str(e_val_dispatch)}"
                        print(error_message); AI_CONVERSATION_HISTORY.append({"role": "tool", "parts": [{"function_response": {"name": function_name, "response": {"error": error_message}}}]}); last_executed_function_call_details = None; break 
                    except Exception as e_exec:
                        error_message = f"Error executing {function_name}: {str(e_exec)}"; print(error_message); traceback.print_exc()
                        AI_CONVERSATION_HISTORY.append({"role": "tool", "parts": [{"function_response": {"name": function_name, "response": {"error": error_message}}}]}); last_executed_function_call_details = None; break 
                else: 
                    error_message = f"Function '{function_name}' is not mapped in AVAILABLE_PYTHON_FUNCTIONS."; print(error_message)
                    AI_CONVERSATION_HISTORY.append({"role": "tool", "parts": [{"function_response": {"name": function_name, "response": {"error": error_message}}}]}); last_executed_function_call_details = None; break 
            
            elif text_response_from_gemini.strip():
                print("--- Gemini's Text Response ---"); print(text_response_from_gemini.strip()); print("-----------------------------")
                processed_text_response = text_response_from_gemini.lower().strip()
                end_phrases = ["session ended", "goodbye", "that's all", "no further assistance", "request complete", "i'm done", "let me know if you need anything else!"] 
                is_question_from_ai = any(q_phrase in processed_text_response for q_phrase in ["anything else?", "what next?", "can i help with another task?", "how else can i help?", "do you want to proceed?"])
                
                if any(phrase in processed_text_response for phrase in end_phrases) and not is_question_from_ai:
                    print("AI indicated end of conversation. Current chat session context will be cleared by /ai_end_chat or next /ai_start_chat.")
                break 
            
            else: 
                print("Gemini did not propose a function call or provide text. Ending this interaction turn.")
                break 
        except Exception as e_turn: 
            print(f"An error occurred during AI interaction turn: {e_turn}"); traceback.print_exc()
            if 'response' in locals() and hasattr(response, '_raw_response'): print(f"Raw response proto: {response._raw_response}")
            elif 'response' in locals(): print(f"Response object: {response}")
            break 
    
    if turn == max_internal_turns - 1 and function_call_to_execute : 
        print("Max internal turns reached while processing function calls for this message. Waiting for next user input.")
    
    print("\n--- End of AI processing for this message. Awaiting next input or command. ---")
    return "AI processing for current message complete."
  
# This dictionary should be defined where handle_ai_prompt can access it.
AVAILABLE_PYTHON_FUNCTIONS = {
    "run_breakout_analysis_singularity": run_breakout_analysis_singularity,
    "handle_quickscore_command": handle_quickscore_command,
    "handle_market_command": handle_market_command,
    "handle_risk_command": handle_risk_command,
    "handle_assess_command": handle_assess_command,
    "handle_history_command": handle_history_command,
    "handle_invest_command": handle_invest_command,
    "handle_custom_command": handle_custom_command,
    "handle_cultivate_command": handle_cultivate_command,
    "save_breakout_stock_data": handle_breakout_command, # Tool name maps to the refactored command handler
}

# --- Main Singularity Loop ---
async def main_singularity():
    global AI_CONVERSATION_HISTORY, risk_persistent_signal, risk_signal_day, gemini_model # Add other globals used in main

    # --- Initialize Loggers and Gemini API model (ensure this is done robustly) ---
    if not risk_logger.hasHandlers(): 
        risk_file_handler_main = logging.FileHandler(RISK_LOG_FILE) 
        risk_formatter_main = logging.Formatter('%(asctime)s - %(levelname)s - Module:%(module)s - Func:%(funcName)s - %(message)s')
        risk_file_handler_main.setFormatter(risk_formatter_main)
        risk_logger.addHandler(risk_file_handler_main)
    risk_logger.info("Singularity Application Started.")

    # Ensure gemini_model is initialized globally (e.g., from API key)
    # Example:
    # if 'gemini_model' not in globals() or gemini_model is None:
    #     GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Or your hardcoded key
    #     if GEMINI_API_KEY:
    #         try:
    #             genai.configure(api_key=GEMINI_API_KEY)
    #             gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    #             print("Gemini API configured successfully in main.")
    #         except Exception as e_cfg:
    #             print(f"FATAL: Error configuring Gemini API in main: {e_cfg}")
    #             gemini_model = None
    #     else:
    #         print("FATAL: GEMINI_API_KEY not found.")
    #         gemini_model = None
            
    if not gemini_model:
        print("FATAL: Gemini model not initialized. AI commands will not function. Please check API key and configuration.")
        # Decide if you want the script to exit or continue with AI features disabled.
        # For now, it will continue, but /ai commands will report errors.

    display_welcome_message() 
    display_commands()    

    while True:
        try:
            prompt_indicator = "[AI Chat Active] > " if AI_CONVERSATION_HISTORY else "Enter command: "
            user_input_full_line = input(prompt_indicator).strip()
            
            if not user_input_full_line:
                continue

            # Handle hardcoded "End Chat" command first
            if user_input_full_line.lower() == "end chat":
                if AI_CONVERSATION_HISTORY:
                    print("AI chat session ended by user command 'End Chat'. History cleared.")
                    AI_CONVERSATION_HISTORY.clear()
                    # Optionally save CONVERSATION_HISTORY_FOR_LOGGING if you maintain a separate one
                else:
                    print("No active AI chat session to end.")
                continue # Go to next input prompt

            command_parts = user_input_full_line.split()
            command = command_parts[0].lower()
            args = command_parts[1:]

            if command == "/ai_start_chat":
                user_natural_prompt = " ".join(args)
                if not user_natural_prompt:
                    print("Usage: /ai_start_chat <your initial question>")
                elif not gemini_model:
                    print("Gemini API model not available.")
                else:
                    print("Starting new AI chat session...")
                    AI_CONVERSATION_HISTORY.clear() 
                    await handle_ai_prompt(user_natural_prompt, is_new_session=True)
            
            elif command == "/ai_end_chat": # Explicit command to end chat
                if AI_CONVERSATION_HISTORY:
                    print("AI chat session explicitly ended via /ai_end_chat. History cleared.")
                    AI_CONVERSATION_HISTORY.clear()
                else:
                    print("No active AI chat session to end.")

            elif command == "/ai": 
                user_natural_prompt = " ".join(args)
                if not user_natural_prompt:
                    print("Usage: /ai <your request or question>")
                elif not gemini_model:
                    print("Gemini API model not available.")
                else:
                    is_new = not AI_CONVERSATION_HISTORY # Start new if history is empty
                    if is_new:
                        print("No active AI chat. Starting new one with this /ai prompt...")
                    await handle_ai_prompt(user_natural_prompt, is_new_session=is_new)
            
            # --- Standard CLI Command Handlers ---
            # Ensure these functions are defined to take `args` and optionally `ai_params=None`
            # if they are also to be called by the AI via AVAILABLE_PYTHON_FUNCTIONS.
            # If they are PURELY CLI, they might not need the ai_params signature.
            elif command == "/invest": await handle_invest_command(args) 
            elif command == "/custom": await handle_custom_command(args)
            elif command == "/breakout": await handle_breakout_command(args)
            elif command == "/market": await handle_market_command(args)
            elif command == "/cultivate": await handle_cultivate_command(args)
            elif command == "/assess": await handle_assess_command(args)
            elif command == "/quickscore": await handle_quickscore_command(args)
            elif command == "/risk": await handle_risk_command(args)
            elif command == "/history": await handle_history_command(args)
            elif command == "/help": display_commands()
            elif command == "/exit":
                print("Exiting Market Insights Center. Goodbye!"); risk_logger.info("Singularity App Exited."); break
            else: 
                if AI_CONVERSATION_HISTORY and not user_input_full_line.startswith("/"): 
                    print(f"Continuing AI chat with: '{user_input_full_line}'")
                    await handle_ai_prompt(user_input_full_line, is_new_session=False)
                else: 
                    print(f"Unknown command: {command}. Type /help for available commands.")
        
        except KeyboardInterrupt:
            print("\nExiting Market Insights Center. Goodbye!")
            if AI_CONVERSATION_HISTORY : print(f"Final AI chat history had {len(AI_CONVERSATION_HISTORY)} turns before interrupt.")
            risk_logger.info("Singularity Application Terminated by User (KeyboardInterrupt).")
            break
        except Exception as e:
            print(f"An unexpected error occurred in the main loop: {e}")
            risk_logger.exception("Unexpected error in main_singularity loop:")
            traceback.print_exc()

if __name__ == "__main__":
    try:
        asyncio.run(main_singularity())
    except KeyboardInterrupt:
        print("\nApplication terminated by user (main execution).")
    except Exception as e_run: 
        print(f"Critical error running application: {e_run}")
        traceback.print_exc() # Print to console for critical startup errors
        if 'risk_logger' in globals() and risk_logger: # Check if logger was initialized
            risk_logger.critical(f"Bot crashed with an unexpected error: {e_run}", exc_info=True)
        else: # Fallback basic logging if risk_logger failed
            logging.basicConfig(filename=RISK_LOG_FILE, level=logging.ERROR)
            logging.critical(f"Bot crashed critically before full logger setup or after it was lost: {e_run}", exc_info=True)

