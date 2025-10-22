# --- Imports for breakout_command ---
import asyncio
import os
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
import traceback

import pandas as pd
from tabulate import tabulate
from tradingview_screener import Query, Column

# --- Imports from other command modules ---
from invest_command import calculate_ema_invest, calculate_one_year_invest

# --- Constants ---
BREAKOUT_TICKERS_FILE = 'breakout_tickers.csv'
BREAKOUT_HISTORICAL_DB_FILE = 'breakout_historical_database.csv'

# --- Helper Functions ---
def safe_score(value: Any) -> float:
    try:
        if pd.isna(value) or value is None: return 0.0
        if isinstance(value, str): value = value.replace('%', '').replace('$', '').strip()
        return float(value)
    except (ValueError, TypeError): return 0.0

# --- Core Logic Functions (moved from breakout.py) ---
async def run_breakout_analysis_singularity(is_called_by_ai: bool = False) -> dict:
    existing_tickers_data = {}
    if os.path.exists(BREAKOUT_TICKERS_FILE):
        try:
            df_existing = pd.read_csv(BREAKOUT_TICKERS_FILE)
            if not df_existing.empty:
                for col in ["Highest Invest Score", "Lowest Invest Score", "Live Price", "1Y% Change", "Invest Score"]:
                    if col in df_existing.columns:
                        if df_existing[col].dtype == 'object': df_existing[col] = df_existing[col].astype(str).str.replace('%', '', regex=False).str.replace('$', '', regex=False).str.strip()
                        df_existing[col] = pd.to_numeric(df_existing[col], errors='coerce')
                existing_tickers_data = df_existing.set_index('Ticker').to_dict('index')
        except Exception as e:
            # FIX: Make error visible instead of silent
            print(f"  -> Info: Could not read existing breakout file: {e}")
            
    existing_tickers_set = set(existing_tickers_data.keys())
    new_tickers_from_screener = []
    try:
        query = Query().select('name').where(Column('market_cap_basic') >= 1_000_000_000, Column('volume') >= 1_000_000, Column('change|1W') >= 20, Column('close') >= 1, Column('average_volume_90d_calc') >= 1_000_000).order_by('change', ascending=False).limit(100)
        _, new_tickers_df = await asyncio.to_thread(query.get_scanner_data, timeout=60)
        if new_tickers_df is not None and 'name' in new_tickers_df.columns:
            new_tickers_from_screener = sorted(list(set([str(t).split(':')[-1].replace('.', '-') for t in new_tickers_df['name'].tolist() if pd.notna(t)])))
    except Exception as e:
        # FIX: Make error visible instead of silent
        print(f"  -> Warning: TradingView screener failed: {e}")

    all_tickers_to_process = sorted(list(set(list(existing_tickers_data.keys()) + new_tickers_from_screener)))
    temp_updated_data = []
    
    for ticker_b in all_tickers_to_process:
        try:
            live_price, current_invest_score = await calculate_ema_invest(ticker_b, 2, is_called_by_ai=True)
            one_year_change, _ = await calculate_one_year_invest(ticker_b, is_called_by_ai=True)
            if current_invest_score is None: continue
            
            existing_entry = existing_tickers_data.get(ticker_b, {})
            highest_score = max(safe_score(existing_entry.get("Highest Invest Score")), current_invest_score)
            lowest_score = min(safe_score(existing_entry.get("Lowest Invest Score", float('inf'))), current_invest_score)
            
            # This is the filtering logic for breakout stocks
            if not (current_invest_score > 600 or current_invest_score < 100.0 or current_invest_score < (3.0/4.0) * highest_score):
                status = "Repeat" if ticker_b in existing_tickers_set else "New"
                temp_updated_data.append({
                    "Ticker": ticker_b, 
                    "Live Price": f"{live_price:.2f}" if live_price else "N/A", 
                    "Invest Score": f"{current_invest_score:.2f}%", 
                    "Highest Invest Score": f"{highest_score:.2f}%", 
                    "Lowest Invest Score": f"{lowest_score:.2f}%", 
                    "1Y% Change": f"{one_year_change:.2f}%" if one_year_change is not None else "N/A", 
                    "Status": status, 
                    "_sort_score": current_invest_score
                })
        except Exception as e:
            # FIX: Make error visible instead of silent
            print(f"  -> Warning: Could not process ticker {ticker_b}: {e}")
            continue
            
    temp_updated_data.sort(key=lambda x: x['_sort_score'], reverse=True)
    final_data = [{k: v for k, v in item.items() if k != '_sort_score'} for item in temp_updated_data]
    
    return {"current_breakout_stocks": final_data}

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
            price_str = str(row.get('Live Price', 'N/A')).replace('$', '').strip()
            score_str = str(row.get('Invest Score', 'N/A')).replace('%', '').strip()
            price_val = safe_score(price_str)
            score_val = safe_score(score_str)

            historical_data_to_save.append({
                'DATE': date_str,
                'TICKER': row.get('Ticker', 'ERR'),
                'PRICE': f"{price_val:.2f}" if price_val is not None and not pd.isna(price_val) else "N/A",
                'INVEST_SCORE': f"{score_val:.2f}" if score_val is not None and not pd.isna(score_val) else "N/A"
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

    except Exception as e_save_hist:
        msg = f"An unexpected error occurred processing/saving historical breakout data: {e_save_hist}"
        if not is_called_by_ai:
            print(msg)
            traceback.print_exc()
        return msg
   
async def handle_breakout_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """
    Handles breakout stock analysis by running analysis or saving data.
    """
    # FIX: This function is now fully implemented to handle both CLI and AI paths
    # and to correctly process the results from the analysis function.

    action_to_perform = "run"
    date_str_for_save = None

    if ai_params:
        action_to_perform = ai_params.get("action", "run")
        if action_to_perform == "save":
            date_str_for_save = ai_params.get("date_to_save", datetime.now().strftime('%m/%d/%Y'))
    elif args and args[0] == "3725":
        action_to_perform = "save"
        date_str_for_save = input("Enter date (MM/DD/YYYY) to save breakout data: ")

    if action_to_perform == "save":
        if not date_str_for_save:
            print("❌ Error: Date is required for saving breakout data.")
            return "Error: Date not provided for save action." if is_called_by_ai else None
        
        save_summary = await save_breakout_data_singularity(date_str_for_save, is_called_by_ai=is_called_by_ai)
        return save_summary if is_called_by_ai else None

    elif action_to_perform == "run":
        analysis_result = await run_breakout_analysis_singularity(is_called_by_ai=is_called_by_ai)
        breakout_stocks = analysis_result.get("current_breakout_stocks")

        if breakout_stocks:
            print("\n--- Breakout Stocks Analysis ---")
            print(tabulate(breakout_stocks, headers="keys", tablefmt="pretty"))

            try:
                df_to_save = pd.DataFrame(breakout_stocks)
                df_to_save.to_csv(BREAKOUT_TICKERS_FILE, index=False)
                print(f"\n✔ Successfully saved {len(breakout_stocks)} records to {BREAKOUT_TICKERS_FILE}")
            except Exception as e:
                print(f"\n❌ Error saving breakout results to file: {e}")
            
            return "Breakout analysis complete. Results displayed and saved." if is_called_by_ai else None
        else:
            print("\n--- Breakout Stocks Analysis ---")
            print("No breakout stocks found matching the criteria.")
            return "No breakout stocks found." if is_called_by_ai else None