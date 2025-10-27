# prometheus_core.py
import sqlite3
import json
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import traceback
import random
import sys
import io
import google.generativeai as genai
import logging
import yfinance as yf
from dateutil.relativedelta import relativedelta
from typing import Dict, List, Any, Callable, Optional, Tuple
import numpy as np
from tabulate import tabulate
import os
import inspect # For signature checking

# --- Constants ---
SYNTHESIZED_WORKFLOWS_FILE = 'synthesized_workflows.json'

# --- Prometheus Core Logger ---
prometheus_logger = logging.getLogger('PROMETHEUS_CORE')
prometheus_logger.setLevel(logging.DEBUG) # <<< Set logger level to DEBUG for more detail
prometheus_logger.propagate = False
if not prometheus_logger.hasHandlers():
    prometheus_log_file = 'prometheus_core.log'
    # Use RotatingFileHandler for better log management
    from logging.handlers import RotatingFileHandler
    # --- MODIFICATION: Added encoding='utf-8' ---
    prometheus_file_handler = RotatingFileHandler(prometheus_log_file, maxBytes=5*1024*1024, backupCount=2, encoding='utf-8') # 5MB limit, 2 backups
    prometheus_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s') # More detail in format
    prometheus_file_handler.setFormatter(prometheus_formatter)
    prometheus_logger.addHandler(prometheus_file_handler)
    # Add a handler to also print DEBUG messages to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG) # <<< Print DEBUG to console
    # Use a simpler format for console to reduce noise
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    prometheus_logger.addHandler(console_handler)


# --- Robust YFinance Download Helper ---
# (get_yf_download_robustly remains the same)
async def get_yf_download_robustly(tickers: list, **kwargs) -> pd.DataFrame:
    """ Robust wrapper for yf.download with retry logic and standardization. """
    max_retries = 2
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(random.uniform(0.3, 0.8))
            kwargs.setdefault('progress', False); kwargs.setdefault('timeout', 15); kwargs.setdefault('auto_adjust', False)
            prometheus_logger.debug(f"get_yf_download_robustly: Attempt {attempt+1} for {tickers} with kwargs: {kwargs}")
            data = await asyncio.to_thread(yf.download, tickers=tickers, **kwargs)

            if isinstance(data, dict):
                 valid_dfs = {name: df for name, df in data.items() if isinstance(df, pd.DataFrame) and not df.empty}
                 if not valid_dfs: raise IOError(f"yf.download returned dict with no valid DataFrames for {tickers}")
                 data = pd.concat(valid_dfs.values(), axis=1, keys=valid_dfs.keys())
                 if isinstance(data.columns, pd.MultiIndex):
                     if data.columns.names[0] == 'Ticker': data.columns = data.columns.swaplevel(0, 1)
                     data.columns.names = ['Price', 'Ticker']
            if not isinstance(data, pd.DataFrame): raise TypeError(f"yf.download did not return a DataFrame (got {type(data)})")
            if data.empty: raise IOError(f"yf.download returned empty DataFrame for {tickers} (attempt {attempt+1})")
            prometheus_logger.debug(f"get_yf_download_robustly: Download successful for {tickers} (attempt {attempt+1}). Shape: {data.shape}")
            if data.isnull().all().all(): raise IOError(f"yf.download returned DataFrame with all NaN data for {tickers} (attempt {attempt+1})")
            if not isinstance(data.columns, pd.MultiIndex):
                 ticker_name = tickers[0] if len(tickers) == 1 else 'Unknown'
                 data.columns = pd.MultiIndex.from_product([data.columns, [ticker_name]], names=['Price', 'Ticker'])
            elif data.columns.names != ['Price', 'Ticker']:
                 try:
                     level_map = {name: i for i, name in enumerate(data.columns.names)}
                     if 'Price' in level_map and 'Ticker' in level_map:
                          if level_map['Price'] != 0 or level_map['Ticker'] != 1: data.columns = data.columns.reorder_levels(['Price', 'Ticker'])
                          data.columns.names = ['Price', 'Ticker']
                     else: data.columns.names = ['Price', 'Ticker']
                 except Exception as e_reformat: prometheus_logger.warning(f"Could not standardize MultiIndex names: {data.columns.names}. Error: {e_reformat}")
            return data
        except Exception as e:
            error_type = type(e).__name__; error_msg = str(e)
            prometheus_logger.warning(f"get_yf_download_robustly: Attempt {attempt+1} failed for {tickers}. Error ({error_type}): {error_msg}")
            if attempt < max_retries - 1: await asyncio.sleep((attempt + 1) * 1)
            else: prometheus_logger.error(f"All yf download attempts failed for {tickers}. Last error ({error_type}): {error_msg}")
            return pd.DataFrame()
    return pd.DataFrame()


# --- Minimal calculate_ema_invest for context fetching ---
# (Added initialization fix for close_col_tuple)
async def calculate_ema_invest_minimal(ticker: str, ema_interval: int = 2) -> Optional[float]:
    """ Minimal version to get INVEST score for context. """
    interval_map = {1: "1wk", 2: "1d", 3: "1h"}; period_map = {1: "max", 2: "10y", 3: "2y"}
    try:
        data = await get_yf_download_robustly(tickers=[ticker], period=period_map.get(ema_interval, "10y"), interval=interval_map.get(ema_interval, "1d"), auto_adjust=True)
        if data.empty: prometheus_logger.debug(f"calculate_ema_invest_minimal({ticker}): No data from download."); return None
        close_prices = None; price_level_name = 'Price'; ticker_level_name = 'Ticker'; close_col_tuple = None # <<< Initialize here
        if isinstance(data.columns, pd.MultiIndex):
             if ('Close', ticker) in data.columns: close_prices = data[('Close', ticker)]
             elif 'Close' in data.columns.get_level_values(price_level_name): close_col_tuple = next((col for col in data.columns if col[data.columns.names.index(price_level_name)] == 'Close'), None);
             if close_col_tuple: close_prices = data[close_col_tuple]
        elif 'Close' in data.columns: close_prices = data['Close']
        if close_prices is None or close_prices.isnull().all() or len(close_prices.dropna()) < 55: prometheus_logger.warning(f"Insufficient 'Close' data for {ticker} EMA calc ({len(close_prices.dropna()) if close_prices is not None else 0} points)."); return None
        ema_8 = close_prices.ewm(span=8, adjust=False).mean(); ema_55 = close_prices.ewm(span=55, adjust=False).mean(); last_ema_8, last_ema_55 = ema_8.iloc[-1], ema_55.iloc[-1]
        if pd.isna(last_ema_8) or pd.isna(last_ema_55) or abs(last_ema_55) < 1e-9: prometheus_logger.warning(f"NaN or zero EMA_55 for {ticker}."); return None
        ema_invest_score = (((last_ema_8 - last_ema_55) / last_ema_55) * 4 + 0.5) * 100
        return float(ema_invest_score)
    except Exception as e: prometheus_logger.warning(f"Context EMA Invest calc failed for {ticker}: {e}"); return None

# --- Helper for Context Enhancement ---
# (Added initialization fix for close_col_tuple)
async def _calculate_perc_changes(ticker: str) -> Dict[str, str]:
    """Fetches 5 years of data using robust helper and calculates % changes."""
    changes = { "1d": "N/A", "1w": "N/A", "1mo": "N/A", "3mo": "N/A", "1y": "N/A", "5y": "N/A" }
    try:
        data = await get_yf_download_robustly( tickers=[ticker], period="5y", interval="1d", auto_adjust=True )
        if data.empty: prometheus_logger.warning(f"No data returned for {ticker} % changes."); return changes
        close_prices = None; price_level_name = 'Price'; ticker_level_name = 'Ticker'; close_col_tuple = None # <<< Initialize here
        if isinstance(data.columns, pd.MultiIndex):
             if ('Close', ticker) in data.columns: close_prices = data[('Close', ticker)]
             elif 'Close' in data.columns.get_level_values(price_level_name): close_col_tuple = next((col for col in data.columns if col[data.columns.names.index(price_level_name)] == 'Close'), None);
             if close_col_tuple: close_prices = data[close_col_tuple]
        elif 'Close' in data.columns: close_prices = data['Close']
        if close_prices is None or close_prices.dropna().empty or len(close_prices.dropna()) < 2: prometheus_logger.warning(f"Insufficient 'Close' data for {ticker} % changes."); return changes
        close_prices = close_prices.dropna(); latest_close = close_prices.iloc[-1]; now_dt = close_prices.index[-1]
        if now_dt.tzinfo is not None: now_dt = now_dt.tz_localize(None)
        periods = { "1d": now_dt - timedelta(days=1), "1w": now_dt - timedelta(weeks=1), "1mo": now_dt - relativedelta(months=1), "3mo": now_dt - relativedelta(months=3), "1y": now_dt - relativedelta(years=1), "5y": now_dt - relativedelta(years=5) }
        past_closes = {}
        for key, past_date in periods.items():
            if close_prices.index.tzinfo is None and past_date.tzinfo is not None: past_date = past_date.tz_localize(None)
            try:
                potential_past_date = close_prices.index[close_prices.index <= past_date]
                if not potential_past_date.empty:
                    actual_past_date = potential_past_date[-1];
                    if actual_past_date < now_dt: past_close_val = close_prices.asof(actual_past_date);
                    if pd.notna(past_close_val): past_closes[key] = past_close_val
                elif key == "5y" and len(close_prices) > 0 and pd.notna(close_prices.iloc[0]): past_closes[key] = close_prices.iloc[0]
            except IndexError:
                 if key == "5y" and len(close_prices) > 0 and pd.notna(close_prices.iloc[0]): past_closes[key] = close_prices.iloc[0]
        latest_close_scalar = latest_close.item() if isinstance(latest_close, (pd.Series, pd.DataFrame)) else latest_close
        for key in periods.keys():
             past_close = past_closes.get(key); past_close_scalar = past_close.item() if isinstance(past_close, (pd.Series, pd.DataFrame)) else past_close
             if isinstance(past_close_scalar, (int, float, np.number)) and isinstance(latest_close_scalar, (int, float, np.number)) and past_close_scalar != 0 and pd.notna(past_close_scalar) and pd.notna(latest_close_scalar):
                 change = ((latest_close_scalar - past_close_scalar) / past_close_scalar) * 100; changes[key] = f"{change:+.2f}%"
    except Exception as e: prometheus_logger.warning(f"Failed calc within % changes for {ticker}: {e}")
    return changes


# --- Prometheus Class ---
class Prometheus:
    def __init__(self, gemini_api_key: Optional[str], toolbox_map: Dict[str, Callable],
                 risk_command_func: Callable, derivative_func: Callable,
                 mlforecast_func: Callable, screener_func: Callable,
                 powerscore_func: Callable, sentiment_func: Callable,
                 fundamentals_func: Callable, quickscore_func: Callable):
        # (Init remains the same)
        prometheus_logger.info("Initializing Prometheus Core...")
        self.db_path = "prometheus_kb.sqlite"; self._initialize_db(); self.toolbox = toolbox_map
        self.risk_command_func = risk_command_func; self.derivative_func = derivative_func; self.mlforecast_func = mlforecast_func
        self.screener_func = screener_func; self.powerscore_func = powerscore_func; self.sentiment_func = sentiment_func
        self.fundamentals_func = fundamentals_func; self.quickscore_func = quickscore_func
        self.gemini_model = None; self.gemini_api_key = gemini_api_key; self.synthesized_commands = set()
        if gemini_api_key and "AIza" in gemini_api_key:
             try: genai.configure(api_key=gemini_api_key); self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite'); prometheus_logger.info("Gemini model OK."); print("   -> Prometheus Core: Gemini model initialized.")
             except Exception as e: prometheus_logger.error(f"Gemini init failed: {e}"); print(f"   -> Prometheus Core: Warn - Gemini init failed: {e}")
        else: prometheus_logger.warning("Gemini API key missing/invalid."); print("   -> Prometheus Core: Warn - Gemini API key missing/invalid.")
        self._load_and_register_synthesized_commands_sync()
        required_funcs = [self.derivative_func, self.mlforecast_func, self.powerscore_func, self.sentiment_func, self.fundamentals_func, self.quickscore_func]
        if all(required_funcs): self.correlation_task = asyncio.create_task(self.background_correlation_analysis()); prometheus_logger.info("BG correlation task started."); print("   -> Prometheus Core: Background correlation task started.")
        else: missing = [f.__name__ for f, func in zip(["deriv", "mlfcst", "pwsc", "sent", "fund", "qscore"], required_funcs) if not func]; self.correlation_task = None; prometheus_logger.warning(f"BG correlation task NOT started (missing: {', '.join(missing)})."); print(f"   -> Prometheus Core: BG correlation task NOT started (missing: {', '.join(missing)}).")

    def _initialize_db(self):
        # (Keep existing implementation)
        prometheus_logger.info(f"Initializing KB (SQLite) at '{self.db_path}'...")
        print("   -> Prometheus Core: Initializing Knowledge Base (SQLite)...")
        try:
            conn = sqlite3.connect(self.db_path); cursor = conn.cursor()
            cursor.execute("""CREATE TABLE IF NOT EXISTS command_log (id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, command TEXT NOT NULL, parameters TEXT, market_context TEXT, output_summary TEXT, success BOOLEAN, duration_ms INTEGER, user_feedback_rating INTEGER, user_feedback_comment TEXT)""")
            cursor.execute("PRAGMA table_info(command_log)"); columns = [info[1] for info in cursor.fetchall()]
            if 'user_feedback_rating' not in columns: cursor.execute("ALTER TABLE command_log ADD COLUMN user_feedback_rating INTEGER")
            if 'user_feedback_comment' not in columns: cursor.execute("ALTER TABLE command_log ADD COLUMN user_feedback_comment TEXT")
            conn.commit(); conn.close(); prometheus_logger.info("KB schema verified."); print("   -> Prometheus Core: Knowledge Base ready.")
        # --- MODIFICATION: Removed emoji ---
        except Exception as e: prometheus_logger.exception(f"ERROR initializing DB: {e}"); print(f"   -> Prometheus Core: [ERROR] initializing DB: {e}")

    def _load_and_register_synthesized_commands_sync(self):
        # (Keep existing implementation)
        prometheus_logger.info(f"Loading synthesized commands sync from '{SYNTHESIZED_WORKFLOWS_FILE}'...")
        print(f"   -> Prometheus Core: Loading synthesized workflows...")
        loaded_count = 0
        try:
            if not os.path.exists(SYNTHESIZED_WORKFLOWS_FILE):
                with open(SYNTHESIZED_WORKFLOWS_FILE, 'w') as f: json.dump({}, f)
                prometheus_logger.info(f"Created empty synthesized file: '{SYNTHESIZED_WORKFLOWS_FILE}'"); print(f"   -> Prometheus Core: Created empty synthesized workflows file."); return
            with open(SYNTHESIZED_WORKFLOWS_FILE, 'r') as f: workflows = json.load(f)
            if not isinstance(workflows, dict): prometheus_logger.warning(f"Workflows file not a dict. Skipping."); print(f"   -> Prometheus Core: Warn - Workflows file format incorrect."); return
            for command_name_with_slash, sequence in workflows.items():
                if isinstance(sequence, list) and command_name_with_slash.startswith('/'):
                    self._create_and_register_workflow_function_sync(sequence, command_name_with_slash)
                    loaded_count += 1
                else: prometheus_logger.warning(f"Invalid sequence/name for '{command_name_with_slash}' in {SYNTHESIZED_WORKFLOWS_FILE}.")
            prometheus_logger.info(f"Loaded/registered {loaded_count} synthesized commands sync.")
            print(f"   -> Prometheus Core: Loaded {loaded_count} synthesized workflows.")
        except FileNotFoundError: pass
        # --- MODIFICATION: Removed emoji ---
        except json.JSONDecodeError: prometheus_logger.error(f"Error decoding JSON {SYNTHESIZED_WORKFLOWS_FILE}."); print(f"   -> Prometheus Core: [ERROR] Bad JSON in workflows file.")
        except Exception as e: prometheus_logger.exception(f"Error loading synthesized workflows sync: {e}"); print(f"   -> Prometheus Core: [ERROR] loading workflows sync: {e}")

    # --- UPDATED: get_market_context with more DEBUG logging ---
    async def get_market_context(self) -> Dict[str, Any]:
        """ Fetches market context including risk scores and % changes with enhanced logging. """
        prometheus_logger.info("Starting context fetch...")
        print("[CONTEXT DEBUG] Starting context fetch...") # <<< DEBUG
        context: Dict[str, Any] = {"vix_price": "N/A", "spy_score": "N/A", "spy_changes": {}, "vix_changes": {}}
        risk_fetch_success = False

        # --- Try Risk Command ---
        if self.risk_command_func:
            original_stdout = sys.stdout; sys.stdout = io.StringIO() # Suppress command output
            try:
                prometheus_logger.debug("Attempting primary context fetch via risk_command_func...")
                print("[CONTEXT DEBUG] Calling risk_command_func...") # <<< DEBUG
                # --- Increased timeout ---
                risk_result_tuple_or_dict = await asyncio.wait_for(
                     self.risk_command_func(args=[], ai_params={"assessment_type": "standard"}, is_called_by_ai=True),
                     timeout=90.0 # <<< Increased timeout to 90s
                )
                prometheus_logger.debug(f"risk_command_func raw result: {risk_result_tuple_or_dict}")
                print(f"[CONTEXT DEBUG] risk_command_func result type: {type(risk_result_tuple_or_dict)}") # <<< DEBUG
                risk_data_dict = {}; raw_data_dict = {}
                if isinstance(risk_result_tuple_or_dict, dict): risk_data_dict = risk_result_tuple_or_dict
                elif isinstance(risk_result_tuple_or_dict, tuple) and len(risk_result_tuple_or_dict) >= 2: risk_data_dict = risk_result_tuple_or_dict[0] if isinstance(risk_result_tuple_or_dict[0], dict) else {}; raw_data_dict = risk_result_tuple_or_dict[1] if isinstance(risk_result_tuple_or_dict[1], dict) else {}
                elif risk_result_tuple_or_dict is None: prometheus_logger.warning("Risk command returned None.")
                elif isinstance(risk_result_tuple_or_dict, str) and "error" in risk_result_tuple_or_dict.lower(): prometheus_logger.warning(f"Risk error: {risk_result_tuple_or_dict}")
                else: prometheus_logger.warning(f"Unexpected risk result: {type(risk_result_tuple_or_dict)}")

                # Try extracting VIX and Score more robustly
                vix_str = raw_data_dict.get("Live VIX Price") # Check raw dict first
                score_str = risk_data_dict.get(next((k for k in risk_data_dict if 'market invest score' in k.lower()), None))
                if vix_str is None: # Fallback to parsed dict
                    vix_key = next((k for k in risk_data_dict if 'vix price' in k.lower()), None)
                    vix_str = risk_data_dict.get(vix_key)

                if vix_str not in ["N/A", None, ""]: context["vix_price"] = str(vix_str).strip().replace('%','')
                if score_str not in ["N/A", None, ""]: context["spy_score"] = str(score_str).strip().replace('%','')

                if context["vix_price"] != "N/A" and context["spy_score"] != "N/A":
                    risk_fetch_success = True
                    prometheus_logger.info(f"Primary risk fetch OK: VIX={context['vix_price']}, Score={context['spy_score']}")
                    print(f"[CONTEXT DEBUG] Primary risk fetch OK: VIX={context['vix_price']}, Score={context['spy_score']}")
                else:
                    prometheus_logger.warning(f"Primary risk fetch partial/failed: VIX={context['vix_price']}, Score={context['spy_score']}")
                    print(f"[CONTEXT DEBUG] Primary risk fetch partial/failed: VIX={context['vix_price']}, Score={context['spy_score']}")
            except asyncio.TimeoutError:
                prometheus_logger.error("Primary risk context fetch timed out (90s)")
                print("[CONTEXT DEBUG] Primary risk context fetch timed out (90s)") # <<< DEBUG Timeout
            except Exception as e:
                prometheus_logger.exception(f"Primary risk context fetch error: {e}")
                print(f"[CONTEXT DEBUG] Primary risk context fetch error: {type(e).__name__} - {e}") # <<< DEBUG Error
            finally:
                sys.stdout = original_stdout # Restore stdout
        else:
            prometheus_logger.warning("No risk_command_func provided for context.")
            print("[CONTEXT DEBUG] No risk_command_func provided.") # <<< DEBUG

        # --- Fallback SPY Score ---
        if context["spy_score"] == "N/A":
            prometheus_logger.info("Attempting fallback SPY INVEST score...")
            print("[CONTEXT DEBUG] Attempting fallback SPY INVEST score...") # <<< DEBUG
            try:
                spy_invest_score = await asyncio.wait_for(calculate_ema_invest_minimal('SPY', 2), timeout=30.0) # <<< Increased timeout
                if spy_invest_score is not None:
                    context["spy_score"] = f"{spy_invest_score:.2f}%"
                    prometheus_logger.info(f"Fallback SPY Score OK: {context['spy_score']}")
                    print(f"[CONTEXT DEBUG] Fallback SPY Score OK: {context['spy_score']}")
                else:
                    prometheus_logger.warning("Fallback SPY Score failed (returned None).")
                    print("[CONTEXT DEBUG] Fallback SPY Score failed (returned None).")
            except asyncio.TimeoutError:
                prometheus_logger.error("Fallback SPY Score timed out (30s).")
                print("[CONTEXT DEBUG] Fallback SPY Score timed out (30s).") # <<< DEBUG Timeout
            except Exception as e_spy:
                prometheus_logger.exception(f"Fallback SPY Score error: {e_spy}")
                print(f"[CONTEXT DEBUG] Fallback SPY Score error: {type(e_spy).__name__} - {e_spy}") # <<< DEBUG Error

        # --- Fallback VIX Price ---
        if context["vix_price"] == "N/A":
            prometheus_logger.info("Attempting fallback VIX price fetch...")
            print("[CONTEXT DEBUG] Attempting fallback VIX price fetch...") # <<< DEBUG
            try:
                # <<< Use auto_adjust=False for VIX fallback to get raw price easier >>>
                vix_data = await asyncio.wait_for(get_yf_download_robustly(tickers=['^VIX'], period="5d", interval="1d", auto_adjust=False), timeout=30.0) # <<< Increased timeout
                prometheus_logger.debug(f"Fallback VIX yf download result (shape): {vix_data.shape if not vix_data.empty else 'Empty'}")
                if not vix_data.empty:
                    close_prices = None; ticker = '^VIX'; price_level_name = 'Price'; close_col_tuple = None
                    # (Robust close price extraction - Added Debug Logs)
                    if isinstance(vix_data.columns, pd.MultiIndex):
                         prometheus_logger.debug("Fallback VIX: MultiIndex detected")
                         # Try Price/Ticker levels first (standardized)
                         if ('Close', ticker) in vix_data.columns: close_prices = vix_data[('Close', ticker)]; prometheus_logger.debug("Fallback VIX: Found ('Close', ticker)")
                         # Fallback if names are different but structure might be Price/Ticker
                         elif 'Close' in vix_data.columns.get_level_values(0):
                             close_col_tuple = next((c for c in vix_data.columns if c[0] == 'Close'), None);
                             if close_col_tuple: close_prices = vix_data[close_col_tuple]; prometheus_logger.debug(f"Fallback VIX: Found tuple {close_col_tuple}")
                             else: prometheus_logger.debug("Fallback VIX: 'Close' in level 0 but tuple not found?")
                         else: prometheus_logger.debug("Fallback VIX: MultiIndex but no 'Close' found.")
                    elif 'Close' in vix_data.columns:
                         close_prices = vix_data['Close']; prometheus_logger.debug("Fallback VIX: Simple DataFrame with 'Close'")
                    else: prometheus_logger.debug("Fallback VIX: No 'Close' column found at all.")

                    if close_prices is not None and not close_prices.dropna().empty:
                         last_price = close_prices.dropna().iloc[-1]
                         context["vix_price"] = f"{last_price:.2f}"; prometheus_logger.info(f"Fallback VIX price OK: {context['vix_price']}"); print(f"[CONTEXT DEBUG] Fallback VIX price OK: {context['vix_price']}")
                    else: prometheus_logger.warning("Fallback VIX: Could not extract valid 'Close' price series."); print("[CONTEXT DEBUG] Fallback VIX: Could not extract valid 'Close' series.")
                else: prometheus_logger.warning("Fallback VIX: Empty data returned from yfinance."); print("[CONTEXT DEBUG] Fallback VIX: Empty data returned from yfinance.")
            except asyncio.TimeoutError:
                prometheus_logger.error("Fallback VIX price timed out (30s).")
                print("[CONTEXT DEBUG] Fallback VIX price timed out (30s).") # <<< DEBUG Timeout
            except Exception as e_vix:
                prometheus_logger.exception(f"Fallback VIX price fetch error: {e_vix}")
                print(f"[CONTEXT DEBUG] Fallback VIX price fetch error: {type(e_vix).__name__} - {e_vix}") # <<< DEBUG Error

        # --- Percentage Changes ---
        # (Keep as is)
        try:
             spy_changes_task = asyncio.wait_for(_calculate_perc_changes('SPY'), timeout=30.0); vix_changes_task = asyncio.wait_for(_calculate_perc_changes('^VIX'), timeout=30.0)
             spy_changes_result, vix_changes_result = await asyncio.gather(spy_changes_task, vix_changes_task, return_exceptions=True)
             if isinstance(spy_changes_result, dict): context["spy_changes"] = spy_changes_result; prometheus_logger.debug(f"SPY % changes fetched.")
             else: prometheus_logger.warning(f"Failed SPY % changes: {spy_changes_result}")
             if isinstance(vix_changes_result, dict): context["vix_changes"] = vix_changes_result; prometheus_logger.debug(f"VIX % changes fetched.")
             else: prometheus_logger.warning(f"Failed VIX % changes: {vix_changes_result}")
        except asyncio.TimeoutError: prometheus_logger.error("ERROR fetching SPY/VIX % changes: Timeout")
        except Exception as e_changes: prometheus_logger.exception(f"ERROR fetching SPY/VIX % changes: {e_changes}")

        prometheus_logger.info(f"Final market context: VIX={context['vix_price']}, Score={context['spy_score']}")
        print(f"[CONTEXT DEBUG] Final context: VIX={context['vix_price']}, Score={context['spy_score']}")
        return context
    # --- END: get_market_context ---

    # --- UPDATED: execute_and_log fixes arg passing and None return ---
    async def execute_and_log(self, command_name_with_slash: str, args: List[str] = None, ai_params: Optional[Dict] = None, called_by_user: bool = False, internal_call: bool = False) -> Any:
        start_time = datetime.now(); command_name = command_name_with_slash.lstrip('/'); context = {}
        if not internal_call: context = await self.get_market_context()
        command_func = self.toolbox.get(command_name); log_id = None
        if not command_func:
            output_summary = f"Unknown command '{command_name_with_slash}'"; prometheus_logger.error(output_summary); print(output_summary)
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            if not internal_call: log_id = self._log_command(start_time, command_name_with_slash, args or ai_params, context, output_summary, success=False, duration_ms=duration_ms)
            return {"status": "error", "message": output_summary}

        parameters_to_log = args if called_by_user and args is not None else ai_params
        context_str_log = "Context N/A (Internal Call)" if internal_call else ", ".join([f"{k}:{str(v)[:20]}{'...' if len(str(v))>20 else ''}" for k,v in context.items()])
        param_str = ' '.join(map(str, args)) if called_by_user and args else json.dumps(ai_params) if ai_params else ""
        log_msg_start = f"Executing: {command_name_with_slash} {param_str} | Context: {context_str_log}"; prometheus_logger.info(log_msg_start)
        is_synthesis_execution = command_name.startswith("synthesized_")
        if called_by_user or is_synthesis_execution: print(f"[Prometheus Log] {log_msg_start}")
        output_summary = f"Execution started."; success_flag = False; result = None
        try:
            kwargs_to_pass = {}
            sig = inspect.signature(command_func)
            func_params = sig.parameters
            expects_args = 'args' in func_params
            expects_ai_params = 'ai_params' in func_params

            # --- *** Corrected Argument Logic *** ---
            if expects_args:
                # If function expects 'args', pass it whether user or internal call
                kwargs_to_pass["args"] = args if called_by_user and args is not None else []
            if expects_ai_params:
                # If function expects 'ai_params', pass it if internal/AI call, otherwise empty dict
                kwargs_to_pass["ai_params"] = ai_params if not called_by_user and ai_params is not None else {}
            if 'is_called_by_ai' in func_params:
                kwargs_to_pass["is_called_by_ai"] = not called_by_user
            # --- *** End Correction *** ---

            # (Dependency injection remains the same)
            if "gemini_model_obj" in func_params and command_name in ["dev", "report", "compare", "powerscore", "sentiment", "reportgeneration"]: kwargs_to_pass["gemini_model_obj"] = self.gemini_model
            if "api_lock_override" in func_params and command_name in ["powerscore", "sentiment"]:
                 try: from main_singularity import GEMINI_API_LOCK; kwargs_to_pass["api_lock_override"] = GEMINI_API_LOCK
                 except ImportError: prometheus_logger.warning(f"Could not import GEMINI_API_LOCK for {command_name}")
            if "screener_func" in func_params and command_name == "dev": kwargs_to_pass["screener_func"] = self.screener_func
            if command_name == "reportgeneration" and "available_functions" in func_params : kwargs_to_pass["available_functions"] = self.toolbox

            prometheus_logger.debug(f"Calling {command_name} with actual kwargs: {kwargs_to_pass}")
            if asyncio.iscoroutinefunction(command_func): result = await command_func(**kwargs_to_pass)
            else: result = await asyncio.to_thread(lambda: command_func(**kwargs_to_pass))
            prometheus_logger.debug(f"Result from {command_name}: {type(result)} - {str(result)[:100]}...")
            success_flag = True
            # --- Refined Result Summarization Logic (remains the same) ---
            if result is None: output_summary = f"{command_name_with_slash} completed (printed output or None)."
            elif isinstance(result, str):
                 if "error" in result.lower() or "failed" in result.lower(): success_flag = False
                 output_summary = result[:1000]
            elif isinstance(result, dict):
                 if result.get('status') == 'error' or 'error' in result: success_flag = False; output_summary = str(result.get('error') or result.get('message', 'Unknown error dict'))[:1000]
                 elif result.get('status') == 'success' or result.get('status') == 'partial_error':
                     # ... (Keep specific summary logic) ...
                     if command_name == "powerscore" and 'powerscore' in result: output_summary = f"PowerScore for {result.get('ticker','N/A')} (S{result.get('sensitivity','?')}) = {result['powerscore']:.2f}. Errors: {result.get('errors')}"
                     elif command_name == "sentiment" and 'sentiment_score_raw' in result: output_summary = f"Sentiment for {result.get('ticker','N/A')}: Score={result['sentiment_score_raw']:.2f}. Summary: {result.get('summary', 'N/A')}"
                     elif command_name == "fundamentals" and 'fundamental_score' in result: output_summary = f"Fundamentals Score for {result.get('ticker','N/A')}: {result['fundamental_score']:.2f}"
                     elif command_name == "risk": output_summary = f"Risk: Combined={result.get('combined_score', 'N/A')}, MktInv={result.get('market_invest_score', 'N/A')}, IVR={result.get('market_ivr', 'N/A')}"
                     elif command_name == "breakout" and 'current_breakout_stocks' in result: stocks = result['current_breakout_stocks']; count = len(stocks); top_ticker = stocks[0]['Ticker'] if count > 0 else 'None'; output_summary = f"Breakout: Found {count} stocks. Top: {top_ticker}."
                     elif command_name == "reportgeneration" and 'filename' in result: output_summary = f"Report Generation: Success. File '{result['filename']}'."
                     elif command_name == "derivative" and 'summary' in result: output_summary = result['summary'][:1000]
                     elif command_name == "quickscore": output_summary = result.get("summary", result.get("message", str(result)))[:1000] # Check dict first
                     elif command_name.startswith("synthesized_") and 'summary' in result: output_summary = result['summary'][:1000]
                     elif 'summary' in result: output_summary = str(result['summary'])[:1000]
                     elif 'message' in result: output_summary = str(result['message'])[:1000]
                     else: output_summary = f"{command_name_with_slash} success (dict)."
                 else: output_summary = f"{command_name_with_slash} completed (dict)."
            elif isinstance(result, tuple):
                 # ... (Keep tuple handling logic) ...
                 if command_name in ["invest", "cultivate"] and len(result) >= 4:
                     holdings_data = result[3] if len(result[3]) > 0 else result[1]; num_holdings = len(holdings_data) if isinstance(holdings_data, list) else 0; cash_val = result[2]
                     output_summary = f"{command_name.capitalize()} done. Holdings: {num_holdings}. Cash: ${cash_val:,.2f}"
                 else: output_summary = f"{command_name_with_slash} completed (tuple len {len(result)})."
            elif isinstance(result, list): output_summary = f"{command_name_with_slash} completed ({len(result)} items)."
            elif isinstance(result, pd.DataFrame): output_summary = f"{command_name_with_slash} completed (DataFrame[{len(result)} rows])."
            else: output_summary = f"{command_name_with_slash} completed (type: {type(result).__name__})."

            if success_flag: prometheus_logger.info(f"Command {command_name_with_slash} finished successfully.")
            else: prometheus_logger.warning(f"Command {command_name_with_slash} finished with error: {output_summary}")
        except Exception as e:
             success_flag = False; output_summary = f"CRITICAL ERROR executing {command_name_with_slash}: {type(e).__name__} - {e}"; prometheus_logger.exception(f"CRITICAL ERROR executing {command_name_with_slash}");
             if called_by_user or is_synthesis_execution: print(f"[Prometheus Log] CRITICAL ERROR: {output_summary}")
             output_summary += f"\nTraceback:\n{traceback.format_exc()}"
             result = {"status": "error", "message": output_summary}
        finally:
             duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
             if not internal_call or is_synthesis_execution:
                 log_id = self._log_command(start_time, command_name_with_slash, parameters_to_log, context, output_summary[:5000], success=success_flag, duration_ms=duration_ms)
                 if called_by_user and log_id is not None: print(f"[Prometheus Action ID: {log_id}]")
                 if success_flag and called_by_user and not internal_call and not is_synthesis_execution and random.random() < 0.1:
                     await self.analyze_workflows()

        if result is None and success_flag: return {"status": "success", "summary": output_summary}
        return result
    # --- END execute_and_log ---

    def _log_command(self, timestamp: datetime, command: str, parameters: Any, context: Dict[str, Any], output_summary: str, success: bool = True, duration_ms: int = 0) -> Optional[int]:
        # (Keep existing implementation)
        params_str = json.dumps(parameters, default=str) if isinstance(parameters, (dict, list)) else str(parameters); context_str = json.dumps(context, default=str)
        log_entry = { "timestamp": timestamp.isoformat(), "command": command, "parameters": params_str, "market_context": context_str, "output_summary": output_summary, "success": success, "duration_ms": duration_ms };
        log_msg = f"Logging: {command} | Success: {success} | Duration: {duration_ms}ms | Summary: {output_summary[:60]}..."; prometheus_logger.info(log_msg)
        conn = None
        try:
            conn = sqlite3.connect(self.db_path); cursor = conn.cursor()
            cursor.execute("""INSERT INTO command_log (timestamp, command, parameters, market_context, output_summary, success, duration_ms) VALUES (:timestamp, :command, :parameters, :market_context, :output_summary, :success, :duration_ms)""", log_entry)
            conn.commit(); last_id = cursor.lastrowid; conn.close()
            return last_id
        # --- MODIFICATION: Removed emoji ---
        except Exception as e:
            prometheus_logger.exception(f"ERROR logging command to DB: {e}"); print(f"   -> Prometheus Core: [ERROR] logging command to DB: {e}")
            if conn: conn.close()
            return None

    # --- UPDATED: analyze_workflows for 2-step sequences ---
    async def analyze_workflows(self):
        prometheus_logger.info("Analyzing command history for potential 2-step workflows...")
        print("[Prometheus Workflow] Analyzing command history for 2-step patterns...")
        conn = sqlite3.connect(self.db_path)
        try:
            query = """
            SELECT c1.command AS command1, c2.command AS command2, COUNT(*) as frequency
            FROM command_log c1 JOIN command_log c2 ON c1.id + 1 = c2.id
            WHERE c1.success = 1 AND c2.success = 1 AND STRFTIME('%s', c2.timestamp) - STRFTIME('%s', c1.timestamp) < 120
            GROUP BY command1, command2 HAVING frequency >= 2 ORDER BY frequency DESC LIMIT 5;
            """
            df_sequences = pd.read_sql_query(query, conn)
            if not df_sequences.empty:
                prometheus_logger.info(f"Potential 2-step workflows detected: {df_sequences.to_dict('records')}")
                print("-> Prometheus Suggestion: Potential 2-step workflows detected:")
                for _, row in df_sequences.iterrows():
                    sequence = [row['command1'], row['command2']]
                    print(f"  - Sequence `{'` -> `'.join(sequence)}` observed {row['frequency']} times.")
                    known_pattern = ['/breakout', '/quickscore'] # Target the 2-step pattern
                    if sequence == known_pattern:
                        cmd_name_with_slash = f"/synthesized_{'_'.join(s.lstrip('/') for s in sequence)}"
                        if cmd_name_with_slash not in self.synthesized_commands:
                            prometheus_logger.info(f"Triggering synthesis for {sequence}")
                            await self._create_and_register_workflow_function(sequence, cmd_name_with_slash)
                        else:
                            prometheus_logger.debug(f"Synthesis skipped for {sequence}, command already exists.")
                            print(f"    (Synthesis skipped, command '{cmd_name_with_slash}' already created)")
                    else:
                        prometheus_logger.debug(f"Skipping synthesis for unsupported 2-step pattern: {sequence}")
                        print(f"    (Skipping synthesis, pattern '{' -> '.join(sequence)}' not yet supported)")
            else:
                 prometheus_logger.info("No frequent 2-step command sequences (>=2) found.")
                 print("[Prometheus Workflow] No frequent (>=2) 2-step command sequences found.")
        # --- MODIFICATION: Removed emoji ---
        except Exception as e:
            prometheus_logger.exception(f"ERROR analyzing 2-step workflows: {e}"); print(f"[Prometheus Workflow] [ERROR] {e}")
        finally: conn.close()
    # --- END: analyze_workflows update ---


    # --- UPDATED: _create_and_register_workflow_function for 2 steps ---
    async def _create_and_register_workflow_function(self, sequence: List[str], command_name_with_slash: str, load_only: bool = False):
        """ Internal helper for the 2-step /breakout -> /quickscore workflow. """
        prometheus_logger.info(f"{'Loading' if load_only else 'Creating'} 2-step workflow function for '{command_name_with_slash}'")
        command_name_no_slash = command_name_with_slash.lstrip('/')
        if command_name_with_slash in self.synthesized_commands: prometheus_logger.debug(f"Workflow '{command_name_with_slash}' already registered."); return

        async def _workflow_executor(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
            print(f"\n--- Running Synthesized Workflow: {command_name_with_slash} ---"); step_summaries = []; top_ticker = None; success = True
            print("  Step 1: Running /breakout..."); prometheus_logger.debug(f"Workflow {command_name_with_slash}: Step 1 - /breakout")
            # --- Pass empty args list explicitly ---
            breakout_result = await self.execute_and_log("/breakout", args=[], called_by_user=False, internal_call=True)
            prometheus_logger.debug(f"Workflow {command_name_with_slash}: Step 1 Result: {breakout_result}") # <<< DEBUG

            # --- Robust Breakout Result Handling ---
            if isinstance(breakout_result, dict) and breakout_result.get("status") == "success":
                stocks = breakout_result.get("current_breakout_stocks", [])
                if stocks and isinstance(stocks, list) and len(stocks) > 0:
                    top_stock_data = stocks[0];
                    if isinstance(top_stock_data, dict):
                        top_ticker = top_stock_data.get('Ticker')
                        if top_ticker: step_summaries.append(f"Breakout found {len(stocks)} stocks, top: {top_ticker}."); print(f"    -> Top breakout stock: {top_ticker}"); prometheus_logger.debug(f"Workflow {command_name_with_slash}: Step 1 OK, top={top_ticker}")
                        else: step_summaries.append("Breakout success, but top ticker missing key."); print("    -> Top breakout stock missing 'Ticker'."); prometheus_logger.warning(f"Workflow {command_name_with_slash}: Step 1 Warn - Missing 'Ticker'.")
                    else: step_summaries.append("Breakout success, but invalid stock data format."); print("    -> Invalid stock data format."); prometheus_logger.warning(f"Workflow {command_name_with_slash}: Step 1 Warn - Invalid format.")
                else: step_summaries.append(breakout_result.get("message", "Breakout success, but found no stocks.")); print(f"    -> {breakout_result.get('message', '/breakout found no stocks.')}"); prometheus_logger.info(f"Workflow {command_name_with_slash}: Step 1 Info - No stocks.")
            else: error_msg = breakout_result.get("message", "Unknown error or non-dict result") if isinstance(breakout_result, dict) else str(breakout_result); step_summaries.append(f"Breakout step failed: {error_msg}"); print(f"    -> /breakout failed: {error_msg[:100]}..."); prometheus_logger.error(f"Workflow {command_name_with_slash}: Step 1 FAILED: {error_msg}"); success = False

            # --- Step 2: Quickscore ---
            if success and top_ticker:
                print(f"  Step 2: Running /quickscore for {top_ticker}..."); prometheus_logger.debug(f"Workflow {command_name_with_slash}: Step 2 - /quickscore {top_ticker}")
                qs_params = {'ticker': top_ticker};
                qs_result = await self.execute_and_log("/quickscore", ai_params=qs_params, called_by_user=False, internal_call=True)
                prometheus_logger.debug(f"Workflow {command_name_with_slash}: Step 2 Result: {qs_result}") # <<< DEBUG
                if isinstance(qs_result, dict) and qs_result.get("status") == "success": summary = qs_result.get("summary", "No summary.").split(". Graphs:")[0]; step_summaries.append(f"Quickscore ({top_ticker}): {summary}."); print(f"    -> {summary}."); prometheus_logger.debug(f"Workflow {command_name_with_slash}: Step 2 OK.")
                else: error_msg = qs_result.get("message", "Failed or non-dict result") if isinstance(qs_result, dict) else str(qs_result); step_summaries.append(f"Quickscore ({top_ticker}): Failed."); print(f"    -> /quickscore failed: {error_msg[:100]}..."); prometheus_logger.warning(f"Workflow {command_name_with_slash}: Step 2 FAILED/Error: {qs_result}")
            elif success: step_summaries.append("Quickscore skipped."); print("  Step 2: Skipped /quickscore."); prometheus_logger.info(f"Workflow {command_name_with_slash}: Step 2 Skipped.")

            # --- Final Summary ---
            final_summary = f"Synthesized workflow '{command_name_with_slash}' completed. Results: {' | '.join(step_summaries)}"; print(f"--- Workflow {command_name_with_slash} Finished ---"); prometheus_logger.info(f"Workflow {command_name_with_slash} Finished.")
            final_result_for_log = {"summary": final_summary, "status": "success" if success else "error"}
            return final_result_for_log

        self.toolbox[command_name_no_slash] = _workflow_executor; self.synthesized_commands.add(command_name_with_slash)
        if not load_only:
            prometheus_logger.info(f"Saving definition for '{command_name_with_slash}'")
            self._save_synthesized_command_definition(command_name_with_slash, sequence)
            # --- MODIFICATION: Removed emoji ---
            print(f"[Prometheus Synthesis] New command '{command_name_with_slash}' created and saved.")
            print(f"   -> Try running: {command_name_with_slash}")
        else: prometheus_logger.info(f"Registered loaded command '{command_name_with_slash}'")

    # --- Synchronous wrapper for loading ---
    def _create_and_register_workflow_function_sync(self, sequence: List[str], command_name_with_slash: str):
        """ Synchronous version for loading during initialization. """
        # (Keep existing implementation - defines async func but registers sync)
        command_name_no_slash = command_name_with_slash.lstrip('/')
        if command_name_with_slash in self.synthesized_commands: return
        prometheus_logger.info(f"Loading workflow function sync for '{command_name_with_slash}'")
        async def _workflow_executor(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
            # (Keep the _workflow_executor logic exactly the same as in the async version, including only 2 steps)
            print(f"\n--- Running Synthesized Workflow: {command_name_with_slash} ---"); step_summaries = []; top_ticker = None; success = True
            print("  Step 1: Running /breakout..."); prometheus_logger.debug(f"Workflow {command_name_with_slash}: Step 1 - /breakout")
            breakout_result = await self.execute_and_log("/breakout", args=[], called_by_user=False, internal_call=True)
            prometheus_logger.debug(f"Workflow {command_name_with_slash}: Step 1 Result: {breakout_result}") # <<< DEBUG
            if isinstance(breakout_result, dict) and breakout_result.get("status") == "success":
                 stocks = breakout_result.get("current_breakout_stocks", [])
                 if stocks and isinstance(stocks, list) and len(stocks) > 0:
                     top_stock_data = stocks[0];
                     if isinstance(top_stock_data, dict):
                         top_ticker = top_stock_data.get('Ticker')
                         if top_ticker: step_summaries.append(f"Breakout found {len(stocks)} stocks, top: {top_ticker}."); print(f"    -> Top breakout stock: {top_ticker}"); prometheus_logger.debug(f"Workflow {command_name_with_slash}: Step 1 OK, top={top_ticker}")
                         else: step_summaries.append("Breakout success, top ticker missing key."); print("    -> Top breakout stock missing 'Ticker'."); prometheus_logger.warning(f"Workflow {command_name_with_slash}: Step 1 Warn - Missing 'Ticker'.")
                     else: step_summaries.append("Breakout success, invalid stock data format."); print("    -> Invalid stock data format."); prometheus_logger.warning(f"Workflow {command_name_with_slash}: Step 1 Warn - Invalid format.")
                 else: step_summaries.append(breakout_result.get("message", "Breakout success, but found no stocks.")); print(f"    -> {breakout_result.get('message', '/breakout found no stocks.')}"); prometheus_logger.info(f"Workflow {command_name_with_slash}: Step 1 Info - No stocks.")
            else: error_msg = breakout_result.get("message", "Unknown error or non-dict result") if isinstance(breakout_result, dict) else str(breakout_result); step_summaries.append(f"Breakout step failed: {error_msg}"); print(f"    -> /breakout failed: {error_msg[:100]}..."); prometheus_logger.error(f"Workflow {command_name_with_slash}: Step 1 FAILED: {error_msg}"); success = False
            if success and top_ticker:
                 print(f"  Step 2: Running /quickscore for {top_ticker}..."); prometheus_logger.debug(f"Workflow {command_name_with_slash}: Step 2 - /quickscore {top_ticker}")
                 qs_params = {'ticker': top_ticker}; qs_result = await self.execute_and_log("/quickscore", ai_params=qs_params, called_by_user=False, internal_call=True)
                 prometheus_logger.debug(f"Workflow {command_name_with_slash}: Step 2 Result: {qs_result}") # <<< DEBUG
                 if isinstance(qs_result, dict) and qs_result.get("status") == "success": summary = qs_result.get("summary", "No summary.").split(". Graphs:")[0]; step_summaries.append(f"Quickscore ({top_ticker}): {summary}."); print(f"    -> {summary}."); prometheus_logger.debug(f"Workflow {command_name_with_slash}: Step 2 OK.")
                 else: error_msg = qs_result.get("message", "Failed or non-dict result") if isinstance(qs_result, dict) else str(qs_result); step_summaries.append(f"Quickscore ({top_ticker}): Failed."); print(f"    -> /quickscore failed: {error_msg[:100]}..."); prometheus_logger.warning(f"Workflow {command_name_with_slash}: Step 2 FAILED/Error: {qs_result}")
            elif success: step_summaries.append("Quickscore skipped."); print("  Step 2: Skipped /quickscore."); prometheus_logger.info(f"Workflow {command_name_with_slash}: Step 2 Skipped.")
            final_summary = f"Synthesized workflow '{command_name_with_slash}' completed. Results: {' | '.join(step_summaries)}"; print(f"--- Workflow {command_name_with_slash} Finished ---"); prometheus_logger.info(f"Workflow {command_name_with_slash} Finished.")
            final_result_for_log = {"summary": final_summary, "status": "success" if success else "error"}
            return final_result_for_log
        self.toolbox[command_name_no_slash] = _workflow_executor
        self.synthesized_commands.add(command_name_with_slash)
        prometheus_logger.info(f"Registered loaded command '{command_name_with_slash}' sync.")


    def _save_synthesized_command_definition(self, command_name_with_slash: str, sequence: List[str]):
        # (Keep existing implementation)
        try:
            workflows = {}
            if os.path.exists(SYNTHESIZED_WORKFLOWS_FILE):
                with open(SYNTHESIZED_WORKFLOWS_FILE, 'r') as f:
                    try: workflows = json.load(f)
                    except json.JSONDecodeError: prometheus_logger.error(f"Error reading {SYNTHESIZED_WORKFLOWS_FILE}, overwriting."); workflows = {}
            workflows[command_name_with_slash] = sequence
            with open(SYNTHESIZED_WORKFLOWS_FILE, 'w') as f: json.dump(workflows, f, indent=4)
            prometheus_logger.info(f"Saved/Updated {command_name_with_slash} in {SYNTHESIZED_WORKFLOWS_FILE}")
        # --- MODIFICATION: Removed emoji ---
        except Exception as e:
            prometheus_logger.exception(f"Error saving definition for {command_name_with_slash}: {e}"); print(f"   -> Prometheus Synthesis: [ERROR] saving workflow: {e}")

    # --- Background Correlation Analysis ---
    # (background_correlation_analysis remains the same)
    async def background_correlation_analysis(self):
        try: from main_singularity import get_sp500_symbols_singularity
        except ImportError: prometheus_logger.error("Failed import get_sp500_symbols_singularity."); return
        commands_to_correlate = { 'derivative': {'func': self.derivative_func, 'args': [], 'ai_params': {}, 'period': '1y', 'value_key': 'second_derivative_at_end'}, 'mlforecast': {'func': self.mlforecast_func, 'args': [], 'ai_params': {}, 'period': '5-Day', 'value_key': 'Est. % Change'}, 'powerscore': {'func': self.powerscore_func, 'args': [], 'ai_params': {'sensitivity': 2}, 'value_key': 'powerscore'}, 'sentiment': {'func': self.sentiment_func, 'args': [], 'ai_params': {}, 'value_key': 'sentiment_score_raw'}, 'fundamentals': {'func': self.fundamentals_func, 'args': [], 'ai_params': {}, 'value_key': 'fundamental_score'}, 'quickscore': {'func': self.quickscore_func, 'args': [], 'ai_params': {'ema_interval': 2}, 'value_key': 'score'} }
        valid_commands_to_run = { cmd: config for cmd, config in commands_to_correlate.items() if config['func'] is not None };
        if not valid_commands_to_run: prometheus_logger.error("BG Corr: No valid functions."); print("[Prometheus Background] ERROR: No valid functions."); return
        prometheus_logger.info(f"BG Corr: Will analyze: {list(valid_commands_to_run.keys())}")
        while True:
             wait_hours = 6; prometheus_logger.info(f"BG Corr: Waiting {wait_hours} hours."); print(f"\n[Prometheus Background] Next correlation check in ~{wait_hours} hours...")
             await asyncio.sleep(3600 * wait_hours); cycle_start_time = datetime.now(); prometheus_logger.info("Starting BG correlation cycle."); print(f"\n[Prometheus Background] Starting cycle @ {cycle_start_time.strftime('%H:%M:%S')}...")
             try:
                 sp500_tickers = await asyncio.to_thread(get_sp500_symbols_singularity);
                 if not sp500_tickers: prometheus_logger.warning("BG Corr: Failed S&P500 fetch."); continue
                 subset_size = min(len(sp500_tickers), 20); subset_tickers = random.sample(sp500_tickers, subset_size); prometheus_logger.info(f"BG Corr: Analyzing {len(subset_tickers)} tickers: {subset_tickers}"); print(f"[Prometheus Background] Analyzing {len(subset_tickers)} tickers...")
                 all_results_data = {cmd: {} for cmd in valid_commands_to_run}; tasks_by_command = {cmd: [] for cmd in valid_commands_to_run}; tickers_by_command_task = {cmd: [] for cmd in valid_commands_to_run};
                 semaphore = asyncio.Semaphore(5); total_tasks = len(subset_tickers) * len(valid_commands_to_run); completed_tasks = 0
                 for cmd, config in valid_commands_to_run.items():
                     func = config['func']
                     async def run_single_command(ticker, cmd_name, cmd_config):
                         nonlocal completed_tasks
                         async with semaphore:
                             try:
                                 params = cmd_config['ai_params'].copy(); params['ticker'] = ticker; kwargs_exec = {'ai_params': params, 'is_called_by_ai': True}
                                 if cmd_name == 'quickscore': result = await func(ticker=ticker, ema_interval=params.get('ema_interval', 2), is_called_by_ai=True)
                                 else:
                                     import inspect; sig = inspect.signature(func)
                                     if "gemini_model_obj" in sig.parameters: kwargs_exec["gemini_model_obj"] = self.gemini_model
                                     if "api_lock_override" in sig.parameters:
                                          try: from main_singularity import GEMINI_API_LOCK; kwargs_exec["api_lock_override"] = GEMINI_API_LOCK
                                          except ImportError: pass
                                     result = await func(**kwargs_exec)
                                 completed_tasks += 1
                                 if completed_tasks % 5 == 0 or completed_tasks == total_tasks: print(f"\r[Prometheus Background] Progress: {completed_tasks}/{total_tasks} calls...", end="")
                                 return ticker, result
                             except Exception as e:
                                 prometheus_logger.warning(f"BG {cmd_name} task failed {ticker}: {type(e).__name__} - {e}"); completed_tasks += 1
                                 if completed_tasks % 5 == 0 or completed_tasks == total_tasks: print(f"\r[Prometheus Background] Progress: {completed_tasks}/{total_tasks} calls...", end="")
                                 return ticker, e
                     for ticker in subset_tickers: task = asyncio.create_task(run_single_command(ticker, cmd, config)); tasks_by_command[cmd].append(task); tickers_by_command_task[cmd].append(ticker)
                 for cmd, tasks in tasks_by_command.items():
                     if not tasks: continue; raw_cmd_results = await asyncio.gather(*tasks); config = valid_commands_to_run[cmd]; value_key = config['value_key']
                     for i, (ticker, result) in enumerate(raw_cmd_results):
                         if isinstance(result, Exception): continue; extracted_value = None
                         try:
                             # ... (result extraction logic remains the same) ...
                             if cmd == 'derivative' and isinstance(result, dict) and config.get('period') in result.get('periods', {}): period_data = result['periods'][config['period']]; extracted_value = period_data.get(value_key) if period_data.get('status') == 'success' else None
                             elif cmd == 'mlforecast' and isinstance(result, list) and result:
                                 for forecast in result:
                                     if forecast.get("Period") == config.get('period'): val_str = forecast.get(value_key, "0%").replace('%', ''); extracted_value = float(val_str); break
                             elif cmd == 'quickscore' and isinstance(result, tuple) and len(result) == 2: extracted_value = result[1]
                             elif isinstance(result, dict) and value_key in result: extracted_value = result[value_key]
                             if extracted_value is not None: all_results_data[cmd][ticker] = float(extracted_value)
                         except (ValueError, TypeError, KeyError, IndexError) as e_extract: prometheus_logger.warning(f"BG {cmd}: Extract error {ticker}. Err: {e_extract}. Res: {str(result)[:100]}...")
                 df_corr = pd.DataFrame(all_results_data).dropna(); print("\r" + " " * 80 + "\r", end="")
                 if len(df_corr) >= 5:
                     try:
                         correlation_matrix = df_corr.corr(method='pearson'); print("[Prometheus Background] Cross-Tool Correlation Matrix:"); print(correlation_matrix.to_string(float_format="%.3f")); prometheus_logger.info(f"Corr matrix ({len(df_corr)} stocks):\n{correlation_matrix.to_string(float_format='%.3f')}")
                         strong_correlations = correlation_matrix.unstack().sort_values(ascending=False).drop_duplicates(); strong_correlations = strong_correlations[abs(strong_correlations) > 0.5]; strong_correlations = strong_correlations[strong_correlations < 1.0]
                         if not strong_correlations.empty: print("[Prometheus Background] Potential Strong Correlations (>0.5):"); print(strong_correlations.to_string(float_format="%.3f")); prometheus_logger.info(f"Strong correlations:\n{strong_correlations.to_string(float_format='%.3f')}")
                         else: print("[Prometheus Background] No strong correlations (>0.5) found."); prometheus_logger.info("No strong correlations (>0.5) found.")
                     except Exception as ce: prometheus_logger.exception(f"BG corr calc error: {ce}"); print(f"[Prometheus Background] Corr calc error: {ce}")
                 else: print(f"[Prometheus Background] Not enough common data ({len(df_corr)}) for matrix."); prometheus_logger.warning(f"BG Corr: Not enough common data ({len(df_corr)}).")
             except asyncio.CancelledError: prometheus_logger.info("BG correlation task cancelled."); break
             except Exception as e: prometheus_logger.exception(f"ERROR BG correlation cycle: {e}"); print(f"[Prometheus Background] Cycle ERROR: {e}")
             finally: cycle_end_time = datetime.now(); duration = cycle_end_time - cycle_start_time; prometheus_logger.info(f"BG cycle finished. Duration: {duration}"); print(f"[Prometheus Background] Cycle finished @ {cycle_end_time.strftime('%H:%M:%S')} (Duration: {duration}).")


    async def start_interactive_session(self):
        # (Keep existing implementation)
        print("\n--- Prometheus Meta-AI Shell ---"); print("Available commands: analyze patterns, check correlations, query log <limit>, exit"); prometheus_logger.info("Entered Prometheus interactive shell.")
        while True:
            try:
                user_input = await asyncio.to_thread(input, "Prometheus> "); user_input_lower = user_input.lower().strip(); parts = user_input.split(); cmd = parts[0].lower() if parts else ""
                if cmd == 'exit': prometheus_logger.info("Exiting Prometheus shell."); break
                elif cmd == "analyze" and len(parts)>1 and parts[1].lower() == "patterns": await self.analyze_workflows()
                elif cmd == "check" and len(parts)>1 and parts[1].lower() == "correlations":
                     print("Triggering background correlation analysis manually..."); required_funcs = [self.derivative_func, self.mlforecast_func, self.powerscore_func, self.sentiment_func, self.fundamentals_func, self.quickscore_func]; can_run_corr = all(required_funcs)
                     if can_run_corr and (not self.correlation_task or self.correlation_task.done()): self.correlation_task = asyncio.create_task(self.background_correlation_analysis()); print("   -> Correlation task started.")
                     elif self.correlation_task and not self.correlation_task.done(): print("   -> Correlation task is already running.")
                     else: print("   -> Cannot run correlation analysis - required functions missing.")
                elif cmd == "query" and len(parts)>1 and parts[1].lower() == "log": limit = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 10; await self._query_log_db(limit)
                else: print("Unknown command. Available: analyze patterns, check correlations, query log <limit>, exit")
            except EOFError: prometheus_logger.warning("EOF received, exiting Prometheus shell."); break
            except Exception as e: prometheus_logger.exception(f"Error in Prometheus shell: {e}"); print(f"Error: {e}")
        print("Returning to M.I.C. Singularity main shell.")

    async def _query_log_db(self, limit: int = 10):
         # (Keep existing implementation)
         print(f"\n--- Recent Command Logs (Last {limit}) ---")
         try:
             conn = sqlite3.connect(self.db_path); conn.row_factory = sqlite3.Row; cursor = conn.cursor(); cursor.execute("SELECT id, timestamp, command, parameters, success, duration_ms, output_summary FROM command_log ORDER BY id DESC LIMIT ?", (limit,)); rows = cursor.fetchall(); conn.close()
             if not rows: print("No logs found."); return
             log_data = []; headers = ["ID", "Timestamp", "Success", "Duration", "Command", "Parameters", "Summary"]
             for row in reversed(rows):
                 ts = datetime.fromisoformat(row['timestamp']).strftime('%H:%M:%S');
                 # --- MODIFICATION: Removed emojis ---
                 success_str = "OK" if row['success'] else "FAIL"; params_str = "<err>"
                 try:
                     params_data = json.loads(row['parameters'])
                     if isinstance(params_data, list): params_str = " ".join(map(str, params_data))
                     elif isinstance(params_data, dict): params_str = json.dumps(params_data, separators=(',', ':'))
                     else: params_str = str(params_data)
                 except (json.JSONDecodeError, TypeError): params_str = row['parameters'] if row['parameters'] else ""
                 params_str_trunc = params_str[:30] + ('...' if len(params_str) > 30 else '')
                 summary_str_trunc = row['output_summary'].replace('\n', ' ')[:50] + ('...' if len(row['output_summary']) > 50 else '')
                 log_data.append([row['id'], ts, success_str, f"{row['duration_ms']}ms", row['command'], params_str_trunc, summary_str_trunc])
             print(tabulate(log_data, headers=headers, tablefmt="grid"))
         except Exception as e: prometheus_logger.exception(f"Error querying log db: {e}"); print(f"Error: {e}")
