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
import numpy as np # Import numpy

# --- Prometheus Core Logger ---
prometheus_logger = logging.getLogger('PROMETHEUS_CORE')
prometheus_logger.setLevel(logging.INFO)
prometheus_logger.propagate = False
if not prometheus_logger.hasHandlers():
    prometheus_log_file = 'prometheus_core.log'
    prometheus_file_handler = logging.FileHandler(prometheus_log_file)
    prometheus_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    prometheus_file_handler.setFormatter(prometheus_formatter)
    prometheus_logger.addHandler(prometheus_file_handler)

# --- Robust YFinance Download Helper ---
async def get_yf_download_robustly(tickers: list, **kwargs) -> pd.DataFrame:
    """ Robust wrapper for yf.download with retry logic and standardization. """
    max_retries = 2
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(random.uniform(0.3, 0.8))
            kwargs.setdefault('progress', False)
            kwargs.setdefault('timeout', 15)
            # auto_adjust might be passed, let it through but default is False
            kwargs.setdefault('auto_adjust', False)

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
            if data.isnull().all().all(): raise IOError(f"yf.download returned DataFrame with all NaN data for {tickers} (attempt {attempt+1})")

            # --- Standardize columns ---
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
                 except Exception as e_reformat:
                      prometheus_logger.warning(f"Could not standardize MultiIndex names: {data.columns.names}. Error: {e_reformat}")

            return data

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            if attempt < max_retries - 1:
                delay = (attempt + 1) * 1
                # Prometheus logger might not be fully configured yet during early calls
                # print(f"   [WARN Download] yf.download failed ({error_type}, Attempt {attempt+1}/{max_retries}) for {tickers}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                prometheus_logger.error(f"All yf download attempts failed for {tickers}. Last error ({error_type}): {error_msg}")
                # print(f"   [ERROR Download] All yf download attempts failed for {tickers}. Last error ({error_type}): {error_msg}")
                return pd.DataFrame()
    return pd.DataFrame()


# --- Minimal calculate_ema_invest for context fetching ---
# (Copied from isolated test - relies on get_yf_download_robustly defined above)
async def calculate_ema_invest_minimal(ticker: str, ema_interval: int = 2) -> Optional[float]:
    """ Minimal version to get INVEST score for context. """
    interval_map = {1: "1wk", 2: "1d", 3: "1h"}
    period_map = {1: "max", 2: "10y", 3: "2y"}
    try:
        data = await get_yf_download_robustly(
            tickers=[ticker],
            period=period_map.get(ema_interval, "10y"),
            interval=interval_map.get(ema_interval, "1d"),
            auto_adjust=True # OK for EMA calc, use robust downloader
        )
        if data.empty: return None

        # Handle potential MultiIndex returned by robust downloader
        close_prices = None
        price_level_name = 'Price' # From standardization
        ticker_level_name = 'Ticker' # From standardization

        if isinstance(data.columns, pd.MultiIndex):
             # Try standard ('Close', ticker) structure assuming standardization worked
             if ('Close', ticker) in data.columns: close_prices = data[('Close', ticker)]
             # Fallback if standardization failed or structure is different
             elif 'Close' in data.columns.get_level_values(price_level_name):
                 # Select the first column named 'Close' at the correct level
                 close_col_tuple = next((col for col in data.columns if col[data.columns.names.index(price_level_name)] == 'Close'), None)
                 if close_col_tuple: close_prices = data[close_col_tuple]

        elif 'Close' in data.columns: # Standard DataFrame
             close_prices = data['Close']

        if close_prices is None or close_prices.isnull().all():
             prometheus_logger.warning(f"Could not extract 'Close' prices for {ticker} in EMA calc.")
             return None

        if len(close_prices.dropna()) < 55:
            prometheus_logger.warning(f"Insufficient data points ({len(close_prices.dropna())}) for {ticker} EMA calc.")
            return None # Need at least 55 for reliable EMA_55

        ema_8 = close_prices.ewm(span=8, adjust=False).mean()
        ema_55 = close_prices.ewm(span=55, adjust=False).mean()

        last_ema_8 = ema_8.iloc[-1]
        last_ema_55 = ema_55.iloc[-1]

        if pd.isna(last_ema_8) or pd.isna(last_ema_55) or abs(last_ema_55) < 1e-9: # Check for near-zero denominator
            prometheus_logger.warning(f"NaN or zero EMA_55 for {ticker}.")
            return None

        ema_invest_score = (((last_ema_8 - last_ema_55) / last_ema_55) * 4 + 0.5) * 100
        # Return raw score, potentially very large or small
        return float(ema_invest_score)
    except Exception as e:
        prometheus_logger.warning(f"Context EMA Invest calc failed for {ticker}: {e}")
        return None

# --- Helper for Context Enhancement ---
async def _calculate_perc_changes(ticker: str) -> Dict[str, str]:
    """Fetches 5 years of data using robust helper and calculates % changes."""
    # (Keep implementation from previous version - uses get_yf_download_robustly)
    changes = { "1d": "N/A", "1w": "N/A", "1mo": "N/A", "3mo": "N/A", "1y": "N/A", "5y": "N/A" }
    try:
        data = await get_yf_download_robustly(
            tickers=[ticker], period="5y", interval="1d", auto_adjust=True
        )

        # Handle potentially empty or malformed DataFrame
        if data.empty:
            prometheus_logger.warning(f"No data returned for {ticker} % changes.")
            return changes

        # --- Robust Close Price Extraction (Handles MultiIndex) ---
        close_prices = None
        price_level_name = 'Price' # Assumes standardization worked
        ticker_level_name = 'Ticker' # Assumes standardization worked

        if isinstance(data.columns, pd.MultiIndex):
             # Try standard ('Close', ticker) structure
             if ('Close', ticker) in data.columns: close_prices = data[('Close', ticker)]
             # Fallback if structure is different (e.g., auto_adjust=True might return flat 'Close')
             elif 'Close' in data.columns.get_level_values(price_level_name):
                 close_col_tuple = next((col for col in data.columns if col[data.columns.names.index(price_level_name)] == 'Close'), None)
                 if close_col_tuple: close_prices = data[close_col_tuple]
        elif 'Close' in data.columns: # Standard DataFrame
             close_prices = data['Close']

        if close_prices is None or close_prices.dropna().empty or len(close_prices.dropna()) < 2:
            prometheus_logger.warning(f"Insufficient or invalid 'Close' data for {ticker} % changes.")
            return changes
        # --- End Robust Extraction ---

        close_prices = close_prices.dropna()
        latest_close = close_prices.iloc[-1]
        now_dt = close_prices.index[-1]
        if now_dt.tzinfo is not None: now_dt = now_dt.tz_localize(None)

        periods = { "1d": now_dt - timedelta(days=1), "1w": now_dt - timedelta(weeks=1),
                    "1mo": now_dt - relativedelta(months=1), "3mo": now_dt - relativedelta(months=3),
                    "1y": now_dt - relativedelta(years=1), "5y": now_dt - relativedelta(years=5) }
        past_closes = {}
        for key, past_date in periods.items():
            if close_prices.index.tzinfo is None and past_date.tzinfo is not None: past_date = past_date.tz_localize(None)
            try:
                potential_past_date = close_prices.index[close_prices.index <= past_date]
                if not potential_past_date.empty:
                    actual_past_date = potential_past_date[-1]
                    if actual_past_date < now_dt:
                        past_close_val = close_prices.asof(actual_past_date)
                        if pd.notna(past_close_val): past_closes[key] = past_close_val
                elif key == "5y" and len(close_prices) > 0 and pd.notna(close_prices.iloc[0]):
                    past_closes[key] = close_prices.iloc[0]
            except IndexError:
                 if key == "5y" and len(close_prices) > 0 and pd.notna(close_prices.iloc[0]): past_closes[key] = close_prices.iloc[0]

        # Ensure latest_close is scalar
        latest_close_scalar = latest_close.item() if isinstance(latest_close, (pd.Series, pd.DataFrame)) else latest_close

        for key in periods.keys():
             past_close = past_closes.get(key)
             past_close_scalar = past_close.item() if isinstance(past_close, (pd.Series, pd.DataFrame)) else past_close

             if isinstance(past_close_scalar, (int, float, np.number)) and \
                isinstance(latest_close_scalar, (int, float, np.number)) and \
                past_close_scalar != 0 and pd.notna(past_close_scalar) and pd.notna(latest_close_scalar):
                 change = ((latest_close_scalar - past_close_scalar) / past_close_scalar) * 100
                 changes[key] = f"{change:+.2f}%"

    except Exception as e:
        prometheus_logger.warning(f"Failed calculation within % changes for {ticker}: {e}")
    return changes


class Prometheus:
    def __init__(self, gemini_api_key: Optional[str], toolbox_map: Dict[str, Callable],
                 risk_command_func: Callable, derivative_func: Callable,
                 mlforecast_func: Callable, screener_func: Callable):
        # (Keep __init__ implementation from previous version)
        prometheus_logger.info("Initializing Prometheus Core...")
        self.db_path = "prometheus_kb.sqlite"
        self._initialize_db()
        self.toolbox = toolbox_map
        self.risk_command_func = risk_command_func
        self.derivative_func = derivative_func
        self.mlforecast_func = mlforecast_func
        self.screener_func = screener_func
        self.gemini_model = None
        self.gemini_api_key = gemini_api_key

        if gemini_api_key and "AIza" in gemini_api_key:
             try:
                 genai.configure(api_key=gemini_api_key)
                 self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite') # Corrected model
                 prometheus_logger.info("Gemini model initialized successfully.")
                 print("   -> Prometheus Core: Gemini model initialized.")
             except Exception as e:
                 prometheus_logger.error(f"Gemini model initialization failed: {e}")
                 print(f"   -> Prometheus Core: Warning - Gemini model init failed: {e}")
        else:
             prometheus_logger.warning("Gemini API key missing or invalid.")
             print("   -> Prometheus Core: Warning - Gemini API key missing or invalid.")

        if self.derivative_func and self.mlforecast_func:
             self.correlation_task = asyncio.create_task(self.background_correlation_analysis())
             prometheus_logger.info("Background correlation task started.")
             print("   -> Prometheus Core: Background correlation task started.")
        else:
             self.correlation_task = None
             prometheus_logger.warning("Background correlation task NOT started (required functions missing).")
             print("   -> Prometheus Core: Background correlation task NOT started (missing functions).")


    def _initialize_db(self):
        # (Keep existing implementation)
        prometheus_logger.info(f"Initializing Knowledge Base (SQLite) at '{self.db_path}'...")
        print("   -> Prometheus Core: Initializing Knowledge Base (SQLite)...")
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS command_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    command TEXT NOT NULL,
                    parameters TEXT,
                    market_context TEXT,
                    output_summary TEXT,
                    success BOOLEAN,
                    duration_ms INTEGER
                )
            """)
            conn.commit()
            conn.close()
            prometheus_logger.info("Knowledge Base schema verified/created.")
            print("   -> Prometheus Core: Knowledge Base ready.")
        except Exception as e:
            prometheus_logger.exception(f"ERROR initializing database: {e}")
            print(f"   -> Prometheus Core: ❌ ERROR initializing database: {e}")

    async def get_market_context(self) -> Dict[str, Any]:
        """ Fetches market context including risk scores and % changes. """
        prometheus_logger.info("Fetching market context...")
        print("   -> Prometheus Core: Fetching market context...")
        context: Dict[str, Any] = {
            "vix_price": "N/A", "spy_score": "N/A",
            "spy_changes": {}, "vix_changes": {}
        }

        # --- Step 1: Fetch Risk Scores & VIX Price ---
        if self.risk_command_func:
            original_stdout = sys.stdout
            sys.stdout = io.StringIO()
            try:
                risk_result_tuple = await asyncio.wait_for(
                     self.risk_command_func(args=[], ai_params={"assessment_type": "standard"}, is_called_by_ai=True),
                     timeout=90.0
                )
                if risk_result_tuple is None:
                     prometheus_logger.warning("Risk command returned None during context fetch.")
                     print("   -> Prometheus Core: Warning - Risk command returned None.")
                elif isinstance(risk_result_tuple, str) and "error" in risk_result_tuple.lower():
                     prometheus_logger.warning(f"Risk command returned error during context fetch: {risk_result_tuple}")
                     print(f"   -> Prometheus Core: Warning - Risk command failed: {risk_result_tuple}")
                elif isinstance(risk_result_tuple, tuple) and len(risk_result_tuple) >= 2:
                    risk_data_dict = risk_result_tuple[0] if isinstance(risk_result_tuple[0], dict) else {}
                    raw_data_dict = risk_result_tuple[1] if isinstance(risk_result_tuple[1], dict) else {}
                    vix_str = raw_data_dict.get("Live VIX Price", "N/A")
                    # Use market_invest_score which should be the capped 0-100 value
                    score_str = risk_data_dict.get("market_invest_score", "N/A")

                    if vix_str not in ["N/A", None, ""]: context["vix_price"] = str(vix_str).strip().replace('%','')
                    if score_str not in ["N/A", None, ""]: context["spy_score"] = str(score_str).strip().replace('%','') # Keep % if present

                    prometheus_logger.info(f"Risk scores fetched: VIX={context['vix_price']}, Score={context['spy_score']}")
                else:
                     prometheus_logger.warning(f"Unexpected result format from risk command: {type(risk_result_tuple)}")
                     print(f"   -> Prometheus Core: Warning - Unexpected result format from risk command.")
            except asyncio.TimeoutError:
                 prometheus_logger.error("ERROR getting risk scores for market context: Timeout")
                 print("   -> Prometheus Core: ❌ ERROR fetching risk scores: Timed out.")
            except Exception as e:
                prometheus_logger.exception(f"ERROR getting risk scores for market context: {e}")
                print(f"   -> Prometheus Core: ❌ ERROR fetching risk scores: {e}")
            finally:
                sys.stdout = original_stdout
        else:
             prometheus_logger.warning("Cannot fetch risk scores: risk_command_func not provided.")

        # --- **Recalculate SPY Score using minimal function if risk failed** ---
        if context["spy_score"] == "N/A":
            prometheus_logger.info("Attempting fallback SPY INVEST score calculation...")
            print("   -> Prometheus Core: Attempting fallback SPY INVEST score calculation...")
            spy_invest_score = await calculate_ema_invest_minimal('SPY', 2)
            if spy_invest_score is not None:
                context["spy_score"] = f"{spy_invest_score:.2f}%"
                prometheus_logger.info(f"Fallback SPY Score calculated: {context['spy_score']}")
                print(f"   -> Prometheus Core: Fallback SPY Score: {context['spy_score']}")
            else:
                 prometheus_logger.warning("Fallback SPY INVEST score calculation also failed.")
                 print("   -> Prometheus Core: ❌ Fallback SPY Score failed.")
        
        # --- **FIX: Add fallback for VIX price if risk failed** ---
        if context["vix_price"] == "N/A":
            prometheus_logger.info("Attempting fallback VIX price fetch...")
            print("   -> Prometheus Core: Attempting fallback VIX price fetch...")
            try:
                vix_data = await get_yf_download_robustly(
                    tickers=['^VIX'], period="5d", interval="1d", auto_adjust=True
                )
                if not vix_data.empty:
                    # Robustly get the last 'Close' price
                    close_prices = None
                    ticker = '^VIX'
                    price_level_name = 'Price'
                    if isinstance(vix_data.columns, pd.MultiIndex):
                         if ('Close', ticker) in vix_data.columns: close_prices = vix_data[('Close', ticker)]
                         elif 'Close' in vix_data.columns.get_level_values(price_level_name):
                             close_col_tuple = next((col for col in vix_data.columns if col[vix_data.columns.names.index(price_level_name)] == 'Close'), None)
                             if close_col_tuple: close_prices = vix_data[close_col_tuple]
                    elif 'Close' in vix_data.columns:
                         close_prices = vix_data['Close']
                    
                    if close_prices is not None and not close_prices.dropna().empty:
                        last_vix_price = close_prices.dropna().iloc[-1]
                        context["vix_price"] = f"{last_vix_price:.2f}"
                        prometheus_logger.info(f"Fallback VIX price fetched: {context['vix_price']}")
                        print(f"   -> Prometheus Core: Fallback VIX price: {context['vix_price']}")
                    else:
                        raise ValueError("Could not extract 'Close' price from VIX data.")
                else:
                    raise ValueError("Empty data returned for ^VIX.")
            except Exception as e_vix:
                prometheus_logger.warning(f"Fallback VIX price fetch also failed: {e_vix}")
                print(f"   -> Prometheus Core: ❌ Fallback VIX price fetch failed: {e_vix}")
        # --- **END FIX** ---

        # --- Step 2: Fetch Percentage Changes ---
        try:
             spy_changes_task = asyncio.wait_for(_calculate_perc_changes('SPY'), timeout=30.0)
             vix_changes_task = asyncio.wait_for(_calculate_perc_changes('^VIX'), timeout=30.0)
             spy_changes_result, vix_changes_result = await asyncio.gather(
                 spy_changes_task, vix_changes_task, return_exceptions=True
             )
             if isinstance(spy_changes_result, dict): context["spy_changes"] = spy_changes_result; prometheus_logger.info(f"SPY % changes fetched.")
             else: prometheus_logger.warning(f"Failed to fetch SPY % changes: {spy_changes_result}")
             if isinstance(vix_changes_result, dict): context["vix_changes"] = vix_changes_result; prometheus_logger.info(f"VIX % changes fetched.")
             else: prometheus_logger.warning(f"Failed to fetch VIX % changes: {vix_changes_result}")
        except asyncio.TimeoutError:
             prometheus_logger.error("ERROR fetching SPY/VIX percentage changes: Timeout")
             print("   -> Prometheus Core: ❌ ERROR fetching SPY/VIX % changes: Timed out.")
        except Exception as e_changes:
             prometheus_logger.exception(f"ERROR fetching SPY/VIX percentage changes: {e_changes}")
             print(f"   -> Prometheus Core: ❌ ERROR fetching SPY/VIX % changes: {e_changes}")

        prometheus_logger.info(f"Final market context ready: {context}")
        return context

    # --- Other Methods (_log_command, execute_and_log, analyze_workflows, background_correlation_analysis, start_interactive_session, _query_log_db) ---
    # (Keep these methods exactly as provided in the previous response)
    async def execute_and_log(self, command_name_with_slash: str, args: List[str], called_by_user: bool = False):
        """
        Central method to execute a command from the toolbox, gather context,
        log the execution details to the database, and handle errors.
        """
        start_time = datetime.now()
        command_name = command_name_with_slash.lstrip('/')
        context = await self.get_market_context() # Fetch the enhanced context
        command_func = self.toolbox.get(command_name)

        if not command_func:
            output_summary = f"Unknown command '{command_name_with_slash}' intercepted by Prometheus."
            prometheus_logger.error(output_summary)
            print(output_summary)
            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
            self._log_command(start_time, command_name_with_slash, args, context, output_summary, success=False, duration_ms=duration_ms)
            return None

        parameters_to_log = args

        # Convert context dict to a more readable string format for console log
        context_str_parts = [f"VIX:{context.get('vix_price', 'N/A')}", f"Score:{context.get('spy_score', 'N/A')}"]
        # Safely access nested dicts
        spy_chg = context.get('spy_changes', {})
        vix_chg = context.get('vix_changes', {})
        
        # --- FIX: Changed keys from '1d_change' to '1d' and '1y_change' to '1y' ---
        spy_1d = spy_chg.get('1d','N/A') if isinstance(spy_chg, dict) else 'N/A'
        spy_1y = spy_chg.get('1y','N/A') if isinstance(spy_chg, dict) else 'N/A'
        vix_1d = vix_chg.get('1d','N/A') if isinstance(vix_chg, dict) else 'N/A'
        vix_1y = vix_chg.get('1y','N/A') if isinstance(vix_chg, dict) else 'N/A'
        # --- END FIX ---

        context_str_parts.append(f"SPY(1d:{spy_1d},1y:{spy_1y})")
        context_str_parts.append(f"VIX(1d:{vix_1d},1y:{vix_1y})")
        context_str_log = ", ".join(context_str_parts)


        log_msg_start = f"Executing: {command_name_with_slash} {' '.join(args)} | Context: {context_str_log}"
        prometheus_logger.info(log_msg_start)
        print(f"[Prometheus Log] {log_msg_start}")

        output_summary = f"Execution of {command_name_with_slash} started."
        success_flag = False
        result = None

        try:
             kwargs = {"args": args, "is_called_by_ai": False} # Default assumption

             # Check function signature for AI-specific params before adding them
             import inspect
             sig = inspect.signature(command_func)
             if "gemini_model_obj" in sig.parameters and command_name in ["dev", "report", "compare", "powerscore", "sentiment"]:
                  kwargs["gemini_model_obj"] = self.gemini_model
             if "api_lock_override" in sig.parameters and command_name in ["powerscore", "sentiment"]:
                  try:
                       from main_singularity import GEMINI_API_LOCK
                       kwargs["api_lock_override"] = GEMINI_API_LOCK
                  except ImportError:
                       prometheus_logger.warning(f"Could not import GEMINI_API_LOCK for command {command_name}")
             if "screener_func" in sig.parameters and command_name == "dev":
                  kwargs["screener_func"] = self.screener_func

             # Remove is_called_by_ai if the target function doesn't accept it explicitly
             # (Most isolated commands might not have it anymore)
             # However, keep it for functions potentially needing it like risk, invest, etc.
             # A safer approach is to pass **kwargs to handlers that might receive extra args.

             if asyncio.iscoroutinefunction(command_func):
                 result = await command_func(**kwargs)
             else:
                 result = await asyncio.to_thread(lambda: command_func(**kwargs))

             success_flag = True

             # --- Refined Result Summarization ---
             if result is None:
                  output_summary = f"{command_name_with_slash} completed successfully (likely printed output)."
             elif isinstance(result, str):
                 if "error" in result.lower(): success_flag = False
                 output_summary = result[:1000]
             elif isinstance(result, dict):
                 if result.get('status') == 'error' or 'error' in result:
                      success_flag = False
                      output_summary = str(result.get('error') or result.get('message', 'Unknown error in dict result'))[:1000]
                 elif 'summary' in result: output_summary = str(result['summary'])[:1000]
                 elif 'message' in result: output_summary = str(result['message'])[:1000]
                 elif 'filename' in result: output_summary = f"{command_name_with_slash} completed. Output: {result['filename']}"
                 elif result.get('status') == 'success': output_summary = f"{command_name_with_slash} completed successfully."
                 else: output_summary = f"{command_name_with_slash} completed (returned dict)."
             elif isinstance(result, tuple) and len(result) >= 3 and command_name in ["cultivate", "invest"]:
                  cash_idx, holdings_idx = (2, 3) if command_name == "invest" else (2, 1)
                  num_holdings = len(result[holdings_idx]) if isinstance(result[holdings_idx], list) else 0
                  output_summary = f"{command_name.capitalize()} done. {num_holdings} holdings. Cash: ${result[cash_idx]:,.2f}"
             elif isinstance(result, (list, pd.DataFrame)):
                 output_summary = f"{command_name_with_slash} completed ({type(result).__name__}[{len(result)} items])."
             else:
                  output_summary = f"{command_name_with_slash} completed (result type: {type(result)})."

             if success_flag: prometheus_logger.info(f"Command {command_name_with_slash} finished successfully.")
             else: prometheus_logger.warning(f"Command {command_name_with_slash} finished with reported error: {output_summary}")


        except Exception as e:
             success_flag = False
             output_summary = f"CRITICAL ERROR executing {command_name_with_slash}: {type(e).__name__} - {e}"
             prometheus_logger.exception(f"CRITICAL ERROR executing {command_name_with_slash}")
             print(f"[Prometheus Log] {output_summary}")

        finally:
             duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
             self._log_command(start_time, command_name_with_slash, parameters_to_log, context, output_summary, success=success_flag, duration_ms=duration_ms)
             if success_flag and random.random() < 0.1: await self.analyze_workflows()

        return result

    def _log_command(self, timestamp: datetime, command: str, parameters: List[str], context: Dict[str, Any], output_summary: str, success: bool = True, duration_ms: int = 0):
        # (Keep existing implementation)
        log_entry = { "timestamp": timestamp.isoformat(), "command": command, "parameters": json.dumps(parameters),
                      "market_context": json.dumps(context), "output_summary": output_summary, "success": success, "duration_ms": duration_ms }
        log_msg = f"Logging: {command} | Success: {success} | Duration: {duration_ms}ms | Summary: {output_summary[:60]}..."
        prometheus_logger.info(log_msg)
        print(f"[Prometheus Log] {log_msg}")
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO command_log (timestamp, command, parameters, market_context, output_summary, success, duration_ms)
                VALUES (:timestamp, :command, :parameters, :market_context, :output_summary, :success, :duration_ms)
            """, log_entry)
            conn.commit()
            conn.close()
        except Exception as e:
            prometheus_logger.exception(f"ERROR logging command to DB: {e}")
            print(f"   -> Prometheus Core: ❌ ERROR logging command to DB: {e}")

    async def analyze_workflows(self):
        # (Keep existing implementation)
        prometheus_logger.info("Analyzing command history for potential workflows...")
        print("[Prometheus Workflow] Analyzing command history...")
        conn = sqlite3.connect(self.db_path)
        try:
            query = """
            SELECT c1.command AS command1, c2.command AS command2, COUNT(*) as frequency
            FROM command_log c1 JOIN command_log c2 ON c1.id + 1 = c2.id
            WHERE c1.success = 1 AND c2.success = 1
            GROUP BY command1, command2 HAVING frequency >= 2 ORDER BY frequency DESC LIMIT 5;
            """
            df_pairs = pd.read_sql_query(query, conn)
            if not df_pairs.empty:
                prometheus_logger.info(f"Potential workflows detected: {df_pairs.to_dict('records')}")
                print("-> Prometheus Suggestion: Potential workflows detected:")
                for _, row in df_pairs.iterrows():
                    print(f"  - Sequence `{row['command1']}` -> `{row['command2']}` observed {row['frequency']} times.")
            else:
                 prometheus_logger.info("No frequent command sequences (>=2) found.")
                 print("[Prometheus Workflow] No frequent (>=2) command sequences found.")
        except Exception as e:
            prometheus_logger.exception(f"ERROR analyzing workflows: {e}")
            print(f"[Prometheus Workflow] ❌ ERROR analyzing workflows: {e}")
        finally: conn.close()

    async def background_correlation_analysis(self):
        # (Keep existing implementation)
         try: from main_singularity import get_sp500_symbols_singularity
         except ImportError:
             prometheus_logger.error("Failed to import get_sp500_symbols_singularity for background task."); return
         while True:
             await asyncio.sleep(3600 * 4)
             prometheus_logger.info("Starting background cross-tool correlation analysis cycle.")
             print("\n[Prometheus Background] Starting cross-tool correlation analysis...")
             try:
                 sp500_tickers = await asyncio.to_thread(get_sp500_symbols_singularity)
                 if not sp500_tickers: prometheus_logger.warning("Failed to get S&P500 tickers."); continue
                 subset_size = min(len(sp500_tickers), 25); subset_tickers = random.sample(sp500_tickers, subset_size)
                 prometheus_logger.info(f"Analyzing correlation for {len(subset_tickers)} tickers.")
                 print(f"[Prometheus Background] Analyzing {len(subset_tickers)} tickers...")
                 deriv_results: Dict[str, Optional[float]] = {}; ml_results: Dict[str, Optional[float]] = {}
                 deriv_tasks = {}; ml_tasks = {}
                 # --- Derivative ---
                 for ticker in subset_tickers:
                      if self.derivative_func: deriv_tasks[ticker] = asyncio.create_task(self.derivative_func(args=[ticker], **{}))
                      else: deriv_results[ticker] = None
                 deriv_raw_results = await asyncio.gather(*deriv_tasks.values(), return_exceptions=True)
                 for i, ticker in enumerate(subset_tickers):
                      if ticker not in deriv_tasks: continue
                      result = deriv_raw_results[i]
                      if isinstance(result, Exception): deriv_results[ticker] = None; prometheus_logger.warning(f"Deriv task failed {ticker}: {result}")
                      elif isinstance(result, dict) and result.get('ticker') == ticker:
                           periods = result.get('periods', {}); one_year = periods.get('1y', {})
                           deriv_results[ticker] = one_year.get('second_derivative_at_end') if one_year.get('status') == 'success' else None
                      else: deriv_results[ticker] = None; prometheus_logger.warning(f"Unexpected deriv result {ticker}: {type(result)}")
                      await asyncio.sleep(0.01)
                 # --- ML Forecast ---
                 for ticker in subset_tickers:
                      if self.mlforecast_func: ml_tasks[ticker] = asyncio.create_task(self.mlforecast_func(ai_params={'ticker': ticker}, is_called_by_ai=True, **{}))
                      else: ml_results[ticker] = None
                 ml_raw_results = await asyncio.gather(*ml_tasks.values(), return_exceptions=True)
                 for i, ticker in enumerate(subset_tickers):
                     if ticker not in ml_tasks: continue
                     result = ml_raw_results[i]
                     if isinstance(result, Exception): ml_results[ticker] = None; prometheus_logger.warning(f"ML task failed {ticker}: {result}")
                     elif isinstance(result, list) and result:
                          ml_results[ticker] = None # Default
                          for forecast in result:
                              if forecast.get("Period") == "5-Day":
                                  try: ml_results[ticker] = float(forecast.get("Est. % Change", "0%").replace('%', '')); break
                                  except ValueError: pass
                     else: ml_results[ticker] = None; prometheus_logger.warning(f"Unexpected ML result {ticker}: {type(result)}")
                     await asyncio.sleep(0.01)
                 # --- Correlation ---
                 combined = [{'ticker': t, 'deriv': deriv_results.get(t), 'ml': ml_results.get(t)} for t in subset_tickers]
                 valid_data = [d for d in combined if isinstance(d['deriv'],(float,int)) and isinstance(d['ml'],(float,int)) and pd.notna(d['deriv']) and pd.notna(d['ml'])]
                 if len(valid_data) > 5:
                      df_corr = pd.DataFrame(valid_data).set_index('ticker')
                      try:
                           corr_matrix = df_corr.corr(); corr_value = corr_matrix.loc['deriv', 'ml']
                           prometheus_logger.info(f"Correlation(1Y Deriv Accel, 5D ML Fcst): {corr_value:.3f} ({len(df_corr)} stocks)")
                           print(f"[Prometheus Background] Corr Analysis: Deriv Accel vs 5D ML Fcst = {corr_value:.3f}")
                           if abs(corr_value) > 0.4: print(f"  -> Potential correlation ({corr_value:.3f}) found.")
                      except Exception as ce: prometheus_logger.exception(f"Corr calc error: {ce}"); print(f"[Prometheus Background] Corr calc error: {ce}")
                 else: print(f"[Prometheus Background] Not enough data ({len(valid_data)}) for correlation.")
             except asyncio.CancelledError: prometheus_logger.info("Background correlation task cancelled."); break
             except Exception as e: prometheus_logger.exception(f"ERROR background correlation cycle: {e}"); print(f"[Prometheus Background] Cycle ERROR: {e}")
             finally: prometheus_logger.info("Background correlation cycle finished."); print("[Prometheus Background] Cycle finished.")


    async def start_interactive_session(self):
        """ Handles the /prometheus command for direct interaction. """
        # (Keep existing implementation)
        print("\n--- Prometheus Meta-AI Shell ---")
        print("Available commands: analyze patterns, check correlations, query log <limit>, exit")
        prometheus_logger.info("Entered Prometheus interactive shell.")
        while True:
            try:
                user_input = await asyncio.to_thread(input, "Prometheus> ")
                user_input_lower = user_input.lower().strip()
                parts = user_input.split()
                cmd = parts[0].lower() if parts else ""

                if cmd == 'exit': prometheus_logger.info("Exiting Prometheus interactive shell."); break
                elif cmd == "analyze" and len(parts)>1 and parts[1].lower() == "patterns": await self.analyze_workflows()
                elif cmd == "check" and len(parts)>1 and parts[1].lower() == "correlations":
                     print("Triggering background correlation analysis manually...")
                     can_run_corr = self.derivative_func and self.mlforecast_func
                     if can_run_corr and (not self.correlation_task or self.correlation_task.done()):
                          self.correlation_task = asyncio.create_task(self.background_correlation_analysis()); print("   -> Correlation task started.")
                     elif self.correlation_task and not self.correlation_task.done(): print("   -> Correlation task is already running.")
                     else: print("   -> Cannot run correlation analysis - required functions missing.")
                elif cmd == "query" and len(parts)>1 and parts[1].lower() == "log":
                    limit = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 10
                    await self._query_log_db(limit)
                else: print("Unknown command. Available: analyze patterns, check correlations, query log <limit>, exit")
            except EOFError: prometheus_logger.warning("EOF received, exiting Prometheus shell."); break
            except Exception as e: prometheus_logger.exception(f"Error in Prometheus shell: {e}"); print(f"Error: {e}")
        print("Returning to M.I.C. Singularity main shell.")

    async def _query_log_db(self, limit: int = 10):
         """ Helper function to query and display recent logs from the DB. """
         # (Keep existing implementation)
         print(f"\n--- Recent Command Logs (Last {limit}) ---")
         try:
             conn = sqlite3.connect(self.db_path); conn.row_factory = sqlite3.Row
             cursor = conn.cursor()
             cursor.execute("SELECT timestamp, command, parameters, success, duration_ms, output_summary FROM command_log ORDER BY id DESC LIMIT ?", (limit,))
             rows = cursor.fetchall(); conn.close()
             if not rows: print("No logs found."); return
             for row in reversed(rows):
                 ts = datetime.fromisoformat(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S'); success_str = "✅" if row['success'] else "❌"
                 try: params_list = json.loads(row['parameters'])
                 except: params_list = ["<err>"]
                 params_str = " ".join(params_list)[:50] + ('...' if len(" ".join(params_list)) > 50 else '')
                 summary_str = row['output_summary'][:70].replace('\n', ' ') + ('...' if len(row['output_summary']) > 70 else '')
                 print(f"{ts} {success_str} [{row['duration_ms']}ms] {row['command']} {params_str} -> {summary_str}")
         except Exception as e: prometheus_logger.exception(f"Error querying log db: {e}"); print(f"Error: {e}")