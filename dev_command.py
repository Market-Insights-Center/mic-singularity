# --- Imports for dev_command ---
import sys
import os
import uuid
import importlib.util
import traceback
import re
import json
import asyncio
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from io import StringIO

import yfinance as yf
import pandas as pd
import numpy as np
import google.generativeai as genai
from tabulate import tabulate
import matplotlib.pyplot as plt
import humanize
import requests
from dateutil.relativedelta import relativedelta

# --- Globals & Setup ---
STRATEGY_DIR = "generated_strategies"
os.makedirs(STRATEGY_DIR, exist_ok=True)
AI_MODEL_NAME = "gemini-1.5-flash"

# --- Base Code for AI-Generated Strategies ---
BASE_STRATEGY_CODE = """
import pandas as pd
import numpy as np

# --- Technical Indicator Implementations (Included in every strategy file) ---
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rs.replace([np.inf, -np.inf], 0, inplace=True)
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fastperiod=12, slowperiod=26, signalperiod=9):
    fast_ema = series.ewm(span=fastperiod, adjust=False).mean()
    slow_ema = series.ewm(span=slowperiod, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signalperiod, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist

# --- Base Strategy Class (Vectorized) ---
class BaseStrategy:
    '''Base class for AI-generated vectorized strategies.'''
    def __init__(self, data, params=None):
        self.data = data
        self.params = params if params is not None else {}

    def generate_signals(self):
        '''
        This method MUST be implemented by the AI.
        It must return a DataFrame with the same index as self.data
        and columns for each ticker.
        Cell values should be 1 for Buy, -1 for Sell, and 0 for Hold.
        '''
        raise NotImplementedError("'generate_signals' must be implemented by a subclass.")
"""

AI_SYSTEM_PROMPT = f"""
You are an expert Python algorithmic trading strategy generator specializing in vectorized backtesting with pandas.
Your task is to write ONLY the Python class named 'Strategy' that inherits from `BaseStrategy`.

**NEW PARADIGM: VECTORIZED SIGNALS**
Your ONLY job is to implement the `generate_signals(self)` method. This method must perform calculations on the entire dataset (`self.data`) at once and return a DataFrame of signals.

**CRITICAL LAWS YOU MUST OBEY:**
1.  **ABSOLUTE LAW:** You MUST implement the `generate_signals(self)` method. DO NOT implement `next()`.
2.  **RETURN VALUE:** The `generate_signals` method MUST return a pandas DataFrame. This DataFrame must have the same index as `self.data`. The columns must be the ticker symbols. The values must be 1 (buy), -1 (sell), or 0 (hold).
3.  **DATA ACCESS:** Access price data using tuple indexing on `self.data`. Example: `close_prices = self.data.loc[:, pd.IndexSlice[:, 'Adj Close']]`. Then, flatten the column names: `close_prices.columns = close_prices.columns.get_level_values(0)`.
4.  **INDICATOR USAGE:** You can use helper functions like `calculate_rsi` which will be available to you. Apply them to the entire DataFrame at once. Example: `rsi = close_prices.apply(calculate_rsi)`.
5.  **DO NOT** write `import` statements or define `BaseStrategy`. Your code MUST start with `class Strategy(BaseStrategy):`.
"""

# --- Helper Functions ---

def _load_gics_map_from_file(filepath="gics_map.txt") -> Dict[str, str]:
    gics_map = {}
    filepath = os.path.join(os.path.dirname(__file__), '..', filepath)
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if ':' in line:
                    code, name = line.strip().split(':', 1)
                    gics_map[code] = name
    except FileNotFoundError:
        print(f"   -> ⚠️ Warning: GICS map file not found at '{filepath}'.")
    return gics_map

async def _get_all_gics_tickers(db_path="gics_database.txt") -> List[str]:
    db_path = os.path.join(os.path.dirname(__file__), '..', db_path)
    if not os.path.exists(db_path): return []
    all_tickers = set()
    try:
        with open(db_path, 'r') as f:
            for line in f:
                if ':' in line:
                    _, tickers_str = line.split(':', 1)
                    all_tickers.update(t.strip().upper() for t in tickers_str.split(',') if t.strip())
    except Exception as e:
        print(f"   -> ❌ Error reading GICS database: {e}")
    return sorted(list(all_tickers))

def _get_specific_index_tickers_robust(index_symbol: str) -> List[str]:
    index_symbol = index_symbol.upper()
    urls = {
        'SPY': ('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', 0, 'Symbol'),
        'QQQ': ('https://en.wikipedia.org/wiki/Nasdaq-100', 4, 'Ticker'),
        'DIA': ('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average', 1, 'Symbol'),
        'IWM': ('https://en.wikipedia.org/wiki/List_of_Russell_2000_stocks', 2, 'Ticker')
    }
    if index_symbol not in urls: return []
    url, table_index, col_name = urls[index_symbol]
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        df = tables[table_index]
        tickers = [str(s).replace('.', '-') for s in df[col_name].tolist() if isinstance(s, str)]
        return sorted(list(set(s for s in tickers if s)))
    except Exception as e:
        print(f"   -> ❌ ERROR: Failed to fetch tickers for {index_symbol}: {e}")
        return []

def _parse_period_to_dates(period_str: str) -> Tuple[Optional[str], Optional[str]]:
    end_date = datetime.now()
    num_match = re.search(r'(\d+)', period_str.lower())
    if not num_match: return None, None
    num = int(num_match.group(1))
    if 'y' in period_str: start_date = end_date - relativedelta(years=num)
    elif 'mo' in period_str: start_date = end_date - relativedelta(months=num)
    elif 'd' in period_str: start_date = end_date - relativedelta(days=num)
    else: return None, None
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

async def get_yf_data_for_dev_command(tickers: List[str], period_str: str, interval: str) -> Optional[pd.DataFrame]:
    if not tickers: return pd.DataFrame()
    start_date, end_date = _parse_period_to_dates(period_str)
    if not start_date or not end_date: return None
    try:
        data = await asyncio.to_thread(yf.download, tickers=tickers, start=start_date, end=end_date, interval=interval, auto_adjust=False, group_by='ticker', progress=False, timeout=120)
        if data is None or data.empty: return None
        data.dropna(axis=1, how='all', inplace=True)
        return data if not data.empty else None
    except Exception as e:
        print(f"   -> ❌ An error occurred during data fetch: {e}")
        return None

def save_strategy_code(code, filename):
    full_path = os.path.join(STRATEGY_DIR, filename)
    strategy_class_match = re.search(r'class Strategy\(BaseStrategy\):.*', code, re.DOTALL)
    ai_code = strategy_class_match.group(0) if strategy_class_match else code
    final_code = BASE_STRATEGY_CODE + "\n" + ai_code
    with open(full_path, 'w') as f: f.write(final_code)
    print(f"✅ Strategy code saved to: {full_path}")
    return full_path

def load_strategy_from_file(filepath):
    try:
        spec = importlib.util.spec_from_file_location("strategy_module", filepath)
        if spec is None or spec.loader is None: return None
        strategy_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(strategy_module)
        return strategy_module.Strategy
    except Exception as e:
        print(f"❌ Error loading strategy from '{filepath}': {e}"); traceback.print_exc()
        return None

async def generate_code_with_gemini(prompt, gemini_model, code_context=None):
    if not gemini_model: return "class Strategy(BaseStrategy):\n    def generate_signals(self):\n        return pd.DataFrame()"
    try:
        model = genai.GenerativeModel(AI_MODEL_NAME, system_instruction=AI_SYSTEM_PROMPT)
        full_prompt = f"Based on this request: {prompt}"
        if code_context:
            full_prompt = f"Here is the existing code:\n```python\n{code_context}\n```\nModify it based on this request: {prompt}"
        response = await model.generate_content_async(full_prompt)
        return response.text.replace('```python', '').replace('```', '').strip()
    except Exception as e:
        print(f"❌ AI code generation error: {e}")
        return None

# --- New Vectorized Backtest Engine ---
def run_backtest(StrategyClass, data, initial_capital=100000.0):
    print("-> [BACKTEST_ENGINE] Running vectorized backtest...")
    if data.empty: print("❌ Backtest failed: Data is empty."); return

    try:
        strategy = StrategyClass(data=data, params={})
        signals = strategy.generate_signals()
    except Exception as e:
        print(f"❌ CRITICAL: Failed during signal generation. Error: {e}"); traceback.print_exc(); return

    close_prices = data.loc[:, pd.IndexSlice[:, 'Adj Close']]
    close_prices.columns = close_prices.columns.get_level_values(0)
    signals = signals.loc[:, signals.columns.isin(close_prices.columns)] # Align signals with available price data

    portfolio = {'cash': initial_capital}
    positions = {ticker: 0.0 for ticker in close_prices.columns}
    portfolio_values = []
    trade_count = 0

    for i in range(len(close_prices)):
        holdings_value = sum(positions[ticker] * close_prices[ticker].iloc[i] for ticker in positions if not pd.isna(close_prices[ticker].iloc[i]))
        total_value = portfolio['cash'] + holdings_value
        portfolio_values.append(total_value)

        for ticker in signals.columns:
            signal, price = signals[ticker].iloc[i], close_prices[ticker].iloc[i]
            if pd.isna(signal) or pd.isna(price) or price <= 0: continue
            
            if signal == 1 and positions[ticker] == 0:
                buy_amount = portfolio['cash'] * 0.1
                if portfolio['cash'] >= buy_amount:
                    shares_to_buy = buy_amount / price
                    positions[ticker] += shares_to_buy
                    portfolio['cash'] -= buy_amount
                    trade_count += 1
            elif signal == -1 and positions[ticker] > 0:
                sell_value = positions[ticker] * price
                portfolio['cash'] += sell_value
                positions[ticker] = 0
                trade_count += 1
    
    portfolio_df = pd.DataFrame(portfolio_values, index=close_prices.index, columns=['value'])
    returns = portfolio_df['value'].pct_change().dropna()
    final_value = portfolio_df['value'].iloc[-1]
    total_return = (final_value / initial_capital) - 1
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    
    print("\n--- Backtest Results ---")
    results = {"Final Value": f"${final_value:,.2f}", "Total Return": f"{total_return:.2%}", "Sharpe Ratio": f"{sharpe:.2f}", "Trades": trade_count}
    print(tabulate(results.items(), tablefmt="grid"))
    
    # Plotting
    plt.style.use('dark_background'); fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(portfolio_df.index, portfolio_df['value'], label='Portfolio Value', color='cyan')
    ax.set_title("Backtest Performance"); ax.set_ylabel('Portfolio Value ($)'); ax.legend(); ax.grid(True, alpha=0.3)
    filename = f"backtest_results_{uuid.uuid4().hex[:6]}.png"
    plt.savefig(filename); plt.close(fig)
    print(f"\n-> Performance plot saved to '{filename}'")

# --- Command Handlers ---

def get_sanitized_filename(name: str) -> str:
    """Removes invalid characters from a string to make it a valid filename."""
    return re.sub(r'[<>:"/\\|?*]', '_', name) + ".py"

async def _generate_strategy_name_with_gemini(prompt: str, gemini_model) -> str:
    """Uses the AI to generate a descriptive, sanitized base name for a strategy."""
    print("-> Asking AI for a descriptive filename...")
    if not gemini_model:
        return f"Strategy_{uuid.uuid4().hex[:6]}" # Fallback

    try:
        model = genai.GenerativeModel(AI_MODEL_NAME)
        naming_prompt = (
            f"Based on the following trading strategy description, create a concise and descriptive name in PascalCase. "
            f"Do not include '.py' or any other text, explanation, or code. Just the name.\n\n"
            f'Description: "{prompt}"\n\n'
            f"Name:"
        )
        response = await model.generate_content_async(naming_prompt)
        base_name = response.text.strip().replace("`", "")
        print(f"   -> AI suggested name: {base_name}")
        return base_name
    except Exception as e:
        print(f"   -> ⚠️ AI name generation failed: {e}. Falling back to default naming.")
        return f"Strategy_{uuid.uuid4().hex[:6]}"
    
async def handle_dev_new(args, gemini_model):
    prompt = " ".join(args)
    if not prompt:
        print("Usage: /dev new \"<your strategy idea>\"")
        return

    # 1. Generate a descriptive name first
    base_name = await _generate_strategy_name_with_gemini(prompt, gemini_model)
    filename = get_sanitized_filename(base_name)

    # 2. Generate the strategy code
    print(f"-> Generating strategy code for '{filename}'...")
    code = await generate_code_with_gemini(prompt, gemini_model)

    # 3. Save the file
    if code:
        save_strategy_code(code, filename)
        
async def handle_dev_modify(args, gemini_model):
    if len(args) < 2: print("Usage: /dev modify <filename.py> \"<request>\""); return
    filename = os.path.basename(args[0])
    prompt = " ".join(args[1:])
    try:
        with open(os.path.join(STRATEGY_DIR, filename), 'r') as f:
            code = await generate_code_with_gemini(prompt, gemini_model, f.read())
            if code: save_strategy_code(code, filename)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {filename}")

async def handle_dev_backtest(args, gemini_model, data_fetch_func, screener_func):
    """
    Handles the '/dev backtest' subcommand.
    """
    try:
        # Robustly find the indices of 'on' and 'over'
        on_idx = args.index("on")
        over_idx = args.index("over")
        
        filename = os.path.basename(args[0])
        # Join everything between 'on' and 'over' for multi-word tickers/SCREENER
        ticker_str = " ".join(args[on_idx+1:over_idx]).upper()
        # Join everything after 'over' for the period
        period_str = " ".join(args[over_idx+1:])

    except (ValueError, IndexError):
        print("Usage: /dev backtest <file.py> on <TICKER/SCREENER> over <period>")
        print("Example: /dev backtest my_strategy.py on SCREENER over 1y")
        return
    
    strategy_path = os.path.join(STRATEGY_DIR, filename)
    StrategyClass = load_strategy_from_file(strategy_path)
    if not StrategyClass:
        print(f"❌ Failed to load strategy from '{strategy_path}'. Aborting backtest.")
        return

    tickers = []
    if ticker_str == 'SCREENER':
        print("\n--- [DEV Interactive Screener] ---")
        sector_input = input("-> Enter sector identifiers (comma-separated, or 'Market'): ")
        
        print("-> Now enter filter criteria (e.g., fundamental_score > 80). Press Enter on an empty line when done.")
        criteria = []
        while True:
            crit_str = input(f"   Criterion {len(criteria) + 1}: ").strip()
            if not crit_str:
                break
            try:
                # Use regex to handle various spacing and ensure correct format
                match = re.match(r'(\w+)\s*([<>=!]+)\s*([\d\.]+)', crit_str)
                if match:
                    metric, op, val = match.groups()
                    criteria.append({"metric": metric.lower(), "operator": op, "value": float(val)})
                    print(f"     ...Criterion added: {metric} {op} {val}")
                else:
                    print("   -> Invalid format. Use: <metric> <operator> <value> (e.g., invest_score < 50)")
            except (ValueError, IndexError):
                print("   -> Invalid format. Please try again.")

        if sector_input and criteria:
            print("\n-> Running screener with your criteria...")
            ai_params_for_screener = {
                "sector_identifiers": [s.strip() for s in sector_input.split(',')],
                "criteria": criteria
            }
            # The screener_func is 'find_and_screen_stocks' from the main script
            screener_result = await screener_func(args=[], ai_params=ai_params_for_screener, is_called_by_ai=True)
            
            if screener_result and screener_result.get('status') == 'success':
                tickers = [item['Ticker'] for item in screener_result.get('results', [])]
                if tickers:
                    print(f"   -> Screener found {len(tickers)} tickers: {', '.join(tickers[:5])}{'...' if len(tickers)>5 else ''}")
                else:
                    print("   -> Screener ran successfully but found no matching tickers.")
            else:
                print(f"   -> Screener failed to run or returned an error: {screener_result.get('message', 'Unknown error')}")
        else:
            print("-> Screener requires at least one sector and one criterion. Aborting.")
    else:
        tickers = [ticker_str]
    
    if not tickers:
        print("❌ No tickers specified for backtest. Aborting.")
        return
        
    print(f"\n-> Fetching data for {len(tickers)} ticker(s) over '{period_str}'...")
    # The data_fetch_func is 'get_yf_data_for_dev_command' from this module
    data = await data_fetch_func(tickers, period_str, "1d")
    
    if data is not None and not data.empty:
        run_backtest(StrategyClass, data)
    else:
        print(f"❌ Failed to fetch any valid data for the specified tickers and period. Aborting backtest.")
        
async def handle_dev_command(args, gemini_model_obj, screener_func):
    if not args: print("Usage: /dev [new|modify|backtest] ..."); return
    sub_cmd, sub_args = args[0].lower(), args[1:]
    
    if sub_cmd == "new": await handle_dev_new(sub_args, gemini_model_obj)
    elif sub_cmd == "modify": await handle_dev_modify(sub_args, gemini_model_obj)
    elif sub_cmd == "backtest": await handle_dev_backtest(sub_args, gemini_model_obj, get_yf_data_for_dev_command, screener_func)
    else: print(f"Unknown /dev command: {sub_cmd}")