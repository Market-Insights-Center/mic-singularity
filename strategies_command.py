# --- Imports for strategies_command ---
import asyncio
import uuid
from typing import List, Dict, Optional, Any

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# --- Imports from other command modules ---
# from invest_command import calculate_ema_invest # Not used, can be removed

# --- Concurrency Lock for Matplotlib ---
# Matplotlib is not thread-safe, so we use a lock to ensure only one plotting
# operation happens at a time when running strategies concurrently.
plt_lock = asyncio.Lock()

# --- Helper Functions (copied or moved for self-containment) ---

async def get_yf_download_robustly(tickers: list, **kwargs) -> pd.DataFrame:
    """A robust wrapper for yf.download with retry logic."""
    for attempt in range(3):
        try:
            data = await asyncio.to_thread(yf.download, tickers=tickers, progress=False, **kwargs)
            if not data.empty:
                # Ensure columns are flat if MultiIndex is returned
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return data.copy()
        except Exception:
            if attempt < 2:
                await asyncio.sleep((attempt + 1) * 2)
    return pd.DataFrame()

def get_futures_specs() -> Dict[str, Dict[str, Any]]:
    """
    Returns a dictionary of specifications for common futures contracts.
    MODIFIED: Added 'exchange' and 'cycle' type for robust ticker generation.
    """
    return {
        # Indices
        "ES": {"name": "E-mini S&P 500", "ticker": "ES=F", "point_value": 50.0, "tick_size": 0.25, "cycle": "quarterly", "exchange": "CME"},
        "NQ": {"name": "E-mini NASDAQ 100", "ticker": "NQ=F", "point_value": 20.0, "tick_size": 0.25, "cycle": "quarterly", "exchange": "CME"},
        "YM": {"name": "Mini DOW Jones", "ticker": "YM=F", "point_value": 5.0, "tick_size": 1.0, "cycle": "quarterly", "exchange": "CBOT"},
        "RTY": {"name": "E-mini Russell 2000", "ticker": "RTY=F", "point_value": 50.0, "tick_size": 0.1, "cycle": "quarterly", "exchange": "CME"},
        # Energies
        "CL": {"name": "Crude Oil WTI", "ticker": "CL=F", "point_value": 1000.0, "tick_size": 0.01, "cycle": "monthly", "exchange": "NYM"},
        "NG": {"name": "Natural Gas", "ticker": "NG=F", "point_value": 10000.0, "tick_size": 0.001, "cycle": "monthly", "exchange": "NYM"},
        # Metals
        "GC": {"name": "Gold", "ticker": "GC=F", "point_value": 100.0, "tick_size": 0.1, "cycle": "monthly", "exchange": "COMEX"},
        "SI": {"name": "Silver", "ticker": "SI=F", "point_value": 5000.0, "tick_size": 0.005, "cycle": "monthly", "exchange": "COMEX"},
        "HG": {"name": "Copper", "ticker": "HG=F", "point_value": 25000.0, "tick_size": 0.0005, "cycle": "monthly", "exchange": "COMEX"},
        # Currencies
        "6E": {"name": "Euro FX", "ticker": "6E=F", "point_value": 125000.0, "tick_size": 0.00005, "cycle": "quarterly", "exchange": "CME"},
        # Grains
        "ZC": {"name": "Corn", "ticker": "ZC=F", "point_value": 50.0, "tick_size": 0.25, "cycle": "monthly", "exchange": "CBOT"},
    }

def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates the Average Directional Index (ADX)."""
    df = data.copy()
    alpha = 1 / period
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['+DM'] = np.where((df['High'].diff() > df['Low'].diff()) & (df['High'].diff() > 0), df['High'].diff(), 0)
    df['-DM'] = np.where((df['Low'].diff() > df['High'].diff()) & (df['Low'].diff() > 0), df['Low'].diff(), 0)
    df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean()
    df['+DI'] = (df['+DM'].ewm(alpha=alpha, adjust=False).mean() / df['ATR']) * 100
    df['-DI'] = (df['-DM'].ewm(alpha=alpha, adjust=False).mean() / df['ATR']) * 100
    df['DX'] = (abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI']) * 100).fillna(0)
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    return df['ADX']

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss
    rs.replace([np.inf, -np.inf], 0, inplace=True)
    return 100 - (100 / (1 + rs))

def get_signal_score(signal: str) -> int:
    """Converts a signal string to a numerical score."""
    if "BUY" in signal:
        return 1
    if "SELL" in signal:
        return -1
    return 0 # HOLD

# --- Graphing Functions ---

def plot_trend_strategy_graph(data: pd.DataFrame, ticker: str, signal: str, ema_short: int, ema_long: int):
    """Generates a chart for the trend-following strategy."""
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(data.index, data['Close'], label='Price', color='grey')
        ax1.plot(data.index, data[f'EMA_{ema_short}'], label=f'EMA-{ema_short}', color='cyan')
        ax1.plot(data.index, data[f'EMA_{ema_long}'], label=f'EMA-{ema_long}', color='orange')
        buy_signals = data[data['signal'].str.contains('BUY')]
        sell_signals = data[data['signal'].str.contains('SELL')]
        ax1.plot(buy_signals.index, buy_signals['Close'], '^', markersize=8, color='lime', label='Buy Signal')
        ax1.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=8, color='red', label='Sell Signal')
        ax1.set_title(f"Trend Strategy for {ticker} | Latest Signal: {signal}", color='white')
        ax1.legend()
        ax2.plot(data.index, data['ADX'], label='ADX (14)', color='magenta')
        ax2.axhline(25, color='red', linestyle='--', label='Trend Threshold (25)')
        ax2.legend()
        ax2.set_ylim(0, 100)
        fig.tight_layout()
        filename = f"strategy_trend_{ticker.replace('=F', '')}_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black')
        plt.close(fig)
        print(f"📂 Strategy chart saved as: {filename}")
        return filename
    except Exception:
        return "Failed to generate graph."

def plot_mean_reversion_graph(data: pd.DataFrame, ticker: str, signal: str):
    """Generates a chart for the mean reversion strategy."""
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(data.index, data['Close'], label='Price', color='grey')
        ax1.plot(data.index, data['Upper_Band'], label='Upper Band', color='red', linestyle='--')
        ax1.plot(data.index, data['Lower_Band'], label='Lower Band', color='lime', linestyle='--')
        buy_signals = data[data['signal'] == 'BUY 🟢 (Oversold)']
        sell_signals = data[data['signal'] == 'SELL 🔴 (Overbought)']
        ax1.plot(buy_signals.index, buy_signals['Close'], '^', markersize=8, color='lime', label='Buy Signal')
        ax1.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=8, color='red', label='Sell Signal')
        ax1.set_title(f"Mean Reversion for {ticker} | Latest Signal: {signal}", color='white')
        ax1.legend()
        ax2.plot(data.index, data['RSI'], label='RSI (14)', color='magenta')
        ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        ax2.axhline(30, color='lime', linestyle='--', label='Oversold (30)')
        ax2.legend()
        ax2.set_ylim(0, 100)
        fig.tight_layout()
        filename = f"strategy_reversion_{ticker.replace('=F', '')}_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black')
        plt.close(fig)
        print(f"📂 Strategy chart saved as: {filename}")
        return filename
    except Exception:
        return "Failed to generate graph."

def plot_volatility_breakout_graph(data: pd.DataFrame, ticker: str, signal: str):
    """Generates a chart for the volatility breakout strategy."""
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(data.index, data['Close'], label='Price', color='grey')
        ax.plot(data.index, data['Upper_Channel'], label='Upper Channel (20-Day High)', color='lime', linestyle='--')
        ax.plot(data.index, data['Lower_Channel'], label='Lower Channel (20-Day Low)', color='red', linestyle='--')
        buy_signals = data[data['signal'] == 'BUY 🟢 (Bullish Breakout)']
        sell_signals = data[data['signal'] == 'SELL 🔴 (Bearish Breakout)']
        ax.plot(buy_signals.index, buy_signals['Close'], '^', markersize=8, color='lime', label='Buy Signal')
        ax.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=8, color='red', label='Sell Signal')
        ax.set_title(f"Volatility Breakout for {ticker} | Latest Signal: {signal}", color='white')
        ax.legend()
        fig.tight_layout()
        filename = f"strategy_breakout_{ticker.replace('=F', '')}_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black')
        plt.close(fig)
        print(f"📂 Strategy chart saved as: {filename}")
        return filename
    except Exception:
        return "Failed to generate graph."

def plot_ma_crossover_graph(data: pd.DataFrame, ticker: str, signal: str, sma_short: int, sma_long: int):
    """Generates a chart for the MA Crossover strategy."""
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(data.index, data['Close'], label='Price', color='grey', alpha=0.8)
        ax.plot(data.index, data[f'SMA_{sma_short}'], label=f'SMA-{sma_short}', color='cyan')
        ax.plot(data.index, data[f'SMA_{sma_long}'], label=f'SMA-{sma_long}', color='orange')
        buy_signals = data[data['signal'].str.contains('BUY')]
        sell_signals = data[data['signal'].str.contains('SELL')]
        ax.plot(buy_signals.index, buy_signals['Close'], '^', markersize=8, color='lime', label='Buy Signal')
        ax.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=8, color='red', label='Sell Signal')
        ax.set_title(f"MA Crossover Strategy for {ticker} | Latest Signal: {signal}", color='white')
        ax.legend()
        fig.tight_layout()
        filename = f"strategy_macrossover_{ticker.replace('=F', '')}_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black')
        plt.close(fig)
        print(f"📂 Strategy chart saved as: {filename}")
        return filename
    except Exception:
        return "Failed to generate graph."

def plot_simple_rsi_graph(data: pd.DataFrame, ticker: str, signal: str):
    """Generates a chart for the simple RSI strategy."""
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
        ax1.plot(data.index, data['Close'], label='Price', color='grey')
        buy_signals = data[data['signal'] == 'BUY 🟢 (Oversold)']
        sell_signals = data[data['signal'] == 'SELL 🔴 (Overbought)']
        ax1.plot(buy_signals.index, buy_signals['Close'], '^', markersize=8, color='lime', label='Buy Signal')
        ax1.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=8, color='red', label='Sell Signal')
        ax1.set_title(f"Simple RSI Strategy for {ticker} | Latest Signal: {signal}", color='white')
        ax1.legend()
        ax2.plot(data.index, data['RSI'], label='RSI (14)', color='magenta')
        ax2.axhline(70, color='red', linestyle='--', label='Overbought (70)')
        ax2.axhline(30, color='lime', linestyle='--', label='Oversold (30)')
        ax2.legend()
        ax2.set_ylim(0, 100)
        fig.tight_layout()
        filename = f"strategy_simple_rsi_{ticker.replace('=F', '')}_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black')
        plt.close(fig)
        print(f"📂 Strategy chart saved as: {filename}")
        return filename
    except Exception:
        return "Failed to generate graph."

def plot_busd_graph(data: pd.DataFrame, ticker: str, signal: str):
    """Generates a chart for the BUSD (Buy Up Sell Down) strategy."""
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(data.index, data['Close'], label='Price', color='grey')
        buy_signals = data[data['signal'].str.contains('BUY')]
        sell_signals = data[data['signal'].str.contains('SELL')]
        ax.plot(buy_signals.index, buy_signals['Close'], '^', markersize=8, color='lime', label='Buy Signal')
        ax.plot(sell_signals.index, sell_signals['Close'], 'v', markersize=8, color='red', label='Sell Signal')
        ax.set_title(f"BUSD Strategy for {ticker} | Latest Signal: {signal}", color='white')
        ax.legend()
        fig.tight_layout()
        filename = f"strategy_busd_{ticker.replace('=F', '')}_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black')
        plt.close(fig)
        print(f"📂 Strategy chart saved as: {filename}")
        return filename
    except Exception:
        return "Failed to generate graph."


# --- Individual Strategy Functions ---

async def get_strategy_data(ticker_input: str, period: str = "2y") -> Optional[Dict[str, Any]]:
    """Helper to fetch and prepare data for a strategy."""
    ticker_upper = ticker_input.upper().replace('/', '')
    specs = get_futures_specs().get(ticker_upper)
    yf_ticker = specs['ticker'] if specs else ticker_upper
    display_name = specs['name'] if specs else yf_ticker
    
    data = await get_yf_download_robustly([yf_ticker], period=period, auto_adjust=False)
    if data.empty:
        return None
    return {"data": data, "yf_ticker": yf_ticker, "display_name": display_name}

async def run_trend_following_strategy(ticker_input: str, lock: asyncio.Lock) -> Optional[Dict[str, Any]]:
    """Runs the Trend Following (EMA Crossover + ADX) strategy."""
    prep = await get_strategy_data(ticker_input, "2y")
    if not prep: return None
    data, yf_ticker, display_name = prep['data'], prep['yf_ticker'], prep['display_name']
    
    ema_short, ema_long = 25, 75
    data[f'EMA_{ema_short}'] = data['Close'].ewm(span=ema_short, adjust=False).mean()
    data[f'EMA_{ema_long}'] = data['Close'].ewm(span=ema_long, adjust=False).mean()
    data['ADX'] = calculate_adx(data)
    
    conditions = [(data['ADX'] > 25) & (data[f'EMA_{ema_short}'] > data[f'EMA_{ema_long}']), (data['ADX'] > 25) & (data[f'EMA_{ema_short}'] < data[f'EMA_{ema_long}'])]
    choices = ["BUY 🟢 (Trending Up)", "SELL 🔴 (Trending Down)"]
    data['signal'] = np.select(conditions, choices, default="HOLD 🟡 (Weak Trend)")
    
    latest = data.iloc[-1]
    async with lock:
        graph_file = await asyncio.to_thread(plot_trend_strategy_graph, data.tail(252), yf_ticker, latest['signal'], ema_short, ema_long)
    return {"display_name": display_name, "signal": latest['signal'], "graph_file": graph_file}

async def run_mean_reversion_strategy(ticker_input: str, lock: asyncio.Lock) -> Optional[Dict[str, Any]]:
    """Runs the Mean Reversion (Bollinger Bands + RSI) strategy."""
    prep = await get_strategy_data(ticker_input, "1y")
    if not prep: return None
    data, yf_ticker, display_name = prep['data'], prep['yf_ticker'], prep['display_name']
    
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['STD_20'] = data['Close'].rolling(window=20).std()
    data['Upper_Band'] = data['SMA_20'] + (data['STD_20'] * 2)
    data['Lower_Band'] = data['SMA_20'] - (data['STD_20'] * 2)
    data['RSI'] = calculate_rsi(data)
    
    conditions = [(data['Close'] <= data['Lower_Band']) & (data['RSI'] < 30), (data['Close'] >= data['Upper_Band']) & (data['RSI'] > 70)]
    choices = ["BUY 🟢 (Oversold)", "SELL 🔴 (Overbought)"]
    data['signal'] = np.select(conditions, choices, default="HOLD 🟡 (Neutral)")
    
    latest = data.iloc[-1]
    async with lock:
        graph_file = await asyncio.to_thread(plot_mean_reversion_graph, data.tail(252), yf_ticker, latest['signal'])
    return {"display_name": display_name, "signal": latest['signal'], "graph_file": graph_file}

async def run_volatility_breakout_strategy(ticker_input: str, lock: asyncio.Lock) -> Optional[Dict[str, Any]]:
    """Runs the Volatility Breakout (Donchian Channels) strategy."""
    prep = await get_strategy_data(ticker_input, "1y")
    if not prep: return None
    data, yf_ticker, display_name = prep['data'], prep['yf_ticker'], prep['display_name']

    data['Upper_Channel'] = data['High'].rolling(window=20).max().shift(1)
    data['Lower_Channel'] = data['Low'].rolling(window=20).min().shift(1)
    
    conditions = [(data['Close'] > data['Upper_Channel']), (data['Close'] < data['Lower_Channel'])]
    choices = ["BUY 🟢 (Bullish Breakout)", "SELL 🔴 (Bearish Breakout)"]
    data['signal'] = np.select(conditions, choices, default="HOLD 🟡 (In Range)")

    latest = data.iloc[-1]
    async with lock:
        graph_file = await asyncio.to_thread(plot_volatility_breakout_graph, data.tail(252), yf_ticker, latest['signal'])
    return {"display_name": display_name, "signal": latest['signal'], "graph_file": graph_file}

async def run_ma_crossover_strategy(ticker_input: str, lock: asyncio.Lock) -> Optional[Dict[str, Any]]:
    """Runs the MA Crossover (SMA 50/200) strategy."""
    prep = await get_strategy_data(ticker_input, "3y")
    if not prep: return None
    data, yf_ticker, display_name = prep['data'], prep['yf_ticker'], prep['display_name']

    sma_short, sma_long = 50, 200
    data[f'SMA_{sma_short}'] = data['Close'].rolling(window=sma_short).mean()
    data[f'SMA_{sma_long}'] = data['Close'].rolling(window=sma_long).mean()

    conditions = [data[f'SMA_{sma_short}'] > data[f'SMA_{sma_long}'], data[f'SMA_{sma_short}'] < data[f'SMA_{sma_long}']]
    choices = ["BUY 🟢 (Golden Cross)", "SELL 🔴 (Death Cross)"]
    data['signal'] = np.select(conditions, choices, default="HOLD 🟡")

    latest = data.iloc[-1]
    async with lock:
        graph_file = await asyncio.to_thread(plot_ma_crossover_graph, data.tail(350), yf_ticker, latest['signal'], sma_short, sma_long)
    return {"display_name": display_name, "signal": latest['signal'], "graph_file": graph_file}

async def run_simple_rsi_strategy(ticker_input: str, lock: asyncio.Lock) -> Optional[Dict[str, Any]]:
    """Runs the Simple RSI (30/70) strategy."""
    prep = await get_strategy_data(ticker_input, "1y")
    if not prep: return None
    data, yf_ticker, display_name = prep['data'], prep['yf_ticker'], prep['display_name']
    
    data['RSI'] = calculate_rsi(data, period=14)

    conditions = [data['RSI'] < 30, data['RSI'] > 70]
    choices = ["BUY 🟢 (Oversold)", "SELL 🔴 (Overbought)"]
    data['signal'] = np.select(conditions, choices, default="HOLD 🟡 (Neutral)")
    
    latest = data.iloc[-1]
    async with lock:
        graph_file = await asyncio.to_thread(plot_simple_rsi_graph, data.tail(252), yf_ticker, latest['signal'])
    return {"display_name": display_name, "signal": latest['signal'], "graph_file": graph_file}

async def run_busd_strategy(ticker_input: str, lock: asyncio.Lock) -> Optional[Dict[str, Any]]:
    """Runs the BUSD (Buy Up, Sell Down) daily momentum strategy."""
    prep = await get_strategy_data(ticker_input, "1y")
    if not prep: return None
    data, yf_ticker, display_name = prep['data'], prep['yf_ticker'], prep['display_name']
    
    conditions = [data['Close'] > data['Open'], data['Close'] < data['Open']]
    choices = ["BUY 🟢 (Up Day)", "SELL 🔴 (Down Day)"]
    data['signal'] = np.select(conditions, choices, default="HOLD 🟡 (Flat)")

    latest = data.iloc[-1]
    async with lock:
        graph_file = await asyncio.to_thread(plot_busd_graph, data.tail(252), yf_ticker, latest['signal'])
    return {"display_name": display_name, "signal": latest['signal'], "graph_file": graph_file}

# --- Aggregate Strategy Function ---

async def run_average_strategy(ticker: str):
    """Runs all available strategies and calculates a consensus signal."""
    strategy_functions = [
        run_trend_following_strategy,
        run_mean_reversion_strategy,
        run_volatility_breakout_strategy,
        run_ma_crossover_strategy,
        run_simple_rsi_strategy,
        run_busd_strategy
    ]
    strategy_names = [
        "Trend Following (EMA/ADX)",
        "Mean Reversion (BB/RSI)",
        "Volatility Breakout",
        "MA Crossover (SMA 50/200)",
        "Simple RSI (30/70)",
        "Daily Momentum (BUSD)"
    ]

    print(f"\n--- Running All Strategies for {ticker.upper()} ---")
    
    # Run all strategies concurrently for efficiency, passing the lock to each
    all_results = await asyncio.gather(
        *[func(ticker, plt_lock) for func in strategy_functions],
        return_exceptions=True
    )
    
    scores = []
    display_name = ""

    # Process results and calculate scores
    for i, result in enumerate(all_results):
        if isinstance(result, Exception) or not result:
            print(f"- {strategy_names[i]}: ❌ Analysis failed.")
            continue

        if not display_name: # Get display name from the first successful result
            display_name = result['display_name']

        signal = result['signal']
        print(f"- {strategy_names[i]}: {signal}")
        scores.append(get_signal_score(signal))

    if not scores:
        print("\n❌ Could not calculate consensus signal as all analyses failed.")
        return

    # Calculate the average score and determine the final consensus signal
    average_score = np.mean(scores)
    
    if average_score >= 0.4:
        final_signal = f"STRONG BUY 🟢🟢 (Score: {average_score:.2f})"
    elif average_score > 0.15:
        final_signal = f"BUY 🟢 (Score: {average_score:.2f})"
    elif average_score <= -0.4:
        final_signal = f"STRONG SELL 🔴🔴 (Score: {average_score:.2f})"
    elif average_score < -0.15:
        final_signal = f"SELL 🔴 (Score: {average_score:.2f})"
    else:
        final_signal = f"HOLD 🟡 (Score: {average_score:.2f})"
    
    print(f"\n--- Consensus for {display_name} ---")
    print(f"Final Signal: {final_signal}")

# --- Main Command Handler ---

async def handle_strategies_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """
    Main handler for the /strategies command. Routes to the chosen strategy analysis.
    """
    if not args:
        print("\n--- Available Strategies ---")
        print("1. Trend Following (25/75 EMA Crossover with ADX Filter)")
        print("2. Mean Reversion (Bollinger Bands & RSI)")
        print("3. Volatility Breakout (Donchian Channels)")
        print("4. MA Crossover (50/200 SMA)")
        print("5. Simple RSI (Overbought/Oversold)")
        print("6. Daily Momentum (Buy Up-Day/Sell Down-Day)")
        print("avg. Run all strategies for a consensus signal")
        print("\nUsage: /strategies <strategy_number_or_avg> <ticker>")
        return

    if len(args) < 2:
        print("Usage: /strategies <strategy_number_or_avg> <ticker>")
        return

    strategy_num, ticker = args[0], args[1]
    results = None

    if strategy_num.lower() == "avg":
        await run_average_strategy(ticker)
        return

    strategy_map = {
        "1": run_trend_following_strategy,
        "2": run_mean_reversion_strategy,
        "3": run_volatility_breakout_strategy,
        "4": run_ma_crossover_strategy,
        "5": run_simple_rsi_strategy,
        "6": run_busd_strategy,
    }

    strategy_func = strategy_map.get(strategy_num)
    if strategy_func:
        results = await strategy_func(ticker, plt_lock)
    else:
        print(f"Error: Strategy '{strategy_num}' is not valid. Use 1-6 or avg.")
        return

    if results:
        print(f"\n--- Strategy Results for {results['display_name']} ---")
        print(f"Final Signal: {results['signal']}")
    else:
        print(f"❌ Analysis failed for {ticker.upper()}. Could not retrieve necessary data.")
