# --- Imports for backtest_command ---
import asyncio
import uuid
from typing import List, Dict, Any, Optional

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# --- Helper Functions (from strategies_command) ---

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

# --- Core Backtest Logic ---

async def run_strategy_backtest(ticker: str, strategy: str, period: str, params: Dict[str, Any]):
    """
    Core logic for running a strategy backtest. Fetches data, implements strategy
    logic, simulates trades, and plots the results.
    """
    print("   -> Fetching historical data...")
    data_download = await asyncio.to_thread(
        yf.download, ticker, period=period, interval="1d", auto_adjust=False, progress=False
    )

    if data_download.empty:
        print(f"❌ Error: No data downloaded for {ticker}. The ticker may be invalid or delisted.")
        return

    hist_data = data_download.copy()
    if isinstance(hist_data.columns, pd.MultiIndex):
        hist_data.columns = hist_data.columns.get_level_values(0)

    # Use 'Adj Close' for calculations but keep 'Close' for BUSD etc.
    price_col = 'Adj Close'
    if price_col not in hist_data.columns:
        print(f"❌ Error: Required 'Adj Close' column not found.")
        return

    print(f"   -> Applying '{strategy}' logic...")
    hist_data['signal'] = 0

    if strategy == 'ma_crossover':
        short_ma = params['short_ma']
        long_ma = params['long_ma']
        hist_data[f'SMA{short_ma}'] = hist_data[price_col].rolling(window=short_ma).mean()
        hist_data[f'SMA{long_ma}'] = hist_data[price_col].rolling(window=long_ma).mean()
        hist_data['position'] = np.where(hist_data[f'SMA{short_ma}'] > hist_data[f'SMA{long_ma}'], 1, -1)
        hist_data['signal'] = hist_data['position'].diff().fillna(0)

    elif strategy == 'rsi':
        rsi_period = params['rsi_period']
        buy_level = params['rsi_buy']
        sell_level = params['rsi_sell']
        hist_data['RSI'] = calculate_rsi(hist_data, period=rsi_period)
        buy_cond = (hist_data['RSI'].shift(1) > buy_level) & (hist_data['RSI'] <= buy_level)
        sell_cond = (hist_data['RSI'].shift(1) < sell_level) & (hist_data['RSI'] >= sell_level)
        hist_data.loc[buy_cond, 'signal'] = 1
        hist_data.loc[sell_cond, 'signal'] = -1
    
    elif strategy == 'busd':
        hist_data.loc[hist_data['Close'] > hist_data['Open'], 'signal'] = 1
        hist_data.loc[hist_data['Close'] < hist_data['Open'], 'signal'] = -1

    elif strategy == 'trend_following':
        ema_short = params['ema_short']
        ema_long = params['ema_long']
        adx_thresh = params['adx_thresh']
        hist_data[f'EMA_{ema_short}'] = hist_data[price_col].ewm(span=ema_short, adjust=False).mean()
        hist_data[f'EMA_{ema_long}'] = hist_data[price_col].ewm(span=ema_long, adjust=False).mean()
        hist_data['ADX'] = calculate_adx(hist_data)
        long_cond = (hist_data[f'EMA_{ema_short}'] > hist_data[f'EMA_{ema_long}']) & (hist_data['ADX'] > adx_thresh)
        short_cond = (hist_data[f'EMA_{ema_short}'] < hist_data[f'EMA_{ema_long}']) & (hist_data['ADX'] > adx_thresh)
        hist_data['position'] = np.select([long_cond, short_cond], [1, -1], default=0)
        hist_data['position'] = hist_data['position'].replace(0, np.nan).ffill().fillna(0)
        hist_data['signal'] = hist_data['position'].diff().fillna(0)

    elif strategy == 'mean_reversion':
        bb_window = params['bb_window']
        bb_std = params['bb_std']
        rsi_period = params['rsi_period']
        rsi_buy = params['rsi_buy']
        rsi_sell = params['rsi_sell']
        hist_data['SMA'] = hist_data[price_col].rolling(window=bb_window).mean()
        hist_data['STD'] = hist_data[price_col].rolling(window=bb_window).std()
        hist_data['Upper_Band'] = hist_data['SMA'] + (hist_data['STD'] * bb_std)
        hist_data['Lower_Band'] = hist_data['SMA'] - (hist_data['STD'] * bb_std)
        hist_data['RSI'] = calculate_rsi(hist_data, period=rsi_period)
        buy_cond = (hist_data[price_col] <= hist_data['Lower_Band']) & (hist_data['RSI'] < rsi_buy)
        sell_cond = (hist_data[price_col] >= hist_data['Upper_Band']) & (hist_data['RSI'] > rsi_sell)
        hist_data.loc[buy_cond, 'signal'] = 1
        hist_data.loc[sell_cond, 'signal'] = -1

    elif strategy == 'volatility_breakout':
        donchian_window = params['donchian_window']
        hist_data['Upper_Channel'] = hist_data['High'].rolling(window=donchian_window).max().shift(1)
        hist_data['Lower_Channel'] = hist_data['Low'].rolling(window=donchian_window).min().shift(1)
        buy_cond = hist_data['Close'] > hist_data['Upper_Channel']
        sell_cond = hist_data['Close'] < hist_data['Lower_Channel']
        hist_data.loc[buy_cond, 'signal'] = 1
        hist_data.loc[sell_cond, 'signal'] = -1
    
    print("   -> Simulating trades and calculating equity curves...")
    initial_capital = 10000.0
    capital = initial_capital
    position_shares = 0.0
    hist_data['strategy_equity'] = initial_capital
    hist_data['hold_equity'] = initial_capital * (hist_data[price_col] / hist_data[price_col].iloc[0])

    for i in range(1, len(hist_data)):
        signal_value = hist_data['signal'].iloc[i]
        current_price = hist_data[price_col].iloc[i]
        
        # On buy signal, if not in a position, go long. If in a short position, flip to long.
        if signal_value > 0:
            if position_shares <= 0:
                capital += position_shares * current_price # close short position
                position_shares = capital / current_price
                capital = 0
        # On sell signal, if in a long position, go flat or flip short.
        elif signal_value < 0:
            if position_shares > 0:
                capital = position_shares * current_price
                position_shares = 0
                # For strategies that short:
                if strategy in ['ma_crossover', 'trend_following']:
                     position_shares = - (capital / current_price)
                     capital = 0

        current_equity = capital + (position_shares * current_price)
        if not np.isnan(current_equity):
            hist_data.iloc[i, hist_data.columns.get_loc('strategy_equity')] = current_equity
        else: # If equity becomes NaN, carry forward the last valid value
            hist_data.iloc[i, hist_data.columns.get_loc('strategy_equity')] = hist_data.iloc[i-1, hist_data.columns.get_loc('strategy_equity')]
    
    strategy_return = (hist_data['strategy_equity'].iloc[-1] / initial_capital - 1) * 100
    hold_return = (hist_data['hold_equity'].iloc[-1] / initial_capital - 1) * 100

    print("\n--- Backtest Results ---")
    print(f"Strategy Return: {strategy_return:+.2f}%")
    print(f"Buy & Hold Return: {hold_return:+.2f}%")
    print("------------------------")
    
    print("   -> Generating results plot...")
    plot_backtest_results(hist_data, ticker, strategy)

def plot_backtest_results(data: pd.DataFrame, ticker: str, strategy: str):
    """
    Generates a two-panel chart visualizing the backtest results.
    """
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        price_col = 'Adj Close'
        ax1.plot(data.index, data[price_col], label=f'{ticker} Price', color='grey', linewidth=1.5)
        
        buy_signals = data[data['signal'] > 0]
        ax1.plot(buy_signals.index, buy_signals[price_col], '^', markersize=10, color='lime', label='Buy Signal', markeredgecolor='black')
        
        sell_signals = data[data['signal'] < 0]
        ax1.plot(sell_signals.index, sell_signals[price_col], 'v', markersize=10, color='red', label='Sell Signal', markeredgecolor='black')

        ax1.set_title(f"{ticker} Backtest: '{strategy.replace('_', ' ').title()}' Strategy", color='white', fontsize=16)
        ax1.set_ylabel("Adjusted Price (USD)", color='white')
        ax1.legend()
        ax1.grid(True, color='dimgray', linestyle='--', linewidth=0.5, alpha=0.5)

        ax2.plot(data.index, data['strategy_equity'], label='Strategy Equity', color='cyan', linewidth=2)
        ax2.plot(data.index, data['hold_equity'], label='Buy & Hold Equity', color='orange', linestyle='--')
        
        ax2.set_xlabel("Date", color='white')
        ax2.set_ylabel("Portfolio Value ($)", color='white')
        ax2.legend()
        ax2.grid(True, color='dimgray', linestyle='--', linewidth=0.5, alpha=0.5)

        fig.tight_layout()
        filename = f"backtest_{ticker}_{strategy}_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black', edgecolor='black', dpi=300)
        plt.close(fig)
        print(f"📂 Backtest results chart saved as: {filename}")

    except Exception as e:
        print(f"❌ Error plotting backtest results: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)

# --- Main Command Handler ---

async def handle_backtest_command(args: List[str], is_called_by_ai: bool = False):
    """
    Handles the /backtest command for testing specific trading strategies with customizable parameters.
    """
    if is_called_by_ai:
        return "The /backtest command is currently available for CLI use only."

    print("\n--- Trading Strategy Backtest Engine ---")
    if len(args) < 3:
        print("Usage: /backtest <TICKER> <strategy> <period> [params...]")
        print("\n--- Available Strategies & Parameters ---")
        print("  ma_crossover [short_win (def:50)] [long_win (def:200)]")
        print("  rsi [period (def:14)] [buy_lvl (def:30)] [sell_lvl (def:70)]")
        print("  busd (no parameters)")
        print("  trend_following [short_ema (def:25)] [long_ema (def:75)] [adx_thresh (def:25)]")
        print("  mean_reversion [bb_win (def:20)] [bb_std (def:2)] [rsi_p (def:14)] [rsi_buy (def:30)] [rsi_sell (def:70)]")
        print("  volatility_breakout [donchian_win (def:20)]")
        return

    ticker = args[0].upper()
    strategy = args[1].lower()
    period = args[2].lower()
    strategy_params = {}

    valid_strategies = [
        'ma_crossover', 'rsi', 'busd', 'trend_following', 
        'mean_reversion', 'volatility_breakout'
    ]
    if strategy not in valid_strategies:
        print(f"❌ Error: Invalid strategy '{strategy}'.")
        return

    try:
        if strategy == 'ma_crossover':
            strategy_params['short_ma'] = int(args[3]) if len(args) > 3 else 50
            strategy_params['long_ma'] = int(args[4]) if len(args) > 4 else 200
        elif strategy == 'rsi':
            strategy_params['rsi_period'] = int(args[3]) if len(args) > 3 else 14
            strategy_params['rsi_buy'] = int(args[4]) if len(args) > 4 else 30
            strategy_params['rsi_sell'] = int(args[5]) if len(args) > 5 else 70
        elif strategy == 'busd':
            pass # No params
        elif strategy == 'trend_following':
            strategy_params['ema_short'] = int(args[3]) if len(args) > 3 else 25
            strategy_params['ema_long'] = int(args[4]) if len(args) > 4 else 75
            strategy_params['adx_thresh'] = int(args[5]) if len(args) > 5 else 25
        elif strategy == 'mean_reversion':
            strategy_params['bb_window'] = int(args[3]) if len(args) > 3 else 20
            strategy_params['bb_std'] = int(args[4]) if len(args) > 4 else 2
            strategy_params['rsi_period'] = int(args[5]) if len(args) > 5 else 14
            strategy_params['rsi_buy'] = int(args[6]) if len(args) > 6 else 30
            strategy_params['rsi_sell'] = int(args[7]) if len(args) > 7 else 70
        elif strategy == 'volatility_breakout':
            strategy_params['donchian_window'] = int(args[3]) if len(args) > 3 else 20
    except (ValueError, IndexError):
        print("⚠️ Warning: Invalid or missing parameters. Using defaults for the selected strategy.")

    print(f"-> Starting backtest for {ticker} using '{strategy}' over a {period} period...")
    print(f"   -> Parameters: {strategy_params if strategy_params else 'Defaults'}")
    await run_strategy_backtest(ticker, strategy, period, strategy_params)
