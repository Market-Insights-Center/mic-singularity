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
    # Ensure ATR calculation handles potential division by zero if TR is zero
    df['ATR'] = df['TR'].ewm(alpha=alpha, adjust=False).mean().replace(0, 1e-9) # Replace 0 ATR with small value
    df['+DI'] = (df['+DM'].ewm(alpha=alpha, adjust=False).mean() / df['ATR']) * 100
    df['-DI'] = (df['-DM'].ewm(alpha=alpha, adjust=False).mean() / df['ATR']) * 100
    # Ensure DX calculation handles potential division by zero
    di_sum = df['+DI'] + df['-DI']
    df['DX'] = (abs(df['+DI'] - df['-DI']) / di_sum.replace(0, 1e-9) * 100).fillna(0) # Replace 0 sum with small value
    df['ADX'] = df['DX'].ewm(alpha=alpha, adjust=False).mean()
    return df['ADX']

def calculate_rsi(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculates the Relative Strength Index (RSI)."""
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    # Handle potential division by zero if loss is zero
    rs = gain / loss.replace(0, 1e-9) # Replace 0 loss with small value
    rs.replace([np.inf, -np.inf], 0, inplace=True) # Handle potential inf values after division
    return 100 - (100 / (1 + rs))

# --- Core Backtest Logic ---

async def run_strategy_backtest(ticker: str, strategy: str, period: str, params: Dict[str, Any], is_cli_call: bool = True) -> Optional[Dict[str, Any]]: # Added return type hint & is_cli_call
    """
    Core logic for running a strategy backtest. Fetches data, implements strategy
    logic, simulates trades, calculates metrics, and optionally plots results.
    Returns a dictionary with results or an error dictionary on failure.
    """
    if is_cli_call: print("   -> Fetching historical data...")
    # Using auto_adjust=False for consistency, ensure 'Adj Close' is used later
    data_download = await asyncio.to_thread(
        yf.download, ticker, period=period, interval="1d", auto_adjust=False, progress=False
    )

    if data_download.empty:
        err_msg = f"Error: No data downloaded for {ticker}. The ticker may be invalid or delisted."
        if is_cli_call: print(f"❌ {err_msg}")
        return {"status": "error", "message": err_msg}

    hist_data = data_download.copy()
    if isinstance(hist_data.columns, pd.MultiIndex):
        hist_data.columns = hist_data.columns.get_level_values(0)

    # Use 'Adj Close' for return calculations and most indicators
    price_col = 'Adj Close'
    if price_col not in hist_data.columns or hist_data[price_col].isnull().all():
        # Fallback to 'Close' if 'Adj Close' is missing or all NaN
        price_col = 'Close'
        if price_col not in hist_data.columns or hist_data[price_col].isnull().all():
            err_msg = f"Error: Required 'Adj Close' or 'Close' column not found or is all NaN."
            if is_cli_call: print(f"❌ {err_msg}")
            return {"status": "error", "message": err_msg}
        elif is_cli_call:
            print(f"   -> Warning: Using 'Close' prices as 'Adj Close' was unavailable.")

    if is_cli_call: print(f"   -> Applying '{strategy}' logic...")
    hist_data['signal'] = 0 # Initialize signal column

    # --- Strategy Signal Generation ---
    try:
        if strategy == 'ma_crossover':
            short_ma = params['short_ma']
            long_ma = params['long_ma']
            hist_data[f'SMA{short_ma}'] = hist_data[price_col].rolling(window=short_ma).mean()
            hist_data[f'SMA{long_ma}'] = hist_data[price_col].rolling(window=long_ma).mean()
            hist_data['position'] = np.where(hist_data[f'SMA{short_ma}'] > hist_data[f'SMA{long_ma}'], 1, -1)
            hist_data['signal'] = hist_data['position'].diff().fillna(0) # Signal on change in position

        elif strategy == 'rsi':
            rsi_period = params['rsi_period']
            buy_level = params['rsi_buy']
            sell_level = params['rsi_sell']
            hist_data['RSI'] = calculate_rsi(hist_data, period=rsi_period)
            # Generate signals based on crossing the levels
            buy_cond = (hist_data['RSI'].shift(1) >= buy_level) & (hist_data['RSI'] < buy_level) # Cross below buy level (exit short/enter long) - adjusted logic
            sell_cond = (hist_data['RSI'].shift(1) <= sell_level) & (hist_data['RSI'] > sell_level) # Cross above sell level (exit long/enter short) - adjusted logic
            hist_data.loc[buy_cond, 'signal'] = 1 # Buy Signal
            hist_data.loc[sell_cond, 'signal'] = -1 # Sell Signal

        elif strategy == 'busd':
            # Use raw 'Close' and 'Open' for this strategy
            if 'Close' not in hist_data.columns or 'Open' not in hist_data.columns:
                 raise KeyError("BUSD requires 'Open' and 'Close' columns.")
            hist_data.loc[hist_data['Close'] > hist_data['Open'], 'signal'] = 1
            hist_data.loc[hist_data['Close'] < hist_data['Open'], 'signal'] = -1
            # Hold if Close == Open

        elif strategy == 'trend_following':
            ema_short = params['ema_short']
            ema_long = params['ema_long']
            adx_thresh = params['adx_thresh']
            hist_data[f'EMA_{ema_short}'] = hist_data[price_col].ewm(span=ema_short, adjust=False).mean()
            hist_data[f'EMA_{ema_long}'] = hist_data[price_col].ewm(span=ema_long, adjust=False).mean()
            hist_data['ADX'] = calculate_adx(hist_data) # Requires High, Low, Close
            long_cond = (hist_data[f'EMA_{ema_short}'] > hist_data[f'EMA_{ema_long}']) & (hist_data['ADX'] > adx_thresh)
            short_cond = (hist_data[f'EMA_{ema_short}'] < hist_data[f'EMA_{ema_long}']) & (hist_data['ADX'] > adx_thresh)
            # Determine position: 1 for long, -1 for short, 0 for flat (weak trend)
            hist_data['position'] = np.select([long_cond, short_cond], [1, -1], default=0)
            # Fill forward flat periods only if the previous state was trending
            # hist_data['position'] = hist_data['position'].replace(0, np.nan).ffill().fillna(0) # This keeps position during weak trend - might not be desired
            hist_data['signal'] = hist_data['position'].diff().fillna(0) # Signal only on change of position (incl. entering/exiting trend)

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
            hist_data.loc[buy_cond, 'signal'] = 1 # Buy signal
            hist_data.loc[sell_cond, 'signal'] = -1 # Sell signal

        elif strategy == 'volatility_breakout':
            donchian_window = params['donchian_window']
            # Donchian uses High/Low of *previous* N periods
            hist_data['Upper_Channel'] = hist_data['High'].rolling(window=donchian_window).max().shift(1)
            hist_data['Lower_Channel'] = hist_data['Low'].rolling(window=donchian_window).min().shift(1)
            buy_cond = hist_data['Close'] > hist_data['Upper_Channel']
            sell_cond = hist_data['Close'] < hist_data['Lower_Channel']
            hist_data.loc[buy_cond, 'signal'] = 1 # Buy breakout
            hist_data.loc[sell_cond, 'signal'] = -1 # Sell breakout

    except KeyError as e:
        err_msg = f"Error applying strategy logic: Missing expected column - {e}. Check if data download included OHLCV."
        if is_cli_call: print(f"❌ {err_msg}")
        return {"status": "error", "message": err_msg}
    except Exception as e:
        err_msg = f"Unexpected error applying strategy logic: {e}"
        if is_cli_call: print(f"❌ {err_msg}")
        return {"status": "error", "message": err_msg}

    # Drop initial rows where signals/indicators might be NaN due to lookback windows
    first_valid_index = hist_data.dropna(subset=['signal']).index.min() # Find first row with valid signal
    hist_data = hist_data.loc[first_valid_index:]
    if hist_data.empty:
        err_msg = "Error: No valid data remaining after calculating indicators/signals."
        if is_cli_call: print(f"❌ {err_msg}")
        return {"status": "error", "message": err_msg}

    if is_cli_call: print("   -> Simulating trades and calculating equity curves...")
    # --- Trade Simulation ---
    initial_capital = 10000.0
    capital = initial_capital
    position_shares = 0.0 # Shares held, positive for long, negative for short
    hist_data['strategy_equity'] = initial_capital
    # Ensure buy & hold starts calculation from the same point as the strategy
    hist_data['hold_equity'] = initial_capital * (hist_data[price_col] / hist_data[price_col].iloc[0])

    # Iterate through data starting from the second row (index 1)
    for i in range(1, len(hist_data)):
        signal_value = hist_data['signal'].iloc[i]
        current_price = hist_data[price_col].iloc[i]

        # --- Simplified Position Logic ---
        # 1. Buy Signal
        if signal_value > 0:
            if position_shares <= 0: # If flat or short, go long
                 if position_shares < 0: # Close short position first
                     capital += position_shares * current_price # position_shares is negative
                 position_shares = capital / current_price # Allocate all capital to long position
                 capital = 0
        # 2. Sell Signal
        elif signal_value < 0:
            if position_shares >= 0: # If flat or long, go short (if strategy allows)
                if position_shares > 0: # Close long position first
                    capital += position_shares * current_price
                if strategy in ['ma_crossover', 'trend_following', 'rsi', 'mean_reversion', 'volatility_breakout']: # Strategies that allow shorting
                    position_shares = - (capital / current_price) # Allocate all capital to short position
                    capital = 0
                else: # Strategies that don't short (e.g., BUSD) just go flat
                    position_shares = 0

        # Calculate equity for the *end* of the current day i
        current_equity = capital + (position_shares * current_price)
        if not np.isnan(current_equity) and current_equity > 0: # Ensure equity is valid and positive
            hist_data.iloc[i, hist_data.columns.get_loc('strategy_equity')] = current_equity
        else: # If equity becomes invalid, carry forward the last valid value
            hist_data.iloc[i, hist_data.columns.get_loc('strategy_equity')] = hist_data.iloc[i-1, hist_data.columns.get_loc('strategy_equity')]

    # --- Final Metrics Calculation ---
    strategy_return_pct = (hist_data['strategy_equity'].iloc[-1] / initial_capital - 1) * 100
    hold_return_pct = (hist_data['hold_equity'].iloc[-1] / initial_capital - 1) * 100
    strategy_returns = hist_data['strategy_equity'].pct_change().dropna()
    sharpe_ratio = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() != 0 else 0.0
    trade_count = int(np.sum(hist_data['signal'] != 0)) # Count actual signals generated

    if is_cli_call:
        print("\n--- Backtest Results ---")
        results_table = [
            ["Strategy Return", f"{strategy_return_pct:+.2f}%"],
            ["Buy & Hold Return", f"{hold_return_pct:+.2f}%"],
            ["Sharpe Ratio (Annualized)", f"{sharpe_ratio:.3f}"],
            ["Total Trade Signals", f"{trade_count}"]
        ]
        print(tabulate(results_table, tablefmt="fancy_grid"))
        print("------------------------")

        print("   -> Generating results plot...")
        # Plotting is only for CLI calls
        plot_backtest_results(hist_data, ticker, strategy)

    # --- Return results dictionary ---
    return {
        "status": "success",
        "ticker": ticker,
        "strategy": strategy,
        "period": period,
        "parameters": params,
        "total_return_pct": strategy_return_pct,
        "buy_hold_return_pct": hold_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "trade_count": trade_count
    }

# --- Plotting Function ---
def plot_backtest_results(data: pd.DataFrame, ticker: str, strategy: str):
    """
    Generates a two-panel chart visualizing the backtest results.
    """
    try:
        plt.style.use('dark_background')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

        # Use the same price column as the backtest ('Adj Close' or 'Close')
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'

        ax1.plot(data.index, data[price_col], label=f'{ticker} Price', color='grey', linewidth=1.5)

        buy_signals = data[data['signal'] > 0]
        ax1.plot(buy_signals.index, buy_signals[price_col], '^', markersize=10, color='lime', label='Buy Signal', markeredgecolor='black')

        sell_signals = data[data['signal'] < 0]
        ax1.plot(sell_signals.index, sell_signals[price_col], 'v', markersize=10, color='red', label='Sell Signal', markeredgecolor='black')

        ax1.set_title(f"{ticker} Backtest: '{strategy.replace('_', ' ').title()}' Strategy", color='white', fontsize=16)
        ax1.set_ylabel(f"{price_col.replace('_',' ')} Price (USD)", color='white') # Dynamic label
        ax1.legend()
        ax1.grid(True, color='dimgray', linestyle='--', linewidth=0.5, alpha=0.5)
        # Set y-axis limits for price slightly padded
        min_price, max_price = data[price_col].min(), data[price_col].max()
        ax1.set_ylim(min_price * 0.95, max_price * 1.05)


        ax2.plot(data.index, data['strategy_equity'], label='Strategy Equity', color='cyan', linewidth=2)
        ax2.plot(data.index, data['hold_equity'], label='Buy & Hold Equity', color='orange', linestyle='--')

        ax2.set_xlabel("Date", color='white')
        ax2.set_ylabel("Portfolio Value ($)", color='white')
        ax2.legend()
        ax2.grid(True, color='dimgray', linestyle='--', linewidth=0.5, alpha=0.5)
        # Set y-axis limits for equity slightly padded
        min_equity = min(data['strategy_equity'].min(), data['hold_equity'].min())
        max_equity = max(data['strategy_equity'].max(), data['hold_equity'].max())
        ax2.set_ylim(min_equity * 0.95, max_equity * 1.05)

        fig.tight_layout()
        filename = f"backtest_{ticker}_{strategy}_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black', edgecolor='black', dpi=300)
        plt.close(fig)
        print(f"📂 Backtest results chart saved as: {filename}")

    except Exception as e:
        print(f"❌ Error plotting backtest results: {e}")
        # Ensure figure is closed even if plotting fails
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)

# --- Main Command Handler ---
async def handle_backtest_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """
    Handles the /backtest command. Returns results dict for logging/AI, prints for CLI.
    """
    # AI calls might be supported later by passing parameters via ai_params
    if is_called_by_ai:
        return {"status": "error", "message": "AI calls to /backtest are not fully supported for parameter parsing yet."}

    # --- CLI Path ---
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
        return None # Return None for CLI user error

    ticker = args[0].upper()
    strategy = args[1].lower()
    period = args[2].lower() # e.g., "1y", "6mo", "5y"
    strategy_params = {} # Dictionary to hold strategy-specific parameters

    valid_strategies = [
        'ma_crossover', 'rsi', 'busd', 'trend_following',
        'mean_reversion', 'volatility_breakout'
    ]
    if strategy not in valid_strategies:
        print(f"❌ Error: Invalid strategy '{strategy}'. Choose from: {', '.join(valid_strategies)}")
        return None

    # --- Parameter Parsing with Defaults ---
    param_args = args[3:] # Get only the parameter arguments
    try:
        if strategy == 'ma_crossover':
            strategy_params['short_ma'] = int(param_args[0]) if len(param_args) > 0 else 50
            strategy_params['long_ma'] = int(param_args[1]) if len(param_args) > 1 else 200
        elif strategy == 'rsi':
            strategy_params['rsi_period'] = int(param_args[0]) if len(param_args) > 0 else 14
            strategy_params['rsi_buy'] = int(param_args[1]) if len(param_args) > 1 else 30
            strategy_params['rsi_sell'] = int(param_args[2]) if len(param_args) > 2 else 70
        elif strategy == 'busd':
            pass # No parameters needed
        elif strategy == 'trend_following':
            strategy_params['ema_short'] = int(param_args[0]) if len(param_args) > 0 else 25
            strategy_params['ema_long'] = int(param_args[1]) if len(param_args) > 1 else 75
            strategy_params['adx_thresh'] = int(param_args[2]) if len(param_args) > 2 else 25
        elif strategy == 'mean_reversion':
            strategy_params['bb_window'] = int(param_args[0]) if len(param_args) > 0 else 20
            strategy_params['bb_std'] = int(param_args[1]) if len(param_args) > 1 else 2
            strategy_params['rsi_period'] = int(param_args[2]) if len(param_args) > 2 else 14
            strategy_params['rsi_buy'] = int(param_args[3]) if len(param_args) > 3 else 30
            strategy_params['rsi_sell'] = int(param_args[4]) if len(param_args) > 4 else 70
        elif strategy == 'volatility_breakout':
            strategy_params['donchian_window'] = int(param_args[0]) if len(param_args) > 0 else 20

        # Validate parameter values (simple examples)
        if 'short_ma' in strategy_params and strategy_params['short_ma'] >= strategy_params.get('long_ma', 200):
            print("❌ Error: Short MA window must be less than Long MA window.")
            return None
        if 'rsi_buy' in strategy_params and strategy_params['rsi_buy'] >= strategy_params.get('rsi_sell', 70):
            print("❌ Error: RSI Buy Level must be less than Sell Level.")
            return None

    except (ValueError, IndexError):
        # This catches errors if user provides non-numeric params or not enough params
        print("⚠️ Warning: Invalid or missing parameters provided. Using default values for the strategy.")
        # Re-assign defaults explicitly if parsing failed
        if strategy == 'ma_crossover': strategy_params = {'short_ma': 50, 'long_ma': 200}
        elif strategy == 'rsi': strategy_params = {'rsi_period': 14, 'rsi_buy': 30, 'rsi_sell': 70}
        elif strategy == 'busd': strategy_params = {}
        elif strategy == 'trend_following': strategy_params = {'ema_short': 25, 'ema_long': 75, 'adx_thresh': 25}
        elif strategy == 'mean_reversion': strategy_params = {'bb_window': 20, 'bb_std': 2, 'rsi_period': 14, 'rsi_buy': 30, 'rsi_sell': 70}
        elif strategy == 'volatility_breakout': strategy_params = {'donchian_window': 20}

    print(f"-> Starting backtest for {ticker} using '{strategy}' over a {period} period...")
    print(f"   -> Parameters: {strategy_params if strategy_params else 'Defaults'}") # Show actual params used

    # Call core logic and get results dict
    backtest_results = await run_strategy_backtest(ticker, strategy, period, strategy_params, is_cli_call=True)

    # For CLI, results are already printed. Return the dictionary for Prometheus logging.
    return backtest_results
    # --- END CLI Path ---