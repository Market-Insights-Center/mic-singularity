# --- optimize_command.py ---
# Standalone module for the /optimize command.

import asyncio
import os
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO
from tabulate import tabulate
from typing import Optional, List
from pypfopt import EfficientFrontier, risk_models, expected_returns

# --- Dependencies for this Command ---
PORTFOLIO_DB_FILE = 'portfolio_codes_database.csv'

def get_sp500_symbols_singularity() -> List[str]:
    """Fetches S&P 500 symbols from Wikipedia."""
    try:
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(sp500_url, headers=headers, timeout=15)
        response.raise_for_status()
        dfs = pd.read_html(StringIO(response.text))
        sp500_df = dfs[0]
        symbols = [str(s).replace('.', '-') for s in sp500_df['Symbol'].tolist() if isinstance(s, str)]
        return sorted(list(set(s for s in symbols if s)))
    except Exception:
        return []

async def get_specific_index_tickers(index_symbol: str) -> List[str]:
    """Fetches component tickers for a specific major index."""
    index_symbol = index_symbol.upper()
    if index_symbol == 'SPY':
        return await asyncio.to_thread(get_sp500_symbols_singularity)

    urls = {
        'QQQ': ('https://en.wikipedia.org/wiki/Nasdaq-100', 4, 'Ticker'),
        'DIA': ('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average', 1, 'Symbol'),
        'RUT': ('https://en.wikipedia.org/wiki/List_of_Russell_2000_stocks', 2, 'Ticker')
    }
    if index_symbol not in urls:
        return []

    url, table_index, col_name = urls[index_symbol]
    try:
        tables = await asyncio.to_thread(pd.read_html, url)
        df = tables[table_index]
        tickers = [str(s).replace('.', '-') for s in df[col_name].tolist() if isinstance(s, str)]
        return sorted(list(set(s for s in tickers if s)))
    except Exception:
        return []

# --- Main Command Handler ---
async def handle_optimize_command(args: list, ai_params: dict = None, is_called_by_ai: bool = False):
    """
    Calculates the optimal portfolio weights for maximum Sharpe ratio using PyPortfolioOpt.
    """
    if not is_called_by_ai:
        print("\n--- Portfolio Optimization (Max Sharpe Ratio) ---")

    if not args:
        print("Usage: /optimize <TICKER1 TICKER2 ...> or /optimize P-<PortfolioCode>")
        return

    input_arg = args[0].upper()
    initial_ticker_list = []
    source_description = ""

    # Step 1: Get the list of tickers
    if input_arg.startswith('P-'):
        portfolio_code = input_arg.split('P-')[1]
        source_description = f"Portfolio Code '{portfolio_code}'"
        if not os.path.exists(PORTFOLIO_DB_FILE):
            print(f"❌ Error: Portfolio database file '{PORTFOLIO_DB_FILE}' not found.")
            return
        try:
            df = pd.read_csv(PORTFOLIO_DB_FILE)
            portfolio_row = df[df['portfolio_code'].astype(str) == portfolio_code]
            if portfolio_row.empty:
                print(f"❌ Error: Portfolio code '{portfolio_code}' not found.")
                return
            p_series = portfolio_row.iloc[0]
            num_portfolios = int(p_series.get('num_portfolios', 0))
            temp_tickers = set()
            for i in range(1, num_portfolios + 1):
                tickers_str = p_series.get(f'tickers_{i}', '')
                if tickers_str and isinstance(tickers_str, str):
                    temp_tickers.update([t.strip() for t in tickers_str.split(',') if t.strip()])
            initial_ticker_list = sorted(list(temp_tickers))
        except Exception as e:
            print(f"❌ Error reading portfolio data: {e}")
            return
    elif input_arg.startswith('I-'):
         index_symbol = input_arg.split('I-')[1]
         source_description = f"Index '{index_symbol}'"
         initial_ticker_list = await get_specific_index_tickers(index_symbol)
    else:
        initial_ticker_list = [arg.upper() for arg in args]
        source_description = "user-provided list"

    if not initial_ticker_list or len(initial_ticker_list) < 2:
        print("❌ Error: At least two valid tickers are required for optimization.")
        return

    print(f"-> Optimizing for {len(initial_ticker_list)} tickers from {source_description}...")

    # Step 2: Fetch historical data
    print("-> Fetching 3-year historical price data...")
    try:
        prices = await asyncio.to_thread(
            yf.download, tickers=initial_ticker_list, period="3y", progress=False
        )
        if prices.empty or 'Close' not in prices.columns:
            raise ValueError("No data returned from yfinance.")
        prices = prices['Close'].dropna(axis=1) # Drop tickers with no data
        if len(prices.columns) < 2:
            print("❌ Error: Not enough tickers had sufficient historical data for optimization.")
            return

    except Exception as e:
        print(f"❌ Error fetching historical price data: {e}")
        return

    # Step 3: Calculate expected returns and covariance
    mu = expected_returns.mean_historical_return(prices)
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    # Step 4: Optimize for max Sharpe ratio
    ef = EfficientFrontier(mu, S)
    try:
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        return

    # Step 5: Display results
    print("\n--- Optimal Portfolio Weights (Max Sharpe Ratio) ---")
    weights_data = []
    sorted_weights = sorted(cleaned_weights.items(), key=lambda item: item[1], reverse=True)

    for ticker, weight in sorted_weights:
        if weight * 100 >= 0.01:
            weights_data.append([ticker, f"{weight * 100:.2f}%"])
    print(tabulate(weights_data, headers=["Ticker", "Optimal Weight"], tablefmt="pretty"))

    print("\n--- Expected Portfolio Performance ---")
    ef.portfolio_performance(verbose=True)
    print("\nOptimization complete.")