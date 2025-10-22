# fairvalue_command.py

# --- Imports for Fair Value Command ---
import yfinance as yf
import pandas as pd
import asyncio
import re
from typing import List, Optional
from datetime import datetime, timedelta
import traceback
from io import StringIO
from math import sqrt

# --- Global Dependencies & Constants ---
YFINANCE_API_SEMAPHORE = asyncio.Semaphore(8)

# --- Helper Functions ---

def get_start_date_from_period(period: str) -> Optional[datetime]:
    """
    Robustly converts a period string (e.g., '1y', '3mo', '7d') to a start date.
    """
    match = re.match(r"(\d+)([a-zA-Z]+)", period.lower())
    if not match:
        return None

    try:
        value = int(match.group(1))
        unit = match.group(2)
        now = datetime.now()

        if unit == 'd':
            return now - timedelta(days=value)
        elif unit == 'w':
            return now - timedelta(weeks=value)
        elif unit == 'mo':
            return now - timedelta(days=value * 30.44) # Approximate months
        elif unit == 'y':
            return now - timedelta(days=value * 365.25) # Approximate years
        else:
            return None
    except (ValueError, IndexError):
        return None

async def calculate_invest_score_at_date(ticker: str, target_date: datetime, sensitivity: int) -> Optional[float]:
    """
    Calculates the EMA-based INVEST score for a ticker at a specific historical date.
    """
    async with YFINANCE_API_SEMAPHORE:
        sensitivity_map = {
            1: {"period": "max", "interval": "1wk"},
            2: {"period": "10y", "interval": "1d"},
            3: {"period": "730d", "interval": "1h"},
        }
        if sensitivity not in sensitivity_map:
            # print(f"[DEBUG] Error: Invalid sensitivity '{sensitivity}' passed to score calculator.")
            return None

        params = sensitivity_map[sensitivity]
        stock = yf.Ticker(ticker.replace('.', '-'))

        try:
            data = await asyncio.to_thread(stock.history, period=params["period"], interval=params["interval"])
            if data.empty:
                return None

            data.index = data.index.tz_localize(None)
            data = data[~data.index.duplicated(keep='last')]
            
            data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
            data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()

            data_at_date = data.asof(target_date)

            if not isinstance(data_at_date, pd.Series) or data_at_date.empty:
                 return None

            if pd.isna(data_at_date.get('Close')) or pd.isna(data_at_date.get('EMA_8')) or pd.isna(data_at_date.get('EMA_55')):
                return None

            ema_8, ema_55 = data_at_date['EMA_8'], data_at_date['EMA_55']
            
            if pd.isna(ema_55) or ema_55 == 0:
                return None
                
            invest_score = (((ema_8 - ema_55) / ema_55) * 4 + 0.5) * 100
            return float(invest_score)
            
        except Exception as e:
            return None

# --- Main Handler for /fairvalue ---

async def handle_fairvalue_command(args: List[str], is_called_by_ai: bool = False):
    """
    Calculates a 'Fair Price' for a ticker based on the relative change
    in its price versus its INVEST score over a specified time period and sensitivity.
    """
    if len(args) != 3:
        print("\nUsage: /fairvalue <TICKER> <period> <sensitivity>")
        print("Example: /fairvalue AAPL 1y 2")
        print("  <TICKER>: Stock symbol (e.g., NVDA)")
        print("  <period>: 1d, 7d, 1w, 1mo, 3mo, 1y, 5y")
        print("  <sensitivity>: 1 (Weekly), 2 (Daily), 3 (Hourly)")
        return

    ticker = args[0].upper()
    period = args[1].lower()
    try:
        sensitivity = int(args[2])
        if sensitivity not in [1, 2, 3]:
            raise ValueError
    except (ValueError, IndexError):
        print("Error: Sensitivity must be a single integer: 1, 2, or 3.")
        return

    print(f"\n--- Fair Value Analysis for {ticker} (Period: {period}, Sensitivity: {sensitivity}) ---")

    end_date = datetime.now()
    start_date = get_start_date_from_period(period)

    if not start_date:
        print(f"Error: Invalid time period '{period}'.")
        return
    
    try:
        fetch_start_date = start_date - timedelta(days=10)
        # Added auto_adjust=True to handle the FutureWarning
        stock_data = await asyncio.to_thread(yf.download, tickers=[ticker], start=fetch_start_date, end=end_date, progress=False, auto_adjust=True)
        
        if not stock_data.empty:
            stock_data = stock_data[~stock_data.index.duplicated(keep='last')]

        if stock_data.empty or len(stock_data) < 2:
            print(f"Error: Could not fetch sufficient price data for {ticker} in the given period.")
            return

        start_price_row = stock_data.asof(start_date)
        
        if not isinstance(start_price_row, pd.Series) or start_price_row.empty:
            print(f"Error: Could not find a valid price data point at or before the start date {start_date.strftime('%Y-%m-%d')}.")
            return

        start_price = start_price_row['Close'].item()
        end_price = stock_data['Close'].iloc[-1].item()
            
        if pd.isna(start_price):
            print(f"Error: Found a valid data row for the start date, but the 'Close' price was NaN.")
            return

        end_score = await calculate_invest_score_at_date(ticker, end_date, sensitivity)

        if end_score is None:
            print("\nError: Failed to generate the current INVEST score. Cannot continue analysis.")
            return

        price_change_pct = ((end_price - start_price) / start_price) * 100
        
        print(f"\nAnalysis Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Price Change:        {price_change_pct:+.2f}% (${start_price:.2f} -> ${end_price:.2f})")
        print(f"Current INVEST Score: {end_score:.2f}")

        # --- Final, Corrected Calculation Logic ---
        # 1. Check for division by zero.
        if abs(price_change_pct) < 1e-9:
            print("\nCannot calculate Valuation Factor: Price percent change over the period is zero.")
            return

        # 2. Calculate the argument, using abs() to prevent math domain errors.
        argument = (end_score - 50) / price_change_pct
        valuation_factor = sqrt(abs(argument))

        # 3. Calculate the Fair Price.
        fair_price = valuation_factor * end_price

        print("\n--- Results ---")
        print(f"Valuation Factor: {valuation_factor:.4f}")
        print(f"Current Price:    ${end_price:.2f}")
        print(f"Estimated Fair Price: ${fair_price:.2f}")
        
        # 4. Determine the conclusion.
        if fair_price < 0:
            print("Conclusion: The calculation resulted in a negative fair price, which is not meaningful.")
        elif fair_price > end_price:
            print("Conclusion: The stock may be UNDERVALUED.")
        else:
            print("Conclusion: The stock may be OVERVALUED.")

    except Exception as e:
        print(f"\nAn unexpected error occurred in the main handler: {e}")
        traceback.print_exc()