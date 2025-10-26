# --- Imports for powerscore_command ---
import asyncio
import traceback
import math
from typing import List, Dict, Any, Optional, Tuple
import random
import configparser

import yfinance as yf
import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.stats import percentileofscore
from sklearn.ensemble import RandomForestRegressor
import google.generativeai as genai

# --- Imports from other command modules ---
from invest_command import calculate_ema_invest
# Corrected import for sentiment - ensure gemini_model/lock can be passed if needed
from sentiment_command import handle_sentiment_command

# --- Global Variables & Constants ---
YFINANCE_API_SEMAPHORE = asyncio.Semaphore(8)
GEMINI_API_LOCK = asyncio.Lock() # Keep the lock here if sentiment uses it

# --- Initialize Gemini Model within this module ---
# Keep your existing Gemini initialization logic here
gemini_model = None
try:
    config = configparser.ConfigParser()
    config.read('config.ini')
    GEMINI_API_KEY = config.get('API_KEYS', 'GEMINI_API_KEY', fallback=None)
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        # Using the correct model name
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite') # Or 'gemini-pro' etc.
except Exception as e:
    print(f"Warning: Could not configure Gemini model in powerscore_command.py: {e}")

# --- Helper Functions (keep existing helpers as they are) ---
# safe_get, get_yfinance_info_robustly, get_yf_download_robustly, get_yf_data_singularity,
# calculate_portfolio_beta_correlation_singularity, get_single_stock_beta_corr,
# get_market_invest_score_for_powerscore, calculate_volatility_metrics,
# handle_fundamentals_command (simplified internal version),
# calculate_technical_indicators, handle_mlforecast_command (simplified internal version),
# get_powerscore_explanation

# --- (Paste ALL your existing helper functions here) ---
def safe_get(data_dict: Dict, key: str, default: Any = None) -> Any:
    value = data_dict.get(key, default)
    return default if value is None or value == 'None' else value

async def get_yfinance_info_robustly(ticker: str) -> Optional[Dict[str, Any]]:
    async with YFINANCE_API_SEMAPHORE:
        for attempt in range(3):
            try:
                # Use asyncio.sleep for non-blocking delay
                await asyncio.sleep(random.uniform(0.2, 0.5))
                # Run blocking yfinance call in a separate thread
                stock_info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)
                # More robust check for valid info
                if stock_info and ('regularMarketPrice' in stock_info or 'currentPrice' in stock_info):
                    return stock_info
                else:
                    # Raise ValueError if data is incomplete, triggering retry
                    raise ValueError(f"Incomplete data received for {ticker}")
            except Exception as e:
                if attempt < 2:
                    # Exponential backoff for retries
                    await asyncio.sleep((attempt + 1) * 2)
                # Optional: Log the final error after all retries fail
                # else: print(f"   -> ❌ ERROR: All attempts to fetch .info for {ticker} failed. Last error: {type(e).__name__}")
        return None # Return None if all retries fail

async def get_yf_download_robustly(tickers: list, **kwargs) -> pd.DataFrame:
    """
    A robust wrapper for yf.download with retry logic and exponential backoff
    to handle transient network errors and API instability.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(random.uniform(0.3, 0.7)) # Stagger requests

            # Ensure progress=False is explicitly passed if not in kwargs
            kwargs.setdefault('progress', False)

            data = await asyncio.to_thread(
                yf.download,
                tickers=tickers,
                **kwargs # Pass other arguments like period, interval, etc.
            )

            # Check if empty, especially for single tickers which might fail temporarily
            if data.empty and len(tickers) == 1:
                 # yfinance sometimes returns empty for single valid tickers temporarily
                 raise IOError(f"yf.download returned empty DataFrame for single ticker: {tickers[0]}")

            # Do not immediately raise error for empty on multi-ticker downloads.
            # Check later if *all* failed if necessary.

            return data # Success (even if partially empty for multi-ticker)

        except Exception as e:
            if attempt < max_retries - 1:
                delay = (attempt + 1) * 3 # Backoff: 3s, 6s
                print(f"   -> WARNING: yf.download failed (Attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                # Log the final error after all attempts fail
                print(f"   -> ❌ ERROR: All yfinance download attempts failed for {tickers}. Last error: {type(e).__name__}")
                return pd.DataFrame() # Return empty DataFrame on persistent failure
    return pd.DataFrame() # Should be unreachable, but ensures return

async def get_yf_data_singularity(tickers: List[str], period: str = "10y", interval: str = "1d", is_called_by_ai: bool = False) -> pd.DataFrame:
    """
    Downloads historical closing price data for multiple tickers using the robust wrapper.
    Ensures the output DataFrame has datetime index and single-level column index (ticker names).
    """
    if not tickers:
        return pd.DataFrame()

    # Call the robust download wrapper
    data = await get_yf_download_robustly(
        tickers=list(set(tickers)), period=period, interval=interval,
        auto_adjust=False, # Keep False for Close/Adj Close consistency if needed
        group_by='ticker', # Groups data by ticker
        timeout=30 # Example timeout
    )

    if data.empty:
        return pd.DataFrame()

    all_series = []
    # Handle MultiIndex columns (when group_by='ticker' or multiple tickers)
    if isinstance(data.columns, pd.MultiIndex):
        for ticker_name in list(set(tickers)): # Iterate through requested tickers
            # Check if both ticker and 'Close' column exist
            if (ticker_name, 'Close') in data.columns:
                # Select the 'Close' series, convert to numeric (handle errors), drop NaNs
                series = pd.to_numeric(data[(ticker_name, 'Close')], errors='coerce').dropna()
                if not series.empty:
                    series.name = ticker_name # Set series name to ticker
                    all_series.append(series)
    # Handle single ticker case (plain DataFrame)
    elif 'Close' in data.columns:
        series = pd.to_numeric(data['Close'], errors='coerce').dropna()
        if not series.empty:
            # Ensure name is set correctly for single ticker
            series.name = list(set(tickers))[0]
            all_series.append(series)

    # If no valid series were extracted, return empty DataFrame
    if not all_series:
        return pd.DataFrame()

    # Concatenate all valid series into a single DataFrame
    df_out = pd.concat(all_series, axis=1)

    # Ensure index is DatetimeIndex (yf usually does this, but good practice)
    df_out.index = pd.to_datetime(df_out.index)

    # Drop rows/columns that are entirely NaN (can happen with differing histories)
    return df_out.dropna(axis=0, how='all').dropna(axis=1, how='all')

async def calculate_portfolio_beta_correlation_singularity(
    portfolio_holdings: List[Dict[str, Any]], # List of {'ticker': str, 'value': float}
    total_portfolio_value: float,
    backtest_period: str, # e.g., "1y", "3mo"
    is_called_by_ai: bool = False # Keep for potential internal use/logging
) -> Optional[tuple[float, float]]:
    """
    Calculates weighted average Beta and Correlation for a given portfolio against SPY.
    `portfolio_holdings`: List of dicts, each with 'ticker' and 'value' (actual dollar allocation).
    Cash should be one of the tickers with its value if present.
    Returns a tuple (beta, correlation) or None if calculation fails.
    """
    if not portfolio_holdings or total_portfolio_value <= 0:
        print("   -> [DEBUG Beta/Corr] Calculation skipped: No holdings or zero/negative total value.")
        return None

    # Filter out holdings with zero or negligible value
    valid_holdings_for_calc = [h for h in portfolio_holdings if isinstance(h.get('value'), (int, float)) and h['value'] > 1e-9]

    if not valid_holdings_for_calc:
        # Check if the only holding(s) is CASH
        is_only_cash = all(h['ticker'].upper() == 'CASH' for h in portfolio_holdings if h.get('ticker'))
        if is_only_cash:
            print("   -> [DEBUG Beta/Corr] Portfolio is 100% cash. Beta=0, Corr=0.")
            return 0.0, 0.0 # Beta and Correlation of cash is 0
        print("   -> [DEBUG Beta/Corr] Calculation skipped: No valid holdings with positive value found.")
        return None # No valid non-cash holdings

    # Extract stock tickers, excluding CASH
    portfolio_stock_tickers_assess = [h['ticker'] for h in valid_holdings_for_calc if h.get('ticker') and h['ticker'].upper() != 'CASH']

    # If only cash was provided (after filtering small values), return 0, 0
    if not portfolio_stock_tickers_assess:
        print("   -> [DEBUG Beta/Corr] Portfolio effectively 100% cash after filtering. Beta=0, Corr=0.")
        return 0.0, 0.0

    # Fetch historical data for stocks and SPY
    all_tickers_for_hist_fetch = list(set(portfolio_stock_tickers_assess + ['SPY']))
    hist_data_assess = await get_yf_data_singularity(
        all_tickers_for_hist_fetch,
        period=backtest_period,
        interval="1d",
        is_called_by_ai=True # Suppress prints from this helper during internal call
    )

    # Validate fetched data
    if hist_data_assess.empty or 'SPY' not in hist_data_assess.columns or hist_data_assess['SPY'].isnull().all() or len(hist_data_assess['SPY'].dropna()) < 20:
        print(f"   -> [DEBUG Beta/Corr] Calculation failed: Insufficient historical data or SPY data missing/invalid for period {backtest_period}.")
        return None

    # Calculate daily returns (percentage change)
    daily_returns_assess_df = hist_data_assess.pct_change(fill_method=None).iloc[1:] # Use iloc[1:] to drop first NaN row

    if daily_returns_assess_df.empty or 'SPY' not in daily_returns_assess_df.columns:
        print("   -> [DEBUG Beta/Corr] Calculation failed: Returns calculation resulted in empty data or missing SPY returns.")
        return None

    spy_returns_series = daily_returns_assess_df['SPY'].dropna()
    # Check if SPY returns have variance (required for beta calculation)
    if spy_returns_series.empty or spy_returns_series.var() < 1e-12: # Use variance check instead of std dev
        print("   -> [DEBUG Beta/Corr] Calculation failed: SPY returns have zero variance.")
        return None

    stock_metrics_calculated = {} # Store calculated beta/corr for each ticker
    market_variance = spy_returns_series.var() # Pre-calculate market variance

    for ticker_met in portfolio_stock_tickers_assess:
        beta_val, correlation_val = np.nan, np.nan # Default to NaN
        if ticker_met in daily_returns_assess_df.columns and not daily_returns_assess_df[ticker_met].isnull().all():
            ticker_returns_series = daily_returns_assess_df[ticker_met].dropna()

            # Align ticker returns with SPY returns by index (date)
            aligned_data = pd.concat([ticker_returns_series, spy_returns_series], axis=1, join='inner').dropna()

            # Need sufficient overlapping data points
            if len(aligned_data) >= 20:
                aligned_ticker_returns = aligned_data.iloc[:, 0]
                aligned_spy_returns = aligned_data.iloc[:, 1]

                # Check if ticker returns also have variance
                if aligned_ticker_returns.var() > 1e-12:
                    try:
                        # Calculate covariance between ticker and SPY
                        covariance_matrix = np.cov(aligned_ticker_returns, aligned_spy_returns)
                        if covariance_matrix.shape == (2, 2):
                             # Beta = Cov(Stock, Market) / Var(Market)
                             beta_val = covariance_matrix[0, 1] / market_variance

                        # Calculate correlation coefficient
                        correlation_coef_matrix = np.corrcoef(aligned_ticker_returns, aligned_spy_returns)
                        if correlation_coef_matrix.shape == (2, 2):
                            correlation_val = correlation_coef_matrix[0, 1]
                            # Handle potential NaN correlation if data is perfectly linear (rare)
                            if pd.isna(correlation_val): correlation_val = 0.0 # Treat as uncorrelated if calculation fails
                    except (ValueError, IndexError, TypeError, np.linalg.LinAlgError) as calc_err:
                        print(f"   -> [DEBUG Beta/Corr] Calculation warning for {ticker_met}: {calc_err}")
                        # Keep beta/corr as NaN if calculation fails
                else:
                    # If ticker has no variance, beta/corr are effectively 0
                    beta_val, correlation_val = 0.0, 0.0
        stock_metrics_calculated[ticker_met] = {'beta': beta_val, 'correlation': correlation_val}

    # Add CASH metrics (Beta=0, Corr=0)
    stock_metrics_calculated['CASH'] = {'beta': 0.0, 'correlation': 0.0}

    # Calculate weighted averages
    weighted_beta_sum, weighted_correlation_sum = 0.0, 0.0
    total_weight_accounted = 0.0 # Track weight used in calculation

    for holding in valid_holdings_for_calc:
        ticker_h = holding.get('ticker','UNKNOWN').upper()
        value_h = holding['value']
        weight_h = value_h / total_portfolio_value
        total_weight_accounted += weight_h

        metrics_for_ticker = stock_metrics_calculated.get(ticker_h)

        if metrics_for_ticker:
            beta_for_calc = metrics_for_ticker.get('beta', 0.0) # Default to 0 if beta wasn't calculable
            corr_for_calc = metrics_for_ticker.get('correlation', 0.0) # Default to 0

            # Only add to sum if the metric was valid (not NaN)
            if not pd.isna(beta_for_calc):
                 weighted_beta_sum += weight_h * beta_for_calc
            else:
                 print(f"   -> [DEBUG Beta/Corr] Excluding {ticker_h} from Beta average (calculation failed).")

            if not pd.isna(corr_for_calc):
                 weighted_correlation_sum += weight_h * corr_for_calc
            else:
                 print(f"   -> [DEBUG Beta/Corr] Excluding {ticker_h} from Correlation average (calculation failed).")
        else:
             # This case should ideally not happen if CASH is handled, but as a safeguard:
             print(f"   -> [DEBUG Beta/Corr] Warning: Metrics not found for holding {ticker_h}. Excluding from average.")


    # Optional: Check if total weight used is close to 1 (if cash wasn't explicitly included)
    # if not any(h['ticker'].upper() == 'CASH' for h in valid_holdings_for_calc):
    #     if abs(total_weight_accounted - 1.0) > 0.01: # Allow 1% tolerance
    #          print(f"   -> [DEBUG Beta/Corr] Warning: Total weight used ({total_weight_accounted:.2f}) doesn't sum to 1. Results might be skewed if cash wasn't included.")

    print(f"   -> [DEBUG Beta/Corr] Calculation successful: Beta={weighted_beta_sum:.4f}, Corr={weighted_correlation_sum:.4f}")
    return weighted_beta_sum, weighted_correlation_sum

async def get_single_stock_beta_corr(ticker: str, period: str, is_called_by_ai: bool = True) -> tuple[Optional[float], Optional[float]]:
    """Calculates Beta and Correlation for a single stock against SPY."""
    # Simulate a 100% portfolio of this stock
    portfolio = [{'ticker': ticker, 'value': 100.0}]
    # Call the portfolio calculation function
    result = await calculate_portfolio_beta_correlation_singularity(portfolio, 100.0, period, is_called_by_ai)
    return result if result else (None, None) # Return tuple or (None, None)

async def get_market_invest_score_for_powerscore() -> Optional[float]:
    """Calculates the Market Invest Score based on VIX and SPY EMA."""
    try:
        # Fetch VIX price (using robust helper)
        vix_data = await get_yf_download_robustly(['^VIX'], period="5d")
        if vix_data.empty or 'Close' not in vix_data.columns: return None
        vix_price = vix_data['Close'].iloc[-1]
        vix_likelihood = np.clip(0.01384083 * (vix_price ** 2), 0, 100)

        # Fetch SPY monthly data (using robust helper)
        spy_hist = await get_yf_download_robustly(['SPY'], period="5y", interval="1mo")
        if spy_hist.empty or 'Close' not in spy_hist.columns or len(spy_hist) < 55: return None # Need enough data for EMA

        spy_hist['EMA_8'] = spy_hist['Close'].ewm(span=8, adjust=False).mean()
        spy_hist['EMA_55'] = spy_hist['Close'].ewm(span=55, adjust=False).mean()

        # Check for NaN in the last row EMAs
        if spy_hist[['EMA_8', 'EMA_55']].iloc[-1].isna().any(): return None

        ema_8, ema_55 = spy_hist['EMA_8'].iloc[-1], spy_hist['EMA_55'].iloc[-1]

        # Avoid division by zero
        if ema_55 == 0: return None

        # Calculate EMA-based likelihood
        x_value = (((ema_8 - ema_55) / ema_55) + 0.5) * 100
        ema_likelihood = np.clip(100 * np.exp(-((45.622216 * x_value / 2750) ** 4)), 0, 100)

        # Avoid division by zero for ratio
        if ema_likelihood < 1e-9: # Use a small threshold instead of exact zero
            # If EMA likelihood is near zero, behavior depends on VIX likelihood
            # High VIX -> very low score (approaching 0)
            # Low VIX -> less extreme low score, maybe capped at 0 or slightly above
            return 0.0 # Cap at 0 if denominator is essentially zero

        # Calculate Market Invest Score
        ratio = vix_likelihood / ema_likelihood
        market_invest_score = 50.0 - (ratio - 1.0) * 100.0

        # Clip the result to the 0-100 range and return as float
        return float(np.clip(market_invest_score, 0, 100))
    except Exception as e:
        print(f"   -> [DEBUG PowerScore] Error calculating Market Invest Score: {e}")
        return None # Return None on any unexpected error

async def calculate_volatility_metrics(ticker: str, period: str) -> tuple[Optional[float], Optional[float]]:
    """Calculates Historical Volatility Rank. Returns (None, vol_rank)."""
    # IV calculation is removed for simplicity within PowerScore context
    try:
        # Use robust download for history data
        hist_data = await get_yf_download_robustly([ticker], period=period)
        if hist_data.empty or 'Close' not in hist_data.columns or len(hist_data) <= 30:
            return None, None # Not enough data

        # Calculate daily returns
        hist_data['daily_return'] = hist_data['Close'].pct_change()

        # Calculate 30-day rolling annualized historical volatility
        rolling_hv = hist_data['daily_return'].rolling(window=30).std() * (252**0.5)
        hv_series = rolling_hv.dropna() # Drop initial NaNs

        # Calculate percentile rank of the latest volatility value
        if len(hv_series) > 1:
            # Calculate percentile rank of the last value within the series
            vol_rank = percentileofscore(hv_series, hv_series.iloc[-1], kind='rank') # 'rank' handles ties
        else:
            vol_rank = None # Cannot rank with less than 2 points

        return None, vol_rank # Return None for IV, and the calculated rank
    except Exception as e:
        print(f"   -> [DEBUG PowerScore] Error calculating Volatility Rank for {ticker}: {e}")
        return None, None # Return None for both on error


async def handle_fundamentals_command_internal(ai_params: dict, is_called_by_ai: bool = True):
    """Internal, simplified version of fundamentals for PowerScore."""
    ticker = ai_params.get("ticker")
    if not ticker: return {"error": "Ticker not provided"}
    # Use robust info fetch
    info = await get_yfinance_info_robustly(ticker)
    if not info: return {"error": f"Could not retrieve data for {ticker}"}

    # Safely get fundamental data using safe_get helper
    pe = safe_get(info, 'trailingPE');
    rg = safe_get(info, 'revenueGrowth');
    de = safe_get(info, 'debtToEquity');
    pm = safe_get(info, 'profitMargins')

    total_score, possible_score = 0.0, 0.0

    # Calculate scores, ensuring checks for None before calculation
    if pe is not None:
        possible_score += 25
        # Ensure pe is float/int before math operations
        try: pe_num = float(pe); total_score += 25 * np.exp(-0.00042 * pe_num**2) if pe_num > 0 else 0
        except (ValueError, TypeError): pass # Skip score part if conversion fails
    if rg is not None:
        possible_score += 25
        try: rg_num = float(rg); total_score += 25 / (1 + np.exp(-0.11 * ((rg_num * 100) - 12.5)))
        except (ValueError, TypeError): pass
    if de is not None:
        possible_score += 25
        try: de_num = float(de); total_score += 25 * np.exp(-0.00956 * de_num) if de_num > 0 else 25
        except (ValueError, TypeError): pass
    if pm is not None:
        possible_score += 25
        try: pm_num = float(pm); total_score += 25 / (1 + np.exp(-0.11 * ((pm_num * 100) - 12.5)))
        except (ValueError, TypeError): pass

    # Calculate final score, avoiding division by zero
    final_score = (total_score / possible_score) * 100 if possible_score > 0 else 0.0

    return {"fundamental_score": final_score}

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculates RSI, MACD, SMA Diff, Volatility for ML Forecast."""
    try:
        if 'Close' not in data.columns: raise KeyError("Missing 'Close' column")

        # RSI (using rolling mean for stability, similar to original)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean() # min_periods=1
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        with np.errstate(divide='ignore', invalid='ignore'): rs = gain / loss
        rs.replace([np.inf, -np.inf], np.nan, inplace=True) # Replace inf with NaN first
        rs.fillna(method='ffill', inplace=True) # Forward fill NaNs if needed
        rs.fillna(0, inplace=True) # Fill remaining NaNs at the start with 0
        data['RSI'] = 100 - (100 / (1 + rs))
        data['RSI'].fillna(50, inplace=True) # Fill any remaining NaNs with neutral 50

        # MACD
        exp1 = data['Close'].ewm(span=12, adjust=False).mean()
        exp2 = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = exp1 - exp2
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

        # SMA Difference
        sma50 = data['Close'].rolling(window=50, min_periods=1).mean()
        sma200 = data['Close'].rolling(window=200, min_periods=1).mean()
        # Avoid division by zero or NaN issues in SMA_Diff
        data['SMA_Diff'] = np.where(sma200 != 0, ((sma50 - sma200) / sma200) * 100, 0)
        data['SMA_Diff'].fillna(0, inplace=True) # Fill NaNs resulting from sma200 being NaN

        # Volatility (Annualized Standard Deviation of Daily Returns)
        # Ensure pct_change doesn't produce leading NaN issues
        daily_returns = data['Close'].pct_change()
        data['Volatility'] = daily_returns.rolling(window=30, min_periods=10).std() * np.sqrt(252) # Lower min_periods
        data['Volatility'].fillna(method='ffill', inplace=True) # Fill early NaNs
        data['Volatility'].fillna(0, inplace=True) # Fill any remaining NaNs

    except KeyError as ke:
         print(f"   -> [DEBUG Tech Indicators] Missing column: {ke}")
    except Exception as e:
         print(f"   -> [DEBUG Tech Indicators] Error calculating indicators: {e}")
         # Ensure columns exist even if calculation fails, filled with NaN or 0
         for col in ['RSI', 'MACD', 'MACD_Signal', 'SMA_Diff', 'Volatility']:
             if col not in data: data[col] = np.nan

    return data


async def handle_mlforecast_command_internal(ai_params: dict, is_called_by_ai: bool = True):
    """Internal, simplified ML Forecast for PowerScore."""
    ticker = ai_params.get("ticker")
    if not ticker: return [] # Return empty list on error for PowerScore

    # Use robust download for historical data
    data = await get_yf_download_robustly([ticker], period="5y", auto_adjust=True) # auto_adjust=True for simplicity here

    # Basic data validation
    if data.empty or len(data) < 252: # Need at least a year for meaningful features
        print(f"   -> [DEBUG MLForecast] Insufficient data for {ticker} (got {len(data)} rows).")
        return []
    # Handle potential MultiIndex if robust download didn't flatten (though it should)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Calculate technical indicators
    data = calculate_technical_indicators(data)
    features = ['RSI', 'MACD', 'MACD_Signal', 'SMA_Diff', 'Volatility']

    # Check if all required features were calculated successfully
    if not all(feature in data.columns for feature in features):
         print(f"   -> [DEBUG MLForecast] Feature calculation failed for {ticker}. Missing columns.")
         return []

    results = []
    # Define horizons for PowerScore (can be adjusted)
    forecast_horizons_internal = {
        "5-Day": 5,
        "1-Month (21-Day)": 21,
        # Add more if needed by different sensitivities later
        # "3-Month (63-Day)": 63,
        # "6-Month (126-Day)": 126, # ~6 months
        # "1-Year (252-Day)": 252, # ~1 year
    }

    for period_name, horizon in forecast_horizons_internal.items():
        try:
            df = data.copy()
            # Calculate future close and percentage change
            df['Future_Close'] = df['Close'].shift(-horizon)
            # Avoid division by zero/NaN for Pct_Change
            df['Pct_Change'] = np.where(df['Close'] != 0, (df['Future_Close'] - df['Close']) / df['Close'], 0)

            # Drop rows with NaN in features or target variable
            df.dropna(subset=features + ['Pct_Change'], inplace=True)

            # Check if enough data remains after dropping NaNs and shifts
            if len(df) < 50: # Need sufficient data for training
                print(f"   -> [DEBUG MLForecast] Skipping {period_name} for {ticker}: Not enough training data ({len(df)} rows).")
                continue

            X, y_magnitude = df[features], df['Pct_Change']

            # Train Random Forest Regressor
            # Use slightly fewer estimators for faster internal calls
            reg = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1, max_depth=10)
            reg.fit(X, y_magnitude)

            # Predict using the latest available features
            # Ensure the last row doesn't have NaNs in features before predicting
            last_features_row = data[features].iloc[-1:]
            if last_features_row.isnull().any().any():
                 print(f"   -> [DEBUG MLForecast] Skipping {period_name} for {ticker}: NaN found in latest features.")
                 continue

            magnitude_pred = reg.predict(last_features_row)[0] * 100

            results.append({
                "Period": period_name,
                "Est. % Change": f"{magnitude_pred:+.2f}%" # Format as string consistent with original
            })
        except Exception as e_period:
            print(f"   -> [DEBUG MLForecast] Error during {period_name} forecast for {ticker}: {e_period}")
            # Optionally add an error entry to results if needed
            # results.append({"Period": period_name, "Est. % Change": "Error"})

    return results # Return list of results (or empty list)

async def get_powerscore_explanation(ticker: str, component_scores: dict, model_to_use: Any, lock_to_use: asyncio.Lock) -> str:
    """Generates an AI summary for the PowerScore components."""
    if not model_to_use:
        return "AI model unavailable."

    # Prepare the prompt with available scores, handling None values
    score_lines = []
    score_map = {
        'Market': component_scores.get('R_prime'),
        'Beta/Corr': component_scores.get('AB_prime'),
        'Volatility': component_scores.get('AA_prime'),
        'Fundamentals': component_scores.get('F_prime'),
        'Technicals': component_scores.get('Q_prime'),
        'Sentiment': component_scores.get('S_prime'),
        'ML Forecast': component_scores.get('M_prime'),
    }
    for name, score in score_map.items():
        score_lines.append(f"- {name}: {score:.1f}" if score is not None else f"- {name}: N/A")

    prompt = f"""
    Act as a financial analyst. Based *only* on the following Prime component scores for {ticker}, provide a concise (2-4 sentences) summary profile.
    Focus on interpreting the scores to highlight clear strengths and weaknesses relative to each other.
    Do NOT mention the specific score values (e.g., '75.2') in your summary. Use qualitative terms (e.g., 'strong', 'weak', 'neutral', 'positive', 'negative', 'favorable', 'unfavorable').
    Component Scores (0-100 scale, higher is generally better):
    {chr(10).join(score_lines)}
    """
    # Using chr(10) for newline within f-string for clarity

    try:
        # Ensure lock mechanism is correctly used if needed for API rate limiting
        async with lock_to_use:
            response = await asyncio.to_thread(
                model_to_use.generate_content,
                prompt,
                # Optional: Add generation config if needed (e.g., temperature)
                # generation_config=genai.types.GenerationConfig(temperature=0.3)
            )
        # Add robust error handling for response parsing
        if response and response.text:
            return response.text.strip()
        else:
            return "AI summary generation failed (empty response)."
    except Exception as e:
        print(f"   -> [DEBUG PowerScore AI] Error generating AI summary: {e}")
        return f"Error generating AI summary: {type(e).__name__}"


# --- Main Command Handler ---
async def handle_powerscore_command(
    args: List[str] = None,
    ai_params: dict = None,
    is_called_by_ai: bool = False,
    gemini_model_obj: Any = None, # Renamed from override for clarity
    api_lock_override: asyncio.Lock = None,
    **kwargs # Add **kwargs to accept unexpected arguments
):
    """
    Calculates and displays the PowerScore for a stock.

    Accepts gemini_model_obj and api_lock_override for dependency injection.
    Uses **kwargs to ignore other potential arguments passed by Prometheus.
    """
    # Use injected model/lock if provided, otherwise use module's global ones
    model_to_use = gemini_model_obj or gemini_model
    lock_to_use = api_lock_override or GEMINI_API_LOCK

    ticker, sensitivity = None, None
    try:
        if is_called_by_ai and ai_params:
            ticker = ai_params.get("ticker")
            sensitivity_raw = ai_params.get("sensitivity")
            # Validate sensitivity received from AI
            sensitivity = int(sensitivity_raw) if sensitivity_raw is not None else None
        elif args and len(args) == 2:
            ticker = args[0].upper()
            sensitivity = int(args[1])
        else: # Handle CLI case where args might be missing/wrong
             pass # Error handled below

        if not ticker or sensitivity not in [1, 2, 3]:
            message = "Usage: /powerscore <TICKER> <SENSITIVITY 1-3>"
            # Return error structure for AI, print for CLI
            if is_called_by_ai: return {"status": "error", "message": message}
            else: print(message); return None # Return None for CLI failure

    except (ValueError, TypeError):
         message = "Invalid sensitivity. Must be an integer 1, 2, or 3."
         if is_called_by_ai: return {"status": "error", "message": message}
         else: print(message); return None

    # --- Start Analysis ---
    if not is_called_by_ai:
        print(f"\n--- Generating PowerScore for {ticker} (Sensitivity: {sensitivity}) ---")

    # Determine period based on sensitivity for Beta/Corr and Volatility
    period_map = {1: '10y', 2: '5y', 3: '1y'}
    backtest_period = period_map[sensitivity]

    # --- Gather Raw Data Asynchronously ---
    tasks = {
        "R": get_market_invest_score_for_powerscore(),
        "ABB_ABC": get_single_stock_beta_corr(ticker, backtest_period), # Uses portfolio func internally
        "AA": calculate_volatility_metrics(ticker, backtest_period),
        "F": handle_fundamentals_command_internal(ai_params={'ticker': ticker}), # Use internal version
        "Q": calculate_ema_invest(ticker, sensitivity, is_called_by_ai=True), # Directly use invest_command's helper
        "S": handle_sentiment_command( # Use sentiment_command's handler
            ai_params={'ticker': ticker},
            is_called_by_ai=True,
            gemini_model_override=model_to_use, # Pass the correct model
            api_lock_override=lock_to_use      # Pass the correct lock
        ),
        "M": handle_mlforecast_command_internal(ai_params={'ticker': ticker}) # Use internal version
    }
    # Gather results, capturing potential exceptions
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    raw_values = dict(zip(tasks.keys(), results))

    # --- Process Raw Values (Handle potential errors/exceptions from gather) ---
    raw = {}
    component_errors = []

    # Market Score (R)
    if isinstance(raw_values.get('R'), Exception): component_errors.append("R (Market)"); raw['R'] = None
    else: raw['R'] = raw_values.get('R')

    # Beta/Corr (ABB_ABC)
    beta_corr_result = raw_values.get('ABB_ABC')
    if isinstance(beta_corr_result, Exception): component_errors.append("AB (Beta/Corr)"); raw['ABB'], raw['ABC'] = None, None
    elif isinstance(beta_corr_result, tuple) and len(beta_corr_result) == 2: raw['ABB'], raw['ABC'] = beta_corr_result
    else: raw['ABB'], raw['ABC'] = None, None # Handle None or unexpected format

    # Volatility Rank (AA)
    vol_result = raw_values.get('AA')
    if isinstance(vol_result, Exception): component_errors.append("AA (Volatility)"); raw['AA'] = None
    elif isinstance(vol_result, tuple) and len(vol_result) == 2: raw['AA'] = vol_result[1] # Index 1 is vol_rank
    else: raw['AA'] = None

    # Fundamentals (F)
    funda_result = raw_values.get('F')
    if isinstance(funda_result, Exception): component_errors.append("F (Fundamentals)"); raw['F'] = None
    elif isinstance(funda_result, dict) and 'fundamental_score' in funda_result: raw['F'] = funda_result['fundamental_score']
    else: raw['F'] = None

    # QuickScore (Q)
    quick_result = raw_values.get('Q')
    if isinstance(quick_result, Exception): component_errors.append("Q (QuickScore)"); raw['Q'] = None
    elif isinstance(quick_result, tuple) and len(quick_result) == 2: raw['Q'] = quick_result[1] # Index 1 is score
    else: raw['Q'] = None

    # Sentiment (S)
    sentiment_result = raw_values.get('S')
    if isinstance(sentiment_result, Exception): component_errors.append("S (Sentiment)"); raw['S'] = None
    # Check if dict and has the raw score key
    elif isinstance(sentiment_result, dict) and 'sentiment_score_raw' in sentiment_result: raw['S'] = sentiment_result['sentiment_score_raw']
    # Handle case where sentiment returns an error dict itself
    elif isinstance(sentiment_result, dict) and 'error' in sentiment_result: component_errors.append(f"S (Sentiment: {sentiment_result['error']})"); raw['S'] = None
    else: raw['S'] = None

    # ML Forecast (M)
    ml_result = raw_values.get('M')
    m_forecast_val, used_m_period = None, "N/A"
    if isinstance(ml_result, Exception): component_errors.append("M (ML Forecast)"); raw['M'] = None
    elif isinstance(ml_result, list) and ml_result:
        # Define periods relevant to sensitivity
        m_period_map = {1: ["1-Year (52-Week)", "6-Month (26-Week)"], # Adjust if ML internal func changes
                        2: ["6-Month (26-Week)", "3-Month (63-Day)"], # Adjust if ML internal func changes
                        3: ["1-Month (21-Day)", "5-Day"]}
        m_lookup = {item.get("Period"): item for item in ml_result}
        for period in m_period_map[sensitivity]:
            if period in m_lookup:
                pct_str = m_lookup[period].get("Est. % Change", "0%").replace('%', '')
                try:
                    m_forecast_val = float(pct_str)
                    used_m_period = period
                    break # Found the first relevant forecast
                except ValueError: pass # Ignore if parsing fails
        raw['M'] = m_forecast_val # Store the found value (or None)
    else: raw['M'] = None # Handle empty list or other types

    # --- Calculate Prime Scores (only if raw value exists and is valid) ---
    prime = {}
    def calculate_prime(key, value): # Helper to avoid repetition
        try: return float(value) if pd.notna(value) else None
        except (ValueError, TypeError): return None

    raw_abb = calculate_prime('ABB', raw.get('ABB'))
    if raw_abb is not None:
        prime['ABB'] = np.clip(100/(((raw_abb-2)**2)+1) if raw_abb<=2 else 400/((3*(raw_abb-2)**2)+4), 0, 100)

    raw_abc = calculate_prime('ABC', raw.get('ABC'))
    if raw_abc is not None:
        # Clamp input to avoid potential math domain errors with extreme correlations
        clamped_abc = np.clip(raw_abc, -0.999, 0.999)
        prime['ABC'] = np.clip(1.01*(100*(297**clamped_abc))/((297**clamped_abc)+3), 0, 100)

    # Calculate AB prime only if both ABB and ABC prime were calculated
    if 'ABB' in prime and 'ABC' in prime:
        prime['AB'] = (prime['ABB'] + prime['ABC']) / 2
    else: prime['AB'] = None # Explicitly None if components missing

    raw_f = calculate_prime('F', raw.get('F'))
    if raw_f is not None: prime['F'] = np.clip(raw_f, 0, 100) # Fundamentals already 0-100

    raw_s = calculate_prime('S', raw.get('S')) # Raw sentiment is -1 to 1
    if raw_s is not None: prime['S'] = np.clip(50 * (raw_s + 1), 0, 100) # Convert to 0-100

    raw_aa = calculate_prime('AA', raw.get('AA')) # Raw AA is percentile 0-100
    if raw_aa is not None: prime['AA'] = 100 - raw_aa # Lower rank = higher score

    raw_r = calculate_prime('R', raw.get('R')) # Raw R is 0-100
    if raw_r is not None: prime['R'] = raw_r

    raw_m = calculate_prime('M', raw.get('M')) # Raw M is forecast % change
    if raw_m is not None:
        m, s = raw_m, sensitivity
        try:
            if s == 1: prime['M'] = 100/(1+(9*(1.1396**-m)))
            elif s == 2: prime['M'] = 100/(1+(4*(1.23**-m)))
            else: prime['M'] = 100/(1+(3*(3**-m))) # Using 3^m for sensitivity 3
            prime['M'] = np.clip(prime['M'], 0, 100)
        except (OverflowError, ValueError): prime['M'] = 50.0 # Default neutral on math error

    raw_q = calculate_prime('Q', raw.get('Q')) # Raw Q is Invest Score % (0-inf)
    if raw_q is not None:
        q, s = raw_q, sensitivity
        try:
            # Apply sigmoid-like functions based on sensitivity
            if s == 1: prime['Q'] = 100/(1+(math.exp(-0.0879*(q-50))))
            elif s == 2: prime['Q'] = 100/(1+(math.exp(-0.0628*(q-50))))
            else: prime['Q'] = 100/(1+(math.exp(-0.0981*(q-50))))
            prime['Q'] = np.clip(prime['Q'], 0, 100)
        except (OverflowError, ValueError): prime['Q'] = 50.0 # Default neutral on math error


    # --- Calculate Final PowerScore ---
    # Define weights based on sensitivity
    weights_map = {
        1: {'R': 0.15, 'AB': 0.15, 'AA': 0.15, 'F': 0.15, 'Q': 0.20, 'S': 0.10, 'M': 0.10},
        2: {'R': 0.20, 'AB': 0.10, 'AA': 0.10, 'F': 0.05, 'Q': 0.25, 'S': 0.15, 'M': 0.15},
        3: {'R': 0.25, 'AB': 0.05, 'AA': 0.05, 'F': 0.00, 'Q': 0.30, 'S': 0.25, 'M': 0.10} # Adjusted M weight for Sens 3
    }
    current_weights = weights_map[sensitivity]
    weighted_sum, total_weight_used = 0.0, 0.0

    # Iterate through prime scores and apply weights
    for key, weight in current_weights.items():
        prime_score = prime.get(key)
        # Only include components that have a valid prime score and a non-zero weight
        if prime_score is not None and weight > 0:
            weighted_sum += prime_score * weight
            total_weight_used += weight

    # Calculate final score, handle division by zero if no components were valid
    final_powerscore = np.clip((weighted_sum / total_weight_used) if total_weight_used > 0 else 0.0, 0, 100)

    # --- Return for AI or Print for CLI ---
    if is_called_by_ai:
        # Return a dictionary suitable for AI tool use
        return {
            "status": "success" if not component_errors else "partial_error",
            "ticker": ticker,
            "sensitivity": sensitivity,
            "powerscore": final_powerscore,
            # Optionally include component scores if needed by AI later
            "prime_scores": {k: v for k, v in prime.items() if v is not None},
            "errors": component_errors if component_errors else None
        }
    else:
        # --- CLI Output ---
        print("\n--- PowerScore Components ---")
        table_data = []
        for key, name, raw_val_key in [
            ('R', "Market Invest Score", 'R'), ('AB', "Beta/Corr Average", 'ABB'), # Show AB, raw uses ABB/ABC
            ('AA', "Volatility Rank", 'AA'), ('F', "Fundamental Score", 'F'),
            ('Q', "QuickScore", 'Q'), ('S', "Sentiment Score", 'S'),
            ('M', f"ML Forecast ({used_m_period})", 'M')]:

            raw_display = "N/A"
            if key == 'AB':
                 raw_b = raw.get('ABB'); raw_c = raw.get('ABC')
                 if raw_b is not None and raw_c is not None: raw_display = f"{raw_b:.2f} / {raw_c:.2f}"
                 elif raw_b is not None: raw_display = f"{raw_b:.2f} / N/A"
                 elif raw_c is not None: raw_display = f"N/A / {raw_c:.2f}"
            elif key == 'AA': raw_display = f"{raw.get(raw_val_key):.1f}%" if raw.get(raw_val_key) is not None else "N/A"
            elif key == 'Q': raw_display = f"{raw.get(raw_val_key):.1f}%" if raw.get(raw_val_key) is not None else "N/A"
            elif key == 'M': raw_display = f"{raw.get(raw_val_key):.1f}%" if raw.get(raw_val_key) is not None else "N/A"
            else: raw_display = f"{raw.get(raw_val_key):.2f}" if raw.get(raw_val_key) is not None else "N/A"

            prime_display = f"{prime.get(key):.1f}" if prime.get(key) is not None else "N/A"
            weight_display = f"{current_weights.get(key, 0)*100:.0f}%"

            table_data.append([name, raw_display, prime_display, weight_display])

        print(tabulate(table_data, headers=["Metric", "Raw Value", "Prime Score", f"Weight (S{sensitivity})"], tablefmt="grid"))

        if component_errors:
            print("\nWarnings during calculation:")
            for err in component_errors: print(f"  - Failed to calculate: {err}")

        # Generate and print AI summary
        ai_summary = await get_powerscore_explanation(ticker, prime, model_to_use, lock_to_use)
        print(f"\nAI Analyst Summary:\n{ai_summary}")

        print(f"\nFINAL POWERSCORE (Sensitivity {sensitivity}): {final_powerscore:.2f} / 100.00")
        print("-" * (len(f"FINAL POWERSCORE (Sensitivity {sensitivity}): {final_powerscore:.2f} / 100.00")))

        return None # Indicate success for CLI