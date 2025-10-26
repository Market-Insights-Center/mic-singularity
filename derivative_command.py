# derivative_command.py

# --- Imports for derivative_command ---
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-GUI environments
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional, Tuple
import os
import asyncio
import traceback
import uuid
import random # Needed for robust downloader
from dateutil.relativedelta import relativedelta # Needed for robust downloader

# --- Constants ---
YFINANCE_API_SEMAPHORE = asyncio.Semaphore(5) # Consistent with isolated test

# --- Robust YFinance Download Helper (Copied from isolated test) ---
async def get_yf_download_robustly(tickers: list, **kwargs) -> pd.DataFrame:
    """ Robust wrapper for yf.download with retry logic and standardization. """
    max_retries = 2
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(random.uniform(0.3, 0.8))
            kwargs.setdefault('progress', False)
            kwargs.setdefault('timeout', 15)
            # *** Keep auto_adjust=False for consistent derivative calculation ***
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
                          if level_map['Price'] != 0 or level_map['Ticker'] != 1:
                               data.columns = data.columns.reorder_levels(['Price', 'Ticker'])
                          data.columns.names = ['Price', 'Ticker']
                     else: data.columns.names = ['Price', 'Ticker']
                 except Exception as e_reformat:
                      print(f"   [WARN Download] Could not standardize MultiIndex names: {data.columns.names}. Error: {e_reformat}")

            return data

        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            if attempt < max_retries - 1:
                delay = (attempt + 1) * 1
                print(f"   [WARN Download] yf.download failed ({error_type}, Attempt {attempt+1}/{max_retries}) for {tickers}. Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                print(f"   [ERROR Download] All yf download attempts failed for {tickers}. Last error ({error_type}): {error_msg}")
                return pd.DataFrame()
    return pd.DataFrame()


# --- Derivative Command Logic Helpers ---
# (Pasted directly from the working isolated_test.py)

def find_best_polynomial_degree(data: np.ndarray) -> int:
    """ Finds the best-fitting polynomial degree (up to 5). """
    x = np.arange(len(data))
    y = data
    best_r_squared = -np.inf
    best_degree = 1
    max_possible_degree = min(5, len(x) - 1)
    if max_possible_degree < 1: return 1
    for degree in range(1, max_possible_degree + 1):
        try:
            with np.errstate(all='raise'): coeffs = np.polyfit(x, y, degree)
            if not isinstance(coeffs, np.ndarray) or coeffs.ndim != 1: continue
            p = np.poly1d(coeffs)
            y_predicted = p(x)
            ss_total = np.sum((y - np.mean(y)) ** 2)
            ss_residual = np.sum((y - y_predicted) ** 2)
            if ss_total < 1e-12: r_squared = 1.0 if ss_residual < 1e-12 else 0.0
            else: r_squared = 1 - (ss_residual / ss_total)
            if r_squared > best_r_squared:
                best_r_squared = r_squared
                best_degree = degree
        except (np.linalg.LinAlgError, ValueError, TypeError, FloatingPointError): continue
    return best_degree

def fit_and_differentiate(prices: np.ndarray, degree: int) -> tuple[Optional[np.ndarray], Optional[float], Optional[float]]:
    """ Fits polynomial, calculates derivatives. Returns (coeffs, deriv1, deriv2) or (None, None, None) on failure. """
    if not isinstance(prices, np.ndarray):
         print(f"    [ERROR Derivative Fit] Input prices is not a NumPy array! Type: {type(prices)}")
         return None, None, None
    if prices.ndim != 1:
        prices = prices.flatten()
        if prices.ndim != 1:
             print(f"    [ERROR Derivative Fit] Input 'prices' could not be flattened to 1D (final shape: {prices.shape})")
             return None, None, None

    x = np.arange(len(prices))
    coeffs: Optional[np.ndarray] = None
    derivative_at_end: Optional[float] = None
    second_derivative_at_end: Optional[float] = None

    try:
        with np.errstate(all='raise'): coeffs = np.polyfit(x, prices, degree)
        if not isinstance(coeffs, np.ndarray) or coeffs.ndim != 1:
            raise ValueError(f"np.polyfit returned invalid coeffs (shape: {getattr(coeffs, 'shape', 'N/A')})")

        poly_func = np.poly1d(coeffs)
        poly_derivative = poly_func.deriv()
        poly_second_derivative = poly_func.deriv(2)
        derivative_at_end = float(poly_derivative(x[-1]))
        second_derivative_at_end = float(poly_second_derivative(x[-1]))

        if not np.isfinite(derivative_at_end) or not np.isfinite(second_derivative_at_end):
             print(f"    [WARN Derivative Fit] Derivative calculation resulted in NaN/Inf.")
             return coeffs, None, None
        return coeffs, derivative_at_end, second_derivative_at_end
    except (np.linalg.LinAlgError, ValueError, TypeError, FloatingPointError) as fit_error:
        # print(f"    [DEBUG Derivative Fit] Polyfit/Differentiation failed for degree {degree}: {fit_error}")
        return None, None, None

def format_polynomial_equation(coeffs: Optional[np.ndarray]) -> str:
    """ Formats coefficients into an equation string. """
    if coeffs is None or not isinstance(coeffs, np.ndarray) or coeffs.ndim != 1 or np.isnan(coeffs).any():
        return "Equation cannot be determined (fit failed)"
    equation_parts = []
    degree = len(coeffs) - 1
    max_abs_coeff = np.max(np.abs(coeffs)) if len(coeffs) > 0 and not np.isnan(coeffs).all() else 0
    precision = 7
    if max_abs_coeff > 100: precision = 2
    elif max_abs_coeff < 1e-4 and max_abs_coeff != 0: precision = 4
    for i, coeff in enumerate(coeffs):
        if np.isclose(coeff, 0): continue
        power = degree - i
        is_first_term = len(equation_parts) == 0
        sign = "" if is_first_term and coeff > 0 else "- " if coeff < 0 else " + "
        abs_coeff = abs(coeff)
        coeff_str = f"{abs_coeff:.{precision}g}"
        if power == 0: equation_parts.append(f"{sign}{coeff_str}")
        elif power == 1:
            term = f"{coeff_str}x" if not np.isclose(abs_coeff, 1.0) else "x"
            equation_parts.append(f"{sign}{term}")
        else:
            term = f"{coeff_str}x^{power}" if not np.isclose(abs_coeff, 1.0) else f"x^{power}"
            equation_parts.append(f"{sign}{term}")
    equation = "".join(equation_parts).strip()
    return equation if equation else "0"

def plot_polynomial_fit(prices: np.ndarray, coeffs: Optional[np.ndarray], ticker: str, period: str) -> Optional[str]:
    """ Generates and saves a plot. Runs synchronously. """
    try:
        x_data = np.arange(len(prices))
        plot_fit_line = coeffs is not None and isinstance(coeffs, np.ndarray) and coeffs.ndim == 1 and not np.isnan(coeffs).any()

        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            ax.plot(x_data, prices, label=f'{ticker} Price', color='cyan', linewidth=1.5, alpha=0.8, marker='.', markersize=4, linestyle='')

            if plot_fit_line:
                poly_fit = np.poly1d(coeffs)
                x_fit = np.linspace(x_data.min(), x_data.max(), 300)
                y_fit = poly_fit(x_fit)
                ax.plot(x_fit, y_fit, label=f'Best Fit Poly (Deg {len(coeffs)-1})', color='lime', linestyle='--', linewidth=2)
            else:
                ax.text(0.5, 0.5, 'Polynomial Fit Failed', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color='red', fontsize=10)

            ax.set_title(f'Poly Fit: {ticker} ({period.upper()})', color='white', fontsize=12)
            ax.set_xlabel('Time Steps', color='white', fontsize=10)
            ax.set_ylabel('Price', color='white', fontsize=10)
            ax.legend(facecolor='black', edgecolor='grey', fontsize=8)
            ax.grid(True, linestyle=':', alpha=0.5, color='grey')
            ax.tick_params(axis='x', colors='white', labelsize=8)
            ax.tick_params(axis='y', colors='white', labelsize=8)
            fig.tight_layout()

            output_dir = 'derivative_outputs'
            os.makedirs(output_dir, exist_ok=True)
            filename_base = f"derivative_{ticker}_{period}_{uuid.uuid4().hex[:6]}.png" # Reverted UUID length
            filepath = os.path.join(output_dir, filename_base)
            plt.savefig(filepath, bbox_inches='tight', facecolor='black', edgecolor='black', dpi=150)

            print(f"  -> Plot saved to {filepath}")
            return filepath
        finally:
            plt.close(fig)

    except Exception as e:
         print(f"    [ERROR Plotting] {ticker} ({period}): {e}")
         return None


# --- Main Handler ---
async def handle_derivative_command(args: List[str], **kwargs): # Added **kwargs
    """
    Handles the /derivative command by fitting polynomials to different time periods
    and calculating the derivative at the most recent point.
    Returns a dictionary summarizing the results.
    Ignores extra keyword arguments via **kwargs.
    """
    if len(args) != 1:
        usage_msg = "Usage: /derivative <TICKER>"
        print(usage_msg)
        return {"status": "error", "message": usage_msg}

    ticker = args[0].upper()
    print(f"📈 Starting derivative analysis for {ticker}...")

    time_periods = {
        '1d': {'period': '1d', 'interval': '5m'}, # Keep shorter intervals from test
        '1wk': {'period': '7d', 'interval': '30m'},# Keep shorter intervals from test
        '1mo': {'period': '1mo', 'interval': '1d'},
        '3mo': {'period': '3mo', 'interval': '1d'},
        '1y': {'period': '1y', 'interval': '1d'}
    }

    results = {}
    plot_filenames = {}
    success_overall = True

    for label, params in time_periods.items():
        period = params['period']
        interval = params['interval']
        print(f"  -> Analyzing {label} ({period}, {interval})...")
        try:
            # Use Robust Downloader
            data = await get_yf_download_robustly(
                tickers=[ticker], period=period, interval=interval
            )

            if data.empty:
                print(f"     [WARN] No data fetched for {label}.")
                results[label] = {"status": "error", "message": "No data fetched"}
                success_overall = False
                continue

            # --- Reliable Column Selection ---
            close_series = None
            adj_close_col = ('Adj Close', ticker)
            close_col = ('Close', ticker)

            if isinstance(data.columns, pd.MultiIndex):
                # Check standard Price/Ticker levels first
                if ('Close', ticker) in data.columns: close_series = data[('Close', ticker)]
                elif ('Adj Close', ticker) in data.columns: close_series = data[('Adj Close', ticker)]; print(f"     [INFO] Using Adj Close for {ticker} ({label})")
                # Fallback check (less common with standardization)
                elif close_col in data.columns: close_series = data[close_col]
                elif adj_close_col in data.columns: close_series = data[adj_close_col]; print(f"     [INFO] Using Adj Close for {ticker} ({label})")
            # Should not happen often now, but keep as safeguard
            elif 'Close' in data.columns: close_series = data['Close']
            elif 'Adj Close' in data.columns: close_series = data['Adj Close']


            if close_series is None:
                 print(f"     [ERROR] Could not find 'Close' or 'Adj Close' column for {ticker} for {label}.")
                 # print(f"     Available columns: {data.columns}") # Optional debug
                 results[label] = {"status": "error", "message": "Required price column not found"}
                 success_overall = False
                 continue
            # --- End Column Selection ---

            prices_series = close_series.dropna()
            if prices_series.empty:
                 print(f"     [WARN] No valid price data after dropna for {label}.")
                 results[label] = {"status": "error", "message": "No valid data points"}
                 success_overall = False
                 continue

            prices = prices_series.to_numpy().flatten() # Ensure 1D

            min_points_required = 6
            if len(prices) < min_points_required:
                print(f"     [WARN] Not enough valid data points ({len(prices)} < {min_points_required}) for {label}.")
                results[label] = {"status": "error", "message": f"Not enough valid data points ({len(prices)})"}
                success_overall = False
                continue

            best_degree = find_best_polynomial_degree(prices)
            coeffs, derivative_at_end, second_derivative_at_end = fit_and_differentiate(prices, best_degree)

            if coeffs is None or derivative_at_end is None or second_derivative_at_end is None:
                 print(f"     [ERROR] Calculation failed for {label}.")
                 results[label] = {"status": "error", "message": "Polynomial fit/differentiation failed"}
                 success_overall = False
                 plot_filename = await asyncio.to_thread(plot_polynomial_fit, prices, None, ticker, label)
                 if plot_filename: plot_filenames[label] = plot_filename
                 continue

            results[label] = {
                "status": "success", "degree": best_degree,
                "equation": format_polynomial_equation(coeffs),
                "derivative_at_end": derivative_at_end,
                "second_derivative_at_end": second_derivative_at_end,
            }
            # Run synchronous plot function in thread
            plot_filename = await asyncio.to_thread(
                plot_polynomial_fit, prices, coeffs, ticker, label
            )
            if plot_filename:
                plot_filenames[label] = plot_filename

        except Exception as e:
            print(f"     [ERROR] Unhandled exception during {label} analysis: {e}")
            # traceback.print_exc() # Uncomment for full traceback
            results[label] = {"status": "error", "message": f"Unhandled Exception: {str(e)}"}
            success_overall = False

    # --- Print Summary ---
    print("\n--- Summary of Derivative Analysis ---")
    print(f"Ticker: {ticker}")
    summary_lines = []
    has_successful_results = False
    for label, res in results.items():
        print(f"\nPeriod: {label.upper()}")
        if res['status'] == 'success':
            has_successful_results = True
            print(f"  Best Fit Equation (Deg {res['degree']}): {res['equation']}")
            print(f"  Derivative (Rate of Change): {res['derivative_at_end']:.7f}")
            print(f"  Second Derivative (Acceleration): {res['second_derivative_at_end']:.7f}")
            if label in plot_filenames:
                 print(f"  Plot Saved: {plot_filenames[label]}")
            summary_lines.append(f"{label.upper()}: Deriv={res['derivative_at_end']:.3f}, Accel={res['second_derivative_at_end']:.3f}")
        else:
            print(f"  Status: {res['status'].upper()}")
            print(f"  Message: {res['message']}")
            summary_lines.append(f"{label.upper()}: Failed - {res['message']}")

    print("\nAnalysis complete.")
    if plot_filenames: print("Check saved images for plots.")

    # --- Prepare Return Value ---
    final_status = "error"
    if has_successful_results:
        final_status = "success" if success_overall else "partial_error"

    final_return = {
        "status": final_status, "ticker": ticker, "periods": results,
        "summary": f"Derivative analysis for {ticker}: " + " | ".join(summary_lines),
        "plot_files": list(plot_filenames.values())
    }
    return final_return