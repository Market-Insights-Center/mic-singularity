# new_derivative_command.py

# --- Imports for derivative_command ---
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
import os
import asyncio
from scipy.optimize import curve_fit
import traceback

# This function will be called from main_singularity.py
async def handle_derivative_command(args: List[str]):
    """
    Handles the /derivative command by fitting polynomials to different time periods
    and calculating the derivative at the most recent point.
    """
    if len(args) != 1:
        print("Usage: /derivative <TICKER>")
        return

    ticker = args[0].upper()
    print(f"📈 Starting derivative analysis for {ticker}...")

    time_periods = {
        '1d': {'period': '1d', 'interval': '15m'},
        '1wk': {'period': '7d', 'interval': '1h'},
        '1mo': {'period': '1mo', 'interval': '1d'},
        '3mo': {'period': '3mo', 'interval': '1d'},
        '1y': {'period': '1y', 'interval': '1d'}
    }

    results = {}
    for label, params in time_periods.items():
        period = params['period']
        interval = params['interval']
        try:
            print(f"  -> Analyzing {label} period...")
            
            print(f"    [DEBUG] Fetching data for period='{period}', interval='{interval}'...")

            data = await asyncio.to_thread(
                lambda: yf.download(ticker, period=period, interval=interval, progress=False)
            )

            if data.empty:
                print(f"    [DEBUG] Data fetch returned an empty DataFrame for {label}.")
                print(f"  ❌ Not enough data for {label}.")
                results[label] = {"status": "error", "message": "Not enough data"}
                continue
            
            prices = data['Close'].values.flatten()
            
            print(f"    [DEBUG] Successfully fetched data. Shape of prices array: {prices.shape}")

            if len(prices) < 5:
                print(f"    [DEBUG] Not enough data points ({len(prices)}) for a polynomial fit.")
                print(f"  ❌ Not enough data for {label}.")
                results[label] = {"status": "error", "message": "Not enough data"}
                continue

            best_degree = find_best_polynomial_degree(prices)

            # MODIFIED: Unpack the second derivative from the function call
            coeffs, derivative_at_end, second_derivative_at_end = fit_and_differentiate(prices, best_degree)

            # Store results
            results[label] = {
                "status": "success",
                "equation": format_polynomial_equation(coeffs),
                "derivative_at_end": derivative_at_end,
                "second_derivative_at_end": second_derivative_at_end, # ADDED: Store the second derivative
                "coefficients": coeffs.tolist(),
                "degree": best_degree
            }
            
            plot_polynomial_fit(prices, coeffs, ticker, label)

        except Exception as e:
            print(f"    [DEBUG] An exception occurred. See traceback below.")
            traceback.print_exc()
            print(f"  ❌ An error occurred during the analysis for {label}: {e}")
            results[label] = {"status": "error", "message": str(e)}

    print("\n--- Summary of Derivative Analysis ---")
    print(f"Ticker: {ticker}")
    for label, res in results.items():
        if res['status'] == 'success':
            print(f"\nPeriod: {label.upper()}")
            print(f"  Best Fit Equation: {res['equation']}")
            print(f"  Derivative (Rate of Change): {res['derivative_at_end']:.7f}")
            # ADDED: Display the second derivative in the summary
            print(f"  Second Derivative (Acceleration): {res['second_derivative_at_end']:.7f}")
            print(f"  Plot Saved: derivative_{ticker}_{label}.png")
        else:
            print(f"\nPeriod: {label.upper()}")
            print(f"  Status: {res['status'].upper()}")
            print(f"  Message: {res['message']}")
            
    print("\nAnalysis complete. Check the saved images for visual details.")
     
# --- Helper Functions (all new) ---

def find_best_polynomial_degree(data: np.ndarray) -> int:
    """
    Finds the best-fitting polynomial degree (up to 5) based on R-squared value.
    """
    x = np.arange(len(data))
    y = data
    best_r_squared = -1
    best_degree = 1
    
    for degree in range(1, 6): 
        if degree >= len(x):
            continue
            
        coeffs = np.polyfit(x, y, degree)
        p = np.poly1d(coeffs)
        
        y_predicted = p(x)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_predicted) ** 2)
        
        if ss_total == 0:
            r_squared = 1.0
        else:
            r_squared = 1 - (ss_residual / ss_total)
            
        if r_squared > best_r_squared:
            best_r_squared = r_squared
            best_degree = degree
            
    return best_degree

# MODIFIED: The function now returns the second derivative as well
def fit_and_differentiate(prices: np.ndarray, degree: int) -> tuple[np.ndarray, float, float]:
    """
    Fits a polynomial to the price data and calculates the first and second
    derivatives at the most recent data point.
    """
    x = np.arange(len(prices))
    coeffs = np.polyfit(x, prices, degree)
    
    # Create the polynomial function and its derivatives
    poly_func = np.poly1d(coeffs)
    poly_derivative = poly_func.deriv()
    poly_second_derivative = poly_derivative.deriv() # ADDED: Calculate second derivative function
    
    # Calculate the derivatives at the last point (index `len(x) - 1`)
    derivative_at_end = poly_derivative(x[-1])
    second_derivative_at_end = poly_second_derivative(x[-1]) # ADDED: Evaluate second derivative at the end
    
    return coeffs, derivative_at_end, second_derivative_at_end

def format_polynomial_equation(coeffs: np.ndarray) -> str:
    """
    Formats the polynomial coefficients into a readable equation string.
    """
    equation_parts = []
    degree = len(coeffs) - 1
    
    for i, coeff in enumerate(coeffs):
        if np.isclose(coeff, 0):
            continue
            
        power = degree - i
        sign = " + " if coeff > 0 and i > 0 else " - " if coeff < 0 else ""
        abs_coeff = abs(coeff)
        
        if power == 0:
            equation_parts.append(f"{sign}{abs_coeff:.7f}")
        elif power == 1:
            equation_parts.append(f"{sign}{abs_coeff:.7f}x")
        else:
            equation_parts.append(f"{sign}{abs_coeff:.7f}x^{power}")

    equation = "".join(equation_parts).strip()
    return equation.lstrip(' +').lstrip(' -')

def plot_polynomial_fit(prices: np.ndarray, coeffs: np.ndarray, ticker: str, period: str):
    """
    Generates and saves a plot of the stock data with the polynomial fit overlayed.
    """
    x_data = np.arange(len(prices))
    poly_fit = np.poly1d(coeffs)
    x_fit = np.linspace(0, len(prices) - 1, 500)
    y_fit = poly_fit(x_fit)
    
    plt.style.use('dark_background')
    plt.figure(figsize=(10, 6))
    
    plt.plot(x_data, prices, label=f'{ticker} Price', color='cyan', linewidth=2, alpha=0.7)
    plt.plot(x_fit, y_fit, label=f'Best Fit Polynomial (Degree {len(coeffs)-1})', color='lime', linestyle='--', linewidth=2)
    
    plt.title(f'Polynomial Fit for {ticker} ({period.upper()})', color='white')
    plt.xlabel('Days', color='white')
    plt.ylabel('Price', color='white')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    output_dir = 'derivative_outputs'
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"derivative_{ticker}_{period}.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()
    
    print(f"  -> Plot saved to {filepath}")