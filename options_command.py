# --- Imports for options_command ---
import asyncio
import uuid
import traceback
from datetime import datetime
from math import sqrt
from typing import List, Dict, Optional, Any

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tabulate import tabulate

# --- Helper Functions (copied or moved for self-containment) ---

def ask_singularity_input(prompt: str, validation_fn=None, error_msg="Invalid input.", default_val=None) -> Optional[str]:
    """Helper function to ask for user input."""
    while True:
        full_prompt = f"{prompt}"
        if default_val is not None:
            full_prompt += f" (default: {default_val}, press Enter)"
        full_prompt += ": "
        user_response = input(full_prompt).strip()
        if not user_response and default_val is not None:
            return str(default_val)
        if validation_fn:
            if validation_fn(user_response):
                return user_response
            else:
                print(error_msg)
        elif user_response:
            return user_response

def display_strikes_menu(all_strikes: list, current_price: float):
    """Displays available strikes in a multi-column format, highlighting the at-the-money strike."""
    print("\n--- Available Strike Prices ---")
    atm_index = -1
    if current_price and all_strikes:
        try:
            atm_index = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - current_price))
        except ValueError:
            pass # No strikes to check
    
    num_cols = 5
    col_width = 18
    num_rows = (len(all_strikes) + num_cols - 1) // num_cols
    
    for row in range(num_rows):
        line = ""
        for col in range(num_cols):
            idx = row + col * num_rows
            if idx < len(all_strikes):
                strike = all_strikes[idx]
                atm_marker = "->" if idx == atm_index else "  "
                line += f"{atm_marker} {idx+1:>3}. ${strike:<8.2f}".ljust(col_width)
        print(line)
    print("-" * (col_width * num_cols - 2))


def calculate_greeks_numpy(df, flag_col='Flag', underlying_price_col='S', strike_col='K', annualized_tte_col='t', riskfree_rate_col='r', sigma_col='sigma', dividend_col='q'):
    """NumPy-based Black-Scholes-Merton option price and greeks calculator."""
    S, K, t, r, sigma, q = df[underlying_price_col].values, df[strike_col].values, df[annualized_tte_col].values, df[riskfree_rate_col].values, df[sigma_col].values, df[dividend_col].values
    t = np.maximum(t, 1e-9)
    # Added a failsafe for sigma to prevent math errors with bad data
    sigma = np.maximum(sigma, 1e-9)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    results = pd.DataFrame(index=df.index)
    call_flags, put_flags = (df[flag_col] == 'c'), (df[flag_col] != 'c')
    results.loc[call_flags, 'Price'] = (S[call_flags] * np.exp(-q[call_flags] * t[call_flags]) * norm.cdf(d1[call_flags])) - (K[call_flags] * np.exp(-r[call_flags] * t[call_flags]) * norm.cdf(d2[call_flags]))
    results.loc[put_flags, 'Price'] = (K[put_flags] * np.exp(-r[put_flags] * t[put_flags]) * norm.cdf(-d2[put_flags])) - (S[put_flags] * np.exp(-q[put_flags] * t[put_flags]) * norm.cdf(-d1[put_flags]))
    results.loc[call_flags, 'delta'] = np.exp(-q[call_flags] * t[call_flags]) * norm.cdf(d1[call_flags])
    results.loc[put_flags, 'delta'] = -np.exp(-q[put_flags] * t[put_flags]) * norm.cdf(-d1[put_flags])
    results['gamma'] = np.exp(-q * t) * norm.pdf(d1) / (S * sigma * np.sqrt(t))
    results['vega'] = S * np.exp(-q * t) * norm.pdf(d1) * np.sqrt(t)
    theta_term1 = -(S * np.exp(-q * t) * norm.pdf(d1) * sigma) / (2 * np.sqrt(t))
    
    # --- BUG FIX ---
    # Corrected the theta formulas to properly filter all numpy arrays.
    # This prevents the ValueError by ensuring array shapes always match.
    results.loc[call_flags, 'theta'] = theta_term1[call_flags] - r[call_flags] * K[call_flags] * np.exp(-r[call_flags] * t[call_flags]) * norm.cdf(d2[call_flags]) + q[call_flags] * S[call_flags] * np.exp(-q[call_flags] * t[call_flags]) * norm.cdf(d1[call_flags])
    results.loc[put_flags, 'theta'] = theta_term1[put_flags] + r[put_flags] * K[put_flags] * np.exp(-r[put_flags] * t[put_flags]) * norm.cdf(-d2[put_flags]) - q[put_flags] * S[put_flags] * np.exp(-q[put_flags] * t[put_flags]) * norm.cdf(-d1[put_flags])
    
    return results

async def get_option_contract_details(ticker_str: str, exp_date: str, strike: float, option_type: str, opt_chain, current_price: float, dividend_yield: float) -> Optional[Dict[str, Any]]:
    """Fetches all necessary data for a single option contract for BSM modeling."""
    try:
        dte = (datetime.strptime(exp_date, "%Y-%m-%d").date() - datetime.now().date()).days
        time_to_exp = max(dte, 1) / 365.0
        chain = opt_chain.calls if option_type == 'call' else opt_chain.puts
        contract = chain[chain['strike'] == strike]
        if contract.empty: 
            return None
        return {"underlying_price": current_price, "strike": strike, "time_to_exp": time_to_exp, "iv": contract['impliedVolatility'].iloc[0], "dividend_yield": dividend_yield, "risk_free_rate": 0.045, "contract_price": contract['lastPrice'].iloc[0], "flag": 'c' if option_type == 'call' else 'p'}
    except (KeyError, IndexError) as e:
        print(f"-> Data lookup error for strike ${strike}: {e}")
        return None
    except Exception as e:
        print(f"-> An unexpected error occurred while fetching contract details: {e}")
        traceback.print_exc()
        return None

async def plot_option_visualizations(ticker: str, exp_date: str, strike: float, option_type: str, opt_chain, current_price: float, dividend_yield: float):
    """Generates 3D surface plots for an option's price and Delta."""
    details = await get_option_contract_details(ticker, exp_date, strike, option_type, opt_chain, current_price, dividend_yield)
    
    # --- BUG FIX ---
    # Added pd.isna() check to handle cases where Yahoo Finance returns invalid IV data.
    if not details or pd.isna(details.get('iv')) or details.get('iv', 0) <= 0 or details.get('time_to_exp', 0) <= 0:
        print(f"-> ERROR: Cannot generate plots for strike ${strike:.2f}. The contract may have invalid data (e.g., zero, negative, or missing Implied Volatility).")
        return

    try:
        S_range = np.linspace(details['underlying_price'] * 0.7, details['underlying_price'] * 1.3, 50)
        t_range = np.linspace(details['time_to_exp'], 0.001, 50)
        S_grid, t_grid = np.meshgrid(S_range, t_range)
        df = pd.DataFrame({'S': S_grid.flatten(), 'K': strike, 't': t_grid.flatten(), 'r': details['risk_free_rate'], 'sigma': details['iv'], 'q': details['dividend_yield'], 'Flag': details['flag']})
        
        greeks_df = calculate_greeks_numpy(df)
        
        price_grid = greeks_df['Price'].values.reshape(S_grid.shape)
        delta_grid = greeks_df['delta'].values.reshape(S_grid.shape)
        
        for grid, greek_name in [(price_grid, "Price"), (delta_grid, "Delta")]:
            fig = plt.figure(figsize=(12, 8)); ax = fig.add_subplot(111, projection='3d')
            fig.suptitle(f'{ticker} {exp_date} ${strike:.2f} {option_type.capitalize()} {greek_name} Surface', color='white')
            ax.plot_surface(S_grid, t_grid * 365, grid, cmap='viridis' if greek_name == "Price" else 'plasma')
            ax.set_xlabel('Stock Price ($)'); ax.set_ylabel('Days to Expiration'); ax.set_zlabel(f'Option {greek_name}')
            filename = f"option_{greek_name.lower()}_surface_{ticker}_{uuid.uuid4().hex[:6]}.png"
            plt.savefig(filename, facecolor='black'); plt.close(fig)
            print(f"-> {greek_name} surface plot saved as: {filename}")
    except Exception as e:
        print(f"-> An unexpected error occurred during plot generation: {e}")
        traceback.print_exc()

async def analyze_single_contract_menu(ticker: str, exp_date: str, opt_chain, current_price: float, all_strikes: list, dividend_yield: float):
    """Handles the user interaction for analyzing a single contract."""
    option_type = ask_singularity_input("Analyze a 'call' or a 'put'?")
    if not option_type or option_type.lower() not in ['call', 'put']: return
    
    if not all_strikes:
        print("-> No strikes available.")
        return
    
    display_strikes_menu(all_strikes, current_price)

    strike_idx_str = ask_singularity_input(f"Select a strike price (1-{len(all_strikes)})")
    try:
        strike_idx = int(strike_idx_str) - 1
        if not (0 <= strike_idx < len(all_strikes)):
            raise IndexError("Selected strike is out of range.")
        strike = all_strikes[strike_idx]
        await plot_option_visualizations(ticker, exp_date, strike, option_type.lower(), opt_chain, current_price, dividend_yield)
    except (ValueError, IndexError):
        # This will now only catch genuine user input errors.
        print("-> Invalid selection.")
    except Exception as e:
        print(f"-> An unexpected error occurred in the analysis menu: {e}")
        traceback.print_exc()

async def model_strategy_pnl_menu(ticker: str, exp_date: str, opt_chain, current_price: float, all_strikes: list, dividend_yield: float):
    """Handles user interaction for modeling P/L of common strategies using numbered selection."""
    print("\n    --- P/L Modeling Menu ---")
    print("    1. Long Call"); print("    2. Long Put")
    print("    3. Covered Call"); print("    4. Cash-Secured Put")
    strategy_choice = ask_singularity_input("    Select a strategy")
    
    if not all_strikes:
        print("-> No strikes available to model.")
        return
    
    display_strikes_menu(all_strikes, current_price)

    try:
        strike_idx_str = ask_singularity_input(f"    Select a strike price (1-{len(all_strikes)})")
        strike_idx = int(strike_idx_str) - 1
        if not (0 <= strike_idx < len(all_strikes)):
            raise ValueError("Index out of range.")
        
        strike = all_strikes[strike_idx]
        print(f"    -> Modeling with strike ${strike:.2f}...")

        sT_range, pnl, title = None, None, "P/L Diagram"

        if not current_price:
             print("-> Could not determine underlying price to model strategy.")
             return
        sT_range = np.linspace(current_price * 0.7, current_price * 1.3, 100)

        option_map = {'1': 'call', '2': 'put', '3': 'call', '4': 'put'}
        details = await get_option_contract_details(ticker, exp_date, strike, option_map.get(strategy_choice), opt_chain, current_price, dividend_yield)
        if not details:
            print("-> Could not retrieve contract details to model strategy.")
            return

        if strategy_choice == '1': # Long Call
            pnl = np.maximum(sT_range - strike, 0) - details['contract_price']
            title = f"Long Call P/L (Strike ${strike:.2f})"
            max_loss, break_even = -details['contract_price'], strike + details['contract_price']
            print(f"-> Max Loss: ${max_loss*100:.2f} | Max Profit: Unlimited | Break-Even: ${break_even:.2f}")

        elif strategy_choice == '2': # Long Put
            pnl = np.maximum(strike - sT_range, 0) - details['contract_price']
            title = f"Long Put P/L (Strike ${strike:.2f})"
            max_loss, break_even = -details['contract_price'], strike - details['contract_price']
            print(f"-> Max Loss: ${max_loss*100:.2f} | Max Profit: ${(strike-details['contract_price'])*100:.2f} | Break-Even: ${break_even:.2f}")
        
        elif strategy_choice == '3': # Covered Call
             pnl = np.minimum(sT_range, strike) - details['underlying_price'] + details['contract_price']
             title = f"Covered Call P/L (Strike ${strike:.2f})"
             break_even = details['underlying_price'] - details['contract_price']
             print(f"-> Max Profit: ${(strike - break_even)*100:.2f} | Break-Even: ${break_even:.2f}")
        
        elif strategy_choice == '4': # Cash-Secured Put
            pnl = np.minimum(strike - sT_range, 0) + details['contract_price']
            title = f"Cash-Secured Put P/L (Strike ${strike:.2f})"
            max_profit = details['contract_price']
            break_even = strike - details['contract_price']
            print(f"-> Max Profit: ${max_profit*100:.2f} | Max Loss: ${-break_even*100:.2f} | Break-Even: ${break_even:.2f}")

        else:
             print("-> Invalid strategy choice.")
             return

        plt.style.use('dark_background'); fig, ax = plt.subplots()
        ax.plot(sT_range, pnl, label="Profit/Loss", color="cyan"); ax.axhline(0, color='red', linestyle='--'); ax.grid(True, linestyle=':')
        ax.set_title(title, color='white'); ax.set_xlabel("Stock Price at Expiration"); ax.set_ylabel("Profit/Loss per Share")
        filename = f"pnl_diagram_{ticker}_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black'); plt.close(fig)
        print(f"-> P/L diagram saved as: {filename}")

    except (ValueError, TypeError):
        print("-> Invalid selection.")

async def recommend_short_strangle(ticker: str, exp_date: str, opt_chain, current_price: float):
    """Calculates and recommends a short strangle based on a reliable IV calculation."""
    print("\n--- Recommended Short Strangle (IV-Based) ---")
    try:
        combined_df = pd.concat([
            opt_chain.calls[['strike', 'impliedVolatility', 'openInterest']],
            opt_chain.puts[['strike', 'impliedVolatility', 'openInterest']]
        ])
        
        combined_df['distance'] = abs(combined_df['strike'] - current_price)
        nearest_strikes_df = combined_df.nsmallest(12, 'distance')
        
        liquid_contracts = nearest_strikes_df[nearest_strikes_df['openInterest'] > 10]
        if liquid_contracts.empty:
            print("-> Not enough liquid contracts near the money to calculate a reliable IV.")
            atm_strike_fallback = min(opt_chain.calls['strike'], key=lambda x: abs(x - current_price))
            reliable_iv = (opt_chain.calls[opt_chain.calls['strike'] == atm_strike_fallback]['impliedVolatility'].iloc[0] + 
                           opt_chain.puts[opt_chain.puts['strike'] == atm_strike_fallback]['impliedVolatility'].iloc[0]) / 2
            if pd.isna(reliable_iv) or reliable_iv <= 0.01:
                print(f"-> Fallback IV is also too low or invalid. Cannot proceed.")
                return
        else:
            reliable_iv = liquid_contracts['impliedVolatility'].mean()

        if pd.isna(reliable_iv) or reliable_iv <= 0.01:
            print(f"-> Cannot calculate expected move with a low or invalid IV of {reliable_iv*100:.2f}%.")
            return

        dte = (datetime.strptime(exp_date, "%Y-%m-%d").date() - datetime.now().date()).days
        expected_move = current_price * reliable_iv * sqrt(max(dte, 1) / 365.0)
        target_call_strike = current_price + expected_move
        target_put_strike = current_price - expected_move

        actual_call_strike = min(opt_chain.calls['strike'], key=lambda x: abs(x - target_call_strike))
        actual_put_strike = min(opt_chain.puts['strike'], key=lambda x: abs(x - target_put_strike))

        call_contract = opt_chain.calls[opt_chain.calls['strike'] == actual_call_strike]
        put_contract = opt_chain.puts[opt_chain.puts['strike'] == actual_put_strike]
        
        if call_contract.empty or put_contract.empty:
            print("-> Could not find matching contracts for the calculated strangle strikes.")
            return

        call_premium = call_contract['lastPrice'].iloc[0]
        put_premium = put_contract['lastPrice'].iloc[0]
        total_premium = call_premium + put_premium

        break_even_high = actual_call_strike + total_premium
        break_even_low = actual_put_strike - total_premium

        print(f"  Underlying Price: ${current_price:,.2f}")
        print(f"  Reliable IV (from {len(liquid_contracts)} liquid contracts): {reliable_iv * 100:.2f}%")
        print(f"  Calculated Expected Move: ±${expected_move:,.2f}")
        print("-" * 50)
        print(f"  Sell Call: Strike ${actual_call_strike:,.2f} (Premium: ${call_premium:.2f})")
        print(f"  Sell Put:  Strike ${actual_put_strike:,.2f} (Premium: ${put_premium:.2f})")
        print("-" * 50)
        print(f"  Total Premium Collected: ${total_premium:.2f} (per share)")
        print(f"  Max Profit: ${total_premium * 100:,.2f}")
        print(f"  Max Loss: Unlimited")
        print(f"  Break-Even Points: ${break_even_low:,.2f} and ${break_even_high:,.2f}")

    except Exception as e:
        print(f"-> Failed to generate short strangle recommendation: {e}")
        traceback.print_exc()

# --- Main Command Handler ---

async def handle_options_command(args: List[str], is_called_by_ai: bool = False):
    if is_called_by_ai:
        return "This is an interactive CLI-only command."
    try:
        ticker = ask_singularity_input("Enter the stock ticker for options analysis")
        if not ticker: return
        stock_yf = yf.Ticker(ticker.upper())
        
        try:
            info = await asyncio.to_thread(lambda: stock_yf.info)
            current_price = info.get('regularMarketPrice') or info.get('currentPrice')
            dividend_yield = info.get('dividendYield', 0.0)
            exp_dates = await asyncio.to_thread(lambda: stock_yf.options)
        except Exception:
            print(f"-> Error fetching data for {ticker.upper()}. It may be an invalid ticker or a network issue.")
            return
        
        if not exp_dates or not current_price: 
            print(f"-> No options data or price found for {ticker.upper()}."); return

        print("\nAvailable expiration dates:")
        for i, date in enumerate(exp_dates): print(f"  {i+1}. {date}")
        date_idx_str = ask_singularity_input(f"Select an expiration date (1-{len(exp_dates)})")
        selected_exp_date = exp_dates[int(date_idx_str) - 1]
        
        opt_chain = await asyncio.to_thread(stock_yf.option_chain, selected_exp_date)
        all_strikes = sorted(opt_chain.calls['strike'].tolist()) if not opt_chain.calls.empty else []
        
        while True:
            print(f"\n--- Options Menu for {ticker.upper()} on {selected_exp_date} ---")
            print("1. Analyze Single Contract"); print("2. Model Strategy P/L"); print("3. Recommend Short Strangle"); print("4. Exit")
            choice = ask_singularity_input("Select an option (1-4)")
            if choice == '1': 
                await analyze_single_contract_menu(ticker, selected_exp_date, opt_chain, current_price, all_strikes, dividend_yield)
            elif choice == '2': 
                await model_strategy_pnl_menu(ticker, selected_exp_date, opt_chain, current_price, all_strikes, dividend_yield)
            elif choice == '3': 
                await recommend_short_strangle(ticker, selected_exp_date, opt_chain, current_price)
            elif choice == '4': 
                break
    except (ValueError, IndexError):
        print("-> Invalid selection. Returning to main menu.")
    except Exception as e:
        print(f"An error occurred in the options module: {e}")
        traceback.print_exc()