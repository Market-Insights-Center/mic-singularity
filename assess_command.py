# --- Imports for assess_command ---
import asyncio
import os
import csv
import uuid
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import percentileofscore

# --- Imports from other command modules ---
from invest_command import plot_ticker_graph, calculate_ema_invest, process_custom_portfolio
from cultivate_command import run_cultivate_analysis_singularity

# --- Constants (copied for self-containment) ---
PORTFOLIO_DB_FILE = 'portfolio_codes_database.csv'

# --- Helper Functions (copied or moved for self-containment) ---

async def get_yf_download_robustly(tickers: list, **kwargs) -> pd.DataFrame:
    for attempt in range(3):
        try:
            data = await asyncio.to_thread(yf.download, tickers=tickers, progress=False, **kwargs)
            if not data.empty:
                return data
        except Exception:
            if attempt < 2: await asyncio.sleep((attempt + 1) * 2)
    return pd.DataFrame()

async def get_yf_data_singularity(tickers: List[str], period: str = None, interval: str = "1d", start: str = None, end: str = None, is_called_by_ai: bool = False) -> pd.DataFrame:
    if not tickers: return pd.DataFrame()
    data = await get_yf_download_robustly(tickers=list(set(tickers)), period=period, interval=interval, start=start, end=end, auto_adjust=False, group_by='ticker', timeout=30)
    if data.empty: return pd.DataFrame()
    all_series = []
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in list(set(tickers)):
            if (ticker, 'Close') in data.columns:
                series = pd.to_numeric(data[(ticker, 'Close')], errors='coerce').dropna()
                if not series.empty: series.name = ticker; all_series.append(series)
    elif 'Close' in data.columns:
        series = pd.to_numeric(data['Close'], errors='coerce').dropna()
        if not series.empty: series.name = list(set(tickers))[0]; all_series.append(series)
    if not all_series: return pd.DataFrame()
    df_out = pd.concat(all_series, axis=1)
    df_out.index = pd.to_datetime(df_out.index)
    return df_out.dropna(how='all')

async def calculate_volatility_metrics(ticker: str, period: str) -> tuple[Optional[float], Optional[float]]:
    try:
        stock_yf = yf.Ticker(ticker)
        hist_data = await asyncio.to_thread(stock_yf.history, period=period)
        if hist_data.empty or len(hist_data) <= 30:
            return None, None

        # --- Volatility Rank Calculation (Existing Logic) ---
        hist_data['daily_return'] = hist_data['Close'].pct_change()
        rolling_hv = hist_data['daily_return'].rolling(window=30).std() * (252**0.5)
        hv_series = rolling_hv.dropna()
        vol_rank = percentileofscore(hv_series, hv_series.iloc[-1]) if len(hv_series) > 1 else None

        # --- New Implied Volatility (IV) Calculation Logic ---
        current_iv = None
        try:
            expirations = stock_yf.options
            if expirations:
                # Find the expiration date closest to 30 days from now
                today = datetime.now()
                target_date = today + timedelta(days=30)
                exp_dates = [datetime.strptime(d, '%Y-%m-%d') for d in expirations]
                closest_date = min(exp_dates, key=lambda d: abs(d - target_date))
                closest_date_str = closest_date.strftime('%Y-%m-%d')
                
                # Fetch the options chain for that date
                opt_chain = await asyncio.to_thread(stock_yf.option_chain, closest_date_str)
                last_price = hist_data['Close'].iloc[-1]

                # Find the at-the-money (ATM) call and put
                atm_call = opt_chain.calls.iloc[(opt_chain.calls['strike'] - last_price).abs().idxmin()]
                atm_put = opt_chain.puts.iloc[(opt_chain.puts['strike'] - last_price).abs().idxmin()]

                # Average the IV of the ATM call and put
                iv_call = atm_call.get('impliedVolatility')
                iv_put = atm_put.get('impliedVolatility')
                
                valid_ivs = [iv for iv in [iv_call, iv_put] if pd.notna(iv) and iv > 0]
                if valid_ivs:
                    current_iv = sum(valid_ivs) / len(valid_ivs)

        except Exception:
            # Silently fail if options data is unavailable or calculation fails
            current_iv = None

        return current_iv, vol_rank
        
    except Exception:
        return None, None

def ask_singularity_input(prompt: str, validation_fn=None, error_msg: str = "Invalid input.", is_called_by_ai: bool = False) -> Optional[str]:
    if is_called_by_ai: return None
    while True:
        user_response = input(f"{prompt}: ").strip()
        if validation_fn:
            if validation_fn(user_response): return user_response
            else: print(error_msg)
        else: return user_response

async def calculate_portfolio_beta_correlation_singularity(portfolio_holdings: List[Dict], total_value: float, period: str, is_called_by_ai: bool) -> Optional[tuple[float, float]]:
    if not portfolio_holdings or total_value <= 0: return None
    stock_tickers = [h['ticker'] for h in portfolio_holdings if h['ticker'].upper() != 'CASH']
    if not stock_tickers: return 0.0, 0.0
    hist_data = await get_yf_data_singularity(stock_tickers + ['SPY'], period=period, is_called_by_ai=True)
    if hist_data.empty or 'SPY' not in hist_data.columns: return None
    returns = hist_data.pct_change().dropna()
    if returns.empty: return None
    
    beta_sum, corr_sum = 0.0, 0.0
    for holding in portfolio_holdings:
        ticker, value = holding['ticker'], holding['value']
        if ticker.upper() == 'CASH': continue
        weight = value / total_value
        if ticker in returns.columns:
            cov = returns[ticker].cov(returns['SPY'])
            market_var = returns['SPY'].var()
            beta = cov / market_var if market_var != 0 else 0.0
            corr = returns[ticker].corr(returns['SPY'])
            beta_sum += weight * beta
            corr_sum += weight * (corr if pd.notna(corr) else 0.0)
    return beta_sum, corr_sum

def calculate_backtest_performance(portfolio_value_history: pd.Series) -> dict:
    if portfolio_value_history.empty: return {"final_value": 0, "total_return_pct": 0, "max_drawdown_pct": 0}
    initial = portfolio_value_history.iloc[0]
    final = portfolio_value_history.iloc[-1]
    total_return = ((final / initial) - 1) * 100
    running_max = portfolio_value_history.cummax()
    drawdown = (portfolio_value_history - running_max) / running_max
    return {"final_value": final, "total_return_pct": total_return, "max_drawdown_pct": drawdown.min() * 100}

def plot_backtest_performance_graph(portfolio_history: pd.Series, spy_history: pd.Series):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))
    portfolio_norm = (portfolio_history / portfolio_history.iloc[0]) * 100
    spy_norm = (spy_history / spy_history.iloc[0]) * 100
    ax.plot(portfolio_norm.index, portfolio_norm, label='Your Portfolio', color='cyan')
    ax.plot(spy_norm.index, spy_norm, label='SPY', color='red', linestyle='--')
    ax.set_title('Backtest Performance vs. SPY', color='white')
    ax.set_ylabel('Performance (Normalized to 100)', color='white')
    ax.legend()
    ax.grid(True, alpha=0.3)
    filename = f"backtest_performance_{uuid.uuid4().hex[:6]}.png"
    plt.savefig(filename, facecolor='black')
    plt.close(fig)
    print(f"📂 Performance graph saved: {filename}")

async def run_portfolio_backtest(portfolio_code: str, start_date: str, end_date: str, initial_value: float = 10000.0, is_called_by_ai: bool = False):
    if not os.path.exists(PORTFOLIO_DB_FILE):
        print(f"❌ Error: Portfolio database file '{PORTFOLIO_DB_FILE}' not found.")
        return
    try:
        df = pd.read_csv(PORTFOLIO_DB_FILE)
        config = df[df['portfolio_code'].astype(str) == portfolio_code].iloc[0]
        tickers = set()
        for i in range(1, int(config.get('num_portfolios', 0)) + 1):
            tickers.update([t.strip() for t in config.get(f'tickers_{i}', '').split(',') if t.strip()])
        
        fetch_start = (pd.to_datetime(start_date) - timedelta(weeks=75)).strftime('%Y-%m-%d')
        hist_data = await get_yf_data_singularity(list(tickers) + ['SPY'], start=fetch_start, end=end_date)
        if hist_data.empty: return

        sim_data = hist_data.loc[start_date:end_date].copy()
        buying_power = initial_value
        shares = {}
        value_history = pd.Series(index=sim_data.index, dtype=float)
        last_rebalance_month = -1

        for date, prices in sim_data.iterrows():
            if date.month != last_rebalance_month:
                current_value = buying_power + sum(shares.get(t, 0) * prices.get(t, 0) for t in shares if pd.notna(prices.get(t)))
                # This block now correctly defines the config and uses the is_called_by_ai flag
                portfolio_config_for_rebalance = df[df['portfolio_code'].astype(str) == portfolio_code].iloc[0].to_dict()
                portfolio_result = await process_custom_portfolio(    
                    portfolio_data_config=portfolio_config_for_rebalance,
                    tailor_portfolio_requested=True,
                    total_value_singularity=current_value,
                    frac_shares_singularity=True,
                    is_custom_command_simplified_output=True,
                    is_called_by_ai=True # Internal call should be treated as AI
                )

                if len(portfolio_result) == 5:
                    _, _, _, structured_holdings, _ = portfolio_result
                    if structured_holdings:
                        shares = {h['ticker']: h['shares'] for h in structured_holdings}
                        buying_power = current_value - sum(h.get('actual_money_allocation', 0) for h in structured_holdings)
                    else:
                        shares = {}
                        buying_power = current_value
                else:
                    # This print statement now correctly uses the passed-in variable
                    if not is_called_by_ai:
                        print(f"⚠️ Warning: Rebalancing failed on {date.strftime('%Y-%m-%d')} due to missing data. Holding cash.")
                    shares = {}
                    buying_power = current_value

                last_rebalance_month = date.month

            current_holdings_value = sum(shares.get(t, 0) * prices.get(t, 0) for t in shares if pd.notna(prices.get(t)))
            value_history[date] = buying_power + current_holdings_value
        
        performance = calculate_backtest_performance(value_history.dropna())
        print(f"\n--- Backtest Results for '{portfolio_code}' ---")
        print(f"Final Value: ${performance['final_value']:,.2f} | Total Return: {performance['total_return_pct']:+.2f}%")
        spy_hist = sim_data['SPY'].loc[value_history.dropna().index]
        plot_backtest_performance_graph(value_history.dropna(), spy_hist)
    except Exception as e:
        print(f"❌ Error during backtest: {e}")

# --- Main Command Handler ---
async def handle_assess_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """
    Handles the /assess command for CLI and AI.
    """
    summary_for_ai = "Assessment initiated."
    assess_code_input, specific_params_dict = None, {}

    if ai_params: 
        assess_code_input = ai_params.get("assess_code", "").upper()
        specific_params_dict = ai_params 
    elif args: 
        assess_code_input = args[0].upper()
    else: 
        msg = "Usage: /assess <AssessCode A/B/C/D/E> [additional_args...]. Type /help for details."
        if not is_called_by_ai: print(msg)
        return "Error: Assess code (A, B, C, D, or E) not specified." if is_called_by_ai else None

    # --- Code A: Stock Volatility Assessment ---
    if assess_code_input == 'A':
        if not is_called_by_ai: print("--- Assess Code A: Stock Volatility ---")
        tickers_str_a, timeframe_str_a, risk_tolerance_a_int = None, None, None
        try:
            if ai_params:
                tickers_str_a = specific_params_dict.get("tickers_str")
                timeframe_str_a = specific_params_dict.get("timeframe_str", "1Y") 
                risk_tolerance_a_int = int(specific_params_dict.get("risk_tolerance", 3)) 
                if not tickers_str_a:
                    return "Error for AI (Assess A): 'tickers_str' (comma-separated tickers) is required."
            elif len(args) >= 4: 
                tickers_str_a = args[1]
                timeframe_str_a = args[2].upper()
                risk_tolerance_a_int = int(args[3])
            else:
                msg_cli_a = "CLI Usage: /assess A <tickers_comma_sep> <timeframe 1Y/3Y/5Y/10Y/20Y/3M/1M> <risk_tolerance 1-5>"
                if not is_called_by_ai: print(msg_cli_a)
                return "Error: Insufficient arguments for Assess Code A." if is_called_by_ai else None

            tickers_list_a = [t.strip().upper() for t in tickers_str_a.split(',') if t.strip()]
            if not tickers_list_a:
                return f"Error (Assess A): No valid tickers found in '{tickers_str_a}'."

            timeframe_upper_a = timeframe_str_a.upper()
            timeframe_map_a = {'1Y': "1y", '3Y': "3y", '5Y': "5y", '10Y': "10y", '20Y': "20y", '3M': "3mo", '1M': "1mo"}
            plot_ema_map_a = {'1Y': 2, '3Y': 1, '5Y': 1, '10Y': 1, '20Y': 1, '3M': 3, '1M': 3} 

            if timeframe_upper_a not in timeframe_map_a:
                return f"Error (Assess A): Invalid timeframe '{timeframe_str_a}'. Use one of: 1Y, 3Y, 5Y, 10Y, 20Y, 3M, 1M."
            selected_yf_period_a = timeframe_map_a[timeframe_upper_a]
            plot_ema_sensitivity_a = plot_ema_map_a[timeframe_upper_a]

            if not (1 <= risk_tolerance_a_int <= 5):
                return f"Error (Assess A): Invalid risk tolerance '{risk_tolerance_a_int}'. Must be 1-5."

            results_for_table_a = []
            assessment_summaries_ai_list = []

            if not is_called_by_ai:
                for ticker_a_graph in tickers_list_a:
                    await asyncio.to_thread(plot_ticker_graph, ticker_a_graph, plot_ema_sensitivity_a, is_called_by_ai=is_called_by_ai)

            for ticker_a_item in tickers_list_a:
                try:
                    hist_df_a = await get_yf_data_singularity([ticker_a_item], period=selected_yf_period_a, is_called_by_ai=True)
                    if hist_df_a.empty or ticker_a_item not in hist_df_a.columns or len(hist_df_a[ticker_a_item].dropna()) <= 1:
                        results_for_table_a.append([ticker_a_item, "N/A", "N/A", "N/A", "N/A", "N/A", "Data Error"])
                        assessment_summaries_ai_list.append(f"{ticker_a_item}: Data Error.")
                        continue
                    
                    current_iv, vol_rank = await calculate_volatility_metrics(ticker_a_item, selected_yf_period_a)
                    iv_display = f"{(current_iv * 100):.1f}%" if current_iv is not None else "N/A"
                    vr_display = f"{vol_rank:.1f}%" if vol_rank is not None else "N/A"

                    close_prices_series_a = hist_df_a[ticker_a_item].dropna()
                    abs_daily_pct_change_a = close_prices_series_a.pct_change().abs() * 100
                    aapc_val_a = abs_daily_pct_change_a.iloc[1:].mean() if len(abs_daily_pct_change_a.iloc[1:]) > 0 else 0.0
                    vol_score_map_a = [(1,0),(2,1),(3,2),(4,3),(5,4),(6,5),(7,6),(8,7),(9,8),(10,9)] 
                    volatility_score_a = 10 
                    for aapc_thresh, score_val in vol_score_map_a:
                        if aapc_val_a <= aapc_thresh: volatility_score_a = score_val; break
                    start_price_a = close_prices_series_a.iloc[0]
                    end_price_a = close_prices_series_a.iloc[-1]
                    period_change_pct_a = ((end_price_a - start_price_a) / start_price_a) * 100 if start_price_a != 0 else 0.0
                    risk_tolerance_ranges_map = {1:(0,1), 2:(2,3), 3:(4,5), 4:(6,7), 5:(8,10)} 
                    vol_score_min, vol_score_max = risk_tolerance_ranges_map[risk_tolerance_a_int]
                    correspondence_a = "Matches" if vol_score_min <= volatility_score_a <= vol_score_max else "No Match"
                    
                    results_for_table_a.append([ticker_a_item, f"{period_change_pct_a:.2f}%", f"{aapc_val_a:.2f}%", volatility_score_a, iv_display, vr_display, correspondence_a])
                    assessment_summaries_ai_list.append(f"{ticker_a_item}({timeframe_upper_a},RT{risk_tolerance_a_int}):Chg {period_change_pct_a:.1f}%,VolSc {volatility_score_a},IV {iv_display},VolRank {vr_display},Match:{correspondence_a}")

                except Exception as e_item_a:
                    results_for_table_a.append([ticker_a_item, "CalcErr", "CalcErr", "CalcErr", "CalcErr", "CalcErr", f"Error"])
                    assessment_summaries_ai_list.append(f"{ticker_a_item}: Calculation Error ({e_item_a}).")

            if results_for_table_a and not is_called_by_ai: 
                print("\n**Stock Volatility Assessment Results (Code A)**")
                results_for_table_a.sort(key=lambda x: x[3] if isinstance(x[3], (int,float)) else float('inf'))
                
                headers = ["Ticker", f"{timeframe_upper_a} Change", "AAPC (%)", "Vol Score (0-9)", "Current IV", "Volatility Rank", "Risk Match"]
                print(tabulate(results_for_table_a, headers=headers, tablefmt="pretty"))
                
            summary_for_ai = "Assess A (Stock Volatility) results: " + " | ".join(assessment_summaries_ai_list) if assessment_summaries_ai_list else "No results for Assess A."
        
        except ValueError as ve_a:
            summary_for_ai = f"Error (Assess A): Invalid parameter type (e.g., risk tolerance not a number). {ve_a}"
            if not is_called_by_ai: print(summary_for_ai)
        except Exception as e_assess_a:
            summary_for_ai = f"An unexpected error occurred in Assess Code A: {e_assess_a}"
            if not is_called_by_ai: print(summary_for_ai); traceback.print_exc()
        return summary_for_ai

    # --- Code B: Manual Portfolio Assessment ---
    elif assess_code_input == 'B':
        if not is_called_by_ai: print("--- Assess Code B: Manual Portfolio Risk (Beta/Correlation) ---")
        backtest_period_b_str, manual_portfolio_holdings_b_list = None, []
        try:
            if ai_params:
                backtest_period_b_str = specific_params_dict.get("backtest_period_str", "1y") 
                raw_holdings_ai = specific_params_dict.get("manual_portfolio_holdings")
                if not raw_holdings_ai or not isinstance(raw_holdings_ai, list):
                    return "Error for AI (Assess B): 'manual_portfolio_holdings' (list of dicts with ticker/shares or ticker/value) is required."
                for holding_item_ai in raw_holdings_ai:
                    ticker_b_item = str(holding_item_ai.get("ticker", "")).upper().strip()
                    shares_b_item_raw = holding_item_ai.get("shares")
                    value_b_item_raw = holding_item_ai.get("value") 
                    if not ticker_b_item: continue 
                    if ticker_b_item == "CASH":
                        if value_b_item_raw is not None:
                            manual_portfolio_holdings_b_list.append({'ticker': 'CASH', 'value': float(value_b_item_raw)})
                        else: return "Error for AI (Assess B): Cash holding needs a 'value'."
                    elif shares_b_item_raw is not None: 
                        shares_b_item = float(shares_b_item_raw)
                        live_price_b_item, _ = await calculate_ema_invest(ticker_b_item, 2, is_called_by_ai=True) 
                        if live_price_b_item is not None and live_price_b_item > 0:
                            holding_value_b = shares_b_item * live_price_b_item
                            manual_portfolio_holdings_b_list.append({'ticker': ticker_b_item, 'value': holding_value_b, 'shares': shares_b_item, 'price_at_eval': live_price_b_item})
                        else: return f"Error for AI (Assess B): Could not fetch live price for {ticker_b_item} to calculate value from shares."
                    elif value_b_item_raw is not None: 
                         manual_portfolio_holdings_b_list.append({'ticker': ticker_b_item, 'value': float(value_b_item_raw)})
                    else: 
                        return f"Error for AI (Assess B): Holding '{ticker_b_item}' needs 'shares' or 'value'."
            elif len(args) >= 2: 
                backtest_period_b_str = args[1].lower()
                if not is_called_by_ai: 
                    print(f"CLI Backtesting Period for Manual Portfolio: {backtest_period_b_str}")
                    print("Enter portfolio holdings (ticker and shares/value). Type 'cash' for cash, 'done' when finished.")
                    while True:
                        ticker_input_b = ask_singularity_input("Enter ticker (or 'cash', 'done')", is_called_by_ai=is_called_by_ai)
                        if ticker_input_b is None: break 
                        ticker_input_b = ticker_input_b.upper().strip()
                        if ticker_input_b == 'DONE': break
                        if ticker_input_b == 'CASH':
                            cash_value_str = ask_singularity_input("Enter cash value", lambda x: float(x) >= 0, "Cash value must be non-negative.", is_called_by_ai=is_called_by_ai)
                            if cash_value_str is None: continue 
                            manual_portfolio_holdings_b_list.append({'ticker': 'CASH', 'value': float(cash_value_str)})
                        else: 
                            shares_str_b = ask_singularity_input(f"Enter shares for {ticker_input_b}", lambda x: float(x) > 0, "Shares must be positive.", is_called_by_ai=is_called_by_ai)
                            if shares_str_b is None: continue
                            shares_b = float(shares_str_b)
                            live_price_b_cli, _ = await calculate_ema_invest(ticker_input_b, 2, is_called_by_ai=True) 
                            if live_price_b_cli is not None and live_price_b_cli > 0:
                                value_b_cli = shares_b * live_price_b_cli
                                manual_portfolio_holdings_b_list.append({'ticker': ticker_input_b, 'value': value_b_cli, 'shares': shares_b, 'price_at_eval': live_price_b_cli})
                                if not is_called_by_ai: print(f"Added {shares_b} shares of {ticker_input_b} at ${live_price_b_cli:.2f} (Value: ${value_b_cli:.2f})")
                            else:
                                if not is_called_by_ai: print(f"Could not fetch price for {ticker_input_b}. Not added.")
            else: 
                msg_cli_b = "CLI Usage for Code B: /assess B <backtest_period 1y/5y/10y> (then prompts for holdings)"
                if not is_called_by_ai: print(msg_cli_b)
                return "Error: Insufficient arguments for Assess Code B." if is_called_by_ai else None

            if backtest_period_b_str not in ['1y', '3y', '5y', '10y']: 
                return f"Error (Assess B): Invalid backtest period '{backtest_period_b_str}'. Use 1y, 3y, 5y, or 10y."
            if not manual_portfolio_holdings_b_list:
                return "Error (Assess B): No portfolio holdings were provided or derived."
            total_value_b_calc = sum(h.get('value', 0.0) for h in manual_portfolio_holdings_b_list)
            if total_value_b_calc <= 0:
                return "Error (Assess B): Total portfolio value is zero or negative based on inputs."

            if not is_called_by_ai:
                print(f"\n--- Manual Portfolio Summary (Assess B) ---")
                print(f"Total Calculated Portfolio Value: ${total_value_b_calc:,.2f}")
                print("Holdings for Beta/Correlation Calculation:")
                for h_disp in manual_portfolio_holdings_b_list:
                    share_price_info = ""
                    if 'shares' in h_disp and 'price_at_eval' in h_disp:
                        share_price_info = f" ({h_disp['shares']} shares @ ${h_disp['price_at_eval']:.2f})"
                    print(f"  - {h_disp['ticker']}: Value ${h_disp['value']:,.2f}{share_price_info}")

            beta_corr_tuple_b = await calculate_portfolio_beta_correlation_singularity(
                manual_portfolio_holdings_b_list, total_value_b_calc, backtest_period_b_str, is_called_by_ai=is_called_by_ai
            )
            if beta_corr_tuple_b:
                weighted_beta_b, weighted_corr_b = beta_corr_tuple_b
                result_text_b = (f"Manual Portfolio Assessment (Value: ${total_value_b_calc:,.2f}, Period: {backtest_period_b_str}): "
                                 f"Weighted Avg Beta vs SPY: {weighted_beta_b:.4f}, Weighted Avg Correlation to SPY: {weighted_corr_b:.4f}.")
                if not is_called_by_ai:
                    print("\n**Manual Portfolio Risk Assessment Results (Code B)**")
                    print(f"  Backtest Period: {backtest_period_b_str}")
                    print(f"  Weighted Average Beta vs SPY: {weighted_beta_b:.4f}")
                    print(f"  Weighted Average Correlation to SPY: {weighted_corr_b:.4f}")
                summary_for_ai = result_text_b
            else:
                summary_for_ai = f"Could not calculate Beta/Correlation for the manual portfolio (Period: {backtest_period_b_str}). Check holdings data or market conditions."
                if not is_called_by_ai: print(summary_for_ai)
        except ValueError as ve_b: 
            summary_for_ai = f"Error (Assess B): Invalid numerical input or data format. {ve_b}"
            if not is_called_by_ai: print(summary_for_ai)
        except Exception as e_assess_b:
            summary_for_ai = f"An unexpected error occurred in Assess Code B: {e_assess_b}"
            if not is_called_by_ai: print(summary_for_ai); traceback.print_exc()
        return summary_for_ai

    # --- Code C: Custom Saved Portfolio Risk Assessment ---
    elif assess_code_input == 'C':
        if not is_called_by_ai: print("--- Assess Code C: Custom Saved Portfolio Risk (Beta/Correlation) ---")
        custom_portfolio_code_c, value_for_assess_c_float, backtest_period_c_str = None, None, None
        try:
            if ai_params:
                custom_portfolio_code_c = specific_params_dict.get("custom_portfolio_code")
                value_raw_c = specific_params_dict.get("value_for_assessment")
                backtest_period_c_str = specific_params_dict.get("backtest_period_str", "3y") 

                if not custom_portfolio_code_c or value_raw_c is None:
                    return "Error for AI (Assess C): 'custom_portfolio_code' and 'value_for_assessment' are required."
                value_for_assess_c_float = float(value_raw_c)
                if value_for_assess_c_float <= 0: return "Error for AI (Assess C): 'value_for_assessment' must be positive."
            elif len(args) >= 4: # assess C CODE VALUE PERIOD
                custom_portfolio_code_c = args[1] # Switched from 2 to 1
                value_for_assess_c_float = float(args[2]) # Switched from 3 to 2
                if value_for_assess_c_float <= 0: print("CLI Error: Value for assessment must be positive."); return None
                backtest_period_c_str = args[3].lower() # Switched from 4 to 3
            else:
                msg_cli_c = "CLI Usage for Code C: /assess C <custom_portfolio_code> <value_for_assessment> <backtest_period 1y/3y/5y/10y>"
                if not is_called_by_ai: print(msg_cli_c)
                return "Error: Insufficient arguments for Assess Code C." if is_called_by_ai else None

            if backtest_period_c_str not in ['1y', '3y', '5y', '10y']:
                return f"Error (Assess C): Invalid backtest period '{backtest_period_c_str}'. Use 1y, 3y, 5y, or 10y."

            custom_config_data_c = None
            if not os.path.exists(PORTFOLIO_DB_FILE): 
                return f"Error (Assess C): Portfolio database '{PORTFOLIO_DB_FILE}' not found."
            with open(PORTFOLIO_DB_FILE, 'r', encoding='utf-8', newline='') as file_c_db:
                reader_c_db = csv.DictReader(file_c_db)
                for row_c_db in reader_c_db:
                    if row_c_db.get('portfolio_code','').strip().lower() == custom_portfolio_code_c.lower():
                        custom_config_data_c = row_c_db; break
            if not custom_config_data_c:
                return f"Error (Assess C): Custom portfolio code '{custom_portfolio_code_c}' not found in database."
            
            csv_frac_shares_str_c = custom_config_data_c.get('frac_shares', 'false').strip().lower()
            frac_shares_from_config_c = csv_frac_shares_str_c in ['true', 'yes']

            if not is_called_by_ai: print(f"  Generating tailored holdings for '{custom_portfolio_code_c}' with value ${value_for_assess_c_float:,.2f}...")
            _, _, final_cash_val_c, structured_holdings_list_c = await process_custom_portfolio(
                portfolio_data_config=custom_config_data_c,
                tailor_portfolio_requested=True, 
                frac_shares_singularity=frac_shares_from_config_c,
                total_value_singularity=value_for_assess_c_float,
                is_custom_command_simplified_output=True, 
                is_called_by_ai=True 
            )

            portfolio_holdings_for_beta_c = [] 
            if structured_holdings_list_c: 
                for item_h_c in structured_holdings_list_c:
                    if item_h_c.get('actual_money_allocation', 0) > 1e-9 : 
                        portfolio_holdings_for_beta_c.append({'ticker': item_h_c['ticker'], 'value': float(item_h_c['actual_money_allocation'])})
            if final_cash_val_c > 1e-9: 
                portfolio_holdings_for_beta_c.append({'ticker': 'CASH', 'value': float(final_cash_val_c)})

            if not portfolio_holdings_for_beta_c :
                summary_for_ai = f"Assess C for '{custom_portfolio_code_c}' (Value ${value_for_assess_c_float:,.0f}): No holdings derived from portfolio configuration for beta/correlation analysis."
                if not is_called_by_ai: print(summary_for_ai)
                return summary_for_ai 

            beta_corr_tuple_c = await calculate_portfolio_beta_correlation_singularity(
                portfolio_holdings_for_beta_c, value_for_assess_c_float, backtest_period_c_str, is_called_by_ai=is_called_by_ai
            )

            if beta_corr_tuple_c:
                weighted_beta_c, weighted_corr_c = beta_corr_tuple_c
                summary_for_ai = (f"Custom Portfolio '{custom_portfolio_code_c}' (Assessed Value ${value_for_assess_c_float:,.2f}, Period {backtest_period_c_str}): "
                                  f"Weighted Avg Beta vs SPY: {weighted_beta_c:.4f}, Weighted Avg Correlation to SPY: {weighted_corr_c:.4f}.")
                if not is_called_by_ai:
                    print("\n**Custom Portfolio Risk Assessment Results (Code C)**")
                    print(f"  Portfolio Code: {custom_portfolio_code_c}, Assessed Value: ${value_for_assess_c_float:,.2f}")
                    print(f"  Backtest Period: {backtest_period_c_str}")
                    print(f"  Weighted Average Beta vs SPY: {weighted_beta_c:.4f}")
                    print(f"  Weighted Average Correlation to SPY: {weighted_corr_c:.4f}")
            else:
                summary_for_ai = f"Could not calculate Beta/Correlation for custom portfolio '{custom_portfolio_code_c}' (Period {backtest_period_c_str})."
                if not is_called_by_ai: print(summary_for_ai)
        except ValueError as ve_c: 
            summary_for_ai = f"Error (Assess C): Invalid numerical input for value_for_assessment. {ve_c}"
            if not is_called_by_ai: print(summary_for_ai)
        except Exception as e_assess_c:
            summary_for_ai = f"An unexpected error occurred in Assess Code C: {e_assess_c}"
            if not is_called_by_ai: print(summary_for_ai); traceback.print_exc()
        return summary_for_ai

    # --- Code D: Cultivate Portfolio Risk Assessment ---
    elif assess_code_input == 'D':
        if not is_called_by_ai: print("--- Assess Code D: Cultivate Portfolio Risk (Beta/Correlation) ---")
        cultivate_code_d_str, value_epsilon_d_float, frac_s_d_bool, backtest_period_d_str = None, None, None, None
        try:
            if ai_params:
                cultivate_code_d_str = specific_params_dict.get("cultivate_portfolio_code", "").upper()
                value_eps_raw_d = specific_params_dict.get("value_for_assessment") 
                frac_s_d_bool = specific_params_dict.get("use_fractional_shares", False) 
                backtest_period_d_str = specific_params_dict.get("backtest_period_str", "5y") 

                if cultivate_code_d_str not in ['A', 'B'] or value_eps_raw_d is None:
                    return "Error for AI (Assess D): Valid 'cultivate_portfolio_code' (A/B) and 'value_for_assessment' required."
                value_epsilon_d_float = float(value_eps_raw_d)
                if value_epsilon_d_float <= 0: return "Error for AI (Assess D): 'value_for_assessment' must be positive."
            elif len(args) >= 5: # assess D CODE VALUE FRAC_S PERIOD
                cultivate_code_d_str = args[1].upper() # Switched from 2 to 1
                value_epsilon_d_float = float(args[2]) # Switched from 3 to 2
                if value_epsilon_d_float <= 0: print("CLI Error: Value for Epsilon must be positive."); return None
                frac_s_str_d = args[3].lower() # Switched from 4 to 3
                if frac_s_str_d not in ['yes', 'no']: print("CLI Error: Fractional shares must be 'yes' or 'no'."); return None
                frac_s_d_bool = frac_s_str_d == 'yes'
                backtest_period_d_str = args[4].lower() # Switched from 5 to 4
            else:
                msg_cli_d = "CLI Usage for Code D: /assess D <cultivate_code A/B> <value_epsilon> <frac_shares y/n> <backtest_period 1y/3y/5y/10y>"
                if not is_called_by_ai: print(msg_cli_d)
                return "Error: Insufficient arguments for Assess Code D." if is_called_by_ai else None

            if cultivate_code_d_str not in ['A', 'B']: return f"Error (Assess D): Invalid Cultivate Code '{cultivate_code_d_str}'. Use A or B."
            if backtest_period_d_str not in ['1y', '3y', '5y', '10y']:
                 return f"Error (Assess D): Invalid backtest period '{backtest_period_d_str}'. Use 1y, 3y, 5y, or 10y."

            if not is_called_by_ai: print(f"  Running Cultivate analysis (Code {cultivate_code_d_str}, Epsilon ${value_epsilon_d_float:,.0f}) to get holdings for assessment...")
            _, tailored_holdings_list_d, final_cash_val_d, _, _, _, err_msg_cult_d = await run_cultivate_analysis_singularity(
                portfolio_value=value_epsilon_d_float,
                frac_shares=frac_s_d_bool,
                cultivate_code_str=cultivate_code_d_str,
                is_called_by_ai=True, 
                is_saving_run=True    
            )
            if err_msg_cult_d:
                return f"Error (Assess D) during underlying Cultivate run: {err_msg_cult_d}"

            portfolio_holdings_for_beta_d = [] 
            if tailored_holdings_list_d: 
                for item_h_d in tailored_holdings_list_d:
                    if item_h_d.get('actual_money_allocation', 0) > 1e-9:
                        portfolio_holdings_for_beta_d.append({'ticker': item_h_d['ticker'], 'value': float(item_h_d['actual_money_allocation'])})
            if final_cash_val_d > 1e-9:
                portfolio_holdings_for_beta_d.append({'ticker': 'CASH', 'value': float(final_cash_val_d)})

            if not portfolio_holdings_for_beta_d:
                summary_for_ai = f"Assess D for Cultivate Code '{cultivate_code_d_str}' (Epsilon ${value_epsilon_d_float:,.0f}): No holdings derived from Cultivate analysis."
                if not is_called_by_ai: print(summary_for_ai)
                return summary_for_ai

            beta_corr_tuple_d = await calculate_portfolio_beta_correlation_singularity(
                portfolio_holdings_for_beta_d, value_epsilon_d_float, backtest_period_d_str, is_called_by_ai=is_called_by_ai
            )
            if beta_corr_tuple_d:
                weighted_beta_d, weighted_corr_d = beta_corr_tuple_d
                summary_for_ai = (f"Cultivate Portfolio Risk (Code {cultivate_code_d_str}, Epsilon ${value_epsilon_d_float:,.2f}, Period {backtest_period_d_str}): "
                                  f"Weighted Avg Beta vs SPY: {weighted_beta_d:.4f}, Weighted Avg Correlation to SPY: {weighted_corr_d:.4f}.")
                if not is_called_by_ai:
                    print("\n**Cultivate Portfolio Risk Assessment Results (Code D)**")
                    print(f"  Cultivate Code: {cultivate_code_d_str}, Epsilon Value: ${value_epsilon_d_float:,.2f}, Fractional Shares: {'Yes' if frac_s_d_bool else 'No'}")
                    print(f"  Backtest Period: {backtest_period_d_str}")
                    print(f"  Weighted Average Beta vs SPY: {weighted_beta_d:.4f}")
                    print(f"  Weighted Average Correlation to SPY: {weighted_corr_d:.4f}")
            else:
                summary_for_ai = f"Could not calculate Beta/Correlation for Cultivate portfolio (Code {cultivate_code_d_str}, Period {backtest_period_d_str})."
                if not is_called_by_ai: print(summary_for_ai)
        except ValueError as ve_d: 
            summary_for_ai = f"Error (Assess D): Invalid numerical input for Epsilon value. {ve_d}"
            if not is_called_by_ai: print(summary_for_ai)
        except Exception as e_assess_d:
            summary_for_ai = f"An unexpected error occurred in Assess Code D: {e_assess_d}"
            if not is_called_by_ai: print(summary_for_ai); traceback.print_exc()
        return summary_for_ai

    # --- NEW: Code E: Portfolio Backtesting ---
    elif assess_code_input == 'E':
        if not is_called_by_ai: print("--- Assess Code E: Portfolio Backtesting ---")
        try:
            if len(args) < 4:
                print("CLI Usage: /assess E <portfolio_code> <start_date_YYYY-MM-DD> <end_date_YYYY-MM-DD>")
                return
            
            portfolio_code_e = args[1]
            start_date_e = args[2]
            end_date_e = args[3]

            # Validate dates
            try:
                pd.to_datetime(start_date_e)
                pd.to_datetime(end_date_e)
            except ValueError:
                print("❌ Error: Invalid date format. Please use YYYY-MM-DD.")
                return

            # Run the backtest using the new helper function
            await run_portfolio_backtest(portfolio_code_e, start_date_e, end_date_e)

        except Exception as e_assess_e:
            summary_for_ai = f"An unexpected error occurred in Assess Code E: {e_assess_e}"
            if not is_called_by_ai: print(summary_for_ai); traceback.print_exc()
        return  # This command is CLI-only for now

    else: 
        msg = f"Unknown or unsupported Assess Code: '{assess_code_input}'. Use A, B, C, D, or E."
        if not is_called_by_ai: print(msg)
        return msg