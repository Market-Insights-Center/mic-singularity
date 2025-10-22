# --- Imports for cultivate_command ---
import asyncio
import csv
import math
import os
import traceback
from datetime import datetime, timedelta
from io import StringIO
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tabulate import tabulate
from tradingview_screener import Column, Query

# --- Imports from other command modules ---
from invest_command import (calculate_ema_invest, generate_portfolio_pie_chart,
                            get_allocation_score)

# --- Constants (copied for self-containment) ---
MARKET_HEDGING_TICKERS = ['SPY', 'DIA', 'QQQ']
RESOURCE_HEDGING_TICKERS = ['GLD', 'SLV']
HEDGING_TICKERS = MARKET_HEDGING_TICKERS + RESOURCE_HEDGING_TICKERS
CULTIVATE_INITIAL_METRICS_FILE = 'cultivate_initial_metrics.csv'
CULTIVATE_T1_FILE = 'cultivate_ticker_list_one.csv'
CULTIVATE_T_MINUS_1_FILE = 'cultivate_ticker_list_negative_one.csv'
CULTIVATE_TF_FINAL_FILE = 'cultivate_ticker_list_final.csv'
CULTIVATE_COMBINED_DATA_FILE_PREFIX = 'cultivate_combined_'

# --- Helper Functions (copied or moved for self-containment) ---

def safe_score(value: Any) -> float:
    try:
        if pd.isna(value) or value is None: return 0.0
        if isinstance(value, str): value = value.replace('%', '').replace('$', '').strip()
        return float(value)
    except (ValueError, TypeError): return 0.0

async def get_yf_download_robustly(tickers: list, **kwargs) -> pd.DataFrame:
    max_retries = 3
    for attempt in range(max_retries):
        try:
            data = await asyncio.to_thread(yf.download, tickers=tickers, progress=False, **kwargs)
            if data.empty:
                 raise IOError(f"yf.download returned empty DataFrame for {tickers}")
            return data
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep((attempt + 1) * 2)
            else:
                return pd.DataFrame()
    return pd.DataFrame()

async def get_yf_data_singularity(tickers: List[str], period: str = "10y", interval: str = "1d", is_called_by_ai: bool = False) -> pd.DataFrame:
    if not tickers: return pd.DataFrame()
    data = await get_yf_download_robustly(tickers=list(set(tickers)), period=period, interval=interval, auto_adjust=False, group_by='ticker', timeout=30)
    if data.empty: return pd.DataFrame()
    all_series = []
    if isinstance(data.columns, pd.MultiIndex):
        for ticker_name in list(set(tickers)):
            if (ticker_name, 'Close') in data.columns:
                series = pd.to_numeric(data[(ticker_name, 'Close')], errors='coerce').dropna()
                if not series.empty: series.name = ticker_name; all_series.append(series)
    elif 'Close' in data.columns:
        series = pd.to_numeric(data['Close'], errors='coerce').dropna()
        if not series.empty: series.name = list(set(tickers))[0]; all_series.append(series)
    if not all_series: return pd.DataFrame()
    df_out = pd.concat(all_series, axis=1)
    df_out.index = pd.to_datetime(df_out.index)
    return df_out.dropna(axis=0, how='all').dropna(axis=1, how='all')

def get_sp500_symbols_singularity(is_called_by_ai: bool = False) -> List[str]:
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        df = pd.read_html(StringIO(response.text))[0]
        if 'Symbol' not in df.columns:
            if not is_called_by_ai:
                print("     ... Error: Could not find 'Symbol' column in the Wikipedia table.")
            return []
        symbols = [str(s).replace('.', '-') for s in df['Symbol'].tolist() if isinstance(s, str)]
        return sorted(list(set(s for s in symbols if s)))
    except Exception as e:
        if not is_called_by_ai:
            print(f"     ... Error: Failed to fetch S&P 500 list. Reason: {type(e).__name__}")
        return []
    
def screen_stocks_singularity(is_called_by_ai: bool = False) -> List[str]:
    try:
        query = Query().select('name').where(Column('market_cap_basic') >= 10_000_000_000, Column('average_volume_90d_calc') >= 500_000).limit(500)
        _, df = query.get_scanner_data(timeout=60)
        if df is not None and 'name' in df.columns:
            tickers = [str(t).split(':')[-1].replace('.', '-') for t in df['name'].tolist() if pd.notna(t)]
            return sorted(list(set(tickers)))
        return []
    except Exception:
        return []

def calculate_cultivate_formulas_singularity(allocation_score: float, is_called_by_ai: bool = False) -> Optional[Dict[str, Any]]:
    sigma = safe_score(allocation_score)
    sigma_safe = max(0.0001, min(99.9999, sigma))
    sigma_ratio_term = sigma_safe / (100.0 - sigma_safe) if (100.0 - sigma_safe) > 1e-9 else np.inf
    results = {}
    try:
        results['omega'] = max(0.0, min(100.0, 100.0 - ((49.0/60.0 * sigma_safe * np.exp(-(np.log(7.0/6.0) / 50.0) * sigma_safe) + 40.0))))
        results['lambda'] = 100.0 - results['omega']
        results['lambda_hedge'] = max(0.0, min(100.0, 100 - ((1/1000) * sigma_safe**2 + (7/20) * sigma_safe + 40)))
        results['alpha'] = 100.0 - results['lambda_hedge']
        results['beta_alloc'] = results['lambda_hedge']
        mu_center = -1/4 + (11/4) * (1 - np.exp(-np.log(11.0/4.0) * sigma_ratio_term))
        results['mu_range'] = (mu_center - 2/3, mu_center + 2/3)
        rho_center = 3/4 - np.exp(-np.log(4.0) * sigma_ratio_term)
        results['rho_range'] = (rho_center - 1/8, rho_center + 1/8)
        omega_target_center = -1/2 + (7/2) * (1 - np.exp(-np.log(7.0/3.0) * sigma_ratio_term))
        results['omega_target_range'] = (omega_target_center - 1/2, omega_target_center + 1/2)
        results['delta'] = max(0.25, min(5.0, 1/4 + (11/4) * (1 - np.exp(-np.log(11.0/8.0) * sigma_ratio_term))))
        results['eta'] = max(0.0, min(100.0, -sigma_safe**2 / 500.0 - 3.0*sigma_safe / 10.0 + 60.0))
        results['kappa'] = 100.0 - results['eta']
        return results
    except Exception:
        return None

async def calculate_metrics_singularity(tickers_list: List[str], spy_data_10y: pd.DataFrame, is_called_by_ai: bool = False) -> Dict[str, Dict[str, float]]:
    metrics = {}
    if spy_data_10y.empty or not tickers_list:
        return {}
    chunk_size = 25
    total_chunks = (len(tickers_list) + chunk_size - 1) // chunk_size
    for i in range(0, len(tickers_list), chunk_size):
        chunk = tickers_list[i:i + chunk_size]
        current_chunk_num = (i // chunk_size) + 1
        if not is_called_by_ai:
            print(f"     -> Processing metrics chunk {current_chunk_num}/{total_chunks} ({len(chunk)} tickers)...")
        chunk_hist_data = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                download_task = get_yf_data_singularity(chunk + ['SPY'], period="1y", is_called_by_ai=True)
                chunk_hist_data = await asyncio.wait_for(download_task, timeout=45.0)
                if not chunk_hist_data.empty:
                    break
                else:
                    raise ValueError("Download returned an empty DataFrame")
            except (asyncio.TimeoutError, Exception) as e:
                if not is_called_by_ai:
                    print(f"        ... chunk {current_chunk_num} failed on attempt {attempt + 1}/{max_retries}. Error: {type(e).__name__}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                else:
                    if not is_called_by_ai:
                        print(f"        ... SKIPPING chunk {current_chunk_num} after all attempts failed.")
        if chunk_hist_data is None or chunk_hist_data.empty:
            continue
        daily_returns_chunk = chunk_hist_data.pct_change(fill_method=None).dropna()
        for ticker in chunk:
            if ticker in daily_returns_chunk.columns and 'SPY' in daily_returns_chunk.columns:
                ticker_returns = daily_returns_chunk[ticker]
                spy_returns_aligned = daily_returns_chunk['SPY']
                aligned_data = pd.concat([ticker_returns, spy_returns_aligned], axis=1).dropna()
                if len(aligned_data) > 20:
                    try:
                        cov_matrix = np.cov(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
                        spy_variance = cov_matrix[1, 1]
                        if spy_variance != 0:
                            beta = cov_matrix[0, 1] / spy_variance
                            correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
                            leverage = (ticker_returns / spy_returns_aligned).replace([np.inf, -np.inf], np.nan).mean()
                            metrics[ticker] = {'beta_1y': beta, 'correlation_1y': correlation, 'avg_leverage_general_1y': leverage}
                    except Exception:
                        continue
    return metrics

def save_initial_metrics_singularity(metrics: Dict[str, Dict[str, float]], tickers_processed: List[str], is_called_by_ai: bool = False):
    if not metrics: return
    df = pd.DataFrame.from_dict(metrics, orient='index').reset_index().rename(columns={'index': 'Ticker'})
    df.to_csv(CULTIVATE_INITIAL_METRICS_FILE, index=False)

async def select_tickers_singularity(tickers_to_filter: list, metrics: dict, invest_scores_all: dict, formula_results: dict, portfolio_value: float, is_called_by_ai: bool = False) -> tuple[list, str | None, dict, int]:
    mu_range, rho_range, omega_range = formula_results['mu_range'], formula_results['rho_range'], formula_results['omega_target_range']
    num_target_tickers = max(1, math.ceil(0.3 * math.sqrt(portfolio_value)) - len(HEDGING_TICKERS))
    T1, T_minus_1 = [], []
    for ticker in tickers_to_filter:
        if ticker in metrics and invest_scores_all.get(ticker, {}).get('score', 0) > 0:
            m = metrics[ticker]
            in_mu = mu_range[0] <= m.get('beta_1y', 0) <= mu_range[1]
            in_rho = rho_range[0] <= m.get('correlation_1y', 0) <= rho_range[1]
            in_omega = omega_range[0] <= m.get('avg_leverage_general_1y', 0) <= omega_range[1]
            candidate = {'ticker': ticker, 'score': invest_scores_all[ticker]['score']}
            if in_mu and in_rho and in_omega:
                T1.append(candidate)
            else:
                T_minus_1.append(candidate)
    T1.sort(key=lambda x: x['score'], reverse=True)
    T_minus_1.sort(key=lambda x: x['score'], reverse=True)
    final_selection = T1[:num_target_tickers]
    needed = num_target_tickers - len(final_selection)
    if needed > 0:
        final_selection.extend(T_minus_1[:needed])
    final_tickers = [item['ticker'] for item in final_selection]
    warning = f"Warning: Target tickers ({num_target_tickers}) not reached. Selected {len(final_tickers)}." if len(final_tickers) < num_target_tickers else None
    return final_tickers, warning, invest_scores_all, num_target_tickers

def build_and_process_portfolios_singularity(common_stock_tickers: list, formula_results: dict, total_portfolio_value: float, frac_shares: bool, invest_scores_all: dict, is_called_by_ai: bool = False) -> tuple:
    epsilon, omega_pct, lambda_pct = total_portfolio_value, formula_results['omega'], formula_results['lambda']
    alpha_pct, hedge_pct = formula_results['alpha'], formula_results['beta_alloc']
    kappa_pct, eta_pct = formula_results['kappa'], formula_results['eta']
    delta = formula_results['delta']
    lambda_value = epsilon * (lambda_pct / 100.0)
    combined_data_for_save = []
    def get_allocations(tickers: list, total_value_segment: float, group_name: str) -> list:
        sub_portfolio, total_amp_score = [], 0
        for t in tickers:
            if t in invest_scores_all and invest_scores_all[t].get('live_price', 0) > 0:
                score = invest_scores_all[t]['raw_invest_score']
                amp_score = max(0, (score * delta) - (delta - 1) * 50)
                sub_portfolio.append({'ticker': t, 'amplified_score': amp_score, 'group': group_name, **invest_scores_all[t]})
                total_amp_score += amp_score
        for item in sub_portfolio:
            weight = (item['amplified_score'] / total_amp_score) if total_amp_score > 0 else 0
            item['target_value'] = total_value_segment * weight
            item['combined_percent_allocation_of_lambda'] = (item['target_value'] / lambda_value) * 100 if lambda_value > 0 else 0
        return sub_portfolio
    common_alloc = get_allocations(common_stock_tickers, lambda_value * (alpha_pct / 100.0), "COMMON")
    market_hedge_alloc = get_allocations(MARKET_HEDGING_TICKERS, lambda_value * (hedge_pct / 100.0) * (kappa_pct / 100.0), "MARKET_HEDGE")
    resource_hedge_alloc = get_allocations(RESOURCE_HEDGING_TICKERS, lambda_value * (hedge_pct / 100.0) * (eta_pct / 100.0), "RESOURCE_HEDGE")
    all_allocations = common_alloc + market_hedge_alloc + resource_hedge_alloc
    combined_data_for_save.extend(all_allocations)
    tailored_holdings = []
    total_spent = 0
    for item in all_allocations:
        price = item.get('live_price', 0)
        if price <= 0: continue
        shares = item['target_value'] / price
        final_shares = round(shares, 2) if frac_shares else math.floor(shares)
        spent = final_shares * price
        if spent > 0:
            item['actual_percent_allocation_total_epsilon'] = (spent / epsilon) * 100
            tailored_holdings.append({'ticker': item['ticker'], 'shares': final_shares, 'actual_money_allocation': spent, **item})
            total_spent += spent
    final_cash = epsilon - total_spent
    return combined_data_for_save, tailored_holdings, final_cash

async def save_cultivate_data_internal_singularity(combined_portfolio_data_to_save: List[Dict], date_str_to_save: str, cultivate_code_for_save: str, epsilon_for_save: float, is_called_by_ai: bool = False):
    if not combined_portfolio_data_to_save: return
    save_file = f"{CULTIVATE_COMBINED_DATA_FILE_PREFIX}{cultivate_code_for_save.upper()}_{int(epsilon_for_save)}.csv"
    file_exists = os.path.isfile(save_file)
    with open(save_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['DATE', 'TICKER', 'PORTFOLIO_GROUP', 'PRICE', 'RAW_INVEST_SCORE', 'COMBINED_ALLOCATION_PERCENT_OF_LAMBDA'])
        for item in combined_portfolio_data_to_save:
             writer.writerow([date_str_to_save, item['ticker'], 'N/A', f"{item['live_price']:.2f}", f"{item['raw_invest_score']:.2f}%", f"{item.get('combined_percent_allocation_of_lambda', 0):.2f}%"])

async def run_cultivate_analysis_singularity(
    portfolio_value: float,
    frac_shares: bool,
    cultivate_code_str: str,
    is_called_by_ai: bool = False,
    is_saving_run: bool = False
) -> tuple[list[dict], list[dict], float, str, float, bool, str | None]:
    suppress_sub_prints = is_called_by_ai or is_saving_run
    if not suppress_sub_prints:
        print(f"\n--- Cultivate Analysis (Code: {cultivate_code_str.upper()}, Value: ${portfolio_value:,.0f}) ---")
    epsilon_val = safe_score(portfolio_value)
    if epsilon_val <= 0:
        return [], [], 0.0, cultivate_code_str, epsilon_val, frac_shares, "Error: Invalid portfolio value"
    allocation_score_cult, _, _ = get_allocation_score(is_called_by_ai=suppress_sub_prints)
    if allocation_score_cult is None:
        return [], [], 0.0, cultivate_code_str, epsilon_val, frac_shares, "Error: Failed to get Allocation Score"
    formula_results_cult = calculate_cultivate_formulas_singularity(allocation_score_cult, is_called_by_ai=suppress_sub_prints)
    if formula_results_cult is None:
        return [], [], 0.0, cultivate_code_str, epsilon_val, frac_shares, "Error: Formula calculation failed"
    if not suppress_sub_prints:
        step1_desc = "Screening for large-cap stocks (this may take a moment)" if cultivate_code_str.upper() == 'A' else "Fetching S&P 500 list"
        print(f"  -> Step 1/5: {step1_desc}...")
    tickers_to_process_cult = []
    if cultivate_code_str.upper() == 'A':
        tickers_to_process_cult = await asyncio.to_thread(screen_stocks_singularity, is_called_by_ai=suppress_sub_prints)
    elif cultivate_code_str.upper() == 'B':
        tickers_to_process_cult = await asyncio.to_thread(get_sp500_symbols_singularity, is_called_by_ai=suppress_sub_prints)
    if not tickers_to_process_cult:
        return [], [], 0.0, cultivate_code_str, epsilon_val, frac_shares, f"Error: Step 1 (Code {cultivate_code_str.upper()}) failed to produce a ticker list."
    if not suppress_sub_prints:
        print(f"     ...found {len(tickers_to_process_cult)} tickers.")
    if not suppress_sub_prints:
        print(f"  -> Step 2/5: Calculating financial metrics (beta, correlation)...")
    spy_hist_data_metrics_cult = await get_yf_data_singularity(['SPY'], period="10y", interval="1d", is_called_by_ai=suppress_sub_prints)
    metrics_dict_cult = {}
    if not spy_hist_data_metrics_cult.empty and tickers_to_process_cult:
        metrics_dict_cult = await calculate_metrics_singularity(tickers_to_process_cult, spy_hist_data_metrics_cult, is_called_by_ai=suppress_sub_prints)
        if not is_saving_run:
            await asyncio.to_thread(save_initial_metrics_singularity, metrics_dict_cult, tickers_to_process_cult, is_called_by_ai=suppress_sub_prints)
    if not suppress_sub_prints:
        print(f"  -> Step 3/5: Calculating Invest Scores for all tickers...")
    invest_scores_all_cult = {}
    all_tickers_for_scoring_cult = list(set((tickers_to_process_cult or []) + HEDGING_TICKERS))
    if all_tickers_for_scoring_cult:
        chunk_size = 25
        total_chunks = (len(all_tickers_for_scoring_cult) + chunk_size - 1) // chunk_size
        for i in range(0, len(all_tickers_for_scoring_cult), chunk_size):
            chunk = all_tickers_for_scoring_cult[i:i + chunk_size]
            current_chunk_num = (i // chunk_size) + 1
            if not suppress_sub_prints:
                print(f"     -> Processing scores chunk {current_chunk_num}/{total_chunks} ({len(chunk)} tickers)...")
            score_tasks = [calculate_ema_invest(ticker, 2, is_called_by_ai=True) for ticker in chunk]
            score_results_tuples = await asyncio.gather(*score_tasks, return_exceptions=True)
            for j, ticker_sc in enumerate(chunk):
                res_sc_tuple = score_results_tuples[j]
                if isinstance(res_sc_tuple, Exception) or not res_sc_tuple or res_sc_tuple[1] is None:
                    invest_scores_all_cult[ticker_sc] = {'score': -float('inf'), 'live_price': 0.0, 'raw_invest_score': -float('inf')}
                else:
                    live_price_sc, score_val_sc = res_sc_tuple
                    invest_scores_all_cult[ticker_sc] = {'score': safe_score(score_val_sc), 'live_price': safe_score(live_price_sc), 'raw_invest_score': safe_score(score_val_sc)}
    if not suppress_sub_prints:
        print(f"     ...scores calculated.")
    if not suppress_sub_prints:
        print(f"  -> Step 4/5: Selecting final tickers based on formula ranges...")
    final_common_stock_tickers_cult, warning_msg, _, num_target = await select_tickers_singularity(
        tickers_to_filter=tickers_to_process_cult, metrics=metrics_dict_cult,
        invest_scores_all=invest_scores_all_cult, formula_results=formula_results_cult,
        portfolio_value=epsilon_val, is_called_by_ai=suppress_sub_prints)
    if not suppress_sub_prints:
        print(f"     ...selected {len(final_common_stock_tickers_cult)} of {num_target} target tickers.")
        if warning_msg:
            print(f"     {warning_msg}")
    if not suppress_sub_prints:
        print(f"  -> Step 5/5: Building final portfolio allocations...")
    (combined_data_for_save, tailored_holdings_final, final_cash_value_cult
     ) = build_and_process_portfolios_singularity(
        final_common_stock_tickers_cult, formula_results_cult, epsilon_val,
        frac_shares, invest_scores_all_cult, is_called_by_ai=suppress_sub_prints)
    if not suppress_sub_prints:
        print("     ...analysis complete.")
        if tailored_holdings_final:
            print(tabulate([{'Ticker': i['ticker'], 'Shares': f"{i['shares']:.2f}" if frac_shares else int(i['shares']), '$ Allocation': f"${i['actual_money_allocation']:,.2f}"} for i in tailored_holdings_final], headers="keys", tablefmt="pretty"))
            print(f"Final Cash: ${final_cash_value_cult:,.2f}")
    return (combined_data_for_save, tailored_holdings_final, final_cash_value_cult,
            cultivate_code_str, epsilon_val, frac_shares, None)

# --- Main Command Handler ---
async def handle_cultivate_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """
    Handles the /cultivate command for CLI and AI.
    """
    # --- FIX: Restructure to ensure all failure paths return a string for the AI ---
    try:
        cult_code, portfolio_val, frac_s_bool, action_type, date_to_save_val = None, None, None, "run_analysis", None

        if ai_params:
            cult_code = ai_params.get("cultivate_code", "").upper()
            if cult_code not in ['A', 'B']:
                raise ValueError(f"Invalid Cultivate code '{cult_code}'. Must be 'A' or 'B'.")
            raw_val = ai_params.get("portfolio_value")
            if raw_val is None:
                raise ValueError("'portfolio_value' is required.")
            portfolio_val = float(raw_val)
            if portfolio_val <= 0:
                raise ValueError("'portfolio_value' must be positive.")
            frac_s_bool = bool(ai_params.get("use_fractional_shares", False))
            action_type = ai_params.get("action", "run_analysis").lower()
            if action_type == "save_data":
                date_str_raw = ai_params.get("date_to_save")
                if not date_str_raw:
                    raise ValueError("'date_to_save' (MM/DD/YYYY) is required for save action.")
                datetime.strptime(date_str_raw, '%m/%d/%Y')
                date_to_save_val = date_str_raw
        else: # CLI Path
            if len(args) < 3:
                print("CLI Usage: /cultivate <Code A/B> <PortfolioValue> <FracShares yes/no> [save_code 3725]")
                return
            cult_code = args[0].upper()
            if cult_code not in ['A', 'B']:
                print("CLI Error: Cultivate Code must be 'A' or 'B'.")
                return
            portfolio_val = float(args[1])
            if portfolio_val <= 0:
                print("CLI Error: Portfolio value must be positive.")
                return
            frac_s_str = args[2].lower()
            if frac_s_str not in ['yes', 'no']:
                print("CLI Error: Fractional shares must be 'yes' or 'no'.")
                return
            frac_s_bool = frac_s_str == 'yes'
            if len(args) > 3 and args[3] == "3725":
                action_type = "save_data"
                date_str_cli = input(f"CLI: Enter date (MM/DD/YYYY) to save Cultivate (Code {cult_code}, Val ${portfolio_val:,.0f}): ")
                try:
                    datetime.strptime(date_str_cli, '%m/%d/%Y')
                    date_to_save_val = date_str_cli
                except ValueError:
                    print("CLI Error: Invalid date format. Save cancelled.")
                    return

        is_for_saving_only = action_type == "save_data"
        
        (combined_data, tailored_entries, final_cash, code_used, eps_used, frac_s_used, err_msg
         ) = await run_cultivate_analysis_singularity(
            portfolio_value=portfolio_val, frac_shares=frac_s_bool,
            cultivate_code_str=cult_code,
            is_called_by_ai=is_called_by_ai,
            is_saving_run=is_for_saving_only
        )

        if err_msg:
            raise Exception(err_msg)

        if action_type == "save_data" and date_to_save_val:
            if combined_data:
                await save_cultivate_data_internal_singularity(
                    combined_portfolio_data_to_save=combined_data,
                    date_str_to_save=date_to_save_val,
                    cultivate_code_for_save=code_used,
                    epsilon_for_save=eps_used,
                    is_called_by_ai=is_called_by_ai
                )
                summary = f"Cultivate analysis (Code {code_used}, Val ${eps_used:,.0f}) data generated and saved for {date_to_save_val}."
                if is_called_by_ai: return summary
                else: print(summary)
            else:
                summary = f"Cultivate 'save_data' action requested for Code {code_used}, but no data was generated to save."
                if is_called_by_ai: return summary
                else: print(summary)
        
        elif action_type == "run_analysis":
            if is_called_by_ai:
                return tailored_entries, final_cash
            # For CLI, detailed printout already happened in run_cultivate_analysis_singularity
        
        return # Success for CLI

    except Exception as e:
        error_message = f"Error in /cultivate command: {e}"
        if is_called_by_ai:
            return error_message
        else:
            print(error_message)
            traceback.print_exc()