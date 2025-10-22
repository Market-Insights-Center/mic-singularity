# --- Imports for compare_command ---
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import humanize
import math
from scipy.stats import percentileofscore

# --- Imports from other command modules ---
# Assuming they are in the same directory and the path is set correctly
from invest_command import calculate_ema_invest
from sentiment_command import handle_sentiment_command
from fundamentals_command import handle_fundamentals_command # Assuming this will also be migrated
from mlforecast_command import handle_mlforecast_command # Assuming this will also be migrated

# --- Global Variables & Constants (copied for self-containment) ---
YFINANCE_API_SEMAPHORE = asyncio.Semaphore(8) # Concurrency limiter

# --- Helper Functions (copied for self-containment) ---

def safe_get(data_dict, key, default=None):
    value = data_dict.get(key, default)
    if value is None or value == 'None':
        return default
    return value

async def get_yfinance_info_robustly(ticker: str) -> Optional[Dict[str, Any]]:
    async with YFINANCE_API_SEMAPHORE:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                stock_info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)
                if stock_info and not stock_info.get('regularMarketPrice'):
                    raise ValueError(f"Incomplete data for {ticker}")
                return stock_info
            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep((attempt + 1) * 2)
                else:
                    return None
    return None

async def get_powerscore_explanation(ticker: str, component_scores: dict, gemini_model: Any, GEMINI_API_LOCK: asyncio.Lock) -> str:
    if not gemini_model:
        return "AI model is not available to generate an explanation."
    prompt = f"""
    Act as a financial analyst. Based on the following component scores for {ticker}, provide a concise 3-4 sentence summary of the stock's profile.
    Highlight its strongest and weakest areas based on these scores. Do not mention the raw or prime scores themselves, but interpret what they mean.
    Component Scores:
    - Market Invest Score (Prime): {component_scores.get('R_prime', 'N/A')}
    - Beta/Correlation Score (Prime): {component_scores.get('AB_prime', 'N/A')}
    - Volatility Rank Score (Prime): {component_scores.get('AA_prime', 'N/A')}
    - Fundamental Score (Prime): {component_scores.get('F_prime', 'N/A')}
    - QuickScore (Technicals, Prime): {component_scores.get('Q_prime', 'N/A')}
    - Sentiment Score (Prime): {component_scores.get('S_prime', 'N/A')}
    - ML Forecast Score (Prime): {component_scores.get('M_prime', 'N/A')}
    """
    try:
        async with GEMINI_API_LOCK:
            response = await asyncio.to_thread(
                gemini_model.generate_content, prompt
            )
        return response.text.strip()
    except Exception as e:
        return f"An error occurred while generating the AI summary: {e}"

async def get_single_stock_beta_corr(ticker: str, period: str) -> tuple[Optional[float], Optional[float]]:
    # This is a simplified version for PowerScore, not needing the full Assess B/C logic
    try:
        data = await asyncio.to_thread(yf.download, tickers=[ticker, 'SPY'], period=period, progress=False)
        if data.empty or data['Adj Close'].isnull().all().any(): return None, None
        returns = data['Adj Close'].pct_change().dropna()
        if len(returns) < 20: return None, None
        
        covariance = returns.cov().iloc[0, 1]
        market_variance = returns['SPY'].var()
        beta = covariance / market_variance if market_variance != 0 else 0.0
        
        correlation = returns.corr().iloc[0, 1]
        return beta, correlation
    except Exception:
        return None, None

async def get_market_invest_score_for_powerscore() -> Optional[float]:
    try:
        vix_hist = await asyncio.to_thread(yf.Ticker('^VIX').history, period="5d")
        vix_price = vix_hist['Close'].iloc[-1]
        vix_likelihood = np.clip(0.01384083 * (vix_price ** 2), 0, 100)

        spy_hist = await asyncio.to_thread(yf.Ticker('SPY').history, period="5y", interval="1mo")
        spy_hist['EMA_8'] = spy_hist['Close'].ewm(span=8, adjust=False).mean()
        spy_hist['EMA_55'] = spy_hist['Close'].ewm(span=55, adjust=False).mean()
        ema_8, ema_55 = spy_hist['EMA_8'].iloc[-1], spy_hist['EMA_55'].iloc[-1]
        x_value = (((ema_8 - ema_55) / ema_55) + 0.5) * 100
        ema_likelihood = np.clip(100 * np.exp(-((45.622216 * x_value / 2750) ** 4)), 0, 100)
        
        if ema_likelihood == 0: return 0.0
        ratio = vix_likelihood / ema_likelihood
        return float(np.clip(50.0 - (ratio - 1.0) * 100.0, 0, 100))
    except Exception:
        return None

async def calculate_volatility_metrics(ticker: str, period: str) -> tuple[Optional[float], Optional[float]]:
    try:
        hist_data = await asyncio.to_thread(yf.Ticker(ticker).history, period=period)
        if hist_data.empty or len(hist_data) <= 30: return None, None
        hist_data['daily_return'] = hist_data['Close'].pct_change()
        rolling_hv = hist_data['daily_return'].rolling(window=30).std() * (252**0.5)
        hv_series = rolling_hv.dropna()
        if len(hv_series) > 1:
            vol_rank = percentileofscore(hv_series, hv_series.iloc[-1])
        else:
            vol_rank = None
        # Simplified IV fetch for this context
        return None, vol_rank
    except Exception:
        return None, None

async def handle_powerscore_command(ai_params: dict, is_called_by_ai: bool, gemini_model: Any, GEMINI_API_LOCK: asyncio.Lock):
    ticker = ai_params.get("ticker")
    sensitivity = ai_params.get("sensitivity")
    beta_corr_period = {1: '10y', 2: '5y', 3: '1y'}[sensitivity]
    
    tasks = {
        "R": get_market_invest_score_for_powerscore(),
        "ABB_ABC": get_single_stock_beta_corr(ticker, beta_corr_period),
        "AA": calculate_volatility_metrics(ticker, beta_corr_period),
        "F": handle_fundamentals_command(ai_params={'ticker': ticker}, is_called_by_ai=True),
        "Q": calculate_ema_invest(ticker, sensitivity, is_called_by_ai=True),
        "S": handle_sentiment_command(ai_params={'ticker': ticker}, is_called_by_ai=True),
        "M": handle_mlforecast_command(ai_params={'ticker': ticker}, is_called_by_ai=True)
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    raw_values = dict(zip(tasks.keys(), results))
    
    # Simplified processing logic for brevity
    ps_components = {
        'R': raw_values.get('R'),
        'ABB': raw_values.get('ABB_ABC')[0] if isinstance(raw_values.get('ABB_ABC'), tuple) else None,
        'AA': raw_values.get('AA')[1] if isinstance(raw_values.get('AA'), tuple) else None,
        'F': raw_values.get('F').get('fundamental_score') if isinstance(raw_values.get('F'), dict) else None,
        'Q': raw_values.get('Q')[1] if isinstance(raw_values.get('Q'), tuple) else None,
        'S': raw_values.get('S').get('sentiment_score') if isinstance(raw_values.get('S'), dict) else None,
        'M': float(raw_values.get('M')[0].get("Est. % Change", "0%").replace('%','')) if isinstance(raw_values.get('M'), list) and raw_values.get('M') else None
    }
    
    weights = {'R': 0.20, 'AB': 0.10, 'AA': 0.10, 'F': 0.05, 'Q': 0.25, 'S': 0.15, 'M': 0.15} # Sens 2
    final_ps = sum(v * weights[k] for k, v in ps_components.items() if v is not None and k in weights and weights[k] > 0)
    total_weight = sum(weights[k] for k, v in ps_components.items() if v is not None and k in weights and weights[k] > 0)
    
    return {"powerscore": (final_ps / total_weight) if total_weight > 0 else 0.0}

def plot_comparison_performance_graph(tickers: List[str]):
    if not tickers: return
    all_tickers_to_fetch = list(set(tickers + ['SPY']))
    print(f"-> Fetching 1-year performance data for: {', '.join(all_tickers_to_fetch)}")
    try:
        data = yf.download(all_tickers_to_fetch, period="1y", progress=False, auto_adjust=True)['Close']
        if data.empty: return
        data.dropna(axis=1, how='all', inplace=True)
        normalized_data = (data / data.iloc[0]) * 100
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))
        for ticker in normalized_data.columns:
            linestyle = '--' if ticker == 'SPY' else '-'
            ax.plot(normalized_data.index, normalized_data[ticker], label=ticker, linestyle=linestyle)
        ax.set_title('1-Year Normalized Stock Performance vs. SPY', color='white')
        ax.set_ylabel('Normalized Performance (Starts at 100)', color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
        filename = f"comparison_performance_{'_'.join(tickers)}_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black')
        plt.close(fig)
        print(f"📂 Performance comparison graph saved: {filename}")
    except Exception as e:
        print(f"❌ Error generating performance graph: {e}")

# --- Main Command Handler ---

async def handle_compare_command(args: List[str], is_called_by_ai: bool, gemini_model: Any, GEMINI_API_LOCK: asyncio.Lock):
    if is_called_by_ai:
        return "The /compare command is currently available for CLI use only."

    print("\n--- Head-to-Head Stock Comparison ---")
    tickers = [arg.upper() for arg in args]
    if not tickers:
        print("Usage: /compare <TICKER1> <TICKER2> ...")
        return

    print(f"-> Comparing: {', '.join(tickers)}")
    results_for_table = []
    
    for ticker in tickers:
        print(f"   -> Analyzing {ticker}...")
        try:
            funda_task = handle_fundamentals_command(ai_params={'ticker': ticker}, is_called_by_ai=True)
            ps_task = handle_powerscore_command(ai_params={'ticker': ticker, 'sensitivity': 2}, is_called_by_ai=True, gemini_model=gemini_model, GEMINI_API_LOCK=GEMINI_API_LOCK)
            invest_task = calculate_ema_invest(ticker, 2, is_called_by_ai=True)
            funda_results, ps_results, invest_results = await asyncio.gather(funda_task, ps_task, invest_task)
            
            metrics = {'Ticker': ticker, 'PowerScore': 'N/A', 'Invest Score': 'N/A', 'P/E Ratio': 'N/A', 'Rev. Growth %': 'N/A', 'D/E Ratio': 'N/A', 'Profit Margin %': 'N/A'}
            if isinstance(ps_results, dict) and 'powerscore' in ps_results: metrics['PowerScore'] = f"{ps_results['powerscore']:.2f}"
            if invest_results and invest_results[1] is not None: metrics['Invest Score'] = f"{invest_results[1]:.2f}%"
            if isinstance(funda_results, dict) and 'error' not in funda_results:
                pe, rev_g, de, margin = funda_results.get('pe_ratio'), funda_results.get('revenue_growth'), funda_results.get('debt_to_equity'), funda_results.get('profit_margin')
                if pe is not None: metrics['P/E Ratio'] = f"{pe:.2f}"
                if rev_g is not None: metrics['Rev. Growth %'] = f"{rev_g * 100:.2f}"
                if de is not None: metrics['D/E Ratio'] = f"{de:.2f}"
                if margin is not None: metrics['Profit Margin %'] = f"{margin * 100:.2f}"
            results_for_table.append(metrics)
        except Exception as e:
            print(f"   -> ❌ An error occurred while analyzing {ticker}: {e}")
            results_for_table.append({'Ticker': ticker, 'PowerScore': 'Error', 'Invest Score': 'Error', 'P/E Ratio': 'Error', 'Rev. Growth %': 'Error', 'D/E Ratio': 'Error', 'Profit Margin %': 'Error'})
    
    results_for_table.sort(key=lambda x: float(x.get('PowerScore', -1.0)), reverse=True)
    print("\n--- Comparison Summary (Sorted by PowerScore) ---")
    if results_for_table:
        print(tabulate(results_for_table, headers="keys", tablefmt="pretty"))
    
    await asyncio.to_thread(plot_comparison_performance_graph, tickers)