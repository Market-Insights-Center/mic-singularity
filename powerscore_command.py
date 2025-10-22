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
from sentiment_command import handle_sentiment_command

# --- Global Variables & Constants ---
YFINANCE_API_SEMAPHORE = asyncio.Semaphore(8)
GEMINI_API_LOCK = asyncio.Lock()

# --- Initialize Gemini Model within this module ---
gemini_model = None
try:
    config = configparser.ConfigParser()
    config.read('config.ini')
    GEMINI_API_KEY = config.get('API_KEYS', 'GEMINI_API_KEY', fallback=None)
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        # --- FIX 1: Corrected model name ---
        gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
except Exception as e:
    print(f"Warning: Could not configure Gemini model in powerscore_command.py: {e}")

# --- Helper Functions (copied or moved for self-containment) ---

def safe_get(data_dict: Dict, key: str, default: Any = None) -> Any:
    value = data_dict.get(key, default)
    return default if value is None or value == 'None' else value

async def get_yfinance_info_robustly(ticker: str) -> Optional[Dict[str, Any]]:
    async with YFINANCE_API_SEMAPHORE:
        for attempt in range(3):
            try:
                stock_info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)
                if stock_info and stock_info.get('regularMarketPrice'):
                    return stock_info
            except Exception:
                if attempt < 2: await asyncio.sleep((attempt + 1) * 2)
        return None

async def get_yf_download_robustly(tickers: list, **kwargs) -> pd.DataFrame:
    """
    A robust wrapper for yf.download with retry logic and exponential backoff
    to handle transient network errors and API instability.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            await asyncio.sleep(random.uniform(0.3, 0.7)) # Stagger requests
            
            data = await asyncio.to_thread(
                yf.download,
                tickers=tickers,
                progress=False,
                **kwargs
            )

            if data.empty:
                 raise IOError(f"yf.download returned an empty DataFrame for tickers: {tickers}")

            return data # Success
            
        except Exception as e:
            if attempt < max_retries - 1:
                delay = (attempt + 1) * 3 # Backoff: 3s, 6s
                print(f"   -> WARNING: yf.download failed (Attempt {attempt+1}/{max_retries}). Retrying in {delay}s...")
                await asyncio.sleep(delay)
            else:
                print(f"   -> ❌ ERROR: All yfinance download attempts failed. Last error: {type(e).__name__}")
                return pd.DataFrame()
    return pd.DataFrame()

async def get_yf_data_singularity(tickers: List[str], period: str = "10y", interval: str = "1d", is_called_by_ai: bool = False) -> pd.DataFrame:
    """
    Downloads historical closing price data for multiple tickers using the robust wrapper.
    """
    if not tickers:
        return pd.DataFrame()

    data = await get_yf_download_robustly(
        tickers=list(set(tickers)), period=period, interval=interval,
        auto_adjust=False, group_by='ticker', timeout=30
    )

    if data.empty:
        return pd.DataFrame()

    all_series = []
    if isinstance(data.columns, pd.MultiIndex):
        for ticker_name in list(set(tickers)):
            if (ticker_name, 'Close') in data.columns:
                series = pd.to_numeric(data[(ticker_name, 'Close')], errors='coerce').dropna()
                if not series.empty:
                    series.name = ticker_name
                    all_series.append(series)
    elif 'Close' in data.columns: # Single ticker case
        series = pd.to_numeric(data['Close'], errors='coerce').dropna()
        if not series.empty:
            series.name = list(set(tickers))[0]
            all_series.append(series)

    if not all_series:
        return pd.DataFrame()

    df_out = pd.concat(all_series, axis=1)
    df_out.index = pd.to_datetime(df_out.index)
    return df_out.dropna(axis=0, how='all').dropna(axis=1, how='all')

async def calculate_portfolio_beta_correlation_singularity(
    portfolio_holdings: List[Dict[str, Any]], # List of {'ticker': str, 'value': float}
    total_portfolio_value: float,
    backtest_period: str, # e.g., "1y", "3mo"
    is_called_by_ai: bool = False
) -> Optional[tuple[float, float]]:
    """
    Calculates weighted average Beta and Correlation for a given portfolio against SPY.
    `portfolio_holdings`: List of dicts, each with 'ticker' and 'value' (actual dollar allocation).
    Cash should be one of the tickers with its value if present.
    """
    if not portfolio_holdings or total_portfolio_value <= 0:
        return None

    valid_holdings_for_calc = [h for h in portfolio_holdings if isinstance(h.get('value'), (int, float)) and h['value'] > 1e-9]
    if not valid_holdings_for_calc:
        is_only_cash = all(h['ticker'].upper() == 'CASH' for h in portfolio_holdings) or \
                       (not any(h['ticker'].upper() != 'CASH' for h in valid_holdings_for_calc) and \
                        any(h['ticker'].upper() == 'CASH' for h in valid_holdings_for_calc))
        if is_only_cash:
            return 0.0, 0.0
        return None

    portfolio_stock_tickers_assess = [h['ticker'] for h in valid_holdings_for_calc if h['ticker'].upper() != 'CASH']
    if not portfolio_stock_tickers_assess:
        return 0.0, 0.0

    all_tickers_for_hist_fetch = list(set(portfolio_stock_tickers_assess + ['SPY']))
    hist_data_assess = await get_yf_data_singularity(all_tickers_for_hist_fetch, period=backtest_period, interval="1d", is_called_by_ai=True)

    if hist_data_assess.empty or 'SPY' not in hist_data_assess.columns or hist_data_assess['SPY'].isnull().all() or len(hist_data_assess['SPY'].dropna()) < 20:
        return None

    daily_returns_assess_df = hist_data_assess.pct_change(fill_method=None).iloc[1:]
    
    if daily_returns_assess_df.empty or 'SPY' not in daily_returns_assess_df.columns:
        return None

    spy_returns_series = daily_returns_assess_df['SPY'].dropna()
    if spy_returns_series.empty or spy_returns_series.std() == 0:
        return None

    stock_metrics_calculated = {}
    for ticker_met in portfolio_stock_tickers_assess:
        beta_val, correlation_val = np.nan, np.nan
        if ticker_met in daily_returns_assess_df.columns and not daily_returns_assess_df[ticker_met].isnull().all():
            ticker_returns_series = daily_returns_assess_df[ticker_met].dropna()
            aligned_data = pd.concat([ticker_returns_series, spy_returns_series], axis=1, join='inner').dropna()
            if len(aligned_data) >= 20:
                aligned_ticker_returns = aligned_data.iloc[:, 0]
                aligned_spy_returns = aligned_data.iloc[:, 1]
                if aligned_ticker_returns.std() > 1e-9 and aligned_spy_returns.std() > 1e-9:
                    try:
                        covariance_matrix = np.cov(aligned_ticker_returns, aligned_spy_returns)
                        if covariance_matrix.shape == (2,2) and covariance_matrix[1, 1] != 0:
                             beta_val = covariance_matrix[0, 1] / covariance_matrix[1, 1]
                        correlation_coef_matrix = np.corrcoef(aligned_ticker_returns, aligned_spy_returns)
                        if correlation_coef_matrix.shape == (2,2):
                            correlation_val = correlation_coef_matrix[0, 1]
                            if pd.isna(correlation_val): correlation_val = 0.0
                    except (ValueError, IndexError, TypeError, np.linalg.LinAlgError):
                        pass
                else:
                    beta_val, correlation_val = 0.0, 0.0
        stock_metrics_calculated[ticker_met] = {'beta': beta_val, 'correlation': correlation_val}

    stock_metrics_calculated['CASH'] = {'beta': 0.0, 'correlation': 0.0}
    weighted_beta_sum, weighted_correlation_sum = 0.0, 0.0

    for holding in valid_holdings_for_calc:
        ticker_h, value_h = holding['ticker'].upper(), holding['value']
        weight_h = value_h / total_portfolio_value
        metrics_for_ticker = stock_metrics_calculated.get(ticker_h)
        if metrics_for_ticker:
            beta_for_calc = metrics_for_ticker.get('beta', 0.0)
            corr_for_calc = metrics_for_ticker.get('correlation', 0.0)
            if not pd.isna(beta_for_calc): weighted_beta_sum += weight_h * beta_for_calc
            if not pd.isna(corr_for_calc): weighted_correlation_sum += weight_h * corr_for_calc
            
    return weighted_beta_sum, weighted_correlation_sum

async def get_single_stock_beta_corr(ticker: str, period: str, is_called_by_ai: bool = True) -> tuple[Optional[float], Optional[float]]:
    portfolio = [{'ticker': ticker, 'value': 100.0}]
    result = await calculate_portfolio_beta_correlation_singularity(portfolio, 100.0, period, is_called_by_ai)
    return result if result else (None, None)

async def get_market_invest_score_for_powerscore() -> Optional[float]:
    try:
        vix_hist = await asyncio.to_thread(yf.Ticker('^VIX').history, period="5d")
        vix_price = vix_hist['Close'].iloc[-1]
        vix_likelihood = np.clip(0.01384083 * (vix_price ** 2), 0, 100)
        spy_hist = await asyncio.to_thread(yf.Ticker('SPY').history, period="5y", interval="1mo")
        spy_hist['EMA_8'] = spy_hist['Close'].ewm(span=8, adjust=False).mean()
        spy_hist['EMA_55'] = spy_hist['Close'].ewm(span=55, adjust=False).mean()
        ema_8, ema_55 = spy_hist['EMA_8'].iloc[-1], spy_hist['EMA_55'].iloc[-1]
        if ema_55 == 0: return None
        x_value = (((ema_8 - ema_55) / ema_55) + 0.5) * 100
        ema_likelihood = np.clip(100 * np.exp(-((45.622216 * x_value / 2750) ** 4)), 0, 100)
        if ema_likelihood == 0: return 0.0
        return float(np.clip(50.0 - ((vix_likelihood / ema_likelihood) - 1.0) * 100.0, 0, 100))
    except Exception:
        return None

async def calculate_volatility_metrics(ticker: str, period: str) -> tuple[Optional[float], Optional[float]]:
    try:
        hist_data = await asyncio.to_thread(yf.Ticker(ticker).history, period=period)
        if hist_data.empty or len(hist_data) <= 30: return None, None
        hist_data['daily_return'] = hist_data['Close'].pct_change()
        rolling_hv = hist_data['daily_return'].rolling(window=30).std() * (252**0.5)
        hv_series = rolling_hv.dropna()
        vol_rank = percentileofscore(hv_series, hv_series.iloc[-1]) if len(hv_series) > 1 else None
        return None, vol_rank
    except Exception:
        return None, None

async def handle_fundamentals_command(ai_params: dict, is_called_by_ai: bool):
    ticker = ai_params.get("ticker")
    if not ticker: return {"error": "Ticker not provided"}
    info = await get_yfinance_info_robustly(ticker)
    if not info: return {"error": f"Could not retrieve data for {ticker}"}
    pe = safe_get(info, 'trailingPE'); rg = safe_get(info, 'revenueGrowth'); de = safe_get(info, 'debtToEquity'); pm = safe_get(info, 'profitMargins')
    total_score, possible_score = 0.0, 0.0
    if pe is not None: possible_score += 25; total_score += 25 * np.exp(-0.00042 * pe**2) if pe > 0 else 0
    if rg is not None: possible_score += 25; total_score += 25 / (1 + np.exp(-0.11 * ((rg * 100) - 12.5)))
    if de is not None: possible_score += 25; total_score += 25 * np.exp(-0.00956 * de) if de > 0 else 25
    if pm is not None: possible_score += 25; total_score += 25 / (1 + np.exp(-0.11 * ((pm * 100) - 12.5)))
    final_score = (total_score / possible_score) * 100 if possible_score > 0 else 0.0
    return {"fundamental_score": final_score}

def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff().where(lambda x: x > 0, 0).rolling(14).mean() / -data['Close'].diff().where(lambda x: x < 0, 0).rolling(14).mean())))
    exp1 = data['Close'].ewm(span=12, adjust=False).mean(); exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2; data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    sma50 = data['Close'].rolling(window=50).mean(); sma200 = data['Close'].rolling(window=200).mean()
    data['SMA_Diff'] = ((sma50 - sma200) / sma200) * 100
    data['Volatility'] = data['Close'].pct_change().rolling(window=30).std() * np.sqrt(252)
    return data

async def handle_mlforecast_command(ai_params: dict, is_called_by_ai: bool = True):
    ticker = ai_params.get("ticker")
    if not ticker: return []
    data = await asyncio.to_thread(yf.download, ticker, period="5y", progress=False, auto_adjust=True)
    
    if data.empty or len(data) < 252:
        return []
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    data = calculate_technical_indicators(data)
    features = ['RSI', 'MACD', 'MACD_Signal', 'SMA_Diff', 'Volatility']
    results = []
    for period_name, horizon in [("5-Day", 5), ("1-Month (21-Day)", 21), ("3-Month (63-Day)", 63)]:
        df = data.copy(); df['Future_Close'] = df['Close'].shift(-horizon); df['Pct_Change'] = (df['Future_Close'] - df['Close']) / df['Close']
        df.dropna(subset=features + ['Pct_Change'], inplace=True)
        if len(df) < 50: continue
        X, y = df[features], df['Pct_Change']
        reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1); reg.fit(X, y)
        magnitude_pred = reg.predict(X.iloc[-1:])[0] * 100
        results.append({"Period": period_name, "Est. % Change": f"{magnitude_pred:+.2f}%"})
    return results

async def get_powerscore_explanation(ticker: str, component_scores: dict, model_to_use: Any, lock_to_use: asyncio.Lock) -> str:
    if not model_to_use: return "AI model unavailable."
    prompt = f"As a financial analyst, concisely summarize the profile of {ticker} based on these Prime component scores: Market={component_scores.get('R_prime')}, Beta/Corr={component_scores.get('AB_prime')}, Volatility={component_scores.get('AA_prime')}, Fundamentals={component_scores.get('F_prime')}, Technicals={component_scores.get('Q_prime')}, Sentiment={component_scores.get('S_prime')}, ML Forecast={component_scores.get('M_prime')}. Highlight strengths and weaknesses without stating the scores."
    try:
        async with lock_to_use:
            response = await asyncio.to_thread(model_to_use.generate_content, prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error generating AI summary: {e}"

# --- Main Command Handler ---

async def handle_powerscore_command(args: List[str] = None, ai_params: dict = None, is_called_by_ai: bool = False, gemini_model_override: Any = None, api_lock_override: asyncio.Lock = None):
    # --- FIX 2: Use the globally defined model if no override is provided ---
    model_to_use = gemini_model_override or gemini_model
    lock_to_use = api_lock_override or GEMINI_API_LOCK
    
    ticker, sensitivity = None, None
    if is_called_by_ai and ai_params:
        ticker, sensitivity = ai_params.get("ticker"), int(ai_params.get("sensitivity"))
    elif args and len(args) == 2:
        ticker, sensitivity = args[0].upper(), int(args[1])
    if not ticker or sensitivity not in [1, 2, 3]:
        message = "Usage: /powerscore <TICKER> <SENSITIVITY 1-3>"
        return {"error": message} if is_called_by_ai else print(message)

    if not is_called_by_ai: print(f"\n--- Generating PowerScore for {ticker} (Sensitivity: {sensitivity}) ---")
    
    period_map = {1: '10y', 2: '5y', 3: '1y'}
    backtest_period = period_map[sensitivity]
    
    tasks = { 
        "R": get_market_invest_score_for_powerscore(), 
        "ABB_ABC": get_single_stock_beta_corr(ticker, backtest_period), 
        "AA": calculate_volatility_metrics(ticker, backtest_period), 
        "F": handle_fundamentals_command(ai_params={'ticker': ticker}, is_called_by_ai=True), 
        "Q": calculate_ema_invest(ticker, sensitivity, is_called_by_ai=True), 
        "S": handle_sentiment_command(
            ai_params={'ticker': ticker}, 
            is_called_by_ai=True, 
            gemini_model_override=model_to_use, 
            api_lock_override=lock_to_use
        ), 
        "M": handle_mlforecast_command(ai_params={'ticker': ticker}) 
    }
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    raw_values = dict(zip(tasks.keys(), results))
    
    m_period_map = {1: ["1-Year (52-Week)", "6-Month (26-Week)"], 2: ["6-Month (26-Week)", "3-Month (63-Day)"], 3: ["1-Month (21-Day)", "5-Day"]}
    m_forecast_val, used_m_period = None, "N/A"
    if isinstance(raw_values.get('M'), list) and raw_values['M']:
        m_lookup = {item.get("Period"): item for item in raw_values['M']}
        for period in m_period_map[sensitivity]:
            if period in m_lookup: 
                m_forecast_val = float(m_lookup[period].get("Est. % Change", "0%").replace('%', ''))
                used_m_period = period
                break
    
    raw = {
        'R': raw_values.get('R'),
        'ABB': raw_values.get('ABB_ABC')[0] if isinstance(raw_values.get('ABB_ABC'), tuple) else None,
        'ABC': raw_values.get('ABB_ABC')[1] if isinstance(raw_values.get('ABB_ABC'), tuple) else None,
        'AA': raw_values.get('AA')[1] if isinstance(raw_values.get('AA'), tuple) else None,
        'F': raw_values.get('F').get('fundamental_score') if isinstance(raw_values.get('F'), dict) else None,
        'Q': raw_values.get('Q')[1] if isinstance(raw_values.get('Q'), tuple) else None,
        'S': raw_values.get('S').get('sentiment_score_raw') if isinstance(raw_values.get('S'), dict) else None,
        'M': m_forecast_val,
    }

    prime = {}
    if raw['ABB'] is not None: prime['ABB'] = np.clip(100/(((raw['ABB']-2)**2)+1) if raw['ABB']<=2 else 400/((3*(raw['ABB']-2)**2)+4),0,100)
    if raw['ABC'] is not None: prime['ABC'] = np.clip(1.01*(100*(297**raw['ABC']))/((297**raw['ABC'])+3),0,100)
    if 'ABB' in prime and 'ABC' in prime: prime['AB'] = (prime['ABB'] + prime['ABC']) / 2
    if raw['F'] is not None: prime['F'] = raw['F']
    if raw['S'] is not None: prime['S'] = 50 * (raw['S'] + 1)
    if raw['AA'] is not None: prime['AA'] = 100 - raw['AA']
    if raw['R'] is not None: prime['R'] = raw['R']
    if raw['M'] is not None:
        m, s = raw['M'], sensitivity
        if s == 1: prime['M'] = 100/(1+(9*(1.1396**-m)))
        elif s == 2: prime['M'] = 100/(1+(4*(1.23**-m)))
        else: prime['M'] = 100/(1+(3*(3**-m)))
    if raw['Q'] is not None:
        q, s = raw['Q'], sensitivity
        if s == 1: prime['Q'] = 100/(1+(math.e**(-0.0879*(q-50))))
        elif s == 2: prime['Q'] = 100/(1+(math.e**(-0.0628*(q-50))))
        else: prime['Q'] = 100/(1+(math.e**(-0.0981*(q-50))))

    weights = {1: {'R':.15,'AB':.15,'AA':.15,'F':.15,'Q':.2,'S':.1,'M':.1}, 2: {'R':.2,'AB':.1,'AA':.1,'F':.05,'Q':.25,'S':.15,'M':.15}, 3: {'R':.25,'AB':.05,'AA':.05,'F':0,'Q':.3,'S':.25,'M':.2}}
    ps, total_w = 0.0, 0.0
    for key, weight in weights[sensitivity].items():
        if key in prime and prime[key] is not None: ps += prime[key] * weight; total_w += weight
    
    final_ps = np.clip((ps / total_w) if total_w > 0 else 0, 0, 100)
    
    if is_called_by_ai:
        return {"powerscore": final_ps}

    ai_summary = await get_powerscore_explanation(ticker, {'R_prime': prime.get('R'), 'AB_prime': prime.get('AB'), 'AA_prime': prime.get('AA'), 'F_prime': prime.get('F'), 'Q_prime': prime.get('Q'), 'S_prime': prime.get('S'), 'M_prime': prime.get('M')}, model_to_use, lock_to_use)
    
    # CLI Output
    table_data = [
        ["Market Invest Score", f"{raw.get('R'):.2f}" if raw.get('R') is not None else "N/A", f"{prime.get('R'):.2f}" if prime.get('R') is not None else "N/A"],
        ["Beta/Corr Average", f"{raw.get('ABB'):.2f}/{raw.get('ABC'):.2f}" if raw.get('ABB') is not None else "N/A", f"{prime.get('AB'):.2f}" if prime.get('AB') is not None else "N/A"],
        ["Volatility Rank", f"{raw.get('AA'):.2f}%" if raw.get('AA') is not None else "N/A", f"{prime.get('AA'):.2f}" if prime.get('AA') is not None else "N/A"],
        ["Fundamental Score", f"{raw.get('F'):.2f}" if raw.get('F') is not None else "N/A", f"{prime.get('F'):.2f}" if prime.get('F') is not None else "N/A"],
        ["QuickScore", f"{raw.get('Q'):.2f}%" if raw.get('Q') is not None else "N/A", f"{prime.get('Q'):.2f}" if prime.get('Q') is not None else "N/A"],
        ["Sentiment Score", f"{raw.get('S'):.2f}" if raw.get('S') is not None else "N/A", f"{prime.get('S'):.2f}" if prime.get('S') is not None else "N/A"],
        [f"ML Forecast ({used_m_period})", f"{raw.get('M'):.2f}%" if raw.get('M') is not None else "N/A", f"{prime.get('M'):.2f}" if prime.get('M') is not None else "N/A"]
    ]
    print(tabulate(table_data, headers=["Metric", "Raw Value", "Prime Score"], tablefmt="grid"))
    print(f"\nAI Analyst Summary:\n{ai_summary}")
    print(f"\nFINAL POWERSCORE: {final_ps:.2f} / 100.00")