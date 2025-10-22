# --- Imports for fundamentals_command ---
import asyncio
import random
from typing import List, Dict, Any, Optional

import yfinance as yf
import numpy as np
from tabulate import tabulate

# --- Global Variables & Constants ---
YFINANCE_API_SEMAPHORE = asyncio.Semaphore(8)

# --- Helper Functions (copied for self-containment) ---

def safe_get(data_dict: Dict, key: str, default: Any = None) -> Any:
    """Safely gets a value from a dictionary."""
    value = data_dict.get(key, default)
    if value is None or value == 'None':
        return default
    return value

async def get_yfinance_info_robustly(ticker: str) -> Optional[Dict[str, Any]]:
    """A robust, centralized function to fetch yfinance .info data."""
    async with YFINANCE_API_SEMAPHORE:
        for attempt in range(3):
            try:
                await asyncio.sleep(random.uniform(0.2, 0.5))
                stock_info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)
                if stock_info and not stock_info.get('regularMarketPrice'):
                    raise ValueError(f"Incomplete data received for {ticker}")
                return stock_info
            except Exception:
                if attempt < 2:
                    await asyncio.sleep((attempt + 1) * 2)
    return None

# --- Main Command Handler ---

async def handle_fundamentals_command(args: list = None, ai_params: dict = None, is_called_by_ai: bool = False):
    """
    Fetches key fundamental data for a stock and calculates a dynamic fundamental score.
    """
    if is_called_by_ai:
        ticker = ai_params.get("ticker") if ai_params else None
    else:
        ticker = args[0] if args else None

    if not ticker:
        message = "Please provide a stock ticker. Usage: /fundamentals <TICKER>"
        if not is_called_by_ai:
            print(message)
        return {"error": message} if is_called_by_ai else None

    if not is_called_by_ai:
        print(f"\n--- Fundamental Analysis for {ticker.upper()} ---")

    try:
        info = await get_yfinance_info_robustly(ticker)

        if not info:
             message = f"Could not retrieve any data for '{ticker}'. The ticker may be invalid or delisted."
             if not is_called_by_ai:
                 print(message)
             return {"error": message} if is_called_by_ai else None

        pe_ratio_raw = safe_get(info, 'trailingPE')
        revenue_growth_raw = safe_get(info, 'revenueGrowth')
        debt_to_equity_raw = safe_get(info, 'debtToEquity')
        profit_margin_raw = safe_get(info, 'profitMargins')
        
        total_score, possible_score, pe_score, rev_g_score, de_score, margin_score = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
        if pe_ratio_raw is not None:
            possible_score += 25
            if pe_ratio_raw > 0: pe_score = 25 * np.exp(-0.00042 * pe_ratio_raw**2)
            total_score += pe_score
        
        if revenue_growth_raw is not None:
            possible_score += 25
            rev_g_score = 25 / (1 + np.exp(-0.11 * ((revenue_growth_raw * 100) - 12.5)))
            total_score += rev_g_score
        
        if debt_to_equity_raw is not None:
            possible_score += 25
            de_score = 25.0 if debt_to_equity_raw <= 0 else 25 * np.exp(-0.00956 * debt_to_equity_raw)
            total_score += de_score
        
        if profit_margin_raw is not None:
            possible_score += 25
            margin_score = 25 / (1 + np.exp(-0.11 * ((profit_margin_raw * 100) - 12.5)))
            total_score += margin_score
        
        final_score = (total_score / possible_score) * 100 if possible_score > 0 else 0.0
        
        table_data = [
            ["P/E Ratio", f"{pe_ratio_raw:.2f}" if pe_ratio_raw is not None else "N/A", f"{pe_score:.2f} / 25"],
            ["Revenue Growth", f"{revenue_growth_raw*100:.2f}%" if revenue_growth_raw is not None else "N/A", f"{rev_g_score:.2f} / 25"],
            ["Debt-to-Equity", f"{debt_to_equity_raw:.2f}" if debt_to_equity_raw is not None else "N/A", f"{de_score:.2f} / 25"],
            ["Profit Margin", f"{profit_margin_raw*100:.2f}%" if profit_margin_raw is not None else "N/A", f"{margin_score:.2f} / 25"],
            ["---", "---", "---"],
            ["Fundamental Score", "", f"{final_score:.2f} / 100"]
        ]

        if not is_called_by_ai:
            print(tabulate(table_data, headers=["Metric", "Value", "Score"], tablefmt="pretty"))
        
        return {
            "ticker": ticker.upper(),
            "pe_ratio": pe_ratio_raw,
            "revenue_growth": revenue_growth_raw,
            "debt_to_equity": debt_to_equity_raw,
            "profit_margin": profit_margin_raw,
            "fundamental_score": final_score
        }

    except Exception as e:
        message = f"An error occurred while fetching data for {ticker}: {e}"
        if not is_called_by_ai:
            print(message)
        return {"error": message} if is_called_by_ai else None