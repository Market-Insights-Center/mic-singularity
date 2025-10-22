# --- Imports for heatmap_command ---
import asyncio
import os
import uuid
from io import StringIO
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
import yfinance as yf

# --- Constants (copied for self-containment) ---
PORTFOLIO_DB_FILE = 'portfolio_codes_database.csv'

# --- Helper Functions (copied or moved for self-containment) ---

async def get_yf_download_robustly(tickers: list, **kwargs) -> pd.DataFrame:
    """A robust wrapper for yf.download with retry logic."""
    for attempt in range(3):
        try:
            data = await asyncio.to_thread(yf.download, tickers=tickers, progress=False, **kwargs)
            if not data.empty:
                return data
        except Exception:
            if attempt < 2:
                await asyncio.sleep((attempt + 1) * 2)
    return pd.DataFrame()

async def get_yf_data_singularity(tickers: List[str], period: str = "10y") -> pd.DataFrame:
    """Downloads historical closing price data for multiple tickers."""
    if not tickers:
        return pd.DataFrame()
    data = await get_yf_download_robustly(tickers=list(set(tickers)), period=period, auto_adjust=False, group_by='ticker')
    if data.empty:
        return pd.DataFrame()
    
    close_prices = data.xs('Close', level=1, axis=1) if isinstance(data.columns, pd.MultiIndex) else data[['Close']]
    return close_prices.dropna(axis=0, how='all').dropna(axis=1, how='all')

# --- START OF FIX ---
# Replaced with the more robust version from the main singularity script.
def get_sp500_symbols_singularity() -> List[str]:
    """Fetches S&P 500 symbols from Wikipedia using the requests library for reliability."""
    try:
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(sp500_url, headers=headers, timeout=15)
        response.raise_for_status()
        dfs = pd.read_html(StringIO(response.text))
        if not dfs: return []
        sp500_df = dfs[0]
        if 'Symbol' not in sp500_df.columns: return []
        symbols = [str(s).replace('.', '-') for s in sp500_df['Symbol'].tolist() if isinstance(s, str)]
        return sorted(list(set(s for s in symbols if s)))
    except Exception as e:
        print(f"-> Error fetching S&P 500 symbols: {e}")
        return []

# Updated to use the robust requests pattern for all indices.
async def get_specific_index_tickers(index_symbol: str) -> List[str]:
    """Fetches component tickers for a major index from Wikipedia."""
    index_symbol = index_symbol.upper()
    
    if index_symbol == 'SPY':
        # Use the now-robust, blocking function in a separate thread.
        return await asyncio.to_thread(get_sp500_symbols_singularity)

    urls = {
        'QQQ': ('https://en.wikipedia.org/wiki/Nasdaq-100', 4, 'Ticker'),
        'DIA': ('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average', 1, 'Symbol'),
        'RUT': ('https://en.wikipedia.org/wiki/List_of_Russell_2000_stocks', 2, 'Ticker')
    }
    if index_symbol not in urls:
        print(f"-> Warning: Index '{index_symbol}' is not supported for heatmap generation.")
        return []

    url, table_index, col_name = urls[index_symbol]
    
    try:
        # Define a blocking function to be run in a separate thread.
        def fetch_and_parse():
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            tables = pd.read_html(StringIO(response.text))
            
            if len(tables) <= table_index:
                print(f"-> Error: Table index {table_index} not found for {index_symbol}. Page structure may have changed.")
                return []
            df = tables[table_index]
            
            if col_name not in df.columns:
                 print(f"-> Error: Column '{col_name}' not found for {index_symbol}. Page structure may have changed.")
                 return []
            tickers = [str(s).replace('.', '-') for s in df[col_name].tolist() if isinstance(s, str)]
            return sorted(list(set(t for t in tickers if t)))
        
        # Run the blocking network I/O in a thread to keep the app responsive.
        return await asyncio.to_thread(fetch_and_parse)
        
    except Exception as e:
        print(f"-> Error fetching data for index '{index_symbol}': {e}")
        return []
# --- END OF FIX ---

def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, title: str) -> Optional[str]:
    """Generates and saves a seaborn heatmap for a given correlation matrix."""
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(16, 12))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(correlation_matrix, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                    linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        ax.set_title(title, fontsize=18, color='white', pad=20)
        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        fig.tight_layout(pad=3.0)
        filename = f"heatmap_{title.replace(' ', '_').lower()}_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black', edgecolor='black', dpi=300)
        plt.close(fig)
        print(f"📂 Heatmap saved successfully: {filename}")
        return filename
    except Exception as e:
        print(f"❌ Error generating heatmap: {e}")
        return None

# --- Main Command Handler ---

async def handle_heatmap_command(args: List[str], is_called_by_ai: bool = False):
    """
    Generates correlation heatmaps for a given list of tickers, an index, or a portfolio.
    """
    if not is_called_by_ai:
        print("\n--- Correlation Heatmap Generator ---")

    if not args:
        print("Usage: /heatmap <tickers | I-Index | P-PortfolioCode>")
        return

    input_arg = args[0].upper()
    ticker_list = []
    title_prefix = ""

    if input_arg.startswith('I-'):
        index_symbol = input_arg.split('I-')[1]
        print(f"-> Fetching component tickers for index: {index_symbol}...")
        title_prefix = f"Index_{index_symbol}"
        ticker_list = await get_specific_index_tickers(index_symbol)
    elif input_arg.startswith('P-'):
        portfolio_code = input_arg.split('P-')[1]
        print(f"-> Loading tickers from portfolio: {portfolio_code}...")
        title_prefix = f"Portfolio_{portfolio_code}"
        if os.path.exists(PORTFOLIO_DB_FILE):
            df = pd.read_csv(PORTFOLIO_DB_FILE)
            p_row = df[df['portfolio_code'].astype(str) == portfolio_code]
            if not p_row.empty:
                p_series = p_row.iloc[0]
                temp_tickers = set()
                for i in range(1, int(p_series.get('num_portfolios', 0)) + 1):
                    tickers_str = p_series.get(f'tickers_{i}', '')
                    if tickers_str:
                        temp_tickers.update([t.strip() for t in tickers_str.split(',')])
                ticker_list = sorted(list(temp_tickers))
    else:
        ticker_list = [arg.upper() for arg in args]
        title_prefix = "_".join(ticker_list[:3])
        if len(ticker_list) > 3: title_prefix += "_and_others"

    if not ticker_list:
        print("-> Error: No valid tickers to process. Aborting.")
        return

    print(f"-> Found {len(ticker_list)} tickers. Downloading historical data for correlation analysis...")
    hist_data = await get_yf_data_singularity(ticker_list, period="1y")
    if hist_data.empty:
        print("-> Error: Failed to download historical data. Aborting.")
        return
        
    daily_returns = hist_data.pct_change().dropna()
    correlation_matrix = daily_returns.corr(method='pearson')

    print("-> Generating heatmap...")
    heatmap_title = f"Daily_Returns_Correlation_{title_prefix}"
    await asyncio.to_thread(plot_correlation_heatmap, correlation_matrix, heatmap_title)