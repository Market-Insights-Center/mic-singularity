# --- Imports for history_command ---
import os
import uuid
import asyncio
from typing import List, Dict, Optional

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Constants ---
RISK_CSV_FILE = 'market_data.csv'

# --- Core Logic (moved from the old risk.py) ---

async def generate_risk_graphs_singularity(is_called_by_ai: bool = False):
    """
    Generates and saves a series of historical graphs for the R.I.S.K. module.
    """
    if not is_called_by_ai:
        print("\n--- Generating R.I.S.K. Historical Graphs ---")

    if not os.path.exists(RISK_CSV_FILE):
        print(f"Error: Data file '{RISK_CSV_FILE}' not found. Cannot generate graphs.")
        return

    try:
        df = pd.read_csv(RISK_CSV_FILE, on_bad_lines='skip')
        if df.empty:
            print("Data file is empty. No graphs generated.")
            return

        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce').dt.tz_localize(None)
        df = df.sort_values(by='Timestamp').dropna(subset=['Timestamp'])
        plt.style.use('dark_background')

        # Graph 1: Market Scores
        fig1, ax1 = plt.subplots(figsize=(14, 7))
        for col in ['General Market Score', 'Large Market Cap Score', 'Combined Score']:
            if col in df.columns:
                ax1.plot(df['Timestamp'], pd.to_numeric(df[col], errors='coerce'), label=col)
        ax1.set_title('Historical Market Scores'); ax1.legend(); ax1.grid(True, linestyle=':', alpha=0.6)
        filename1 = f"risk_market_scores_hist_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename1, facecolor='black'); plt.close(fig1)
        if not is_called_by_ai: print(f"📂 Graph saved: {filename1}")

        # Graph 2: SPY & VIX Prices
        fig2, ax2 = plt.subplots(figsize=(14, 7))
        ax2.plot(df['Timestamp'], pd.to_numeric(df['Live SPY Price'], errors='coerce'), color='lime', label='SPY Price')
        ax2.set_ylabel('SPY Price ($)', color='lime')
        ax3 = ax2.twinx()
        ax3.plot(df['Timestamp'], pd.to_numeric(df['Live VIX Price'], errors='coerce'), color='red', label='VIX Price')
        ax3.set_ylabel('VIX Price', color='red')
        fig2.suptitle('Historical SPY & VIX Prices'); fig2.legend(loc="upper left"); ax2.grid(True, linestyle=':', alpha=0.6)
        filename2 = f"risk_spy_vix_prices_hist_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename2, facecolor='black'); plt.close(fig2)
        if not is_called_by_ai: print(f"📂 Graph saved: {filename2}")

    except Exception as e:
        print(f"❌ An error occurred during graph generation: {e}")

# --- Main Command Handler ---

async def handle_history_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """
    Handles the /history command by calling the graph generation logic directly.
    """
    await generate_risk_graphs_singularity(is_called_by_ai=is_called_by_ai)

    if is_called_by_ai:
        return "The R.I.S.K. history graph generation has been executed."