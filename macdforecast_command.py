# --- Imports for macdforecast_command ---
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --- Helper Function (moved for self-containment) ---

def plot_macd_forecast_graph(ticker, data, forecast_day_count, avg_price_change, forecast_date, is_called_by_ai):
    """Helper to generate and save the forecast graph. This is synchronous."""
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(12, 7))

        price_column = 'Adj Close'
        past_month_data = data.iloc[-22:]
        ax.plot(past_month_data.index, past_month_data[price_column], label='Past Month Prices', color='grey', marker='.')

        if forecast_day_count > 0 and pd.notna(avg_price_change):
            last_price = data[price_column].iloc[-1]
            last_date = data.index[-1]
            forecast_dates_range = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_day_count)
            forecast_prices = [last_price * ((1 + avg_price_change) ** (i+1)) for i in range(forecast_day_count)]
            ax.plot(forecast_dates_range, forecast_prices, label='Forecasted Prices', linestyle='--', color='cyan', marker='o')
            ax.annotate(f"${forecast_prices[-1]:.2f}\n{forecast_date.strftime('%Y-%m-%d')}",
                        xy=(forecast_dates_range[-1], forecast_prices[-1]),
                        xytext=(10, -10), textcoords='offset points',
                        color='white', backgroundcolor='black',
                        arrowprops=dict(arrowstyle="->", color="cyan"))

        ax.set_title(f"{ticker} Price: Past Month and MACD CTC Forecast", color='white')
        ax.set_xlabel("Date", color='white')
        ax.set_ylabel("Price (USD)", color='white')
        ax.legend(facecolor='black', edgecolor='white', labelcolor='white')
        ax.grid(True, color='dimgray', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.tick_params(axis='x', colors='white', rotation=25)
        ax.tick_params(axis='y', colors='white')
        fig.tight_layout()

        filename = f"macd_forecast_graph_{ticker.replace('.','-')}_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black', edgecolor='black')
        plt.close(fig)
        if not is_called_by_ai:
            print(f"📂 MACD forecast graph saved: {filename}")
        return filename
    except Exception as e:
        if not is_called_by_ai:
            print(f"❌ Error plotting MACD forecast graph for {ticker}: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        return None

# --- Main Command Handler ---

async def handle_macd_forecast_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """
    Handles the /macdforecast command.
    """
    if not is_called_by_ai:
        print("\n--- /macdforecast Command ---")

    tickers = []
    if ai_params:
        tickers_str = ai_params.get("tickers")
        if not tickers_str:
            return "Error for AI (/macdforecast): 'tickers' parameter (string of tickers) is required."
        tickers = [t.strip().upper() for t in tickers_str.replace(',', ' ').split() if t.strip()]
    elif args:
        tickers = [t.strip().upper() for t in args]
    else:
        tickers_input = input("Enter a list of stock tickers separated by spaces or commas: ")
        tickers = [ticker.strip().upper() for ticker in tickers_input.replace(',', ' ').split() if ticker.strip()]

    if not tickers:
        msg = "No tickers provided for MACD forecast."
        if not is_called_by_ai: print(msg)
        return f"Error: {msg}" if is_called_by_ai else None

    if not is_called_by_ai:
        print(f"\nProcessing the following tickers for MACD CTC Forecast: {tickers}")
        print("-" * 60)

    all_summaries = []
    for ticker in tickers:
        try:
            end_date = datetime.today()
            start_date = end_date - timedelta(days=365)
            data = await asyncio.to_thread(yf.download, ticker, start=start_date, end=end_date, progress=False, auto_adjust=False)

            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            price_column = 'Adj Close'
            if data.empty or price_column not in data.columns or data[price_column].isnull().all():
                summary = f"Forecast for {ticker}: Could not be produced. No valid '{price_column}' data."
                all_summaries.append(summary)
                continue

            data['macd'] = data[price_column].ewm(span=25).mean() - data[price_column].ewm(span=75).mean()
            if len(data['macd'].dropna()) < 4:
                all_summaries.append(f"Forecast for {ticker}: Not enough data for MACD analysis.")
                continue

            macd_changes = data['macd'].diff().iloc[-3:]
            
            if len(macd_changes.dropna()) < 2:
                all_summaries.append(f"Forecast for {ticker}: Not enough data for trend analysis.")
                continue

            failed_conditions = False
            if macd_changes.isnull().any():
                failed_conditions = True
            if not failed_conditions:
                try:
                    if not (abs(macd_changes.iloc[-1]) < abs(macd_changes.iloc[-2])):
                        failed_conditions = True
                except IndexError:
                    failed_conditions = True
            if not failed_conditions:
                is_consistent = (macd_changes > 0).all() or (macd_changes < 0).all()
                if not is_consistent:
                    failed_conditions = True

            if failed_conditions:
                all_summaries.append(f"Forecast for {ticker}: Could not be produced due to failed conditions.")
                continue

            avg_macd_change = macd_changes.mean()
            last_macd = data['macd'].iloc[-1]
            if pd.isna(avg_macd_change) or avg_macd_change == 0:
                all_summaries.append(f"Forecast for {ticker}: Could not be produced. Average MACD change is zero or invalid.")
                continue

            forecast_day_count = int(abs(last_macd / avg_macd_change))
            if not (0 < forecast_day_count < 365):
                all_summaries.append(f"Forecast for {ticker}: Projected day count ({forecast_day_count}) is not reasonable.")
                continue

            data['price_change_pct'] = data[price_column].pct_change()
            data['macd_change_full'] = data['macd'].diff()
            tolerance = 0.1 * abs(avg_macd_change)
            similar_macd_changes = data[abs(data['macd_change_full'] - avg_macd_change) < tolerance]
            
            if similar_macd_changes.empty:
                all_summaries.append(f"Forecast for {ticker}: No similar historical instances found.")
                continue
            
            # --- START OF TYPO FIX ---
            # Corrected 'price_pct_change' to 'price_change_pct' to match the column name created above.
            avg_price_change = similar_macd_changes['price_change_pct'].mean()
            # --- END OF TYPO FIX ---

            if pd.isna(avg_price_change):
                all_summaries.append(f"Forecast for {ticker}: Could not calculate a valid average price change.")
                continue

            last_price = data[price_column].iloc[-1]
            forecasted_price = last_price * ((1 + avg_price_change) ** forecast_day_count)
            percent_change = ((forecasted_price - last_price) / last_price) * 100
            forecast_date = data.index[-1] + pd.tseries.offsets.BusinessDay(n=forecast_day_count)
            
            summary = (f"Forecast for {ticker}: Successful. "
                       f"Price: ${forecasted_price:.2f} ({percent_change:+.2f}%) "
                       f"on {forecast_date.strftime('%Y-%m-%d')}.")
            all_summaries.append(summary)
            
            if not is_called_by_ai:
                await asyncio.to_thread(plot_macd_forecast_graph, ticker, data, forecast_day_count, avg_price_change, forecast_date, is_called_by_ai)

        except Exception as e:
            all_summaries.append(f"An error occurred for {ticker}: {e}")
    
    if not is_called_by_ai:
        for s in all_summaries: print(s); print("-" * 60)

    return " | ".join(all_summaries) if is_called_by_ai else None