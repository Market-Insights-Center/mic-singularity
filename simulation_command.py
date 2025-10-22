# --- Imports for simulation_command ---
import asyncio
import random
import uuid
from datetime import datetime, timedelta
from io import StringIO
from typing import List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from tabulate import tabulate

# --- Helper Functions (copied or moved for self-containment) ---

def get_sp500_fallback_list() -> List[str]:
    """Returns a hardcoded list of S&P 500 symbols as a fallback."""
    print("⚠️ Wikipedia scrape failed. Using a hardcoded fallback list of S&P 500 tickers. This list may be slightly outdated.")
    return ['A', 'AAL', 'AAP', 'AAPL', 'ABBV', 'ABC', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP', 'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALK', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN', 'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'AVB', 'AVGO', 'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF-B', 'BIIB', 'BIO', 'BK', 'BKNG', 'BKR', 'BLK', 'BLDR', 'BMY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BWA', 'BX', 'BXP', 'C', 'CAG', 'CAH', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CFG', 'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMA', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF', 'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CSCO', 'CSGP', 'CSX', 'CTAS', 'CTLT', 'CTRA', 'CTS', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DAY', 'DD', 'DE', 'DECK', 'DFS', 'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA', 'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'ENPH', 'EOG', 'EPAM', 'EQIX', 'EQR', 'EQT', 'ERIE', 'ES', 'ESS', 'ETN', 'ETR', 'ETSY', 'EVA', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR', 'F', 'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FMC', 'FOX', 'FOXA', 'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GE', 'GEHC', 'GEN', 'GILD', 'GIS', 'GL', 'GLW', 'GM', 'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG', 'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE', 'IDXX', 'IEX', 'IFF', 'ILMN', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW', 'IVZ', 'J', 'JBL', 'JCI', 'JCP', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY', 'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD', 'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST', 'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MRO', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU', 'NCLH', 'NDAQ', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE', 'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OGN', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW', 'PARA', 'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEAK', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSX', 'PTC', 'PWR', 'PXD', 'PYPL', 'QCOM', 'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG', 'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SCL', 'SJM', 'SLB', 'SNA', 'SNPS', 'SO', 'SPG', 'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG', 'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV', 'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP', 'UPS', 'URI', 'USB', 'V', 'VEEV', 'VLO', 'VLTO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VTR', 'VTRS', 'VZ', 'WAB', 'WAT', 'WBD', 'WCN', 'WDC', 'WEC', 'WELL', 'WFC', 'WHR', 'WM', 'WMB', 'WMT', 'WRB', 'WRK', 'WST', 'WY', 'WYNN', 'X', 'XEL', 'XOM', 'XRAY', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZTS']

def get_sp500_symbols_singularity() -> List[str]:
    """
    Fetches S&P 500 symbols from Wikipedia using a robust method with headers.
    """
    print("-> Fetching S&P 500 symbols from Wikipedia...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        # Set a User-Agent to mimic a browser and avoid being blocked
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()  # This will raise an error for bad status codes (4xx or 5xx)
        
        # Use StringIO to parse the HTML text with pandas
        df = pd.read_html(StringIO(response.text))[0]
        symbols = [str(s).replace('.', '-') for s in df['Symbol'].tolist() if isinstance(s, str)]
        
        print(f"   Successfully fetched {len(symbols)} S&P 500 symbols.")
        return sorted(list(set(symbols)))
    except Exception as e:
        print(f"   FAILED to fetch S&P 500 symbols: {e}")
        # Return an empty list on failure, which the main handler will catch
        return []

def ask_singularity_input(prompt: str, validation_fn=None) -> Optional[str]:
    """Helper function to ask for user input with optional validation."""
    while True:
        user_response = input(f"{prompt}: ").strip()
        if not user_response:
            return None
        if validation_fn:
            if validation_fn(user_response):
                return user_response
            else:
                print("Invalid input.")
        else:
            return user_response

def plot_simulation_performance_graph(user_portfolio_history: list, index_data: pd.DataFrame, simulation_period_str: str, simulation_dates: pd.Index):
    """Generates and saves a graph comparing user's portfolio performance against market indices."""
    try:
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(14, 8))
        if isinstance(index_data.index, pd.DatetimeIndex):
            index_data.index = index_data.index.tz_localize(None)
        user_dates = simulation_dates[:len(user_portfolio_history)].tz_localize(None)
        user_df = pd.DataFrame({'Value': user_portfolio_history}, index=user_dates)
        user_normalized = (user_df['Value'] / user_df['Value'].iloc[0]) * 100
        ax.plot(user_df.index, user_normalized, label='Your Portfolio', color='cyan', linewidth=2.5)
        for column in index_data.columns:
            aligned_index_data = index_data[column].reindex(user_dates, method='ffill').dropna()
            if not aligned_index_data.empty:
                index_normalized = (aligned_index_data / aligned_index_data.iloc[0]) * 100
                ax.plot(aligned_index_data.index, index_normalized, label=column, linestyle='--')
        ax.set_title(f'Simulation Performance vs. Market Indices\n({simulation_period_str})', color='white')
        ax.set_ylabel('Performance (Normalized to 100)', color='white')
        ax.legend()
        ax.grid(True, alpha=0.3)
        filename = f"simulation_performance_{uuid.uuid4().hex[:6]}.png"
        plt.savefig(filename, facecolor='black')
        plt.close(fig)
        print(f"📂 Performance graph saved: {filename}")
    except Exception as e:
        print(f"❌ Error plotting performance graph: {e}")

# --- Main Command Handler ---

async def handle_simulation_command(args: List[str], is_called_by_ai: bool = False):
    """
    Orchestrates an interactive stock market simulation with robust stock selection.
    """
    if is_called_by_ai:
        return "This command is interactive and designed for direct user use in the CLI."

    print("\n--- Market Simulation Engine ---")

    # --- 1. SETUP ---
    num_stocks_str = ask_singularity_input("How many stocks to simulate? (3-10)", validation_fn=lambda x: 3 <= int(x) <= 10)
    if not num_stocks_str: return
    num_stocks_to_find = int(num_stocks_str)
    
    print("-> Setting up simulation, please wait...")

    sp500_tickers = await asyncio.to_thread(get_sp500_symbols_singularity)
    if not sp500_tickers:
        print("-> Error: Could not fetch the S&P 500 list, which is required for the simulation.")
        return

    simulation_df = None
    simulation_period_str = ""
    selected_tickers = []
    
    max_setup_attempts = 10
    for attempt in range(max_setup_attempts):
        try:
            random.shuffle(sp500_tickers)
            candidate_tickers = sp500_tickers[:num_stocks_to_find]

            def fetch_history_sync(ticker):
                stock = yf.Ticker(ticker)
                return stock.history(period="max", interval="1wk", auto_adjust=False)

            async def get_full_history(ticker):
                try:
                    data = await asyncio.to_thread(fetch_history_sync, ticker)
                    if data.empty or data['Close'].isnull().all(): return None
                    series = data['Close'].dropna()
                    series.name = ticker 
                    return series
                except Exception: return None

            tasks = [get_full_history(t) for t in candidate_tickers]
            histories = await asyncio.gather(*tasks)
            
            valid_histories = [h for h in histories if h is not None]
            if len(valid_histories) < num_stocks_to_find:
                await asyncio.sleep(0.5)
                continue

            latest_start_date = max(h.index.min() for h in valid_histories)
            earliest_end_date = min(h.index.max() for h in valid_histories)

            if (earliest_end_date - latest_start_date).days < (365 * 5 + 1):
                await asyncio.sleep(0.5)
                continue

            latest_possible_sim_start = earliest_end_date - timedelta(days=365*5)
            if latest_possible_sim_start < latest_start_date:
                await asyncio.sleep(0.5)
                continue

            random_days_offset = random.randint(0, (latest_possible_sim_start - latest_start_date).days)
            simulation_start_date = latest_start_date + timedelta(days=random_days_offset)
            simulation_end_date = simulation_start_date + timedelta(days=365*5)
            simulation_period_str = f"{simulation_start_date.strftime('%Y-%m-%d')} to {simulation_end_date.strftime('%Y-%m-%d')}"

            final_data_series = []
            for h in valid_histories:
                series_slice = h.loc[simulation_start_date:simulation_end_date]
                final_data_series.append(series_slice)

            simulation_df = pd.concat(final_data_series, axis=1).dropna()
            
            if len(simulation_df) > 250:
                selected_tickers = simulation_df.columns.tolist()
                break
            else:
                simulation_df = None
                
        except Exception:
            await asyncio.sleep(0.5)
            continue

    if simulation_df is None or not selected_tickers:
        print("-> Error: Could not find a suitable set of stocks after multiple attempts. Please try again.")
        return
        
    print("-> Setup complete. Starting simulation...")
    
    fake_names_pool = ["Aether Corp", "QuantumLeap Inc", "Stellar Solutions", "Nova Dynamics", "BioSynth", "Helios Energy", "Cybernetics Co", "Zenith Motors", "FusionWorks", "TerraForm"]
    random.shuffle(fake_names_pool)
    ticker_map, name_map, fake_ticker_map = {}, {}, {}

    for i, real_ticker in enumerate(selected_tickers):
        fake_name = fake_names_pool[i]
        fake_ticker = "".join([word[0] for word in fake_name.split()]) + "X" if len(fake_name.split()) > 1 else fake_name[:3].upper() + "X"
        ticker_map[real_ticker] = {'name': fake_name, 'ticker': fake_ticker}
        name_map[fake_name.lower()] = real_ticker
        fake_ticker_map[fake_ticker.upper()] = real_ticker

    portfolio_value_str = ask_singularity_input("Enter your starting portfolio value (e.g., 100000)", validation_fn=lambda x: float(x) > 0)
    if not portfolio_value_str: return
    
    buying_power = float(portfolio_value_str)
    user_portfolio = {info['name']: 0.0 for info in ticker_map.values()}
    trade_history = []
    user_portfolio_value_history = []
    current_week = 0
    
    loop = asyncio.get_running_loop()
    while current_week < len(simulation_df):
        current_prices = simulation_df.iloc[current_week]
        
        print("\n" + "="*80)
        print(f"Week: {current_week + 1} / {len(simulation_df)}")
        
        table_data = []
        headers = ["Company Name", "Ticker", "Price", "Weekly Change"]
        for real_ticker, fake_info in ticker_map.items():
            price = current_prices[real_ticker]
            change_pct = ((price / simulation_df[real_ticker].iloc[current_week-1]) - 1) * 100 if current_week > 0 else 0.0
            table_data.append([fake_info['name'], fake_info['ticker'], f"${price:,.2f}", f"{change_pct:+.2f}%"])
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))
        
        current_holdings_value = sum(user_portfolio[info['name']] * current_prices[real] for real, info in ticker_map.items())
        total_portfolio_value = buying_power + current_holdings_value
        user_portfolio_value_history.append(total_portfolio_value)
        
        print(f"\nPortfolio Value: ${total_portfolio_value:,.2f} | Buying Power: ${buying_power:,.2f}")
        
        if current_week == 0 or (current_week > 0 and current_week % 10 == 0):
            sample_name = random.choice(list(ticker_map.values()))['name']
            sample_ticker = random.choice(list(ticker_map.values()))['ticker']
            print("\n--- Sample Commands ---")
            print(f"  buy 10 {sample_ticker}")
            print(f"  sell max \"{sample_name}\"")
            print(f"  buy $5000 {sample_ticker}, sell 25 \"{sample_name}\"")
            print("  end simulation")
            print("-----------------------\n")

        if current_week >= len(simulation_df) - 1:
            print("-> Final week reached. Ending simulation...")
            await asyncio.sleep(2)
            break

        command_line = await loop.run_in_executor(None, lambda: input("Enter command(s) or press Enter to advance: "))
        
        commands = [cmd.strip() for cmd in command_line.split(',') if cmd.strip()]
        if not commands:
            current_week += 1
            continue
        
        for command in commands:
            parts = command.strip().split()
            if not parts: continue
            
            action = parts[0].lower()
            
            try:
                if action == "end":
                    print("-> Ending simulation by user command...")
                    current_week = len(simulation_df)
                    break

                if action in ["buy", "sell"]:
                    identifier = " ".join(parts[2:]).upper()
                    real_ticker, fake_name_traded = None, None

                    if identifier in fake_ticker_map:
                        real_ticker = fake_ticker_map[identifier]
                        fake_name_traded = ticker_map[real_ticker]['name']
                    else:
                        name_to_check = " ".join(parts[2:]).lower()
                        if name_to_check in name_map:
                            real_ticker = name_map[name_to_check]
                            fake_name_traded = ticker_map[real_ticker]['name']

                    if not real_ticker:
                        print(f"-> Error: '{' '.join(parts[2:])}' is not a valid company name or ticker.")
                    else:
                        price_at_trade = current_prices[real_ticker]
                        amount_str = parts[1].lower()
                        shares_to_trade = 0.0
                        
                        if amount_str == "max":
                            if action == "buy": shares_to_trade = buying_power / price_at_trade if price_at_trade > 0 else 0
                            else: shares_to_trade = user_portfolio[fake_name_traded]
                        elif amount_str.startswith('$'):
                            dollar_amount = float(amount_str[1:])
                            shares_to_trade = dollar_amount / price_at_trade if price_at_trade > 0 else 0
                        else:
                            shares_to_trade = float(amount_str)

                        if shares_to_trade > 1e-9:
                            cost = shares_to_trade * price_at_trade
                            if action == "buy" and cost > buying_power + 1e-9:
                                print(f"-> Error: Not enough buying power. Need ${cost:,.2f}, have ${buying_power:,.2f}.")
                            elif action == "sell" and shares_to_trade > user_portfolio[fake_name_traded] + 1e-9:
                                print(f"-> Error: Cannot sell more shares. Have {user_portfolio[fake_name_traded]:.4f}, tried {shares_to_trade:.4f}.")
                            else:
                                if action == "buy":
                                    buying_power -= cost
                                    user_portfolio[fake_name_traded] += shares_to_trade
                                    trade_history.append({'type': 'Buy', 'name': fake_name_traded, 'shares': shares_to_trade, 'price': price_at_trade, 'week': current_week})
                                    print(f"-> BOUGHT {shares_to_trade:.4f} shares of {fake_name_traded} for ${cost:,.2f}.")
                                else: # Sell
                                    buying_power += cost
                                    user_portfolio[fake_name_traded] -= shares_to_trade
                                    trade_history.append({'type': 'Sell', 'name': fake_name_traded, 'shares': shares_to_trade, 'price': price_at_trade, 'week': current_week})
                                    print(f"-> SOLD {shares_to_trade:.4f} shares of {fake_name_traded} for ${cost:,.2f}.")
                        else:
                            print("-> Error: Cannot trade zero or a negligible number of shares.")
            except (ValueError, IndexError):
                print(f"-> Error: Invalid command format for '{command}'. Example: 'Buy 10 QLIX'.")
        
        if current_week < len(simulation_df) and action != 'end':
            current_week += 1

    print("\n" + "#"*80 + "\nSIMULATION ENDED\n" + "#"*80)

    print(f"\nSimulation Period: {simulation_period_str}")
    print("Stock Identities:")
    for real, fake_info in ticker_map.items():
        print(f"  - {fake_info['name']} ({fake_info['ticker']}): {real}")

    final_portfolio_value = user_portfolio_value_history[-1] if user_portfolio_value_history else buying_power
    total_return_pct = ((final_portfolio_value / float(portfolio_value_str)) - 1) * 100
    print(f"\nYour Final Portfolio Value: ${final_portfolio_value:,.2f}")
    print(f"Total Return: {total_return_pct:+.2f}%")

    print("\n-> Comparing performance against market indices...")
    start_date_obj = datetime.strptime(simulation_period_str.split(' to ')[0], '%Y-%m-%d')
    end_date_obj = datetime.strptime(simulation_period_str.split(' to ')[1], '%Y-%m-%d')
    
    index_tickers = ['SPY', 'QQQ', 'DIA', 'IWM']
    index_hist_data = await asyncio.to_thread(yf.download, index_tickers, start=start_date_obj, end=end_date_obj, interval="1wk", progress=False)
    
    if not index_hist_data.empty and 'Close' in index_hist_data.columns:
        index_hist_data = index_hist_data['Close']
        index_performance = []
        for idx in index_tickers:
            if idx in index_hist_data.columns and not index_hist_data[idx].dropna().empty:
                idx_return = ((index_hist_data[idx].dropna().iloc[-1] / index_hist_data[idx].dropna().iloc[0]) - 1) * 100
                index_performance.append([idx, f"{idx_return:+.2f}%"])
        print(tabulate(index_performance, headers=["Index", "Return"], tablefmt="pretty"))
        if user_portfolio_value_history:
            plot_simulation_performance_graph(user_portfolio_value_history, index_hist_data, simulation_period_str, simulation_df.index)
    else:
        print("   -> Could not download index data for comparison.")

    stock_perf = []
    for real_ticker in selected_tickers:
        perf = ((simulation_df[real_ticker].iloc[-1] / simulation_df[real_ticker].iloc[0]) - 1) * 100
        stock_perf.append({'name': ticker_map[real_ticker]['name'], 'perf': perf})
    
    best_stock = max(stock_perf, key=lambda x: x['perf'])
    worst_stock = min(stock_perf, key=lambda x: x['perf'])
    print(f"\nBest Performing Stock in Simulation: {best_stock['name']} ({best_stock['perf']:+.2f}%)")
    print(f"Worst Performing Stock in Simulation: {worst_stock['name']} ({worst_stock['perf']:+.2f}%)")
    
    best_trade_profit = 0
    worst_trade_loss = 0
    best_trade_desc = "None"
    worst_trade_desc = "None"
    
    sells = sorted([t for t in trade_history if t['type'] == 'Sell'], key=lambda x: x['week'])
    temp_buys = [b.copy() for b in sorted([t for t in trade_history if t['type'] == 'Buy'], key=lambda x: x['week'])]

    for sell_trade in sells:
        shares_to_account_for = sell_trade['shares']
        total_cost_basis = 0
        
        for buy_trade in temp_buys:
            if buy_trade['name'] == sell_trade['name'] and buy_trade['week'] < sell_trade['week'] and buy_trade['shares'] > 1e-9:
                
                shares_to_use = min(shares_to_account_for, buy_trade['shares'])
                cost_basis_for_this_chunk = shares_to_use * buy_trade['price']
                total_cost_basis += cost_basis_for_this_chunk
                
                buy_trade['shares'] -= shares_to_use
                shares_to_account_for -= shares_to_use
                
                if shares_to_account_for <= 1e-9:
                    break
        
        if abs(shares_to_account_for) > sell_trade['shares'] * 0.001:
            continue

        total_sale_value = sell_trade['shares'] * sell_trade['price']
        total_profit = total_sale_value - total_cost_basis

        if worst_trade_loss == 0 and total_profit < 0:
            worst_trade_loss = total_profit
            worst_trade_desc = f"Selling {sell_trade['shares']:.2f} shares of {sell_trade['name']} for a total loss of ${abs(total_profit):,.2f}"

        if total_profit > best_trade_profit:
            best_trade_profit = total_profit
            best_trade_desc = f"Selling {sell_trade['shares']:.2f} shares of {sell_trade['name']} for a total profit of ${total_profit:,.2f}"
        
        if total_profit < worst_trade_loss:
            worst_trade_loss = total_profit
            worst_trade_desc = f"Selling {sell_trade['shares']:.2f} shares of {sell_trade['name']} for a total loss of ${abs(total_profit):,.2f}"

    print(f"\nBest Single Trade: {best_trade_desc}")
    print(f"Worst Single Trade: {worst_trade_desc}")
    
    print("\n-> Pausing for 10 seconds before returning to menu...")
    await asyncio.sleep(10)
