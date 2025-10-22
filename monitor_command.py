# --- Imports for monitor_command ---
import asyncio
import os
import csv
import smtplib
import configparser
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Dict, Optional
from collections import defaultdict

import yfinance as yf
import pandas as pd
from tabulate import tabulate

# --- Imports from other command modules ---
from invest_command import calculate_ema_invest
from tracking_command import _load_portfolio_run

# --- Module-level Globals & Constants ---
active_alerts: List[Dict] = []
alert_lock = asyncio.Lock()
ALERTS_FILE = 'alerts.csv'

# --- Configuration for Notifications ---
config = configparser.ConfigParser()
config.read('config.ini')

# --- Helper Functions (moved or copied for self-containment) ---

async def send_notification(subject: str, body: str, recipient_email_override: Optional[str] = None):
    """Sends an email notification."""
    try:
        smtp_server = config.get('EMAIL_CONFIG', 'SMTP_SERVER')
        smtp_port = config.getint('EMAIL_CONFIG', 'SMTP_PORT')
        sender_email = config.get('EMAIL_CONFIG', 'SENDER_EMAIL')
        sender_password = config.get('EMAIL_CONFIG', 'SENDER_PASSWORD')
        recipient = recipient_email_override or config.get('EMAIL_CONFIG', 'RECIPIENT_EMAIL', fallback=None)

        if not all([smtp_server, smtp_port, sender_email, sender_password, recipient]):
            print("⚠️ Email config incomplete. Cannot send notification.")
            return

        msg = MIMEMultipart()
        msg['From'], msg['To'], msg['Subject'] = sender_email, recipient, subject
        msg.attach(MIMEText(body, 'plain'))

        # Define the synchronous function to be run in a thread
        def _send_email_sync():
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()  # Secure the connection
                server.login(sender_email, sender_password) # Login
                server.send_message(msg) # Send the email
        
        # Run the blocking email code in a separate thread
        await asyncio.to_thread(_send_email_sync)
        print(f"✔ Email notification sent successfully to {recipient}.")
    except Exception as e:
        print(f"❌ Failed to send email notification: {e}")
        
def save_alerts_to_csv():
    """Saves the current state of active_alerts to the CSV file."""
    try:
        with open(ALERTS_FILE, mode='w', newline='', encoding='utf-8') as f:
            # Add 'portfolio_code' to the list of field names
            fieldnames = ['ticker', 'metric', 'operator', 'value', 'sensitivity', 'recipient_email', 'portfolio_code']
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(active_alerts)
    except Exception as e:
        print(f"Error saving alerts to {ALERTS_FILE}: {e}")

async def load_alerts_from_csv():
    """Loads alerts from the CSV file into memory at startup."""
    if not os.path.exists(ALERTS_FILE): return
    try:
        with open(ALERTS_FILE, mode='r', newline='', encoding='utf-8') as f:
            temp_alerts = []
            for row in csv.DictReader(f):
                try:
                    row['value'] = float(row['value'])
                    row['sensitivity'] = int(row['sensitivity']) if row.get('sensitivity') else None
                    temp_alerts.append(row)
                except (ValueError, KeyError):
                    continue
            async with alert_lock:
                global active_alerts
                active_alerts = temp_alerts
            if active_alerts: print(f"✅ Loaded {len(active_alerts)} alert(s) from {ALERTS_FILE}.")
    except Exception as e:
        print(f"Error loading alerts from {ALERTS_FILE}: {e}")

async def alert_worker():
    """
    A continuous background task that checks active alerts every 10 seconds.
    """
    print("🚀 Real-time Alert Worker has been started.")
    while True:
        await asyncio.sleep(10)  # Check every 10 seconds
        
        alerts_to_check = []
        triggered_indices = []

        async with alert_lock:
            if not active_alerts:
                continue
            alerts_to_check = list(active_alerts)

        # --- Efficiently check all PRICE alerts first ---
        price_alerts = [(i, a) for i, a in enumerate(alerts_to_check) if a['metric'] == 'price']
        if price_alerts:
            tickers_to_fetch_price = list(set([alert['ticker'] for _, alert in price_alerts]))
            try:
                data = await asyncio.to_thread(yf.download, tickers=tickers_to_fetch_price, period="1d", progress=False, auto_adjust=False)
                close_prices = data.get('Close')
                if close_prices is not None:
                    for index, alert in price_alerts:
                        current_price = None
                        if isinstance(close_prices, pd.DataFrame):
                            current_price = close_prices[alert['ticker']].iloc[-1] if alert['ticker'] in close_prices.columns else None
                        elif isinstance(close_prices, pd.Series):
                            current_price = close_prices.iloc[-1]

                        if current_price is None or pd.isna(current_price): continue
                        
                        op, val = alert['operator'], alert['value']
                        if (op == '>' and current_price > val) or (op == '<' and current_price < val) or \
                           (op == '>=' and current_price >= val) or (op == '<=' and current_price <= val):
                            print("\n" + "!"*80 + f"\n🔔 PRICE ALERT TRIGGERED! 🔔\n   Ticker:    {alert['ticker']}\n   Condition: Price {op} {val}\n   Live Price:  ${current_price:,.2f}\n" + "!"*80)
                            print(f"\nEnter command: ", end="", flush=True)
                            
                            recipient = alert.get('recipient_email')
                            if recipient:
                                subject = f"M.I.C. Singularity Price Alert: {alert['ticker']}"
                                body = (f"A price alert for {alert['ticker']} has been triggered.\n\n"
                                        f"Details:\n - Ticker: {alert['ticker']}\n"
                                        f" - Condition: Price {op} {val}\n"
                                        f" - Current Price: ${current_price:,.2f}\n\n"
                                        f"This alert has now been removed from the active list.")
                                await send_notification(subject, body, recipient_email_override=recipient)

                            triggered_indices.append(index)
            except Exception: pass

        # --- Check all INVEST score alerts individually ---
        invest_alerts = [(i, a) for i, a in enumerate(alerts_to_check) if a['metric'] == 'invest']
        if invest_alerts:
            for index, alert in invest_alerts:
                try:
                    _, current_score = await calculate_ema_invest(alert['ticker'], alert['sensitivity'], is_called_by_ai=True)
                    if current_score is None: continue

                    op, val = alert['operator'], alert['value']
                    if (op == '>' and current_score > val) or (op == '<' and current_score < val) or \
                       (op == '>=' and current_score >= val) or (op == '<=' and current_score <= val):
                        print("\n" + "!"*80 + f"\n🔔 INVEST SCORE ALERT TRIGGERED! 🔔\n   Ticker:      {alert['ticker']}\n   Condition:   INVEST Score (Sens: {alert['sensitivity']}) {op} {val}\n   Live Score:  {current_score:,.2f}%\n" + "!"*80)
                        print(f"\nEnter command: ", end="", flush=True)

                        recipient = alert.get('recipient_email')
                        if recipient:
                            subject = f"M.I.C. Singularity Invest Score Alert: {alert['ticker']}"
                            body = (f"An INVEST score alert for {alert['ticker']} has been triggered.\n\n"
                                    f"Details:\n - Ticker: {alert['ticker']}\n"
                                    f" - Sensitivity: {alert['sensitivity']}\n"
                                    f" - Condition: INVEST Score {op} {val}\n"
                                    f" - Current Score: {current_score:,.2f}%\n\n"
                                    f"This alert has now been removed from the active list.")
                            await send_notification(subject, body, recipient_email_override=recipient)

                        triggered_indices.append(index)
                except Exception: pass

        # --- Check all P&L alerts ---
        pnl_alerts = [(i, a) for i, a in enumerate(alerts_to_check) if a['metric'] == 'pnl']
        if pnl_alerts:
            # Group alerts by portfolio code to minimize file loading
            alerts_by_portfolio = defaultdict(list)
            for index, alert in pnl_alerts:
                alerts_by_portfolio[alert['portfolio_code']].append((index, alert))

            for portfolio_code, alerts in alerts_by_portfolio.items():
                try:
                    old_run_data = await _load_portfolio_run(portfolio_code) # From tracking_command.py
                    if not old_run_data: continue

                    tickers_to_fetch = [row['Ticker'] for row in old_run_data if row.get('Ticker') != 'Cash']
                    
                    # Fetch both opening and live prices in one call
                    data = await asyncio.to_thread(yf.download, tickers=tickers_to_fetch, period="1d", interval="1m", progress=False, auto_adjust=False)
                    
                    if data is None or data.empty: continue
                    
                    # Get the most recent open prices
                    open_prices = data['Open'].iloc[0] if isinstance(data['Open'], pd.DataFrame) else data['Open']
                    # Get the most recent live prices (close of the last minute)
                    live_prices = data['Close'].iloc[-1] if isinstance(data['Close'], pd.DataFrame) else data['Close']

                    opening_value = sum(
                        float(row.get('Shares', 0)) * open_prices.get(row['Ticker'], 0)
                        for row in old_run_data if row.get('Ticker') != 'Cash'
                    )
                    current_value = sum(
                        float(row.get('Shares', 0)) * live_prices.get(row['Ticker'], 0)
                        for row in old_run_data if row.get('Ticker') != 'Cash'
                    )
                    
                    daily_pnl = current_value - opening_value
                    
                    for index, alert in alerts:
                        op, val = alert['operator'], alert['value']
                        # Check if the P&L crosses the specified interval
                        if (op == '>' and daily_pnl > val) or (op == '<' and daily_pnl < val) or \
                           (op == '>=' and daily_pnl >= val) or (op == '<=' and daily_pnl <= val):
                            print("\n" + "!"*80 + f"\n📈 DAILY P&L ALERT TRIGGERED! 📈\n   Portfolio: {portfolio_code}\n   Condition: Daily P&L {op} {val:,.2f}\n   Live P&L:  ${daily_pnl:,.2f}\n" + "!"*80)
                            print(f"\nEnter command: ", end="", flush=True)

                            recipient = alert.get('recipient_email')
                            if recipient:
                                subject = f"M.I.C. Singularity P&L Alert: {portfolio_code}"
                                body = (f"A daily P&L alert for portfolio '{portfolio_code}' has been triggered.\n\n"
                                        f"Details:\n - Portfolio: {portfolio_code}\n"
                                        f" - Current Portfolio Value: ${current_value:,.2f}\n"
                                        f" - Current Daily P&L: ${daily_pnl:,.2f}\n"
                                        f" - Triggered Condition: Daily P&L {op} {val:,.2f}\n\n"
                                        f"This alert has now been removed from the active list.")
                                await send_notification(subject, body, recipient_email_override=recipient)
                            
                            triggered_indices.append(index)
                except Exception as e:
                    print(f"❌ Error during P&L check for portfolio {portfolio_code}: {e}")
                    pass


        # Safely remove all triggered alerts from the main list
        if triggered_indices:
            async with alert_lock:
                # Use set to ensure unique indices before popping
                for index in sorted(list(set(triggered_indices)), reverse=True):
                    if index < len(active_alerts):
                        active_alerts.pop(index)
                # After removing, save the updated list
                save_alerts_to_csv()
                
# --- Main Command Handler ---
async def handle_monitor_command(args: list, is_called_by_ai: bool = False):
    """
    Manages real-time monitoring alerts. Now accepts an optional recipient email.
    """
    if is_called_by_ai:
        return "This command is interactive and designed for direct user use in the CLI."

    if not args:
        print("Usage: /monitor <add|list|remove> [options]")
        return

    action = args[0].lower()
    async with alert_lock:
        if action == "add":
            # --- New logic to parse optional email ---
            recipient_email = None
            command_args = args
            if len(args) > 2 and args[-2].lower() == 'to':
                recipient_email = args[-1]
                command_args = args[:-2] # Remove 'to <email>' for further parsing
            
            if len(command_args) < 5:
                print("Usage: /monitor add <TICKER> price <op> <value> [to <email>]")
                print("       /monitor add <TICKER> invest <sens> <op> <value> [to <email>]")
                print("       /monitor add <PORTFOLIO_CODE> pnl <op> <value> [to <email>]")
                return

            identifier = command_args[1].upper()
            metric = command_args[2].lower()
            alert_to_add = None

            if metric == 'price':
                if len(command_args) != 5:
                    print("Usage: /monitor add <TICKER> price <op> <value> [to <email>]")
                    return
                try:
                    operator = command_args[3]
                    value = float(command_args[4])
                    if operator not in ['>', '<', '>=', '<=']:
                        print(f"Error: Invalid operator '{operator}'.")
                        return
                    alert_to_add = {"ticker": identifier, "metric": "price", "operator": operator, "value": value, "sensitivity": None, "recipient_email": recipient_email}
                except ValueError:
                    print("Error: The value for the alert must be a valid number.")

            elif metric == 'invest':
                if len(command_args) != 6:
                    print("Usage: /monitor add <TICKER> invest <sens> <op> <value> [to <email>]")
                    return
                try:
                    sensitivity = int(command_args[3])
                    operator = command_args[4]
                    value = float(command_args[5])
                    if sensitivity not in [1, 2, 3]:
                        print("Error: Invalid sensitivity. Use 1, 2, or 3.")
                        return
                    if operator not in ['>', '<', '>=', '<=']:
                        print(f"Error: Invalid operator '{operator}'.")
                        return
                    alert_to_add = {"ticker": identifier, "metric": "invest", "sensitivity": sensitivity, "operator": operator, "value": value, "recipient_email": recipient_email}
                except ValueError:
                    print("Error: Sensitivity and value must be valid numbers.")
            
            elif metric == 'pnl':
                if len(command_args) != 5:
                    print("Usage: /monitor add <PORTFOLIO_CODE> pnl <op> <value> [to <email>]")
                    return
                try:
                    operator = command_args[3]
                    value = float(command_args[4])
                    if operator not in ['>', '<', '>=', '<=']:
                        print(f"Error: Invalid operator '{operator}'.")
                        return
                    alert_to_add = {"portfolio_code": identifier, "metric": "pnl", "operator": operator, "value": value, "recipient_email": recipient_email}
                except ValueError:
                    print("Error: The value for the P&L interval must be a valid number.")
            
            else:
                print(f"Error: Invalid metric '{metric}'. Use 'price', 'invest', or 'pnl'.")

            if alert_to_add:
                active_alerts.append(alert_to_add)
                save_alerts_to_csv()
                if recipient_email:
                    print(f"✅ Alert added and saved. Email notification will be sent to {recipient_email}.")
                else:
                    print(f"✅ Alert added and saved. Notification will be printed to the terminal only (no email).")

        elif action == "list":
            if not active_alerts:
                print("No active alerts to display.")
                return
            print("\n--- Active Alerts ---")
            table_data = []
            for i, alert in enumerate(active_alerts):
                condition_str = ""
                identifier = alert.get('ticker') or alert.get('portfolio_code')
                
                if alert['metric'] == 'price':
                    condition_str = f"price {alert['operator']} {alert['value']}"
                elif alert['metric'] == 'invest':
                    condition_str = f"INVEST (sens: {alert.get('sensitivity', 'N/A')}) {alert['operator']} {alert['value']}"
                elif alert['metric'] == 'pnl':
                    condition_str = f"P&L {alert['operator']} {alert['value']}"
                
                recipient_display = alert.get('recipient_email') or "Terminal Only"
                table_data.append([i + 1, identifier, condition_str, recipient_display])
            print(tabulate(table_data, headers=["Index", "Identifier", "Condition", "Recipient"], tablefmt="pretty"))

        elif action == "remove":
            if len(args) < 2:
                print("Usage: /monitor remove <index>")
                return
            try:
                index_to_remove = int(args[1]) - 1
                if 0 <= index_to_remove < len(active_alerts):
                    active_alerts.pop(index_to_remove)
                    save_alerts_to_csv()  # Save the list immediately after manual removal
                    print(f"✅ Alert at index {args[1]} removed and file updated.")
                else:
                    print("Error: Invalid index. Use '/monitor list' to see active alerts.")
            except ValueError:
                print("Error: Please provide a valid number for the index.")
        else:
            print(f"Unknown monitor command: {action}. Use 'add', 'list', or 'remove'.")