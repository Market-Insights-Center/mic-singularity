import robin_stocks.robinhood as r
import configparser
import time
# Add pyotp to imports if you are using the fully automated 2FA login
import pyotp
from typing import List, Dict, Any

# Load Configuration
config = configparser.ConfigParser()
config.read('config.ini')

def login_to_robinhood():
    """Logs into Robinhood using credentials and auto-generates 2FA token."""
    try:
        username = config.get('ROBINHOOD', 'RH_USERNAME', fallback=None)
        password = config.get('ROBINHOOD', 'RH_PASSWORD', fallback=None)
        mfa_secret = config.get('ROBINHOOD', 'RH_MFA_CODE', fallback=None)

        if not username or not password:
            print("❌ Error: Robinhood credentials missing.")
            return False

        print("🔐 Logging into Robinhood...")
        
        # Generate the token automatically if a secret is provided
        totp_code = None
        if mfa_secret:
            totp = pyotp.TOTP(mfa_secret)
            totp_code = totp.now()

        # Pass the generated code to the login function
        r.login(username, password, mfa_code=totp_code)
        print("✅ Login successful.")
        return True
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False

def get_robinhood_equity() -> float:
    """
    Logs in and fetches the total equity of the account.
    Returns 0.0 if retrieval fails.
    """
    if not login_to_robinhood():
        return 0.0
    
    try:
        # Fetch portfolio profile
        profile = r.profiles.load_portfolio_profile()
        if profile and 'equity' in profile:
            # Robinhood API returns strings, must cast to float
            return float(profile['equity'])
    except Exception as e:
        print(f"❌ Error fetching Robinhood equity: {e}")
    
    return 0.0

def execute_portfolio_rebalance(trades: List[Dict[str, Any]]):
    """
    Takes a list of calculated trades and executes them sequentially.
    Format: [{'ticker': 'AAPL', 'side': 'buy', 'quantity': 1.5}, ...]
    """
    if not trades:
        print("No trades to execute.")
        return

    print(f"\n--- 🏹 Robinhood Trade Execution ({len(trades)} orders) ---")
    
    confirm = input(f"⚠️  Are you sure you want to execute these {len(trades)} trades on Robinhood REAL MONEY account? (yes/no): ").lower().strip()
    if confirm != 'yes':
        print("🚫 Execution cancelled.")
        return

    if not login_to_robinhood():
        return

    print("\n🚀 Executing orders...")
    successful_trades = 0
    failed_trades = 0

    # Sort Sells first to free up cash before buying
    trades.sort(key=lambda x: x['side'] == 'buy') 

    for trade in trades:
        ticker = trade['ticker']
        raw_qty = float(trade['quantity'])
        side = trade['side']
        
        # --- FIX: ROUNDING TO 6 DECIMAL PLACES ---
        # Robinhood API rejects orders with >8 decimal places. 
        # Python floats can often result in 0.20000000000000004.
        qty = round(raw_qty, 6) 

        if qty <= 0: continue

        try:
            print(f"   Processing: {side.upper()} {qty} {ticker}...", end=" ")
            
            order = None
            if side == 'buy':
                order = r.orders.order_buy_fractional_by_quantity(ticker, qty)
            elif side == 'sell':
                order = r.orders.order_sell_fractional_by_quantity(ticker, qty)
            
            if order and 'id' in order:
                print(f"✅ Order Placed (ID: {order['id']})")
                successful_trades += 1
            elif order and 'detail' in order:
                print(f"❌ Failed: {order['detail']}")
                failed_trades += 1
            elif order and 'non_field_errors' in order:
                # Catch generic non-field errors often returned by RH
                print(f"❌ Failed: {order['non_field_errors']}")
                failed_trades += 1
            else:
                # Fallback for unknown error structures
                print(f"⚠️  Unknown response: {order}")
                failed_trades += 1
                
            time.sleep(2) 
            
        except Exception as e:
            print(f"❌ Error executing {ticker}: {e}")
            failed_trades += 1

    print("-" * 50)
    print(f"Execution Complete. Success: {successful_trades} | Failed: {failed_trades}")
    print("-" * 50)
    
    r.logout()