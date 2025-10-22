# --- Imports for spear_command ---
import asyncio
import uuid
import math
from math import sqrt
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate
import fear_and_greed
import humanize
from nltk.tokenize import sent_tokenize
from sentiment_command import handle_sentiment_command # type:ignore

# --- Helper Functions (copied or moved for self-containment) ---

def ask_singularity_input(prompt: str, validation_fn=None, error_msg="Invalid input.", default_val=None, is_called_by_ai: bool = False) -> Optional[str]:
    """Helper function to ask for user input in Singularity CLI."""
    if is_called_by_ai:
        return None
    while True:
        full_prompt = f"{prompt}"
        if default_val is not None:
            full_prompt += f" (default: {default_val}, press Enter)"
        full_prompt += ": "
        user_response = input(full_prompt).strip()
        if not user_response and default_val is not None:
            return str(default_val)
        if validation_fn:
            if validation_fn(user_response):
                return user_response
            else:
                print(error_msg)
        else:
            return user_response

def fetch_yahoo_finance_data_spear(ticker):
    """Fetches key financial metrics for a ticker."""
    stock = yf.Ticker(ticker)
    info = stock.info
    today = datetime.today()
    # MODIFICATION: Added auto_adjust=False to all yf.download calls to suppress warnings and ensure consistency.
    data_1d = yf.download(ticker, start=today - timedelta(days=2), progress=False, timeout=10, auto_adjust=False)
    data_1m = yf.download(ticker, start=today - timedelta(days=31), progress=False, timeout=10, auto_adjust=False)
    data_6m = yf.download(ticker, start=today - timedelta(days=181), progress=False, timeout=10, auto_adjust=False)
    spy_1d = yf.download("SPY", start=today - timedelta(days=2), progress=False, timeout=10, auto_adjust=False)
    
    # MODIFICATION: Wrapped calculations in float() to prevent pandas Series ambiguity errors.
    return {
        '1D% Change': float(data_1d['Close'].iloc[-1] / data_1d['Open'].iloc[0] - 1) if not data_1d.empty else 0.0,
        '1M% Change': float(data_1m['Close'].iloc[-1] / data_1m['Open'].iloc[0] - 1) if not data_1m.empty else 0.0,
        '3M% Change': float(data_6m['Close'].iloc[-1] / data_6m['Open'].iloc[0] - 1) if not data_6m.empty else 0.0,
        '1Y% Change': float(info.get('52WeekChange', 0.0)),
        'SPY 1D% Change': float(spy_1d['Close'].iloc[-1] / spy_1d['Open'].iloc[0] - 1) if not spy_1d.empty else 0.0,
        'Market Cap In Billions of USD': info.get('marketCap', 0)
    }

def plot_spear_graph(ticker, recommended_price, is_called_by_ai: bool = False):
    """Generates and saves the SPEAR forecast graph."""
    # MODIFICATION: Added auto_adjust=False to suppress warnings.
    data = yf.download(ticker, period="1y", progress=False, timeout=15, auto_adjust=False)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price', color='grey')
    if recommended_price is not None:
        ax.axhline(y=recommended_price, color='r', linestyle='--', label=f'Predicted Price: ${recommended_price:.2f}')
    ax.set_title(f'{ticker.upper()} Stock Price - Past Year', color='white')
    ax.legend()
    ax.grid(True, alpha=0.3)
    filename = f"spear_graph_{ticker.replace('.','-')}_{uuid.uuid4().hex[:6]}.png"
    plt.savefig(filename, facecolor='black')
    plt.close(fig)
    if not is_called_by_ai: print(f"SPEAR graph saved: {filename}")
    return filename

def business_summary_spear(ticker, business_summary_length_sentences):
    """Fetches and formats a company's business summary."""
    info = yf.Ticker(ticker).info
    summary_raw = info.get('longBusinessSummary', 'N/A')
    market_cap_formatted = humanize.intword(info.get('marketCap')) if info.get('marketCap') else 'N/A'
    summary_trim = ' '.join(sent_tokenize(summary_raw)[:business_summary_length_sentences])
    return summary_trim, info.get('industry', 'N/A'), info.get('sector', 'N/A'), market_cap_formatted

def calculate_spear_prediction(ticker, sector, relevance, hype, fear, earnings_date_str, earnings_time, trend, reversal, meme_stock, trade, actual_price):
    finance_data = fetch_yahoo_finance_data_spear(ticker)
    one_month_chg_per = finance_data['1M% Change']
    six_month_chg_per = finance_data['3M% Change']
    one_year_chg_per = finance_data['1Y% Change']
    spy_one_day_chg_per = finance_data['SPY 1D% Change']
    market_cap = finance_data['Market Cap In Billions of USD']

    def fetch_options_data(ticker_opt: str):
        stock = yf.Ticker(ticker_opt)
        today = datetime.today().date()
        try:
            exp_dates = [datetime.strptime(date, '%Y-%m-%d').date() for date in stock.options]
            future_exp_dates = [d for d in exp_dates if d > today]
            if not future_exp_dates: raise ValueError("No future expiration dates.")
            next_exp_date = min(future_exp_dates)
            opt_chain = stock.option_chain(next_exp_date.strftime('%Y-%m-%d'))
            pcr = opt_chain.puts['openInterest'].sum() / opt_chain.calls['openInterest'].sum() if opt_chain.calls['openInterest'].sum() != 0 else 0
            avg_iv = (opt_chain.puts['impliedVolatility'].mean() + opt_chain.calls['impliedVolatility'].mean()) / 2 * 100
            return {"put_call_ratio": pcr, "average_implied_volatility": avg_iv}
        except Exception:
            return {"put_call_ratio": 1, "average_implied_volatility": 0}

    c1 = "1" if one_year_chg_per < 0.5 else "2" if one_year_chg_per < 0.75 else "3" if one_year_chg_per < 1 else "4" if one_year_chg_per < 1.5 else "5"
    c2 = "5" if market_cap < 5e9 else "3" if market_cap < 2.5e10 else "2" if market_cap < 5e10 else "3" if market_cap < 1e11 else "4" if market_cap < 2e11 else "5"
    c3 = sector
    c4 = relevance
    c5 = ((float(c1) + float(c2) + float(c3) + float(c4)) + hype)/4
    c6 = math.sqrt(abs((c5**2) - 1)) if c5 < 3 else math.sqrt(abs((c5**2) + 1)) if c5 > 3 else 3.25
    c7 = c6 * 1.25 if c4 == "5" else c6
    c8 = c7 - 3
    c9 = c8 * -c8 if c8 < 0 else c8**2
    c10 = c9 / 3 * 0.1 if market_cap > 1e11 else c9 * 0.1
    c11 = one_month_chg_per*100/30
    days_difference = (datetime.strptime(earnings_date_str, "%Y-%m-%d") - datetime.today()).days + (1 if earnings_time == 'a' else 0)
    c12 = int(days_difference) * c11
    c13 = ((((spy_one_day_chg_per * 100) - 0.12)) * c10) + c10
    # MODIFICATION: Added auto_adjust=False to suppress warnings.
    vix_data = yf.download("^VIX", start=(datetime.today() - pd.Timedelta(days=2)).strftime("%Y-%m-%d"), progress=False, auto_adjust=False)
    vix_live = vix_data['Open'].values[-1] if not vix_data.empty else 17.0
    c14 = "-1" if vix_live < 13.5 else "-0.5" if vix_live < 15.25 else "0" if vix_live < 17.0 else "0.5" if vix_live < 18.75 else "1"
    c15 = ((0.5 * float(c14) * float(c13)) + float(c13))
    uncertain = 1 if trend in ["No Trend", "n"] else 0
    c16 = "3" if fear == 50 else "2" if trend in ["Upwards","u"] else "1" if trend in ["Downwards","d"] else "0" if trend == "Use Stock" else "-1" if fear > 55 else "-2" if market_cap >= 1e11 else "-3" if uncertain == 1 else "-4"
    c17 = float(-1*c15) if c16=="3" and trend in ["Upwards","u"] else float(c15) if c16=="3" else float(-1*(((((fear-50)/50))*c15)-c15)) if c16=="2" else float(-1*(((((fear-50)/50))*c15)+c15)) if c16=="1" else float(((((fear-50)/50))*c15)+c15) if c16=="0" and (one_month_chg_per>=0.1) else float(((((fear-50)/50))*c15)-c15) if c16=="0" and (one_month_chg_per<=0.1) else float(((((fear-50)/50))*c15)+c15) if c16=="-1" else float(((((fear-50)/50))*c15)-c15) if c16=="-2" else float(c15) if c16=="-3" else float(((((fear-50)/50))*c15)+c15)
    c18 = ((c17+c15)/4) if uncertain==1 and fear==50 and (one_month_chg_per>=0.1) else ((c17-c15)/4) if uncertain==1 and fear==50 and (one_month_chg_per<=0.1) else (((c17+(float(-1*(((fear-50)/50))))*(3*c15))+c15)/4) if uncertain==1 and (one_month_chg_per>=0.1) else (((c17+(float(-1*(((fear-50)/50))))*(3*c15))-c15)/4) if uncertain==1 and (one_month_chg_per<=0.1) else float(c17)
    c19 = c18 * 3 if uncertain == 1 and -0.05 < c18 < 0.05 else float(c18)
    c20 = c19 * 2 if market_cap < 5e9 and -0.05 < c19 < 0.05 else float(c19)
    c21 = sqrt(abs(c20))*-2 if reversal in ["Yes","y"] and c20>0 else sqrt(abs(c20*-1))*2 if reversal in ["Yes","y"] and c20<0 else float(c20)
    price_in_value = 1 if six_month_chg_per > 0.3 or six_month_chg_per < 0.1 else 0
    price_in_hype = 1 if abs(hype) > 0.56 else 0
    price_in_affirmation = 0 if price_in_value == 1 and price_in_hype == 1 else 1
    c22 = -1 if price_in_affirmation == 1 else 1
    c23 = c21 * float(c22) if price_in_affirmation == 1 else float(c21)
    c24 = (10*c23)+c23 if price_in_affirmation==1 and 0<c23<=0.005 else (10*c23)-c23 if price_in_affirmation==1 and -0.005<=c23<0 else (5*c23)+c23 if price_in_affirmation==1 and 0<c23<=0.01 else (5*c23)-c23 if price_in_affirmation==1 and -0.01<=c23<0 else (2*c23)+c23 if price_in_affirmation==1 and 0<c23<=0.5 else (2*c23)-c23 if price_in_affirmation==1 and -0.5<=c23<0 else (1.5*c23)+c23 if price_in_affirmation==1 and 0<c23<=0.1 else (1.5*c23)-c23 if price_in_affirmation==1 and -0.1<=c23<0 else float(c23)
    c25 = (c24*2) if abs(hype)>0.5 and -0.05<c23<0.05 and price_in_affirmation==1 else (c24*1.5) if abs(hype)>0.5 and -0.1<c23<0.1 and price_in_affirmation==1 else float(c24)
    c26 = c25*20 if -0.01<=c25<=0.01 and meme_stock in ["Yes","y"] else c25*10 if -0.025<=c25<=0.025 and meme_stock in ["Yes","y"] else c25*5 if -0.05<=c25<=0.05 and meme_stock in ["Yes","y"] else float(c25)
    c27 = c26 * 7.5 if -0.025 <= c26 <= 0.025 else float(c26)
    c28 = c27/5 if abs(c27)>=0.3 and meme_stock in ["No","n"] else float(c27)
    c29 = c28/8 if abs(c28)>=1 else c28/5 if abs(c28)>=0.75 else c28/2 if abs(c28)>=0.4 else float(c28)
    c30 = c29 * (((50-fear)*8/100)+1) if fear<=55 else c29 * (((50-fear)*4/100)+1) if fear<=60 else c29*(((50-fear)*2/100)+1)
    result_options = fetch_options_data(ticker)
    put_call, iv = result_options['put_call_ratio'], result_options['average_implied_volatility']
    c31 = ((iv-300)/-600) if iv>=300 and market_cap>=2e10 else c30
    c32 = (((iv-150)/-600)-0.1) if iv>=150 and c30<0 else (((iv-150)/600)+0.1) if iv>=150 and c30>=0 else c30
    c33 = (iv/-1200) if c30<0 else (iv/1200)
    c34 = ((put_call-2)/-25) if put_call>2 and c30<0 else ((put_call-2)/25) if put_call>2 and c30>=0 else c30
    c35 = abs(one_month_chg_per-(one_year_chg_per/12)) if meme_stock not in ["Yes","y"] and c30>=0 else (-1*abs(one_month_chg_per-(one_year_chg_per/12))) if meme_stock not in ["Yes","y"] and c30<0 else c30
    c36 = (c31+c32+c34)/3
    c37 = (c30+c33+c36)/3
    c38 = (c35+c37)/2 if -0.4 < c35 < 0.4 else c37
    c39 = 0.15 if c38>0.35 and market_cap>1e11 else -0.15 if c38<-0.35 and market_cap>1e11 else 0.35 if c38>0.35 else -0.35 if c38<-0.35 else c38

    return {
        'price_in': price_in_affirmation, 'prediction': float(c39),
        'prediction_in_time': float(c12 / 100), 'finance_data': finance_data
    }

# --- Main Command Handler ---

# --- Replace the existing handle_spear_command function with this one ---

async def handle_spear_command(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False):
    """Handles the /spear command for CLI and AI, now with automated hype scoring."""
    if not is_called_by_ai:
        print("\n--- /spear Command ---")

    params = {}
    try:
        if is_called_by_ai and ai_params:
            params = ai_params.copy()

            if not params.get('ticker'):
                return {"error": "Ticker is a required parameter for SPEAR analysis."}
            
            params['ticker'] = params.get('ticker', '').replace(".", "-")

            # Intelligently load from databases if info is not already provided by AI
            try:
                spear_bank_df = pd.read_csv('spear_bank.csv')
                stock_data = spear_bank_df[spear_bank_df['Ticker'].str.lower() == params['ticker'].lower()]
                if not stock_data.empty:
                    params.setdefault('sector_relevance', float(stock_data['Sector to Market'].iloc[0]))
                    params.setdefault('stock_relevance', float(stock_data['Stock to Sector'].iloc[0]))
                    params.setdefault('is_meme_stock', stock_data['Meme Stock'].iloc[0])
            except (FileNotFoundError, KeyError):
                pass

            try:
                spear_trend_df = pd.read_csv('spear_trend.csv')
                if not spear_trend_df.empty:
                    params.setdefault('market_trend', spear_trend_df['Market Trend'].iloc[0])
                    params.setdefault('market_reversal_likely', spear_trend_df['Reversal Likely'].iloc[0])
            except (FileNotFoundError, KeyError):
                pass

            # Handle natural language variations for AI
            if 'earnings_date' in params and params['earnings_date'].lower() == 'tomorrow':
                params['earnings_date'] = (datetime.now() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            if 'earnings_time' in params:
                time_str = params['earnings_time'].lower()
                if 'after' in time_str: params['earnings_time'] = 'a'
                elif 'pre' in time_str or 'before' in time_str: params['earnings_time'] = 'p'
            
            if 'market_trend' in params:
                trend_str = params['market_trend'].lower()
                if 'upward' in trend_str: params['market_trend'] = 'Upwards'
                elif 'downward' in trend_str: params['market_trend'] = 'Downwards'
                elif 'no' in trend_str: params['market_trend'] = 'No Trend'

            # Final check for any remaining missing required parameters (hype is now auto-generated)
            required_keys = ['ticker', 'sector_relevance', 'stock_relevance', 'earnings_date', 'earnings_time']
            missing_keys = [key for key in required_keys if key not in params]
            if missing_keys:
                return f"Error: Missing required parameters for SPEAR analysis: {', '.join(missing_keys)}. Please provide them."

        else: # CLI path
            ticker_in = ask_singularity_input('Enter the ticker', validation_fn=lambda x: x.strip(), is_called_by_ai=is_called_by_ai)
            if not ticker_in: return
            params['ticker'] = ticker_in.replace(".", "-")
            
            stock = yf.Ticker(params['ticker'])
            market_cap = stock.info.get('marketCap', 0)

            try:
                spear_bank_df = pd.read_csv('spear_bank.csv')
                stock_data = spear_bank_df[spear_bank_df['Ticker'].str.lower() == params['ticker'].lower()]
                if not stock_data.empty:
                    params['sector_relevance'] = float(stock_data['Sector to Market'].iloc[0])
                    params['stock_relevance'] = float(stock_data['Stock to Sector'].iloc[0])
                    params['is_meme_stock'] = stock_data['Meme Stock'].iloc[0]
                    print("Info: Found database entry in 'spear_bank.csv'. Auto-filling some inputs.")
            except FileNotFoundError:
                pass 

            try:
                spear_trend_df = pd.read_csv('spear_trend.csv')
                if not spear_trend_df.empty:
                    params['market_trend'] = spear_trend_df['Market Trend'].iloc[0]
                    params['market_reversal_likely'] = spear_trend_df['Reversal Likely'].iloc[0]
                    print("Info: Found database entry in 'spear_trend.csv'. Auto-filling trend inputs.")
            except FileNotFoundError:
                pass

            if 'sector_relevance' not in params:
                params['sector_relevance'] = float(ask_singularity_input('Enter The Sector To Market Relevance Number (1 to 5)', lambda x: 1<=float(x)<=5, is_called_by_ai=is_called_by_ai))
            if 'stock_relevance' not in params:
                params['stock_relevance'] = float(ask_singularity_input('Enter The Stock To Sector Relevance Number (1 to 5)', lambda x: 1<=float(x)<=5, is_called_by_ai=is_called_by_ai))
            
            if 'is_meme_stock' not in params:
                 params['is_meme_stock'] = ask_singularity_input('Is it a meme stock? (Yes/y or No/n)', default_val="No", is_called_by_ai=is_called_by_ai) if market_cap < 5e10 else "No"
            
            if 'market_trend' not in params:
                params['market_trend'] = ask_singularity_input('What Is The Market Trend (Upwards/u, Downwards/d, No Trend/n)', default_val="No Trend", is_called_by_ai=is_called_by_ai) if params['is_meme_stock'].lower() in ["no", "n"] else "No Trend"

            if 'market_reversal_likely' not in params:
                 params['market_reversal_likely'] = ask_singularity_input('Is A Market Reversal Likely (Yes/y or No/n)', default_val="No", is_called_by_ai=is_called_by_ai) if params['market_trend'].lower() in ["no trend", "n"] and params['is_meme_stock'].lower() in ["no", "n"] else "No"

            if 'earnings_date' not in params:
                params['earnings_date'] = ask_singularity_input('Enter The Earnings Date (YYYY-MM-DD)', lambda x: datetime.strptime(x, "%Y-%m-%d"), is_called_by_ai=is_called_by_ai)
            if 'earnings_time' not in params:
                params['earnings_time'] = ask_singularity_input('Enter The Earnings Time (p for Pre-Market, a for After Hours)', lambda x: x.lower() in ['p', 'a'], is_called_by_ai=is_called_by_ai)

    except (ValueError, TypeError, AttributeError) as e:
        err_msg = f"Error: Invalid input provided. {e}"
        if not is_called_by_ai: print(err_msg)
        return {"error": err_msg} if is_called_by_ai else None

    # --- Automated Hype Score Generation ---
    if not is_called_by_ai:
        print("\n-> Running automated sentiment analysis to determine hype score...")
    
    sentiment_result = await handle_sentiment_command(
        ai_params={'ticker': params['ticker']}, 
        is_called_by_ai=True
    )
    
    if sentiment_result and isinstance(sentiment_result, dict) and 'sentiment_score_raw' in sentiment_result:
        params['hype'] = sentiment_result['sentiment_score_raw']
        if not is_called_by_ai:
            print(f"-> Automated Hype Score: {params['hype']:.4f}")
    else:
        params['hype'] = 0.0 # Default to neutral if sentiment analysis fails
        if not is_called_by_ai:
            print("-> Warning: Could not determine sentiment. Defaulting hype score to 0.0.")

    # --- Core Logic ---
    tckr = yf.Ticker(params['ticker'])
    hist_price = tckr.history(period="1d", auto_adjust=False)
    actual_price = hist_price['Close'].iloc[-1] if not hist_price.empty else None

    if not actual_price:
        err_msg = f"Error: Could not fetch live price for {params['ticker']}."
        if not is_called_by_ai: print(err_msg)
        return {"error": err_msg} if is_called_by_ai else None
        
    fear_and_greed_data = fear_and_greed.get()
    fear_value = round(fear_and_greed_data[0])
    
    prediction_data = calculate_spear_prediction(
        params['ticker'], params['sector_relevance'], params['stock_relevance'], params['hype'],
        fear_value, params['earnings_date'], params['earnings_time'],
        params.get('market_trend', "No Trend"), params.get('market_reversal_likely', "No"), 
        params.get('is_meme_stock', "No"), 1, actual_price
    )
    
    # --- Output Formatting ---
    prediction = prediction_data['prediction']
    d1 = (actual_price * ((prediction / 2) + 1))
    
    trade_recommendation = "No trade recommendation."
    if -0.025 <= prediction <= 0.025:
        trade_recommendation = "Do Not Trade Options As The Expected Change Is Too Small. Place a Buy Stop Order At ±1% On Shares If Desired."
    elif 0.025 < prediction <= 0.05:
        trade_recommendation = "Do Not Trade Options. Place A Buy Stop Order For A Long Position At +2%."
    elif prediction > 0.05:
        strike_price = round(d1, -1) if actual_price > 100 else round(d1, 0) if actual_price > 10 else round(d1, 1)
        trade_recommendation = f"Buy Calls (${strike_price:.2f} Strike Price) Or Place An Order For A Long Position. Prepare With Proper Risk Management Such As Taking A Smaller Position."
    elif -0.05 <= prediction < -0.025:
        trade_recommendation = "Do Not Trade Options. Place A Buy Stop Order For A Short Position At -2%."
    elif prediction < -0.05:
        strike_price = round(d1, -1) if actual_price > 100 else round(d1, 0) if actual_price > 10 else round(d1, 1)
        trade_recommendation = f"Buy Puts (${strike_price:.2f} Strike Price) Or Place An Order For A Short Position. Prepare With Proper Risk Management Such As Taking A Smaller Position."

    if is_called_by_ai:
        summary = (f"SPEAR analysis for {params['ticker'].upper()} predicts a {prediction:.2%} change at earnings. "
                   f"The model's recommendation is: '{trade_recommendation}'. "
                   f"The automated hype score used was {params['hype']:.4f}. "
                   f"A graph showing the predicted price has been saved.")
        plot_spear_graph(params['ticker'], actual_price * (1 + prediction), is_called_by_ai=True)
        return {"summary": summary, "predicted_change_percent": prediction * 100, "recommendation": trade_recommendation}
    else:
        # CLI prints full details
        print("\n" + "="*50)
        summary_trim, industry, sector, market_cap_summary = business_summary_spear(params['ticker'], 3)
        print(f"**Business Summary for {params['ticker'].upper()}:**\n{summary_trim}")
        print(f"\nIndustry: {industry}\nSector: {sector}\nMarket Cap: {market_cap_summary}")
        
        print("\n**--- SPEAR Analysis Results ---**")
        print(f"# Ticker: {params['ticker'].upper()}")
        
        print(f"\n**Trade Recommendation:**\n{trade_recommendation}\n")
        
        data_table = {
            'Prediction At Earnings': f"{prediction:.2%}",
            'Growth From Now To Earnings': f"{prediction_data['prediction_in_time']:.2%}",
            'Live Price': f"${actual_price:,.2f}",
            'Earnings Date': params['earnings_date'],
            'Earnings Time': "After Hours" if params['earnings_time'] == 'a' else "Pre-Market",
            '1D% Change': f"{prediction_data['finance_data']['1D% Change']:.2%}",
            '1M% Change': f"{prediction_data['finance_data']['1M% Change']:.2%}",
            '1Y% Change': f"{prediction_data['finance_data']['1Y% Change']:.2%}",
        }
        print(tabulate(data_table.items(), headers=["Metric", "Value"], tablefmt="grid"))
        
        plot_spear_graph(params['ticker'], actual_price * (1 + prediction), is_called_by_ai=False)
        print("="*50 + "\n")
        return None