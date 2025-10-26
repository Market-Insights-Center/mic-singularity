
# --- Imports for reportgeneration_command ---
import asyncio
import os
import sys
import uuid
import traceback
import json
import csv
from datetime import datetime
from typing import List, Dict, Any, Optional
import re
from collections import Counter
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from nltk.tokenize import sent_tokenize
import humanize
import requests
from io import StringIO

# --- Imports from other command modules ---
from invest_command import handle_invest_command
from risk_command import perform_risk_calculations_singularity #type: ignore
from breakout_command import run_breakout_analysis_singularity #type: ignore
# --- FIX: Direct imports for self-contained parallel tasks ---
from powerscore_command import handle_powerscore_command #type: ignore
from sentiment_command import handle_sentiment_command #type: ignore

REPORTS_DATA_FILE = 'generated_reports.json'

# --- Helper Functions (Copied or Moved for Self-Containment) ---

def ask_singularity_input(prompt: str, validation_fn=None, error_msg: str = "Invalid input.", default_val=None) -> Optional[str]:
    while True:
        full_prompt = f"{prompt}"
        if default_val is not None:
            full_prompt += f" (default: {default_val}, press Enter)"
        full_prompt += ": "
        user_response = input(full_prompt).strip()
        if not user_response and default_val is not None:
            return str(default_val)
        if not user_response and default_val is None:
            print("Input is required.")
            continue
        if validation_fn:
            if validation_fn(user_response):
                return user_response
            else:
                print(error_msg)
        else:
            return user_response

async def generate_ai_filename(strategy_name: str, industries: List[str], gemini_model: Any) -> str:
    if not gemini_model:
        safe_strategy_name = re.sub(r'\W+', '', strategy_name.replace(' ', '_'))
        return f"Report_{safe_strategy_name}_{datetime.now().strftime('%Y%m%d')}.pdf"
    try:
        industry_str = ", ".join(industries) if industries else "Diversified"
        prompt = f"""
        Generate a single, short, descriptive, and filesystem-safe filename for an investment report.
        - The filename should NOT include the '.pdf' extension.
        - Use snake_case or PascalCase.
        - It must be based on this Strategy: "{strategy_name}" and these Top Industries: "{industry_str}".
        - Your entire response must be ONLY the filename. Do not add any text, examples, or explanations.
        Example of a valid response from you: Balanced_EnergyFinancials_Report
        """
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        filename_base = response.text.strip().split()[-1].replace('.pdf', '')
        filename_sanitized = re.sub(r'[^a-zA-Z0-9_-]', '', filename_base)
        if not filename_sanitized:
            raise ValueError("AI returned an empty or invalid filename base.")
        return f"{filename_sanitized}_{datetime.now().strftime('%m%d')}.pdf"
    except Exception as e:
        print(f"Could not generate AI filename, using fallback: {e}")
        safe_strategy_name = re.sub(r'\W+', '', strategy_name.replace(' ', '_'))
        return f"Report_{safe_strategy_name}_{datetime.now().strftime('%Y%m%d')}.pdf"

def get_gics_map(filepath="gics_map.txt") -> Dict[str, str]:
    gics_map = {}
    if not os.path.exists(filepath): return {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if ':' in line:
                    code, name = line.strip().split(':', 1)
                    gics_map[code] = name
    except Exception: pass
    return gics_map

def filter_stocks_by_gics(user_inputs_str: str, txt_path: str = 'gics_database.txt') -> set:
    if not os.path.exists(txt_path): return set()
    user_inputs_list = [item.strip() for item in user_inputs_str.split(',')]
    if any(item.lower() == 'market' for item in user_inputs_list):
        all_tickers = set()
        with open(txt_path, 'r') as f:
            for line in f:
                if ':' in line:
                    _, tickers = line.split(':', 1)
                    all_tickers.update(t.strip().upper() for t in tickers.split(',') if t.strip())
        return all_tickers
    gics_map = get_gics_map()
    name_to_code_map = {name.lower(): code for code, name in gics_map.items()}
    gics_data = {}
    with open(txt_path, 'r') as f:
        for line in f:
            if ':' in line:
                code, tickers = line.split(':', 1)
                gics_data[code.strip()] = tickers.strip()
    target_codes = set()
    selected_tickers = set()

    # If the input looks like a list of tickers, use them directly
    if all(re.match(r'^[A-Z]{1,5}$', item) for item in user_inputs_list) and len(user_inputs_list) > 1:
        return set(user_inputs_list)

    for item in user_inputs_list:
        item_lower = item.lower()
        if item.isdigit():
            target_codes.add(item)
        else:
            for name, code in name_to_code_map.items():
                if item_lower in name:
                    target_codes.add(code)

    for user_code in target_codes:
        for db_code, tickers_str in gics_data.items():
            if db_code.startswith(user_code):
                selected_tickers.update(t.strip().upper() for t in tickers_str.split(',') if t.strip())
    return selected_tickers

async def pre_screen_stocks_by_sensitivity(tickers: list, sensitivity: int) -> list:
    if sensitivity >= 3: return tickers
    if not tickers: return []
    cap_thresh, vol_thresh = (5e9, 1e6) if sensitivity == 1 else (1e9, 5e5)
    screened_tickers = []
    chunk_size = 25
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            data = await asyncio.to_thread(yf.download, tickers=chunk, period="3mo", progress=False, timeout=30)
            if data.empty: continue
            avg_vol = data['Volume'].mean()
            for ticker in chunk:
                try:
                    info = await asyncio.to_thread(lambda t=ticker: yf.Ticker(t).info)
                    if info.get('marketCap', 0) >= cap_thresh and avg_vol.get(ticker, 0) >= vol_thresh:
                        screened_tickers.append(ticker)
                except Exception: continue
        except Exception: continue
    return screened_tickers

async def calculate_ema_invest(ticker: str, ema_interval: int, is_called_by_ai: bool = False) -> tuple[Optional[float], Optional[float]]:
    ticker_yf_format = ticker.replace('.', '-')
    stock = yf.Ticker(ticker_yf_format)
    interval_map = {1: "1wk", 2: "1d", 3: "1h"}
    period_map = {1: "max", 2: "10y", 3: "2y"}
    interval_str = interval_map.get(ema_interval, "1h")
    period_str = period_map.get(ema_interval, "2y")
    try:
        data = await asyncio.to_thread(stock.history, period=period_str, interval=interval_str)
    except Exception:
        return None, None
    if data.empty or 'Close' not in data.columns: return None, None
    try:
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()
    except Exception:
        return None, None
    if data.empty or data.iloc[-1][['Close', 'EMA_8', 'EMA_55']].isna().any():
        return (data['Close'].iloc[-1] if not data.empty and pd.notna(data['Close'].iloc[-1]) else None), None
    latest = data.iloc[-1]
    live_price, ema_8, ema_55 = latest['Close'], latest['EMA_8'], latest['EMA_55']
    if pd.isna(live_price) or pd.isna(ema_8) or pd.isna(ema_55) or ema_55 == 0: return live_price, None
    ema_enter = (ema_8 - ema_55) / ema_55
    ema_invest_score = ((ema_enter * 4) + 0.5) * 100
    return float(live_price), float(ema_invest_score)

def safe_score(value: Any) -> float:
    try:
        if pd.isna(value) or value is None: return 0.0
        if isinstance(value, str): value = value.replace('%', '').replace('$', '').strip()
        return float(value)
    except (ValueError, TypeError): return 0.0

async def calculate_market_invest_scores_singularity(tickers: List[str], ema_sens: int, is_called_by_ai: bool = False) -> List[Dict[str, Any]]:
    result_data_market = []
    total_tickers = len(tickers)
    if not is_called_by_ai:
        print(f"\nCalculating Invest scores for {total_tickers} market tickers (Sensitivity: {ema_sens})...")
    chunk_size = 25
    processed_count_market = 0
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i + chunk_size]
        tasks = [calculate_ema_invest(ticker, ema_sens, is_called_by_ai=True) for ticker in chunk]
        results_chunk = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, res_item in enumerate(results_chunk):
            ticker_processed = chunk[idx]
            if isinstance(res_item, Exception):
                result_data_market.append({'ticker': ticker_processed, 'live_price': None, 'score': None, 'error': str(res_item)})
            elif res_item is not None:
                live_price_market, ema_invest_score_market = res_item
                result_data_market.append({'ticker': ticker_processed, 'live_price': live_price_market, 'score': ema_invest_score_market})
            processed_count_market += 1
            if not is_called_by_ai and (processed_count_market % 50 == 0 or processed_count_market == total_tickers):
                print(f"  ...market scores calculated for {processed_count_market}/{total_tickers} tickers.")
    result_data_market.sort(key=lambda x: safe_score(x.get('score', -float('inf'))), reverse=True)
    if not is_called_by_ai:
        print("Finished calculating all market scores.")
    return result_data_market
  
def business_summary_spear(ticker, business_summary_length_sentences):
    try:
        company = yf.Ticker(ticker)
        info = company.info
        if not info or info.get('regularMarketPrice') is None:
            return 'Business Summary not available', 'N/A', 'N/A', 'N/A'
        summary_raw = info.get('longBusinessSummary', 'Business Summary not available')
        industry = info.get('industry', 'N/A')
        sector = info.get('sector', 'N/A')
        market_cap_raw = info.get('marketCap')
        market_cap_formatted = humanize.intword(market_cap_raw) if market_cap_raw else 'N/A'
        summary_trim = ' '.join(sent_tokenize(summary_raw)[:business_summary_length_sentences])
        return summary_trim, industry, sector, market_cap_formatted
    except (TypeError, IndexError, Exception) as e:
        print(f"   -> [DEBUG] Handled yfinance crash in business_summary_spear for {ticker}. Error: {e}")
        return 'Business Summary not available', 'N/A', 'N/A', 'N/A'

async def get_ai_stock_rationale(ticker: str, investment_goals: str, risk_tolerance: int, gemini_model: Any) -> str:
    if not gemini_model: return "AI model not available."
    try:
        summary, _, _, _ = await asyncio.to_thread(business_summary_spear, ticker, 2)
        prompt = f"""Given a user with risk tolerance {risk_tolerance}/5 and investment goals "{investment_goals}", provide a concise (2-3 sentences) rationale for including {ticker} ({summary}) in their portfolio."""
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate AI rationale for {ticker}: {e}"

async def get_ai_trading_strategy(risk_tolerance: int, market_invest_score: float, strategy_name: str, gemini_model: Any) -> str:
    if not gemini_model: return "AI model not available."
    try:
        prompt = f"""Generate a brief (3-4 sentences) 'Proposed Trading Strategy' section for an investment report. Inputs: Risk Tolerance: {risk_tolerance}/5, Market Score: {market_invest_score:.2f}/100, Strategy: "{strategy_name}". Mention review frequency and rebalancing. This is not financial advice."""
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        return response.text.strip()
    except Exception as e:
        return f"Could not generate AI trading strategy: {e}"

async def save_chart_for_pdf(portfolio_data, cash_value, total_value):
    all_holdings = [{'ticker': h['ticker'], 'value': h['actual_money_allocation']} for h in portfolio_data]
    if cash_value > 1e-9:
        all_holdings.append({'ticker': 'Cash', 'value': cash_value})
    all_holdings.sort(key=lambda x: x['value'], reverse=True)
    top_15 = all_holdings[:15]
    other_value = sum(h['value'] for h in all_holdings[15:])
    labels = [h['ticker'] for h in top_15]
    sizes = [h['value'] for h in top_15]
    if other_value > 0:
        labels.append('Other')
        sizes.append(other_value)
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(8, 5))
    wedges, _, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=90, textprops=dict(color="w"))
    ax.legend(wedges, labels, title="Assets", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), labelcolor='white')
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title(f"Portfolio Allocation (Total Value: ${total_value:,.2f})", color='white')
    filename = f"report_pie_chart_{uuid.uuid4().hex[:6]}.png"
    plt.savefig(filename, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    return filename

def create_pdf_report_reportlab(filename, report_data):
    doc = SimpleDocTemplate(filename, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='TitleStyle', fontName='Helvetica-Bold', fontSize=18, alignment=TA_CENTER, spaceAfter=24))
    styles.add(ParagraphStyle(name='HeaderStyle', fontName='Helvetica-Bold', fontSize=14, spaceAfter=12))
    styles.add(ParagraphStyle(name='BodyStyle', fontName='Helvetica', fontSize=11, leading=14, spaceAfter=12))
    styles.add(ParagraphStyle(name='RationaleStyle', fontName='Helvetica', fontSize=10, leading=12, leftIndent=15, spaceAfter=12))
    story = []
    story.append(Paragraph("M.I.C. Singularity - Investment Recommendation", styles['TitleStyle']))
    story.append(Paragraph(f"<i>Report Date: {report_data['date']}</i>", styles['Normal']))
    story.append(Spacer(1, 0.25*inch))
    story.append(Paragraph("Client Profile", styles['HeaderStyle']))
    profile_text = (f"<b>Risk Tolerance:</b> {report_data['risk_tolerance']}/5<br/>"
                    f"<b>Investment Goals:</b> {report_data['investment_goals']}<br/>"
                    f"<b>Portfolio Value:</b> ${report_data['portfolio_value']:,.2f}")
    story.append(Paragraph(profile_text, styles['BodyStyle']))
    story.append(Spacer(1, 0.25*inch))
    story.append(Paragraph("Proposed Solution", styles['HeaderStyle']))
    story.append(Paragraph(report_data['solution_text'], styles['BodyStyle']))
    story.append(Spacer(1, 0.25*inch))
    story.append(Paragraph("Proposed Trading Strategy", styles['HeaderStyle']))
    story.append(Paragraph(report_data['trading_strategy'], styles['BodyStyle']))
    story.append(Spacer(1, 0.25*inch))
    story.append(Paragraph("Portfolio Allocation Chart", styles['HeaderStyle']))
    img = Image(report_data['chart_filename'], width=7*inch, height=4.375*inch) 
    story.append(img)
    story.append(Spacer(1, 0.25*inch))
    story.append(Paragraph("Holdings Analysis & Rationale", styles['HeaderStyle']))
    story.append(Paragraph(f"<i>Strategy: {report_data['strategy_name']}</i>", styles['Normal']))
    story.append(Spacer(1, 0.1*inch))
    for holding in report_data['holdings_data']:
        holding_text = f"<b>{holding['ticker']} ({holding['industry']})</b>"
        story.append(Paragraph(holding_text, styles['BodyStyle']))
        rationale_text = (f"<b>Allocation:</b> ${holding['actual_money_allocation']:,.2f} ({holding['percentage_allocation']:.2f}%)<br/>"
                          f"<b>Rationale:</b> {holding['rationale']}")
        story.append(Paragraph(rationale_text, styles['RationaleStyle']))
    story.append(Paragraph("Final Portfolio Summary", styles['HeaderStyle']))
    table_data = [report_data['table_header']] + report_data['table_data']
    cash_row = ['CASH', '-', '-', f"{report_data['cash_percentage']:.2f}%", 'Cash Reserve']
    table_data.append(cash_row)
    t = Table(table_data, colWidths=[1.0*inch, 1.0*inch, 0.75*inch, 1.0*inch, 2.25*inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0,0), (-1,0), 12),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige),
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))
    story.append(t)
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"<b>Remaining Cash: ${report_data['final_cash']:,.2f}</b>", styles['BodyStyle']))
    doc.build(story)

async def save_report_holdings(report_filename: str, holdings_data: List[Dict[str, Any]]):
    print(f"-> Saving holdings data for report: {report_filename}...")
    try:
        if os.path.exists(REPORTS_DATA_FILE):
            with open(REPORTS_DATA_FILE, 'r') as f:
                all_reports_data = json.load(f)
        else:
            all_reports_data = {}
        holdings_to_save = []
        for holding in holdings_data:
            price_at_generation = holding.get('live_price') 
            if price_at_generation is not None:
                 holdings_to_save.append({
                    'ticker': holding['ticker'],
                    'shares': holding['shares'],
                    'price_at_generation': price_at_generation
                })
        report_key = report_filename.replace('.pdf', '')
        all_reports_data[report_key] = {
            'filename': report_filename,
            'generation_date': datetime.now().isoformat(),
            'holdings': holdings_to_save
        }
        with open(REPORTS_DATA_FILE, 'w') as f:
            json.dump(all_reports_data, f, indent=4)
        print(f"   -> Successfully saved data for {len(holdings_to_save)} holdings.")
    except Exception as e:
        print(f"   -> ERROR: Could not save report holdings data. {e}")

async def handle_performance_check():
    print("\n--- Portfolio Performance Check ---")
    if not os.path.exists(REPORTS_DATA_FILE):
        print("No report data found. Please generate a report first.")
        return
    with open(REPORTS_DATA_FILE, 'r') as f:
        try:
            all_reports_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: Could not read the report data file ('{REPORTS_DATA_FILE}'). It might be corrupted.")
            return
    if not all_reports_data:
        print("No reports have been saved yet.")
        return
    print("Available reports:")
    for report_name in all_reports_data.keys():
        print(f"  - {report_name}")
    report_name_input = ask_singularity_input("\nEnter the name of the report to check performance")
    if not report_name_input or report_name_input not in all_reports_data:
        print("Invalid report name. Aborting.")
        return
    selected_report = all_reports_data[report_name_input]
    saved_holdings = selected_report.get('holdings', [])
    if not saved_holdings:
        print(f"No holdings data found for report '{report_name_input}'.")
        return
    tickers = [h['ticker'] for h in saved_holdings]
    print(f"\nFetching current market data for {len(tickers)} assets...")
    try:
        data = await asyncio.to_thread(yf.download, tickers=tickers, period="1d", progress=False)
        if data.empty:
            print("Could not fetch market data.")
            return
        current_prices = data['Close'].iloc[-1].to_dict()
    except Exception as e:
        print(f"An error occurred while fetching prices: {e}")
        return
    print(f"\n--- Performance for Report: {report_name_input} ---")
    print(f"--- Generated on: {datetime.fromisoformat(selected_report['generation_date']).strftime('%Y-%m-%d %H:%M')} ---")
    table_header = f"{'Ticker':<10} | {'Shares':>10} | {'Saved Price':>15} | {'Current Price':>15} | {'P&L':>15}"
    print(table_header)
    print("-" * len(table_header))
    total_pnl = 0.0
    total_initial_value = 0.0
    total_current_value = 0.0
    for holding in saved_holdings:
        ticker = holding['ticker']
        shares = holding['shares']
        saved_price = holding['price_at_generation']
        current_price = current_prices.get(ticker.replace('.', '-'))
        if current_price is None or pd.isna(current_price):
            print(f"{ticker:<10} | {shares:>10.4f} | {f'${saved_price:,.2f}':>15} | {'N/A':>15} | {'PRICE N/A':>15}")
            continue
        initial_value = saved_price * shares
        current_value = current_price * shares
        pnl = current_value - initial_value
        total_pnl += pnl
        total_initial_value += initial_value
        total_current_value += current_value
        print(f"{ticker:<10} | {shares:>10.4f} | {f'${saved_price:,.2f}':>15} | {f'${current_price:,.2f}':>15} | {f'${pnl:,.2f}':>15}")
    print("-" * len(table_header))
    if total_initial_value != 0:
        percentage_change = (total_pnl / total_initial_value) * 100
        print(f"\nInitial Portfolio Value: ${total_initial_value:,.2f}")
        print(f"Current Portfolio Value: ${total_current_value:,.2f}")
        print(f"Total Profit/Loss: ${total_pnl:,.2f} ({percentage_change:+.2f}%)")
    else:
        print("\nCould not calculate percentage change as initial value was zero.")
    print("--- End of Report ---")

async def _scrape_website_text(url: str) -> Optional[str]:
    print(f"-> Attempting to scrape content from {url}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = await asyncio.to_thread(requests.get, url, headers=headers, timeout=15)
        response.raise_for_status()
        text = response.text
        text = re.sub(r'<(script|style).*?>.*?</\1>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        print("   -> Scrape successful.")
        return text
    except Exception as e:
        print(f"   -> ERROR: Failed to scrape website: {e}")
        return None

async def _get_inputs_from_text_ai(text: str, gemini_model: Any) -> Optional[Dict[str, Any]]:
    if not gemini_model:
        print("   -> ERROR: AI model is not available for input extraction.")
        return None
    print("-> Asking AI to extract report parameters from text...")
    prompt = f"""
    Analyze the following user-provided text to determine the four key inputs for an investment report.
    The four inputs are:
    1.  'risk_tolerance': An integer between 1 and 5.
    2.  'investment_goals': A brief string describing the user's objectives.
    3.  'portfolio_value': A floating-point number representing the total investment amount.
    4.  'gics_input': A comma-separated string of GICS codes, industry names, or the word 'market'.
    If a value is not mentioned, use a sensible default (e.g., risk_tolerance: 3, gics_input: 'market'). For portfolio value, if not found, use 100000. For goals, if not found, use 'Balanced Growth'.
    Your response MUST be a single, clean JSON object and nothing else. Do not add any explanatory text before or after the JSON.
    Example Response:
    {{
      "risk_tolerance": 4,
      "investment_goals": "Aggressive growth with a focus on tech",
      "portfolio_value": 250000.00,
      "gics_input": "Technology, Software"
    }}
    --- USER TEXT TO ANALYZE ---
    {text}
    """
    try:
        response = await asyncio.to_thread(gemini_model.generate_content, prompt)
        json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if not json_match:
            raise ValueError("AI did not return a valid JSON object.")
        extracted_data = json.loads(json_match.group(0))
        print("   -> AI extraction successful.")
        return extracted_data
    except Exception as e:
        print(f"   -> ERROR: Failed to get or parse AI response: {e}")
        return None

async def generate_ai_driven_report(ai_params: Dict[str, Any], gemini_model: Any, **kwargs) -> Dict[str, str]:
    print("\n--- [DEBUG] Entering Armored PDF Report Generation ---")
    chart_filename = ""
    try:
        print("[DEBUG] Step 1/5: Extracting parameters...")
        risk_tolerance = int(ai_params.get('risk_tolerance', 3))
        investment_goals = str(ai_params.get('investment_goals', 'Balanced Growth'))
        portfolio_value = float(ai_params.get('portfolio_value', 100000.0))
        gics_input = str(ai_params.get('gics_input', 'Market'))
        print(f"   -> OK: Parameters received: Risk={risk_tolerance}, Value=${portfolio_value:,.2f}, Goals='{investment_goals}', Sectors='{gics_input}'")
        print("[DEBUG] Step 2/5: Determining strategy...")
        risk_results = await perform_risk_calculations_singularity(is_called_by_ai=True)
        market_invest_score_str = risk_results[0].get("market_invest_score", "50.0").replace('%', '')
        market_invest_score = float(market_invest_score_str) if market_invest_score_str != 'N/A' else 50.0
        strategy = {}
        is_high_score = market_invest_score >= 50
        if is_high_score:
            if risk_tolerance <= 2: strategy = {'name': 'High Score / Low Tolerance', 'amplification': 1.0, 'weights': {'Market Index': 20, 'Hedging': 20, 'Filtered Stocks': 60}, 'sub_portfolios': {'Market Index': 'SPY,DIA,QQQ', 'Hedging': 'GLD,SLV'}}
            elif risk_tolerance == 3: strategy = {'name': 'High Score / Medium Tolerance', 'amplification': 1.5, 'weights': {'Market Index': 15, 'Hedging': 15, 'Filtered Stocks': 70}, 'sub_portfolios': {'Market Index': 'SPY,QQQ', 'Hedging': 'GLD,SLV'}}
            else: strategy = {'name': 'High Score / High Tolerance', 'amplification': 2.0, 'weights': {'Market Index': 7.5, 'Hedging': 7.5, 'Breakouts': 10, 'Filtered Stocks': 75}, 'sub_portfolios': {'Market Index': 'SPY,QQQ', 'Hedging': 'GLD,SLV'}}
        else:
            if risk_tolerance <= 2: strategy = {'name': 'Low Score / Low Tolerance', 'amplification': 0.5, 'weights': {'Market Index': 30, 'Hedging': 30, 'Filtered Stocks': 40}, 'sub_portfolios': {'Market Index': 'SPY,DIA', 'Hedging': 'GLD,SLV'}}
            elif risk_tolerance == 3: strategy = {'name': 'Low Score / Medium Tolerance', 'amplification': 1.0, 'weights': {'Market Index': 25, 'Hedging': 25, 'Filtered Stocks': 50}, 'sub_portfolios': {'Market Index': 'SPY,DIA,QQQ', 'Hedging': 'GLD,SLV'}}
            else: strategy = {'name': 'Low Score / High Tolerance', 'amplification': 1.5, 'weights': {'Market Index': 15, 'Hedging': 15, 'Breakouts': 10, 'Filtered Stocks': 60}, 'sub_portfolios': {'Market Index': 'SPY,QQQ', 'Hedging': 'GLD,SLV'}}
        print(f"   -> OK: Strategy selected: '{strategy['name']}'")
        print("[DEBUG] Step 3/5: Filtering and scoring stocks...")
        ema_sensitivity = 1 if risk_tolerance <= 2 else (2 if risk_tolerance == 3 else 3)
        initial_gics_tickers = filter_stocks_by_gics(gics_input)
        print(f"   -> DEBUG: Found {len(initial_gics_tickers)} initial tickers from GICS input '{gics_input}'.")
        if not initial_gics_tickers:
            return {"status": "error", "message": f"Could not find any stocks matching the sector/industry '{gics_input}'. Please try a different or broader category."}
        screened_tickers = await pre_screen_stocks_by_sensitivity(list(initial_gics_tickers), ema_sensitivity)
        if not screened_tickers:
            return {"status": "error", "message": f"Found stocks for '{gics_input}', but none passed the pre-screening for market cap and volume. Try a higher sensitivity or a different sector."}
        all_stocks_with_scores = await calculate_market_invest_scores_singularity(screened_tickers, ema_sensitivity, is_called_by_ai=True)
        top_25_stocks = sorted(all_stocks_with_scores, key=lambda x: x.get('score') or -float('inf'), reverse=True)[:25]
        filtered_stocks_final = [d['ticker'] for d in top_25_stocks]
        print(f"   -> OK: Filtering complete. {len(filtered_stocks_final)} stocks selected for portfolio.")
        if not filtered_stocks_final:
            return {"status": "error", "message": "Stocks were found and screened, but none had a valid score to be included in the report. This may indicate a data issue."}
        print("[DEBUG] Step 4/5: Allocating portfolio...")
        top_5_breakouts = []
        if 'Breakouts' in strategy['weights']:
            breakout_results = await run_breakout_analysis_singularity(is_called_by_ai=True)
            top_5_breakouts = [d['Ticker'] for d in breakout_results.get('current_breakout_stocks', [])[:5]]
        invest_params = {"ema_sensitivity": ema_sensitivity, "amplification": strategy['amplification'], "tailor_to_value": True, "total_value": portfolio_value, "use_fractional_shares": True, "sub_portfolios": []}
        for name, tickers in strategy['sub_portfolios'].items():
            invest_params['sub_portfolios'].append({'tickers': tickers, 'weight': strategy['weights'][name]})
        invest_params['sub_portfolios'].append({'tickers': ",".join(filtered_stocks_final), 'weight': strategy['weights']['Filtered Stocks']})
        if 'Breakouts' in strategy['weights'] and top_5_breakouts:
            invest_params['sub_portfolios'].append({'tickers': ",".join(top_5_breakouts), 'weight': strategy['weights']['Breakouts']})
        _, _, final_cash, tailored_portfolio_data = await handle_invest_command(args=[], ai_params=invest_params, is_called_by_ai=True, return_structured_data=True)
        live_price_map = {stock['ticker']: stock['live_price'] for stock in all_stocks_with_scores if stock.get('live_price') is not None}
        for holding in tailored_portfolio_data:
            if holding['ticker'] in live_price_map:
                holding['live_price'] = live_price_map[holding['ticker']]
        print(f"   -> OK: Allocation complete. Portfolio has {len(tailored_portfolio_data)} holdings with ${final_cash:,.2f} remaining cash.")
        print("[DEBUG] Step 5/5: Generating PDF...")
        chart_filename = await save_chart_for_pdf(tailored_portfolio_data, final_cash, portfolio_value)
        holdings_data_for_pdf = []
        all_tickers_to_score = {h['ticker'] for h in tailored_portfolio_data}; all_scores_calculated = await calculate_market_invest_scores_singularity(list(all_tickers_to_score), ema_sensitivity, is_called_by_ai=True); invest_scores_map = {d['ticker']: d['score'] for d in all_scores_calculated if d.get('score') is not None}; etf_list = {'SPY', 'QQQ', 'DIA', 'IWM', 'GLD', 'SLV'}
        for holding in tailored_portfolio_data:
            if holding['ticker'] in etf_list: holding['industry'] = "ETF"
            else: _, industry, _, _ = await asyncio.to_thread(business_summary_spear, holding['ticker'], 1); holding['industry'] = industry
            holding['invest_score'] = invest_scores_map.get(holding['ticker'], 'N/A'); holding['percentage_allocation'] = (holding['actual_money_allocation'] / portfolio_value) * 100; holding['rationale'] = await get_ai_stock_rationale(holding['ticker'], investment_goals, risk_tolerance, gemini_model=gemini_model); holdings_data_for_pdf.append(holding)
        table_header = ["Ticker", "Invest Score", "Shares", "% Allocation", "Industry"]; table_data = [[h['ticker'], f"{h.get('invest_score'):.2f}%" if isinstance(h.get('invest_score'), (int, float)) else "N/A", f"{h['shares']:.2f}", f"{h['percentage_allocation']:.2f}%", h.get('industry', 'N/A')] for h in holdings_data_for_pdf]
        report_data_package = { "date": datetime.now().strftime('%Y-%m-%d'), "risk_tolerance": risk_tolerance, "investment_goals": investment_goals, "portfolio_value": portfolio_value, "solution_text": f"A diversified portfolio based on your risk tolerance of '{risk_tolerance}/5' and market score of {market_invest_score:.2f}, structured via the '{strategy['name']}' strategy.", "trading_strategy": await get_ai_trading_strategy(risk_tolerance, market_invest_score, strategy['name'], gemini_model=gemini_model), "strategy_name": strategy['name'], "chart_filename": chart_filename, "holdings_data": holdings_data_for_pdf, "table_header": table_header, "table_data": table_data, "final_cash": final_cash, "cash_percentage": (final_cash / portfolio_value) * 100 if portfolio_value > 0 else 0 }
        all_industries = [h.get('industry', 'Other') for h in holdings_data_for_pdf if h.get('industry') and h.get('industry') != 'ETF']; top_industries = [item[0] for item in Counter(all_industries).most_common(2)]; report_filename = await generate_ai_filename(strategy['name'], top_industries, gemini_model)
        create_pdf_report_reportlab(report_filename, report_data_package)
        await save_report_holdings(report_filename, tailored_portfolio_data)
        print(f"   -> OK: PDF generation complete: '{report_filename}'")
        print("[DEBUG] ✅ Armored report generation process finished successfully.")
        return {"status": "success", "filename": report_filename}
    except Exception as e:
        error_type = type(e).__name__
        error_message = f"Failed during AI report generation: {traceback.format_exc()}"
        print(f"❌ [DEBUG] CRITICAL ERROR ({error_type}) in generate_ai_driven_report: {error_message}")
        return {"status": "error", "message": f"An internal error occurred ({error_type}). Please check the console logs for the full traceback."}
    finally:
        if chart_filename and os.path.exists(chart_filename):
            os.remove(chart_filename)

async def create_dynamic_investment_plan(ai_params: Dict[str, Any], gemini_model: Any, available_functions: Dict, **kwargs) -> Dict[str, str]:
    """
    AI Orchestrator: Dynamically plans and executes a series of tool calls to build a complex, customized report.
    VERSION 2.9: Added explicit context handling to the planner prompt.
    """
    print("\n--- AI Dynamic Report Orchestrator v2.9 ---")
    user_request = ai_params.get('user_request')
    if not user_request:
        return {"status": "error", "message": "No user request was provided to the orchestrator."}

    chart_filename = ""
    try:
        print("-> Asking AI to create a dynamic execution plan...")
        
        # --- FIX: Added a rule to prioritize the CONTEXT block for data sourcing. ---
        planning_prompt = f"""
        Analyze the user's request, paying close attention to any "CONTEXT:" block, to create a JSON execution plan. Your response MUST be a single, clean JSON object.

        **USER REQUEST:** "{user_request}"

        **JSON TEMPLATE (fill this in):**
        {{
          "data_source": {{
            "tool": "risk_based_gics",
            "parameters": {{ "gics_input": "..." }}
          }},
          "filters": [ ... ],
          "allocation_details": {{ ... }}
        }}

        **CRITICAL RULES FOR DATA SOURCE:**
        1.  **PRIORITY 1 (CONTEXT):** If the request starts with "CONTEXT: The user's favorite tickers have been retrieved and are: [TICKERS]...", you MUST use those exact tickers for the 'gics_input' parameter. IGNORE any other sector names mentioned in the original request.
        2.  **PRIORITY 2 (EXPLICIT TICKERS/SECTORS):** If there is no CONTEXT block, but the user mentions specific tickers (e.g., 'using AAPL and MSFT') or sectors (e.g., 'from the Technology sector'), use those for the 'gics_input'.
        3.  **PRIORITY 3 (KEYWORDS):** If no specific tickers/sectors are mentioned, check for keywords. 'cultivate' -> use 'cultivate' tool. 'breakout' -> use 'breakout' tool.
        4.  **FALLBACK:** If none of the above apply, default 'gics_input' to 'Market'.
        
        **OTHER RULES:**
        - Populate the 'filters' list based on user criteria like "PowerScore over 70".
        - ALWAYS extract 'portfolio_value' and 'risk_tolerance'.
        """

        response = await asyncio.to_thread(gemini_model.generate_content, planning_prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if not json_match: raise ValueError("AI planner did not return a valid JSON-like object.")
            data = json_match.group(0)
            plan = json.loads(data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"   -> CRITICAL ERROR: AI planner returned malformed data. Cannot proceed. Error: {e}")
            return {"status": "error", "message": "The AI planner failed to create a valid execution plan. Please try rephrasing your request."}

        source_tool = plan.get('data_source', {}).get('tool')
        source_params = plan.get('data_source', {}).get('parameters', {})
        filters = plan.get('filters', [])
        alloc_details = plan.get('allocation_details', {})

        if not source_tool or not alloc_details:
             print(f"   -> CRITICAL ERROR: AI plan is missing 'data_source' or 'allocation_details'. Plan: {plan}")
             return {"status": "error", "message": "The AI planner created an incomplete plan. It missed either the data source or allocation details."}

        print(f"   -> Plan received. Sourcing from '{source_tool}' with {len(filters)} filter(s).")

        initial_tickers = []
        if source_tool == 'cultivate':
            cultivate_return_value = await available_functions['handle_cultivate_command'](args=[], ai_params=source_params, is_called_by_ai=True)
            cultivate_results = cultivate_return_value[0] if isinstance(cultivate_return_value, tuple) else cultivate_return_value
            if isinstance(cultivate_results, list):
                initial_tickers = [d['Ticker'] for d in cultivate_results]
            else:
                print(f"-> ERROR: The 'cultivate' tool returned an error: {cultivate_results}")
                return {"status": "error", "message": f"Failed to get data from Cultivate. The tool reported: '{cultivate_results}'"}
        elif source_tool == 'breakout':
            breakout_results = await run_breakout_analysis_singularity(is_called_by_ai=True)
            initial_tickers = [d['Ticker'] for d in breakout_results.get('current_breakout_stocks', [])]
        elif source_tool == 'risk_based_gics':
            gics_input = source_params.get('gics_input', 'Market')
            if isinstance(gics_input, list): gics_input = ",".join(gics_input)
            initial_tickers = list(filter_stocks_by_gics(gics_input))
        
        print(f"-> Source data acquired: {len(initial_tickers)} tickers from '{source_tool}'.")

        filtered_tickers = initial_tickers.copy()
        semaphore = asyncio.Semaphore(5)

        for f in filters:
            if not filtered_tickers: break
            filter_tool = f.get('tool')
            if not filter_tool: continue
            print(f"-> Applying filter: {filter_tool} on {len(filtered_tickers)} tickers (in parallel)...")
            condition = f.get('condition', {})
            operator = str(condition.get('operator', '>')).lower()
            value = condition.get('value')
            if value is None: continue
            
            tasks = []
            if filter_tool == 'powerscore':
                sensitivity = str(f.get('parameters', {}).get('sensitivity', '2'))
                async def _process_ps(ticker):
                    try:
                        async with semaphore:
                            ps_data = await handle_powerscore_command(args=[], ai_params={'ticker': ticker, 'sensitivity': sensitivity}, is_called_by_ai=True)
                            score = ps_data.get('PowerScore', 0)
                            if (operator == '>' and score > value) or (operator == '<' and score < value): return ticker
                    except Exception as e:
                        print(f"    -> WARNING: Failed to process PowerScore for {ticker}. Error: {type(e).__name__}")
                    return None
                tasks = [_process_ps(ticker) for ticker in filtered_tickers]

            elif filter_tool == 'sentiment':
                async def _process_sentiment(ticker):
                    try:
                        async with semaphore:
                            sentiment_data = await handle_sentiment_command(args=[], ai_params={'ticker': ticker}, is_called_by_ai=True)
                            score = sentiment_data.get('sentiment_score', 0.0)
                            if (operator == '>' and score > value) or (operator == '<' and score < value): return ticker
                    except Exception as e:
                        print(f"    -> WARNING: Failed to process Sentiment for {ticker}. Error: {type(e).__name__}")
                    return None
                tasks = [_process_sentiment(ticker) for ticker in filtered_tickers]

            if tasks:
                results = await asyncio.gather(*tasks)
                passed_filter = [ticker for ticker in results if ticker is not None]
                filtered_tickers = passed_filter
            print(f"   -> {len(filtered_tickers)} tickers remaining after '{filter_tool}' filter.")

        if not filtered_tickers:
            print("-> ERROR: No tickers remained after applying all filters. Aborting plan.")
            return {"status": "error", "message": "The filtering criteria were too strict and eliminated all stocks from the initial list. Please try again with less restrictive filters."}
        
        portfolio_value = alloc_details['portfolio_value']
        risk_tolerance_raw = alloc_details.get('risk_tolerance', 'balanced')
        risk_map = {'conservative': 1, 'balanced': 3, 'aggressive': 5, 'speculative': 5}
        risk_tolerance = risk_map.get(risk_tolerance_raw.lower(), 3) if isinstance(risk_tolerance_raw, str) else int(risk_tolerance_raw)
        
        ema_sensitivity = 1 if risk_tolerance <= 2 else (2 if risk_tolerance == 3 else 3)

        invest_params = {"ema_sensitivity": ema_sensitivity, "amplification": 1.5, "tailor_to_value": True, "total_value": portfolio_value, "use_fractional_shares": True, "sub_portfolios": [{'tickers': ",".join(filtered_tickers), 'weight': 100}]}
        _, _, final_cash, tailored_portfolio_data = await handle_invest_command(args=[], ai_params=invest_params, is_called_by_ai=True, return_structured_data=True)
        print("-> Generating final PDF report...")
        investment_goals = alloc_details.get('investment_goals') or "Dynamically generated based on user request."
        chart_filename = await save_chart_for_pdf(tailored_portfolio_data, final_cash, portfolio_value)
        holdings_data_for_pdf = []
        for holding in tailored_portfolio_data:
            _, industry, _, _ = await asyncio.to_thread(business_summary_spear, holding['ticker'], 1)
            holding['industry'] = industry
            holding['percentage_allocation'] = (holding['actual_money_allocation'] / portfolio_value) * 100
            holding['rationale'] = await get_ai_stock_rationale(holding['ticker'], investment_goals, risk_tolerance, gemini_model=gemini_model)
            holdings_data_for_pdf.append(holding)
        table_header = ["Ticker", "Shares", "% Allocation", "Industry"]
        table_data = [[h['ticker'], f"{h['shares']:.2f}", f"{h['percentage_allocation']:.2f}%", h.get('industry', 'N/A')] for h in holdings_data_for_pdf]
        report_data_package = {"date": datetime.now().strftime('%Y-%m-%d'), "risk_tolerance": risk_tolerance, "investment_goals": investment_goals, "portfolio_value": portfolio_value, "solution_text": f"This dynamically generated portfolio was constructed based on your request: '{user_request}'.", "trading_strategy": "Review portfolio quarterly or when investment goals change.", "strategy_name": "Dynamic AI-Orchestrated Strategy", "chart_filename": chart_filename, "holdings_data": holdings_data_for_pdf, "table_header": table_header, "table_data": table_data, "final_cash": final_cash, "cash_percentage": (final_cash / portfolio_value) * 100}
        report_filename = await generate_ai_filename("DynamicPlan", [], gemini_model)
        create_pdf_report_reportlab(report_filename, report_data_package)
        await save_report_holdings(report_filename, tailored_portfolio_data)
        print(f"✅ Success! Dynamic report saved as '{report_filename}'.")
        return {"status": "success", "filename": report_filename}
    except Exception as e:
        error_message = f"Failed during dynamic plan execution: {traceback.format_exc()}"
        print(f"❌ ERROR: {error_message}")
        return {"status": "error", "message": str(e)}
    finally:
        if chart_filename and os.path.exists(chart_filename):
            os.remove(chart_filename)

async def handle_report_generation(args: List[str], ai_params: Optional[Dict] = None, is_called_by_ai: bool = False, gemini_model_obj: Any = None, **kwargs):
    # Map the argument from Prometheus/AI calls to the internal variable name
    gemini_model = gemini_model_obj

    if args and args[0].lower() == "performance":
        await handle_performance_check()
        return

    print("\n--- M.I.C. PDF Report Generation (using ReportLab) ---")
    report_inputs = None
    user_text_for_tti = ""
    mode = args[0].lower() if args else "interactive"
    if mode == "tti":
        print("\nPlease enter a detailed description for the report. Type 'END' on a new line when finished.")
        lines = []
        while True:
            line = input()
            if line.strip().upper() == 'END': break
            lines.append(line)
        user_text_for_tti = "\n".join(lines)
        if user_text_for_tti:
            report_inputs = await _get_inputs_from_text_ai(user_text_for_tti, gemini_model)
    elif mode == "web":
        url = ask_singularity_input("Enter the website URL to analyze for report parameters")
        if url:
            scraped_text = await _scrape_website_text(url)
            if scraped_text:
                report_inputs = await _get_inputs_from_text_ai(scraped_text, gemini_model)
    elif mode == "interactive":
        try:
            risk_tolerance_str = ask_singularity_input("Enter Risk Tolerance (1-5, 1=low, 5=high)", validation_fn=lambda x: 1 <= int(x) <= 5, error_msg="Please enter a number between 1 and 5.")
            if risk_tolerance_str is None: return
            investment_goals = ask_singularity_input("Describe your investment goals (e.g., 'long-term growth', 'stable income')")
            if investment_goals is None: return
            portfolio_value_str = ask_singularity_input("Enter total portfolio value", validation_fn=lambda x: float(x) > 0, error_msg="Please enter a positive number.")
            if portfolio_value_str is None: return
            gics_input = ask_singularity_input("Enter a comma-separated list of GICS codes or sector/industry names")
            if gics_input is None: return
            report_inputs = {
                "risk_tolerance": int(risk_tolerance_str),
                "investment_goals": investment_goals,
                "portfolio_value": float(portfolio_value_str),
                "gics_input": gics_input
            }
        except (ValueError, TypeError):
            print("Invalid input. Please start over.")
            return
    else:
        print(f"Unknown mode: '{mode}'. Use 'tti', 'web', or no argument for interactive mode.")
        return
    if mode in ["tti", "web"]:
        if not report_inputs:
            print("Could not determine report inputs automatically. Aborting.")
            if mode == 'tti' and user_text_for_tti:
                print("\n--- Your Original Text ---")
                print(user_text_for_tti)
            return
        print("\n--- AI Compiled Inputs ---")
        print(f"  - Risk Tolerance:   {report_inputs.get('risk_tolerance')}")
        print(f"  - Investment Goals: {report_inputs.get('investment_goals')}")
        print(f"  - Portfolio Value:  ${report_inputs.get('portfolio_value', 0):,.2f}")
        print(f"  - Target Sectors:   {report_inputs.get('gics_input')}")
        proceed = ask_singularity_input("\nProceed with these inputs? (y/n)", validation_fn=lambda x: x.lower() in ['y', 'n'], default_val='y')
        if proceed.lower() == 'n':
            print("Operation cancelled by user.")
            if mode == 'tti':
                print("\n--- Your Original Text ---")
                print(user_text_for_tti)
            return
    if not report_inputs:
        print("Report generation cancelled or failed to get inputs.")
        return
    try:
        risk_tolerance = int(report_inputs['risk_tolerance'])
        investment_goals = str(report_inputs['investment_goals'])
        portfolio_value = float(report_inputs['portfolio_value'])
        gics_input = str(report_inputs['gics_input'])
    except (KeyError, ValueError, TypeError) as e:
        print(f"Error processing the compiled inputs: {e}. Please try again.")
        return
    print("-> Determining strategy and filtering stocks by GICS...")
    risk_results = await perform_risk_calculations_singularity(is_called_by_ai=True)
    market_invest_score_str = risk_results[0].get("market_invest_score", "50.0").replace('%', '')
    market_invest_score = float(market_invest_score_str) if market_invest_score_str != 'N/A' else 50.0
    strategy = {}
    is_high_score = market_invest_score >= 50
    if is_high_score:
        if risk_tolerance <= 2: strategy = {'name': 'High Score / Low Tolerance', 'amplification': 1.0, 'weights': {'Market Index': 20, 'Hedging': 20, 'Filtered Stocks': 60}, 'sub_portfolios': {'Market Index': 'SPY,DIA,QQQ', 'Hedging': 'GLD,SLV'}}
        elif risk_tolerance == 3: strategy = {'name': 'High Score / Medium Tolerance', 'amplification': 1.5, 'weights': {'Market Index': 15, 'Hedging': 15, 'Filtered Stocks': 70}, 'sub_portfolios': {'Market Index': 'SPY,QQQ', 'Hedging': 'GLD,SLV'}}
        else: strategy = {'name': 'High Score / High Tolerance', 'amplification': 2.0, 'weights': {'Market Index': 7.5, 'Hedging': 7.5, 'Breakouts': 10, 'Filtered Stocks': 75}, 'sub_portfolios': {'Market Index': 'SPY,QQQ', 'Hedging': 'GLD,SLV'}}
    else:
        if risk_tolerance <= 2: strategy = {'name': 'Low Score / Low Tolerance', 'amplification': 0.5, 'weights': {'Market Index': 30, 'Hedging': 30, 'Filtered Stocks': 40}, 'sub_portfolios': {'Market Index': 'SPY,DIA', 'Hedging': 'GLD,SLV'}}
        elif risk_tolerance == 3: strategy = {'name': 'Low Score / Medium Tolerance', 'amplification': 1.0, 'weights': {'Market Index': 25, 'Hedging': 25, 'Filtered Stocks': 50}, 'sub_portfolios': {'Market Index': 'SPY,DIA,QQQ', 'Hedging': 'GLD,SLV'}}
        else: strategy = {'name': 'Low Score / High Tolerance', 'amplification': 1.5, 'weights': {'Market Index': 15, 'Hedging': 15, 'Breakouts': 10, 'Filtered Stocks': 60}, 'sub_portfolios': {'Market Index': 'SPY,QQQ', 'Hedging': 'GLD,SLV'}}
    ema_sensitivity = 1 if risk_tolerance <= 2 else (2 if risk_tolerance == 3 else 3)
    print(f"   Strategy Selected: {strategy['name']} | Amplification: {strategy['amplification']} | EMA Sensitivity: {ema_sensitivity}")
    initial_gics_tickers = filter_stocks_by_gics(gics_input)
    if not initial_gics_tickers:
        print("   -> ERROR: No stocks found for the specified GICS sectors/industries. Aborting report.")
        return
    print("-> Pre-screening stocks and calculating scores...")
    screened_tickers = await pre_screen_stocks_by_sensitivity(list(initial_gics_tickers), ema_sensitivity)
    if not screened_tickers:
        print("   -> ERROR: No stocks passed the pre-screening criteria. Aborting report.")
        return
    all_stocks_with_scores = await calculate_market_invest_scores_singularity(screened_tickers, ema_sensitivity, is_called_by_ai=False)
    all_stocks_with_scores.sort(key=lambda x: x.get('score') if x.get('score') is not None else -float('inf'), reverse=True)
    top_25_stocks = all_stocks_with_scores[:25]
    filtered_stocks_final = [d['ticker'] for d in top_25_stocks]
    if not filtered_stocks_final:
        print("   -> ERROR: Could not determine top stocks after scoring. Aborting report.")
        return
    print("-> Allocating portfolio and enhancing data...")
    top_5_breakouts = []
    if 'Breakouts' in strategy['weights']:
        breakout_results = await run_breakout_analysis_singularity(is_called_by_ai=True)
        top_5_breakouts = [d['Ticker'] for d in breakout_results.get('current_breakout_stocks', [])[:5]]
    invest_params = {
        "ema_sensitivity": ema_sensitivity, "amplification": strategy['amplification'],
        "tailor_to_value": True, "total_value": portfolio_value, "use_fractional_shares": True,
        "sub_portfolios": []
    }
    for name, tickers in strategy['sub_portfolios'].items():
        invest_params['sub_portfolios'].append({'tickers': tickers, 'weight': strategy['weights'][name]})
    invest_params['sub_portfolios'].append({'tickers': ",".join(filtered_stocks_final), 'weight': strategy['weights']['Filtered Stocks']})
    if 'Breakouts' in strategy['weights']:
        if top_5_breakouts:
            invest_params['sub_portfolios'].append({'tickers': ",".join(top_5_breakouts), 'weight': strategy['weights']['Breakouts']})
        else:
            if invest_params['sub_portfolios']:
                largest_sub_p = max(invest_params['sub_portfolios'], key=lambda p: p['weight'])
                largest_sub_p['weight'] += strategy['weights']['Breakouts']
    _, _, final_cash, tailored_portfolio_data = await handle_invest_command(args=[], ai_params=invest_params, is_called_by_ai=True, return_structured_data=True)

    print("-> Attaching live prices to final holdings for performance tracking...")
    final_tickers_in_portfolio = [h['ticker'] for h in tailored_portfolio_data]
    final_price_data = await calculate_market_invest_scores_singularity(final_tickers_in_portfolio, ema_sensitivity, is_called_by_ai=True)
    live_price_map = {stock['ticker']: stock['live_price'] for stock in final_price_data if stock.get('live_price') is not None}
    for holding in tailored_portfolio_data:
        if holding['ticker'] in live_price_map:
            holding['live_price'] = live_price_map[holding['ticker']]

    print("-> Generating PDF report...")
    chart_filename = ""
    try:
        chart_filename = await save_chart_for_pdf(tailored_portfolio_data, final_cash, portfolio_value)
        all_tickers_to_score = {h['ticker'] for h in tailored_portfolio_data}
        all_scores_calculated = await calculate_market_invest_scores_singularity(list(all_tickers_to_score), ema_sensitivity, is_called_by_ai=True)
        invest_scores_map = {d['ticker']: d['score'] for d in all_scores_calculated if d.get('score') is not None}
        etf_list = {'SPY', 'QQQ', 'DIA', 'IWM', 'GLD', 'SLV'}
        holdings_data_for_pdf = []
        for holding in tailored_portfolio_data:
            if holding['ticker'] in etf_list:
                holding['industry'] = "ETF"
            else:
                _, industry, _, _ = await asyncio.to_thread(business_summary_spear, holding['ticker'], 1)
                holding['industry'] = industry
            holding['invest_score'] = invest_scores_map.get(holding['ticker'], 'N/A')
            holding['percentage_allocation'] = (holding['actual_money_allocation'] / portfolio_value) * 100
            holding['rationale'] = await get_ai_stock_rationale(holding['ticker'], investment_goals, risk_tolerance, gemini_model=gemini_model)
            holdings_data_for_pdf.append(holding)
        table_header = ["Ticker", "Invest Score", "Shares", "% Allocation", "Industry"]
        table_data = []
        for holding in holdings_data_for_pdf:
            score_val = holding.get('invest_score')
            score_str = f"{score_val:.2f}%" if isinstance(score_val, (int, float)) else "N/A"
            table_data.append([
                holding['ticker'], score_str, f"{holding['shares']:.2f}",
                f"{holding['percentage_allocation']:.2f}%", holding.get('industry', 'N/A')
            ])
        report_data_package = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "risk_tolerance": risk_tolerance,
            "investment_goals": investment_goals,
            "portfolio_value": portfolio_value,
            "solution_text": f"Our solution is to create a diversified portfolio based on your risk tolerance of '{risk_tolerance}/5' and the market score of {market_invest_score:.2f}. The portfolio is structured according to the '{strategy['name']}' strategy, allocating capital across market indices, hedges, and a curated list of high-potential stocks.",
            "trading_strategy": await get_ai_trading_strategy(risk_tolerance, market_invest_score, strategy['name'], gemini_model=gemini_model),
            "strategy_name": strategy['name'],
            "chart_filename": chart_filename,
            "holdings_data": holdings_data_for_pdf,
            "table_header": table_header,
            "table_data": table_data,
            "final_cash": final_cash,
            "cash_percentage": (final_cash / portfolio_value) * 100 if portfolio_value > 0 else 0
        }
        all_industries = [h.get('industry', 'Other') for h in holdings_data_for_pdf if h.get('industry') and h.get('industry') != 'ETF']
        industry_counts = Counter(all_industries)
        top_industries = [item[0] for item in industry_counts.most_common(2)]
        report_filename = await generate_ai_filename(strategy['name'], top_industries, gemini_model)
        
        create_pdf_report_reportlab(report_filename, report_data_package)
        print(f"\n✅ Success! PDF report saved as '{report_filename}'.")
        await save_report_holdings(report_filename, tailored_portfolio_data)
    except Exception as e:
        print(f"\n❌ Error: Could not save the PDF report. {e}")
        traceback.print_exc()
    finally:
        if chart_filename and os.path.exists(chart_filename):
            os.remove(chart_filename)
