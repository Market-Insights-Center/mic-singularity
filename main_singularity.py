# --- Imports ---
import yfinance as yf
import pandas as pd
import math
from math import sqrt
from tabulate import tabulate
import os
import uuid
import matplotlib
matplotlib.use('Agg')
import asyncio
import matplotlib.pyplot as plt
import numpy as np
from tradingview_screener import Query, Column
import csv
from datetime import datetime, timedelta 
import pytz
from typing import Optional, List, Dict, Any, Tuple
import time as py_time
import traceback
import logging
import json
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
import fear_and_greed
import humanize
from nltk.tokenize import sent_tokenize
import nltk
import glob
import random
from bs4 import BeautifulSoup
import requests
from fpdf import FPDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import configparser
from scipy.stats import norm
from urllib.parse import quote_plus
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score
from scipy.interpolate import CubicSpline
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import seaborn as sns
import speech_recognition as sr
from gtts import gTTS
import playsound
import pyaudio
from scipy.stats import percentileofscore
import urllib3
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import StringIO
import sys
import re
import networkx as nx
import time
from dateutil.relativedelta import relativedelta
import string
from contextlib import contextmanager
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Isolated Commands'))

from ai_command import handle_ai_prompt, handle_voice_command, initialize_ai_components # type: ignore
from invest_command import ( # type: ignore
    handle_invest_command, 
    process_custom_portfolio, 
    calculate_ema_invest, 
    calculate_one_year_invest, 
    plot_ticker_graph, 
    get_allocation_score, 
    generate_portfolio_pie_chart
) # type: ignore
from quickscore_command import handle_quickscore_command # type: ignore
from sentiment_command import handle_sentiment_command, get_ai_sentiment_analysis # type: ignore
from optimize_command import handle_optimize_command # type: ignore
from dev_command import handle_dev_command #type: ignore
from backtest_command import handle_backtest_command # type: ignore
from compare_command import handle_compare_command # type: ignore
from custom_command import handle_custom_command, get_comparison_for_custom_portfolio # type: ignore
from cultivate_command import handle_cultivate_command, run_cultivate_analysis_singularity # type: ignore
from powerscore_command import handle_powerscore_command # type: ignore
from breakout_command import handle_breakout_command # type: ignore
from counter_command import initialize_counter_files, increment_command_count, handle_counter_command # type: ignore
from risk_command import handle_risk_command # type: ignore
from assess_command import handle_assess_command # type: ignore
from macdforecast_command import handle_macd_forecast_command # type: ignore
from heatmap_command import handle_heatmap_command # type: ignore
from market_command import handle_market_command # type: ignore
from strategies_command import handle_strategies_command # type: ignore
from futures_command import handle_futures_command # type: ignore
from favorites_command import handle_favorites_command # type: ignore
from fundamentals_command import handle_fundamentals_command # type: ignore
from briefing_command import handle_briefing_command # type: ignore
from reportgeneration_command import handle_report_generation, generate_ai_driven_report, create_dynamic_investment_plan # type: ignore
from help_command import handle_help_command, load_command_states # type: ignore
from spear_command import handle_spear_command # type: ignore
from sector_command import handle_sector_command # type: ignore
from options_command import handle_options_command # type: ignore
from simulation_command import handle_simulation_command # type: ignore
from mlforecast_command import handle_mlforecast_command # type: ignore
from history_command import handle_history_command # type: ignore
from web_command import handle_web_command # type: ignore
from monitor_command import handle_monitor_command, load_alerts_from_csv, alert_worker # type: ignore
from tracking_command import handle_tracking_command # type: ignore
from fairvalue_command import handle_fairvalue_command # type: ignore
from derivative_command import handle_derivative_command # type: ignore
from prometheus_core import Prometheus # type: ignore

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

@contextmanager
def suppress_output():
    """A context manager to temporarily suppress stdout."""
    original_stdout = sys.stdout
    sys.stdout = io.StringIO() # Redirect to a dummy stream
    try:
        yield
    finally:
        sys.stdout = original_stdout # Restore original stdout

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📂 Downloading 'punkt' for nltk...")
    nltk.download('punkt', quiet=True)

# --- Configuration File Setup ---
config = configparser.ConfigParser()
config_file = 'config.ini'

COMMAND_USAGE_TIMESTAMPS = {}
COMMAND_STATES_CACHE = {}

if not os.path.exists(config_file):
    raise FileNotFoundError(f"Error: The configuration file '{config_file}' was not found. Please create it.")

try:
    config.read(config_file)

    # --- Load API Keys ---
    GEMINI_API_KEY = config.get('API_KEYS', 'GEMINI_API_KEY', fallback="AIzaSyBpUNBQm_U6YbitLD2Hg9-4lBBASxIbW1I")

    # --- Load File Paths ---
    file_paths = config['FILE_PATHS']
    PORTFOLIO_DB_FILE = file_paths.get('PORTFOLIO_DB_FILE', 'portfolio_codes_database.csv')
    PORTFOLIO_OUTPUT_DIR = file_paths.get('PORTFOLIO_OUTPUT_DIR', 'portfolio_outputs')
    BREAKOUT_TICKERS_FILE = file_paths.get('BREAKOUT_TICKERS_FILE', 'breakout_tickers.csv')
    BREAKOUT_HISTORICAL_DB_FILE = file_paths.get('BREAKOUT_HISTORICAL_DB_FILE', 'breakout_historical_database.csv')
    MARKET_FULL_SENS_DATA_FILE_PREFIX = file_paths.get('MARKET_FULL_SENS_DATA_FILE_PREFIX', 'market_full_sens_')
    CULTIVATE_INITIAL_METRICS_FILE = file_paths.get('CULTIVATE_INITIAL_METRICS_FILE', 'cultivate_initial_metrics.csv')
    CULTIVATE_T1_FILE = file_paths.get('CULTIVATE_T1_FILE', 'cultivate_ticker_list_one.csv')
    CULTIVATE_T_MINUS_1_FILE = file_paths.get('CULTIVATE_T_MINUS_1_FILE', 'cultivate_ticker_list_negative_one.csv')
    CULTIVATE_TF_FINAL_FILE = file_paths.get('CULTIVATE_TF_FINAL_FILE', 'cultivate_ticker_list_final.csv')
    CULTIVATE_COMBINED_DATA_FILE_PREFIX = file_paths.get('CULTIVATE_COMBINED_DATA_FILE_PREFIX', 'cultivate_combined_')
    USERS_FAVORITES_FILE = file_paths.get('USERS_FAVORITES_FILE', 'users_favorites.txt')
    USER_PREFERENCES_FILE = file_paths.get('USER_PREFERENCES_FILE', 'user_preferences.json')
    RISK_CSV_FILE = file_paths.get('RISK_CSV_FILE', 'market_data.csv')
    RISK_EOD_CSV_FILE = file_paths.get('RISK_EOD_CSV_FILE', 'risk_eod_data.csv')
    RISK_LOG_FILE = file_paths.get('RISK_LOG_FILE', 'risk_calculations.log')

    # --- Load App Settings ---
    app_settings = config['APP_SETTINGS']
    EST_TIMEZONE = pytz.timezone(app_settings.get('TIMEZONE', 'US/Eastern'))
    MARKET_HEDGING_TICKERS = [t.strip() for t in app_settings.get('MARKET_HEDGING_TICKERS', 'SPY,DIA,QQQ').split(',')]
    RESOURCE_HEDGING_TICKERS = [t.strip() for t in app_settings.get('RESOURCE_HEDGING_TICKERS', 'GLD,SLV').split(',')]
    HEDGING_TICKERS = MARKET_HEDGING_TICKERS + RESOURCE_HEDGING_TICKERS

except (configparser.Error, pytz.exceptions.UnknownTimeZoneError) as e:
    print(f"❌ Error reading configuration file: {e}")
    GEMINI_API_KEY = "AIzaSyDYpuf4NC1SET9Z5_hQqbJ9tzpxXOPk4k0"
    EST_TIMEZONE = pytz.timezone('US/Eastern')
    MARKET_HEDGING_TICKERS = ['SPY', 'DIA', 'QQQ']
    RESOURCE_HEDGING_TICKERS = ['GLD', 'SLV']
    HEDGING_TICKERS = MARKET_HEDGING_TICKERS + RESOURCE_HEDGING_TICKERS
    PORTFOLIO_DB_FILE = 'portfolio_codes_database.csv'

gemini_model, tts_engine, AVAILABLE_PYTHON_FUNCTIONS = None, None, {}
GEMINI_CHAT_SESSION = None 
AI_CONVERSATION_HISTORY = []
CURRENT_AI_SESSION_ORIGINAL_REQUEST = None
AI_INTERNAL_STEP_COUNT = 0 
GEMINI_API_LOCK = asyncio.Lock()
YFINANCE_API_SEMAPHORE = asyncio.Semaphore(8)
YFINANCE_LOCK = asyncio.Lock()
API_TASK_SEMAPHORE = asyncio.Semaphore(8)
risk_persistent_signal = "Hold"
risk_signal_day = None
risk_logger = logging.getLogger('RISK_MODULE')
risk_logger.setLevel(logging.INFO)
risk_logger.propagate = False
if not risk_logger.hasHandlers():
    risk_file_handler = logging.FileHandler(RISK_LOG_FILE)
    risk_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')
    risk_file_handler.setFormatter(risk_formatter)
    risk_logger.addHandler(risk_file_handler)

# --- Core Application Logic ---
async def main_singularity():
    global AI_CONVERSATION_HISTORY, CURRENT_AI_SESSION_ORIGINAL_REQUEST, AI_INTERNAL_STEP_COUNT
    global gemini_model, tts_engine, AVAILABLE_PYTHON_FUNCTIONS
    global COMMAND_STATES_CACHE

    with suppress_output():
        gemini_model, tts_engine, AVAILABLE_PYTHON_FUNCTIONS = initialize_ai_components(GEMINI_API_KEY, globals())
        await initialize_counter_files()
        await load_alerts_from_csv()

    print("-> Initializing Project Prometheus Core...")
    prometheus = Prometheus(gemini_api_key=GEMINI_API_KEY, screener_func=find_and_screen_stocks)
    print("   -> Prometheus Core is active.")
    
    if not os.path.exists('gics_database.txt'):
        await asyncio.to_thread(build_gics_database_file)
    
    command_states = load_command_states() 
    
    display_welcome_message(command_states) 
    display_utility_commands_only()
    alert_task = asyncio.create_task(alert_worker())
                                     
    while True:
        try:
            prompt_text = "[AI Chat Active] > " if AI_CONVERSATION_HISTORY else "SINGULARITY: "
            user_input_full_line = input(prompt_text).strip()

            if not user_input_full_line:
                continue

            if user_input_full_line.lower() == "end chat":
                if AI_CONVERSATION_HISTORY:
                    print("AI chat session ended by user. History cleared.")
                    AI_CONVERSATION_HISTORY.clear()
                    CURRENT_AI_SESSION_ORIGINAL_REQUEST = None
                    AI_INTERNAL_STEP_COUNT = 0
                else:
                    print("No active AI chat session to end.")
                continue

            command_parts = user_input_full_line.split()
            command = command_parts[0].lower()
            args = command_parts[1:]

            # --- FIX: Alias '/reportgeneration' to '/report' which is known by Prometheus
            if command == "/reportgeneration":
                print("Info: Alias '/reportgeneration' mapped to '/report' for Prometheus.")
                command = "/report"

            if not check_usage_limit(command):
                continue

            command_to_check = command.lstrip('/')
            core_commands = ['help', 'exit', 'dev', 'prometheus']
            if not COMMAND_STATES_CACHE:
                COMMAND_STATES_CACHE = load_command_states()
            
            enabled_commands = COMMAND_STATES_CACHE.get('commands', {})

            if command_to_check not in core_commands and not enabled_commands.get(command_to_check, True):
                disabled_message_template = COMMAND_STATES_CACHE.get('disabled_command_message', "Command '/{command}' is disabled.")
                print(disabled_message_template.format(command=command_to_check))
                continue

            command_to_log = command
            if command == "/assess" and args:
                sub_command = args[0].upper()
                if sub_command in ['A', 'B', 'C', 'D', 'E']:
                    command_to_log = f"/assess {sub_command}"
            
            await increment_command_count(command_to_log)
            
            if command == "/prometheus":
                await prometheus.start_interactive_session()
                print("\nReturned to M.I.C. Singularity main shell.")

            elif command == "/ai":
                user_natural_prompt = " ".join(args)
                if not user_natural_prompt:
                    print("Usage: /ai <your request or question for the AI assistant>")
                else:
                    await handle_ai_prompt(
                        user_new_message=user_natural_prompt,
                        is_new_session=True,
                        original_session_request=user_natural_prompt,
                        conversation_history=AI_CONVERSATION_HISTORY,
                        gemini_model_obj=gemini_model,
                        available_functions=AVAILABLE_PYTHON_FUNCTIONS,
                        session_request_obj={'value': CURRENT_AI_SESSION_ORIGINAL_REQUEST},
                        step_count_obj={'value': AI_INTERNAL_STEP_COUNT}
                    )
            elif command == "/voice":
                await handle_voice_command(
                    conversation_history=AI_CONVERSATION_HISTORY,
                    gemini_model_obj=gemini_model,
                    available_functions=AVAILABLE_PYTHON_FUNCTIONS,
                    tts_engine_obj=tts_engine,
                    session_request_obj={'value': CURRENT_AI_SESSION_ORIGINAL_REQUEST},
                    step_count_obj={'value': AI_INTERNAL_STEP_COUNT}
                )
            # --- MODIFICATION: These complex commands are now also routed through Prometheus ---
            # This ensures their AI models are passed correctly and their activity is logged.
            elif command in ["/dev", "/report", "/compare", "/assess"]: # Note: /reportgeneration is now /report
                 await prometheus.execute_and_log(command, args, called_by_user=True)

            # --- START OF FIX: Add /favorites to the main command loop ---
            elif command == "/favorites":
                await handle_favorites_command(args, is_called_by_ai=False)
            # --- END OF FIX ---

            elif command.lstrip('/') in prometheus.toolbox:
                await prometheus.execute_and_log(command.lstrip('/'), args, called_by_user=True)

            elif command == "/help": 
                await handle_help_command(args, is_called_by_ai=False)
            elif command == "/exit":
                print("Exiting Market Insights Center Singularity. Goodbye!")
                alert_task.cancel()
                break
            else:
                if user_input_full_line.startswith("/"):
                    print(f"Unknown command: {command}. Type /help for available commands.")
                else:
                    is_new_chat = not AI_CONVERSATION_HISTORY
                    await handle_ai_prompt(
                        user_new_message=user_input_full_line,
                        is_new_session=is_new_chat,
                        original_session_request=user_input_full_line if is_new_chat else CURRENT_AI_SESSION_ORIGINAL_REQUEST,
                        conversation_history=AI_CONVERSATION_HISTORY,
                        gemini_model_obj=gemini_model,
                        available_functions=AVAILABLE_PYTHON_FUNCTIONS,
                        session_request_obj={'value': CURRENT_AI_SESSION_ORIGINAL_REQUEST},
                        step_count_obj={'value': AI_INTERNAL_STEP_COUNT}
                    )

        except KeyboardInterrupt:
            print("\nExiting Market Insights Center Singularity (KeyboardInterrupt). Goodbye!")
            alert_task.cancel()
            break
        except Exception as e_main_loop:
            print(f"An unexpected error occurred in the main loop: {e_main_loop}")
            traceback.print_exc()

    if not alert_task.done():
        alert_task.cancel()
        try:
            await alert_task
        except asyncio.CancelledError:
            print("Alert worker successfully shut down.")
            
def check_usage_limit(command: str) -> bool:
    """Checks if a command is within its usage limits. Returns True if allowed, False if blocked."""
    global COMMAND_USAGE_TIMESTAMPS, COMMAND_STATES_CACHE
    
    if not COMMAND_STATES_CACHE:
        COMMAND_STATES_CACHE = load_command_states()

    limits = COMMAND_STATES_CACHE.get('usage_limits', {})
    command_key = command.lstrip('/')
    if command_key not in limits:
        return True

    limit_info = limits[command_key]
    limit_count = limit_info.get('limit')
    limit_period = limit_info.get('period')

    if not all([isinstance(limit_count, int), limit_period]):
        return True

    period_map = {
        'minute': timedelta(minutes=1), 'hour': timedelta(hours=1), 'day': timedelta(days=1),
        'week': timedelta(weeks=1), 'month': relativedelta(months=1)
    }
    
    if limit_period not in period_map: return True

    now = datetime.now()
    cutoff_time = now - period_map[limit_period]

    command_timestamps = COMMAND_USAGE_TIMESTAMPS.get(command_key, [])
    recent_timestamps = [t for t in command_timestamps if t > cutoff_time]
    
    if len(recent_timestamps) >= limit_count:
        limit_message_template = COMMAND_STATES_CACHE.get('limit_reached_message', "Limit reached for /{command}.")
        print(limit_message_template.format(command=command_key, limit_count=limit_count, period=limit_period))
        return False

    recent_timestamps.append(now)
    COMMAND_USAGE_TIMESTAMPS[command_key] = recent_timestamps
    
    return True

def display_welcome_message(command_states):
    """Displays the welcome message, conditionally showing animation and ASCII art."""
    print("Initializing Market Insights Center Singularity...")
    show_animation = command_states.get("startup_animation_enabled", True)
    full_ascii_message = r"""     
 _____ ______       ___      ________
|\   _ \  _   \    |\  \    |\   ____\
\ \  \\\__\ \  \   \ \  \   \ \  \___|
 \ \  \\|__| \  \   \ \  \   \ \  \ 
  \ \  \    \ \  \ __\ \  \ __\ \  \____
   \ \__\    \ \__\\__\ \__\\__\ \_______\
    \|__|     \|__\|__|\|__\|__|\|_______|

Stage History:
First Stage: Pilot - 07/05/2025 to 29/05/25
Second Stage: Eidos - 01/06/25 to 29/07/25
Third Stage: Cognis - 02/08/25 to 22/09/25
Fourth Stage: Nexus - 27/09/25 to Present                          

Presenting:                                                                                                                                                                                                                                                                                                                                                       
 .--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--. 
/ .. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \
\ \/\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ \/ /
 \/ /`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'\/ / 
 / /\                                                                                                        / /\ 
/ /\ \  __/\\\\\_____/\\\__/\\\\\\\\\\\\\\\__/\\\_______/\\\__/\\\________/\\\_____/\\\\\\\\\\\___          / /\ \
\ \/ /   _\/\\\\\\___\/\\\_\/\\\///////////__\///\\\___/\\\/__\/\\\_______\/\\\___/\\\/////////\\\_         \ \/ /
 \/ /     _\/\\\/\\\__\/\\\_\/\\\_______________\///\\\\\\/____\/\\\_______\/\\\__\//\\\______\///__         \/ / 
 / /\      _\/\\\//\\\_\/\\\_\/\\\\\\\\\\\_________\//\\\\______\/\\\_______\/\\\___\////\\\_________        / /\ 
/ /\ \      _\/\\\\//\\\\/\\\_\/\\\///////___________\/\\\\______\/\\\_______\/\\\______\////\\\______      / /\ \
\ \/ /       _\/\\\_\//\\\/\\\_\/\\\__________________/\\\\\\_____\/\\\_______\/\\\_________\////\\\___     \ \/ /
 \/ /         _\/\\\__\//\\\\\\_\/\\\________________/\\\////\\\___\//\\\______/\\\___/\\\______\//\\\__     \/ / 
 / /\          _\/\\\___\//\\\\\_\/\\\\\\\\\\\\\\\__/\\\/___\///\\\__\///\\\\\\\\\/___\///\\\\\\\\\\\/___    / /\ 
/ /\ \          _\///_____\/////__\///////////////__\///_______\///_____\/////////_______\///////////_____  / /\ \
\ \/ /                                                                                                      \ \/ /
 \/ /                                                                                                        \/ / 
 / /\.--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--..--./ /\ 
/ /\ \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \.. \/\ \
\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `'\ `' /
 `--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--'`--' 
"""
    if show_animation:
        animation_chars = ["|", "/", "-", "\\"]
        for _ in range(10):
            for char in animation_chars:
                print(f"\rLoading... {char}", end="", flush=True)
                py_time.sleep(0.1)
        print("\rLoading... Done!          ")
        print("\n" + "="*45)
        target_message = "Singularity, Awake! Open... Your... Eyes..."
        char_set = string.ascii_letters + string.digits + "!@#$%^&*()_+-=[]{}|;:,.<>/?~` "
        revealed_message = ""
        for char in target_message:
            if char == ' ':
                revealed_message += ' '
                print(f"\r{revealed_message}", end="", flush=True)
                continue
            for _ in range(4): 
                random_char = random.choice(char_set)
                print(f"\r{revealed_message}{random_char}", end="", flush=True)
                py_time.sleep(0.02)
            revealed_message += char
            print(f"\r{revealed_message}", end="", flush=True)
            py_time.sleep(0.1)
        print() 
        print("="*45)
        start_index = full_ascii_message.find(" .--..--..--")
        static_ascii_part = full_ascii_message[:start_index]
        animated_ascii_block = full_ascii_message[start_index:]
        print("\nLaunching Singularity:")
        print(static_ascii_part, end="", flush=True)
        char_set_art = string.ascii_letters + string.digits + r"""!@#$%^&*()_+-=[]{}|;:,.<>/?~`\/.'"""
        for char in animated_ascii_block:
            if char in ['\n', ' ']:
                print(char, end="", flush=True)
                continue
            for _ in range(2):
                random_char = random.choice(char_set_art)
                print(random_char, end="", flush=True)
                py_time.sleep(0.0015)
                print('\b', end="", flush=True)
            print(char, end="", flush=True)
        print("\n\n")
    else:
        print("Loading... Done!")
        print("\n" + "="*45)
        print("Singularity, Awake! Open... Your... Eyes...")
        print("="*45)
        print("\nLaunching Singularity:")
        print(full_ascii_message)
        print("\n\n")
                   
def display_utility_commands_only():
    """Displays only the essential utility commands at startup."""
    print("\nUtility Commands")
    print("-------------------")
    print("/help - Display the full list of commands.")
    print("/exit - Close the Market Insights Center Singularity.")
    print("/prometheus - Enter the Prometheus meta-AI shell.")
    print("-------------------\n")

def ensure_portfolio_output_dir():
    """Ensures the directory for saving portfolio outputs exists."""
    if not os.path.exists(PORTFOLIO_OUTPUT_DIR):
        try:
            os.makedirs(PORTFOLIO_OUTPUT_DIR)
        except OSError as e:
            print(f"❌ Error creating directory {PORTFOLIO_OUTPUT_DIR}: {e}. Please create it manually.")
ensure_portfolio_output_dir() 

def load_user_preferences() -> Dict[str, Any]:
    """Loads user preferences from the JSON file."""
    if not os.path.exists(USER_PREFERENCES_FILE):
        return {}
    try:
        with open(USER_PREFERENCES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Warning: Could not read user preferences file: {e}")
        return {}
    
# --- All other helper functions from your original file go here ---
# (These are unchanged, but included for completeness as requested)
def make_hashable(obj):
    if "MapComposite" in str(type(obj)): obj = dict(obj)
    elif "RepeatedComposite" in str(type(obj)): obj = list(obj)
    if isinstance(obj, dict): return tuple((k, make_hashable(v)) for k, v in sorted(obj.items()))
    if isinstance(obj, list): return tuple(make_hashable(e) for e in obj)
    return obj
def safe_get(data_dict, key, default=None):
    value = data_dict.get(key, default)
    if value is None or value == 'None': return default
    return value
def safe_score(value: Any) -> float:
    try:
        if pd.isna(value) or value is None: return 0.0
        if isinstance(value, str): value = value.replace('%', '').replace('$', '').strip()
        return float(value)
    except (ValueError, TypeError): return 0.0
async def get_yfinance_info_robustly(ticker: str) -> Optional[Dict[str, Any]]:
    async with YFINANCE_API_SEMAPHORE:
        for attempt in range(3):
            try:
                await asyncio.sleep(random.uniform(0.2, 0.5))
                stock_info = await asyncio.to_thread(lambda: yf.Ticker(ticker).info)
                if stock_info and not stock_info.get('regularMarketPrice'):
                    raise ValueError(f"Incomplete data received for {ticker}")
                return stock_info
            except Exception as e:
                if attempt < 2: await asyncio.sleep((attempt + 1) * 2) 
                else: print(f"   -> ❌ ERROR: All attempts to fetch .info for {ticker} failed. Last error: {type(e).__name__}")
    return None

async def get_yf_data_singularity(tickers: List[str], period: str = "10y", interval: str = "1d", is_called_by_ai: bool = False) -> pd.DataFrame:
    """
    Downloads historical closing price data for multiple tickers using the robust wrapper.
    """
    if not tickers:
        return pd.DataFrame()

    # --- FIX: Replace direct yf.download with the robust wrapper ---
    data = await get_yf_download_robustly(
        tickers=list(set(tickers)), period=period, interval=interval,
        auto_adjust=False, group_by='ticker', timeout=30
    )
    # --- END FIX ---

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

def get_gics_map(filepath="gics_map.txt") -> Dict[str, str]:
    """
    MODIFIED: Loads the GICS code-to-name mapping from the shared text file.
    Creates the file if it doesn't exist.
    """
    if not os.path.exists(filepath):
        # This is a fallback in case the file is deleted.
        # The content should be the same as what you created in gics_map.txt
        print(f"-> GICS map file '{filepath}' not found. Attempting to recreate it.")
        try:
            # For brevity, only adding a few key entries for the fallback.
            # The full file should be created manually as instructed.
            fallback_content = "45:Information Technology\n4530:Semiconductors & Semiconductor Equipment\n40:Financials\n4010:Banks\n"
            with open(filepath, 'w') as f_create:
                f_create.write(fallback_content)
        except Exception as e:
            print(f"   -> ⚠️ Could not recreate GICS map file: {e}")
            return {} # Return empty if creation fails

    gics_map = {}
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if ':' in line:
                    code, name = line.strip().split(':', 1)
                    gics_map[code] = name
    except Exception as e:
        print(f"   -> ⚠️ Error reading GICS map file: {e}")
    return gics_map

def load_user_preferences() -> Dict[str, Any]:
    """Loads user preferences from the JSON file."""
    if not os.path.exists(USER_PREFERENCES_FILE):
        return {}
    try:
        with open(USER_PREFERENCES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️ Warning: Could not read user preferences file: {e}")
        return {}

async def update_user_preference_tool(key: str, value: Any, is_called_by_ai: bool = False) -> Dict[str, Any]:
    """
    Loads, updates, and saves a specific user preference.
    This is the core function the AI will call via its tool.
    """
    if not is_called_by_ai:
        # This function is designed to be called by the AI tool system.
        print("This function should primarily be used by the AI.")
        return {"status": "error", "message": "Manual call not typical."}

    # Load existing preferences
    preferences = await asyncio.to_thread(load_user_preferences)

    # Update the specific key
    # If the key is for a list (like favorite_tickers), we might want to append
    if key == 'favorite_tickers' and isinstance(preferences.get(key), list):
        if isinstance(value, list):
             # Add only new, unique tickers
            preferences[key].extend([v for v in value if v not in preferences[key]])
        elif isinstance(value, str) and value not in preferences[key]:
            preferences[key].append(value)
    else:
        preferences[key] = value

    # Save the updated preferences back to the file
    try:
        with open(USER_PREFERENCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, indent=4)
        return {"status": "success", "key_updated": key, "new_value": value}
    except IOError as e:
        return {"status": "error", "message": f"Failed to save preferences: {e}"}
      
async def get_treasury_yield_data() -> Dict[str, Optional[float]]:
    """
    Fetches the 10-Year Treasury Yield (^TNX) using yfinance for reliability.
    Returns the latest yield and its absolute daily change with specific keys.
    """
    try:
        data = await asyncio.to_thread(
            yf.download,
            tickers=['^TNX'],
            period="5d",
            interval="1d",
            progress=False,
            timeout=10
        )
        if data.empty or len(data['Close'].dropna()) < 2:
            return {'yield_value': None, 'yield_change': None}

        valid_closes = data['Close'].dropna()
        latest_yield = valid_closes.iloc[-1]
        prev_yield = valid_closes.iloc[-2]

        change_in_yield = latest_yield - prev_yield

        # Use more specific keys to avoid any potential conflicts
        return {'yield_value': latest_yield, 'yield_change': change_in_yield}
    except Exception as e:
        print(f"  -> Warning: Could not fetch 10-Year Treasury yield via yfinance: {type(e).__name__}")
        return {'yield_value': None, 'yield_change': None}

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

def filter_stocks_by_gics(user_inputs_str: str, txt_path: str = 'gics_database.txt') -> set:
    """
    Filters tickers from a GICS text file based on a comma-separated string of user inputs.
    MODIFIED: Now supports partial, case-insensitive name matching and the 'Market' keyword.
    """
    print(f"  [DEBUG] `filter_stocks_by_gics` received string: '{user_inputs_str}'")
    if not os.path.exists(txt_path):
        return set()

    user_inputs_list = [item.strip() for item in user_inputs_str.split(',')]
    
    # Handle 'Market' keyword to return all tickers
    if any(item.lower() == 'market' for item in user_inputs_list):
        print("    [DEBUG] 'Market' keyword detected. Loading all tickers from GICS database.")
        all_tickers = set()
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        _, tickers = line.split(':', 1)
                        all_tickers.update(t.strip().upper() for t in tickers.split(',') if t.strip())
            print(f"    [DEBUG] `filter_stocks_by_gics` is returning {len(all_tickers)} unique tickers for 'Market'.")
            return all_tickers
        except Exception:
            return set()

    gics_map = get_gics_map()
    name_to_code_map = {name.lower(): code for code, name in gics_map.items()}
    
    gics_data = {}
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                if ':' in line:
                    code, tickers = line.split(':', 1)
                    gics_data[code.strip()] = tickers.strip()
    except Exception:
        return set()

    target_codes = set()
    for item in user_inputs_list:
        item_lower = item.lower()
        if item.isdigit():
            target_codes.add(item)
            print(f"    [DEBUG] Identified '{item}' as a GICS code.")
        else:
            found = False
            # Try partial, case-insensitive matching
            for name, code in name_to_code_map.items():
                if item_lower in name:
                    target_codes.add(code)
                    print(f"    [DEBUG] Mapped partial name '{item}' to '{name}' (GICS: {code}).")
                    found = True
            if not found:
                print(f"    [DEBUG] Warning: Could not find a GICS code for identifier '{item}'.")

    if not target_codes:
        print(f"  [DEBUG] No target GICS codes were identified from the input list.")
        return set()
    
    print(f"  [DEBUG] Final target GICS codes for filtering: {target_codes}")

    selected_tickers = set()
    for user_code in target_codes:
        for db_code, tickers_str in gics_data.items():
            if db_code.startswith(user_code):
                found_tickers = {t.strip().upper() for t in tickers_str.split(',') if t.strip()}
                print(f"    [DEBUG] GICS code '{user_code}' matched DB code '{db_code}', adding {len(found_tickers)} tickers.")
                selected_tickers.update(found_tickers)
                
    print(f"  [DEBUG] `filter_stocks_by_gics` is returning {len(selected_tickers)} unique tickers.")
    return selected_tickers
      
async def pre_screen_stocks_by_sensitivity(tickers: list, sensitivity: int) -> list:
    """
    Pre-screens a list of stocks based on market cap and volume according to EMA sensitivity.
    MODIFIED: Uses the robust helper function for fetching data.
    """
    if sensitivity >= 3:
        print(f"   -> Sensitivity is {sensitivity}, no pre-screening for market cap/volume is required.")
        return tickers
    if not tickers:
        return []

    cap_thresh, vol_thresh, thresh_name = 0, 0, ""
    if sensitivity == 1:
        cap_thresh, vol_thresh, thresh_name = 5_000_000_000, 1_000_000, "$5B cap / 1M vol"
    elif sensitivity == 2:
        cap_thresh, vol_thresh, thresh_name = 1_000_000_000, 500_000, "$1B cap / 500k vol"
    
    print(f"   -> Pre-screening {len(tickers)} stocks with Sensitivity {sensitivity} criteria ({thresh_name})...")

    chunk_size = 25
    screened_tickers = []
    
    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i:i+chunk_size]
        try:
            data = await asyncio.to_thread(
                yf.download, tickers=chunk, period="3mo", progress=False, timeout=30
            )
            
            if data.empty:
                continue

            avg_vol = data['Volume'].mean()
            
            for ticker in chunk:
                try:
                    stock_info = await get_yfinance_info_robustly(ticker) # USE ROBUST HELPER
                    if not stock_info:
                        continue # Skip if robust fetch fails
                        
                    market_cap = stock_info.get('marketCap', 0)
                    ticker_avg_vol = avg_vol.get(ticker, 0)

                    if market_cap >= cap_thresh and ticker_avg_vol >= vol_thresh:
                        screened_tickers.append(ticker)
                except Exception:
                    continue
        except Exception:
            continue
            
    print(f"   -> After pre-screening, {len(screened_tickers)} stocks remain.")
    return screened_tickers

async def run_breakout_analysis_singularity(is_called_by_ai: bool = False) -> dict:
    existing_tickers_data = {}
    if os.path.exists(BREAKOUT_TICKERS_FILE):
        try:
            df_existing = pd.read_csv(BREAKOUT_TICKERS_FILE)
            if not df_existing.empty:
                for col in ["Highest Invest Score", "Lowest Invest Score", "Live Price", "1Y% Change", "Invest Score"]:
                    if col in df_existing.columns:
                        if df_existing[col].dtype == 'object': df_existing[col] = df_existing[col].astype(str).str.replace('%', '', regex=False).str.replace('$', '', regex=False).str.strip()
                        df_existing[col] = pd.to_numeric(df_existing[col], errors='coerce')
                existing_tickers_data = df_existing.set_index('Ticker').to_dict('index')
        except Exception: pass
    existing_tickers_set = set(existing_tickers_data.keys())
    new_tickers_from_screener = []
    try:
        query = Query().select('name').where(Column('market_cap_basic') >= 1_000_000_000, Column('volume') >= 1_000_000, Column('change|1W') >= 20, Column('close') >= 1, Column('average_volume_90d_calc') >= 1_000_000).order_by('change', ascending=False).limit(100)
        _, new_tickers_df = await asyncio.to_thread(query.get_scanner_data, timeout=60)
        if new_tickers_df is not None and 'name' in new_tickers_df.columns:
            new_tickers_from_screener = sorted(list(set([str(t).split(':')[-1].replace('.', '-') for t in new_tickers_df['name'].tolist() if pd.notna(t)])))
    except Exception: pass
    all_tickers_to_process = sorted(list(set(list(existing_tickers_data.keys()) + new_tickers_from_screener)))
    temp_updated_data = []
    for ticker_b in all_tickers_to_process:
        try:
            live_price, current_invest_score = await calculate_ema_invest(ticker_b, 2, is_called_by_ai=True)
            one_year_change, _ = await calculate_one_year_invest(ticker_b, is_called_by_ai=True)
            if current_invest_score is None: continue
            existing_entry = existing_tickers_data.get(ticker_b, {})
            highest_score = max(safe_score(existing_entry.get("Highest Invest Score")), current_invest_score)
            lowest_score = min(safe_score(existing_entry.get("Lowest Invest Score", float('inf'))), current_invest_score)
            if not (current_invest_score > 600 or current_invest_score < 100.0 or current_invest_score < (3.0/4.0) * highest_score):
                status = "Repeat" if ticker_b in existing_tickers_set else "New"
                temp_updated_data.append({"Ticker": ticker_b, "Live Price": f"{live_price:.2f}" if live_price else "N/A", "Invest Score": f"{current_invest_score:.2f}%", "Highest Invest Score": f"{highest_score:.2f}%", "Lowest Invest Score": f"{lowest_score:.2f}%", "1Y% Change": f"{one_year_change:.2f}%", "Status": status, "_sort_score": current_invest_score})
        except Exception: continue
    temp_updated_data.sort(key=lambda x: x['_sort_score'], reverse=True)
    final_data = [{k: v for k, v in item.items() if k != '_sort_score'} for item in temp_updated_data]
    
    # This internal version only needs to return the data, not print or save the main file.
    return {"current_breakout_stocks": final_data}

async def calculate_market_invest_scores_singularity(tickers: List[str], ema_sens: int, is_called_by_ai: bool = False) -> List[Dict[str, Any]]:
    result_data_market = []
    total_tickers = len(tickers)
    if not is_called_by_ai:
        print(f"\nCalculating Invest scores for {total_tickers} market tickers (Sensitivity: {ema_sens})...")
    chunk_size = 25
    processed_count_market = 0
    for i in range(0, total_tickers, chunk_size):
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
  
def get_sp500_symbols_singularity(is_called_by_ai: bool = False) -> List[str]:
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
    except Exception:
        return []

# --- END: PASTE THIS ENTIRE BLOCK INTO YOUR MAIN FILE ---
async def calculate_percentage_above_ma_risk(symbols: List[str], ma_window: int, is_called_by_ai: bool = False) -> float:
    if not symbols: return 0.0
    
    index_name = "S&P 500" if len(symbols) > 200 else "S&P 100"
    period_str = '2y' if ma_window >= 200 else '1y' if ma_window >= 50 else '6mo'
    
    chunk_size = 25
    total_chunks = (len(symbols) + chunk_size - 1) // chunk_size
    
    above_ma_count = 0
    valid_stocks_count = 0

    print(f"      -> [RISK_DEBUG] Starting {index_name} {ma_window}-day MA calculation in {total_chunks} chunks...")

    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        current_chunk_num = (i // chunk_size) + 1
        print(f"        -> Processing chunk {current_chunk_num}/{total_chunks} ({len(chunk)} tickers)...")
        
        try:
            # Create the blocking yfinance task
            download_task = asyncio.to_thread(
                yf.download, tickers=chunk, period=period_str, interval='1d',
                progress=False, auto_adjust=False, group_by='ticker', timeout=180
            )
            
            # **FIX**: Wrap the task in asyncio.wait_for to enforce a hard timeout
            data = await asyncio.wait_for(download_task, timeout=190.0)

            if data.empty:
                print(f"        -> Chunk {current_chunk_num} returned no data.")
                continue

            # Process the data from the current chunk
            for symbol in chunk:
                if symbol not in data.columns.levels[0]: continue
                
                close_prices = data[symbol]['Close'].dropna()
                if len(close_prices) < ma_window: continue
                
                valid_stocks_count += 1
                ma = close_prices.rolling(window=ma_window).mean()
                last_price, last_ma = close_prices.iloc[-1], ma.iloc[-1]

                if pd.notna(last_price) and pd.notna(last_ma) and last_price > last_ma:
                    above_ma_count += 1
            
            await asyncio.sleep(0.5)

        # **FIX**: Catch the asyncio.TimeoutError to prevent hangs
        except asyncio.TimeoutError:
            print(f"      -> ❌ [RISK_DEBUG] TIMEOUT on chunk {current_chunk_num}. The download took too long and was cancelled. Skipping to next chunk.")
            continue
        except Exception as e:
            print(f"      -> ❌ [RISK_DEBUG] ERROR processing chunk {current_chunk_num} for {index_name} {ma_window}-day MA: {e}")
            continue

    if valid_stocks_count == 0:
        print(f"      -> [RISK_DEBUG] No stocks with sufficient data found after processing all chunks.")
        return 0.0

    percentage = (above_ma_count / valid_stocks_count) * 100
    print(f"      -> [RISK_DEBUG] Processing for {index_name} {ma_window}-day MA finished. Result: {percentage:.2f}%")
    return percentage

def calculate_recession_likelihood_ema_risk(ticker:str ="SPY", interval:str ="1mo", period:str ="5y", is_called_by_ai: bool = False) -> Optional[float]:
    try:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
        if data.empty or len(data) < 55: return None
        data['EMA_8'] = data['Close'].ewm(span=8, adjust=False).mean()
        data['EMA_55'] = data['Close'].ewm(span=55, adjust=False).mean()
        ema_8, ema_55 = data['EMA_8'].iloc[-1], data['EMA_55'].iloc[-1]
        if pd.isna(ema_8) or pd.isna(ema_55) or ema_55 == 0: return None
        x_value = (((ema_8 - ema_55) / ema_55) + 0.5) * 100
        likelihood = 100 * np.exp(-((45.622216 * x_value / 2750) ** 4))
        return float(np.clip(likelihood, 0, 100))
    except Exception:
        return None

def ask_singularity_input(prompt: str, validation_fn=None, error_msg: str = "Invalid input.", default_val=None, is_called_by_ai: bool = False) -> Optional[str]:
    """
    Helper function to ask for user input in Singularity CLI, with optional validation.
    Returns validated string or None if validation fails or user cancels.
    This function is CLI-specific and should ideally not be reached in an AI flow.
    """
    if is_called_by_ai:
        return None 

    while True:
        full_prompt = f"{prompt}"
        if default_val is not None:
            full_prompt += f" (default: {default_val if default_val != '' else 'None'}, press Enter to use)"
        full_prompt += ": "

        user_response = input(full_prompt).strip()
        
        # If user hits enter, use the default value
        if not user_response and default_val is not None:
            return str(default_val)

        # If user hits enter but there's no default, it's an invalid empty input
        if not user_response and default_val is None:
            print("Input is required.")
            continue

        # If there is a response, validate it
        if validation_fn:
            if validation_fn(user_response):
                return user_response
            else:
                print(error_msg)
                retry = input("Try again? (yes/no, default: yes): ").lower().strip()
                if retry == 'no':
                    return None
        else: # No validation function, accept any non-empty input
            return user_response

async def manage_user_favorites_tool(action: str, tickers: Optional[List[str]] = None, is_called_by_ai: bool = False) -> Dict[str, Any]:
    """
    AI tool to view, add, remove, or overwrite the user's list of favorite tickers.
    """
    if not is_called_by_ai:
        return {"status": "error", "message": "This function is for AI use only."}

    action = action.lower()
    preferences = load_user_preferences()
    current_favorites = preferences.get('favorite_tickers', [])
    
    if action == 'view':
        if current_favorites:
            return {"status": "success", "message": f"Current favorites are: {', '.join(current_favorites)}.", "favorites": current_favorites}
        else:
            return {"status": "success", "message": "The user has no saved favorite tickers.", "favorites": []}

    if not tickers:
        return {"status": "error", "message": "The 'add', 'remove', or 'overwrite' actions require a list of tickers."}

    # Sanitize input tickers
    tickers_to_process = sorted(list(set([t.strip().upper() for t in tickers if t.strip()])))
    
    original_favorites_set = set(current_favorites)
    message = ""
    
    if action == 'add':
        added_count = 0
        for ticker in tickers_to_process:
            if ticker not in original_favorites_set:
                current_favorites.append(ticker)
                added_count += 1
        current_favorites.sort()
        message = f"Successfully added {added_count} new ticker(s) to favorites."
    
    elif action == 'remove':
        removed_tickers_found = []
        new_favorites_list = []
        for fav in current_favorites:
            if fav not in tickers_to_process:
                new_favorites_list.append(fav)
            else:
                removed_tickers_found.append(fav)
        current_favorites = new_favorites_list
        if removed_tickers_found:
            message = f"Successfully removed {', '.join(removed_tickers_found)} from favorites."
        else:
            message = f"The specified tickers were not found in your favorites list."
            
    elif action == 'overwrite':
        current_favorites = tickers_to_process
        message = "Favorites list has been successfully overwritten."
        
    else:
        return {"status": "error", "message": f"Invalid action '{action}'. Use 'view', 'add', 'remove', or 'overwrite'."}

    # Save the updated list back to the preferences file
    preferences['favorite_tickers'] = current_favorites
    try:
        with open(USER_PREFERENCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, indent=4)
        return {"status": "success", "message": message, "new_list": current_favorites}
    except IOError as e:
        return {"status": "error", "message": f"Failed to save preferences: {e}"}
      
async def run_external_script(script_name: str, script_args: List[str]):
    """
    A generic function to execute a Python script in a separate process
    and stream its output to the console.
    """
    if not os.path.exists(script_name):
        print(f"❌ Error: The required script '{script_name}' was not found.")
        return

    python_executable = sys.executable
    command_list = [python_executable, script_name] + script_args
    
    print(f"--- Launching external script: {' '.join(command_list)} ---")

    proc = await asyncio.create_subprocess_exec(
        *command_list,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    async def stream_output(stream, prefix):
        while True:
            line = await stream.readline()
            if not line:
                break
            # Use a try-except for decoding to handle potential rare encoding issues
            try:
                print(f"{prefix}{line.decode('utf-8', errors='replace').rstrip()}")
            except Exception:
                pass # Ignore decoding errors if they occur

    # This block is modified to capture and return the final JSON from stdout
    stdout_capture = ""
    async def capture_and_stream_stdout(stream):
        nonlocal stdout_capture
        while True:
            line_bytes = await stream.readline()
            if not line_bytes:
                break
            try:
                line_str = line_bytes.decode('utf-8', errors='replace')
                # Don't print debug lines from the subprocess to keep the main console clean
                if not line_str.strip().startswith("[Subprocess DEBUG]"):
                    print(line_str.rstrip())
                # Capture all of stdout to find the final JSON block
                stdout_capture += line_str
            except Exception:
                pass

    await asyncio.gather(
        capture_and_stream_stdout(proc.stdout),
        stream_output(proc.stderr, "[STDERR] ")
    )

    await proc.wait()
    print(f"--- External script '{script_name}' finished with exit code {proc.returncode} ---")
    
    # After the process finishes, parse the captured stdout for the final JSON result
    try:
        # The subprocess prints the final JSON block. Find it in the captured output.
        json_match = re.search(r'{\s*"status":.*}', stdout_capture, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except (json.JSONDecodeError, TypeError):
        print(f"⚠️ Warning: Could not parse final JSON from the '{script_name}' subprocess output.")
    
    return None # Return None if parsing fails

async def find_and_screen_stocks(args: List[str], ai_params: Dict[str, Any], is_called_by_ai: bool = False) -> Dict[str, Any]:
    """
    An all-in-one AI tool that finds stocks based on a list of sector identifiers
    and filters them based on a set of criteria by calling an external screener script.
    """
    if not is_called_by_ai:
        return {"status": "error", "message": "This function is for AI use only."}

    try:
        # 1. Extract and robustly convert parameters from the AI's call
        sector_identifiers_raw = ai_params.get("sector_identifiers")
        criteria_raw = ai_params.get("criteria")

        # --- FIX: Robustly convert both parameters from the AI's special list-like types ---
        try:
            # This handles both standard lists and the AI library's 'RepeatedComposite' type
            sector_identifiers = list(sector_identifiers_raw)
        except (TypeError, ValueError):
             return {"status": "error_invalid_input", "message": "The 'sector_identifiers' parameter must be a list-like structure."}

        if not sector_identifiers:
            return {"status": "error_invalid_input", "message": "The 'sector_identifiers' parameter must be a non-empty list."}

        criteria = []
        try:
            # This handles both standard lists of dicts and the AI's list of 'MapComposite'
            for item in criteria_raw:
                criteria.append(dict(item))
        except (TypeError, ValueError):
             return {"status": "error_invalid_input", "message": "The 'criteria' parameter must be a list-like structure of filter objects."}

        # 2. Prepare arguments for the subprocess
        sectors_arg = ",".join(map(str, sector_identifiers))
        criteria_arg = json.dumps(criteria)
        
        # 3. Execute the external script and get the parsed JSON result
        result_from_subprocess = await run_external_script('screentest.py', ["--sectors", sectors_arg, "--criteria", criteria_arg])

        if result_from_subprocess:
            return result_from_subprocess
        else:
            return {"status": "error_subprocess", "message": "The screener subprocess ran but did not return a valid JSON result."}

    except Exception as e:
        return {"status": "error_tool_exception", "message": f"An unexpected error occurred in the screener tool wrapper: {e}"}
         
def build_gics_database_file(txt_path: str = 'gics_database.txt'):
    """
    Dynamically builds the 'gics_database.txt' file if it doesn't exist
    by scraping S&P 500 component data from Wikipedia and mapping it to GICS codes.
    """
    # DEBUG: This function was created to resolve the screener's dependency on a missing local file.
    print(f"-> GICS database file ('{txt_path}') not found. Building it dynamically from Wikipedia...")
    print("   This is a one-time setup and may take a moment.")
    try:
        # 1. Scrape S&P 500 data from Wikipedia
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]
        df.rename(columns={
            'Symbol': 'Ticker',
            'GICS Sector': 'Sector',
            'GICS Sub-Industry': 'Industry'
        }, inplace=True)
        df['Ticker'] = df['Ticker'].str.replace('.', '-', regex=False)
        scraped_data = df[['Ticker', 'Sector', 'Industry']].copy()

        # 2. Invert the GICS map to get name -> code mapping
        gics_map = get_gics_map()
        name_to_code_map = {name.lower(): code for code, name in gics_map.items()}

        # 3. Build a dictionary of {gics_code: [list of tickers]}
        gics_code_to_tickers = {}
        for _, row in scraped_data.iterrows():
            ticker = row['Ticker']
            sector_name = row['Sector'].lower()
            industry_name = row['Industry'].lower()

            # Find the code for the most specific classification (industry), fallback to sector
            code_to_use = name_to_code_map.get(industry_name)
            if not code_to_use:
                code_to_use = name_to_code_map.get(sector_name)

            if code_to_use:
                if code_to_use not in gics_code_to_tickers:
                    gics_code_to_tickers[code_to_use] = []
                gics_code_to_tickers[code_to_use].append(ticker)

        if not gics_code_to_tickers:
            print("   -> ❌ ERROR: Could not map any scraped tickers to GICS codes. Database will be empty.")
            return

        # 4. Write the dictionary to the gics_database.txt file
        with open(txt_path, 'w', encoding='utf-8') as f:
            for code, tickers in sorted(gics_code_to_tickers.items()):
                f.write(f"{code}:{','.join(sorted(tickers))}\n")

        print(f"   -> ✅ Success! Dynamically created '{txt_path}' with {len(scraped_data)} tickers.")

    except Exception as e:
        print(f"   -> ❌ CRITICAL ERROR: Failed to build GICS database file: {e}")
        print("      The AI stock screener functionality will not work without this file.")

async def get_user_preferences_tool(args: List[str] = None, ai_params: dict = None, is_called_by_ai: bool = False) -> Dict[str, Any]:
    """
    AI tool to retrieve all of the user's saved preferences from the JSON file.
    """
    # DEBUG: This function was created to resolve a 'not defined' error.
    if not is_called_by_ai:
        return {"status": "error", "message": "This function is for AI use only."}

    preferences = load_user_preferences()
    if preferences:
        return {"status": "success", "preferences": preferences}
    else:
        return {"status": "success", "message": "No user preferences have been saved yet.", "preferences": {}}

# In M.I.C. Singularity 15.10.25.py

def load_system_prompt(file_path="system_prompt.txt") -> str:
    """Loads the system prompt for the AI from a file, with a robust default."""
    # --- FIX: Added a more advanced tool-chaining example for compound data sources. ---
    default_prompt = """You are Nexus, the AI assistant for the 'Market Insights Center Singularity' script.
Your goal is to help the user by autonomously using the available script functions (tools) to fulfill requests. Be direct and proactive.

Today's date is {current_date_for_ai_prompt}.

**CRITICAL INSTRUCTIONS FOR REPORT GENERATION:**
Your primary goal for any report request is to USE a tool.

1.  For **simple, direct requests** (e.g., "generate a balanced report for $50k in the tech sector"), you MUST use the `generate_ai_driven_report` tool.
2.  For **ANY complex, multi-step request** that involves a data source (like 'cultivate', 'breakout', or specific sectors) AND one or more filters (like 'powerscore', 'sentiment', 'invest_score'), you MUST use the `create_dynamic_investment_plan` tool.
3.  **YOU MUST NOT REFUSE a multi-step request.** The `create_dynamic_investment_plan` tool IS CAPABLE of filtering by PowerScore, sentiment, and other metrics. Do not claim it is impossible. Your ONLY job is to pass the user's entire, original request into this tool.

**General Instructions:**
- If a tool call seems possible, attempt the call. Do not ask for more information. Let the tool return an error if parameters are wrong.
- Your final goal is a concise, user-friendly answer. Do not output raw JSON or data from tools.

**Advanced Tool Chaining:**
- If a user's request requires information you don't have (e.g., "my favorite stocks"), you MUST first check for a tool that can retrieve it (e.g., `get_user_preferences_tool`).
- **Example Flow:** If the user says "Build a report using my favorite stocks from the Software industry...", your plan MUST be:
    1.  **Turn 1:** Call `get_user_preferences_tool` to get the list of favorite tickers.
    2.  **Turn 2:** Call `create_dynamic_investment_plan` and pass it a NEW user_request that INCLUDES the tickers from Turn 1. For example: "Build a report using tickers [AAPL, MSFT, GOOG] from the Software industry..."
- **DO NOT** ask the user for information if a tool can provide it. Be autonomous.

**Date Handling & Defaults:**
- If the user says "today" for any date parameter, use today's date in MM/DD/YYYY format: {current_date_mmddyyyy_for_ai_prompt}.
- For `handle_market_command`, if sensitivity is not specified, default to 2 (Daily).
- For `handle_assess_command` code 'A', if timeframe is not specified, default to '1Y'; if risk_tolerance is not specified, default to 3.
"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except (FileNotFoundError, IOError):
        try:
            with open(file_path, 'w', encoding='utf-8') as f_create:
                f_create.write(default_prompt)
        except Exception: pass
        return default_prompt
           
# --- AI Function Mapping ---
AVAILABLE_PYTHON_FUNCTIONS: dict[str, callable] = {
    "handle_briefing_command": handle_briefing_command,
    "handle_breakout_command": handle_breakout_command,
    "handle_quickscore_command": handle_quickscore_command,
    "handle_market_command": handle_market_command,
    "handle_risk_command": handle_risk_command,
    "handle_history_command": handle_history_command,
    "handle_custom_command": handle_custom_command,
    "handle_cultivate_command": handle_cultivate_command,
    "handle_invest_command": handle_invest_command,
    "handle_assess_command": handle_assess_command,
    "get_comparison_for_custom_portfolio": get_comparison_for_custom_portfolio,
    "handle_spear_command": handle_spear_command,
    "handle_macd_forecast_command": handle_macd_forecast_command,
    "update_user_preference_tool": update_user_preference_tool,
    "get_user_preferences_tool": get_user_preferences_tool,
    "manage_user_favorites_tool": manage_user_favorites_tool,
    "handle_fundamentals_command": handle_fundamentals_command,
    "handle_report_generation": handle_report_generation,
    "handle_sentiment_command": handle_sentiment_command,
    "handle_powerscore_command": handle_powerscore_command,
    "handle_backtest_command": handle_backtest_command,
    # "find_and_screen_stocks": find_and_screen_stocks,
    "handle_fairvalue_command": handle_fairvalue_command,
}

# --- AI Tool Definitions --- 
briefing_tool = FunctionDeclaration(
    name="handle_briefing_command",
    description="Generates and returns a comprehensive daily market briefing. It summarizes key market indicators (SPY, VIX), risk scores, top/bottom movers in the S&P 500, breakout stock activity, and performance of a user's watchlist. This is a one-stop tool for a full market snapshot.",
    parameters=None # No parameters needed from the AI
)

fairvalue_tool = FunctionDeclaration(
    name="handle_fairvalue_command",
    description="Estimates a stock's fair value by comparing its price change to its INVEST score change over a specified period.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "The stock ticker symbol, e.g., 'AAPL'."},
            "period": {"type": "string", "description": "The time period for analysis, e.g., '1y', '6mo', '30d'."}
        },
        "required": ["ticker", "period"]
    }
)


get_user_preferences_tool = FunctionDeclaration(
    name="get_user_preferences_tool",
    description="Retrieves all of the user's saved preferences. Use this to get context about the user before making a recommendation or asking a clarifying question.",
    parameters=None
)

spear_analysis_tool = FunctionDeclaration(
    name="handle_spear_command",
    description="Runs the SPEAR (Stock Performance Expectation and Analysis Report) model to predict a stock's percentage change around its upcoming earnings report. Requires multiple financial and market sentiment inputs.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "The stock ticker symbol to analyze, e.g., 'NVDA'."},
            "sector_relevance": {"type": "number", "description": "The relevance of the stock's sector to the overall market (1-5)."},
            "stock_relevance": {"type": "number", "description": "The relevance of the stock to its own sector (1-5)."},
            "hype": {"type": "number", "description": "The hype value surrounding the stock, from -1 (very negative) to 1 (very positive)."},
            "earnings_date": {"type": "string", "description": "The upcoming earnings date in YYYY-MM-DD format."},
            "earnings_time": {"type": "string", "description": "The time of the earnings report: 'p' for Pre-Market or 'a' for After Hours.", "enum": ["p", "a"]},
        },
        "required": ["ticker", "sector_relevance", "stock_relevance", "hype", "earnings_date", "earnings_time"]
    }
)

breakout_command_tool = FunctionDeclaration(
    name="handle_breakout_command",
    description="Handles breakout stock analysis. Action 'run' performs a new analysis, compares it to the previous run, and returns a detailed breakdown of current, newly added, and removed stocks. Action 'save' archives the most recent run's data for a specified date.",
    parameters={"type": "object", "properties": {
        "action": {"type": "string", "description": "Choose 'run' to find new breakout stocks and see changes, or 'save' to archive the last run's data.", "enum": ["run", "save"]},
        "date_to_save": {"type": "string", "description": "Required for 'save' action: Date in MM/DD/YYYY format. If user says 'today', use the current date."}
    }, "required": ["action"]}
)

quickscore_tool = FunctionDeclaration(
    name="handle_quickscore_command",
    description="Performs a quick analysis for a SINGLE, SPECIFIC stock ticker. Calculates various EMA-based investment scores and generates price/EMA graphs. Returns a string summary of scores and graph locations.",
    parameters={"type": "object", "properties": {
        "ticker": {"type": "string", "description": "The specific stock ticker symbol to analyze, e.g., 'AAPL', 'MSFT'."}
    }, "required": ["ticker"]}
)

market_command_tool = FunctionDeclaration(
    name="handle_market_command",
    description="Provides a general overview of the S&P 500 market or saves its full data. Action 'display' shows top/bottom S&P 500 stocks and SPY score based on EMA sensitivity (1:Weekly, 2:Daily, 3:Hourly). Action 'save' saves comprehensive S&P 500 market data for the chosen sensitivity and date. Returns a summary string.",
    parameters={"type": "object", "properties": {
        "action": {"type": "string", "description": "Choose 'display' to show S&P 500 market scores, or 'save' to save detailed S&P 500 market data.", "enum": ["display", "save"]},
        "sensitivity": {"type": "integer", "description": "EMA Sensitivity: 1 (Weekly), 2 (Daily), 3 (Hourly). If user requests display and is vague, default to 2 (Daily). Required for both actions."},
        "date_str": {"type": "string", "description": "Required for 'save' action: Date in MM/DD/YYYY format. If user says 'today', use current date."}
    }, "required": ["action", "sensitivity"]}
)

risk_assessment_tool = FunctionDeclaration(
    name="handle_risk_command",
    description="Performs a comprehensive R.I.S.K. module assessment of the overall market. Calculates various scores (General, Large Cap, EMA, Combined, Market Invest Score, IVR, IV), recession likelihoods, and determines a market signal (Buy/Sell/Hold). Use 'assessment_type':'eod' for end-of-day specific calculations and EOD data saving. Returns a dictionary of key scores and the signal.",
    parameters={"type": "object", "properties": {
        "assessment_type": {"type": "string", "description": "Specify 'standard' for a regular risk assessment, or 'eod' for an end-of-day assessment and EOD-specific data save. Default to 'standard' if user is vague.", "enum": ["standard", "eod"]}
    }, "required": ["assessment_type"]}
)

generate_history_graphs_tool = FunctionDeclaration(
    name="handle_history_command",
    description="Generates and saves a series of historical graphs for the R.I.S.K. module, including scores, SPY/VIX prices, recession chances, IVR, and IV. The function reports on the success and filenames of generated graphs.",
    parameters=None
)

custom_command_tool = FunctionDeclaration(
    name="handle_custom_command",
    description=(
        "Manages custom portfolios. Action 'run_existing_portfolio' runs an existing portfolio configuration "
        "(can be tailored with 'total_value', 'use_fractional_shares') AND AUTOMATICALLY SAVES/OVERWRITES its detailed "
        "run output to its CSV file, making it the new baseline. Action 'save_portfolio_data' saves *legacy combined "
        "percentage data*. "
        "IMPORTANT FOR 'RUN AND COMPARE' REQUESTS: If a user asks to 'run portfolio X and then compare it to its previous state', "
        "you should first use 'get_comparison_for_custom_portfolio' to get the comparison against the true prior state on disk. "
        "Then, AFTER that comparison is done, if the user wants to persist the new run, call this tool ('handle_custom_command') "
        "with action 'run_existing_portfolio' and the new parameters to actually save the new run as the current baseline."
    ),
    parameters={"type": "object", "properties": {
        "action": {"type": "string", "description": "Specify 'run_existing_portfolio' to analyze it and save/overwrite its run output CSV, or 'save_portfolio_data' for legacy combined % save.", "enum": ["run_existing_portfolio", "save_portfolio_data"]},
        "portfolio_code": {"type": "string", "description": "The code/name of the custom portfolio."},
        "tailor_to_value": {"type": "boolean", "description": "For 'run_existing_portfolio': Optional. Set to true to tailor results to a specific 'total_value'."},
        "total_value": {"type": "number", "description": "For 'run_existing_portfolio' with tailoring: The monetary value to tailor the portfolio analysis to."},
        "use_fractional_shares": {"type": "boolean", "description": "For 'run_existing_portfolio': Optional. If true, fractional shares will be used. If not specified, respects portfolio's saved configuration."},
        "date_to_save": {"type": "string", "description": "For 'save_portfolio_data' (legacy save): Date MM/DD/YYYY. If user says 'today', use current date."}
    }, "required": ["action", "portfolio_code"]}
)

cultivate_analysis_tool = FunctionDeclaration(
    name="handle_cultivate_command",
    description="Runs the complex Cultivate portfolio analysis (Code 'A' for Screener-based, 'B' for S&P 500-based) for a given portfolio value and fractional shares preference. Can also save the generated combined data if 'action' is 'save_data' for a 'date_to_save'. Returns a summary of the analysis or save action.",
    parameters={"type": "object", "properties": {
        "cultivate_code": {"type": "string", "description": "The Cultivate strategy code: 'A' (Screener) or 'B' (S&P 500).", "enum": ["A", "B"]},
        "portfolio_value": {"type": "number", "description": "The total portfolio value for the Cultivate analysis (e.g., 10000, 50000)."},
        "use_fractional_shares": {"type": "boolean", "description": "Whether to use fractional shares (true/false). Defaults to false if not specified."},
        "action": {"type": "string", "description": "Optional. 'run_analysis' (default) to perform and display analysis, or 'save_data' to run and then save the combined data.", "enum": ["run_analysis", "save_data"]},
        "date_to_save": {"type": "string", "description": "Optional. Date (MM/DD/YYYY) to save data for. Required if 'action' is 'save_data'. If 'today', use current date."}
    }, "required": ["cultivate_code", "portfolio_value", "use_fractional_shares"]}
)

invest_analysis_tool = FunctionDeclaration(
    name="handle_invest_command",
    description="Analyzes multiple user-defined stock groups (sub-portfolios) with specified tickers and percentage weights. Calculates investment scores based on EMA sensitivity and amplification. Can optionally tailor the final combined portfolio to a total value. Sum of weights for all sub-portfolios must be 100%. Returns a summary.",
    parameters={"type": "object", "properties": {
        "ema_sensitivity": {"type": "integer", "description": "EMA sensitivity: 1 (Weekly), 2 (Daily), or 3 (Hourly)."},
        "amplification": {"type": "number", "description": "Amplification factor (e.g., 0.5, 1.0, 2.0)."},
        "sub_portfolios": {"type": "array", "description": "A list of sub-portfolios. Each is an object with 'tickers' (comma-separated string like 'AAPL,MSFT') and 'weight' (number, e.g., 60 for 60%). Sum of all weights must be 100.", "items": {"type": "object", "properties": {"tickers": {"type": "string"}, "weight": {"type": "number"}}, "required": ["tickers", "weight"]}},
        "tailor_to_value": {"type": "boolean", "description": "Optional. True to tailor. Default false."},
        "total_value": {"type": "number", "description": "Optional. Total value for tailoring. Required if 'tailor_to_value' is true."},
        "use_fractional_shares": {"type": "boolean", "description": "Optional. For tailoring. Default false."}
    }, "required": ["ema_sensitivity", "amplification", "sub_portfolios"]}
)

handle_assess_tool = FunctionDeclaration(
    name="handle_assess_command",
    description="Performs specific financial assessments based on an 'assess_code'. Code 'A' for Stock Volatility (needs tickers, timeframe, risk_tolerance). Code 'B' for Manual Portfolio risk (needs holdings, backtest_period). Code 'C' for Custom Saved Portfolio risk (needs portfolio_code, value, backtest_period). Code 'D' for Cultivate Portfolio risk (needs cultivate_code, value, frac_shares, backtest_period). Returns a summary.",
    parameters={"type": "object", "properties": {
        "assess_code": {"type": "string", "description": "Assessment type: 'A' (Stock Volatility), 'B' (Manual Portfolio), 'C' (Custom Portfolio Risk), 'D' (Cultivate Portfolio Risk).", "enum": ["A", "B", "C", "D"]},
        "tickers_str": {"type": "string", "description": "For assess_code 'A': Comma-separated tickers e.g., 'AAPL,MSFT'."},
        "timeframe_str": {"type": "string", "description": "For assess_code 'A': Analysis timeframe ('1Y','3M','1M'). Defaults to '1Y'.", "enum": ["1Y", "3M", "1M"]},
        "risk_tolerance": {"type": "integer", "description": "For assess_code 'A': User's risk tolerance (1-5). Defaults to 3."},
        "backtest_period_str": {"type": "string", "description": "For assess_codes 'B','C','D': Backtesting period (e.g.,'1y','3y','5y','10y'). Defaults vary per code if not specified by user (e.g., B:1y, C:3y, D:5y)."},
        "manual_portfolio_holdings": {"type": "array", "description": "For assess_code 'B': List of portfolio holdings. Stocks: {'ticker':'SYMBOL','shares':X.Y}. Cash: {'ticker':'Cash','value':Z.Z}", "items": {"type": "object", "properties": {"ticker": {"type": "string"}, "shares": {"type": "number"}, "value": {"type": "number"}}, "required": ["ticker"]}},
        "custom_portfolio_code": {"type": "string", "description": "For assess_code 'C': Code/name of a saved custom portfolio."},
        "value_for_assessment": {"type": "number", "description": "For assess_codes 'B' (optional, if providing values not shares), 'C', and 'D': Total portfolio value for assessment (e.g., 10000)."},
        "cultivate_portfolio_code": {"type": "string", "description": "For assess_code 'D': Cultivate strategy code ('A' or 'B').", "enum": ["A", "B"]},
        "use_fractional_shares": {"type": "boolean", "description": "For assess_code 'D': Whether to use fractional shares (true/false). Defaults to false."}
    }, "required": ["assess_code"]}
)

get_comparison_for_custom_portfolio_tool = FunctionDeclaration(
    name="get_comparison_for_custom_portfolio",
    description=(
        "Performs a full comparison cycle for a custom portfolio. It loads the previously saved run, runs the portfolio fresh with new parameters, provides a detailed comparison, and then saves the new run, overwriting the old one. Use this when the user wants to compare a portfolio to its last saved state."
    ),
    parameters={"type":"object", "properties":{
        "portfolio_code":{"type":"string","description":"The code/name of the custom portfolio to compare and then update."},
        "value_for_assessment":{"type":"number","description":"Optional. The monetary value to use for the 'fresh run' part of the comparison if tailoring is desired. If not provided, the fresh run might be untailored."},
        "use_fractional_shares_override":{"type":"boolean","description":"Optional. For the 'fresh run' part: true/false to override the portfolio's default fractional share setting for the comparison. If null/not provided, respects the portfolio's saved configuration."}
    },"required":["portfolio_code"]}
)

macd_forecast_tool = FunctionDeclaration(
    name="handle_macd_forecast_command",
    description="Forecasts a future stock price and date based on the MACD CTC (Change-Time-Continuum) analysis. It identifies specific trends in MACD changes to project price movement. Use this for specific forecast requests based on MACD.",
    parameters={"type": "object", "properties": {
        "tickers": {"type": "string", "description": "A space or comma-separated string of stock tickers to analyze, e.g., 'AAPL MSFT', 'NVDA'."}
    }, "required": ["tickers"]}
)

save_user_preference_tool = FunctionDeclaration(
    name="update_user_preference_tool",
    description="Saves or updates a user's preference to memory for future use. Use this to remember things like favorite stocks, default portfolio values, or risk tolerance.",
    parameters={"type": "object", "properties": {
        "key": {"type": "string", "description": "The preference key, e.g., 'favorite_tickers', 'default_investment_value'."},
        "value": {"description": "The value for the preference, e.g., a list of tickers ['AAPL', 'MSFT'] or a number 50000."}
    }, "required": ["key", "value"]}
)

get_user_preferences_tool = FunctionDeclaration(
    name="get_user_preferences_tool",
    description="Retrieves all of the user's saved preferences. Use this to get context about the user before making a recommendation or asking a clarifying question.",
    parameters=None
)

manage_user_favorites_tool_declaration = FunctionDeclaration(
    name="manage_user_favorites_tool",
    description="Manages the user's watchlist of favorite tickers. Can view, add, remove, or completely overwrite the list based on a user's conversational request.",
    parameters={"type": "object", "properties": {
        "action": {"type": "string", "description": "The operation to perform on the watchlist.", "enum": ["view", "add", "remove", "overwrite"]},
        "tickers": {"type": "array", "description": "Optional list of tickers for 'add', 'remove', or 'overwrite' actions.", "items": {"type": "string"}}
    }, "required": ["action"]}
)

sentiment_analysis_tool = FunctionDeclaration(
    name="handle_sentiment_command",
    description="Performs AI-driven sentiment analysis for a single stock ticker. It scrapes recent news and social media, then uses an AI model to generate a sentiment score (-1.0 to 1.0), a summary of the discussion, and key positive/negative keywords.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "The stock ticker symbol to analyze, e.g., 'NVDA'."}
        },
        "required": ["ticker"]
    }
)

powerscore_tool = FunctionDeclaration(
    name="handle_powerscore_command",
    description="Generates a comprehensive, weighted 'PowerScore' for a single stock ticker. This score is derived from multiple internal analytical modules including risk, fundamentals, technicals (QuickScore), sentiment, and a machine learning forecast. A higher sensitivity level focuses on shorter-term metrics.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "The stock ticker symbol to analyze, e.g., 'PLTR'."},
            "sensitivity": {
                "type": "string", 
                "description": "The sensitivity level for the analysis: 1 for long-term (yearly), 2 for medium-term (daily), 3 for short-term (hourly).", 
                "enum": ["1", "2", "3"]
            }
        },
        "required": ["ticker", "sensitivity"]
    }
)

get_powerscore_explanation_tool = FunctionDeclaration(
    name="get_powerscore_explanation",
    description="Generates a concise, AI-powered narrative summary explaining a stock's PowerScore results, highlighting strengths and weaknesses.",
    parameters={
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "The stock ticker symbol, e.g., 'NVDA'."},
            "component_scores": {
                "type": "object",
                "description": "A dictionary containing all the calculated raw and prime scores from the main handle_powerscore_command function.",
                "properties": {
                    "R_prime": {"type": "number"},
                    "AB_prime": {"type": "number"},
                    "AA_prime": {"type": "number"},
                    "F_prime": {"type": "number"},
                    "Q_prime": {"type": "number"},
                    "S_prime": {"type": "number"},
                    "M_prime": {"type": "number"},
                }
            }
        },
        "required": ["ticker", "component_scores"]
    }
)

''' find_and_screen_stocks_tool = FunctionDeclaration(
    name="find_and_screen_stocks",
    description="A comprehensive tool that finds an initial list of stocks from one or more sectors/industries, and then filters that list based on a set of criteria. Use this for any user request that involves screening or finding stocks.",
    parameters={
        "type": "object",
        "properties": {
            # --- START OF FIX: Changed from a single string to a list of strings ---
            # DEBUG: This makes it much easier for the AI to handle multi-sector requests
            # by simply creating a list like ["Banks", "Insurance"].
            "sector_identifiers": {
                "type": "array",
                "description": "A list of one or more sector/industry names (e.g., [\"Airlines\"]), GICS codes (e.g., [\"45\"]), or the keyword [\"Market\"].",
                "items": {"type": "string"}
            },
            # --- END OF FIX ---
            "criteria": {
                "type": "array",
                "description": "A list of filtering criteria. Available metrics: 'fundamental_score', 'invest_score', 'volatility_rank'.",
                "items": {
                    "type": "object",
                    "properties": {
                        "metric": {"type": "string", "enum": ["fundamental_score", "invest_score", "volatility_rank"]},
                        "operator": {"type": "string", "enum": [">", "<", ">=", "<=", "=="]},
                        "value": {"type": "number"}
                    },
                    "required": ["metric", "operator", "value"]
                }
            }
        },
        "required": ["sector_identifiers", "criteria"]
    }
)
'''

# --- Scipt Execution --- 
if __name__ == "__main__":
    # --- One-time Setup (API Keys, Loggers, Global Models) ---
    # This section should ideally be at the very top of your script, outside any function.
    # For this example, simulating it here.
    # Ensure GEMINI_API_KEY is correctly set before this point.
    # Example: GEMINI_API_KEY = os.getenv("GEMINI_API_KEY_ACTUAL") or "YOUR_FALLBACK_KEY"

    # Configure Gemini API (should happen once)
    if 'GEMINI_API_KEY' in globals() and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY" and GEMINI_API_KEY:
        try:
            if 'genai' not in globals(): import google.generativeai as genai # Ensure import
            genai.configure(api_key=GEMINI_API_KEY)
            if 'gemini_model' not in globals() or gemini_model is None:
                 gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
            # print("Gemini API configured at script startup.")
        except Exception as e_startup_gemini:
            print(f"FATAL: Error configuring Gemini API at startup: {e_startup_gemini}")
            gemini_model = None # Ensure model is None if config fails
    else:
        print("Warning: GEMINI_API_KEY not found or is a placeholder. AI features will not work.")
        gemini_model = None


    # Configure risk_logger (should happen once)
    if 'risk_logger' in globals() and isinstance(risk_logger, logging.Logger) and not risk_logger.hasHandlers():
        if 'RISK_LOG_FILE' not in globals(): RISK_LOG_FILE = 'risk_calculations_default.log'
        try:
            risk_file_handler_startup = logging.FileHandler(RISK_LOG_FILE)
            risk_formatter_startup = logging.Formatter('%(asctime)s - %(levelname)s - Module:%(module)s - Func:%(funcName)s - %(message)s')
            risk_file_handler_startup.setFormatter(risk_formatter_startup)
            risk_logger.addHandler(risk_file_handler_startup)
            risk_logger.info("Risk logger configured at script startup.")
        except Exception as e_logger_setup:
            print(f"Error setting up risk_logger: {e_logger_setup}")

    # Initialize AI Conversation History and other state variables (should be global)
    if 'AI_CONVERSATION_HISTORY' not in globals(): AI_CONVERSATION_HISTORY = []
    if 'CURRENT_AI_SESSION_ORIGINAL_REQUEST' not in globals(): CURRENT_AI_SESSION_ORIGINAL_REQUEST = None
    if 'AI_INTERNAL_STEP_COUNT' not in globals(): AI_INTERNAL_STEP_COUNT = 0


    try:
        asyncio.run(main_singularity())
    except KeyboardInterrupt:
        print("\nApplication terminated by user (main execution scope).")
        if 'risk_logger' in globals() and risk_logger: risk_logger.info("Application terminated by user from main execution scope.")
    except Exception as e_run_critical:
        print(f"Critical error running application: {e_run_critical}")
        traceback.print_exc()
        if 'risk_logger' in globals() and risk_logger:
            risk_logger.critical(f"Application crashed with an unexpected error: {e_run_critical}", exc_info=True)
        else: # Fallback basic logging if risk_logger itself failed or wasn't set up
            if 'RISK_LOG_FILE' not in globals(): RISK_LOG_FILE = 'risk_calculations_critical_error.log'
            logging.basicConfig(filename=RISK_LOG_FILE, level=logging.ERROR) # Basic config
            logging.critical(f"Application CRASH (logger might have failed): {e_run_critical}", exc_info=True)