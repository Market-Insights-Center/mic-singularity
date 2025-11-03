# kronos_command.py
# --- Imports ---
import asyncio
import traceback
import json
import os
import shutil
import re
import pandas as pd
import numpy as np
import sqlite3
import pytz
from tabulate import tabulate
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from prometheus_core import Prometheus # Import Prometheus class
from dateutil.relativedelta import relativedelta
# (No risk_command import needed)

# --- Constants ---
PROMETHEUS_STATE_FILE = 'prometheus_state.json'
KRONOS_SCHEDULE_FILE = 'kronos_schedule.json'
SANDBOX_DIR = 'kronos_sandbox' # Directory for test outputs

# Define cache file paths directly to avoid import issues
SP500_CACHE_FILE = 'sp500_risk_cache.csv'
SP100_CACHE_FILE = 'sp100_risk_cache.csv'

# Define default background task intervals
DEFAULT_CORR_INTERVAL_HOURS = 6
DEFAULT_WORKFLOW_CHANCE = 0.1

# --- Kronos Helper Functions ---

def _load_kronos_config() -> Dict[str, Any]:
    """Loads configurable parameters from the Prometheus state file."""
    config = {
        "correlation_interval_hours": DEFAULT_CORR_INTERVAL_HOURS,
        "workflow_analysis_chance": DEFAULT_WORKFLOW_CHANCE
    }
    try:
        if os.path.exists(PROMETHEUS_STATE_FILE):
            with open(PROMETHEUS_STATE_FILE, 'r') as f:
                state = json.load(f)
                config["correlation_interval_hours"] = state.get("correlation_interval_hours", DEFAULT_CORR_INTERVAL_HOURS)
                config["workflow_analysis_chance"] = state.get("workflow_analysis_chance", DEFAULT_WORKFLOW_CHANCE)
    except (IOError, json.JSONDecodeError):
        pass
    return config

def _save_kronos_config(config_key: str, new_value: Any, prometheus_instance: Prometheus):
    """Saves a specific config key to the state file and updates the instance if active."""
    try:
        state = {}
        if os.path.exists(PROMETHEUS_STATE_FILE):
            with open(PROMETHEUS_STATE_FILE, 'r') as f:
                state = json.load(f)
        
        state[config_key] = new_value
        state["is_active"] = prometheus_instance.is_active 
        
        with open(PROMETHEUS_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        
        print(f"✅ Config updated: '{config_key}' set to {new_value}.")
        
        if prometheus_instance.is_active:
            if config_key == 'correlation_interval_hours':
                print("   -> Restarting background correlation task to apply new interval...")
                if prometheus_instance.correlation_task and not prometheus_instance.correlation_task.done():
                    prometheus_instance.correlation_task.cancel()
                # Re-check and restart the task
                required_funcs = [
                    prometheus_instance.derivative_func, prometheus_instance.mlforecast_func, 
                    prometheus_instance.sentiment_func, prometheus_instance.fundamentals_func, 
                    prometheus_instance.quickscore_func
                ]
                if all(required_funcs):
                    prometheus_instance.correlation_task = asyncio.create_task(prometheus_instance.background_correlation_analysis())
                    print("   -> Background task restarted with new interval.")
                else:
                    print("   -> Background task not restarted (required functions missing).")
                    
            elif config_key == 'workflow_analysis_chance':
                # This value is read dynamically, so no restart is needed.
                print("   -> Workflow analysis chance updated for next user command.")
                
    except (IOError, json.JSONDecodeError) as e:
        print(f"❌ Error saving Kronos config: {e}")
    except Exception as e:
        print(f"❌ Unexpected error saving config: {e}")

def _load_schedule() -> List[Dict[str, Any]]:
    """Loads the schedule from kronos_schedule.json."""
    if not os.path.exists(KRONOS_SCHEDULE_FILE):
        return []
    try:
        with open(KRONOS_SCHEDULE_FILE, 'r') as f:
            schedule_data = json.load(f)
            # Re-parse next_run times into datetime objects
            for job in schedule_data:
                job['next_run'] = datetime.fromisoformat(job['next_run'])
            return schedule_data
    except (IOError, json.JSONDecodeError, TypeError):
        print("⚠️ Warning: Could not load or parse schedule file. Creating a new one.")
        return []

def _save_schedule(schedule_data: List[Dict[str, Any]]):
    """Saves the schedule to kronos_schedule.json."""
    try:
        # Convert datetime objects to strings for JSON serialization
        serializable_data = []
        for job in schedule_data:
            job_copy = job.copy()
            job_copy['next_run'] = job['next_run'].isoformat()
            serializable_data.append(job_copy)
            
        with open(KRONOS_SCHEDULE_FILE, 'w') as f:
            json.dump(serializable_data, f, indent=4)
    except (IOError, TypeError) as e:
        print(f"❌ Error saving schedule: {e}")

def _parse_interval_to_timedelta(interval_str: str) -> Optional[timedelta]:
    """Converts interval string (e.g., '3h', '1d', '30m') to timedelta."""
    match = re.match(r"(\d+)([mhd])", interval_str.lower())
    if not match:
        return None
    try:
        value = int(match.group(1))
        unit = match.group(2)
        if unit == 'm':
            return timedelta(minutes=value)
        elif unit == 'h':
            return timedelta(hours=value)
        elif unit == 'd':
            return timedelta(days=value)
    except (ValueError, TypeError):
        return None
    return None

def _is_market_open() -> bool:
    """Checks if the US stock market is open (Mon-Fri, 8:30-15:00 CST/CDT)."""
    try:
        # Using US/Central as it's less ambiguous with DST than EST/EDT
        tz = pytz.timezone('US/Central') 
        now_ct = datetime.now(tz)
        
        # Market open 8:30 AM, close 3:00 PM (15:00) Central Time
        market_open = now_ct.time() >= datetime.time(8, 30)
        market_close = now_ct.time() < datetime.time(15, 0)
        is_weekday = now_ct.weekday() < 5 # 0=Monday, 4=Friday
        
        return is_weekday and market_open and market_close
    except Exception:
        # Fail safe: if time check fails, assume market is closed
        return False

# --- Kronos Command Handlers ---

async def _handle_kronos_status(parts: List[str], prometheus_instance: Prometheus):
    """Handles the 'status' command in the Kronos shell."""
    current_status_str = "ACTIVE" if prometheus_instance.is_active else "INACTIVE"
    
    if len(parts) == 1:
        print(f"Prometheus is currently {current_status_str}.")
        print("Usage: status <on|off>")
        return

    new_status = parts[1].lower()
    
    if new_status == "on":
        if prometheus_instance.is_active:
            print("Prometheus is already ACTIVE.")
        else:
            print("Activating Prometheus...")
            prometheus_instance.is_active = True
            
            required_funcs = [
                prometheus_instance.derivative_func, prometheus_instance.mlforecast_func, 
                prometheus_instance.sentiment_func, prometheus_instance.fundamentals_func, 
                prometheus_instance.quickscore_func
            ]
            if all(required_funcs):
                if not prometheus_instance.correlation_task or prometheus_instance.correlation_task.done():
                    print("   -> Starting background correlation task...")
                    prometheus_instance.correlation_task = asyncio.create_task(prometheus_instance.background_correlation_analysis())
                else:
                    print("   -> Background correlation task is already running.")
            else:
                print("   -> Background correlation task NOT started (required functions missing).")
                
            print("   -> Loading synthesized commands...")
            prometheus_instance._load_and_register_synthesized_commands_sync()
            print("   -> Context fetching and workflow analysis enabled.")
            prometheus_instance._save_prometheus_state()
            print("✅ Prometheus is now ACTIVE.")
            
    elif new_status == "off":
        if not prometheus_instance.is_active:
            print("Prometheus is already INACTIVE.")
        else:
            print("Deactivating Prometheus...")
            prometheus_instance.is_active = False
            
            if prometheus_instance.correlation_task and not prometheus_instance.correlation_task.done():
                prometheus_instance.correlation_task.cancel()
                print("   -> Background correlation task cancelled.")
            prometheus_instance.correlation_task = None
            
            prometheus_instance.toolbox = prometheus_instance.base_toolbox.copy()
            prometheus_instance.synthesized_commands.clear()
            print("   -> Synthesized commands unloaded.")
            print("   -> Context fetching and workflow analysis disabled.")
            prometheus_instance._save_prometheus_state()
            print("✅ Prometheus is now INACTIVE.")
    else:
        print(f"Unknown status: '{new_status}'. Use 'on' or 'off'.")

async def _handle_kronos_optimize(parts: List[str], prometheus_instance: Prometheus):
    """Handles the 'optimize' command in the Kronos shell."""
    try:
        if len(parts) < 4:
            print("Usage: optimize <strategy_name> <ticker> <period> [generations] [population]")
            print("Example: optimize rsi SPY 1y 10 20")
            return

        strategy_arg = parts[1].lower()
        ticker_arg = parts[2].upper()
        period_arg = parts[3].lower()
        generations_arg = int(parts[4]) if len(parts) > 4 else 10
        population_size_arg = int(parts[5]) if len(parts) > 5 else 20
        num_parents_arg = population_size_arg // 2

        optimizable_strategies = prometheus_instance.optimizable_params_config.get("/backtest", {}).keys()
        if strategy_arg not in optimizable_strategies:
            print(f"❌ Error: Strategy '{strategy_arg}' is not defined as optimizable for /backtest.")
            print(f"   Available strategies for optimization: {', '.join(optimizable_strategies)}")
            return

        await prometheus_instance.run_parameter_optimization(
            command_name="/backtest",
            strategy_name=strategy_arg,
            ticker=ticker_arg,
            period=period_arg,
            generations=generations_arg,
            population_size=population_size_arg,
            num_parents=num_parents_arg
        )

    except (ValueError, TypeError):
        print("❌ Error: Invalid number for generations or population size.")
    except Exception as e:
        print(f"❌ An error occurred during optimization: {e}")
        traceback.print_exc()

async def _handle_kronos_test(parts: List[str], prometheus_instance: Prometheus):
    """Handles the 'test' command in the Kronos shell."""
    try:
        if len(parts) < 4:
            print("Usage: test <command_file.py> <ticker> <period> [mode:manual|auto]")
            print("Example: test backtest_command.py SPY 2y manual")
            return

        filename_to_improve = parts[1]
        ticker_arg = parts[2].upper()
        period_arg = parts[3].lower()
        mode_arg = parts[4].lower() if len(parts) > 4 else "manual"

        if not filename_to_improve.endswith(".py"):
            print("❌ Error: File must be a .py file.")
            return
        if mode_arg not in ["manual", "auto"]:
            print("❌ Error: Mode must be 'manual' or 'auto'.")
            return
            
        print(f"--- Initiating Automated Test for {filename_to_improve} ---")
        print(f"    Mode: {mode_arg.upper()}, Ticker: {ticker_arg}, Period: {period_arg}")

        # 1. Generate Hypothesis
        hypothesis_result = await prometheus_instance.generate_improvement_hypothesis(filename_to_improve)
        if not (isinstance(hypothesis_result, dict) and hypothesis_result.get("status") == "success"):
            print(f"❌ Skipping test because hypothesis failed: {hypothesis_result.get('message', 'Unknown error')}")
            return

        original_code = hypothesis_result.get("original_code")
        hypothesis_text = hypothesis_result.get("hypothesis")
        if not original_code or not hypothesis_text:
            print("❌ Error: Hypothesis generated, but original code or text missing.")
            return

        # 2. Generate Improved Code (to temporary file)
        temp_filepath = await prometheus_instance._generate_improved_code(
            command_filename=filename_to_improve,
            original_code=original_code,
            improvement_hypothesis=hypothesis_text
        )
        if not temp_filepath:
            print(f"❌ Failed to generate or save improved code for {filename_to_improve}.")
            return

        # 3. Compare Performance (if backtestable)
        print("\n-> Checking if code is backtestable for comparison...")
        # Construct the path to the *original* command file
        original_target_path_check = os.path.join(os.path.dirname(__file__), 'Isolated Commands', filename_to_improve)
        
        OriginalStratClass = prometheus_instance._load_strategy_class_from_file(original_target_path_check)
        ImprovedStratClass = prometheus_instance._load_strategy_class_from_file(temp_filepath)
        
        is_backtestable = (OriginalStratClass and hasattr(OriginalStratClass(pd.DataFrame()), 'generate_signals') and
                           ImprovedStratClass and hasattr(ImprovedStratClass(pd.DataFrame()), 'generate_signals'))

        if not is_backtestable:
            print("-> Files do not appear to be standard backtest strategies. Skipping performance comparison.")
            print(f"   -> Generated code saved temporarily for manual review: {temp_filepath}")
            return

        print(f"-> Files appear backtestable. Running comparison on {ticker_arg} ({period_arg})...")
        comparison_results = await prometheus_instance._compare_command_performance(
            original_filename=filename_to_improve,
            improved_filepath=temp_filepath,
            ticker=ticker_arg,
            period=period_arg
        )

        if not comparison_results:
            print("⚠️ Comparison failed or produced no results. Aborting approval step.")
            print(f"   -> Improved code remains available at: {temp_filepath}")
            return

        # 4. Handle Approval and Overwrite
        original_results, improved_results = comparison_results
        
        is_improved = improved_results.get('sharpe_ratio', -np.inf) > original_results.get('sharpe_ratio', -np.inf)
        
        print("\n--- Confirmation ---")
        original_target_path = os.path.join(os.path.dirname(__file__), 'Isolated Commands', filename_to_improve)
        
        user_approval = "no"
        if mode_arg == "auto":
            if is_improved:
                print(f"   -> AUTO-APPROVE: Improved Sharpe Ratio ({improved_results.get('sharpe_ratio', -np.inf):.3f} > {original_results.get('sharpe_ratio', -np.inf):.3f}).")
                user_approval = "yes"
            else:
                print(f"   -> AUTO-REJECT: No improvement in Sharpe Ratio.")
                user_approval = "no"
        else: # Manual mode
            prompt_message = f"❓ Overwrite original file '{original_target_path}' with the improved version? (yes/no): "
            user_approval = await asyncio.to_thread(input, prompt_message)

        if user_approval.lower() == 'yes':
            try:
                print(f"   -> Overwriting '{original_target_path}'...")
                shutil.move(temp_filepath, original_target_path)
                print(f"✅ Original file overwritten successfully.")
            except Exception as e_move:
                print(f"❌ Error overwriting file: {e_move}")
                print(f"   -> Improved code remains available at: {temp_filepath}")
        else:
            print("   -> Overwrite cancelled/rejected.")
            print(f"   -> Improved code remains available at: {temp_filepath}")

    except Exception as e:
        print(f"❌ An error occurred during the test: {e}")
        traceback.print_exc()

async def _handle_kronos_config(parts: List[str], prometheus_instance: Prometheus):
    """Handles the 'config' command in the Kronos shell."""
    current_config = _load_kronos_config()
    
    if len(parts) == 1:
        print("\n--- Current Kronos Configuration ---")
        print(f"  correlation_interval_hours = {current_config.get('correlation_interval_hours')}")
        print(f"  workflow_analysis_chance   = {current_config.get('workflow_analysis_chance')}")
        print("\nUsage: config <key> <value>")
        print("Example: config correlation_interval_hours 8")
        return
        
    if len(parts) != 3:
        print("Usage: config <key> <value>")
        return

    key, value_str = parts[1].lower(), parts[2]
    
    try:
        if key == "correlation_interval_hours":
            new_value = float(value_str)
            if new_value < 0.1: raise ValueError("Interval must be at least 0.1 hours.")
            _save_kronos_config(key, new_value, prometheus_instance)
            
        elif key == "workflow_analysis_chance":
            new_value = float(value_str)
            if not (0.0 <= new_value <= 1.0): raise ValueError("Chance must be between 0.0 and 1.0.")
            _save_kronos_config(key, new_value, prometheus_instance)
            
        else:
            print(f"❌ Error: Unknown config key '{key}'.")
            print("   Available keys: correlation_interval_hours, workflow_analysis_chance")

    except ValueError as e:
        print(f"❌ Error: Invalid value '{value_str}'. {e}")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

async def _handle_kronos_schedule(parts: List[str], prometheus_instance: Prometheus):
    """Handles the 'schedule' command for managing cron-like tasks."""
    if len(parts) < 2 or parts[1].lower() not in ['add', 'list', 'remove']:
        print("Usage: schedule <add|list|remove> [options]")
        print("  - add <interval> \"<command>\" [--market-hours] : e.g., schedule add 15m \"/risk\" --market-hours")
        print("  - list                                   : Show active scheduled jobs.")
        print("  - remove <job_id>                        : Remove a job by its ID.")
        return

    action = parts[1].lower()
    schedule = _load_schedule()

    if action == 'list':
        if not schedule:
            print("No commands are currently scheduled.")
            return
        print("\n--- Active Command Schedule ---")
        table_data = []
        for i, job in enumerate(schedule):
            table_data.append([
                i + 1,
                job['command_str'],
                job['interval_str'],
                job['next_run'].strftime('%Y-%m-%d %H:%M:%S'),
                "Yes" if job.get('market_hours_only', False) else "No" # Show market hours status
            ])
        print(tabulate(table_data, headers=["Job ID", "Command", "Interval", "Next Run (UTC)", "Market Hours Only"], tablefmt="grid"))

    elif action == 'remove':
        if len(parts) < 3:
            print("Usage: schedule remove <job_id>")
            return
        try:
            job_id_to_remove = int(parts[2])
            if 1 <= job_id_to_remove <= len(schedule):
                removed_job = schedule.pop(job_id_to_remove - 1)
                _save_schedule(schedule)
                print(f"✅ Removed scheduled job: \"{removed_job['command_str']}\"")
            else:
                print(f"❌ Error: Invalid Job ID. Use 'schedule list' to see valid IDs.")
        except ValueError:
            print("❌ Error: Job ID must be a number.")

    elif action == 'add':
        # Re-join all parts to find the quoted command
        full_input_str = " ".join(parts[2:])
        command_match = re.search(r"\"(.*?)\"", full_input_str)
        if not command_match:
            print("Usage: schedule add <interval> \"<command>\" [--market-hours]")
            print("Error: Command must be enclosed in double quotes.")
            return
        
        command_str = command_match.group(1)
        # Get the part *before* the quoted command as the interval
        interval_str = full_input_str[:command_match.start()].strip()
        # Get the part *after* the quoted command for flags
        flags_str = full_input_str[command_match.end():].strip()
        
        market_hours_only = "--market-hours" in flags_str
        
        interval_delta = _parse_interval_to_timedelta(interval_str)
        if not interval_delta or interval_delta.total_seconds() < 60:
            print("❌ Error: Invalid interval. Must be at least '1m' and use 'm', 'h', or 'd'.")
            return
            
        if not command_str.startswith(('/', '#')): # Allow internal commands
            print("❌ Error: Command must be a valid command string starting with '/' (e.g., \"/briefing\").")
            return
            
        # Add the new job
        new_job = {
            "command_str": command_str,
            "interval_str": interval_str,
            "interval_seconds": interval_delta.total_seconds(),
            "next_run": datetime.utcnow() + interval_delta, # Set first run
            "market_hours_only": market_hours_only # Store the new flag
        }
        schedule.append(new_job)
        _save_schedule(schedule)
        print(f"✅ Scheduled job added. Next run at: {new_job['next_run'].strftime('%Y-%m-%d %H:%M:%S')} UTC")
        if market_hours_only:
            print("   -> This job will only run during US market hours (Mon-Fri, 8:30-15:00 US/Central).")

async def _handle_kronos_analyze(parts: List[str], prometheus_instance: Prometheus):
    """Handles the 'analyze' command for querying the log database."""
    if len(parts) < 2 or parts[1].lower() not in ['logs']:
        print("Usage: analyze logs <command_name | errors>")
        print("  - <command_name>: e.g., /backtest. Shows performance evolution.")
        print("  - errors          : Shows most common errors.")
        return

    db_path = prometheus_instance.db_path
    if not os.path.exists(db_path):
        print(f"❌ Error: Prometheus database not found at '{db_path}'.")
        return

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        target = " ".join(parts[2:])
        
        if target.lower() == 'errors':
            print(f"\n--- Top 10 Errors (Last 7 Days) ---")
            cursor.execute("""
                SELECT output_summary, COUNT(*) as count
                FROM command_log
                WHERE success = 0 AND timestamp >= ?
                GROUP BY output_summary
                ORDER BY count DESC
                LIMIT 10
            """, ((datetime.now() - timedelta(days=7)).isoformat(),))
            rows = cursor.fetchall()
            if not rows:
                print("No errors found in the last 7 days.")
                return
            table_data = [[row[0][:100] + "...", row[1]] for row in rows]
            print(tabulate(table_data, headers=["Error Message", "Count"], tablefmt="grid"))
            
        elif target.startswith('/'):
            command_name = target
            print(f"\n--- Performance Analysis for {command_name} (Last 30 Days) ---")
            
            # Check for backtest metrics
            if command_name == '/backtest':
                cursor.execute("""
                    SELECT 
                        STRFTIME('%Y-%m-%d', timestamp) as day,
                        AVG(backtest_sharpe_ratio),
                        AVG(backtest_return_pct),
                        COUNT(*)
                    FROM command_log
                    WHERE command = ? AND success = 1 AND backtest_sharpe_ratio IS NOT NULL
                          AND timestamp >= ?
                    GROUP BY day
                    ORDER BY day ASC
                """, (command_name, (datetime.now() - timedelta(days=30)).isoformat()))
                
                rows = cursor.fetchall()
                if not rows:
                    print(f"No successful backtest logs with metrics found for {command_name} in the last 30 days.")
                    return
                
                table_data = [[row[0], f"{row[1]:.3f}", f"{row[2]:.2f}%", row[3]] for row in rows]
                print(tabulate(table_data, headers=["Date", "Avg Sharpe", "Avg Return %", "Runs"], tablefmt="grid"))
            else:
                # Generic command analysis
                cursor.execute("""
                    SELECT 
                        STRFTIME('%Y-%m-%d', timestamp) as day,
                        AVG(duration_ms),
                        SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successes,
                        SUM(CASE WHEN success = 0 THEN 1 ELSE 0 END) as failures
                    FROM command_log
                    WHERE command = ? AND timestamp >= ?
                    GROUP BY day
                    ORDER BY day ASC
                """, (command_name, (datetime.now() - timedelta(days=30)).isoformat()))
                
                rows = cursor.fetchall()
                if not rows:
                    print(f"No logs found for {command_name} in the last 30 days.")
                    return
                
                table_data = [[row[0], f"{row[1]:.0f} ms", row[2], row[3]] for row in rows]
                print(tabulate(table_data, headers=["Date", "Avg Duration", "Successes", "Failures"], tablefmt="grid"))

        else:
            print("Invalid target. Use 'errors' or a command name starting with '/'.")

    except sqlite3.Error as e:
        print(f"❌ Database error: {e}")
    except Exception as e:
        print(f"❌ An error occurred during analysis: {e}")
    finally:
        if conn:
            conn.close()

async def _handle_kronos_cache(parts: List[str]):
    """Handles the 'cache' command for managing data caches."""
    if len(parts) < 2 or parts[1].lower() not in ['list', 'clear']:
        print("Usage: cache <list|clear>")
        return
        
    action = parts[1].lower()
    
    # Define cache files
    cache_files = {
        "S&P 500": SP500_CACHE_FILE,
        "S&P 100": SP100_CACHE_FILE,
        # Add other cache files here as they are created
    }
    
    if action == 'list':
        print("\n--- Data Cache Status ---")
        table_data = []
        for name, filepath in cache_files.items():
            status = "❌ Not Found"
            size_mb = 0
            mod_time = "N/A"
            if os.path.exists(filepath):
                try:
                    stats = os.stat(filepath)
                    size_mb = stats.st_size / (1024 * 1024)
                    mod_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    status = "✅ Found"
                except Exception:
                    status = "⚠️ Error Reading"
            table_data.append([name, status, f"{size_mb:.2f} MB", mod_time])
        print(tabulate(table_data, headers=["Cache Name", "Status", "Size", "Last Modified"], tablefmt="grid"))

    elif action == 'clear':
        print("\n--- Clearing Data Caches ---")
        for name, filepath in cache_files.items():
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"   -> ✅ Removed '{name}' cache ({filepath})")
                except Exception as e:
                    print(f"   -> ❌ Failed to remove '{name}' cache: {e}")
            else:
                print(f"   -> ℹ️ '{name}' cache not found, skipping.")
        print("Cache clearing complete.")

# --- Main Kronos Shell ---

async def handle_kronos_command(args: List[str], prometheus_instance: Prometheus):
    """
    Opens the interactive Kronos shell to manage the Prometheus instance.
    """
    if not prometheus_instance:
        print("❌ CRITICAL: Kronos command cannot start. The Prometheus instance was not provided.")
        return

    print("\n--- Kronos Meta-Control Shell ---")
    print("Welcome, Kronos. Manage Prometheus's autonomy.")
    print("Type 'help' for commands, 'exit' to return to Singularity.")
    
    while True:
        try:
            active_str = "ACTIVE" if prometheus_instance.is_active else "INACTIVE"
            user_input = await asyncio.to_thread(input, f"Kronos ({active_str})> ")
            
            if not user_input:
                continue
                
            parts = user_input.split()
            cmd = parts[0].lower()
            
            if cmd == 'exit':
                print("Exiting Kronos shell.")
                break
                
            elif cmd == 'help':
                print("\n--- Kronos Commands ---")
                print("  status <on|off>      : Toggle Prometheus autonomous features ON or OFF.")
                print("  optimize <strat> <t> <p>... : Run Genetic Algorithm parameter optimization for a /backtest strategy.")
                print("                         (e.g., optimize rsi SPY 1y)")
                print("  test <file> <t> <p> [auto|manual] : Run the full 'Hypothesize -> Generate -> Test -> Overwrite' loop.")
                print("                         (e.g., test backtest_command.py SPY 2y manual)")
                print("  schedule <add|list|remove>... : Manage scheduled tasks.")
                print("                         (e.g., schedule add 4h \"/briefing\" [--market-hours])")
                print("  analyze logs <cmd|errors> : Analyze the Prometheus command log database.")
                print("                         (e.g., analyze logs /backtest)")
                print("  cache <list|clear>   : View or clear data caches (e.g., for /risk).")
                print("  config [key] [value] : View or set automation parameters.")
                print("                         (e.g., config correlation_interval_hours 8)")
                print("  help                 : Show this help message.")
                print("  exit                 : Return to the main Singularity shell.")
                
            elif cmd == 'status':
                await _handle_kronos_status(parts, prometheus_instance)
                
            elif cmd == 'optimize':
                if not prometheus_instance.is_active:
                    print("   -> Cannot optimize. Prometheus is INACTIVE.")
                    continue
                await _handle_kronos_optimize(parts, prometheus_instance)
                
            elif cmd == 'test':
                if not prometheus_instance.is_active:
                    print("   -> Cannot test. Prometheus is INACTIVE.")
                    continue
                await _handle_kronos_test(parts, prometheus_instance)
                
            elif cmd == 'config':
                await _handle_kronos_config(parts, prometheus_instance)
                
            elif cmd == 'schedule':
                await _handle_kronos_schedule(parts, prometheus_instance)
                
            elif cmd == 'analyze':
                await _handle_kronos_analyze(parts, prometheus_instance)
                
            elif cmd == 'cache':
                await _handle_kronos_cache(parts)
                
            else:
                print(f"Unknown Kronos command: '{cmd}'. Type 'help'.")
                
        except EOFError:
            print("\nExiting Kronos shell (EOF).")
            break
        except KeyboardInterrupt:
            print("\nExiting Kronos shell (Interrupt).")
            break
        except Exception as e:
            print(f"❌ An error occurred in the Kronos shell: {e}")
            traceback.print_exc()

# --- Background Scheduler Worker (MODIFIED) ---

async def kronos_scheduler_worker(prometheus_instance: Prometheus):
    """
    The background worker that runs scheduled tasks.
    This should be started as an asyncio.Task in main_singularity.py.
    """
    print("🚀 Kronos Scheduler Worker has been started.")
    while True:
        await asyncio.sleep(60) # Check every minute
        
        # Only run jobs if Prometheus is active
        if not prometheus_instance or not prometheus_instance.is_active:
            continue
            
        schedule = _load_schedule()
        now_utc = datetime.utcnow()
        schedule_updated = False
        
        for job in schedule:
            if now_utc >= job['next_run']:
                print(f"\n[Kronos Scheduler] Triggering job: {job['command_str']}")
                
                # --- NEW: Market Hours Check ---
                is_market_hours_job = job.get('market_hours_only', False)
                if is_market_hours_job and not _is_market_open():
                    print(f"   -> Skipping job: '{job['command_str']}'. Reason: Market is closed.")
                    # Reschedule for the next interval
                    interval_delta = timedelta(seconds=job['interval_seconds'])
                    job['next_run'] = now_utc + interval_delta
                    schedule_updated = True
                    continue # Skip this job
                # --- END NEW ---

                try:
                    # --- FIX: Robust command parsing for sub-shells ---
                    command_str = job['command_str']
                    
                    # Split the command from its arguments
                    parts = command_str.split()
                    command_with_slash = parts[0]
                    args = parts[1:]

                    # Check for meta-commands (commands that are shells themselves)
                    if command_with_slash == "/prometheus":
                        # This is a command *for* the Prometheus shell
                        # Re-map it to the *actual* Singularity command.
                        if args and args[0] == "generate" and args[1] == "memo":
                            print(f"   -> Remapping {command_str} to /memo")
                            await prometheus_instance.execute_and_log("/memo", [], called_by_user=False, internal_call=True)
                        # Add other mappings here as needed
                        # e.g., elif args and args[0] == "analyze" and args[1] == "patterns":
                        #   await prometheus_instance.analyze_workflows()
                        else:
                            print(f"   -> ERROR: Scheduled /prometheus command '{' '.join(args)}' is not recognized by Kronos.")

                    elif command_with_slash == "/kronos":
                        # This is a command *for* the Kronos shell itself
                        if args and args[0] == "test":
                            print(f"   -> Executing Kronos command: test {' '.join(args[1:])}")
                            await _handle_kronos_test(args, prometheus_instance)
                        # Add other internal Kronos commands here if needed
                        else:
                            print(f"   -> ERROR: Scheduled /kronos command '{' '.join(args)}' is not supported for automation.")
                    
                    else:
                        # This is a standard Singularity command (e.g., /risk, /briefing)
                        print(f"   -> Executing standard command: {command_with_slash}")
                        await prometheus_instance.execute_and_log(
                            command_name_with_slash=command_with_slash,
                            args=args,
                            called_by_user=False,
                            internal_call=True
                        )
                    # --- END FIX ---
                    
                except Exception as e:
                    print(f"❌ [Kronos Scheduler] Error running job '{job['command_str']}': {e}")
                
                # Reschedule for the next interval
                interval_delta = timedelta(seconds=job['interval_seconds'])
                job['next_run'] = now_utc + interval_delta
                schedule_updated = True
        
        if schedule_updated:
            _save_schedule(schedule)