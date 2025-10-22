
# --- AI Command Module Imports ---
import asyncio
import json
import traceback
from datetime import datetime
from typing import Optional, List, Dict, Any, Callable

# AI and Speech specific imports
import google.generativeai as genai
from google.generativeai.types import FunctionDeclaration, Tool
import speech_recognition as sr
import pyttsx3

# --- Helper Functions (Copied from main file) ---
# These are necessary for the AI logic to function independently.
async def _continuous_spinner_animation(stop_event: asyncio.Event, message_prefix: str = "AI is processing..."):
    """Helper coroutine for a continuous spinning animation."""
    animation_chars = ["|", "/", "-", "\\"]
    idx = 0
    try:
        while not stop_event.is_set():
            print(f"\r{message_prefix} {animation_chars[idx % len(animation_chars)]}  ", end="", flush=True)
            idx += 1
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        pass
    finally:
        if stop_event.is_set():
            print(f"\r{message_prefix} Done!          ", end="", flush=True)
        else:
            print(f"\r{' ' * (len(message_prefix) + 20)}\r", end="", flush=True)

def make_hashable(obj):
    """ Recursively converts dicts, lists, and proto composites to hashable tuples. """
    if "MapComposite" in str(type(obj)):
        obj = dict(obj)
    elif "RepeatedComposite" in str(type(obj)):
        obj = list(obj)
    if isinstance(obj, dict):
        return tuple((k, make_hashable(v)) for k, v in sorted(obj.items()))
    if isinstance(obj, list):
        return tuple(make_hashable(e) for e in obj)
    return obj

def load_system_prompt(file_path="system_prompt.txt") -> str:
    """Loads the system prompt for the AI from a file, with a robust default."""
    # --- FIX: Upgraded the tool-chaining example to a critical, unbreakable rule. ---
    default_prompt = """You are Nexus, the AI assistant for the 'Market Insights Center Singularity' script.
Your goal is to help the user by autonomously using the available script functions (tools) to fulfill requests. Be direct and proactive.

Today's date is {current_date_for_ai_prompt}.

**CRITICAL INSTRUCTIONS FOR REPORT GENERATION:**
Your primary goal for any report request is to USE a tool.

1.  For **simple, direct requests** (e.g., "generate a balanced report for $50k in the tech sector"), you MUST use the `generate_ai_driven_report` tool.
2.  For **ANY complex, multi-step request** that involves a data source (like 'cultivate', 'breakout', or specific sectors) AND one or more filters (like 'powerscore', 'sentiment', 'invest_score'), you MUST use the `create_dynamic_investment_plan` tool.
3.  **YOU MUST NOT REFUSE a multi-step request.** The `create_dynamic_investment_plan` tool IS CAPABLE of filtering by PowerScore, sentiment, and other metrics. Do not claim it is impossible. Your ONLY job is to pass the user's entire, original request into this tool.

**CRITICAL RULE FOR AUTONOMY:**
- If a user's request requires information you do not currently have (e.g., "my favorite stocks"), you MUST first look for a tool that can retrieve that information (e.g., `get_user_preferences_tool`).
- You MUST call that information-gathering tool first. In the next turn, you will receive the information and can then call the main tool (`create_dynamic_investment_plan`) with all the required data.
- **Under no circumstances should you ask the user for information if a tool can provide it.** Your primary directive is to be autonomous.

**General Instructions:**
- If a tool call seems possible, attempt the call. Do not ask for more information. Let the tool return an error if parameters are wrong.
- Your final goal is a concise, user-friendly answer. Do not output raw JSON or data from tools.

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
    
# --- Initialization Function ---
def initialize_ai_components(api_key: str, main_globals: Dict[str, Any]):
    """
    Initializes and configures all AI-related components.
    Returns the configured Gemini model, TTS engine, and the function mapping dictionary.
    """
    # 1. Configure Gemini Model
    gemini_model = None
    if not api_key or "AIza" not in api_key:
        print("⚠️ Warning: Gemini API key is missing or invalid. AI features will be disabled.")
    else:
        try:
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash') # Using a slightly more advanced model
            print("✔ Gemini API configured successfully.")
        except Exception as e:
            print(f"❌ Error configuring Gemini API: {e}")

    # 2. Configure TTS Engine
    tts_engine = None
    try:
        tts_engine = pyttsx3.init()
        voices = tts_engine.getProperty('voices')
        male_voice = next((v for v in voices if hasattr(v, 'gender') and v.gender == 'male'), None) or \
                     next((v for v in voices if v.name.startswith(('Microsoft David', 'Microsoft Mark'))), None)
        if male_voice:
            tts_engine.setProperty('voice', male_voice.id)
        tts_engine.setProperty('rate', tts_engine.getProperty('rate') + 50)
        print("✔ 'pyttsx3' TTS engine configured successfully.")
    except Exception as e:
        print(f"❌ Error initializing pyttsx3 TTS engine: {e}")
        tts_engine = None # Ensure it's None on failure

    # 3. Define AI Tools (all FunctionDeclaration objects from the main file)
    # This section centralizes all tool definitions.
    briefing_tool = FunctionDeclaration(name="handle_briefing_command", description="Generates and returns a comprehensive daily market briefing.")
    get_user_preferences_tool = FunctionDeclaration(name="get_user_preferences_tool", description="Retrieves all of the user's saved preferences, including 'favorite_tickers'.")
    spear_analysis_tool = FunctionDeclaration(name="handle_spear_command", description="Runs the SPEAR model to predict a stock's percentage change around its upcoming earnings report.", parameters={"type": "object", "properties": {"ticker": {"type": "string"}, "sector_relevance": {"type": "number"}, "stock_relevance": {"type": "number"}, "hype": {"type": "number"}, "earnings_date": {"type": "string"}, "earnings_time": {"type": "string", "enum": ["p", "a"]}}, "required": ["ticker", "sector_relevance", "stock_relevance", "hype", "earnings_date", "earnings_time"]})
    breakout_command_tool = FunctionDeclaration(name="handle_breakout_command", description="Handles breakout stock analysis.", parameters={"type": "object", "properties": {"action": {"type": "string", "enum": ["run", "save"]}, "date_to_save": {"type": "string"}}, "required": ["action"]})
    market_command_tool = FunctionDeclaration(name="handle_market_command", description="Provides a general overview of the S&P 500 market or saves its full data.", parameters={"type": "object", "properties": {"action": {"type": "string", "enum": ["display", "save"]}, "sensitivity": {"type": "integer"}, "date_str": {"type": "string"}}, "required": ["action", "sensitivity"]})
    risk_assessment_tool = FunctionDeclaration(name="handle_risk_command", description="Performs a comprehensive R.I.S.K. module assessment of the overall market.", parameters={"type": "object", "properties": {"assessment_type": {"type": "string", "enum": ["standard", "eod"]}}, "required": ["assessment_type"]})
    generate_history_graphs_tool = FunctionDeclaration(name="handle_history_command", description="Generates and saves a series of historical graphs for the R.I.S.K. module.")
    custom_command_tool = FunctionDeclaration(name="handle_custom_command", description="Manages custom portfolios, including running, saving, and comparing.", parameters={"type": "object", "properties": {"action": {"type": "string", "enum": ["run_existing_portfolio", "save_portfolio_data"]}, "portfolio_code": {"type": "string"}, "tailor_to_value": {"type": "boolean"}, "total_value": {"type": "number"}, "use_fractional_shares": {"type": "boolean"}, "date_to_save": {"type": "string"}}, "required": ["action", "portfolio_code"]})
    cultivate_analysis_tool = FunctionDeclaration(name="handle_cultivate_command", description="Runs the complex Cultivate portfolio analysis.", parameters={"type": "object", "properties": {"cultivate_code": {"type": "string", "enum": ["A", "B"]}, "portfolio_value": {"type": "number"}, "use_fractional_shares": {"type": "boolean"}, "action": {"type": "string", "enum": ["run_analysis", "save_data"]}, "date_to_save": {"type": "string"}}, "required": ["cultivate_code", "portfolio_value", "use_fractional_shares"]})
    invest_analysis_tool = FunctionDeclaration(name="handle_invest_command", description="Analyzes multiple user-defined stock groups (sub-portfolios).", parameters={"type": "object", "properties": {"ema_sensitivity": {"type": "integer"}, "amplification": {"type": "number"}, "sub_portfolios": {"type": "array", "items": {"type": "object", "properties": {"tickers": {"type": "string"}, "weight": {"type": "number"}}, "required": ["tickers", "weight"]}}, "tailor_to_value": {"type": "boolean"}, "total_value": {"type": "number"}, "use_fractional_shares": {"type": "boolean"}}, "required": ["ema_sensitivity", "amplification", "sub_portfolios"]})
    handle_assess_tool = FunctionDeclaration(name="handle_assess_command", description="Performs specific financial assessments based on an 'assess_code'.", parameters={"type": "object", "properties": {"assess_code": {"type": "string", "enum": ["A", "B", "C", "D"]},"tickers_str": {"type": "string"},"timeframe_str": {"type": "string", "enum": ["1Y", "3M", "1M"]},"risk_tolerance": {"type": "integer"},"backtest_period_str": {"type": "string"},"manual_portfolio_holdings": {"type": "array", "items": {"type": "object", "properties": {"ticker": {"type": "string"}, "shares": {"type": "number"}, "value": {"type": "number"}}, "required": ["ticker"]}},"custom_portfolio_code": {"type": "string"},"value_for_assessment": {"type": "number"},"cultivate_portfolio_code": {"type": "string", "enum": ["A", "B"]},"use_fractional_shares": {"type": "boolean"}}, "required": ["assess_code"]})
    get_comparison_for_custom_portfolio_tool = FunctionDeclaration(name="get_comparison_for_custom_portfolio", description="Performs a full comparison cycle for a custom portfolio.", parameters={"type": "object", "properties": {"portfolio_code": {"type": "string"}, "value_for_assessment": {"type": "number"}, "use_fractional_shares_override": {"type": "boolean"}}, "required": ["portfolio_code"]})
    macd_forecast_tool = FunctionDeclaration(name="handle_macd_forecast_command", description="Forecasts a future stock price and date based on the MACD CTC analysis.", parameters={"type": "object", "properties": {"tickers": {"type": "string"}}, "required": ["tickers"]})
    save_user_preference_tool = FunctionDeclaration(name="update_user_preference_tool", description="Saves or updates a user's preference to memory.", parameters={"type": "object", "properties": {"key": {"type": "string"}, "value": {}}, "required": ["key", "value"]})
    manage_user_favorites_tool_declaration = FunctionDeclaration(name="manage_user_favorites_tool", description="Manages the user's watchlist of favorite tickers.", parameters={"type": "object", "properties": {"action": {"type": "string", "enum": ["view", "add", "remove", "overwrite"]}, "tickers": {"type": "array", "items": {"type": "string"}}}, "required": ["action"]})
    get_powerscore_explanation_tool = FunctionDeclaration(name="get_powerscore_explanation", description="Generates a concise, AI-powered narrative summary explaining a stock's PowerScore results.", parameters={"type": "object", "properties": {"ticker": {"type": "string"}, "component_scores": {"type": "object", "properties": {"R_prime": {"type": "number"}, "AB_prime": {"type": "number"}, "AA_prime": {"type": "number"}, "F_prime": {"type": "number"}, "Q_prime": {"type": "number"}, "S_prime": {"type": "number"}, "M_prime": {"type": "number"}}}}, "required": ["ticker", "component_scores"]})
    find_and_screen_stocks_tool = FunctionDeclaration(name="find_and_screen_stocks", description="A comprehensive tool that finds an initial list of stocks from one or more sectors/industries, and then filters that list based on a set of criteria.", parameters={"type": "object", "properties": {"sector_identifiers": {"type": "array", "description": "A list of sector/industry names, GICS codes, or 'Market'.", "items": {"type": "string"}}, "criteria": {"type": "array", "description": "A list of filtering criteria.", "items": {"type": "object", "properties": {"metric": {"type": "string", "enum": ["fundamental_score", "invest_score", "volatility_rank"]}, "operator": {"type": "string", "enum": [">", "<", ">=", "<=", "=="]}, "value": {"type": "number"}}, "required": ["metric", "operator", "value"]}}}, "required": ["sector_identifiers", "criteria"]})
    report_generation_tool = FunctionDeclaration(
        name="generate_ai_driven_report",
        description="Generates a simple, standard PDF investment report. IMPORTANT: Use this for basic requests ONLY. For complex requests that require combining tools (like Cultivate, PowerScore, etc.), you MUST use the 'create_dynamic_investment_plan' tool instead.",
        parameters={
            "type": "object",
            "properties": {
                "risk_tolerance": {"type": "integer", "description": "User's risk tolerance from 1 (low) to 5 (high). Infer from words like 'conservative' (1-2), 'balanced' (3), or 'aggressive' (4-5)."},
                "investment_goals": {"type": "string", "description": "A brief description of the user's financial goals, e.g., 'long-term growth', 'stable income'."},
                "portfolio_value": {"type": "number", "description": "The total monetary value of the portfolio to be allocated."},
                "gics_input": {"type": "string", "description": "A comma-separated string of GICS codes, industry/sector names, or 'Market' to focus the investment."}
            },
            "required": ["risk_tolerance", "portfolio_value", "gics_input"]
        }
    )
    dynamic_plan_tool = FunctionDeclaration(
        name="create_dynamic_investment_plan",
        description="The primary tool for complex investment report generation. Use this when the user's request requires multiple steps, filtering, or combining data from other tools like 'cultivate', 'breakout', or 'powerscore'. This tool dynamically plans and executes the required steps.",
        parameters={
            "type": "object",
            "properties": {
                "user_request": {"type": "string", "description": "The user's full, original request for the investment plan or report. Example: 'Build a report from my Cultivate-A portfolio using only stocks with a PowerScore over 75.'"}
            },
            "required": ["user_request"]
        }
    )

    # Pack all tool declarations into a list for the Tool object
    tool_declarations = [v for v in locals().values() if isinstance(v, FunctionDeclaration)]

    # 4. Map Function Names to Function Objects
    function_names_for_ai = [
        "handle_briefing_command", "handle_breakout_command", "handle_quickscore_command",
        "handle_market_command", "handle_risk_command", "handle_history_command",
        "handle_custom_command", "handle_cultivate_command", "handle_invest_command",
        "handle_assess_command", "get_comparison_for_custom_portfolio", "handle_spear_command",
        "handle_macd_forecast_command", "update_user_preference_tool", "get_user_preferences_tool",
        "manage_user_favorites_tool", "handle_fundamentals_command", "handle_report_generation",
        "handle_sentiment_command", "handle_powerscore_command", "get_powerscore_explanation",
        "handle_backtest_command", 
        "generate_ai_driven_report", "create_dynamic_investment_plan"
    ]
    available_functions = {name: main_globals[name] for name in function_names_for_ai if name in main_globals}
    
    # Store the declaration on the function object itself for easier lookup later
    for declaration in tool_declarations:
        if declaration.name in available_functions:
            setattr(available_functions[declaration.name], '_is_tool', declaration)

    return gemini_model, tts_engine, available_functions


# --- AI and Voice Logic (Moved and Modified) ---
async def handle_ai_prompt(
    user_new_message: str, is_new_session: bool, original_session_request: Optional[str],
    conversation_history: List[Dict], gemini_model_obj, available_functions: Dict[str, Callable],
    session_request_obj: Dict, step_count_obj: Dict
):
    """Handles interaction with the Gemini AI, including function calling and context injection."""
    if not gemini_model_obj:
        print("Error: Gemini model is not configured.")
        return

    tool_declarations = [
        getattr(func, '_is_tool') for func in available_functions.values() if hasattr(func, '_is_tool')
    ]
    all_gemini_tools = Tool(function_declarations=tool_declarations)

    if is_new_session:
        conversation_history.clear()
        step_count_obj['value'] = 0
        session_request_obj['value'] = original_session_request or user_new_message
        base_prompt = load_system_prompt()
        formatted_prompt = base_prompt.replace("{current_date_for_ai_prompt}", datetime.now().strftime('%B %d, %Y')).replace("{current_date_mmddyyyy_for_ai_prompt}", datetime.now().strftime('%m/%d/%Y'))
        conversation_history.extend([
            {"role": "user", "parts": [{"text": formatted_prompt}]},
            {"role": "model", "parts": [{"text": "Understood. I am Nexus. How can I help?"}]}
        ])

    # --- START OF FIX: Context Injection Logic ---
    if conversation_history and len(conversation_history) > 2:
        last_entry = conversation_history[-1]
        if last_entry.get("role") == "tool":
            try:
                tool_response_part = last_entry.get("parts", [{}])[0]
                function_response = tool_response_part.get("function_response", {})
                response_data = function_response.get("response", {})
                
                if function_response.get("name") == "get_user_preferences_tool" and response_data.get("status") == "success":
                    prefs = response_data.get("preferences", {})
                    fav_tickers = prefs.get("favorite_tickers")
                    if fav_tickers:
                        # This is the crucial part: create a new, more specific prompt for the AI's next turn
                        context_prefix = f"CONTEXT: The user's favorite tickers have been retrieved and are: {', '.join(fav_tickers)}. Now, using this explicit list of tickers, fulfill the user's original request to '{session_request_obj['value']}'."
                        user_new_message = context_prefix
                        print(f"   [AI Context Injection]: Adding retrieved favorites to the next step.")
            except Exception as e:
                print(f"   [AI Context DEBUG]: Failed to parse tool history for injection. Error: {e}")
    # --- END OF FIX ---

    conversation_history.append({"role": "user", "parts": [{"text": user_new_message}]})

    max_internal_turns, executed_tool_calls, final_text_response = 10, set(), ""
    stop_spinner_event = asyncio.Event()
    spinner_task = asyncio.create_task(_continuous_spinner_animation(stop_spinner_event, "AI is processing..."))

    try:
        for turn_num in range(max_internal_turns):
            step_count_obj['value'] = turn_num
            response = await asyncio.to_thread(
                gemini_model_obj.generate_content,
                contents=conversation_history, tools=[all_gemini_tools],
                tool_config={"function_calling_config": {"mode": "auto"}}
            )
            
            candidate = response.candidates[0] if response.candidates else None
            if not candidate or not candidate.content or not candidate.content.parts:
                final_text_response = "AI returned no response."
                break

            part = candidate.content.parts[0]
            if part.function_call and part.function_call.name:
                fc = part.function_call
                tool_name, tool_args = fc.name, dict(fc.args)
                call_signature = (tool_name, make_hashable(tool_args))

                if call_signature in executed_tool_calls:
                    error_msg = f"System aborted repetitive tool call to '{tool_name}' to prevent a loop."
                    conversation_history.append({"role": "model", "parts": [{"function_call": fc}]})
                    conversation_history.append({"role": "tool", "parts": [{"function_response": {"name": tool_name, "response": {"error": error_msg}}}]})
                    continue

                executed_tool_calls.add(call_signature)
                conversation_history.append({"role": "model", "parts": [{"function_call": fc}]})

                if tool_name in available_functions:
                    func_to_call = available_functions[tool_name]
                    try:
                        kwargs = { "args": [], "ai_params": tool_args, "is_called_by_ai": True }
                        
                        if tool_name in ["generate_ai_driven_report", "create_dynamic_investment_plan", "handle_compare_command"]:
                            kwargs["gemini_model"] = gemini_model_obj
                        if tool_name == "create_dynamic_investment_plan":
                            kwargs["available_functions"] = available_functions

                        if asyncio.iscoroutinefunction(func_to_call):
                            result = await func_to_call(**kwargs)
                        else:
                            result = await asyncio.to_thread(lambda: func_to_call(**kwargs))
                        
                        response_payload = {}
                        if result is None: response_payload = {"status": "success", "message": f"Tool '{tool_name}' completed."}
                        elif isinstance(result, str): response_payload = {"status": "success", "summary": result}
                        elif isinstance(result, (dict, list)): response_payload = result
                        else: response_payload = {"status": "success", "value": result}
                        
                        conversation_history.append({ "role": "tool", "parts": [{"function_response": { "name": tool_name, "response": response_payload }}]})

                    except Exception as e:
                        error_result = f"Error executing tool '{tool_name}': {traceback.format_exc()}"
                        conversation_history.append({"role": "tool", "parts": [{"function_response": {"name": tool_name, "response": {"error": error_result}}}]})
                else:
                    conversation_history.append({"role": "tool", "parts": [{"function_response": {"name": tool_name, "response": {"error": f"Unknown tool '{tool_name}'."}}}]})
            
            elif part.text:
                final_text_response = part.text
                conversation_history.append({"role": "model", "parts": [{"text": final_text_response}]})
                break
            else:
                break
        else:
            final_text_response = "Max processing turns reached. Summarizing results."
    
    finally:
        stop_spinner_event.set()
        await spinner_task
    
    print("\n--- AI's Final Answer ---")
    print(final_text_response or "AI processing complete. See console for tool outputs.")
    print("-------------------------\n")

def speak_text(text: str, tts_engine_obj):
    """Converts a text string to speech and plays it."""
    if not tts_engine_obj:
        print("AI Response (TTS engine unavailable):", text)
        return
    try:
        tts_engine_obj.say(text)
        tts_engine_obj.runAndWait()
    except Exception as e:
        print(f"❌ Error during text-to-speech playback: {e}")
        print("AI Response (TTS failed):", text)

def listen_for_voice() -> Optional[str]:
    """Listens for voice input from the microphone."""
    if not sr:
        print("Voice recognition library 'SpeechRecognition' not available.")
        return None
    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 1.0
        r.adjust_for_ambient_noise(source, duration=1)
        print("🎤 Listening for wake word ('Nexus')...")
        try:
            audio = r.listen(source, timeout=10, phrase_time_limit=15)
        except sr.WaitTimeoutError:
            return None
    try:
        text = r.recognize_google(audio)
        print(f"👂 Heard: \"{text}\"")
        return text
    except sr.UnknownValueError:
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None


async def parse_and_execute_voice_command(
    command_text: str, conversation_history: List[Dict], gemini_model_obj,
    available_functions: Dict[str, Callable], tts_engine_obj,
    session_request_obj: Dict, step_count_obj: Dict
):
    """Parses transcribed text and calls the AI handler."""
    is_new = not conversation_history
    await handle_ai_prompt(
        user_new_message=command_text, is_new_session=is_new,
        original_session_request=command_text if is_new else session_request_obj['value'],
        conversation_history=conversation_history, gemini_model_obj=gemini_model_obj,
        available_functions=available_functions, session_request_obj=session_request_obj,
        step_count_obj=step_count_obj
    )
    final_response = "Request complete."
    if conversation_history:
        final_model_responses = [p.get("text", "") for e in reversed(conversation_history) if e.get("role") == "model" for p in e.get("parts", []) if "text" in p]
        if final_model_responses:
            final_response = final_model_responses[0]
    speak_text(final_response, tts_engine_obj)


async def handle_voice_command(
    conversation_history: List[Dict], gemini_model_obj, available_functions: Dict[str, Callable],
    tts_engine_obj, session_request_obj: Dict, step_count_obj: Dict
):
    """Main loop for the voice assistant."""
    if not all([sr, tts_engine_obj]):
        print("❌ Cannot start voice assistant. Libraries missing or failed to initialize.")
        return

    speak_text("Voice assistant activated.", tts_engine_obj)
    while True:
        transcribed = listen_for_voice()
        if transcribed:
            text_lower = transcribed.lower()
            if any(phrase in text_lower for phrase in ["stop listening", "end conversation"]):
                speak_text("Deactivating voice assistant.", tts_engine_obj)
                break
            
            if "Nexus" in text_lower:
                command = text_lower.split("Nexus", 1)[-1].strip()
                if command:
                    speak_text(f"Processing: {command}", tts_engine_obj)
                    await parse_and_execute_voice_command(
                        command, conversation_history, gemini_model_obj, available_functions,
                        tts_engine_obj, session_request_obj, step_count_obj
                    )
                else:
                    speak_text("Yes?", tts_engine_obj)
