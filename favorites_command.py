# favorites_command.py

# --- Imports for favorites_command ---
import os
import json
from typing import List, Dict, Any, Optional

# --- Constants (copied for self-containment) ---
USER_PREFERENCES_FILE = 'user_preferences.json'

# --- Helper Functions (copied for self-containment) ---

def load_user_preferences() -> Dict[str, Any]:
    """Loads user preferences from the JSON file."""
    if not os.path.exists(USER_PREFERENCES_FILE):
        return {}
    try:
        with open(USER_PREFERENCES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

async def update_user_preference_tool(key: str, value: Any) -> Dict[str, Any]:
    """
    Loads, updates, and saves a specific user preference.
    """
    preferences = load_user_preferences()
    preferences[key] = value
    try:
        with open(USER_PREFERENCES_FILE, 'w', encoding='utf-8') as f:
            json.dump(preferences, f, indent=4)
        return {"status": "success", "key_updated": key, "new_value": value}
    except IOError as e:
        return {"status": "error", "message": f"Failed to save preferences: {e}"}

# --- Main Command Handler (Unified for User and AI) ---

async def handle_favorites_command(
    args: Optional[List[str]] = None, 
    ai_params: Optional[Dict[str, Any]] = None,
    is_called_by_ai: bool = False
) -> Dict[str, Any]:
    """
    Manages the user's list of favorite tickers.
    Supports viewing, adding, removing, and overwriting the list.
    This function is universal and handles calls from both the CLI and the AI.
    """
    action: Optional[str] = None
    tickers_to_process: set = set()

    # --- Step 1: Parse input from either the user (args) or the AI (ai_params) ---
    if is_called_by_ai and ai_params:
        action = ai_params.get('action', 'view').lower()
        tickers_list = ai_params.get('tickers', [])
        if tickers_list:
            tickers_to_process = {t.strip().upper() for t in tickers_list if t.strip()}
    elif args:
        action = args[0].lower() if args else 'view'
        tickers_str = " ".join(args[1:]) if len(args) > 1 else ''
        if tickers_str:
            tickers_to_process = {t.strip().upper() for t in tickers_str.replace(',', ' ').split() if t.strip()}
    else: # Default action if no args are provided
        action = 'view'

    # --- Step 2: Load current state ---
    preferences = load_user_preferences()
    current_favorites = preferences.get('favorite_tickers', [])
    current_favorites_set = set(current_favorites)
    
    message = ""
    updated_list = current_favorites # Default to the current list

    # --- Step 3: Perform the requested action ---
    if action == 'view':
        if current_favorites:
            message = f"Your current saved list is: {', '.join(sorted(current_favorites))}"
        else:
            message = "You do not have any saved favorite tickers yet."
        
        if not is_called_by_ai:
            print(message)
        return {"status": "success", "message": message, "favorites": sorted(current_favorites)}

    if action in ['add', 'remove', 'overwrite'] and not tickers_to_process:
        error_msg = f"Error: The '{action}' action requires a list of tickers."
        if not is_called_by_ai:
            print(error_msg)
        return {"status": "error", "message": error_msg}

    if action == 'add':
        new_tickers = tickers_to_process - current_favorites_set
        if not new_tickers:
            message = "No new tickers to add. The provided tickers are already in your list."
        else:
            updated_list = sorted(list(current_favorites_set.union(new_tickers)))
            message = f"Added {len(new_tickers)} ticker(s): {', '.join(sorted(list(new_tickers)))}."
    
    elif action == 'remove':
        removed_tickers = current_favorites_set.intersection(tickers_to_process)
        if not removed_tickers:
            message = "None of the specified tickers were found in your favorites list."
        else:
            updated_list = sorted(list(current_favorites_set - removed_tickers))
            message = f"Removed {len(removed_tickers)} ticker(s): {', '.join(sorted(list(removed_tickers)))}."

    elif action == 'overwrite':
        updated_list = sorted(list(tickers_to_process))
        message = "Your favorites list has been overwritten."

    else:
        error_msg = f"Unknown action: '{action}'. Valid actions are view, add, remove, overwrite."
        if not is_called_by_ai:
            print(error_msg)
        return {"status": "error", "message": error_msg}

    # --- Step 4: Save the changes and prepare the response ---
    save_result = await update_user_preference_tool(key='favorite_tickers', value=updated_list)

    if save_result.get("status") == "success":
        final_message = f"Success! {message}"
        if updated_list:
            final_message += f" Your new list is: {', '.join(updated_list)}"
        else:
            final_message += " Your favorites list is now empty."
            
        if not is_called_by_ai:
            print(f"\n✅ {final_message}")
            
        return {"status": "success", "message": final_message, "new_list": updated_list}
    else:
        error_msg = f"Could not save your new list. {save_result.get('message')}"
        if not is_called_by_ai:
            print(f"\n❌ Error: {error_msg}")
        return {"status": "error", "message": error_msg}