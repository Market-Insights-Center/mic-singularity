
# --- Imports for favorites_command ---
import os
import json
from typing import List, Dict, Any

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

# --- Main Command Handler ---

async def handle_favorites_command(args: List[str], is_called_by_ai: bool = False, **kwargs):
    """
    Manages the user's list of favorite tickers from the CLI.
    Supports viewing, adding, removing, and overwriting the list.
    """
    if is_called_by_ai:
        # This function is designed for CLI interaction. The AI uses a different tool.
        return {"status": "error", "message": "This specific handler is for CLI use. AI should use 'manage_user_favorites_tool'."}

    print("\n--- Manage Favorite Tickers ---")
    
    action = args[0].lower() if args else 'view'
    tickers_str = " ".join(args[1:]) if len(args) > 1 else ''

    preferences = load_user_preferences()
    current_favorites = preferences.get('favorite_tickers', [])
    current_favorites_set = set(current_favorites)

    if action == 'view':
        if current_favorites:
            print(f"Your current saved list is: {', '.join(sorted(current_favorites))}")
        else:
            print("You do not have any saved favorite tickers yet.")
        return

    if action in ['add', 'remove', 'overwrite'] and not tickers_str:
        print(f"Error: The '{action}' action requires a comma-separated list of tickers.")
        print("Usage: /favorites add AAPL,MSFT")
        return

    tickers_to_process = {t.strip().upper() for t in tickers_str.replace(',', ' ').split() if t.strip()}
    
    if action == 'add':
        new_tickers = tickers_to_process - current_favorites_set
        if not new_tickers:
            print("No new tickers to add. The provided tickers are already in your list.")
            return
        updated_list = sorted(list(current_favorites_set.union(new_tickers)))
        message = f"Added {len(new_tickers)} ticker(s): {', '.join(sorted(list(new_tickers)))}."
    
    elif action == 'remove':
        removed_tickers = current_favorites_set.intersection(tickers_to_process)
        if not removed_tickers:
            print("None of the specified tickers were found in your favorites list.")
            return
        updated_list = sorted(list(current_favorites_set - removed_tickers))
        message = f"Removed {len(removed_tickers)} ticker(s): {', '.join(sorted(list(removed_tickers)))}."

    elif action == 'overwrite':
        updated_list = sorted(list(tickers_to_process))
        message = "Your favorites list has been overwritten."

    else:
        print(f"Unknown action: '{action}'.")
        print("Usage: /favorites [view|add|remove|overwrite] [TICKERS]")
        print("Example: /favorites add AAPL,MSFT")
        return

    save_result = await update_user_preference_tool(key='favorite_tickers', value=updated_list)

    if save_result.get("status") == "success":
        print(f"\n✅ Success! {message}")
        if updated_list:
            print(f"Your new list is: {', '.join(updated_list)}")
        else:
            print("Your favorites list is now empty.")
    else:
        print(f"\n❌ Error: Could not save your new list. {save_result.get('message')}")