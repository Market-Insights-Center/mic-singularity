# feedback_command.py

import sqlite3
from typing import List, Dict, Any

def handle_feedback_command(args: List[str], is_called_by_ai: bool = False) -> Dict[str, Any]:
    """
    Allows the user to provide feedback on the result of a previously executed action.
    """
    if is_called_by_ai:
        return {"status": "error", "message": "This command is intended for direct user feedback."}

    # --- FIX: Robust argument parsing ---
    if len(args) < 2:
        return {"status": "error", "message": "Usage: /feedback <action_id> <rating_1_to_5> [optional comment]"}

    try:
        action_id = int(args[0])
        rating = int(args[1])
        comment = " ".join(args[2:]) if len(args) > 2 else None

        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5.")

    except ValueError:
        return {"status": "error", "message": "Invalid input. The first argument must be a number (Action ID) and the second must be a rating from 1 to 5."}

    db_path = 'prometheus_kb.sqlite'
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT id FROM actions WHERE id = ?", (action_id,))
        if cursor.fetchone() is None:
            return {"status": "error", "message": f"Action ID {action_id} not found in the knowledge base."}

        cursor.execute(
            "UPDATE actions SET user_feedback_rating = ?, user_feedback_comment = ? WHERE id = ?",
            (rating, comment, action_id)
        )
        conn.commit()

        return {"status": "success", "message": f"Feedback successfully recorded for Action ID {action_id}."}

    except sqlite3.Error as e:
        return {"status": "error", "message": f"Database error: {e}"}
    finally:
        if conn:
            conn.close()