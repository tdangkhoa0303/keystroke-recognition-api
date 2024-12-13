import base64
import json
from typing import Optional


def extract_session_id(session: Optional[dict]) -> Optional[str]:
    if not session or "access_token" not in session:
        return None

    try:
        access_token = session["access_token"]
        # Split the token into parts
        session_token_parts = access_token.split(".")
        if len(session_token_parts) < 2:
            return None  # Invalid token structure

        # Decode the payload (second part of the token)
        payload_part = session_token_parts[1]
        payload_json = base64.urlsafe_b64decode(payload_part + "==").decode("utf-8")
        payload = json.loads(payload_json)

        # Return the session_id from the payload if it exists
        return payload.get("session_id")
    except (ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"Error decoding token: {error}")
        return None
