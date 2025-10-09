import json
from datetime import datetime

# Dummy user database with daily credit limits
USERS = {
    "user_123": {"subscription": "free", "limit": 2},
    "user_456": {"subscription": "pro", "limit": 20},
}

# In-memory usage tracking (resets daily)
USAGE = {
    user_id: {"credits_used": 0, "date": None}
    for user_id in USERS
}

def reset_daily_credits(user_id: str):
    today = datetime.utcnow().date()
    usage = USAGE[user_id]

    if usage["date"] != today:
        usage["date"] = today
        usage["credits_used"] = 0

def check_user(user_id: str):
    if user_id not in USERS:
        raise Exception("Invalid user ID")

    reset_daily_credits(user_id)

    user_limit = USERS[user_id]["limit"]
    used = USAGE[user_id]["credits_used"]
    if used >= user_limit:
        raise Exception("No credits left for today.")

    # Increment usage count for this request
    USAGE[user_id]["credits_used"] += 1

    # Calculate remaining credits AFTER increment
    remaining = user_limit - USAGE[user_id]["credits_used"]
    return remaining

def log_request(user_id, endpoint, method, input_data, output_data, remaining_credits):
    entry = {
        "user_id": user_id,
        "endpoint": endpoint,
        "method": method,
        "input": input_data,
        "output": output_data,
        "remaining_credits": remaining_credits,
        "timestamp": datetime.utcnow().isoformat()
    }

    with open("logs.json", "a") as log_file:
        log_file.write(json.dumps(entry) + "\n")
