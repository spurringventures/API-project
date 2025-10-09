from fastapi import FastAPI, Request, HTTPException, Header
from models import Numbers
from database import USERS, USAGE, log_request, check_user, reset_daily_credits
from slowapi import Limiter
from slowapi.util import get_remote_address

app = FastAPI()

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

def validate_numbers(a, b):
    if not (-1000 <= a <= 1000 and -1000 <= b <= 1000):
        raise HTTPException(status_code=400, detail="Inputs must be between -1000 and 1000.")

def validate_user(user_id: str):
    try:
        # check_user increments usage and returns remaining credits
        remaining = check_user(user_id)
        return remaining
    except Exception as e:
        raise HTTPException(status_code=403, detail=str(e))

@app.post("/multiply")

async def multiply(numbers: Numbers, request: Request, x_user_id: str = Header(...)):
    remaining = validate_user(x_user_id)
    validate_numbers(numbers.a, numbers.b)

    result = numbers.a * numbers.b
    log_request(x_user_id, "/multiply", "POST", numbers.dict(), {"result": result}, remaining)

    return {"result": result, "remaining_credits": remaining}

@app.post("/divide")
@limiter.limit("5/minute")
async def divide(numbers: Numbers, request: Request, x_user_id: str = Header(...)):
    remaining = validate_user(x_user_id)
    validate_numbers(numbers.a, numbers.b)

    if numbers.b == 0:
        raise HTTPException(status_code=400, detail="Cannot divide by zero.")

    result = numbers.a / numbers.b
    log_request(x_user_id, "/divide", "POST", numbers.dict(), {"result": result}, remaining)

    return {"result": result, "remaining_credits": remaining}

@app.get("/credits")
async def get_credits(x_user_id: str = Header(...)):
    if x_user_id not in USAGE:
        raise HTTPException(status_code=404, detail="User not found")

    reset_daily_credits(x_user_id)
    usage = USAGE[x_user_id]
    limit = USERS[x_user_id]["limit"]

    return {
        "user_id": x_user_id,
        "subscription": USERS[x_user_id]["subscription"],
        "credits_used_today": usage["credits_used"],
        "credits_remaining": limit - usage["credits_used"],
        "daily_limit": limit,
        "date": str(usage["date"])
    }
