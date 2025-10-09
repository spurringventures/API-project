from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# ------------------------------
# Configuration
# ------------------------------
usage_count = 0
DAILY_LIMIT = 10

# ------------------------------
# Models
# ------------------------------
class CalculatorInput(BaseModel):
    a: float
    b: float

# ------------------------------
# Helper Function
# ------------------------------
def check_limit():
    global usage_count
    if usage_count >= DAILY_LIMIT:
        print("⚠️ API usage limit reached! Notify admin or client.")  # Notification in console
        raise HTTPException(status_code=403, detail="Daily API limit reached! Please try again tomorrow.")
    usage_count += 1

# ------------------------------
# API Endpoints
# ------------------------------

@app.post("/add")
def add_numbers(data: CalculatorInput):
    check_limit()
    result = data.a + data.b
    return {
        "operation": "addition",
        "a": data.a,
        "b": data.b,
        "result": result,
        "remaining_credits": DAILY_LIMIT - usage_count
    }

@app.post("/subtract")
def subtract_numbers(data: CalculatorInput):
    check_limit()
    result = data.a - data.b
    return {
        "operation": "subtraction",
        "a": data.a,
        "b": data.b,
        "result": result,
        "remaining_credits": DAILY_LIMIT - usage_count
    }

@app.get("/usage")
def get_usage():
    return {
        "total_used": usage_count,
        "remaining": DAILY_LIMIT - usage_count
    }
