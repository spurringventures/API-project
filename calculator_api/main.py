# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import math
import csv
from database import init_db, insert_activity, get_today_usage_count, get_usage_report
from fastapi.responses import FileResponse

# Initialize FastAPI
app = FastAPI(title="Calculator API", description="Square Root & Average API with Daily Limit", version="1.0")

# Initialize Database
init_db()

# ------------------------------
# Request Models
# ------------------------------
class SquareRootRequest(BaseModel):
    username: str
    number: float

class AverageRequest(BaseModel):
    username: str
    numbers: list[float]

# ------------------------------
# Helper Function
# ------------------------------
DAILY_LIMIT = 10

def check_limit(username: str):
    usage_count = get_today_usage_count(username)
    remaining = DAILY_LIMIT - usage_count
    if usage_count >= DAILY_LIMIT:
        raise HTTPException(status_code=403, detail="Daily limit reached. Try again tomorrow.")
    return usage_count, remaining


# ------------------------------
# Endpoints
# ------------------------------
@app.post("/square_root")
def square_root(req: SquareRootRequest):
    usage_count, remaining = check_limit(req.username)
    result = math.sqrt(req.number)
    insert_activity(req.username, "square_root", str(req.number), str(result))
    return {
        "operation": "square_root",
        "input": req.number,
        "result": result,
        "used_today": usage_count + 1,
        "remaining_today": remaining - 1
    }


@app.post("/average")
def average(req: AverageRequest):
    usage_count, remaining = check_limit(req.username)
    if len(req.numbers) == 0:
        raise HTTPException(status_code=400, detail="List of numbers cannot be empty.")
    result = sum(req.numbers) / len(req.numbers)
    insert_activity(req.username, "average", str(req.numbers), str(result))
    return {
        "operation": "average",
        "input": req.numbers,
        "result": result,
        "used_today": usage_count + 1,
        "remaining_today": remaining - 1
    }


@app.get("/report/{username}")
def get_report(username: str):
    report = get_usage_report(username)
    if not report:
        return {"message": f"No data found for user '{username}'"}
    return {"username": username, "usage": report}

@app.get("/download_report/{username}")
def download_report(username: str):
    report = get_usage_report(username)
    if not report:
        raise HTTPException(status_code=404, detail=f"No data found for user '{username}'")
    
    filename = f"{username}_usage_report.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["date", "count"])
        writer.writeheader()
        writer.writerows(report)
    
    return FileResponse(filename, media_type='text/csv', filename=filename)