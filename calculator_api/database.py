# database.py
import sqlite3
from datetime import datetime, date

DB_NAME = "calculator.db"

# ------------------------------
# Initialize the database
# ------------------------------
def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS user_activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            operation TEXT,
            input_data TEXT,
            output_data TEXT,
            timestamp TEXT,
            date TEXT
        )
    """)
    conn.commit()
    conn.close()

# ------------------------------
# Insert user activity
# ------------------------------
def insert_activity(username, operation, input_data, output_data):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    today = date.today().isoformat()

    c.execute("""
        INSERT INTO user_activity (username, operation, input_data, output_data, timestamp, date)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (username, operation, input_data, output_data, timestamp, today))
    conn.commit()
    conn.close()

# ------------------------------
# Count user activity for today
# ------------------------------
def get_today_usage_count(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    today = date.today().isoformat()

    c.execute("""
        SELECT COUNT(*) FROM user_activity WHERE username = ? AND date = ?
    """, (username, today))
    count = c.fetchone()[0]
    conn.close()
    return count

# ------------------------------
# Get usage report for a user
# ------------------------------
def get_usage_report(username):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        SELECT date, COUNT(*) FROM user_activity
        WHERE username = ?
        GROUP BY date
    """, (username,))
    data = c.fetchall()
    conn.close()

    report = [{"date": d, "count": c} for d, c in data]
    return report
