import os
import sqlite3
from datetime import datetime

DATABASE_PATH = os.getenv("DATABASE_PATH", "visionguard.db")

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the SQLite database schema for Task 5 audit trail."""
    conn = get_db_connection()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                verdict TEXT NOT NULL,
                confidence REAL NOT NULL,
                raw_score REAL NOT NULL,
                model_name TEXT NOT NULL,
                threshold REAL NOT NULL
            );
        """)
    conn.close()
    print(f"[+] Initialized SQLite database at '{DATABASE_PATH}'")

def log_inference(filename, verdict, confidence, raw_score, model_name, threshold=0.5):
    """Sub-task 5: Logs every prediction query permanently into the database."""
    conn = get_db_connection()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    with conn:
        conn.execute("""
            INSERT INTO inferences (timestamp, filename, verdict, confidence, raw_score, model_name, threshold)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (timestamp, filename, verdict, confidence, raw_score, model_name, threshold))
    conn.close()

def get_all_inferences(limit=100):
    """Retrieves prediction history for the web UI."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM inferences ORDER BY id DESC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows

if __name__ == "__main__":
    init_db()
