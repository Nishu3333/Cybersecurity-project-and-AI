import sqlite3
from contextlib import contextmanager
from utils.config import Config

def init_db():
    conn = sqlite3.connect(Config.DB_PATH)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT UNIQUE,
        amount REAL,
        timestamp TEXT,
        hour INTEGER,
        latenight INTEGER,
        sender_account TEXT,
        receiver_account TEXT,
        country TEXT,
        transaction_type TEXT,
        currency TEXT,
        channel TEXT,
        risk_rating TEXT,
        risk_score REAL,
        prediction INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    cur.execute("""CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id TEXT UNIQUE,
        transaction_id TEXT,
        risk_score REAL,
        amount REAL,
        priority TEXT,
        reason TEXT,
        status TEXT DEFAULT 'Active',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )""")
    conn.commit()
    conn.close()

@contextmanager
def get_conn():
    conn = sqlite3.connect(Config.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
