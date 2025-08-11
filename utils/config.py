import os
from dotenv import load_dotenv
load_dotenv()

class Config:
    MODELS_DIR = os.getenv("MODELS_DIR", "models")
    DATA_DIR = os.getenv("DATA_DIR", "data")
    DB_PATH = os.getenv("DB_PATH", "aml_system.db")
    ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.7"))
    API_HOST = os.getenv("API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("API_PORT", "8000"))
