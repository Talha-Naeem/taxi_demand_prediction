# src/settings.py
import os
from dotenv import load_dotenv
from src.paths import PARENT_DIR

# Load .env file
load_dotenv(PARENT_DIR / ".env")

# ---- Hopsworks credentials ----
HOPSWORKS_PROJECT_NAME = "taxidemand_predict"
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

if not HOPSWORKS_API_KEY:
    raise Exception("❌ Missing HOPSWORKS_API_KEY in .env file!")
