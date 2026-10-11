import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

SIMILARITY_THRESHOLD = float(
os.getenv("SIMILARITY_THRESHOLD", "0.95")
)

