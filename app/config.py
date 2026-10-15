import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# API Auth
API_KEY = os.getenv("API_KEY", "changeme")  # mude no .env

# Vector search
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "1.2"))
TOP_K = int(os.getenv("TOP_K", "4"))

# Chunking
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# Upload
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "15"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

UPLOAD_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)