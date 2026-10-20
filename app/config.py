import os
from math import isfinite
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

EMBEDDING_REVISION = os.getenv("EMBEDDING_REVISION", "").strip()

# API Auth
API_KEY = os.getenv("API_KEY", "").strip()

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

if CHUNK_SIZE <= 0 or not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError("CHUNK_SIZE must be positive and CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
if TOP_K <= 0 or MAX_UPLOAD_SIZE_MB <= 0 or not isfinite(SIMILARITY_THRESHOLD) or SIMILARITY_THRESHOLD < 0:
    raise ValueError("TOP_K and MAX_UPLOAD_SIZE_MB must be positive; SIMILARITY_THRESHOLD must be finite and nonnegative")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
