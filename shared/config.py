# shared/config.py
"""Centralised configuration — all env vars are read here and nowhere else."""

from dotenv import load_dotenv
import os

load_dotenv()

# --- OpenAI / LLM ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or ""
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY must be set in environment")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")

# --- Qdrant ---
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "documents")

# --- API ---
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
try:
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
except ValueError as exc:
    raise ValueError(
        f"API_PORT must be a valid integer, got: {os.getenv('API_PORT')!r}"
    ) from exc

# --- Streamlit / API client ---
# Named API_URL to match the env var that app/main.py reads ("API_URL").
# Both this module and app/main.py default to http://localhost:8000.
API_URL: str = os.getenv("API_URL", "http://localhost:8000")
