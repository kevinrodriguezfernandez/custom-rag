# shared/config.py
"""Centralised configuration — all env vars are read here and nowhere else."""

from dotenv import load_dotenv
import os

load_dotenv()

# --- OpenAI / LLM ---
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")

# --- Qdrant ---
QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "documents")

# --- API ---
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# --- Streamlit ---
API_BASE_URL: str = os.getenv("API_BASE_URL", "http://localhost:8000")
