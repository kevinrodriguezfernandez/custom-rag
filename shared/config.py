# shared/config.py
"""Centralised configuration — all env vars are read here and nowhere else."""

from dotenv import load_dotenv
import os

load_dotenv()

# --- OpenAI / LLM ---
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY") or ""
OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL: str = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")

# --- Embedding provider ---
# Set EMBEDDING_PROVIDER=ollama to use Ollama for embeddings instead of OpenAI.
# When using Ollama embeddings, pull the model first: ollama pull nomic-embed-text
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "openai")  # "openai" | "ollama"
OLLAMA_EMBEDDING_MODEL: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
# Vector size must match the embedding model:
#   openai/text-embedding-3-small → 1536
#   ollama/nomic-embed-text       → 768
VECTOR_SIZE: int = 768 if EMBEDDING_PROVIDER == "ollama" else 1536

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

# --- Ollama ---
OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")

# --- Anthropic ---
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
