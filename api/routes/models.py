# api/routes/models.py
"""Models endpoint — returns available models from configured providers."""

import logging

import httpx
from fastapi import APIRouter

from shared.config import OLLAMA_URL

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/ollama")
async def list_ollama_models() -> dict:
    """Fetch the list of locally available Ollama models from /api/tags."""
    url = f"{OLLAMA_URL}/api/tags"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url)
            response.raise_for_status()
        models = [m["name"] for m in response.json().get("models", [])]
        logger.info("Fetched %d Ollama models", len(models))
        return {"models": models}
    except Exception as exc:
        logger.warning("Could not reach Ollama at %s — %s", url, exc)
        return {"models": [], "error": str(exc)}
