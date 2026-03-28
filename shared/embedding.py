# shared/embedding.py
"""Shared embedding client factory used by both ingestion and api modules."""

import openai

from shared.config import (
    EMBEDDING_PROVIDER,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_URL,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
)


def get_embedding_client() -> tuple[openai.OpenAI, str]:
    """Return (client, model) for the configured embedding provider.

    Returns
    -------
    tuple[openai.OpenAI, str]
        An OpenAI-compatible client and the model name to use for embeddings.
    """
    if EMBEDDING_PROVIDER == "ollama":
        return (
            openai.OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama"),
            OLLAMA_EMBEDDING_MODEL,
        )
    return openai.OpenAI(api_key=OPENAI_API_KEY), OPENAI_EMBEDDING_MODEL
