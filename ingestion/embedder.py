# ingestion/embedder.py
"""Embedding pipeline — converts text chunks into vectors and upserts to Qdrant."""

import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from shared.config import (
    EMBEDDING_PROVIDER,
    OLLAMA_EMBEDDING_MODEL,
    OLLAMA_URL,
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    QDRANT_COLLECTION,
    QDRANT_URL,
    VECTOR_SIZE,
)
from shared.models import DocumentChunk


def _embedding_client() -> tuple[openai.OpenAI, str]:
    """Return (client, model) for the configured embedding provider."""
    if EMBEDDING_PROVIDER == "ollama":
        return (
            openai.OpenAI(base_url=f"{OLLAMA_URL}/v1", api_key="ollama"),
            OLLAMA_EMBEDDING_MODEL,
        )
    return openai.OpenAI(api_key=OPENAI_API_KEY), OPENAI_EMBEDDING_MODEL


def embed_chunks(chunks: list[DocumentChunk]) -> list[list[float]]:
    """Generate embedding vectors for a list of document chunks.

    Uses the configured embedding provider (OpenAI or Ollama).

    Parameters
    ----------
    chunks:
        The chunks to embed.

    Returns
    -------
    list[list[float]]
        A list of embedding vectors, one per chunk, in the same order.
    """
    client, model = _embedding_client()
    response = client.embeddings.create(
        model=model,
        input=[c.content for c in chunks],
    )
    return [item.embedding for item in response.data]


def upsert_to_store(chunks: list[DocumentChunk], vectors: list[list[float]]) -> int:
    """Upsert chunk vectors into the Qdrant vector store.

    Creates the collection if it does not already exist (Cosine distance,
    1536-dimensional vectors for text-embedding-3-small).

    Parameters
    ----------
    chunks:
        The document chunks (used for payload/metadata).
    vectors:
        Corresponding embedding vectors.

    Returns
    -------
    int
        Number of points successfully upserted.
    """
    qdrant = QdrantClient(url=QDRANT_URL)

    existing = {c.name for c in qdrant.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    points = [
        PointStruct(
            id=abs(hash(chunk.chunk_id)) % (2**63),
            vector=vector,
            payload=chunk.model_dump(),
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)
