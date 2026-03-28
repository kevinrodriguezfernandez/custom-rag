# ingestion/embedder.py
"""Embedding pipeline — converts text chunks into vectors and upserts to Qdrant."""

import hashlib
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from shared.config import (
    QDRANT_COLLECTION,
    QDRANT_URL,
    VECTOR_SIZE,
)
from shared.embedding import get_embedding_client
from shared.models import DocumentChunk

logger = logging.getLogger(__name__)

_qdrant = QdrantClient(url=QDRANT_URL)


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
    client, model = get_embedding_client()
    logger.info("Embedding chunks — chunk_count=%d model=%s", len(chunks), model)
    response = client.embeddings.create(
        model=model,
        input=[c.content for c in chunks],
    )
    return [item.embedding for item in response.data]


def _chunk_id_to_point_id(chunk_id: str) -> int:
    """Deterministically convert a chunk_id string to a Qdrant-compatible uint64.

    Uses SHA-256 so the same chunk_id always maps to the same point ID across
    processes (unlike Python's hash() which is randomised per-process).
    """
    return int.from_bytes(hashlib.sha256(chunk_id.encode()).digest()[:8], "big")


def upsert_to_store(chunks: list[DocumentChunk], vectors: list[list[float]]) -> int:
    """Upsert chunk vectors into the Qdrant vector store.

    Creates the collection if it does not already exist, using the vector size
    from config (768 for Ollama/nomic-embed-text, 1536 for OpenAI embeddings).

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
    existing = {c.name for c in _qdrant.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        _qdrant.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )

    points = [
        PointStruct(
            id=_chunk_id_to_point_id(chunk.chunk_id),
            vector=vector,
            payload=chunk.model_dump(),
        )
        for chunk, vector in zip(chunks, vectors)
    ]

    _qdrant.upsert(collection_name=QDRANT_COLLECTION, points=points)
    logger.info("Upsert complete — upserted_count=%d collection=%s", len(points), QDRANT_COLLECTION)
    return len(points)
