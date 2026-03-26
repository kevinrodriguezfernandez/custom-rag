# ingestion/embedder.py
"""Embedding pipeline — converts text chunks into vectors and upserts to Qdrant."""

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from shared.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, QDRANT_URL, QDRANT_COLLECTION
from shared.models import DocumentChunk

VECTOR_SIZE = 1536  # text-embedding-3-small output dimension


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
    client = OpenAI(api_key=OPENAI_API_KEY)
    texts = [chunk.content for chunk in chunks]
    response = client.embeddings.create(model=OPENAI_EMBEDDING_MODEL, input=texts)
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
    client = QdrantClient(url=QDRANT_URL)
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    points = [
        PointStruct(
            id=chunk.chunk_id,
            vector=vector,
            payload={
                "document_id": chunk.document_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
            },
        )
        for chunk, vector in zip(chunks, vectors)
    ]
    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)
