# ingestion/embedder.py
"""Embedding pipeline — converts text chunks into vectors and upserts to Qdrant."""

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from shared.config import OPENAI_API_KEY, OPENAI_EMBEDDING_MODEL, QDRANT_URL, QDRANT_COLLECTION
from shared.models import DocumentChunk

VECTOR_SIZE = 1536  # text-embedding-3-small output dimension


def embed_chunks(chunks: list[DocumentChunk]) -> list[list[float]]:
    """Generate embedding vectors for a list of document chunks.

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
