# api/services/retriever.py
"""Retriever service — embed query, search Qdrant, return ranked chunks."""

from openai import OpenAI
from qdrant_client import QdrantClient

from shared.config import (
    OPENAI_API_KEY,
    OPENAI_EMBEDDING_MODEL,
    QDRANT_URL,
    QDRANT_COLLECTION,
)
from shared.models import DocumentChunk, RetrievedChunk


def retrieve(query: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Embed *query* and return the top-k most relevant chunks from the vector store.

    Parameters
    ----------
    query:
        The user's natural-language question.
    top_k:
        Maximum number of chunks to return.

    Returns
    -------
    list[RetrievedChunk]
        Ranked list of relevant chunks with similarity scores, highest first.
    """
    # 1. Embed the query
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    response = openai_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=[query],
    )
    query_vector = response.data[0].embedding

    # 2. Search Qdrant
    qdrant_client = QdrantClient(url=QDRANT_URL)
    results = qdrant_client.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=query_vector,
        limit=top_k,
        with_payload=True,
    )

    # 3. Reconstruct RetrievedChunk objects from Qdrant payloads
    retrieved: list[RetrievedChunk] = []
    for hit in results:
        payload = hit.payload or {}
        chunk = DocumentChunk(
            chunk_id=str(hit.id),
            document_id=payload.get("document_id", ""),
            content=payload.get("content", ""),
            metadata=payload.get("metadata", {}),
        )
        retrieved.append(RetrievedChunk(chunk=chunk, score=hit.score))
    return retrieved
