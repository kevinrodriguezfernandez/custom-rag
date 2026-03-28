# api/services/retriever.py
"""Retriever service — queries Qdrant for relevant document chunks."""

import asyncio
import logging

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from shared.config import QDRANT_COLLECTION, QDRANT_URL
from shared.embedding import get_embedding_client
from shared.models import DocumentChunk, RetrievedChunk

logger = logging.getLogger(__name__)

_qdrant = QdrantClient(url=QDRANT_URL)


async def retrieve(query: str, top_k: int = 5, chat_id: str | None = None) -> list[RetrievedChunk]:
    """Embed *query* and return the top-k most relevant chunks from the vector store.

    Embeds the query using the configured embedding provider (OpenAI or Ollama),
    then performs a vector similarity search against the Qdrant collection.
    Both calls run in a thread pool to avoid blocking the event loop.

    Parameters
    ----------
    query:
        The user's natural-language question.
    top_k:
        Maximum number of chunks to return.
    chat_id:
        Optional chat session ID to scope results to a specific session.

    Returns
    -------
    list[RetrievedChunk]
        Ranked list of relevant chunks with scores, highest first.
    """
    loop = asyncio.get_running_loop()

    # Embed the query (blocking I/O — run in thread pool)
    emb_client, emb_model = get_embedding_client()
    logger.debug("Embedding query — model=%s", emb_model)

    def _embed() -> list[float]:
        response = emb_client.embeddings.create(
            model=emb_model,
            input=query,
        )
        return response.data[0].embedding

    query_vector: list[float] = await loop.run_in_executor(None, _embed)

    # Search Qdrant (blocking I/O — run in thread pool)
    logger.debug(
        "Searching Qdrant — collection=%s top_k=%d chat_id=%s",
        QDRANT_COLLECTION,
        top_k,
        chat_id,
    )

    def _search() -> list:
        query_filter = Filter(
            must=[FieldCondition(key="metadata.chat_id", match=MatchValue(value=chat_id))]
        ) if chat_id else None
        response = _qdrant.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )
        return response.points

    results = await loop.run_in_executor(None, _search)

    retrieved: list[RetrievedChunk] = []
    for point in results:
        chunk = DocumentChunk(**point.payload)
        retrieved.append(RetrievedChunk(chunk=chunk, score=point.score))

    logger.info("Retrieval returned %d result(s)", len(retrieved))
    return retrieved
