# api/services/retriever.py
"""Retriever service — queries Qdrant for relevant document chunks."""

from shared.models import RetrievedChunk


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
        Ranked list of relevant chunks with scores.
    """
    raise NotImplementedError("Retriever is not yet implemented.")
