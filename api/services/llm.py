# api/services/llm.py
"""LLM service — builds a prompt from retrieved context and calls the chat model."""

from shared.models import RetrievedChunk


def generate_answer(query: str, context_chunks: list[RetrievedChunk]) -> str:
    """Generate an answer to *query* using the retrieved *context_chunks*.

    Parameters
    ----------
    query:
        The user's original question.
    context_chunks:
        Chunks retrieved from the vector store to use as context.

    Returns
    -------
    str
        The LLM-generated answer.
    """
    raise NotImplementedError("LLM answer generation is not yet implemented.")
