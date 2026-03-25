# ingestion/embedder.py
"""Embedding pipeline — converts text chunks into vectors and upserts to Qdrant."""

from shared.models import DocumentChunk


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
    raise NotImplementedError("Embedding is not yet implemented.")


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
    raise NotImplementedError("Vector store upsert is not yet implemented.")
