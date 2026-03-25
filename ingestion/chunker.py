# ingestion/chunker.py
"""Text chunking — splits raw text into overlapping chunks for embedding."""

from shared.models import DocumentChunk


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[DocumentChunk]:
    """Split *text* into chunks and return a list of ``DocumentChunk`` objects.

    Parameters
    ----------
    text:
        The full plain-text content to split.
    document_id:
        Identifier of the parent document (propagated to each chunk).
    chunk_size:
        Target size in characters for each chunk.
    chunk_overlap:
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    list[DocumentChunk]
        Ordered list of chunks.
    """
    raise NotImplementedError("Chunking is not yet implemented.")
