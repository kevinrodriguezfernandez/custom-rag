# ingestion/chunker.py
"""Text chunking — splits raw text into overlapping chunks for embedding."""

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.models import DocumentChunk

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = 1024,
    chunk_overlap: int = 128,
    chat_id: str = "",
) -> list[DocumentChunk]:
    """Split *text* into chunks and return a list of ``DocumentChunk`` objects.

    Uses RecursiveCharacterTextSplitter which tries to split on paragraph
    boundaries first, then sentence boundaries, before falling back to
    character-level splits.

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
    logger.info("Chunking text — input_length=%d", len(text))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = splitter.split_text(text)
    chunks = [
        DocumentChunk(
            chunk_id=f"{document_id}-{i}",
            document_id=document_id,
            content=chunk_text,
            metadata={"chunk_index": i, "chat_id": chat_id},
        )
        for i, chunk_text in enumerate(texts)
    ]
    logger.info("Chunking complete — chunk_count=%d", len(chunks))
    return chunks
