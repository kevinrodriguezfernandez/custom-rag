# ingestion/chunker.py
"""Text chunking — splits raw text into overlapping chunks for embedding."""

import uuid

from langchain_text_splitters import RecursiveCharacterTextSplitter

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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    raw_chunks = splitter.split_text(text)
    return [
        DocumentChunk(
            chunk_id=str(uuid.uuid4()),
            document_id=document_id,
            content=chunk,
            metadata={"chunk_index": i},
        )
        for i, chunk in enumerate(raw_chunks)
    ]
