# ingestion/chunker.py
"""Text chunking — splits raw text into overlapping chunks for embedding."""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from shared.models import DocumentChunk


def chunk_text(
    text: str,
    document_id: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
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
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    texts = splitter.split_text(text)
    return [
        DocumentChunk(
            chunk_id=f"{document_id}-{i}",
            document_id=document_id,
            content=chunk_text,
            metadata={"chunk_index": i},
        )
        for i, chunk_text in enumerate(texts)
    ]
