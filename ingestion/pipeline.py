# ingestion/pipeline.py
"""Ingestion pipeline — orchestrates load → chunk → embed → upsert."""

import logging
import tempfile
from pathlib import Path

from ingestion.chunker import chunk_text
from ingestion.embedder import embed_chunks, upsert_to_store
from ingestion.loader import load_document

logger = logging.getLogger(__name__)


def run_pipeline(
    file_bytes: bytes,
    filename: str,
    document_id: str,
    chat_id: str = "",
) -> int:
    """Load, chunk, embed, and store a document. Returns the number of chunks created.

    This is the single entry point into the ingestion module for the API layer.
    All internal steps (loading, chunking, embedding, upserting) are encapsulated here.

    Parameters
    ----------
    file_bytes:
        Raw bytes of the uploaded file.
    filename:
        Original filename (used to determine file type by extension).
    document_id:
        Caller-supplied identifier for the document.
    chat_id:
        Optional chat session ID to scope the document to.

    Returns
    -------
    int
        Number of chunks successfully upserted into the vector store.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    Exception
        Propagated from any pipeline stage on unexpected failure.
    """
    suffix = Path(filename).suffix or ".txt"
    logger.info(
        "Pipeline started — document_id=%s filename=%s chat_id=%s",
        document_id,
        filename,
        chat_id,
    )

    with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        text = load_document(Path(tmp.name))

    chunks = chunk_text(text, document_id, chat_id=chat_id)
    logger.info("Chunking complete — chunk_count=%d", len(chunks))

    vectors = embed_chunks(chunks)
    logger.info("Embedding complete — vector_count=%d", len(vectors))

    count = upsert_to_store(chunks, vectors)
    logger.info("Pipeline complete — upserted_count=%d", count)
    return count
