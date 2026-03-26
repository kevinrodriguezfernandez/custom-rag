# api/routes/ingest.py
"""Ingest endpoint — accepts a document upload, chunks it, and stores embeddings."""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ingestion.chunker import chunk_text
from ingestion.embedder import embed_chunks, upsert_to_store
from ingestion.loader import load_document
from shared.models import IngestResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(..., description="Document file to ingest"),
    document_id: str = Form(..., description="Caller-supplied document identifier"),
    chat_id: str = Form(default="", description="Chat session ID to scope this document to"),
) -> IngestResponse:
    """Ingest a document: load, chunk, embed, and store.

    1. Save the uploaded bytes to a temporary file.
    2. Extract plain text via the loader.
    3. Chunk the text using RecursiveCharacterTextSplitter.
    4. Embed chunks via OpenAI and upsert to Qdrant.
    5. Return the document_id and chunk count.
    """
    # Determine suffix from the uploaded filename so the loader can pick the right
    # strategy (PDF vs text).
    suffix = Path(file.filename or "upload.txt").suffix or ".txt"

    logger.info(
        "Ingest request received — filename=%s document_id=%s chat_id=%s",
        file.filename,
        document_id,
        chat_id,
    )

    try:
        file_bytes = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file — error=%s", exc)
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {exc}") from exc

    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            text = load_document(Path(tmp.name))
        logger.info("Document loaded — text_length=%d", len(text))
    except ValueError as exc:
        logger.error("Unsupported file type — error=%s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Failed to load document — error=%s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to load document: {exc}") from exc

    try:
        chunks = chunk_text(text, document_id, chat_id=chat_id)
        logger.info("Chunking complete — chunk_count=%d", len(chunks))
        vectors = embed_chunks(chunks)
        logger.info("Embedding complete — vector_count=%d", len(vectors))
        count = upsert_to_store(chunks, vectors)
        logger.info("Upsert complete — upserted_count=%d", count)
    except Exception as exc:
        logger.error("Ingestion pipeline failed — error=%s", exc)
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {exc}") from exc

    return IngestResponse(document_id=document_id, chunks_created=count)
