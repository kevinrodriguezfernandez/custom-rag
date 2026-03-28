# api/routes/ingest.py
"""Ingest endpoint — accepts a document upload, chunks it, and stores embeddings."""

import asyncio
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ingestion.pipeline import run_pipeline
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

    Delegates the full pipeline to ``ingestion.pipeline.run_pipeline`` and
    runs it in a thread pool to avoid blocking the async event loop.
    """
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

    loop = asyncio.get_running_loop()
    try:
        count: int = await loop.run_in_executor(
            None,
            run_pipeline,
            file_bytes,
            file.filename or "upload.txt",
            document_id,
            chat_id,
        )
    except ValueError as exc:
        logger.error("Unsupported file type — error=%s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Ingestion pipeline failed — error=%s", exc)
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {exc}") from exc

    logger.info("Ingest complete — document_id=%s chunks_created=%d", document_id, count)
    return IngestResponse(document_id=document_id, chunks_created=count)
