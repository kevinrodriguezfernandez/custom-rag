# api/routes/ingest.py
"""Ingest endpoint — accepts a document upload, chunks it, and stores embeddings."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ingestion.chunker import chunk_text
from ingestion.embedder import embed_chunks, upsert_to_store
from ingestion.loader import load_document
from shared.models import IngestResponse

router = APIRouter()


@router.post("/", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(..., description="Document file to ingest"),
    document_id: str = Form(..., description="Caller-supplied document identifier"),
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

    try:
        file_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read uploaded file: {exc}") from exc

    try:
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            text = load_document(Path(tmp.name))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to load document: {exc}") from exc

    try:
        chunks = chunk_text(text, document_id)
        vectors = embed_chunks(chunks)
        count = upsert_to_store(chunks, vectors)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion pipeline failed: {exc}") from exc

    return IngestResponse(document_id=document_id, chunks_created=count)
