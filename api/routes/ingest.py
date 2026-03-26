# api/routes/ingest.py
"""Ingest endpoint — accepts a document upload, chunks it, and stores embeddings."""

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from ingestion.chunker import chunk_text
from ingestion.embedder import embed_chunks, upsert_to_store
from ingestion.loader import load_document
from shared.models import IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".txt", ".pdf"})


@router.post("/", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(..., description="Document file to ingest (.txt or .pdf)"),
    document_id: str = Form(..., description="Caller-supplied document identifier"),
) -> IngestResponse:
    """Ingest a document: validate, load, chunk, embed, and store.

    Steps:
    1. Validate file extension.
    2. Buffer the upload to a temporary file on disk.
    3. Extract plain text via the loader.
    4. Chunk the text into overlapping windows.
    5. Embed the chunks and upsert to the vector store.
    6. Return the number of chunks stored.
    """
    # 1. Validate file extension
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type {suffix!r}. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    # 2. Buffer upload to a temp file (loader requires a real Path on disk)
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(contents)
            tmp_path = Path(tmp.name)
    except OSError as exc:
        logger.exception("Failed to buffer upload for document_id=%r", document_id)
        raise HTTPException(
            status_code=500, detail=f"Failed to save upload: {exc}"
        ) from exc

    # 3. Extract text — synchronous file I/O, offloaded to a thread
    try:
        text: str = await asyncio.to_thread(load_document, tmp_path)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("load_document failed for document_id=%r", document_id)
        raise HTTPException(
            status_code=500, detail=f"Failed to load document: {exc}"
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    # 4. Chunk text — CPU-bound but fast; offloaded for consistency
    chunks = await asyncio.to_thread(chunk_text, text, document_id)
    if not chunks:
        raise HTTPException(
            status_code=422, detail="Document produced no text chunks."
        )

    # 5. Embed chunks — network I/O to OpenAI, must not block event loop
    try:
        vectors: list[list[float]] = await asyncio.to_thread(embed_chunks, chunks)
    except Exception as exc:
        logger.exception("embed_chunks failed for document_id=%r", document_id)
        raise HTTPException(
            status_code=502, detail=f"Embedding failed: {exc}"
        ) from exc

    # 6. Upsert to vector store — network I/O to Qdrant, must not block event loop
    try:
        n_upserted: int = await asyncio.to_thread(upsert_to_store, chunks, vectors)
    except Exception as exc:
        logger.exception("upsert_to_store failed for document_id=%r", document_id)
        raise HTTPException(
            status_code=502, detail=f"Vector store upsert failed: {exc}"
        ) from exc

    logger.info(
        "Ingested document_id=%r: %d chunks stored", document_id, n_upserted
    )
    return IngestResponse(document_id=document_id, chunks_created=n_upserted)
