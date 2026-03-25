# api/routes/ingest.py
"""Ingest endpoint — accepts a document upload, chunks it, and stores embeddings."""

from fastapi import APIRouter, UploadFile, File, Form

from shared.models import IngestResponse

router = APIRouter()


@router.post("/", response_model=IngestResponse)
async def ingest(
    file: UploadFile = File(..., description="Document file to ingest"),
    document_id: str = Form(..., description="Caller-supplied document identifier"),
) -> IngestResponse:
    """Ingest a document: load, chunk, embed, and store.

    Steps (to be implemented):
    1. Read the uploaded file.
    2. Extract text via the loader.
    3. Chunk the text.
    4. Embed the chunks and upsert to Qdrant.
    5. Return the number of chunks created.
    """
    raise NotImplementedError("Ingest endpoint is not yet implemented.")
