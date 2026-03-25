# shared/models.py
"""ALL Pydantic models for the monorepo live in this single file.

Other modules import from here — they never define their own request/response
or domain models.
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------

class DocumentChunk(BaseModel):
    """A single chunk of a source document, ready for embedding and storage."""

    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    document_id: str = Field(..., description="Parent document identifier")
    content: str = Field(..., description="Plain-text content of the chunk")
    metadata: dict = Field(default_factory=dict, description="Arbitrary metadata (source, page, etc.)")


class RetrievedChunk(BaseModel):
    """A chunk returned from the vector store with a relevance score."""

    chunk: DocumentChunk
    score: float = Field(..., description="Similarity score from the vector store")


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")


class ChatResponse(BaseModel):
    """Response returned to the frontend after RAG pipeline execution."""

    answer: str = Field(..., description="LLM-generated answer")
    sources: list[RetrievedChunk] = Field(default_factory=list, description="Chunks used to generate the answer")


class IngestRequest(BaseModel):
    """Request to ingest a new document (metadata only; file comes via multipart)."""

    document_id: str = Field(..., description="Caller-supplied document identifier")
    metadata: dict = Field(default_factory=dict, description="Optional metadata to attach")


class IngestResponse(BaseModel):
    """Acknowledgement after a successful ingestion."""

    document_id: str
    chunks_created: int


class HealthResponse(BaseModel):
    """Health-check response."""

    status: str = "ok"
    version: str = "0.1.0"
