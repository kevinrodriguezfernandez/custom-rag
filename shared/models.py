# shared/models.py
"""ALL Pydantic models for the monorepo live in this single file.

Other modules import from here — they never define their own request/response
or domain models.
"""

from typing import Literal
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
# Conversation primitives
# ---------------------------------------------------------------------------

class ChatTurn(BaseModel):
    """A single turn in a conversation history (user or assistant message)."""

    role: Literal["user", "assistant"] = Field(..., description="Speaker role for this turn")
    content: str = Field(..., description="Text content of the turn")


# ---------------------------------------------------------------------------
# API request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Incoming chat request from the frontend."""

    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    model: str = Field(default="gpt-4o-mini", description="OpenAI model name to use for generation")
    chat_history: list[ChatTurn] = Field(
        default_factory=list,
        description="Previous conversation turns, ordered oldest-first",
    )


class ChatResponse(BaseModel):
    """Response returned to the frontend after RAG pipeline execution."""

    answer: str = Field(..., description="LLM-generated answer")
    sources: list[RetrievedChunk] = Field(default_factory=list, description="Chunks used to generate the answer")
    model: str | None = Field(default=None, description="OpenAI model that produced the answer, echoed from the request")


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
