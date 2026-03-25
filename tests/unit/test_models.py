# tests/unit/test_models.py
"""Smoke tests for shared Pydantic models."""

from shared.models import (
    ChatRequest,
    ChatResponse,
    DocumentChunk,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RetrievedChunk,
)


def test_health_response_defaults() -> None:
    resp = HealthResponse()
    assert resp.status == "ok"
    assert resp.version == "0.1.0"


def test_chat_request_minimum() -> None:
    req = ChatRequest(query="What is RAG?")
    assert req.query == "What is RAG?"
    assert req.top_k == 5


def test_document_chunk_round_trip() -> None:
    chunk = DocumentChunk(
        chunk_id="c-1",
        document_id="d-1",
        content="Hello world",
        metadata={"page": 1},
    )
    assert chunk.chunk_id == "c-1"
    assert chunk.metadata["page"] == 1


def test_ingest_request() -> None:
    req = IngestRequest(document_id="d-1")
    assert req.document_id == "d-1"
    assert req.metadata == {}


def test_ingest_response() -> None:
    resp = IngestResponse(document_id="d-1", chunks_created=10)
    assert resp.chunks_created == 10


def test_retrieved_chunk() -> None:
    chunk = DocumentChunk(chunk_id="c-1", document_id="d-1", content="text")
    retrieved = RetrievedChunk(chunk=chunk, score=0.95)
    assert retrieved.score == 0.95


def test_chat_response() -> None:
    resp = ChatResponse(answer="42", sources=[])
    assert resp.answer == "42"
    assert resp.sources == []
