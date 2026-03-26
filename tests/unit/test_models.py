# tests/unit/test_models.py
"""Smoke tests for shared Pydantic models."""

import pytest
from pydantic import ValidationError

from shared.models import (
    ChatRequest,
    ChatResponse,
    ChatTurn,
    DocumentChunk,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RetrievedChunk,
)


class TestHealthResponse:
    """Contract tests for HealthResponse."""

    def test_health_response_defaults(self) -> None:
        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.version == "0.1.0"

    def test_health_response_custom_values(self) -> None:
        resp = HealthResponse(status="degraded", version="0.2.0")
        assert resp.status == "degraded"
        assert resp.version == "0.2.0"


class TestDocumentChunk:
    """Contract tests for DocumentChunk."""

    def test_document_chunk_round_trip(self) -> None:
        chunk = DocumentChunk(
            chunk_id="c-1",
            document_id="d-1",
            content="Hello world",
            metadata={"page": 1},
        )
        assert chunk.chunk_id == "c-1"
        assert chunk.document_id == "d-1"
        assert chunk.content == "Hello world"
        assert chunk.metadata["page"] == 1

    def test_document_chunk_required_fields(self) -> None:
        """chunk_id, document_id, content are required."""
        with pytest.raises(ValidationError):
            DocumentChunk(
                document_id="d-1",
                content="text",
            )  # Missing chunk_id

        with pytest.raises(ValidationError):
            DocumentChunk(
                chunk_id="c-1",
                content="text",
            )  # Missing document_id

        with pytest.raises(ValidationError):
            DocumentChunk(
                chunk_id="c-1",
                document_id="d-1",
            )  # Missing content

    def test_document_chunk_metadata_default(self) -> None:
        chunk = DocumentChunk(chunk_id="c-1", document_id="d-1", content="text")
        assert chunk.metadata == {}

    def test_document_chunk_metadata_arbitrary(self) -> None:
        chunk = DocumentChunk(
            chunk_id="c-1",
            document_id="d-1",
            content="text",
            metadata={"source": "pdf", "page": 5, "custom": "value"},
        )
        assert chunk.metadata["source"] == "pdf"
        assert chunk.metadata["page"] == 5
        assert chunk.metadata["custom"] == "value"


class TestRetrievedChunk:
    """Contract tests for RetrievedChunk."""

    def test_retrieved_chunk_requires_chunk_and_score(self) -> None:
        chunk = DocumentChunk(chunk_id="c-1", document_id="d-1", content="text")
        retrieved = RetrievedChunk(chunk=chunk, score=0.95)
        assert retrieved.score == 0.95
        assert retrieved.chunk == chunk

    def test_retrieved_chunk_missing_chunk(self) -> None:
        with pytest.raises(ValidationError):
            RetrievedChunk(score=0.95)  # Missing chunk

    def test_retrieved_chunk_missing_score(self) -> None:
        chunk = DocumentChunk(chunk_id="c-1", document_id="d-1", content="text")
        with pytest.raises(ValidationError):
            RetrievedChunk(chunk=chunk)  # Missing score

    def test_retrieved_chunk_score_range(self) -> None:
        chunk = DocumentChunk(chunk_id="c-1", document_id="d-1", content="text")
        # Score should accept any float (no validation on range in the model)
        retrieved = RetrievedChunk(chunk=chunk, score=0.0)
        assert retrieved.score == 0.0

        retrieved = RetrievedChunk(chunk=chunk, score=1.0)
        assert retrieved.score == 1.0

        retrieved = RetrievedChunk(chunk=chunk, score=0.5)
        assert retrieved.score == 0.5


class TestChatTurn:
    """Contract tests for ChatTurn."""

    def test_chat_turn_user_role(self) -> None:
        turn = ChatTurn(role="user", content="What is AI?")
        assert turn.role == "user"
        assert turn.content == "What is AI?"

    def test_chat_turn_assistant_role(self) -> None:
        turn = ChatTurn(role="assistant", content="AI is...")
        assert turn.role == "assistant"
        assert turn.content == "AI is..."

    def test_chat_turn_invalid_role(self) -> None:
        """Only 'user' and 'assistant' roles are allowed."""
        with pytest.raises(ValidationError):
            ChatTurn(role="system", content="Invalid role")

    def test_chat_turn_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            ChatTurn(role="user")  # Missing content

        with pytest.raises(ValidationError):
            ChatTurn(content="Hello")  # Missing role


class TestChatRequest:
    """Contract tests for ChatRequest."""

    def test_chat_request_minimum(self) -> None:
        req = ChatRequest(query="What is RAG?")
        assert req.query == "What is RAG?"
        assert req.top_k == 5
        assert req.model == "gpt-4o-mini"
        assert req.chat_history == []

    def test_chat_request_custom_values(self) -> None:
        req = ChatRequest(query="Test", top_k=10, model="claude-opus-4-1")
        assert req.query == "Test"
        assert req.top_k == 10
        assert req.model == "claude-opus-4-1"

    def test_chat_request_query_required(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest()  # Missing query

    def test_chat_request_query_min_length(self) -> None:
        """Query must have min_length=1."""
        with pytest.raises(ValidationError):
            ChatRequest(query="")

    def test_chat_request_top_k_bounds(self) -> None:
        """top_k must be ge=1, le=20."""
        with pytest.raises(ValidationError):
            ChatRequest(query="Test", top_k=0)  # Below minimum

        with pytest.raises(ValidationError):
            ChatRequest(query="Test", top_k=21)  # Above maximum

        # Valid values
        req = ChatRequest(query="Test", top_k=1)
        assert req.top_k == 1

        req = ChatRequest(query="Test", top_k=20)
        assert req.top_k == 20

    def test_chat_request_with_chat_history(self) -> None:
        history = [
            ChatTurn(role="user", content="First question"),
            ChatTurn(role="assistant", content="First answer"),
        ]
        req = ChatRequest(query="Follow-up?", chat_history=history)
        assert len(req.chat_history) == 2
        assert req.chat_history[0].role == "user"

    def test_chat_request_model_names(self) -> None:
        """Model parameter accepts various model names."""
        models = [
            "gpt-4o-mini",
            "gpt-4",
            "claude-opus-4-1",
            "claude-3-sonnet-20240229",
            "llama3",
            "custom-model",
        ]
        for model in models:
            req = ChatRequest(query="Test", model=model)
            assert req.model == model


class TestChatResponse:
    """Contract tests for ChatResponse."""

    def test_chat_response_minimum(self) -> None:
        resp = ChatResponse(answer="42")
        assert resp.answer == "42"
        assert resp.sources == []
        assert resp.model is None

    def test_chat_response_with_sources(self) -> None:
        chunk = DocumentChunk(chunk_id="c-1", document_id="d-1", content="text")
        source = RetrievedChunk(chunk=chunk, score=0.95)
        resp = ChatResponse(answer="42", sources=[source])
        assert resp.answer == "42"
        assert len(resp.sources) == 1
        assert resp.sources[0].score == 0.95

    def test_chat_response_with_model(self) -> None:
        resp = ChatResponse(answer="42", model="gpt-4o-mini")
        assert resp.model == "gpt-4o-mini"

    def test_chat_response_answer_required(self) -> None:
        with pytest.raises(ValidationError):
            ChatResponse()  # Missing answer

    def test_chat_response_answer_must_be_string(self) -> None:
        # Answer is required and must be a string
        resp = ChatResponse(answer="Valid string")
        assert isinstance(resp.answer, str)


class TestIngestRequest:
    """Contract tests for IngestRequest."""

    def test_ingest_request_minimum(self) -> None:
        req = IngestRequest(document_id="d-1")
        assert req.document_id == "d-1"
        assert req.metadata == {}

    def test_ingest_request_with_metadata(self) -> None:
        req = IngestRequest(
            document_id="d-1",
            metadata={"source": "upload", "user_id": "user-123"},
        )
        assert req.document_id == "d-1"
        assert req.metadata["source"] == "upload"
        assert req.metadata["user_id"] == "user-123"

    def test_ingest_request_document_id_required(self) -> None:
        with pytest.raises(ValidationError):
            IngestRequest()  # Missing document_id


class TestIngestResponse:
    """Contract tests for IngestResponse."""

    def test_ingest_response(self) -> None:
        resp = IngestResponse(document_id="d-1", chunks_created=10)
        assert resp.document_id == "d-1"
        assert resp.chunks_created == 10

    def test_ingest_response_required_fields(self) -> None:
        with pytest.raises(ValidationError):
            IngestResponse(document_id="d-1")  # Missing chunks_created

        with pytest.raises(ValidationError):
            IngestResponse(chunks_created=10)  # Missing document_id

    def test_ingest_response_chunks_created_zero(self) -> None:
        resp = IngestResponse(document_id="d-1", chunks_created=0)
        assert resp.chunks_created == 0
