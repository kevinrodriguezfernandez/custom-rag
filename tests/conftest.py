# tests/conftest.py
"""Shared pytest fixtures for the entire test suite."""

import asyncio
from unittest.mock import MagicMock, AsyncMock

import pytest
from fastapi.testclient import TestClient

from api.main import app
from shared.models import (
    ChatRequest,
    ChatTurn,
    DocumentChunk,
    HealthResponse,
    RetrievedChunk,
)


@pytest.fixture()
def client() -> TestClient:
    """Return a synchronous FastAPI test client."""
    return TestClient(app)


@pytest.fixture()
def sample_source() -> DocumentChunk:
    """A single valid DocumentChunk for testing."""
    return DocumentChunk(
        chunk_id="doc-1-0",
        document_id="doc-1",
        content="The quick brown fox jumps over the lazy dog.",
        metadata={"page": 1, "chunk_index": 0},
    )


@pytest.fixture()
def sample_chunks() -> list[DocumentChunk]:
    """Multiple realistic DocumentChunks for integration tests."""
    return [
        DocumentChunk(
            chunk_id="doc-2-0",
            document_id="doc-2",
            content="Machine learning is a subset of artificial intelligence "
                    "that focuses on the development of algorithms and statistical models.",
            metadata={"page": 1, "chunk_index": 0},
        ),
        DocumentChunk(
            chunk_id="doc-2-1",
            document_id="doc-2",
            content="These models enable computers to learn from data without being "
                    "explicitly programmed.",
            metadata={"page": 1, "chunk_index": 1},
        ),
        DocumentChunk(
            chunk_id="doc-2-2",
            document_id="doc-2",
            content="Common applications include classification, regression, clustering, "
                    "and dimensionality reduction.",
            metadata={"page": 2, "chunk_index": 2},
        ),
    ]


@pytest.fixture()
def sample_chat_request() -> ChatRequest:
    """A valid ChatRequest for testing."""
    return ChatRequest(
        query="What is machine learning?",
        top_k=5,
        model="gpt-4o-mini",
        chat_history=[],
    )


@pytest.fixture()
def sample_chat_request_with_history() -> ChatRequest:
    """A ChatRequest with conversation history for testing."""
    return ChatRequest(
        query="Tell me more about that.",
        top_k=3,
        model="gpt-4o-mini",
        chat_history=[
            ChatTurn(role="user", content="What is AI?"),
            ChatTurn(role="assistant", content="Artificial Intelligence is..."),
        ],
    )


@pytest.fixture()
def mock_openai_client() -> MagicMock:
    """Mock OpenAI client that returns valid embeddings."""
    client = MagicMock()
    # Mock embeddings response
    mock_embedding = MagicMock()
    mock_embedding.embedding = [0.1] * 1536  # text-embedding-3-small is 1536-dim
    client.embeddings.create.return_value.data = [mock_embedding]
    # Mock chat completion response
    mock_choice = MagicMock()
    mock_choice.message.content = "This is a mocked LLM response."
    client.chat.completions.create.return_value.choices = [mock_choice]
    return client


@pytest.fixture()
def mock_anthropic_client() -> MagicMock:
    """Mock Anthropic client for Claude model testing."""
    client = MagicMock()
    mock_response = MagicMock()
    mock_content = MagicMock()
    mock_content.text = "This is a mocked Claude response."
    mock_response.content = [mock_content]
    client.messages.create.return_value = mock_response
    return client


@pytest.fixture()
def mock_qdrant_client() -> MagicMock:
    """Mock Qdrant client with in-memory behavior."""
    client = MagicMock()

    # Mock get_collections
    mock_collection = MagicMock()
    mock_collection.name = "documents"
    client.get_collections.return_value.collections = [mock_collection]

    # Mock search results
    def mock_search(collection_name, query_vector, limit, with_payload=False):
        # Return a mock search result
        mock_point = MagicMock()
        mock_point.payload = {
            "chunk_id": "doc-1-0",
            "document_id": "doc-1",
            "content": "Sample retrieved chunk.",
            "metadata": {"page": 1},
        }
        mock_point.score = 0.95
        return [mock_point]

    client.search.side_effect = mock_search
    client.upsert.return_value = None
    client.create_collection.return_value = None

    return client
