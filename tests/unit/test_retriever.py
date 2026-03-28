# tests/unit/test_retriever.py
"""Unit tests for the retriever service."""

from unittest.mock import MagicMock, patch

import pytest

from api.services.retriever import retrieve
from shared.models import DocumentChunk, RetrievedChunk


def _mock_qdrant_response(points):
    """Return a MagicMock that mimics query_points() response."""
    mock_response = MagicMock()
    mock_response.points = points
    return mock_response


class TestRetrieveBasic:
    """Tests for basic retriever functionality."""

    @pytest.mark.asyncio
    async def test_retrieve_returns_list_of_retrieved_chunks(self) -> None:
        """retrieve() returns a list of RetrievedChunk objects."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai, patch(
            "api.services.retriever._qdrant"
        ) as mock_qdrant:
            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            mock_point = MagicMock()
            mock_point.payload = {
                "chunk_id": "c-1",
                "document_id": "d-1",
                "content": "Retrieved content",
                "metadata": {},
            }
            mock_point.score = 0.95
            mock_qdrant.query_points.return_value = _mock_qdrant_response([mock_point])

            result = await retrieve("Test query", top_k=5)

            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], RetrievedChunk)
            assert result[0].score == 0.95

    @pytest.mark.asyncio
    async def test_retrieve_embeds_query(self) -> None:
        """retrieve() embeds the query using OpenAI."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai, patch(
            "api.services.retriever._qdrant"
        ) as mock_qdrant:
            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            mock_qdrant.query_points.return_value = _mock_qdrant_response([])

            await retrieve("Test query", top_k=5)

            oai_instance.embeddings.create.assert_called_once()
            call_kwargs = oai_instance.embeddings.create.call_args.kwargs
            assert call_kwargs.get("input") == "Test query"

    @pytest.mark.asyncio
    async def test_retrieve_searches_qdrant(self) -> None:
        """retrieve() searches Qdrant with the query embedding."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai, patch(
            "api.services.retriever._qdrant"
        ) as mock_qdrant:
            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            test_vector = [0.1, 0.2, 0.3] + [0.1] * 1533
            mock_embedding = MagicMock()
            mock_embedding.embedding = test_vector
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            mock_qdrant.query_points.return_value = _mock_qdrant_response([])

            await retrieve("Test query", top_k=10)

            mock_qdrant.query_points.assert_called_once()
            call_kwargs = mock_qdrant.query_points.call_args.kwargs
            assert call_kwargs.get("query") == test_vector
            assert call_kwargs.get("limit") == 10

    @pytest.mark.asyncio
    async def test_retrieve_respects_top_k_parameter(self) -> None:
        """retrieve() passes top_k to Qdrant."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai, patch(
            "api.services.retriever._qdrant"
        ) as mock_qdrant:
            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            mock_qdrant.query_points.return_value = _mock_qdrant_response([])

            await retrieve("Query", top_k=20)

            call_kwargs = mock_qdrant.query_points.call_args.kwargs
            assert call_kwargs.get("limit") == 20


class TestRetrieveEdgeCases:
    """Tests for edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_retrieve_empty_results(self) -> None:
        """retrieve() returns empty list when no chunks are found."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai, patch(
            "api.services.retriever._qdrant"
        ) as mock_qdrant:
            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            mock_qdrant.query_points.return_value = _mock_qdrant_response([])

            result = await retrieve("Obscure query", top_k=5)

            assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_multiple_results(self) -> None:
        """retrieve() handles multiple search results."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai, patch(
            "api.services.retriever._qdrant"
        ) as mock_qdrant:
            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            mock_points = []
            for i in range(5):
                mock_point = MagicMock()
                mock_point.payload = {
                    "chunk_id": f"c-{i}",
                    "document_id": "d-1",
                    "content": f"Content {i}",
                    "metadata": {},
                }
                mock_point.score = 0.9 - (i * 0.05)
                mock_points.append(mock_point)

            mock_qdrant.query_points.return_value = _mock_qdrant_response(mock_points)

            result = await retrieve("Query", top_k=5)

            assert len(result) == 5
            for i in range(len(result) - 1):
                assert result[i].score >= result[i + 1].score

    @pytest.mark.asyncio
    async def test_retrieve_with_low_score_results(self) -> None:
        """retrieve() returns results even with low similarity scores."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai, patch(
            "api.services.retriever._qdrant"
        ) as mock_qdrant:
            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            mock_point = MagicMock()
            mock_point.payload = {
                "chunk_id": "c-1",
                "document_id": "d-1",
                "content": "Weakly relevant content",
                "metadata": {},
            }
            mock_point.score = 0.1
            mock_qdrant.query_points.return_value = _mock_qdrant_response([mock_point])

            result = await retrieve("Query", top_k=1)

            assert len(result) == 1
            assert result[0].score == 0.1


class TestRetrieveConfiguration:
    """Tests for configuration and model usage."""

    @pytest.mark.asyncio
    async def test_retrieve_uses_configured_embedding_model(self) -> None:
        """retrieve() uses the configured embedding model."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai, patch(
            "api.services.retriever._qdrant"
        ) as mock_qdrant, patch(
            "shared.embedding.OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ), patch(
            "shared.embedding.EMBEDDING_PROVIDER",
            "openai",
        ):
            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            mock_qdrant.query_points.return_value = _mock_qdrant_response([])

            await retrieve("Query", top_k=5)

            call_kwargs = oai_instance.embeddings.create.call_args.kwargs
            assert call_kwargs.get("model") == "text-embedding-3-small"

    @pytest.mark.asyncio
    async def test_retrieve_uses_configured_qdrant_collection(self) -> None:
        """retrieve() searches the configured Qdrant collection."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai, patch(
            "api.services.retriever._qdrant"
        ) as mock_qdrant, patch(
            "api.services.retriever.QDRANT_COLLECTION", "my-documents"
        ):
            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            mock_qdrant.query_points.return_value = _mock_qdrant_response([])

            await retrieve("Query", top_k=5)

            call_kwargs = mock_qdrant.query_points.call_args.kwargs
            assert call_kwargs.get("collection_name") == "my-documents"


class TestRetrieveChunkIntegrity:
    """Tests for chunk payload integrity."""

    @pytest.mark.asyncio
    async def test_retrieve_preserves_chunk_content(self) -> None:
        """Retrieved chunks preserve the original content."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai, patch(
            "api.services.retriever._qdrant"
        ) as mock_qdrant:
            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            original_content = "Important context information for the answer."
            mock_point = MagicMock()
            mock_point.payload = {
                "chunk_id": "c-1",
                "document_id": "d-1",
                "content": original_content,
                "metadata": {"source": "user_upload"},
            }
            mock_point.score = 0.95
            mock_qdrant.query_points.return_value = _mock_qdrant_response([mock_point])

            result = await retrieve("Query", top_k=5)

            assert result[0].chunk.content == original_content

    @pytest.mark.asyncio
    async def test_retrieve_includes_metadata(self) -> None:
        """Retrieved chunks include their metadata."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai, patch(
            "api.services.retriever._qdrant"
        ) as mock_qdrant:
            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            metadata = {"page": 5, "source": "document.pdf", "chunk_index": 10}
            mock_point = MagicMock()
            mock_point.payload = {
                "chunk_id": "c-1",
                "document_id": "d-1",
                "content": "Content",
                "metadata": metadata,
            }
            mock_point.score = 0.95
            mock_qdrant.query_points.return_value = _mock_qdrant_response([mock_point])

            result = await retrieve("Query", top_k=5)

            assert result[0].chunk.metadata == metadata
