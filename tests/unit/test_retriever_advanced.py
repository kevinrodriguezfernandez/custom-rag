# tests/unit/test_retriever_advanced.py
"""Advanced unit tests for the retriever service."""

from unittest.mock import MagicMock, patch
import pytest

from api.services.retriever import retrieve
from shared.models import DocumentChunk, RetrievedChunk


def _mock_qdrant_response(points):
    """Return a MagicMock that mimics query_points() response."""
    mock_response = MagicMock()
    mock_response.points = points
    return mock_response


class TestRetrieveAdvanced:
    """Advanced tests for the retrieve function."""

    @pytest.mark.asyncio
    async def test_retrieve_with_unicode_query(self) -> None:
        """retrieve() handles unicode characters in query."""
        with patch("api.services.retriever.openai.OpenAI") as mock_openai, \
             patch("api.services.retriever.QdrantClient") as mock_qdrant:

            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            qdrant_instance = MagicMock()
            mock_qdrant.return_value = qdrant_instance
            qdrant_instance.query_points.return_value = _mock_qdrant_response([])

            result = await retrieve("你好 مرحبا café", top_k=5)

            # Verify unicode query was passed
            call_kwargs = oai_instance.embeddings.create.call_args.kwargs
            assert call_kwargs["input"] == "你好 مرحبا café"
            assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_with_very_long_query(self) -> None:
        """retrieve() handles very long query strings."""
        long_query = "Q?" * 5000

        with patch("api.services.retriever.openai.OpenAI") as mock_openai, \
             patch("api.services.retriever.QdrantClient") as mock_qdrant:

            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            qdrant_instance = MagicMock()
            mock_qdrant.return_value = qdrant_instance
            qdrant_instance.query_points.return_value = _mock_qdrant_response([])

            result = await retrieve(long_query, top_k=5)

            call_kwargs = oai_instance.embeddings.create.call_args.kwargs
            assert call_kwargs["input"] == long_query
            assert result == []

    @pytest.mark.asyncio
    async def test_retrieve_with_special_characters(self) -> None:
        """retrieve() handles special characters in query."""
        special_query = "!@#$%^&*()_+-=[]{}|;:',.<>?/`~"

        with patch("api.services.retriever.openai.OpenAI") as mock_openai, \
             patch("api.services.retriever.QdrantClient") as mock_qdrant:

            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            qdrant_instance = MagicMock()
            mock_qdrant.return_value = qdrant_instance
            qdrant_instance.query_points.return_value = _mock_qdrant_response([])

            result = await retrieve(special_query, top_k=5)

            call_kwargs = oai_instance.embeddings.create.call_args.kwargs
            assert call_kwargs["input"] == special_query

    @pytest.mark.asyncio
    async def test_retrieve_payload_reconstruction(self) -> None:
        """retrieve() correctly reconstructs DocumentChunk from Qdrant payload."""
        with patch("api.services.retriever.openai.OpenAI") as mock_openai, \
             patch("api.services.retriever.QdrantClient") as mock_qdrant:

            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            qdrant_instance = MagicMock()
            mock_qdrant.return_value = qdrant_instance

            # Create a realistic payload
            payload = {
                "chunk_id": "doc-1-5",
                "document_id": "doc-1",
                "content": "Detailed context information",
                "metadata": {
                    "page": 5,
                    "source": "document.pdf",
                    "section": "Introduction"
                }
            }

            mock_point = MagicMock()
            mock_point.payload = payload
            mock_point.score = 0.87
            qdrant_instance.query_points.return_value = _mock_qdrant_response([mock_point])

            result = await retrieve("Query", top_k=5)

            assert len(result) == 1
            chunk = result[0].chunk
            assert chunk.chunk_id == "doc-1-5"
            assert chunk.document_id == "doc-1"
            assert chunk.content == "Detailed context information"
            assert chunk.metadata["page"] == 5
            assert chunk.metadata["source"] == "document.pdf"

    @pytest.mark.asyncio
    async def test_retrieve_with_boundary_top_k_values(self) -> None:
        """retrieve() handles edge case top_k values."""
        with patch("api.services.retriever.openai.OpenAI") as mock_openai, \
             patch("api.services.retriever.QdrantClient") as mock_qdrant:

            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            qdrant_instance = MagicMock()
            mock_qdrant.return_value = qdrant_instance
            qdrant_instance.query_points.return_value = _mock_qdrant_response([])

            # Test with top_k = 1 (minimum)
            await retrieve("Query", top_k=1)
            call_kwargs = qdrant_instance.query_points.call_args.kwargs
            assert call_kwargs["limit"] == 1

            # Test with top_k = 100 (large)
            qdrant_instance.reset_mock()
            qdrant_instance.query_points.return_value = _mock_qdrant_response([])
            await retrieve("Query", top_k=100)
            call_kwargs = qdrant_instance.query_points.call_args.kwargs
            assert call_kwargs["limit"] == 100

    @pytest.mark.asyncio
    async def test_retrieve_scores_not_filtered(self) -> None:
        """retrieve() returns all results regardless of score."""
        with patch("api.services.retriever.openai.OpenAI") as mock_openai, \
             patch("api.services.retriever.QdrantClient") as mock_qdrant:

            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            qdrant_instance = MagicMock()
            mock_qdrant.return_value = qdrant_instance

            # Create results with varying scores including very low ones
            mock_points = []
            for score in [0.99, 0.50, 0.25, 0.01]:
                mock_point = MagicMock()
                mock_point.payload = {
                    "chunk_id": "c-1",
                    "document_id": "d-1",
                    "content": f"Content with score {score}",
                    "metadata": {}
                }
                mock_point.score = score
                mock_points.append(mock_point)

            qdrant_instance.query_points.return_value = _mock_qdrant_response(mock_points)

            result = await retrieve("Query", top_k=5)

            # All results should be returned, even low-scoring ones
            assert len(result) == 4
            scores = [r.score for r in result]
            assert 0.01 in scores  # Very low score is included

    @pytest.mark.asyncio
    async def test_retrieve_with_missing_payload_fields(self) -> None:
        """retrieve() reconstructs DocumentChunk with default values for missing fields."""
        with patch("api.services.retriever.openai.OpenAI") as mock_openai, \
             patch("api.services.retriever.QdrantClient") as mock_qdrant:

            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            qdrant_instance = MagicMock()
            mock_qdrant.return_value = qdrant_instance

            # Payload with missing metadata (should default to {})
            payload = {
                "chunk_id": "c-1",
                "document_id": "d-1",
                "content": "Content"
            }

            mock_point = MagicMock()
            mock_point.payload = payload
            mock_point.score = 0.95
            qdrant_instance.query_points.return_value = _mock_qdrant_response([mock_point])

            result = await retrieve("Query", top_k=5)

            chunk = result[0].chunk
            assert chunk.metadata == {}  # Default empty dict

    @pytest.mark.asyncio
    async def test_retrieve_embedding_order_preservation(self) -> None:
        """retrieve() uses the exact embedding vector for search."""
        with patch("api.services.retriever.openai.OpenAI") as mock_openai, \
             patch("api.services.retriever.QdrantClient") as mock_qdrant:

            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance

            # Create a distinctive embedding
            distinctive_vector = [0.1, 0.2, 0.3, 0.4, 0.5] + [0.0] * 1531
            mock_embedding = MagicMock()
            mock_embedding.embedding = distinctive_vector
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            qdrant_instance = MagicMock()
            mock_qdrant.return_value = qdrant_instance
            qdrant_instance.query_points.return_value = _mock_qdrant_response([])

            await retrieve("Query", top_k=5)

            # Verify the exact vector was passed to query_points
            call_kwargs = qdrant_instance.query_points.call_args.kwargs
            assert call_kwargs["query"] == distinctive_vector

    @pytest.mark.asyncio
    async def test_retrieve_constructs_retrieved_chunks_correctly(self) -> None:
        """retrieve() builds RetrievedChunk objects with correct structure."""
        with patch("api.services.retriever.openai.OpenAI") as mock_openai, \
             patch("api.services.retriever.QdrantClient") as mock_qdrant:

            oai_instance = MagicMock()
            mock_openai.return_value = oai_instance
            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            oai_instance.embeddings.create.return_value.data = [mock_embedding]

            qdrant_instance = MagicMock()
            mock_qdrant.return_value = qdrant_instance

            mock_point = MagicMock()
            mock_point.payload = {
                "chunk_id": "c-1",
                "document_id": "d-1",
                "content": "Test",
                "metadata": {}
            }
            mock_point.score = 0.95
            qdrant_instance.query_points.return_value = _mock_qdrant_response([mock_point])

            result = await retrieve("Query", top_k=5)

            assert len(result) == 1
            assert isinstance(result[0], RetrievedChunk)
            assert isinstance(result[0].chunk, DocumentChunk)
            assert result[0].score == 0.95
            assert result[0].chunk.chunk_id == "c-1"
