# tests/unit/test_embedder.py
"""Unit tests for the embedding module."""

from unittest.mock import MagicMock, patch


from ingestion.embedder import embed_chunks, upsert_to_store
from shared.models import DocumentChunk


class TestEmbedChunks:
    """Tests for the embed_chunks function."""

    def test_embed_chunks_returns_vectors(self, sample_chunks) -> None:
        """embed_chunks returns a list of vectors."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai:
            client_instance = MagicMock()
            mock_openai.return_value = client_instance

            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536  # 1536-dim vector
            client_instance.embeddings.create.return_value.data = [
                mock_embedding
            ] * len(sample_chunks)

            result = embed_chunks(sample_chunks)

            assert isinstance(result, list)
            assert len(result) == len(sample_chunks)
            assert all(isinstance(v, list) for v in result)
            assert all(len(v) == 1536 for v in result)

    def test_embed_chunks_passes_correct_content(self, sample_chunks) -> None:
        """embed_chunks passes chunk content to the OpenAI API."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai:
            client_instance = MagicMock()
            mock_openai.return_value = client_instance

            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            client_instance.embeddings.create.return_value.data = [
                mock_embedding
            ] * len(sample_chunks)

            embed_chunks(sample_chunks)

            client_instance.embeddings.create.assert_called_once()
            call_kwargs = client_instance.embeddings.create.call_args.kwargs
            assert "input" in call_kwargs
            input_content = call_kwargs["input"]
            assert isinstance(input_content, list)
            assert len(input_content) == len(sample_chunks)
            assert all(
                input_content[i] == sample_chunks[i].content
                for i in range(len(sample_chunks))
            )

    def test_embed_chunks_uses_configured_model(self, sample_chunks) -> None:
        """embed_chunks uses the configured embedding model."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai, patch(
            "shared.embedding.OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ), patch(
            "shared.embedding.EMBEDDING_PROVIDER", "openai"
        ):
            client_instance = MagicMock()
            mock_openai.return_value = client_instance

            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            client_instance.embeddings.create.return_value.data = [
                mock_embedding
            ] * len(sample_chunks)

            embed_chunks(sample_chunks)

            call_kwargs = client_instance.embeddings.create.call_args.kwargs
            assert call_kwargs.get("model") == "text-embedding-3-small"

    def test_embed_chunks_single_chunk(self) -> None:
        """embed_chunks handles a single chunk correctly."""
        chunks = [
            DocumentChunk(
                chunk_id="c-1",
                document_id="doc-1",
                content="Single chunk content",
                metadata={},
            )
        ]

        with patch("shared.embedding.openai.OpenAI") as mock_openai:
            client_instance = MagicMock()
            mock_openai.return_value = client_instance

            mock_embedding = MagicMock()
            mock_embedding.embedding = [0.1] * 1536
            client_instance.embeddings.create.return_value.data = [mock_embedding]

            result = embed_chunks(chunks)

            assert len(result) == 1
            assert len(result[0]) == 1536

    def test_embed_chunks_empty_list(self) -> None:
        """embed_chunks handles an empty chunk list."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai:
            client_instance = MagicMock()
            mock_openai.return_value = client_instance
            client_instance.embeddings.create.return_value.data = []

            result = embed_chunks([])

            assert result == []

    def test_embed_chunks_preserves_order(self, sample_chunks) -> None:
        """Returned vectors are in the same order as input chunks."""
        with patch("shared.embedding.openai.OpenAI") as mock_openai:
            client_instance = MagicMock()
            mock_openai.return_value = client_instance

            mock_embeddings = []
            for i in range(len(sample_chunks)):
                mock_embedding = MagicMock()
                mock_embedding.embedding = [float(i)] * 1536
                mock_embeddings.append(mock_embedding)

            client_instance.embeddings.create.return_value.data = mock_embeddings

            result = embed_chunks(sample_chunks)

            for i, vector in enumerate(result):
                assert vector[0] == float(i)


class TestUpsertToStore:
    """Tests for the upsert_to_store function."""

    def test_upsert_returns_count(self, sample_chunks) -> None:
        """upsert_to_store returns the number of upserted points."""
        vectors = [[0.1] * 1536 for _ in sample_chunks]

        with patch("ingestion.embedder._qdrant") as mock_qdrant:
            mock_collection = MagicMock()
            mock_collection.name = "documents"
            mock_qdrant.get_collections.return_value.collections = [mock_collection]

            count = upsert_to_store(sample_chunks, vectors)

            assert count == len(sample_chunks)

    def test_upsert_creates_collection_if_missing(self, sample_chunks) -> None:
        """upsert_to_store creates the collection if it doesn't exist."""
        vectors = [[0.1] * 1536 for _ in sample_chunks]

        with patch("ingestion.embedder._qdrant") as mock_qdrant, patch(
            "ingestion.embedder.QDRANT_COLLECTION", "documents"
        ):
            mock_qdrant.get_collections.return_value.collections = []

            upsert_to_store(sample_chunks, vectors)

            mock_qdrant.create_collection.assert_called_once()
            call_kwargs = mock_qdrant.create_collection.call_args.kwargs
            assert call_kwargs["collection_name"] == "documents"

    def test_upsert_does_not_recreate_existing_collection(
        self, sample_chunks
    ) -> None:
        """upsert_to_store skips collection creation if it exists."""
        vectors = [[0.1] * 1536 for _ in sample_chunks]

        with patch("ingestion.embedder._qdrant") as mock_qdrant, patch(
            "ingestion.embedder.QDRANT_COLLECTION", "documents"
        ):
            mock_collection = MagicMock()
            mock_collection.name = "documents"
            mock_qdrant.get_collections.return_value.collections = [mock_collection]

            upsert_to_store(sample_chunks, vectors)

            mock_qdrant.create_collection.assert_not_called()

    def test_upsert_calls_qdrant_upsert(self, sample_chunks) -> None:
        """upsert_to_store calls the Qdrant upsert method."""
        vectors = [[0.1] * 1536 for _ in sample_chunks]

        with patch("ingestion.embedder._qdrant") as mock_qdrant:
            mock_collection = MagicMock()
            mock_collection.name = "documents"
            mock_qdrant.get_collections.return_value.collections = [mock_collection]

            upsert_to_store(sample_chunks, vectors)

            mock_qdrant.upsert.assert_called_once()

    def test_upsert_payload_structure(self, sample_source) -> None:
        """Upserted points have correct payload structure."""
        chunks = [sample_source]
        vectors = [[0.1] * 1536]

        with patch("ingestion.embedder._qdrant") as mock_qdrant:
            mock_collection = MagicMock()
            mock_collection.name = "documents"
            mock_qdrant.get_collections.return_value.collections = [mock_collection]

            upsert_to_store(chunks, vectors)

            call_kwargs = mock_qdrant.upsert.call_args.kwargs
            points = call_kwargs.get("points", [])
            assert len(points) == 1
            point = points[0]

            assert hasattr(point, "vector")
            assert hasattr(point, "payload")
            assert point.payload["chunk_id"] == sample_source.chunk_id
            assert point.payload["document_id"] == sample_source.document_id
            assert point.payload["content"] == sample_source.content

    def test_upsert_single_chunk(self, sample_source) -> None:
        """upsert_to_store handles a single chunk."""
        vectors = [[0.1] * 1536]

        with patch("ingestion.embedder._qdrant") as mock_qdrant:
            mock_collection = MagicMock()
            mock_collection.name = "documents"
            mock_qdrant.get_collections.return_value.collections = [mock_collection]

            count = upsert_to_store([sample_source], vectors)

            assert count == 1

    def test_upsert_empty_chunks(self) -> None:
        """upsert_to_store handles empty chunk list."""
        with patch("ingestion.embedder._qdrant") as mock_qdrant:
            mock_collection = MagicMock()
            mock_collection.name = "documents"
            mock_qdrant.get_collections.return_value.collections = [mock_collection]

            count = upsert_to_store([], [])

            assert count == 0
