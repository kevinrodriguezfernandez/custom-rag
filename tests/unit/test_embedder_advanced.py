# tests/unit/test_embedder_advanced.py
"""Advanced unit tests for the embedding module."""

from unittest.mock import MagicMock, patch

from ingestion.embedder import embed_chunks, upsert_to_store
from shared.models import DocumentChunk


class TestEmbedChunksAdvanced:
    """Advanced tests for embed_chunks function."""

    def test_embed_chunks_with_unicode_content(self) -> None:
        """embed_chunks handles unicode content correctly."""
        chunks = [
            DocumentChunk(
                chunk_id="unicode-1",
                document_id="doc-1",
                content="Hello 你好 مرحبا café"
            ),
            DocumentChunk(
                chunk_id="unicode-2",
                document_id="doc-1",
                content="Москва Αθήνα 日本"
            ),
        ]

        with patch("shared.embedding.openai.OpenAI") as mock_openai:
            client_instance = MagicMock()
            mock_openai.return_value = client_instance

            mock_embeddings = [
                MagicMock(embedding=[0.1] * 1536),
                MagicMock(embedding=[0.2] * 1536),
            ]
            client_instance.embeddings.create.return_value.data = mock_embeddings

            result = embed_chunks(chunks)

            assert len(result) == 2
            call_kwargs = client_instance.embeddings.create.call_args.kwargs
            input_content = call_kwargs["input"]
            assert input_content[0] == "Hello 你好 مرحبا café"
            assert input_content[1] == "Москва Αθήνα 日本"

    def test_embed_chunks_with_very_long_content(self) -> None:
        """embed_chunks handles very long chunk content."""
        long_content = "A" * 10000
        chunks = [
            DocumentChunk(
                chunk_id="long-1",
                document_id="doc-1",
                content=long_content
            )
        ]

        with patch("shared.embedding.openai.OpenAI") as mock_openai:
            client_instance = MagicMock()
            mock_openai.return_value = client_instance

            mock_embedding = MagicMock(embedding=[0.1] * 1536)
            client_instance.embeddings.create.return_value.data = [mock_embedding]

            result = embed_chunks(chunks)

            assert len(result) == 1
            assert len(result[0]) == 1536
            call_kwargs = client_instance.embeddings.create.call_args.kwargs
            assert call_kwargs["input"][0] == long_content

    def test_embed_chunks_with_special_characters(self) -> None:
        """embed_chunks preserves special characters."""
        special_content = "!@#$%^&*()_+-=[]{}|;:',.<>?/`~\n\t"
        chunks = [
            DocumentChunk(
                chunk_id="special-1",
                document_id="doc-1",
                content=special_content
            )
        ]

        with patch("shared.embedding.openai.OpenAI") as mock_openai:
            client_instance = MagicMock()
            mock_openai.return_value = client_instance

            mock_embedding = MagicMock(embedding=[0.1] * 1536)
            client_instance.embeddings.create.return_value.data = [mock_embedding]

            result = embed_chunks(chunks)

            call_kwargs = client_instance.embeddings.create.call_args.kwargs
            assert call_kwargs["input"][0] == special_content

    def test_embed_chunks_different_dimensions(self) -> None:
        """embed_chunks handles various embedding dimensions."""
        chunks = [
            DocumentChunk(
                chunk_id="c-1",
                document_id="doc-1",
                content="Content"
            )
        ]

        for vector_size in [512, 768, 1536, 3072]:
            with patch("shared.embedding.openai.OpenAI") as mock_openai:
                client_instance = MagicMock()
                mock_openai.return_value = client_instance

                mock_embedding = MagicMock(embedding=[0.1] * vector_size)
                client_instance.embeddings.create.return_value.data = [mock_embedding]

                result = embed_chunks(chunks)

                assert len(result[0]) == vector_size

    def test_embed_chunks_with_metadata(self) -> None:
        """embed_chunks only passes content, not metadata."""
        chunks = [
            DocumentChunk(
                chunk_id="c-1",
                document_id="doc-1",
                content="Content to embed",
                metadata={"page": 5, "source": "doc.pdf"}
            )
        ]

        with patch("shared.embedding.openai.OpenAI") as mock_openai:
            client_instance = MagicMock()
            mock_openai.return_value = client_instance

            mock_embedding = MagicMock(embedding=[0.1] * 1536)
            client_instance.embeddings.create.return_value.data = [mock_embedding]

            result = embed_chunks(chunks)

            call_kwargs = client_instance.embeddings.create.call_args.kwargs
            input_content = call_kwargs["input"]
            assert input_content == ["Content to embed"]
            assert "metadata" not in str(input_content)


class TestUpsertToStoreAdvanced:
    """Advanced tests for upsert_to_store function."""

    def test_upsert_with_mismatched_chunk_vector_count(self) -> None:
        """upsert_to_store handles mismatched chunk/vector counts."""
        chunks = [
            DocumentChunk(chunk_id="c-1", document_id="d-1", content="C1"),
            DocumentChunk(chunk_id="c-2", document_id="d-1", content="C2"),
        ]
        vectors = [[0.1] * 1536]  # Only 1 vector for 2 chunks

        with patch("ingestion.embedder._qdrant") as mock_qdrant:
            mock_collection = MagicMock()
            mock_collection.name = "documents"
            mock_qdrant.get_collections.return_value.collections = [mock_collection]

            # zip truncates to the shorter list — only 1 point upserted
            count = upsert_to_store(chunks, vectors)

            assert count == 1

    def test_upsert_preserves_chunk_metadata(self) -> None:
        """upsert_to_store includes all chunk metadata in payload."""
        chunk = DocumentChunk(
            chunk_id="c-1",
            document_id="d-1",
            content="Content",
            metadata={"page": 5, "source": "doc.pdf", "custom": "value"}
        )
        chunks = [chunk]
        vectors = [[0.1] * 1536]

        with patch("ingestion.embedder._qdrant") as mock_qdrant:
            mock_collection = MagicMock()
            mock_collection.name = "documents"
            mock_qdrant.get_collections.return_value.collections = [mock_collection]

            upsert_to_store(chunks, vectors)

            call_kwargs = mock_qdrant.upsert.call_args.kwargs
            points = call_kwargs["points"]
            payload = points[0].payload

            assert payload["metadata"]["page"] == 5
            assert payload["metadata"]["source"] == "doc.pdf"
            assert payload["metadata"]["custom"] == "value"

    def test_upsert_with_empty_metadata(self) -> None:
        """upsert_to_store handles chunks with empty metadata."""
        chunk = DocumentChunk(
            chunk_id="c-1",
            document_id="d-1",
            content="Content",
            metadata={}
        )
        chunks = [chunk]
        vectors = [[0.1] * 1536]

        with patch("ingestion.embedder._qdrant") as mock_qdrant:
            mock_collection = MagicMock()
            mock_collection.name = "documents"
            mock_qdrant.get_collections.return_value.collections = [mock_collection]

            count = upsert_to_store(chunks, vectors)

            assert count == 1
            call_kwargs = mock_qdrant.upsert.call_args.kwargs
            payload = call_kwargs["points"][0].payload
            assert payload["metadata"] == {}

    def test_upsert_hash_distribution(self) -> None:
        """upsert_to_store creates unique deterministic IDs for different chunk_ids."""
        chunks = [
            DocumentChunk(chunk_id=f"chunk-{i}", document_id="d-1", content=f"C{i}")
            for i in range(5)
        ]
        vectors = [[0.1 + i * 0.01] * 1536 for i in range(5)]

        with patch("ingestion.embedder._qdrant") as mock_qdrant:
            mock_collection = MagicMock()
            mock_collection.name = "documents"
            mock_qdrant.get_collections.return_value.collections = [mock_collection]

            upsert_to_store(chunks, vectors)

            call_kwargs = mock_qdrant.upsert.call_args.kwargs
            points = call_kwargs["points"]
            ids = [p.id for p in points]

            # All IDs should be unique
            assert len(ids) == len(set(ids))

    def test_upsert_with_very_large_vectors(self) -> None:
        """upsert_to_store handles large vector dimensions."""
        chunk = DocumentChunk(chunk_id="c-1", document_id="d-1", content="C")
        chunks = [chunk]
        large_vector = [[0.5] * 4096]

        with patch("ingestion.embedder._qdrant") as mock_qdrant:
            mock_collection = MagicMock()
            mock_collection.name = "documents"
            mock_qdrant.get_collections.return_value.collections = [mock_collection]

            count = upsert_to_store(chunks, large_vector)

            assert count == 1
            call_kwargs = mock_qdrant.upsert.call_args.kwargs
            point = call_kwargs["points"][0]
            assert len(point.vector) == 4096

    def test_upsert_collection_name_from_config(self) -> None:
        """upsert_to_store uses configured collection name."""
        chunk = DocumentChunk(chunk_id="c-1", document_id="d-1", content="C")
        chunks = [chunk]
        vectors = [[0.1] * 1536]

        with patch("ingestion.embedder._qdrant") as mock_qdrant, \
             patch("ingestion.embedder.QDRANT_COLLECTION", "custom-collection"):

            mock_collection = MagicMock()
            mock_collection.name = "custom-collection"
            mock_qdrant.get_collections.return_value.collections = [mock_collection]

            upsert_to_store(chunks, vectors)

            call_kwargs = mock_qdrant.upsert.call_args.kwargs
            assert call_kwargs["collection_name"] == "custom-collection"

    def test_upsert_deterministic_point_ids(self) -> None:
        """Same chunk_id always produces the same point ID (deterministic hash)."""
        chunk = DocumentChunk(chunk_id="stable-id", document_id="d-1", content="C")
        chunks = [chunk]
        vectors = [[0.1] * 1536]

        with patch("ingestion.embedder._qdrant") as mock_qdrant:
            mock_collection = MagicMock()
            mock_collection.name = "documents"
            mock_qdrant.get_collections.return_value.collections = [mock_collection]

            upsert_to_store(chunks, vectors)
            first_id = mock_qdrant.upsert.call_args.kwargs["points"][0].id

            mock_qdrant.reset_mock()
            mock_qdrant.get_collections.return_value.collections = [mock_collection]

            upsert_to_store(chunks, vectors)
            second_id = mock_qdrant.upsert.call_args.kwargs["points"][0].id

            assert first_id == second_id
