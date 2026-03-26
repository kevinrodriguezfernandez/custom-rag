# tests/integration/test_ingest_endpoint.py
"""Integration tests for the /ingest endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from shared.models import DocumentChunk


class TestIngestEndpointBasic:
    """Tests for basic ingest endpoint functionality."""

    def test_ingest_endpoint_returns_200(self) -> None:
        """POST /ingest/ with valid file returns 200 OK."""
        client = TestClient(app)

        file_content = b"This is test content to ingest."
        files = {"file": ("test.txt", file_content, "text/plain")}
        data = {"document_id": "test-doc-1"}

        with patch("api.routes.ingest.load_document") as mock_load, patch(
            "api.routes.ingest.chunk_text"
        ) as mock_chunk, patch(
            "api.routes.ingest.embed_chunks"
        ) as mock_embed, patch(
            "api.routes.ingest.upsert_to_store"
        ) as mock_upsert:
            mock_load.return_value = "Loaded text content"
            mock_chunk.return_value = [
                DocumentChunk(
                    chunk_id="test-doc-1-0",
                    document_id="test-doc-1",
                    content="Chunk 1",
                ),
                DocumentChunk(
                    chunk_id="test-doc-1-1",
                    document_id="test-doc-1",
                    content="Chunk 2",
                ),
            ]
            mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]
            mock_upsert.return_value = 2

            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 200

    def test_ingest_endpoint_returns_ingest_response(self) -> None:
        """POST /ingest/ returns an IngestResponse with document_id and chunks_created."""
        client = TestClient(app)

        files = {"file": ("test.txt", b"Test content", "text/plain")}
        data = {"document_id": "my-doc"}

        with patch("api.routes.ingest.load_document") as mock_load, patch(
            "api.routes.ingest.chunk_text"
        ) as mock_chunk, patch(
            "api.routes.ingest.embed_chunks"
        ) as mock_embed, patch(
            "api.routes.ingest.upsert_to_store"
        ) as mock_upsert:
            mock_load.return_value = "Content"
            mock_chunk.return_value = [
                DocumentChunk(chunk_id="c-0", document_id="my-doc", content="C1"),
                DocumentChunk(chunk_id="c-1", document_id="my-doc", content="C2"),
                DocumentChunk(chunk_id="c-2", document_id="my-doc", content="C3"),
            ]
            mock_embed.return_value = [[0.1] * 1536] * 3
            mock_upsert.return_value = 3

            response = client.post("/ingest/", files=files, data=data)

            resp_data = response.json()
            assert resp_data["document_id"] == "my-doc"
            assert resp_data["chunks_created"] == 3

    def test_ingest_endpoint_accepts_txt_file(self) -> None:
        """POST /ingest/ accepts .txt files."""
        client = TestClient(app)

        files = {"file": ("document.txt", b"Text content", "text/plain")}
        data = {"document_id": "doc-txt"}

        with patch("api.routes.ingest.load_document") as mock_load, patch(
            "api.routes.ingest.chunk_text"
        ) as mock_chunk, patch(
            "api.routes.ingest.embed_chunks"
        ) as mock_embed, patch(
            "api.routes.ingest.upsert_to_store"
        ) as mock_upsert:
            mock_load.return_value = "Text content"
            mock_chunk.return_value = [
                DocumentChunk(chunk_id="c-0", document_id="doc-txt", content="C")
            ]
            mock_embed.return_value = [[0.1] * 1536]
            mock_upsert.return_value = 1

            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 200
            # Verify load_document was called
            mock_load.assert_called_once()

    def test_ingest_endpoint_accepts_md_file(self) -> None:
        """POST /ingest/ accepts .md (Markdown) files."""
        client = TestClient(app)

        files = {"file": ("document.md", b"# Markdown\nContent", "text/markdown")}
        data = {"document_id": "doc-md"}

        with patch("api.routes.ingest.load_document") as mock_load, patch(
            "api.routes.ingest.chunk_text"
        ) as mock_chunk, patch(
            "api.routes.ingest.embed_chunks"
        ) as mock_embed, patch(
            "api.routes.ingest.upsert_to_store"
        ) as mock_upsert:
            mock_load.return_value = "# Markdown\nContent"
            mock_chunk.return_value = [
                DocumentChunk(chunk_id="c-0", document_id="doc-md", content="C")
            ]
            mock_embed.return_value = [[0.1] * 1536]
            mock_upsert.return_value = 1

            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 200


class TestIngestEndpointValidation:
    """Tests for request validation."""

    def test_ingest_endpoint_missing_file(self) -> None:
        """POST /ingest/ without file returns 422."""
        client = TestClient(app)

        data = {"document_id": "doc-1"}
        response = client.post("/ingest/", data=data)

        assert response.status_code == 422

    def test_ingest_endpoint_missing_document_id(self) -> None:
        """POST /ingest/ without document_id returns 422."""
        client = TestClient(app)

        files = {"file": ("test.txt", b"Content")}
        response = client.post("/ingest/", files=files)

        assert response.status_code == 422

    def test_ingest_endpoint_empty_file(self) -> None:
        """POST /ingest/ with empty file is processed."""
        client = TestClient(app)

        files = {"file": ("empty.txt", b"", "text/plain")}
        data = {"document_id": "empty-doc"}

        with patch("api.routes.ingest.load_document") as mock_load, patch(
            "api.routes.ingest.chunk_text"
        ) as mock_chunk, patch(
            "api.routes.ingest.embed_chunks"
        ) as mock_embed, patch(
            "api.routes.ingest.upsert_to_store"
        ) as mock_upsert:
            mock_load.return_value = ""
            mock_chunk.return_value = []
            mock_embed.return_value = []
            mock_upsert.return_value = 0

            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 200
            resp_data = response.json()
            assert resp_data["chunks_created"] == 0


class TestIngestEndpointErrorHandling:
    """Tests for error handling in ingest endpoint."""

    def test_ingest_endpoint_unsupported_file_type_returns_422(self) -> None:
        """POST /ingest/ with unsupported file type returns 422."""
        client = TestClient(app)

        files = {"file": ("document.docx", b"Binary content", "application/vnd.openxmlformats")}
        data = {"document_id": "doc-unsupported"}

        with patch("api.routes.ingest.load_document") as mock_load:
            mock_load.side_effect = ValueError("Unsupported file type: .docx")

            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 422
            resp_data = response.json()
            assert "Unsupported file type" in resp_data.get("detail", "")

    def test_ingest_endpoint_loader_error_returns_500(self) -> None:
        """POST /ingest/ returns 500 if document loading fails."""
        client = TestClient(app)

        files = {"file": ("test.txt", b"Content")}
        data = {"document_id": "doc-load-error"}

        with patch("api.routes.ingest.load_document") as mock_load:
            mock_load.side_effect = Exception("Failed to read file")

            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 500
            resp_data = response.json()
            assert "Failed to load document" in resp_data.get("detail", "")

    def test_ingest_endpoint_chunking_error_returns_500(self) -> None:
        """POST /ingest/ returns 500 if chunking fails."""
        client = TestClient(app)

        files = {"file": ("test.txt", b"Content")}
        data = {"document_id": "doc-chunk-error"}

        with patch("api.routes.ingest.load_document") as mock_load, patch(
            "api.routes.ingest.chunk_text"
        ) as mock_chunk:
            mock_load.return_value = "Content"
            mock_chunk.side_effect = Exception("Chunking error")

            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 500
            resp_data = response.json()
            assert "Ingestion pipeline failed" in resp_data.get("detail", "")

    def test_ingest_endpoint_embedding_error_returns_500(self) -> None:
        """POST /ingest/ returns 500 if embedding fails."""
        client = TestClient(app)

        files = {"file": ("test.txt", b"Content")}
        data = {"document_id": "doc-embed-error"}

        with patch("api.routes.ingest.load_document") as mock_load, patch(
            "api.routes.ingest.chunk_text"
        ) as mock_chunk, patch(
            "api.routes.ingest.embed_chunks"
        ) as mock_embed:
            mock_load.return_value = "Content"
            mock_chunk.return_value = [
                DocumentChunk(chunk_id="c-0", document_id="doc-embed-error", content="C")
            ]
            mock_embed.side_effect = Exception("OpenAI API error")

            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 500

    def test_ingest_endpoint_upsert_error_returns_500(self) -> None:
        """POST /ingest/ returns 500 if upsert fails."""
        client = TestClient(app)

        files = {"file": ("test.txt", b"Content")}
        data = {"document_id": "doc-upsert-error"}

        with patch("api.routes.ingest.load_document") as mock_load, patch(
            "api.routes.ingest.chunk_text"
        ) as mock_chunk, patch(
            "api.routes.ingest.embed_chunks"
        ) as mock_embed, patch(
            "api.routes.ingest.upsert_to_store"
        ) as mock_upsert:
            mock_load.return_value = "Content"
            mock_chunk.return_value = [
                DocumentChunk(chunk_id="c-0", document_id="doc-upsert-error", content="C")
            ]
            mock_embed.return_value = [[0.1] * 1536]
            mock_upsert.side_effect = Exception("Qdrant connection error")

            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 500


class TestIngestEndpointPipeline:
    """Integration tests for the full ingest pipeline."""

    def test_ingest_endpoint_calls_full_pipeline(self) -> None:
        """POST /ingest/ calls load, chunk, embed, and upsert in sequence."""
        client = TestClient(app)

        files = {"file": ("test.txt", b"Document content")}
        data = {"document_id": "pipeline-test"}

        with patch("api.routes.ingest.load_document") as mock_load, patch(
            "api.routes.ingest.chunk_text"
        ) as mock_chunk, patch(
            "api.routes.ingest.embed_chunks"
        ) as mock_embed, patch(
            "api.routes.ingest.upsert_to_store"
        ) as mock_upsert:
            chunks = [
                DocumentChunk(chunk_id="c-0", document_id="pipeline-test", content="C1"),
                DocumentChunk(chunk_id="c-1", document_id="pipeline-test", content="C2"),
            ]
            mock_load.return_value = "Loaded content"
            mock_chunk.return_value = chunks
            mock_embed.return_value = [[0.1] * 1536, [0.2] * 1536]
            mock_upsert.return_value = 2

            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 200

            # Verify all functions were called
            mock_load.assert_called_once()
            mock_chunk.assert_called_once()
            mock_embed.assert_called_once()
            mock_upsert.assert_called_once()

    def test_ingest_endpoint_document_id_propagation(self) -> None:
        """POST /ingest/ propagates document_id through the pipeline."""
        client = TestClient(app)

        doc_id = "my-special-doc-42"
        files = {"file": ("test.txt", b"Content")}
        data = {"document_id": doc_id}

        with patch("api.routes.ingest.load_document") as mock_load, patch(
            "api.routes.ingest.chunk_text"
        ) as mock_chunk, patch(
            "api.routes.ingest.embed_chunks"
        ) as mock_embed, patch(
            "api.routes.ingest.upsert_to_store"
        ) as mock_upsert:
            mock_load.return_value = "Content"
            mock_chunk.return_value = [
                DocumentChunk(chunk_id=f"{doc_id}-0", document_id=doc_id, content="C")
            ]
            mock_embed.return_value = [[0.1] * 1536]
            mock_upsert.return_value = 1

            response = client.post("/ingest/", files=files, data=data)

            # Verify chunk_text was called with the correct document_id
            call_args = mock_chunk.call_args
            assert call_args.args[1] == doc_id

    def test_ingest_endpoint_file_suffix_detection(self) -> None:
        """POST /ingest/ detects file type from filename suffix."""
        client = TestClient(app)

        # Test with .md file
        files = {"file": ("readme.md", b"# Heading\nContent")}
        data = {"document_id": "md-test"}

        with patch("api.routes.ingest.load_document") as mock_load, patch(
            "api.routes.ingest.chunk_text"
        ) as mock_chunk, patch(
            "api.routes.ingest.embed_chunks"
        ) as mock_embed, patch(
            "api.routes.ingest.upsert_to_store"
        ) as mock_upsert:
            mock_load.return_value = "# Heading\nContent"
            mock_chunk.return_value = [
                DocumentChunk(chunk_id="c-0", document_id="md-test", content="C")
            ]
            mock_embed.return_value = [[0.1] * 1536]
            mock_upsert.return_value = 1

            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 200
            # load_document should be called once
            mock_load.assert_called_once()

    def test_ingest_endpoint_chunk_count_matches_response(self) -> None:
        """POST /ingest/ response chunks_created matches actual upsets."""
        client = TestClient(app)

        files = {"file": ("test.txt", b"Content" * 100)}  # Longer content = more chunks
        data = {"document_id": "multi-chunk"}

        num_chunks = 5
        chunks = [
            DocumentChunk(chunk_id=f"multi-chunk-{i}", document_id="multi-chunk", content=f"C{i}")
            for i in range(num_chunks)
        ]

        with patch("api.routes.ingest.load_document") as mock_load, patch(
            "api.routes.ingest.chunk_text"
        ) as mock_chunk, patch(
            "api.routes.ingest.embed_chunks"
        ) as mock_embed, patch(
            "api.routes.ingest.upsert_to_store"
        ) as mock_upsert:
            mock_load.return_value = "Content" * 100
            mock_chunk.return_value = chunks
            mock_embed.return_value = [[0.1] * 1536] * num_chunks
            mock_upsert.return_value = num_chunks

            response = client.post("/ingest/", files=files, data=data)

            resp_data = response.json()
            assert resp_data["chunks_created"] == num_chunks
