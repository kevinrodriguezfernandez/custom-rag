# tests/integration/test_ingest_endpoint.py
"""Integration tests for the /ingest endpoint."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app


class TestIngestEndpointBasic:
    """Tests for basic ingest endpoint functionality."""

    def test_ingest_endpoint_returns_200(self) -> None:
        """POST /ingest/ with valid file returns 200 OK."""
        client = TestClient(app)

        files = {"file": ("test.txt", b"This is test content to ingest.", "text/plain")}
        data = {"document_id": "test-doc-1"}

        with patch("api.routes.ingest.run_pipeline", return_value=2) as mock_pipeline:
            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 200
            mock_pipeline.assert_called_once()

    def test_ingest_endpoint_returns_ingest_response(self) -> None:
        """POST /ingest/ returns an IngestResponse with document_id and chunks_created."""
        client = TestClient(app)

        files = {"file": ("test.txt", b"Test content", "text/plain")}
        data = {"document_id": "my-doc"}

        with patch("api.routes.ingest.run_pipeline", return_value=3):
            response = client.post("/ingest/", files=files, data=data)

            resp_data = response.json()
            assert resp_data["document_id"] == "my-doc"
            assert resp_data["chunks_created"] == 3

    def test_ingest_endpoint_accepts_txt_file(self) -> None:
        """POST /ingest/ accepts .txt files."""
        client = TestClient(app)

        files = {"file": ("document.txt", b"Text content", "text/plain")}
        data = {"document_id": "doc-txt"}

        with patch("api.routes.ingest.run_pipeline", return_value=1) as mock_pipeline:
            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 200
            mock_pipeline.assert_called_once()

    def test_ingest_endpoint_accepts_md_file(self) -> None:
        """POST /ingest/ accepts .md (Markdown) files."""
        client = TestClient(app)

        files = {"file": ("document.md", b"# Markdown\nContent", "text/markdown")}
        data = {"document_id": "doc-md"}

        with patch("api.routes.ingest.run_pipeline", return_value=1):
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

        with patch("api.routes.ingest.run_pipeline", return_value=0):
            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 200
            resp_data = response.json()
            assert resp_data["chunks_created"] == 0


class TestIngestEndpointErrorHandling:
    """Tests for error handling in ingest endpoint."""

    def test_ingest_endpoint_unsupported_file_type_returns_422(self) -> None:
        """POST /ingest/ with unsupported file type returns 422."""
        client = TestClient(app)

        files = {"file": ("document.csv", b"col1,col2", "text/csv")}
        data = {"document_id": "doc-unsupported"}

        with patch(
            "api.routes.ingest.run_pipeline",
            side_effect=ValueError("Unsupported file type: .csv"),
        ):
            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 422
            resp_data = response.json()
            assert "Unsupported file type" in resp_data.get("detail", "")

    def test_ingest_endpoint_pipeline_error_returns_500(self) -> None:
        """POST /ingest/ returns 500 if the pipeline raises an unexpected error."""
        client = TestClient(app)

        files = {"file": ("test.txt", b"Content")}
        data = {"document_id": "doc-error"}

        with patch(
            "api.routes.ingest.run_pipeline",
            side_effect=Exception("Unexpected pipeline failure"),
        ):
            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 500
            resp_data = response.json()
            assert "Ingestion pipeline failed" in resp_data.get("detail", "")


class TestIngestEndpointPipeline:
    """Integration tests for the full ingest pipeline."""

    def test_ingest_endpoint_calls_run_pipeline_with_correct_args(self) -> None:
        """POST /ingest/ delegates to run_pipeline with correct arguments."""
        client = TestClient(app)

        doc_id = "my-special-doc-42"
        file_content = b"Document content"
        files = {"file": ("test.txt", file_content, "text/plain")}
        data = {"document_id": doc_id}

        with patch("api.routes.ingest.run_pipeline", return_value=1) as mock_pipeline:
            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 200
            mock_pipeline.assert_called_once()
            call_args = mock_pipeline.call_args
            # run_pipeline(file_bytes, filename, document_id, chat_id)
            assert call_args.args[0] == file_content
            assert call_args.args[1] == "test.txt"
            assert call_args.args[2] == doc_id

    def test_ingest_endpoint_chunk_count_matches_response(self) -> None:
        """POST /ingest/ response chunks_created matches pipeline return value."""
        client = TestClient(app)

        num_chunks = 5
        files = {"file": ("test.txt", b"Content" * 100)}
        data = {"document_id": "multi-chunk"}

        with patch("api.routes.ingest.run_pipeline", return_value=num_chunks):
            response = client.post("/ingest/", files=files, data=data)

            resp_data = response.json()
            assert resp_data["chunks_created"] == num_chunks

    def test_ingest_endpoint_passes_chat_id(self) -> None:
        """POST /ingest/ passes chat_id to run_pipeline."""
        client = TestClient(app)

        files = {"file": ("test.txt", b"Content")}
        data = {"document_id": "doc-1", "chat_id": "session-abc"}

        with patch("api.routes.ingest.run_pipeline", return_value=1) as mock_pipeline:
            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 200
            call_args = mock_pipeline.call_args
            assert call_args.args[3] == "session-abc"
