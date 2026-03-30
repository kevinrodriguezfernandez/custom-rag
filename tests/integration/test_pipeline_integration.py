# tests/integration/test_pipeline_integration.py
"""End-to-end integration tests for the complete RAG pipeline."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app
from shared.models import DocumentChunk, RetrievedChunk


class TestIngestAndChatPipeline:
    """Tests for complete document ingestion and retrieval pipeline."""

    def test_ingest_then_chat_flow(self) -> None:
        """Ingest a document then retrieve it via chat."""
        client = TestClient(app)

        # Step 1: Ingest document
        files = {"file": ("doc.txt", b"AI is artificial intelligence", "text/plain")}
        ingest_data = {"document_id": "ai-doc"}
        chunk = DocumentChunk(
            chunk_id="ai-doc-0",
            document_id="ai-doc",
            content="AI is artificial intelligence"
        )

        with patch("api.routes.ingest.run_pipeline", return_value=1):
            ingest_response = client.post("/ingest/", files=files, data=ingest_data)
            assert ingest_response.status_code == 200
            assert ingest_response.json()["chunks_created"] == 1

        # Step 2: Chat with the ingested content
        chat_request = {
            "query": "What is AI?",
            "top_k": 5,
            "model": "gpt-4o-mini",
        }

        with patch("api.routes.chat.retrieve") as mock_retrieve, \
             patch("api.routes.chat.generate_answer") as mock_generate:

            retrieved = [RetrievedChunk(chunk=chunk, score=0.95)]
            mock_retrieve.return_value = retrieved
            mock_generate.return_value = "Artificial Intelligence is..."

            chat_response = client.post("/chat/", json=chat_request)
            assert chat_response.status_code == 200
            assert "Artificial Intelligence" in chat_response.json()["answer"]


class TestErrorPropagation:
    """Tests for error propagation across layers."""

    def test_ingest_with_unsupported_file_extension(self) -> None:
        """Ingest endpoint properly rejects unsupported file types."""
        client = TestClient(app)

        files = {"file": ("test.csv", b"col1,col2")}
        data = {"document_id": "fail-doc"}

        with patch(
            "api.routes.ingest.run_pipeline",
            side_effect=ValueError("Unsupported file type: .csv"),
        ):
            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 422
            assert "Unsupported file type" in response.json().get("detail", "")

    def test_chat_retrieval_error_continues_without_context(self) -> None:
        """Chat endpoint continues with empty context when retrieval fails."""
        client = TestClient(app)

        request = {"query": "Test?", "top_k": 5}

        with patch("api.routes.chat.retrieve") as mock_retrieve, patch(
            "api.routes.chat.generate_answer"
        ) as mock_generate:
            mock_retrieve.side_effect = RuntimeError("Qdrant unreachable")
            mock_generate.return_value = "Answer without context"

            response = client.post("/chat/", json=request)

            assert response.status_code == 200
            assert response.json()["sources"] == []

    def test_chat_generation_error_returns_502(self) -> None:
        """Chat endpoint returns 502 when LLM generation fails."""
        client = TestClient(app)

        request = {"query": "Test?", "top_k": 5}

        with patch("api.routes.chat.retrieve") as mock_retrieve, \
             patch("api.routes.chat.generate_answer") as mock_generate:

            mock_retrieve.return_value = []
            mock_generate.side_effect = RuntimeError("API timeout")

            response = client.post("/chat/", json=request)

            assert response.status_code == 502
            assert "LLM generation failed" in response.json()["detail"]


class TestLargeDocumentHandling:
    """Tests for handling large documents."""

    def test_ingest_very_large_file(self) -> None:
        """Ingest endpoint handles large file uploads."""
        client = TestClient(app)

        large_content = b"Content " * 131072  # ~1MB
        files = {"file": ("large.txt", large_content, "text/plain")}
        data = {"document_id": "large-doc"}

        with patch("api.routes.ingest.run_pipeline", return_value=10):
            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 200
            assert response.json()["chunks_created"] == 10


class TestChatWithMultipleSources:
    """Tests for chat with multiple retrieved sources."""

    def test_chat_with_many_sources(self) -> None:
        """Chat endpoint returns many sources correctly."""
        client = TestClient(app)

        # Create many source chunks
        chunks = [
            DocumentChunk(
                chunk_id=f"chunk-{i}",
                document_id="doc-1",
                content=f"Information {i}"
            )
            for i in range(10)
        ]
        retrieved = [
            RetrievedChunk(chunk=chunk, score=0.95 - i*0.01)
            for i, chunk in enumerate(chunks)
        ]

        request = {"query": "Multi-source query?", "top_k": 10}

        with patch("api.routes.chat.retrieve") as mock_retrieve, \
             patch("api.routes.chat.generate_answer") as mock_generate:

            mock_retrieve.return_value = retrieved
            mock_generate.return_value = "Answer based on multiple sources"

            response = client.post("/chat/", json=request)

            data = response.json()
            assert len(data["sources"]) == 10
            # Verify scores are in descending order
            for i in range(len(data["sources"]) - 1):
                assert data["sources"][i]["score"] >= data["sources"][i+1]["score"]

    def test_chat_source_format(self) -> None:
        """Chat sources include all required fields."""
        client = TestClient(app)

        chunk = DocumentChunk(
            chunk_id="test-chunk",
            document_id="test-doc",
            content="Test content",
            metadata={"page": 1, "source": "document.pdf"}
        )
        retrieved = [RetrievedChunk(chunk=chunk, score=0.92)]

        request = {"query": "Test?"}

        with patch("api.routes.chat.retrieve") as mock_retrieve, \
             patch("api.routes.chat.generate_answer") as mock_generate:

            mock_retrieve.return_value = retrieved
            mock_generate.return_value = "Test answer"

            response = client.post("/chat/", json=request)

            source = response.json()["sources"][0]
            assert "chunk" in source
            assert source["chunk"]["chunk_id"] == "test-chunk"
            assert source["chunk"]["document_id"] == "test-doc"
            assert source["chunk"]["content"] == "Test content"
            assert source["chunk"]["metadata"]["page"] == 1
            assert source["score"] == 0.92


class TestMultipleDocumentTypes:
    """Tests for handling different document types."""

    def test_ingest_txt_md_pdf_files(self) -> None:
        """Ingest endpoint properly routes different file types."""
        client = TestClient(app)

        file_configs = [
            ("doc.txt", b"Text content", "text/plain"),
            ("doc.md", b"# Markdown", "text/markdown"),
            ("doc.pdf", b"%PDF-1.4", "application/pdf"),
        ]

        for filename, content, mime_type in file_configs:
            files = {"file": (filename, content, mime_type)}
            data = {"document_id": f"doc-{filename.split('.')[1]}"}

            with patch("api.routes.ingest.run_pipeline", return_value=1) as mock_pipeline:
                response = client.post("/ingest/", files=files, data=data)

                assert response.status_code == 200
                mock_pipeline.assert_called_once()


class TestModelVariations:
    """Tests for different model selections."""

    def test_chat_with_various_models(self) -> None:
        """Chat endpoint works with different model names."""
        client = TestClient(app)

        models = [
            "gpt-4o-mini",
            "gpt-4",
            "claude-opus-4-1",
            "claude-3-sonnet-20240229",
            "llama3",
            "mistral",
        ]

        for model in models:
            request = {"query": "Test?", "model": model}

            with patch("api.routes.chat.retrieve") as mock_retrieve, \
                 patch("api.routes.chat.generate_answer") as mock_generate:

                mock_retrieve.return_value = []
                mock_generate.return_value = f"{model} response"

                response = client.post("/chat/", json=request)

                assert response.status_code == 200
                assert response.json()["model"] == model
                assert mock_generate.called
                # Verify the model was passed to generate_answer
                call_kwargs = mock_generate.call_args.kwargs
                assert call_kwargs["model"] == model


class TestDocumentIDPropagation:
    """Tests for document ID propagation through the pipeline."""

    def test_document_id_preserved_through_pipeline(self) -> None:
        """Document ID is preserved from ingest through to the response."""
        client = TestClient(app)

        doc_id = "my-special-document-42"
        files = {"file": ("doc.txt", b"Content", "text/plain")}
        data = {"document_id": doc_id}

        with patch("api.routes.ingest.run_pipeline", return_value=1) as mock_pipeline:
            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 200
            assert response.json()["document_id"] == doc_id
            # Verify run_pipeline was called with the correct document_id
            call_args = mock_pipeline.call_args
            assert call_args.args[2] == doc_id
