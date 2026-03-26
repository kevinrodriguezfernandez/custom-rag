# tests/integration/test_pipeline_integration.py
"""End-to-end integration tests for the complete RAG pipeline."""

from unittest.mock import MagicMock, patch
import tempfile
from pathlib import Path

import pytest
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

        with patch("api.routes.ingest.load_document") as mock_load, \
             patch("api.routes.ingest.chunk_text") as mock_chunk, \
             patch("api.routes.ingest.embed_chunks") as mock_embed, \
             patch("api.routes.ingest.upsert_to_store") as mock_upsert:

            mock_load.return_value = "AI is artificial intelligence"
            chunk = DocumentChunk(
                chunk_id="ai-doc-0",
                document_id="ai-doc",
                content="AI is artificial intelligence"
            )
            mock_chunk.return_value = [chunk]
            mock_embed.return_value = [[0.1] * 1536]
            mock_upsert.return_value = 1

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

        files = {"file": ("test.docx", b"binary content")}
        data = {"document_id": "fail-doc"}

        with patch("api.routes.ingest.load_document") as mock_load:
            mock_load.side_effect = ValueError("Unsupported file type: .docx")

            response = client.post("/ingest/", files=files, data=data)

            # ValueError from load_document is caught and returns 422
            assert response.status_code == 422
            assert "Unsupported file type" in response.json().get("detail", "")

    def test_chat_retrieval_error_returns_502(self) -> None:
        """Chat endpoint returns 502 when retrieval fails."""
        client = TestClient(app)

        request = {"query": "Test?", "top_k": 5}

        with patch("api.routes.chat.retrieve") as mock_retrieve:
            mock_retrieve.side_effect = RuntimeError("Qdrant unreachable")

            response = client.post("/chat/", json=request)

            assert response.status_code == 502
            assert "Retrieval failed" in response.json()["detail"]

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

        # Simulate a large file (1MB of content)
        large_content = b"Content " * 131072  # ~1MB
        files = {"file": ("large.txt", large_content, "text/plain")}
        data = {"document_id": "large-doc"}

        with patch("api.routes.ingest.load_document") as mock_load, \
             patch("api.routes.ingest.chunk_text") as mock_chunk, \
             patch("api.routes.ingest.embed_chunks") as mock_embed, \
             patch("api.routes.ingest.upsert_to_store") as mock_upsert:

            mock_load.return_value = "Content " * 131072
            chunks = [
                DocumentChunk(
                    chunk_id=f"large-doc-{i}",
                    document_id="large-doc",
                    content=f"Chunk {i}"
                )
                for i in range(10)
            ]
            mock_chunk.return_value = chunks
            mock_embed.return_value = [[0.1] * 1536] * 10
            mock_upsert.return_value = 10

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

            with patch("api.routes.ingest.load_document") as mock_load, \
                 patch("api.routes.ingest.chunk_text") as mock_chunk, \
                 patch("api.routes.ingest.embed_chunks") as mock_embed, \
                 patch("api.routes.ingest.upsert_to_store") as mock_upsert:

                mock_load.return_value = "Content"
                chunk = DocumentChunk(
                    chunk_id="c-0",
                    document_id=data["document_id"],
                    content="Content"
                )
                mock_chunk.return_value = [chunk]
                mock_embed.return_value = [[0.1] * 1536]
                mock_upsert.return_value = 1

                response = client.post("/ingest/", files=files, data=data)

                assert response.status_code == 200
                # Verify load_document was called (indicating file was processed)
                mock_load.assert_called_once()


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
        """Document ID is preserved from ingest through chat."""
        client = TestClient(app)

        doc_id = "my-special-document-42"
        files = {"file": ("doc.txt", b"Content", "text/plain")}
        data = {"document_id": doc_id}

        with patch("api.routes.ingest.load_document") as mock_load, \
             patch("api.routes.ingest.chunk_text") as mock_chunk, \
             patch("api.routes.ingest.embed_chunks") as mock_embed, \
             patch("api.routes.ingest.upsert_to_store") as mock_upsert:

            mock_load.return_value = "Content"

            # Verify chunk_text receives the correct document_id
            def verify_doc_id(text, document_id, **kwargs):
                assert document_id == doc_id
                return [
                    DocumentChunk(
                        chunk_id=f"{document_id}-0",
                        document_id=document_id,
                        content="Content"
                    )
                ]

            mock_chunk.side_effect = verify_doc_id
            mock_embed.return_value = [[0.1] * 1536]
            mock_upsert.return_value = 1

            response = client.post("/ingest/", files=files, data=data)

            assert response.status_code == 200
            assert response.json()["document_id"] == doc_id
