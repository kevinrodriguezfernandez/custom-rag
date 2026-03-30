# tests/integration/test_chat_endpoint.py
"""Integration tests for the /chat endpoint."""

from unittest.mock import patch


from shared.models import RetrievedChunk


class TestChatEndpointBasic:
    """Tests for basic chat endpoint functionality."""

    def test_chat_endpoint_returns_200(self, client, sample_chat_request) -> None:
        """POST /chat/ returns 200 OK."""
        with patch("api.routes.chat.retrieve") as mock_retrieve, patch(
            "api.routes.chat.generate_answer"
        ) as mock_generate:
            mock_retrieve.return_value = []
            mock_generate.return_value = "Test response"

            response = client.post(
                "/chat/",
                json=sample_chat_request.model_dump(),
            )

            assert response.status_code == 200

    def test_chat_endpoint_returns_chat_response(
        self, client, sample_chat_request
    ) -> None:
        """POST /chat/ returns a ChatResponse with answer and sources."""
        with patch("api.routes.chat.retrieve") as mock_retrieve, patch(
            "api.routes.chat.generate_answer"
        ) as mock_generate:
            mock_retrieve.return_value = []
            mock_generate.return_value = "Generated answer"

            response = client.post(
                "/chat/",
                json=sample_chat_request.model_dump(),
            )

            data = response.json()
            assert "answer" in data
            assert "sources" in data
            assert data["answer"] == "Generated answer"

    def test_chat_endpoint_echoes_model(self, client) -> None:
        """POST /chat/ echoes back the model name in the response."""
        request = {
            "query": "What is AI?",
            "top_k": 5,
            "model": "claude-opus-4-1",
            "chat_history": [],
        }

        with patch("api.routes.chat.retrieve") as mock_retrieve, patch(
            "api.routes.chat.generate_answer"
        ) as mock_generate:
            mock_retrieve.return_value = []
            mock_generate.return_value = "Claude response"

            response = client.post("/chat/", json=request)

            data = response.json()
            assert data["model"] == "claude-opus-4-1"

    def test_chat_endpoint_includes_sources(self, client, sample_chunks) -> None:
        """POST /chat/ includes retrieved source chunks in response."""
        request = {
            "query": "What is machine learning?",
            "top_k": 3,
            "model": "gpt-4o-mini",
        }

        retrieved = [
            RetrievedChunk(chunk=chunk, score=0.9) for chunk in sample_chunks
        ]

        with patch("api.routes.chat.retrieve") as mock_retrieve, patch(
            "api.routes.chat.generate_answer"
        ) as mock_generate:
            mock_retrieve.return_value = retrieved
            mock_generate.return_value = "ML answer"

            response = client.post("/chat/", json=request)

            data = response.json()
            assert len(data["sources"]) == len(sample_chunks)
            assert data["sources"][0]["score"] == 0.9


class TestChatEndpointValidation:
    """Tests for request validation."""

    def test_chat_endpoint_missing_query(self, client) -> None:
        """POST /chat/ without query returns 422."""
        request = {"top_k": 5, "model": "gpt-4o-mini"}

        response = client.post("/chat/", json=request)

        assert response.status_code == 422

    def test_chat_endpoint_empty_query(self, client) -> None:
        """POST /chat/ with empty query returns 422."""
        request = {"query": "", "top_k": 5}

        response = client.post("/chat/", json=request)

        assert response.status_code == 422

    def test_chat_endpoint_top_k_out_of_range(self, client) -> None:
        """POST /chat/ with invalid top_k returns 422."""
        # top_k too high (max 20)
        request = {"query": "Test?", "top_k": 25}

        response = client.post("/chat/", json=request)

        assert response.status_code == 422

    def test_chat_endpoint_top_k_zero(self, client) -> None:
        """POST /chat/ with top_k=0 returns 422."""
        request = {"query": "Test?", "top_k": 0}

        response = client.post("/chat/", json=request)

        assert response.status_code == 422

    def test_chat_endpoint_invalid_json(self, client) -> None:
        """POST /chat/ with invalid JSON returns 422."""
        response = client.post("/chat/", content="not json")

        assert response.status_code == 422


class TestChatEndpointErrorHandling:
    """Tests for error handling in chat endpoint."""

    def test_chat_endpoint_retrieval_failure_continues_without_context(self, client) -> None:
        """POST /chat/ continues with empty chunks if retrieval fails."""
        request = {"query": "Test?", "top_k": 5}

        with patch("api.routes.chat.retrieve") as mock_retrieve, patch(
            "api.routes.chat.generate_answer"
        ) as mock_generate:
            mock_retrieve.side_effect = Exception("Qdrant unavailable")
            mock_generate.return_value = "Answer without context"

            response = client.post("/chat/", json=request)

            assert response.status_code == 200
            data = response.json()
            assert data["sources"] == []

    def test_chat_endpoint_llm_failure_returns_502(self, client) -> None:
        """POST /chat/ returns 502 if LLM generation fails."""
        request = {"query": "Test?", "top_k": 5}

        with patch("api.routes.chat.retrieve") as mock_retrieve, patch(
            "api.routes.chat.generate_answer"
        ) as mock_generate:
            mock_retrieve.return_value = []
            mock_generate.side_effect = Exception("OpenAI API error")

            response = client.post("/chat/", json=request)

            assert response.status_code == 502
            data = response.json()
            assert "LLM generation failed" in data.get("detail", "")

    def test_chat_endpoint_empty_retrieval_result(self, client) -> None:
        """POST /chat/ works even with no relevant chunks retrieved."""
        request = {"query": "Obscure question?", "top_k": 5}

        with patch("api.routes.chat.retrieve") as mock_retrieve, patch(
            "api.routes.chat.generate_answer"
        ) as mock_generate:
            mock_retrieve.return_value = []  # No chunks found
            mock_generate.return_value = "Sorry, I couldn't find relevant context."

            response = client.post("/chat/", json=request)

            assert response.status_code == 200
            data = response.json()
            assert len(data["sources"]) == 0


class TestChatEndpointWithHistory:
    """Tests for chat endpoint with conversation history."""

    def test_chat_endpoint_accepts_chat_history(
        self, client, sample_chat_request_with_history
    ) -> None:
        """POST /chat/ accepts and processes chat_history."""
        with patch("api.routes.chat.retrieve") as mock_retrieve, patch(
            "api.routes.chat.generate_answer"
        ) as mock_generate:
            mock_retrieve.return_value = []
            mock_generate.return_value = "Response based on history"

            response = client.post(
                "/chat/",
                json=sample_chat_request_with_history.model_dump(),
            )

            assert response.status_code == 200
            # Verify generate_answer was called with the history
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            assert call_args.kwargs.get("chat_history") is not None

    def test_chat_endpoint_empty_history(self, client) -> None:
        """POST /chat/ works with empty chat history."""
        request = {
            "query": "First question?",
            "chat_history": [],
            "model": "gpt-4o-mini",
        }

        with patch("api.routes.chat.retrieve") as mock_retrieve, patch(
            "api.routes.chat.generate_answer"
        ) as mock_generate:
            mock_retrieve.return_value = []
            mock_generate.return_value = "First response"

            response = client.post("/chat/", json=request)

            assert response.status_code == 200


class TestChatEndpointIntegration:
    """Integration tests with actual component interactions."""

    def test_chat_endpoint_passes_query_to_retriever(self, client) -> None:
        """POST /chat/ passes the query to the retrieve function."""
        request = {"query": "Specific question about X?", "top_k": 5}

        with patch("api.routes.chat.retrieve") as mock_retrieve, patch(
            "api.routes.chat.generate_answer"
        ) as mock_generate:
            mock_retrieve.return_value = []
            mock_generate.return_value = "Answer"

            client.post("/chat/", json=request)

            # Verify retrieve was called with correct query
            mock_retrieve.assert_called_once()
            call_args = mock_retrieve.call_args
            assert call_args.args[0] == "Specific question about X?"

    def test_chat_endpoint_passes_top_k_to_retriever(self, client) -> None:
        """POST /chat/ passes top_k to the retrieve function."""
        request = {"query": "Test?", "top_k": 10}

        with patch("api.routes.chat.retrieve") as mock_retrieve, patch(
            "api.routes.chat.generate_answer"
        ) as mock_generate:
            mock_retrieve.return_value = []
            mock_generate.return_value = "Answer"

            client.post("/chat/", json=request)

            # Verify retrieve was called with correct top_k
            call_args = mock_retrieve.call_args
            assert call_args.args[1] == 10

    def test_chat_endpoint_passes_chunks_to_llm(self, client, sample_chunks) -> None:
        """POST /chat/ passes retrieved chunks to generate_answer."""
        request = {"query": "Test?", "top_k": 3}

        retrieved = [
            RetrievedChunk(chunk=chunk, score=0.9) for chunk in sample_chunks
        ]

        with patch("api.routes.chat.retrieve") as mock_retrieve, patch(
            "api.routes.chat.generate_answer"
        ) as mock_generate:
            mock_retrieve.return_value = retrieved
            mock_generate.return_value = "Answer"

            client.post("/chat/", json=request)

            # Verify generate_answer received the chunks
            mock_generate.assert_called_once()
            call_args = mock_generate.call_args
            assert call_args.kwargs.get("context_chunks") == retrieved

    def test_chat_endpoint_passes_model_to_llm(self, client) -> None:
        """POST /chat/ passes the model name to generate_answer."""
        request = {
            "query": "Test?",
            "model": "claude-opus-4-1",
        }

        with patch("api.routes.chat.retrieve") as mock_retrieve, patch(
            "api.routes.chat.generate_answer"
        ) as mock_generate:
            mock_retrieve.return_value = []
            mock_generate.return_value = "Claude answer"

            client.post("/chat/", json=request)

            call_args = mock_generate.call_args
            assert call_args.kwargs.get("model") == "claude-opus-4-1"
