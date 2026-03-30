# tests/unit/test_ollama_models_endpoint.py
"""Unit tests for the GET /models/ollama endpoint."""

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestOllamaModelsEndpoint:
    """Tests for GET /models/ollama."""

    def test_returns_model_list_on_success(self) -> None:
        """Returns parsed model names when Ollama responds correctly."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "llama3.2:latest"},
                {"name": "nomic-embed-text:latest"},
            ]
        }

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

        with patch("api.routes.models.httpx.AsyncClient", return_value=mock_client):
            response = client.get("/models/ollama")

        assert response.status_code == 200
        data = response.json()
        assert data["models"] == ["llama3.2:latest", "nomic-embed-text:latest"]

    def test_returns_empty_list_when_ollama_unreachable(self) -> None:
        """Returns empty models list with error detail when Ollama is down."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get = AsyncMock(
            side_effect=Exception("Connection refused")
        )

        with patch("api.routes.models.httpx.AsyncClient", return_value=mock_client):
            response = client.get("/models/ollama")

        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []
        assert "error" in data

    def test_returns_empty_list_when_no_models_downloaded(self) -> None:
        """Returns empty list when Ollama is running but has no models."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"models": []}

        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get = AsyncMock(return_value=mock_response)

        with patch("api.routes.models.httpx.AsyncClient", return_value=mock_client):
            response = client.get("/models/ollama")

        assert response.status_code == 200
        assert response.json() == {"models": []}

    def test_error_message_included_on_failure(self) -> None:
        """The error field contains the exception message."""
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value.get = AsyncMock(
            side_effect=Exception("timeout")
        )

        with patch("api.routes.models.httpx.AsyncClient", return_value=mock_client):
            response = client.get("/models/ollama")

        assert "timeout" in response.json()["error"]
