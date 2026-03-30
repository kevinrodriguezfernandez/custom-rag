# tests/integration/test_health_endpoint.py
"""Integration tests for the /health endpoint."""

from fastapi.testclient import TestClient

from api.main import app


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_returns_200(self) -> None:
        """GET /health returns 200 OK."""
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_returns_ok_status(self) -> None:
        """GET /health returns status='ok'."""
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "ok"

    def test_health_returns_version(self) -> None:
        """GET /health returns the current version."""
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "0.1.0"

    def test_health_response_schema(self) -> None:
        """GET /health returns a valid HealthResponse."""
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()

        # Verify required fields
        assert "status" in data
        assert "version" in data

        # Verify types
        assert isinstance(data["status"], str)
        assert isinstance(data["version"], str)

    def test_health_endpoint_no_dependencies(self) -> None:
        """GET /health does not require external services."""
        # The health endpoint should always return 200
        # regardless of external service availability
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
