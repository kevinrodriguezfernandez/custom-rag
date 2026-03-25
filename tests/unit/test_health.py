# tests/unit/test_health.py
"""Tests for the /health endpoint."""


def test_health_returns_ok(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
