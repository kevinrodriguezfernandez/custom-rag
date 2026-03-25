# tests/conftest.py
"""Shared pytest fixtures for the entire test suite."""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture()
def client() -> TestClient:
    """Return a synchronous FastAPI test client."""
    return TestClient(app)
