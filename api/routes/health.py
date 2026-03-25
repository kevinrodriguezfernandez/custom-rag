# api/routes/health.py
"""Health-check endpoint."""

from fastapi import APIRouter

from shared.models import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return service health status."""
    return HealthResponse()
