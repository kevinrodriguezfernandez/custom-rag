# api/main.py
"""FastAPI application entry point."""

from fastapi import FastAPI

from api.routes.health import router as health_router
from api.routes.chat import router as chat_router
from api.routes.ingest import router as ingest_router

app = FastAPI(
    title="Custom RAG API",
    version="0.1.0",
    description="Retrieval-Augmented Generation backend service",
)

# --- Register routers ---
app.include_router(health_router, tags=["health"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])


if __name__ == "__main__":
    import uvicorn
    from shared.config import API_HOST, API_PORT

    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
