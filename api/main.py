# api/main.py
"""FastAPI application entry point."""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.routes.chat import router as chat_router
from api.routes.ingest import router as ingest_router
from shared.config import API_URL, EMBEDDING_PROVIDER, QDRANT_COLLECTION, QDRANT_URL

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Custom RAG API",
    version="0.1.0",
    description="Retrieval-Augmented Generation backend service",
)

# --- CORS ---
# allow_credentials=True is incompatible with allow_origins=["*"] per the CORS spec
# (browsers reject credentialed requests to wildcard origins). API keys are passed
# in the request body, not via cookies, so credentials mode is not needed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Register routers ---
app.include_router(health_router, tags=["health"])
app.include_router(chat_router, prefix="/chat", tags=["chat"])
app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])


logger.info(
    "Custom RAG API starting — API_URL=%s QDRANT_URL=%s QDRANT_COLLECTION=%s EMBEDDING_PROVIDER=%s",
    API_URL,
    QDRANT_URL,
    QDRANT_COLLECTION,
    EMBEDDING_PROVIDER,
)

if __name__ == "__main__":
    import uvicorn
    from shared.config import API_HOST, API_PORT

    uvicorn.run("api.main:app", host=API_HOST, port=API_PORT, reload=True)
