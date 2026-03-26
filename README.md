# custom-rag

A Retrieval-Augmented Generation (RAG) system built as a Python monorepo. Upload documents, embed them into a Qdrant vector store, and chat with your data through a Streamlit UI backed by a FastAPI service.

## Architecture

```
custom-rag/
├── api/              # FastAPI backend (chat + ingest endpoints)
│   ├── routes/       # chat.py, ingest.py, health.py
│   └── services/     # llm.py, retriever.py
├── app/              # Streamlit frontend
│   └── main.py
├── ingestion/        # Document processing pipeline
│   ├── loader.py     # Read raw files → plain text
│   ├── chunker.py    # Split text → DocumentChunk list
│   └── embedder.py   # Embed chunks + upsert to Qdrant
├── shared/
│   ├── models.py     # All Pydantic models (single source of truth)
│   └── config.py     # All env vars loaded here
├── tests/
│   ├── unit/
│   ├── integration/
│   └── contract/
└── docker-compose.yml
```

**Data flow:**

```
[User uploads file]
        ↓
  POST /ingest/   →  loader → chunker → embedder → Qdrant
        ↓
[User sends query]
        ↓
  POST /chat/     →  embed query → Qdrant search → LLM → answer + sources
        ↓
  Streamlit UI renders answer and retrieved source chunks
```

## Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit |
| Backend | FastAPI + Uvicorn |
| Vector store | Qdrant |
| Embeddings | OpenAI `text-embedding-3-small` (default) |
| LLM | OpenAI `gpt-4o` (default) — also supports Anthropic and local Llama |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Python | 3.12+ |

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker + Docker Compose (for the full stack)
- An OpenAI API key

## Setup

**1. Clone and install dependencies**

```bash
git clone <repo-url>
cd custom-rag
uv sync
```

**2. Configure environment**

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

```env
OPENAI_API_KEY=sk-...
```

The other values have sensible defaults for local development:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Model used for document and query embeddings |
| `OPENAI_CHAT_MODEL` | `gpt-4o` | Default LLM for generation |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `documents` | Qdrant collection name |
| `API_HOST` | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | `8000` | FastAPI bind port |
| `API_URL` | `http://localhost:8000` | URL the Streamlit app uses to reach the API |

## Running

### Option A — Docker Compose (recommended)

Starts Qdrant, the FastAPI API, and the Streamlit app together:

```bash
docker compose up --build
```

| Service | URL |
|---------|-----|
| Streamlit UI | http://localhost:8501 |
| FastAPI (docs) | http://localhost:8000/docs |
| Qdrant dashboard | http://localhost:6333/dashboard |

### Option B — Local (three terminals)

**Terminal 1 — Qdrant**

```bash
docker run -p 6333:6333 qdrant/qdrant
```

**Terminal 2 — FastAPI backend**

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 3 — Streamlit frontend**

```bash
uv run streamlit run app/main.py
```

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/ingest/` | Upload a document for ingestion |
| `POST` | `/chat/` | Ask a question; returns answer + source chunks |

**Ingest a document**

```bash
curl -X POST http://localhost:8000/ingest/ \
  -F "file=@/path/to/document.pdf" \
  -F "document_id=my-doc-001"
```

**Chat**

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the document say about X?",
    "top_k": 5,
    "model": "gpt-4o-mini"
  }'
```

Interactive API docs are available at `http://localhost:8000/docs`.

## Tests

```bash
# All tests
uv run pytest

# Unit tests only
uv run pytest tests/unit/

# With verbose output
uv run pytest -v
```

## Project status

The scaffold and all interfaces are in place. The following functions are stubbed with `NotImplementedError` and need implementation:

- `ingestion/loader.py` — `load_document()`
- `ingestion/chunker.py` — `chunk_text()`
- `ingestion/embedder.py` — `embed_chunks()`, `upsert_to_store()`
- `api/routes/chat.py` — `chat()`
- `api/routes/ingest.py` — `ingest()`
- `api/services/retriever.py` — retrieval logic
- `api/services/llm.py` — LLM call logic
