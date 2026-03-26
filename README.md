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
│   └── integration/
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
| Embeddings | OpenAI `text-embedding-3-small` (default) or Ollama `nomic-embed-text` |
| LLM | OpenAI, Anthropic (Claude), or Ollama — selectable per chat in the UI |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| Python | 3.12+ |

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)
- Docker + Docker Compose (for the full stack)
- An API key for whichever LLM provider you want to use

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

Edit `.env` and set the key(s) for the provider(s) you want to use:

```env
OPENAI_API_KEY=sk-...        # for OpenAI models
ANTHROPIC_API_KEY=sk-ant-... # for Claude models
```

Full list of supported variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for OpenAI models |
| `ANTHROPIC_API_KEY` | — | Required for Anthropic (Claude) models |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model when using OpenAI |
| `EMBEDDING_PROVIDER` | `openai` | Embedding backend: `openai` or `ollama` |
| `OLLAMA_EMBEDDING_MODEL` | `nomic-embed-text` | Embedding model when using Ollama |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `documents` | Qdrant collection name |
| `API_URL` | `http://localhost:8000` | URL the Streamlit app uses to reach the API |

## LLM providers

The provider and model are selected per chat in the Streamlit sidebar. The API key you enter in the sidebar is sent directly to the backend for that request.

| Provider | Available models |
|----------|-----------------|
| **OpenAI** | `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.2`, `gpt-4o` |
| **Anthropic** | `claude-opus-4-6`, `claude-sonnet-4-6`, `claude-haiku-4-5` |
| **Ollama (local)** | `minimax-m2.7:cloud`, `llama3.2`, `llama3.1`, `llama3`, `mistral`, `gemma2`, `phi3`, `codellama` |

For Ollama, the sidebar asks for the Ollama server URL instead of an API key (defaults to `http://host.docker.internal:11434` when running via Docker Compose).

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

**Supported file types:** `.pdf`, `.txt`, `.md`, `.docx`, `.xlsx`, `.xls`

**Ingest a document**

```bash
curl -X POST http://localhost:8000/ingest/ \
  -F "file=@/path/to/document.pdf" \
  -F "document_id=my-doc-001" \
  -F "chat_id=optional-session-id"
```

**Chat**

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What does the document say about X?",
    "top_k": 5,
    "model": "gpt-4o-mini",
    "chat_id": "optional-session-id",
    "chat_history": []
  }'
```

Each chat session has its own document scope — documents ingested with a given `chat_id` are only retrieved during queries that use the same `chat_id`.

Interactive API docs are available at `http://localhost:8000/docs`.

## Tests

```bash
# All tests
uv run pytest

# With coverage report
uv run pytest --cov=api --cov=ingestion --cov=shared --cov-report=term-missing

# Unit tests only
uv run pytest tests/unit/

# With verbose output
uv run pytest -v
```

## GitHub Actions

| Workflow | Trigger | What it does |
|----------|---------|--------------|
| **Claude Code Review** | Every PR | Runs pytest + coverage, flake8, pylint, then posts an automated Claude code review as a PR comment |
| **Claude Code** | `@claude` mention in issue or PR comment | Claude responds inline to the comment |
