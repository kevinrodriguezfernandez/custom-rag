---
name: RAG project stack and configuration
description: LLM providers, embedding model, vector store config, and key architectural decisions made during initial implementation
type: project
---

LLM provider routing in api/services/llm.py:
- `claude-*` models → Anthropic API (anthropic>=0.40.0)
- Known Ollama models or non-gpt-/non-claude- prefixed models → Ollama at OLLAMA_URL/v1 with api_key="ollama"
- All others (default) → OpenAI

**Why:** Multi-provider support was built in from the start. Ollama enables local/offline inference; Anthropic gives Claude access; OpenAI is the default.

**How to apply:** When adding new providers or model routing, edit the `_KNOWN_OLLAMA_MODELS` set and the routing logic in `generate_answer()` in api/services/llm.py.

Embedding model: `text-embedding-3-small` (1536 dimensions, configured in shared/config.py as OPENAI_EMBEDDING_MODEL).

Vector store: Qdrant, collection name from QDRANT_COLLECTION env var (default: "documents"). Collection is auto-created on first upsert with Cosine distance.

Point IDs in Qdrant are generated as `abs(hash(chunk.chunk_id)) % (2**63)` — integer hashes of the chunk_id string.

Session memory strategy: stateless — chat_history is passed per-request as a list[ChatTurn], no server-side session storage.

CORS: allow_origins=["*"] in api/main.py (all origins, local dev only — tighten for production).

Dependencies added beyond initial pyproject.toml:
- anthropic>=0.40.0
- langchain-community>=0.3.0
- pypdf>=5.0.0
