---
name: Async offloading for synchronous services
description: All service calls (retriever, llm, embedder, loader) are synchronous and must be wrapped in asyncio.to_thread() inside async route handlers
type: feedback
---

All functions in `api/services/retriever.py`, `api/services/llm.py`, `ingestion/embedder.py`, and `ingestion/loader.py` are synchronous (they use blocking OpenAI and Qdrant clients). Route handlers are async.

**Rule:** Every call to these services from an async route handler must be wrapped with `await asyncio.to_thread(fn, *args)` — never called directly.

**Why:** Calling a blocking function directly inside an async handler blocks the entire event loop, preventing FastAPI from handling other requests concurrently.

**How to apply:** Any time a new route or service is added that calls into ingestion or api/services functions, check whether the callee is synchronous. If so, use `asyncio.to_thread`. This applies even to `chunk_text` (CPU-bound) for large document safety.
