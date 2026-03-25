---
name: shared/config.py conventions
description: How configuration is loaded and which keys are required vs optional
type: project
---

All env vars are read exclusively in shared/config.py (python-dotenv + os.getenv). No other file may hardcode or re-read env vars.

Required (raises ValueError at import time if missing):
- OPENAI_API_KEY

Optional (safe defaults provided):
- OPENAI_EMBEDDING_MODEL — default "text-embedding-3-small"
- OPENAI_CHAT_MODEL — default "gpt-4o"
- ANTHROPIC_API_KEY — default "" (absence guarded at call site in llm.py, not at import)
- QDRANT_URL — default "http://localhost:6333"
- QDRANT_COLLECTION — default "documents"
- API_HOST — default "0.0.0.0"
- API_PORT — default 8000
- API_URL — default "http://localhost:8000"

**Why:** Centralising config in one module prevents scattered os.getenv calls and makes it easy to audit what the app depends on.
**How to apply:** Always import from shared.config, never from os.environ directly. If a new env var is needed, add it to shared/config.py first.
