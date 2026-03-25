---
name: Model evolution log
description: Record of models added or modified, which modules consume them, and whether changes were breaking
type: project
---

## 2026-03-25 — shared/config.py hardening (not a model change)

**What changed:**
- `OPENAI_API_KEY`: was `str | None`, now `str` with a module-level `ValueError` guard if unset.
- `API_PORT`: bare `int()` cast wrapped in `try/except ValueError` with a descriptive re-raise.
- `API_BASE_URL` renamed to `API_URL`, env var key changed from `"API_BASE_URL"` to `"API_URL"` to align with `app/main.py` which has always read `os.getenv("API_URL", ...)` directly.

**Why:** Code reviewer flagged all three. The naming mismatch meant `app/main.py` was silently ignoring `shared/config.py`'s value and reading its own env var independently.

**Impact on consumers:** `app/` — any code that imported `API_BASE_URL` from `shared.config` must be updated to `API_URL`. `app/main.py` currently does NOT import from `shared.config` at all; it reads `API_URL` directly from the environment, so it is unaffected today but can now import `API_URL` from `shared.config` to consolidate.

---

## 2026-03-25 — ChatRequest, ChatResponse, ChatTurn

**What changed:**
- `ChatRequest` gained two optional fields: `model: str` (default `"gpt-4o-mini"`) and `chat_history: list[ChatTurn]` (default empty list).
- `ChatResponse` gained one optional field: `model: str | None` (default `None`) to echo back which model was used.
- New shared model `ChatTurn` added with fields `role: Literal["user", "assistant"]` and `content: str`.

**Why `ChatTurn` is shared (not local):** It crosses module boundaries as a nested type inside `ChatRequest`, which is consumed by both `api/` and `app/`. Defining it inline as `dict[str, str]` was rejected in favor of a proper typed model per the "prefer specific types over dict" rule.

**Consumers:** `app/` (Streamlit) and `api/` (FastAPI) both consume `ChatRequest` and `ChatResponse`.

**Breaking change:** No. All new fields are optional with defaults. Existing callers are unaffected.

**Why:** `model` field needed so callers can select the OpenAI model per-request. `chat_history` needed to support multi-turn conversations passed through the RAG pipeline.
