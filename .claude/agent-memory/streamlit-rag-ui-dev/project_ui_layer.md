---
name: Streamlit UI layer — component map and session state inventory
description: Full map of app/main.py components, session state keys, API contract, and disk persistence patterns
type: project
---

## Component functions in app/main.py

| Function | Responsibility |
|---|---|
| `init_session_state()` | Guards all session state keys with `if key not in st.session_state` |
| `render_sidebar()` | Provider selector, model selector (filtered to provider), New Chat button, saved-chat list buttons |
| `render_chat()` | Displays message history; owns the `st.chat_input` widget |
| `render_source_panel(sources)` | Expandable retrieved-chunk display, called after a successful API response |
| `_start_new_chat()` | Thin state mutation — sets new UUID chat_id, clears messages |
| `_load_chat(chat_id)` | Thin state mutation — copies saved chat into active session |
| `_handle_user_input(user_input)` | Only non-trivial logic path: appends user turn, calls API, appends assistant turn, saves to disk, renders sources; shows `st.caption` with "Provider · model" format if response_model present |
| `call_chat_api(query, top_k, model, chat_history)` | Pure httpx POST to /chat/ — returns (answer, sources, response_model) tuple |
| `save_chat(chat_id, title, messages)` | Writes chats/<id>.json and updates st.session_state.chats |
| `load_all_chats()` | Reads all *.json files from chats/ on startup |
| `derive_title(first_user_message)` | Truncates to 40 chars for chat title |
| `main()` | Page config, init_session_state, render_sidebar, render_chat |

## Session state key inventory

| Key | Type | Purpose |
|---|---|---|
| `chat_id` | `str \| None` | UUID of the active chat session |
| `messages` | `list[dict]` | `[{role, content}, ...]` for the active chat |
| `selected_provider` | `str` | Chosen provider name — key into `AVAILABLE_MODELS` dict; defaults to `"OpenAI"` |
| `selected_model` | `str` | Chosen model name within the selected provider; sent to API as `model` field |
| `chats` | `dict[str, dict]` | `{id: {id, title, messages, created_at, updated_at}}` loaded from disk |

## AVAILABLE_MODELS structure

```python
AVAILABLE_MODELS: dict[str, list[str]] = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "Anthropic": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"],
    "Llama (local)": ["llama3.2", "llama3.1", "llama3"],
}
```

When the provider selectbox changes, `selected_model` is reset to the first model in the new provider's list.

## API contract (matches shared/models.py exactly)

- Endpoint: `POST {API_URL}/chat/`
- Request body fields: `query` (str), `top_k` (int, default 5), `model` (str), `chat_history` (list of ChatTurn dicts)
- ChatTurn shape: `{role: "user"|"assistant", content: str}`
- Response body fields: `answer` (str), `sources` (list of RetrievedChunk dicts), `model` (str | None)
- RetrievedChunk shape: `{chunk: {chunk_id, document_id, content, metadata}, score: float}`

### call_chat_api signature
```python
def call_chat_api(
    query: str,
    top_k: int = 5,
    model: str = "gpt-4o-mini",
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, Any]], str | None]:
```
- `model` is sourced from `st.session_state.selected_model` at the call site (just the model string, no provider prefix)
- `chat_history` is the full `st.session_state.messages` list at the time of the call (includes the just-appended user turn)
- Third return value is `response_model` (str | None), shown via `st.caption` as `"Provider · model"` inside the assistant chat bubble

## Disk persistence

- Chat files live in `chats/` at the project root (`Path(__file__).parent.parent / "chats"`)
- Each file: `chats/<uuid>.json` with keys: id, title, messages, created_at, updated_at
- Directory is created on import (`CHATS_DIR.mkdir(exist_ok=True)`)

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `API_URL` | `http://localhost:8000` | FastAPI backend base URL |
| `OPENAI_API_KEY` | (required by API layer) | Not read directly in app/main.py |

## Dependencies added

```
uv add python-dotenv httpx
```

**Why:** python-dotenv for .env loading; httpx for synchronous HTTP calls to the FastAPI backend (chosen over requests for modern API and timeout ergonomics).

**How to apply:** Any future network calls in app/main.py should use httpx. Do not add requests as a duplicate.
