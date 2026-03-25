# app/main.py
"""Streamlit UI entry point for the Custom RAG frontend.

Run with:
    streamlit run app/main.py

Required env vars (place in .env at project root):
    OPENAI_API_KEY   — passed through to the API layer
    API_URL          — base URL for the FastAPI backend (default: http://localhost:8000)

New dependencies (install once):
    uv add python-dotenv httpx
"""

from __future__ import annotations

import datetime
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so shared/ is importable when run via
# `streamlit run app/main.py` from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx
import streamlit as st
from dotenv import load_dotenv

from shared.config import API_URL as _API_URL

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

load_dotenv()

API_URL: str = _API_URL
CHATS_DIR: Path = Path(__file__).resolve().parent.parent / "chats"
CHATS_DIR.mkdir(exist_ok=True)

AVAILABLE_MODELS: dict[str, list[str]] = {
    "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "Anthropic": ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"],
    "Llama (local)": ["llama3.2", "llama3.1", "llama3"],
}

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------


def init_session_state() -> None:
    """Initialise all session-state keys with safe defaults.

    Keys managed:
        chat_id           (str | None)  — UUID of the active chat session
        messages          (list[dict])  — [{role, content}, ...] for the active chat
        selected_provider (str)         — chosen provider name (key in AVAILABLE_MODELS)
        selected_model    (str)         — chosen model name within the selected provider
        chats             (dict)        — {id: {title, messages, created_at}} loaded from disk
    """
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "selected_provider" not in st.session_state:
        st.session_state.selected_provider = "OpenAI"
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = AVAILABLE_MODELS["OpenAI"][0]
    if "chats" not in st.session_state:
        st.session_state.chats = load_all_chats()


# ---------------------------------------------------------------------------
# Disk persistence helpers
# ---------------------------------------------------------------------------


def chat_file(chat_id: str) -> Path:
    """Return the JSON file path for a given chat ID."""
    return CHATS_DIR / f"{chat_id}.json"


def load_all_chats() -> dict[str, Any]:
    """Load all saved chat sessions from the chats/ directory into a dict.

    Returns a dict keyed by chat ID with values {title, messages, created_at}.
    """
    chats: dict[str, Any] = {}
    for path in sorted(CHATS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            chats[data["id"]] = data
        except (json.JSONDecodeError, OSError) as exc:
            st.warning(f"Skipping malformed chat file {path}: {exc}")
    return chats


def save_chat(chat_id: str, title: str, messages: list[dict[str, str]]) -> None:
    """Persist a chat session to disk as JSON.

    Writes/overwrites chats/<chat_id>.json. Also updates st.session_state.chats
    so the sidebar reflects the change immediately.
    """
    payload = {
        "id": chat_id,
        "title": title,
        "messages": messages,
        "created_at": st.session_state.chats.get(chat_id, {}).get(
            "created_at", datetime.datetime.now(datetime.timezone.utc).isoformat()
        ),
        "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    chat_file(chat_id).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    st.session_state.chats[chat_id] = payload


def derive_title(first_user_message: str) -> str:
    """Truncate the first user message to 40 chars to use as a chat title."""
    text = first_user_message.strip()
    return text[:40] + ("..." if len(text) > 40 else "")


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------


def call_chat_api(
    query: str,
    top_k: int = 5,
    model: str = "gpt-4o-mini",
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[str, list[dict[str, Any]], str | None]:
    """POST to /chat/ and return (answer, sources, response_model).

    Uses the exact ChatRequest field names from shared/models.py:
        query        (str)
        top_k        (int, default 5)
        model        (str, default "gpt-4o-mini")
        chat_history (list[ChatTurn], default [])

    Returns:
        answer         — the LLM-generated answer string
        sources        — list of raw RetrievedChunk dicts (may be empty)
        response_model — model name echoed back from ChatResponse, or None

    Raises:
        httpx.HTTPError / httpx.ConnectError on network failure — caller handles these.
    """
    payload: dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "model": model,
        "chat_history": chat_history or [],
    }
    with httpx.Client(timeout=60.0) as client:
        response = client.post(f"{API_URL}/chat/", json=payload)
        response.raise_for_status()
    data = response.json()
    answer: str = data.get("answer", "")
    sources: list[dict[str, Any]] = data.get("sources", [])
    response_model: str | None = data.get("model")
    return answer, sources, response_model


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------


def render_sidebar() -> None:
    """Render the sidebar: model selector, new-chat button, and saved-chat list.

    Reads/writes:
        st.session_state.selected_provider
        st.session_state.selected_model
        st.session_state.chat_id
        st.session_state.messages
        st.session_state.chats
    """
    with st.sidebar:
        st.title("Custom RAG")

        # --- Provider selector ---
        st.subheader("Model")
        providers = list(AVAILABLE_MODELS.keys())
        selected_provider = st.selectbox(
            label="Provider",
            options=providers,
            index=providers.index(st.session_state.selected_provider),
            key="_provider_selector",
        )
        # Reset model to first in provider's list when provider changes
        if selected_provider != st.session_state.selected_provider:
            st.session_state.selected_provider = selected_provider
            st.session_state.selected_model = AVAILABLE_MODELS[selected_provider][0]

        # --- Model selector (filtered to selected provider) ---
        provider_models = AVAILABLE_MODELS[selected_provider]
        current_model = st.session_state.selected_model
        model_index = provider_models.index(current_model) if current_model in provider_models else 0
        selected_model = st.selectbox(
            label="Model",
            options=provider_models,
            index=model_index,
            key="_model_selector",
        )
        st.session_state.selected_provider = selected_provider
        st.session_state.selected_model = selected_model

        st.divider()

        # --- New chat button ---
        if st.button("+ New chat", use_container_width=True):
            _start_new_chat()
            st.rerun()

        st.divider()

        # --- Saved chat list ---
        st.subheader("Chat history")
        if not st.session_state.chats:
            st.caption("No saved chats yet.")
        else:
            for cid, meta in st.session_state.chats.items():
                label = meta.get("title", "Untitled")
                is_active = cid == st.session_state.chat_id
                button_label = f"**{label}**" if is_active else label
                if st.button(button_label, key=f"chat_btn_{cid}", use_container_width=True):
                    _load_chat(cid)
                    st.rerun()


def render_chat() -> None:
    """Render the main chat area: message history and the input box.

    Reads/writes:
        st.session_state.messages
        st.session_state.chat_id
        st.session_state.chats
        st.session_state.selected_model
    """
    st.header("Chat", divider="gray")

    # Display conversation history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about your documents...")
    if user_input:
        _handle_user_input(user_input)


def render_source_panel(sources: list[dict[str, Any]]) -> None:
    """Render retrieved document chunks in an expandable panel below the chat.

    Args:
        sources: list of raw RetrievedChunk dicts from the API response.

    Reads: nothing from session state (pure display).
    """
    if not sources:
        return

    st.subheader("Retrieved sources", divider="gray")
    for i, item in enumerate(sources, start=1):
        chunk = item.get("chunk", {})
        score = item.get("score", 0.0)
        content = chunk.get("content", "")
        metadata = chunk.get("metadata", {})
        doc_id = chunk.get("document_id", "unknown")

        with st.expander(f"Source {i} — {doc_id}  (score: {score:.3f})"):
            st.markdown(content)
            if metadata:
                st.caption("Metadata: " + ", ".join(f"{k}={v}" for k, v in metadata.items()))


# ---------------------------------------------------------------------------
# State mutation helpers (thin — only update session state)
# ---------------------------------------------------------------------------


def _start_new_chat() -> None:
    """Reset session state to begin a blank chat session.

    Writes:
        st.session_state.chat_id  → new UUID string
        st.session_state.messages → []
    """
    st.session_state.chat_id = str(uuid.uuid4())
    st.session_state.messages = []


def _load_chat(chat_id: str) -> None:
    """Load a saved chat from st.session_state.chats into the active session.

    Writes:
        st.session_state.chat_id
        st.session_state.messages
    """
    meta = st.session_state.chats.get(chat_id, {})
    st.session_state.chat_id = chat_id
    st.session_state.messages = list(meta.get("messages", []))


def _handle_user_input(user_input: str) -> None:
    """Process a user message: display it, call the API, display the response.

    This function contains the only business-logic path in the UI layer.
    It is called directly from render_chat() — not from a Streamlit callback —
    so it is allowed to call st.* display functions.

    Writes:
        st.session_state.messages  — appends user + assistant turns
        st.session_state.chat_id   — creates one if absent
        st.session_state.chats     — updates or creates the chat entry on disk
    """
    if not user_input.strip():
        return

    # Ensure we have an active chat ID
    if not st.session_state.chat_id:
        _start_new_chat()

    # Append and display the user message immediately
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call the API and display the assistant response
    sources: list[dict[str, Any]] = []
    response_model: str | None = None
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources, response_model = call_chat_api(
                    user_input,
                    model=st.session_state.selected_model,
                    chat_history=list(st.session_state.messages),
                )
            except httpx.ConnectError:
                st.error(
                    f"Cannot reach the API at **{API_URL}**. "
                    "Make sure the FastAPI backend is running and `API_URL` is set correctly."
                )
                # Remove the user message we just appended so history stays consistent
                st.session_state.messages.pop()
                return
            except httpx.HTTPStatusError as exc:
                st.error(
                    f"API returned an error: **{exc.response.status_code}** — "
                    f"{exc.response.text[:200]}"
                )
                st.session_state.messages.pop()
                return
            except Exception as exc:  # noqa: BLE001
                st.error(f"Unexpected error while contacting the API: {exc}")
                st.session_state.messages.pop()
                return

        st.markdown(answer)
        if response_model:
            st.caption(f"{st.session_state.selected_provider} · {response_model}")

    # Persist both turns
    st.session_state.messages.append({"role": "assistant", "content": answer})

    chat_id = st.session_state.chat_id
    assert chat_id is not None
    existing_title = st.session_state.chats.get(chat_id, {}).get("title")
    title = existing_title or derive_title(user_input)
    save_chat(chat_id, title, st.session_state.messages)

    # Render sources below the chat area if any were returned
    if sources:
        render_source_panel(sources)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Configure the page and orchestrate top-level component rendering."""
    st.set_page_config(
        page_title="Custom RAG",
        page_icon=":books:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_session_state()
    render_sidebar()

    # Ensure there is always an active chat session on first load
    if not st.session_state.chat_id:
        _start_new_chat()

    render_chat()


if __name__ == "__main__":
    main()
