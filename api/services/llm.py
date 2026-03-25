# api/services/llm.py
"""LLM service — builds a prompt from retrieved context and routes to the appropriate model provider."""

from __future__ import annotations

from openai import OpenAI

from shared.config import OPENAI_API_KEY, ANTHROPIC_API_KEY
from shared.models import ChatTurn, RetrievedChunk

# Models that must be routed to the Anthropic API
ANTHROPIC_MODELS: set[str] = {"claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"}

# Models served locally via Ollama's OpenAI-compatible endpoint
LLAMA_MODELS: set[str] = {"llama3.2", "llama3.1", "llama3"}


def generate_answer(
    query: str,
    context_chunks: list[RetrievedChunk],
    model: str = "gpt-4o-mini",
    chat_history: list[ChatTurn] | None = None,
) -> str:
    """Generate an answer to *query* using *context_chunks* as grounding context.

    Routes to the correct provider based on *model*:
    - ANTHROPIC_MODELS  -> Anthropic Messages API
    - LLAMA_MODELS      -> Ollama OpenAI-compatible endpoint (http://localhost:11434/v1)
    - everything else   -> OpenAI Chat Completions API

    Parameters
    ----------
    query:
        The user's original question.
    context_chunks:
        Chunks retrieved from the vector store to use as context.
    model:
        Model identifier. Controls provider routing.
    chat_history:
        Previous conversation turns, oldest-first. Roles must be "user" or "assistant".

    Returns
    -------
    str
        The LLM-generated answer string.

    Raises
    ------
    ValueError
        If an Anthropic model is requested but ANTHROPIC_API_KEY is not set.
    """
    # Build system prompt with retrieved context
    context_text = "\n\n---\n\n".join(
        f"[Source {i + 1}]\n{rc.chunk.content}"
        for i, rc in enumerate(context_chunks)
    )
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "provided context. If the answer is not in the context, say so.\n\n"
        f"CONTEXT:\n{context_text}"
    )

    # Build message list from history, filtering to valid roles only
    history = chat_history or []
    messages: list[dict[str, str]] = [
        {"role": turn.role, "content": turn.content}
        for turn in history
        if turn.role in ("user", "assistant")
    ]

    # Append the current query if it is not already the last message
    if not messages or messages[-1]["content"] != query:
        messages.append({"role": "user", "content": query})

    if model in ANTHROPIC_MODELS:
        if not ANTHROPIC_API_KEY:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set — required for Anthropic models"
            )
        import anthropic

        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        return response.content[0].text

    elif model in LLAMA_MODELS:
        client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
        )
        return response.choices[0].message.content or ""

    else:  # OpenAI (default path)
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt}] + messages,
        )
        return response.choices[0].message.content or ""
