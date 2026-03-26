# api/services/llm.py
"""LLM service — builds a prompt from retrieved context and routes to the appropriate model provider."""

import asyncio

import anthropic
import openai

from shared.config import ANTHROPIC_API_KEY, OLLAMA_URL, OPENAI_API_KEY
from shared.models import ChatTurn, RetrievedChunk

# Models that are served locally via Ollama
_KNOWN_OLLAMA_MODELS = {
    "llama3",
    "llama3.1",
    "llama3.2",
    "mistral",
    "gemma",
    "gemma2",
    "phi3",
    "phi3.5",
    "qwen2",
    "deepseek-r1",
    "codellama",
    "mixtral",
    "neural-chat",
    "starling-lm",
    "solar",
}


def _build_system_prompt(context_chunks: list[RetrievedChunk]) -> str:
    """Construct the system prompt that injects retrieved context."""
    if not context_chunks:
        context_text = "No relevant context was found in the knowledge base."
    else:
        sections = []
        for i, rc in enumerate(context_chunks, start=1):
            sections.append(f"[{i}] {rc.chunk.content}")
        context_text = "\n\n".join(sections)

    return (
        "You are a helpful assistant. Answer the user's question using only the "
        "context below. If the context does not contain enough information to answer, "
        "say so clearly.\n\n"
        f"Context:\n{context_text}"
    )


def _build_messages(
    query: str,
    system_prompt: str,
    chat_history: list[ChatTurn] | None,
) -> list[dict]:
    """Assemble the message list for the chat completion call."""
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for turn in chat_history or []:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": query})
    return messages


async def generate_answer(
    query: str,
    context_chunks: list[RetrievedChunk],
    model: str = "gpt-4o-mini",
    chat_history: list[ChatTurn] | None = None,
    api_key: str | None = None,
    api_url: str | None = None,
) -> str:
    """Generate an answer to *query* using the retrieved *context_chunks*.

    Routes to the appropriate LLM provider based on the model name:
    - ``claude-*`` models go to the Anthropic API.
    - Known Ollama models (or any model not prefixed with ``gpt-`` / ``claude-``)
      go to the local Ollama OpenAI-compatible endpoint.
    - Everything else goes to OpenAI.

    Parameters
    ----------
    query:
        The user's original question.
    context_chunks:
        Chunks retrieved from the vector store to use as context.
    model:
        The model identifier to use for generation.
    chat_history:
        Optional prior conversation turns (oldest-first).

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
