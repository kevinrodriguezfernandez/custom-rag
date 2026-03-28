# api/services/llm.py
"""LLM service — builds a prompt from retrieved context and calls the chat model."""

import asyncio
import logging

import anthropic
import openai

from shared.config import ANTHROPIC_API_KEY, OLLAMA_URL, OPENAI_API_KEY
from shared.models import ChatTurn, RetrievedChunk

logger = logging.getLogger(__name__)

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
    """Construct the system prompt, injecting retrieved context when available."""
    if not context_chunks:
        return (
            "You are a helpful, knowledgeable assistant. "
            "Answer the user's question thoroughly and accurately using your own knowledge."
        )

    sections = [f"[{i}] {rc.chunk.content}" for i, rc in enumerate(context_chunks, start=1)]
    context_text = "\n\n".join(sections)

    return (
        "You are a helpful, knowledgeable assistant. "
        "Use the document context below to ground your answer with specific information from the user's documents. "
        "You may also draw on your own knowledge to provide complete, accurate responses. "
        "Always prioritise information from the context when it is relevant.\n\n"
        f"Document context:\n{context_text}"
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
        The LLM-generated answer.
    """
    system_prompt = _build_system_prompt(context_chunks)
    logger.debug("Prompt built — system_prompt_length=%d", len(system_prompt))
    loop = asyncio.get_running_loop()

    # --- Provider routing ---
    if model.startswith("claude-"):
        effective_key = api_key or ANTHROPIC_API_KEY
        if not effective_key:
            logger.warning("Anthropic API key is empty — request will likely fail")
        logger.info("Provider selected — provider=anthropic model=%s", model)
        return await loop.run_in_executor(
            None,
            _call_anthropic,
            query,
            system_prompt,
            chat_history,
            model,
            effective_key,
        )

    is_known_ollama = model in _KNOWN_OLLAMA_MODELS
    is_not_openai = not model.startswith("gpt-")
    use_ollama = is_known_ollama or (OLLAMA_URL and is_not_openai)

    if use_ollama:
        logger.info("Provider selected — provider=ollama model=%s", model)
        effective_url = f"{api_url.rstrip('/')}/v1" if api_url else f"{OLLAMA_URL}/v1"
        return await loop.run_in_executor(
            None,
            _call_openai_compat,
            query,
            system_prompt,
            chat_history,
            model,
            effective_url,
            "ollama",
        )

    # Default: OpenAI
    effective_key = api_key or OPENAI_API_KEY
    if not effective_key:
        logger.warning("OpenAI API key is empty — request will likely fail")
    logger.info("Provider selected — provider=openai model=%s", model)
    return await loop.run_in_executor(
        None,
        _call_openai_compat,
        query,
        system_prompt,
        chat_history,
        model,
        None,
        effective_key,
    )


def _call_openai_compat(
    query: str,
    system_prompt: str,
    chat_history: list[ChatTurn] | None,
    model: str,
    base_url: str | None,
    api_key: str,
) -> str:
    """Call an OpenAI-compatible endpoint (OpenAI or Ollama)."""
    client = openai.OpenAI(api_key=api_key, base_url=base_url)
    messages = _build_messages(query, system_prompt, chat_history)
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def _call_anthropic(
    query: str,
    system_prompt: str,
    chat_history: list[ChatTurn] | None,
    model: str,
    api_key: str = "",
) -> str:
    """Call the Anthropic Messages API."""
    client = anthropic.Anthropic(api_key=api_key)

    # Anthropic uses a separate system parameter, not a system message in the list
    messages: list[dict] = []
    for turn in chat_history or []:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": query})

    response = client.messages.create(
        model=model,
        max_tokens=2048,
        system=system_prompt,
        messages=messages,
    )
    return response.content[0].text
