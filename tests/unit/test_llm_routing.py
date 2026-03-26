# tests/unit/test_llm_routing.py
"""Unit tests for LLM provider routing logic."""

from unittest.mock import MagicMock, patch, AsyncMock
import asyncio

import pytest

from api.services.llm import generate_answer, _build_system_prompt, _build_messages
from shared.models import ChatTurn, DocumentChunk, RetrievedChunk


class TestBuildSystemPrompt:
    """Tests for the system prompt builder."""

    def test_build_system_prompt_with_context(self, sample_chunks) -> None:
        """System prompt is built correctly with context chunks."""
        retrieved = [
            RetrievedChunk(chunk=chunk, score=0.95) for chunk in sample_chunks
        ]

        prompt = _build_system_prompt(retrieved)

        assert "You are a helpful assistant" in prompt
        assert "context" in prompt.lower()
        # Check that chunk content is included
        for chunk in sample_chunks:
            assert chunk.content in prompt

    def test_build_system_prompt_without_context(self) -> None:
        """System prompt handles empty context gracefully."""
        prompt = _build_system_prompt([])

        assert "You are a helpful assistant" in prompt
        assert "No relevant context" in prompt

    def test_build_system_prompt_multiple_chunks(self, sample_chunks) -> None:
        """System prompt includes all chunks with numbered sections."""
        retrieved = [
            RetrievedChunk(chunk=chunk, score=0.95) for chunk in sample_chunks
        ]

        prompt = _build_system_prompt(retrieved)

        # Check for numbered sections
        assert "[1]" in prompt
        assert "[2]" in prompt
        assert "[3]" in prompt

    def test_build_system_prompt_chunk_order(self) -> None:
        """System prompt maintains chunk order."""
        chunk1 = DocumentChunk(
            chunk_id="c1", document_id="d1", content="First content"
        )
        chunk2 = DocumentChunk(
            chunk_id="c2", document_id="d1", content="Second content"
        )
        chunk3 = DocumentChunk(
            chunk_id="c3", document_id="d1", content="Third content"
        )

        retrieved = [
            RetrievedChunk(chunk=chunk1, score=0.9),
            RetrievedChunk(chunk=chunk2, score=0.8),
            RetrievedChunk(chunk=chunk3, score=0.7),
        ]

        prompt = _build_system_prompt(retrieved)

        # Verify order is preserved
        pos_first = prompt.find("First content")
        pos_second = prompt.find("Second content")
        pos_third = prompt.find("Third content")

        assert pos_first < pos_second < pos_third


class TestBuildMessages:
    """Tests for the message list builder."""

    def test_build_messages_basic(self) -> None:
        """Build messages with query and system prompt."""
        system_prompt = "You are helpful."
        query = "What is AI?"
        messages = _build_messages(query, system_prompt, None)

        assert len(messages) >= 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == system_prompt
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == query

    def test_build_messages_with_chat_history(self) -> None:
        """Build messages includes chat history in correct order."""
        system_prompt = "You are helpful."
        query = "Tell me more."
        history = [
            ChatTurn(role="user", content="What is AI?"),
            ChatTurn(role="assistant", content="AI is..."),
        ]

        messages = _build_messages(query, system_prompt, history)

        assert len(messages) == 4  # system, user, assistant, user
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert messages[1]["content"] == "What is AI?"
        assert messages[2]["role"] == "assistant"
        assert messages[2]["content"] == "AI is..."
        assert messages[3]["role"] == "user"
        assert messages[3]["content"] == query

    def test_build_messages_empty_history(self) -> None:
        """Build messages with empty history list."""
        messages = _build_messages("Query?", "System.", [])

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_build_messages_none_history(self) -> None:
        """Build messages with None history."""
        messages = _build_messages("Query?", "System.", None)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_build_messages_long_history(self) -> None:
        """Build messages handles longer conversation history."""
        history = [
            ChatTurn(role="user", content=f"Question {i}?")
            if i % 2 == 0
            else ChatTurn(role="assistant", content=f"Answer {i}.")
            for i in range(10)
        ]

        messages = _build_messages("Final query?", "System.", history)

        assert len(messages) == 12  # system + 10 history + 1 current user
        # Verify history order is preserved
        for i in range(10):
            assert messages[i + 1]["content"] in [
                f"Question {i}?",
                f"Answer {i}.",
            ]


class TestGenerateAnswerRouting:
    """Tests for LLM provider routing in generate_answer."""

    @pytest.mark.asyncio
    async def test_route_to_anthropic_for_claude_models(self) -> None:
        """claude-* models are routed to Anthropic."""
        with patch(
            "api.services.llm._call_anthropic"
        ) as mock_anthropic, patch(
            "api.services.llm._call_openai_compat"
        ) as mock_openai:
            mock_anthropic.return_value = "Claude response"

            result = await generate_answer(
                query="Test?",
                context_chunks=[],
                model="claude-3-sonnet-20240229",
            )

            assert result == "Claude response"
            # Verify Anthropic was called
            mock_anthropic.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_to_openai_for_gpt_models(self) -> None:
        """gpt-* models are routed to OpenAI."""
        with patch(
            "api.services.llm._call_openai_compat"
        ) as mock_openai:
            mock_openai.return_value = "GPT response"

            result = await generate_answer(
                query="Test?", context_chunks=[], model="gpt-4o-mini"
            )

            assert result == "GPT response"

    @pytest.mark.asyncio
    async def test_route_to_ollama_for_known_models(self) -> None:
        """Known Ollama models are routed to Ollama endpoint."""
        with patch(
            "api.services.llm._call_openai_compat"
        ) as mock_openai:
            mock_openai.return_value = "Ollama response"

            result = await generate_answer(
                query="Test?", context_chunks=[], model="llama3"
            )

            assert result == "Ollama response"

    @pytest.mark.asyncio
    async def test_claude_opus_routing(self) -> None:
        """claude-opus is correctly routed to Anthropic."""
        with patch(
            "api.services.llm._call_anthropic"
        ) as mock_anthropic:
            mock_anthropic.return_value = "Opus response"

            result = await generate_answer(
                query="Test?", context_chunks=[], model="claude-opus-4-1"
            )

            assert result == "Opus response"

    @pytest.mark.asyncio
    async def test_gpt_4_routing(self) -> None:
        """gpt-4 is correctly routed to OpenAI."""
        with patch(
            "api.services.llm._call_openai_compat"
        ) as mock_openai:
            mock_openai.return_value = "GPT-4 response"

            result = await generate_answer(
                query="Test?", context_chunks=[], model="gpt-4"
            )

            assert result == "GPT-4 response"

    @pytest.mark.asyncio
    async def test_unknown_model_defaults_to_openai(self) -> None:
        """Unknown models (non-claude, non-gpt, non-ollama) default to OpenAI."""
        with patch(
            "api.services.llm._call_openai_compat"
        ) as mock_openai:
            mock_openai.return_value = "OpenAI fallback"

            result = await generate_answer(
                query="Test?", context_chunks=[], model="unknown-model-xyz"
            )

            assert result == "OpenAI fallback"

    @pytest.mark.asyncio
    async def test_generate_answer_with_context_chunks(self) -> None:
        """generate_answer includes context in the system prompt."""
        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id="c1",
                    document_id="d1",
                    content="Test context",
                ),
                score=0.9,
            )
        ]

        with patch(
            "api.services.llm._call_openai_compat"
        ) as mock_openai:
            mock_openai.return_value = "Response with context"

            result = await generate_answer(
                query="Test?", context_chunks=chunks, model="gpt-4o-mini"
            )

            assert result == "Response with context"

    @pytest.mark.asyncio
    async def test_generate_answer_with_chat_history(self) -> None:
        """generate_answer passes chat history to LLM."""
        history = [ChatTurn(role="user", content="Previous question")]

        with patch(
            "api.services.llm._call_openai_compat"
        ) as mock_openai:
            mock_openai.return_value = "Response with history"

            result = await generate_answer(
                query="Follow-up?",
                context_chunks=[],
                model="gpt-4o-mini",
                chat_history=history,
            )

            assert result == "Response with history"

    @pytest.mark.asyncio
    async def test_generate_answer_default_model(self) -> None:
        """generate_answer uses default model if not specified."""
        with patch(
            "api.services.llm._call_openai_compat"
        ) as mock_openai:
            mock_openai.return_value = "Default model response"

            result = await generate_answer(
                query="Test?", context_chunks=[]
            )  # No model specified

            assert result == "Default model response"


class TestOllamaModelDetection:
    """Tests for Ollama model detection logic."""

    def test_llama3_is_detected_as_ollama(self) -> None:
        """llama3 is in the known Ollama models set."""
        from api.services.llm import _KNOWN_OLLAMA_MODELS

        assert "llama3" in _KNOWN_OLLAMA_MODELS

    def test_llama31_is_detected_as_ollama(self) -> None:
        """llama3.1 is in the known Ollama models set."""
        from api.services.llm import _KNOWN_OLLAMA_MODELS

        assert "llama3.1" in _KNOWN_OLLAMA_MODELS

    def test_mistral_is_detected_as_ollama(self) -> None:
        """mistral is in the known Ollama models set."""
        from api.services.llm import _KNOWN_OLLAMA_MODELS

        assert "mistral" in _KNOWN_OLLAMA_MODELS

    def test_phi3_is_detected_as_ollama(self) -> None:
        """phi3 is in the known Ollama models set."""
        from api.services.llm import _KNOWN_OLLAMA_MODELS

        assert "phi3" in _KNOWN_OLLAMA_MODELS
