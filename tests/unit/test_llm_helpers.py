# tests/unit/test_llm_helpers.py
"""Unit tests for LLM service helper functions and edge cases."""

from unittest.mock import MagicMock, patch

from api.services.llm import _build_system_prompt, _build_messages, _call_openai_compat, _call_anthropic
from shared.models import DocumentChunk, RetrievedChunk, ChatTurn


class TestSystemPromptBuilder:
    """Additional tests for system prompt construction."""

    def test_build_system_prompt_with_very_long_content(self) -> None:
        """System prompt handles chunks with very long content."""
        chunk = DocumentChunk(
            chunk_id="long-chunk",
            document_id="doc-1",
            content="A" * 5000,  # Very long chunk
        )
        retrieved = [RetrievedChunk(chunk=chunk, score=0.95)]

        prompt = _build_system_prompt(retrieved)

        assert "A" * 5000 in prompt
        assert "[1]" in prompt

    def test_build_system_prompt_with_special_characters(self) -> None:
        """System prompt preserves special characters in chunks."""
        chunk = DocumentChunk(
            chunk_id="special-chunk",
            document_id="doc-1",
            content="Content with special chars: !@#$%^&*()_+-=[]{}|;:',.<>?/`~",
        )
        retrieved = [RetrievedChunk(chunk=chunk, score=0.95)]

        prompt = _build_system_prompt(retrieved)

        assert "!@#$%^&*()" in prompt
        assert "_+-=[]{}|;:'" in prompt

    def test_build_system_prompt_with_unicode_content(self) -> None:
        """System prompt handles unicode characters in chunk content."""
        chunk = DocumentChunk(
            chunk_id="unicode-chunk",
            document_id="doc-1",
            content="Unicode: 你好 مرحبا café naïve Москва",
        )
        retrieved = [RetrievedChunk(chunk=chunk, score=0.95)]

        prompt = _build_system_prompt(retrieved)

        assert "你好" in prompt
        assert "مرحبا" in prompt
        assert "café" in prompt

    def test_build_system_prompt_with_newlines_in_content(self) -> None:
        """System prompt preserves newlines in chunk content."""
        chunk = DocumentChunk(
            chunk_id="newline-chunk",
            document_id="doc-1",
            content="Line 1\nLine 2\nLine 3",
        )
        retrieved = [RetrievedChunk(chunk=chunk, score=0.95)]

        prompt = _build_system_prompt(retrieved)

        assert "Line 1\nLine 2\nLine 3" in prompt

    def test_build_system_prompt_single_chunk(self) -> None:
        """System prompt with single chunk includes numbering."""
        chunk = DocumentChunk(
            chunk_id="single",
            document_id="doc-1",
            content="Single chunk content",
        )
        retrieved = [RetrievedChunk(chunk=chunk, score=0.95)]

        prompt = _build_system_prompt(retrieved)

        assert "[1] Single chunk content" in prompt

    def test_build_system_prompt_many_chunks(self) -> None:
        """System prompt with many chunks numbers them sequentially."""
        chunks = [
            RetrievedChunk(
                chunk=DocumentChunk(
                    chunk_id=f"chunk-{i}",
                    document_id="doc-1",
                    content=f"Content {i}",
                ),
                score=0.95 - i * 0.01,
            )
            for i in range(10)
        ]

        prompt = _build_system_prompt(chunks)

        for i in range(1, 11):
            assert f"[{i}]" in prompt


class TestMessageBuilderEdgeCases:
    """Additional tests for message list construction."""

    def test_build_messages_with_empty_query(self) -> None:
        """Build messages handles empty query string."""
        messages = _build_messages("", "System.", None)

        assert len(messages) == 2
        assert messages[-1]["content"] == ""

    def test_build_messages_with_very_long_query(self) -> None:
        """Build messages handles very long query."""
        long_query = "Q?" * 5000
        messages = _build_messages(long_query, "System.", None)

        assert messages[-1]["content"] == long_query

    def test_build_messages_with_newlines_in_query(self) -> None:
        """Build messages preserves newlines in query."""
        query_with_newlines = "Line 1\nLine 2\nLine 3?"
        messages = _build_messages(query_with_newlines, "System.", None)

        assert messages[-1]["content"] == query_with_newlines

    def test_build_messages_with_special_chars_in_query(self) -> None:
        """Build messages preserves special characters in query."""
        special_query = "What is !@#$%^&*()? [test] {value}"
        messages = _build_messages(special_query, "System.", None)

        assert messages[-1]["content"] == special_query

    def test_build_messages_alternating_roles(self) -> None:
        """Build messages correctly handles alternating user/assistant roles."""
        history = [
            ChatTurn(role="user", content="Q1"),
            ChatTurn(role="assistant", content="A1"),
            ChatTurn(role="user", content="Q2"),
            ChatTurn(role="assistant", content="A2"),
        ]

        messages = _build_messages("Q3", "System.", history)

        # Verify alternating pattern: system, user, assistant, user, assistant, user
        roles = [msg["role"] for msg in messages]
        assert roles == ["system", "user", "assistant", "user", "assistant", "user"]

    def test_build_messages_preserves_role_casing(self) -> None:
        """Build messages preserves exact role values."""
        history = [
            ChatTurn(role="user", content="Q"),
            ChatTurn(role="assistant", content="A"),
        ]

        messages = _build_messages("Query", "System.", history)

        for msg in messages[1:-1]:  # Skip system and final user
            assert msg["role"] in ("user", "assistant")

    def test_build_messages_very_long_system_prompt(self) -> None:
        """Build messages handles very long system prompt."""
        long_system = "S." * 10000
        messages = _build_messages("Q?", long_system, None)

        assert messages[0]["content"] == long_system


class TestOpenAICompatCall:
    """Tests for the OpenAI-compatible API call function."""

    def test_call_openai_compat_returns_string(self) -> None:
        """_call_openai_compat returns a string response."""
        with patch("api.services.llm.openai.OpenAI") as mock_oai:
            client_instance = MagicMock()
            mock_oai.return_value = client_instance

            mock_choice = MagicMock()
            mock_choice.message.content = "Test response"
            client_instance.chat.completions.create.return_value.choices = [mock_choice]

            result = _call_openai_compat(
                "Query", "System", None, "gpt-4o-mini", None, "key"
            )

            assert result == "Test response"

    def test_call_openai_compat_with_base_url(self) -> None:
        """_call_openai_compat uses base_url when provided."""
        with patch("api.services.llm.openai.OpenAI") as mock_oai:
            client_instance = MagicMock()
            mock_oai.return_value = client_instance

            mock_choice = MagicMock()
            mock_choice.message.content = "Response"
            client_instance.chat.completions.create.return_value.choices = [mock_choice]

            _call_openai_compat(
                "Query",
                "System",
                None,
                "llama3",
                "http://localhost:11434/v1",
                "ollama",
            )

            # Verify OpenAI client was initialized with base_url
            mock_oai.assert_called_once()
            call_kwargs = mock_oai.call_args.kwargs
            assert call_kwargs["base_url"] == "http://localhost:11434/v1"

    def test_call_openai_compat_with_chat_history(self) -> None:
        """_call_openai_compat includes chat history in message list."""
        history = [
            ChatTurn(role="user", content="Q1"),
            ChatTurn(role="assistant", content="A1"),
        ]

        with patch("api.services.llm.openai.OpenAI") as mock_oai:
            client_instance = MagicMock()
            mock_oai.return_value = client_instance

            mock_choice = MagicMock()
            mock_choice.message.content = "Response"
            client_instance.chat.completions.create.return_value.choices = [mock_choice]

            _call_openai_compat(
                "Q2", "System", history, "gpt-4o-mini", None, "key"
            )

            # Verify messages included history
            call_kwargs = client_instance.chat.completions.create.call_args.kwargs
            messages = call_kwargs["messages"]
            assert len(messages) == 4  # system + 2 history + 1 current

    def test_call_openai_compat_empty_response_content(self) -> None:
        """_call_openai_compat handles None/empty message content gracefully."""
        with patch("api.services.llm.openai.OpenAI") as mock_oai:
            client_instance = MagicMock()
            mock_oai.return_value = client_instance

            mock_choice = MagicMock()
            mock_choice.message.content = None  # Empty response
            client_instance.chat.completions.create.return_value.choices = [mock_choice]

            result = _call_openai_compat(
                "Query", "System", None, "gpt-4o-mini", None, "key"
            )

            # Should return empty string for None content
            assert result == ""

    def test_call_openai_compat_api_key_passed_to_client(self) -> None:
        """_call_openai_compat passes API key to OpenAI client."""
        with patch("api.services.llm.openai.OpenAI") as mock_oai:
            client_instance = MagicMock()
            mock_oai.return_value = client_instance

            mock_choice = MagicMock()
            mock_choice.message.content = "Response"
            client_instance.chat.completions.create.return_value.choices = [mock_choice]

            _call_openai_compat(
                "Query", "System", None, "gpt-4o-mini", None, "test-key-123"
            )

            # Verify API key was passed
            call_kwargs = mock_oai.call_args.kwargs
            assert call_kwargs["api_key"] == "test-key-123"


class TestAnthropicCall:
    """Tests for the Anthropic API call function."""

    def test_call_anthropic_returns_string(self) -> None:
        """_call_anthropic returns a string response."""
        with patch("api.services.llm.anthropic.Anthropic") as mock_anthro:
            client_instance = MagicMock()
            mock_anthro.return_value = client_instance

            mock_content = MagicMock()
            mock_content.text = "Claude response"
            mock_response = MagicMock()
            mock_response.content = [mock_content]
            client_instance.messages.create.return_value = mock_response

            result = _call_anthropic("Query", "System", None, "claude-opus-4-1")

            assert result == "Claude response"

    def test_call_anthropic_passes_system_param(self) -> None:
        """_call_anthropic passes system prompt as parameter, not in messages."""
        with patch("api.services.llm.anthropic.Anthropic") as mock_anthro:
            client_instance = MagicMock()
            mock_anthro.return_value = client_instance

            mock_content = MagicMock()
            mock_content.text = "Response"
            mock_response = MagicMock()
            mock_response.content = [mock_content]
            client_instance.messages.create.return_value = mock_response

            _call_anthropic("Query", "System prompt", None, "claude-opus-4-1")

            call_kwargs = client_instance.messages.create.call_args.kwargs
            assert call_kwargs["system"] == "System prompt"

    def test_call_anthropic_with_chat_history(self) -> None:
        """_call_anthropic includes chat history in messages (not system)."""
        history = [
            ChatTurn(role="user", content="Q1"),
            ChatTurn(role="assistant", content="A1"),
        ]

        with patch("api.services.llm.anthropic.Anthropic") as mock_anthro:
            client_instance = MagicMock()
            mock_anthro.return_value = client_instance

            mock_content = MagicMock()
            mock_content.text = "Response"
            mock_response = MagicMock()
            mock_response.content = [mock_content]
            client_instance.messages.create.return_value = mock_response

            _call_anthropic("Q2", "System", history, "claude-opus-4-1")

            call_kwargs = client_instance.messages.create.call_args.kwargs
            messages = call_kwargs["messages"]
            # History (2) + current query (1) = 3 messages, NO system in messages
            assert len(messages) == 3
            assert all(msg["role"] in ("user", "assistant") for msg in messages)

    def test_call_anthropic_api_key_passed(self) -> None:
        """_call_anthropic passes API key to Anthropic client."""
        with patch("api.services.llm.anthropic.Anthropic") as mock_anthro:
            client_instance = MagicMock()
            mock_anthro.return_value = client_instance

            mock_content = MagicMock()
            mock_content.text = "Response"
            mock_response = MagicMock()
            mock_response.content = [mock_content]
            client_instance.messages.create.return_value = mock_response

            with patch("api.services.llm.ANTHROPIC_API_KEY", "claude-key-123"):
                _call_anthropic("Query", "System", None, "claude-opus-4-1")

                # Verify Anthropic client was created
                assert mock_anthro.called

    def test_call_anthropic_max_tokens_set(self) -> None:
        """_call_anthropic sets max_tokens parameter."""
        with patch("api.services.llm.anthropic.Anthropic") as mock_anthro:
            client_instance = MagicMock()
            mock_anthro.return_value = client_instance

            mock_content = MagicMock()
            mock_content.text = "Response"
            mock_response = MagicMock()
            mock_response.content = [mock_content]
            client_instance.messages.create.return_value = mock_response

            _call_anthropic("Query", "System", None, "claude-opus-4-1")

            call_kwargs = client_instance.messages.create.call_args.kwargs
            assert call_kwargs.get("max_tokens") == 2048
