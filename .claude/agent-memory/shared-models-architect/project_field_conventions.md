---
name: Field naming and type conventions
description: Standardized field naming, type choices, and patterns established in shared/models.py
type: project
---

## Naming

- Model selection field: `model: str` — plain string, not an Enum, to stay flexible as OpenAI model names evolve.
- Default OpenAI model: `"gpt-4o-mini"` — established as the project default for generation.
- Conversation history field: `chat_history: list[ChatTurn]` — ordered oldest-first.

## Type choices

- Use `Literal["user", "assistant"]` for role fields instead of a `SourceType`-style Enum when the value set is small and stable.
- Use `str | None = Field(default=None, ...)` for optional echo-back fields on response models (e.g., `ChatResponse.model`).
- Avoid `dict[str, str]` for structured data — define a proper `BaseModel` (e.g., `ChatTurn`) even when the shape is simple.
- `default_factory=list` for all list fields; never bare `[]`.

## All fields use `Field(...)` with a `description` argument.
