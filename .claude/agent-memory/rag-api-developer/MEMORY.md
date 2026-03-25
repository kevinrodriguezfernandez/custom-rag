# Agent Memory Index

- [RAG API stack and provider configuration](project_stack.md) — LLM provider routing (OpenAI/Anthropic/Ollama), embedding model, Qdrant collection, default chat model
- [shared/config.py conventions](project_config.md) — which env vars are required vs optional, how config is loaded
- [shared/models.py model inventory](project_models.md) — full list of Pydantic models; never redefine them outside shared/models.py
- [Async offloading for synchronous services](feedback_async_offloading.md) — all service/ingestion calls are sync; always wrap with asyncio.to_thread() in async route handlers
