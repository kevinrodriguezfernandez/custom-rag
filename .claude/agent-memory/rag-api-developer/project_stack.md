---
name: RAG API stack and provider configuration
description: LLM providers, embedding model, vector store, and routing conventions in use
type: project
---

Multi-provider LLM routing is implemented in api/services/llm.py. Provider is selected by model name at call time, not by a global config flag.

- OpenAI (default): any model not in the Anthropic or Llama sets — uses OPENAI_API_KEY from shared/config.py
- Anthropic: {"claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5"} — requires ANTHROPIC_API_KEY (optional env var, empty string default); raises ValueError at call time if key is missing
- Ollama/Llama: {"llama3.2", "llama3.1", "llama3"} — OpenAI-compatible client pointed at http://localhost:11434/v1, api_key="ollama"

Embedding: OpenAI text-embedding-3-small (OPENAI_EMBEDDING_MODEL env var, default "text-embedding-3-small")
Vector store: Qdrant at QDRANT_URL (default http://localhost:6333), collection QDRANT_COLLECTION (default "documents")
Default chat model: gpt-4o-mini

**Why:** Multi-provider routing allows the Streamlit frontend to select any model via ChatRequest.model without backend reconfiguration.
**How to apply:** When adding new model providers, add the model name set and a new routing branch in llm.py. Keep ANTHROPIC_API_KEY optional — do not add it to the required-key guard in shared/config.py.
