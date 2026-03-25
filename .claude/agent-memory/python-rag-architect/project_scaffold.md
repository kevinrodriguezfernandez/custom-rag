---
name: project_scaffold
description: Initial monorepo skeleton created on 2026-03-25 with all modules, stubs, and dependencies
type: project
---

Monorepo skeleton created with full directory structure and stub files.

**Why:** Establish the foundation so all specialist agents (streamlit-rag-ui-dev, rag-api-developer, shared-models-architect, rag-tests-writer) can begin work immediately without creating files.

**How to apply:** All new feature work must follow the build order: ingestion -> shared -> app -> api. Every data shape goes in shared/models.py first. All config reads go through shared/config.py.

Key decisions made:
- LangChain + langchain-openai + langchain-text-splitters included for chunking/embedding pipeline
- pydantic-settings included for future structured settings (not yet used, available)
- pytest-asyncio in auto mode for async test support
- app/ is a uv workspace member with its own pyproject.toml (no deps of its own yet)
- All API routes raise NotImplementedError as stubs
- .env.example provided as the config template
