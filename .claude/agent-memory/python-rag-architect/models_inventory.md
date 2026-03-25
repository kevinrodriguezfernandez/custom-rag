---
name: models_inventory
description: Pydantic models defined in shared/models.py as of initial scaffold
type: project
---

Models in shared/models.py (created 2026-03-25):

**Why:** Single source of truth for all data shapes across the monorepo.

**How to apply:** Any new request/response or domain model MUST be added here before writing logic that uses it.

Domain models:
- DocumentChunk: chunk_id, document_id, content, metadata
- RetrievedChunk: chunk (DocumentChunk), score (float)

API models:
- ChatRequest: query (str), top_k (int, default=5)
- ChatResponse: answer (str), sources (list[RetrievedChunk])
- IngestRequest: document_id (str), metadata (dict)
- IngestResponse: document_id (str), chunks_created (int)
- HealthResponse: status (str, default="ok"), version (str, default="0.1.0")
