---
name: shared/models.py model inventory
description: All Pydantic models defined in the monorepo and their purpose
type: project
---

All models live in shared/models.py. No other file may define domain or API models.

Domain:
- DocumentChunk(chunk_id, document_id, content, metadata)
- RetrievedChunk(chunk: DocumentChunk, score: float)

Conversation:
- ChatTurn(role: Literal["user","assistant"], content)

API request/response:
- ChatRequest(query, top_k=5, model="gpt-4o-mini", chat_history=[])
- ChatResponse(answer, sources: list[RetrievedChunk], model)
- IngestRequest(document_id, metadata={})
- IngestResponse(document_id, chunks_created)
- HealthResponse(status="ok", version="0.1.0")

**Why:** Single source of truth prevents model drift between frontend, API, and ingestion pipeline.
**How to apply:** If a new model is needed, flag it to the shared-models owner — never define models inside api/ or app/.
