# api/routes/chat.py
"""Chat endpoint — accepts a user query, runs the RAG pipeline, returns an answer."""

from fastapi import APIRouter, HTTPException

from api.services.llm import generate_answer
from api.services.retriever import retrieve
from shared.models import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run the RAG pipeline for the given user query.

    Steps:
    1. Embed the query and retrieve the top-k relevant chunks from the vector store.
    2. Build a grounded prompt from the retrieved context.
    3. Call the LLM (OpenAI / Anthropic / Ollama, routed by model name).
    4. Return the generated answer together with the source chunks.
    """
    # 1. Retrieve relevant chunks — synchronous network I/O, offloaded to a thread
    try:
        chunks = await asyncio.to_thread(
            retrieve,
            request.query,
            request.top_k,
        )
    except Exception as exc:
        logger.exception(
            "Retrieval failed for query=%r model=%r", request.query, request.model
        )
        raise HTTPException(
            status_code=502, detail=f"Retrieval failed: {exc}"
        ) from exc

    # 2. Generate answer — synchronous network I/O, offloaded to a thread
    try:
        answer: str = await asyncio.to_thread(
            generate_answer,
            request.query,
            chunks,
            request.model,
            request.chat_history,
        )
    except ValueError as exc:
        # Raised by generate_answer when a required API key is missing
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(
            "LLM generation failed for query=%r model=%r", request.query, request.model
        )
        raise HTTPException(
            status_code=502, detail=f"LLM generation failed: {exc}"
        ) from exc

    return ChatResponse(answer=answer, sources=chunks, model=request.model)
