# api/routes/chat.py
"""Chat endpoint — accepts a user query, runs the RAG pipeline, returns an answer."""

import logging

from fastapi import APIRouter, HTTPException

from api.services.llm import generate_answer
from api.services.retriever import retrieve
from shared.models import ChatRequest, ChatResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run the RAG pipeline for the given user query.

    1. Embed the query and retrieve the top-k relevant chunks from Qdrant.
    2. Build a prompt with retrieved context and call the LLM.
    3. Return the answer along with source chunks for citation.
    """
    logger.info(
        "Chat request received — query=%r model=%s top_k=%d chat_id=%s",
        request.query[:100],
        request.model,
        request.top_k,
        request.chat_id,
    )

    try:
        chunks = await retrieve(request.query, request.top_k, chat_id=request.chat_id)
        logger.info("Retrieval complete — chunks_returned=%d", len(chunks))
    except Exception as exc:
        logger.warning("Retrieval failed, proceeding without context — error=%s", exc)
        chunks = []

    try:
        answer = await generate_answer(
            query=request.query,
            context_chunks=chunks,
            model=request.model,
            chat_history=request.chat_history,
            api_key=request.api_key,
            api_url=request.api_url,
        )
        logger.info("Generation complete — answer_length=%d", len(answer))
    except Exception as exc:
        logger.error("LLM generation failed — error=%s", exc)
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}") from exc

    return ChatResponse(answer=answer, sources=chunks, model=request.model)
