# api/routes/chat.py
"""Chat endpoint — accepts a user query, runs the RAG pipeline, returns an answer."""

from fastapi import APIRouter, HTTPException

from api.services.llm import generate_answer
from api.services.retriever import retrieve
from shared.models import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run the RAG pipeline for the given user query.

    1. Embed the query and retrieve the top-k relevant chunks from Qdrant.
    2. Build a prompt with retrieved context and call the LLM.
    3. Return the answer along with source chunks for citation.
    """
    try:
        chunks = await retrieve(request.query, request.top_k)
    except Exception:
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
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM generation failed: {exc}") from exc

    return ChatResponse(answer=answer, sources=chunks, model=request.model)
