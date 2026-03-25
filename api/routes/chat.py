# api/routes/chat.py
"""Chat endpoint — accepts a user query, runs the RAG pipeline, returns an answer."""

from fastapi import APIRouter

from shared.models import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Run the RAG pipeline for the given user query.

    Steps (to be implemented):
    1. Embed the query.
    2. Retrieve relevant chunks from Qdrant.
    3. Build a prompt with retrieved context.
    4. Call the LLM and return the answer with sources.
    """
    raise NotImplementedError("Chat endpoint is not yet implemented.")
