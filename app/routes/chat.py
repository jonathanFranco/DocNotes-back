from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse, SourceReference
from app.services import rag_service

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    answer = rag_service.answer_question(request.question)
    return ChatResponse(
        answer=answer.text,
        found=answer.found,
        sources=[SourceReference(page=s.page, excerpt=s.excerpt, similarity=s.similarity) for s in answer.sources],
    )
