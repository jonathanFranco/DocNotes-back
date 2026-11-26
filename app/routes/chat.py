from fastapi import APIRouter, HTTPException, status

from app.schemas.chat import ChatRequest, ChatResponse, SourceReference
from app.services import rag_service
from app.services.embedding_service import OllamaError
from app.services.guardrails import QuestionError

router = APIRouter(tags=["chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        answer = rag_service.answer_question(request.question)
    except QuestionError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except rag_service.NoDocumentError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except OllamaError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    return ChatResponse(
        answer=answer.text,
        found=answer.found,
        sources=[SourceReference(page=s.page, excerpt=s.excerpt, similarity=s.similarity) for s in answer.sources],
    )
