from dataclasses import dataclass

import ollama
from fastapi import status

from app.config import get_settings
from app.errors import AppError
from app.services import embedding_service, guardrails, vector_store
from app.services.embedding_service import OllamaError

TOP_K = 4
EXCERPT_CHARS = 300


class NoDocumentError(AppError):
    status_code = status.HTTP_409_CONFLICT


@dataclass(frozen=True)
class Source:
    page: int
    excerpt: str
    similarity: float


@dataclass(frozen=True)
class Answer:
    text: str
    found: bool
    sources: list[Source]


def answer_question(question: str) -> Answer:
    question = guardrails.validate_question(question)
    if not vector_store.has_document():
        raise NoDocumentError("NO_DOCUMENT", "No document loaded. Upload a PDF before asking.")

    hits = vector_store.search(embedding_service.embed_query(question), TOP_K)
    context = guardrails.select_context(hits)
    if not context:
        return Answer(text="", found=False, sources=[])

    text = guardrails.clean_answer(_generate(guardrails.build_user_prompt(question, context)))
    if retry_prompt := guardrails.build_retry_prompt(question, context, text):
        text = guardrails.clean_answer(_generate(retry_prompt))
    if guardrails.is_not_found(text):
        return Answer(text="", found=False, sources=[])

    cited = guardrails.cited_pages(text)
    used = [hit for hit in context if hit.page in cited] or context
    sources = [Source(page=hit.page, excerpt=_excerpt(hit.text), similarity=round(hit.similarity, 3)) for hit in used]
    return Answer(text=text, found=True, sources=sources)


def _generate(user_prompt: str) -> str:
    settings = get_settings()
    try:
        response = embedding_service.get_client().chat(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": guardrails.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": 512},
        )
    except ollama.ResponseError as exc:
        raise OllamaError(
            "OLLAMA_MODEL_ERROR", f"Ollama rejected model '{settings.llm_model}': {exc.error}", model=settings.llm_model
        ) from exc
    except Exception as exc:
        raise OllamaError("OLLAMA_UNAVAILABLE", f"Could not generate the answer via Ollama at {settings.ollama_host}.") from exc
    return (response.message.content or "").strip()


def _excerpt(text: str) -> str:
    if len(text) <= EXCERPT_CHARS:
        return text
    return text[:EXCERPT_CHARS].rsplit(" ", 1)[0] + "…"
