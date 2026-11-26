import ollama
from fastapi import APIRouter

from app.config import get_settings
from app.schemas.health import HealthResponse, ModelStatus, OllamaStatus

router = APIRouter(tags=["health"])


def _with_tag(name: str) -> str:
    return name if ":" in name else f"{name}:latest"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    installed: set[str] = set()
    error: str | None = None

    try:
        client = ollama.Client(host=settings.ollama_host, timeout=5)
        installed = {m.model for m in client.list().models if m.model}
    except Exception as exc:
        error = f"Ollama unavailable at {settings.ollama_host}: {exc}"

    llm = ModelStatus(name=settings.llm_model, available=_with_tag(settings.llm_model) in installed)
    embedding = ModelStatus(name=settings.embedding_model, available=_with_tag(settings.embedding_model) in installed)
    ollama_status = OllamaStatus(reachable=error is None, llm=llm, embedding=embedding, error=error)
    ok = ollama_status.reachable and llm.available and embedding.available
    return HealthResponse(status="ok" if ok else "degraded", ollama=ollama_status)
