import ollama

from app.config import get_settings
from app.services.llm import OllamaError

BATCH_SIZE = 32


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed([f"title: none | text: {t}" for t in texts])


def embed_queries(queries: list[str]) -> list[list[float]]:
    return _embed([f"task: search result | query: {q}" for q in queries])


def _embed(inputs: list[str]) -> list[list[float]]:
    settings = get_settings()
    client = ollama.Client(host=settings.ollama_host, timeout=settings.ollama_timeout_seconds)
    vectors: list[list[float]] = []
    try:
        for start in range(0, len(inputs), BATCH_SIZE):
            response = client.embed(model=settings.embedding_model, input=inputs[start : start + BATCH_SIZE])
            vectors.extend(list(v) for v in response.embeddings)
    except ollama.ResponseError as exc:
        raise OllamaError(
            "OLLAMA_MODEL_ERROR",
            f"Ollama rejected model '{settings.embedding_model}': {exc.error}",
            model=settings.embedding_model,
        ) from exc
    except Exception as exc:
        raise OllamaError(
            "OLLAMA_UNAVAILABLE", f"Could not generate embeddings via Ollama at {settings.ollama_host}."
        ) from exc
    return vectors
