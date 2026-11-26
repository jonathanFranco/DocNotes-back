import ollama

from app.config import get_settings

BATCH_SIZE = 32


class OllamaError(RuntimeError):
    pass


def get_client() -> ollama.Client:
    settings = get_settings()
    return ollama.Client(host=settings.ollama_host, timeout=settings.ollama_timeout_seconds)


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed([f"title: none | text: {t}" for t in texts])


def embed_query(question: str) -> list[float]:
    return _embed([f"task: search result | query: {question}"])[0]


def _embed(inputs: list[str]) -> list[list[float]]:
    settings = get_settings()
    client = get_client()
    vectors: list[list[float]] = []
    try:
        for start in range(0, len(inputs), BATCH_SIZE):
            response = client.embed(model=settings.embedding_model, input=inputs[start : start + BATCH_SIZE])
            vectors.extend(list(v) for v in response.embeddings)
    except ollama.ResponseError as exc:
        raise OllamaError(f"Ollama recusou o modelo '{settings.embedding_model}': {exc.error}") from exc
    except Exception as exc:
        raise OllamaError(f"Não foi possível gerar embeddings via Ollama em {settings.ollama_host}.") from exc
    return vectors
