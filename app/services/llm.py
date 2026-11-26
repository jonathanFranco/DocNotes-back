from collections.abc import Iterator

import ollama
from fastapi import status

from app.config import get_settings
from app.errors import AppError


class OllamaError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


def chat(system: str, user: str, *, num_predict: int) -> str:
    return "".join(stream_chat(system, user, num_predict=num_predict)).strip()


def stream_chat(system: str, user: str, *, num_predict: int) -> Iterator[str]:
    settings = get_settings()
    client = ollama.Client(host=settings.ollama_host, timeout=settings.ollama_timeout_seconds)
    try:
        for chunk in client.chat(
            model=settings.llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": 0, "seed": 42, "num_ctx": 4096, "num_predict": num_predict},
            stream=True,
        ):
            yield chunk.message.content or ""
    except ollama.ResponseError as exc:
        raise OllamaError(
            "OLLAMA_MODEL_ERROR", f"Ollama rejected model '{settings.llm_model}': {exc.error}", model=settings.llm_model
        ) from exc
    except Exception as exc:
        raise OllamaError("OLLAMA_UNAVAILABLE", f"Could not reach Ollama at {settings.ollama_host}.") from exc
