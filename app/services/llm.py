import logging
from collections.abc import Iterator

import ollama
from fastapi import status

from app.config import get_settings
from app.errors import AppError


class OllamaError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE


CONTEXT_TOKENS = 4096

logger = logging.getLogger(__name__)


def chat(system: str, user: str, *, task: str, num_predict: int) -> str:
    return "".join(stream_chat(system, user, task=task, num_predict=num_predict)).strip()


def stream_chat(system: str, user: str, *, task: str, num_predict: int) -> Iterator[str]:
    settings = get_settings()
    client = ollama.Client(host=settings.ollama_host, timeout=settings.ollama_timeout_seconds)
    try:
        for chunk in client.chat(
            model=settings.llm_model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options={"temperature": 0, "seed": 42, "num_ctx": CONTEXT_TOKENS, "num_predict": num_predict},
            stream=True,
        ):
            if chunk.done:
                _log(task, chunk, num_predict=num_predict)
            yield chunk.message.content or ""
    except ollama.ResponseError as exc:
        raise OllamaError(
            "OLLAMA_MODEL_ERROR", f"Ollama rejected model '{settings.llm_model}': {exc.error}", model=settings.llm_model
        ) from exc
    except Exception as exc:
        raise OllamaError("OLLAMA_UNAVAILABLE", f"Could not reach Ollama at {settings.ollama_host}.") from exc


def _log(task: str, chunk: ollama.ChatResponse, *, num_predict: int) -> None:
    prompt_tokens = chunk.prompt_eval_count or 0
    output_tokens = chunk.eval_count or 0
    logger.info(
        "llm task=%s model=%s prompt_tokens=%d output_tokens=%d/%d ctx=%d done=%s seconds=%.1f",
        task,
        chunk.model,
        prompt_tokens,
        output_tokens,
        num_predict,
        CONTEXT_TOKENS,
        chunk.done_reason,
        (chunk.total_duration or 0) / 1e9,
    )
    if prompt_tokens + num_predict > CONTEXT_TOKENS:
        logger.warning(
            "llm task=%s prompt_tokens=%d + num_predict=%d exceeds num_ctx=%d; the prompt may have been cut",
            task,
            prompt_tokens,
            num_predict,
            CONTEXT_TOKENS,
        )
