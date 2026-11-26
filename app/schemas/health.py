from typing import Literal

from pydantic import BaseModel


class ModelStatus(BaseModel):
    name: str
    available: bool


class OllamaStatus(BaseModel):
    reachable: bool
    llm: ModelStatus
    embedding: ModelStatus
    error: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    ollama: OllamaStatus
