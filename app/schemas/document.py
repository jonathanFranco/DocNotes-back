from typing import Any, Literal

from pydantic import BaseModel


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    page_count: int
    pages_with_text: int


class UploadResponse(BaseModel):
    message: str
    document: DocumentInfo
    processing_seconds: float


class DeleteResponse(BaseModel):
    message: str


class SummaryProgress(BaseModel):
    done: int
    total: int


class SummaryError(BaseModel):
    code: str
    params: dict[str, Any] = {}


class Evidence(BaseModel):
    page: int
    excerpt: str


class SummaryPoint(BaseModel):
    text: str
    verified: bool | None
    evidence: Evidence | None


class SummaryResponse(BaseModel):
    status: Literal["generating", "ready", "failed"]
    stage: Literal["reading", "writing", "checking"] | None
    progress: SummaryProgress | None
    points: list[SummaryPoint]
    error: SummaryError | None
