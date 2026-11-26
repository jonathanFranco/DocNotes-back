from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class SourceReference(BaseModel):
    page: int
    excerpt: str
    similarity: float


class ChatResponse(BaseModel):
    answer: str
    found: bool
    sources: list[SourceReference]
