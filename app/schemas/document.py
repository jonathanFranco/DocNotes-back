from pydantic import BaseModel


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    page_count: int
    pages_with_text: int
    chunk_count: int


class UploadResponse(BaseModel):
    message: str
    document: DocumentInfo
    processing_seconds: float


class DeleteResponse(BaseModel):
    message: str
