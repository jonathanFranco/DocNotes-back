import time
import uuid
from pathlib import PurePath

from fastapi import APIRouter, UploadFile, status

from app.config import get_settings
from app.errors import AppError
from app.schemas.document import DeleteResponse, DocumentInfo, UploadResponse
from app.services import embedding_service, pdf_processor, vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentInfo)
def get_current_document() -> DocumentInfo:
    summary = vector_store.current_document()
    if summary is None:
        raise AppError("NO_DOCUMENT", "No document loaded.", status_code=status.HTTP_404_NOT_FOUND)
    return DocumentInfo(
        document_id=summary.document.document_id,
        filename=summary.document.filename,
        page_count=summary.document.page_count,
        pages_with_text=summary.pages_with_text,
        chunk_count=summary.chunk_count,
    )


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
def upload_document(file: UploadFile) -> UploadResponse:
    settings = get_settings()
    filename = PurePath(file.filename or "").name
    if not filename.lower().endswith(".pdf"):
        raise AppError("FILE_NOT_PDF", "Upload a file with the .pdf extension.")

    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise AppError(
            "FILE_TOO_LARGE",
            f"The file exceeds the {settings.max_upload_mb} MB limit.",
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            max_mb=settings.max_upload_mb,
        )
    if not data:
        raise AppError("FILE_EMPTY", "The file is empty.")

    started = time.perf_counter()
    page_count, pages = pdf_processor.extract_pages(data)

    chunks = pdf_processor.chunk_pages(pages)
    embeddings = embedding_service.embed_documents([c.text for c in chunks])

    document = vector_store.StoredDocument(
        document_id=uuid.uuid4().hex, filename=filename, page_count=page_count
    )
    vector_store.replace_document(document, chunks, embeddings)

    return UploadResponse(
        message="Document processed successfully.",
        document=DocumentInfo(
            document_id=document.document_id,
            filename=filename,
            page_count=page_count,
            pages_with_text=len(pages),
            chunk_count=len(chunks),
        ),
        processing_seconds=round(time.perf_counter() - started, 2),
    )


@router.delete("", response_model=DeleteResponse)
def delete_document() -> DeleteResponse:
    if not vector_store.delete_document():
        raise AppError("NO_DOCUMENT", "No document loaded.", status_code=status.HTTP_404_NOT_FOUND)
    return DeleteResponse(message="Document deleted.")
