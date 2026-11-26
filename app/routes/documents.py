import time
import uuid
from pathlib import PurePath

from fastapi import APIRouter, UploadFile, status

from app.config import get_settings
from app.errors import AppError
from app.schemas.document import (
    DeleteResponse,
    DocumentInfo,
    Evidence,
    SummaryError,
    SummaryPoint,
    SummaryProgress,
    SummaryResponse,
    UploadResponse,
)
from app.services import document_store, embedding_service, pdf_processor, summary_service, vector_store
from app.services.document_store import StoredDocument

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentInfo)
def get_current_document() -> DocumentInfo:
    return _info(_current())


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
    document = StoredDocument(document_id=uuid.uuid4().hex, filename=filename, page_count=page_count, pages=pages)

    summary_service.cancel()
    vector_store.replace(document.document_id, chunks, embeddings)
    document_store.save(document)
    summary_service.start(document)

    return UploadResponse(
        message="Document processed successfully. The summary is being generated.",
        document=_info(document),
        processing_seconds=round(time.perf_counter() - started, 2),
    )


@router.get("/summary", response_model=SummaryResponse)
def get_summary() -> SummaryResponse:
    return _summary(summary_service.state(_current()))


@router.post("/summary", response_model=SummaryResponse, status_code=status.HTTP_202_ACCEPTED)
def retry_summary() -> SummaryResponse:
    document = _current()
    if document.summary is None:
        summary_service.start(document, retry=True)
    return _summary(summary_service.state(document))


@router.delete("", response_model=DeleteResponse)
def delete_document() -> DeleteResponse:
    summary_service.cancel()
    vector_store.clear()
    if not document_store.delete():
        raise AppError("NO_DOCUMENT", "No document loaded.", status_code=status.HTTP_404_NOT_FOUND)
    return DeleteResponse(message="Document deleted.")


def _current() -> StoredDocument:
    document = document_store.load()
    if document is None:
        raise AppError("NO_DOCUMENT", "No document loaded.", status_code=status.HTTP_404_NOT_FOUND)
    return document


def _info(document: StoredDocument) -> DocumentInfo:
    return DocumentInfo(
        document_id=document.document_id,
        filename=document.filename,
        page_count=document.page_count,
        pages_with_text=len(document.pages),
    )


def _summary(state: summary_service.SummaryState) -> SummaryResponse:
    return SummaryResponse(
        status=state.status,
        stage=state.stage,
        progress=SummaryProgress(done=state.done, total=state.total) if state.total else None,
        points=[
            SummaryPoint(
                text=point.text,
                verified=point.verified,
                evidence=Evidence(page=point.evidence.page, excerpt=point.evidence.excerpt) if point.evidence else None,
            )
            for point in state.points
        ],
        error=SummaryError(code=state.error_code, params=state.error_params) if state.error_code else None,
    )
