import time
import uuid
from pathlib import PurePath

from fastapi import APIRouter, HTTPException, UploadFile, status

from app.config import get_settings
from app.schemas.document import DeleteResponse, DocumentInfo, UploadResponse
from app.services import embedding_service, pdf_processor, vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("", response_model=DocumentInfo)
def get_current_document() -> DocumentInfo:
    summary = vector_store.current_document()
    if summary is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Nenhum documento carregado.")
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
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Envie um arquivo com extensão .pdf.")

    data = file.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"O arquivo excede o limite de {settings.max_upload_mb} MB.",
        )
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "O arquivo está vazio.")

    started = time.perf_counter()
    try:
        page_count, pages = pdf_processor.extract_pages(data)
    except pdf_processor.InvalidPDFError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    chunks = pdf_processor.chunk_pages(pages)
    try:
        embeddings = embedding_service.embed_documents([c.text for c in chunks])
    except embedding_service.OllamaError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc

    document = vector_store.StoredDocument(
        document_id=uuid.uuid4().hex, filename=filename, page_count=page_count
    )
    vector_store.replace_document(document, chunks, embeddings)

    return UploadResponse(
        message="Documento processado com sucesso.",
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
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Nenhum documento carregado.")
    return DeleteResponse(message="Documento excluído.")
