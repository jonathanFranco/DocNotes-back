import shutil
import threading
import uuid
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.services.pdf_processor import Chunk

COLLECTION_NAME = "documents"

_write_lock = threading.Lock()


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    filename: str
    page_count: int


@lru_cache
def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=get_settings().chroma_path,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _collection() -> chromadb.Collection:
    return _client().get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def replace_document(document: StoredDocument, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    with _write_lock:
        _clear()
        _collection().add(
            ids=[f"{document.document_id}:{c.index}" for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "document_id": document.document_id,
                    "filename": document.filename,
                    "page_count": document.page_count,
                    "page": c.page,
                    "chunk_index": c.index,
                }
                for c in chunks
            ],
        )


def delete_document() -> bool:
    with _write_lock:
        return _clear()


def _clear() -> bool:
    had_data = _collection().count() > 0
    if had_data:
        _client().delete_collection(COLLECTION_NAME)
        _remove_orphan_indexes()
    return had_data


def _remove_orphan_indexes() -> None:
    for entry in get_settings().chroma_path.iterdir():
        if entry.is_dir() and _is_uuid(entry.name):
            shutil.rmtree(entry, ignore_errors=True)


def _is_uuid(name: str) -> bool:
    try:
        uuid.UUID(name)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class SearchHit:
    text: str
    page: int
    similarity: float


def search(embedding: list[float], top_k: int) -> list[SearchHit]:
    collection = _collection()
    if collection.count() == 0:
        return []
    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    return [
        SearchHit(text=text, page=int(meta["page"]), similarity=1 - distance)
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        )
    ]


@dataclass(frozen=True)
class DocumentSummary:
    document: StoredDocument
    pages_with_text: int
    chunk_count: int


def current_document() -> DocumentSummary | None:
    metadatas = _collection().get(include=["metadatas"])["metadatas"]
    if not metadatas:
        return None
    first = metadatas[0]
    return DocumentSummary(
        document=StoredDocument(
            document_id=str(first["document_id"]),
            filename=str(first["filename"]),
            page_count=int(first["page_count"]),
        ),
        pages_with_text=len({m["page"] for m in metadatas}),
        chunk_count=len(metadatas),
    )


def has_document() -> bool:
    return _collection().count() > 0
