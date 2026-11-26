import shutil
import threading
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.services.pdf_processor import Chunk

COLLECTION_NAME = "documents"

_write_lock = threading.Lock()


@dataclass(frozen=True)
class SearchHit:
    text: str
    page: int
    similarity: float


def _path() -> Path:
    return get_settings().data_path / "chroma"


@lru_cache
def _client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=_path(), settings=ChromaSettings(anonymized_telemetry=False))


def _collection() -> chromadb.Collection:
    return _client().get_or_create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )


def replace(document_id: str, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
    with _write_lock:
        _clear()
        _collection().add(
            ids=[f"{document_id}:{c.index}" for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{"document_id": document_id, "page": c.page, "chunk_index": c.index} for c in chunks],
        )


def clear() -> None:
    with _write_lock:
        _clear()


def search(embedding: list[float], top_k: int, document_id: str) -> list[SearchHit]:
    collection = _collection()
    count = collection.count()
    if count == 0:
        return []
    result = collection.query(
        query_embeddings=[embedding],
        n_results=min(top_k, count),
        where={"document_id": document_id},
        include=["documents", "metadatas", "distances"],
    )
    return [
        SearchHit(text=text, page=int(meta["page"]), similarity=1 - distance)
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0], strict=True
        )
    ]


def chunks_on_pages(document_id: str, pages: list[int]) -> list[SearchHit]:
    if not pages:
        return []
    result = _collection().get(
        where={"$and": [{"document_id": document_id}, {"page": {"$in": pages}}]},
        include=["documents", "metadatas"],
    )
    rows = sorted(zip(result["documents"], result["metadatas"], strict=True), key=lambda row: int(row[1]["chunk_index"]))
    return [SearchHit(text=text, page=int(meta["page"]), similarity=0.0) for text, meta in rows]


def _clear() -> None:
    if _collection().count() > 0:
        _client().delete_collection(COLLECTION_NAME)
        _remove_orphan_indexes()


def _remove_orphan_indexes() -> None:
    for entry in _path().iterdir():
        if entry.is_dir() and _is_uuid(entry.name):
            shutil.rmtree(entry, ignore_errors=True)


def _is_uuid(name: str) -> bool:
    try:
        uuid.UUID(name)
    except ValueError:
        return False
    return True
