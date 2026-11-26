import json
import os
import threading
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from app.config import get_settings
from app.services.pdf_processor import Page

_lock = threading.Lock()


@dataclass(frozen=True)
class Evidence:
    page: int
    excerpt: str


@dataclass(frozen=True)
class SummaryPoint:
    text: str
    verified: bool | None = None
    evidence: Evidence | None = None


@dataclass(frozen=True)
class StoredDocument:
    document_id: str
    filename: str
    page_count: int
    pages: list[Page]
    summary: list[SummaryPoint] | None = None


def load() -> StoredDocument | None:
    try:
        raw = json.loads(_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    return StoredDocument(
        document_id=raw["document_id"],
        filename=raw["filename"],
        page_count=raw["page_count"],
        pages=[Page(**page) for page in raw["pages"]],
        summary=None if raw.get("summary") is None else [_point(p) for p in raw["summary"]],
    )


def _point(raw: str | dict) -> SummaryPoint:
    if isinstance(raw, str):
        return SummaryPoint(text=raw)
    evidence = raw.get("evidence")
    return SummaryPoint(text=raw["text"], verified=raw.get("verified"), evidence=Evidence(**evidence) if evidence else None)


def save(document: StoredDocument) -> None:
    with _lock:
        _write(document)


def save_summary(document_id: str, points: list[SummaryPoint]) -> bool:
    with _lock:
        document = load()
        if document is None or document.document_id != document_id:
            return False
        _write(replace(document, summary=points))
        return True


def delete() -> bool:
    with _lock:
        try:
            _path().unlink()
        except FileNotFoundError:
            return False
        return True


def _path() -> Path:
    return get_settings().data_path / "document.json"


def _write(document: StoredDocument) -> None:
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(asdict(document), ensure_ascii=False), encoding="utf-8")
    os.replace(temp, path)
