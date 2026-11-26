import re
import unicodedata
from dataclasses import dataclass

import pymupdf
from fastapi import status

from app.errors import AppError


class InvalidPDFError(AppError):
    status_code = status.HTTP_422_UNPROCESSABLE_CONTENT


CHUNK_CHARS = 1000
CHUNK_OVERLAP = 150


@dataclass(frozen=True)
class Page:
    number: int
    text: str


@dataclass(frozen=True)
class Chunk:
    text: str
    page: int
    index: int


def extract_pages(data: bytes) -> tuple[int, list[Page]]:
    if not data.startswith(b"%PDF-"):
        raise InvalidPDFError("PDF_INVALID", "The file is not a valid PDF.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError) as exc:
        raise InvalidPDFError("PDF_CORRUPTED", "Could not open the PDF; the file may be corrupted.") from exc

    with doc:
        if doc.needs_pass:
            raise InvalidPDFError("PDF_PASSWORD_PROTECTED", "Password-protected PDFs are not supported.")
        page_count = doc.page_count
        if page_count == 0:
            raise InvalidPDFError("PDF_NO_PAGES", "The PDF has no pages.")
        pages = [
            Page(number=i + 1, text=text)
            for i, page in enumerate(doc)
            if (text := clean_text(page.get_text("text")))
        ]

    if not pages:
        raise InvalidPDFError(
            "PDF_NO_TEXT",
            "No extractable text found. Scanned (image-only) PDFs are not supported.",
        )
    return page_count, pages


def clean_text(text: str) -> str:
    text = "".join(ch if ch in "ºª" else unicodedata.normalize("NFKC", ch) for ch in unicodedata.normalize("NFC", text))
    text = "".join(ch for ch in text if ch in "\n\t" or unicodedata.category(ch)[0] != "C")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


def chunk_pages(pages: list[Page]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for text in split_text(page.text, CHUNK_CHARS, overlap=CHUNK_OVERLAP):
            chunks.append(Chunk(text=text, page=page.number, index=len(chunks)))
    return chunks


def split_text(text: str, max_chars: int, *, overlap: int = 0) -> list[str]:
    units = [u for s in re.split(r"(?<=[.!?…])\s+|\n\n", text) for u in _hard_wrap(s.strip(), max_chars) if u]
    pieces: list[str] = []
    current: list[str] = []
    length = 0
    for unit in units:
        if current and length + 1 + len(unit) > max_chars:
            pieces.append(" ".join(current))
            kept: list[str] = []
            for previous in reversed(current):
                if sum(len(s) + 1 for s in kept) + len(previous) > overlap:
                    break
                kept.insert(0, previous)
            current, length = kept, sum(len(s) + 1 for s in kept)
        current.append(unit)
        length += len(unit) + 1
    if current:
        pieces.append(" ".join(current))
    return pieces


def _hard_wrap(sentence: str, max_chars: int) -> list[str]:
    if len(sentence) <= max_chars:
        return [sentence]
    parts: list[str] = []
    current = ""
    for word in sentence.split(" "):
        if current and len(current) + 1 + len(word) > max_chars:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        parts.append(current)
    return [p[i : i + max_chars] for p in parts for i in range(0, len(p), max_chars)]
