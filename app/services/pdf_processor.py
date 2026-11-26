import re
import unicodedata
from dataclasses import dataclass

import pymupdf

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


class InvalidPDFError(ValueError):
    pass


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
        raise InvalidPDFError("O arquivo não é um PDF válido.")
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except (pymupdf.FileDataError, pymupdf.EmptyFileError, RuntimeError) as exc:
        raise InvalidPDFError("Não foi possível abrir o PDF; o arquivo pode estar corrompido.") from exc

    with doc:
        if doc.needs_pass:
            raise InvalidPDFError("PDF protegido por senha não é suportado.")
        page_count = doc.page_count
        if page_count == 0:
            raise InvalidPDFError("O PDF não contém páginas.")
        pages = [
            Page(number=i + 1, text=text)
            for i, page in enumerate(doc)
            if (text := clean_text(page.get_text("text")))
        ]

    if not pages:
        raise InvalidPDFError(
            "Nenhum texto extraível encontrado. PDFs digitalizados (apenas imagem) não são suportados."
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
        for text in _split(page.text):
            chunks.append(Chunk(text=text, page=page.number, index=len(chunks)))
    return chunks


def _split(text: str) -> list[str]:
    units = [u for s in re.split(r"(?<=[.!?…])\s+|\n\n", text) for u in _hard_wrap(s.strip()) if u]
    chunks: list[str] = []
    current: list[str] = []
    length = 0
    for unit in units:
        if current and length + 1 + len(unit) > CHUNK_SIZE:
            chunks.append(" ".join(current))
            overlap: list[str] = []
            for previous in reversed(current):
                if sum(len(s) + 1 for s in overlap) + len(previous) > CHUNK_OVERLAP:
                    break
                overlap.insert(0, previous)
            current, length = overlap, sum(len(s) + 1 for s in overlap)
        current.append(unit)
        length += len(unit) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks


def _hard_wrap(sentence: str) -> list[str]:
    if len(sentence) <= CHUNK_SIZE:
        return [sentence]
    parts: list[str] = []
    current = ""
    for word in sentence.split(" "):
        if current and len(current) + 1 + len(word) > CHUNK_SIZE:
            parts.append(current)
            current = word
        else:
            current = f"{current} {word}" if current else word
    if current:
        parts.append(current)
    return [p[i : i + CHUNK_SIZE] for p in parts for i in range(0, len(p), CHUNK_SIZE)]
