import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Literal

from app.errors import AppError
from app.services import document_store, guardrails, llm, rag_service
from app.services.document_store import StoredDocument, SummaryPoint
from app.services.pdf_processor import Page, split_text

BLOCK_CHARS = 2000
REDUCE_INPUT_CHARS = 6000
MAP_MAX_TOKENS = 600
FINAL_MAX_TOKENS = 1500
MAX_REDUCE_ROUNDS = 5

Status = Literal["generating", "ready", "failed"]
Stage = Literal["reading", "writing", "checking"]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Block:
    sections: list[tuple[int, str]]
    pages: frozenset[int]


@dataclass
class _Job:
    document_id: str
    status: Status = "generating"
    stage: Stage = "reading"
    done: int = 0
    total: int = 0
    draft: str = ""
    draft_pages: frozenset[int] = frozenset()
    written: list[str] = field(default_factory=list)
    checked: list[SummaryPoint] = field(default_factory=list)
    error_code: str | None = None
    error_params: dict[str, Any] = field(default_factory=dict)
    cancelled: bool = False


@dataclass(frozen=True)
class SummaryState:
    status: Status
    stage: Stage | None
    done: int
    total: int
    points: list[SummaryPoint]
    error_code: str | None
    error_params: dict[str, Any]


_jobs: dict[str, _Job] = {}
_lock = threading.Lock()


def state(document: StoredDocument) -> SummaryState:
    if document.summary is not None:
        return SummaryState("ready", None, 0, 0, document.summary, None, {})
    return _snapshot(start(document))


def start(document: StoredDocument, *, retry: bool = False) -> _Job:
    with _lock:
        job = _jobs.get(document.document_id)
        if job and (job.status == "generating" or not retry):
            return job
        _cancel_all()
        job = _jobs[document.document_id] = _Job(document.document_id)
    threading.Thread(target=_run, args=(job, document), daemon=True, name=f"summary-{document.document_id}").start()
    return job


def cancel() -> None:
    with _lock:
        _cancel_all()


def _cancel_all() -> None:
    for job in _jobs.values():
        job.cancelled = True
    _jobs.clear()


def _snapshot(job: _Job) -> SummaryState:
    if job.stage == "writing":
        points = [SummaryPoint(text=text) for text in guardrails.parse_points(job.draft, job.draft_pages)]
    else:
        checked = list(job.checked)
        points = checked + [SummaryPoint(text=text) for text in job.written[len(checked) :]]
    stage = job.stage if job.status == "generating" else None
    return SummaryState(job.status, stage, job.done, job.total, points, job.error_code, job.error_params)


def _run(job: _Job, document: StoredDocument) -> None:
    try:
        written = _summarize(job, document.pages)
        if job.cancelled:
            return
        if not written:
            raise AppError("SUMMARY_EMPTY", "The model did not return any summary points.")
        job.written = written
        job.done, job.total = 0, len(written)
        job.stage = "checking"
        points = rag_service.verify(document.document_id, written, lambda point: _checked(job, point))
        if job.cancelled:
            return
        document_store.save_summary(document.document_id, points)
        job.status = "ready"
    except AppError as exc:
        job.error_code = exc.code
        job.error_params = exc.params
        job.status = "failed"
    except Exception:
        logger.exception("Summary failed for document %s", document.document_id)
        job.error_code = "SUMMARY_FAILED"
        job.status = "failed"


def _checked(job: _Job, point: SummaryPoint) -> None:
    job.checked.append(point)
    job.done += 1


def _summarize(job: _Job, pages: list[Page]) -> list[str]:
    blocks = _blocks(pages)
    job.total = len(blocks)
    notes: list[str] = []
    for block in blocks:
        if job.cancelled:
            return []
        answer = llm.chat(
            guardrails.MAP_SYSTEM_PROMPT,
            guardrails.build_block_prompt(block.sections),
            task="summary.map",
            num_predict=MAP_MAX_TOKENS,
        )
        notes.append(_as_list(guardrails.parse_page_sections(answer, block.pages)))
        job.done += 1

    all_pages = frozenset(page.number for page in pages)
    for _ in range(MAX_REDUCE_ROUNDS):
        if len("\n".join(notes)) <= REDUCE_INPUT_CHARS or job.cancelled:
            break
        notes = [_reduce(batch, all_pages) for batch in _batches(notes)]

    job.draft_pages = all_pages
    job.stage = "writing"
    prompt = guardrails.build_notes_prompt(notes)
    for piece in llm.stream_chat(guardrails.REDUCE_SYSTEM_PROMPT, prompt, task="summary.final", num_predict=FINAL_MAX_TOKENS):
        if job.cancelled:
            return []
        job.draft += piece
    return guardrails.parse_points(job.draft, all_pages)


def _reduce(batch: list[str], valid_pages: frozenset[int]) -> str:
    answer = llm.chat(
        guardrails.REDUCE_SYSTEM_PROMPT,
        guardrails.build_notes_prompt(batch),
        task="summary.reduce",
        num_predict=MAP_MAX_TOKENS * 2,
    )
    return _as_list(guardrails.parse_points(answer, valid_pages))


def _blocks(pages: list[Page]) -> list[Block]:
    pieces = [(page.number, piece) for page in pages for piece in split_text(page.text, BLOCK_CHARS)]
    blocks: list[Block] = []
    current: list[tuple[int, str]] = []
    size = 0
    for page, piece in pieces:
        if current and size + len(piece) > BLOCK_CHARS:
            blocks.append(_block(current))
            current, size = [], 0
        current.append((page, piece))
        size += len(piece)
    if current:
        blocks.append(_block(current))
    return blocks


def _block(pieces: list[tuple[int, str]]) -> Block:
    sections: list[tuple[int, str]] = []
    for page, text in pieces:
        if sections and sections[-1][0] == page:
            sections[-1] = (page, f"{sections[-1][1]} {text}")
        else:
            sections.append((page, text))
    return Block(sections=sections, pages=frozenset(page for page, _ in sections))


def _batches(notes: list[str]) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for note in notes:
        if len(current) >= 2 and size + len(note) > REDUCE_INPUT_CHARS:
            batches.append(current)
            current, size = [], 0
        current.append(note)
        size += len(note) + 1
    if current:
        batches.append(current)
    return batches


def _as_list(points: list[str]) -> str:
    return "\n".join(f"- {point}" for point in points)
