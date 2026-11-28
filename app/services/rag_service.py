import re
from collections.abc import Callable

from app.services import embedding_service, guardrails, llm, vector_store
from app.services.document_store import Evidence, SummaryPoint
from app.services.vector_store import SearchHit

TOP_K = 3
MAX_PASSAGE_CHARS = 3500
EXCERPT_CHARS = 320

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_WORD = re.compile(r"\w{4,}")


def verify(document_id: str, points: list[str], on_checked: Callable[[SummaryPoint], None]) -> list[SummaryPoint]:
    claims = [guardrails.strip_citations(point) for point in points]
    embeddings = embedding_service.embed_queries(claims)
    checked: list[SummaryPoint] = []
    for point, claim, embedding in zip(points, claims, embeddings, strict=True):
        result = _verify_one(document_id, point, claim, embedding)
        checked.append(result)
        on_checked(result)
    return checked


def _verify_one(document_id: str, point: str, claim: str, embedding: list[float]) -> SummaryPoint:
    cited = guardrails.cited_pages(point)
    hits = vector_store.search(embedding, TOP_K, document_id)
    if not hits:
        return SummaryPoint(text=point)

    passages = _passages(hits, vector_store.chunks_on_pages(document_id, cited))
    texts = [hit.text for hit in passages]
    supported = guardrails.numbers_match(claim, texts) and guardrails.judge_says_yes(
        llm.chat(
            guardrails.JUDGE_SYSTEM_PROMPT, guardrails.build_judge_prompt(texts, claim), task="rag.judge_claim", num_predict=4
        )
    )
    best = hits[0]
    return SummaryPoint(text=point, verified=supported, evidence=Evidence(page=best.page, excerpt=_excerpt(claim, best.text)))


def _passages(hits: list[SearchHit], cited: list[SearchHit]) -> list[SearchHit]:
    selected: list[SearchHit] = []
    seen: set[str] = set()
    used = 0
    for hit in [*hits, *cited]:
        if hit.text in seen or used + len(hit.text) > MAX_PASSAGE_CHARS:
            continue
        selected.append(hit)
        seen.add(hit.text)
        used += len(hit.text)
    return selected


def _excerpt(claim: str, passage: str) -> str:
    sentences = _SENTENCE_SPLIT.split(passage)
    words = {w.lower() for w in _WORD.findall(claim)}
    best = max(range(len(sentences)), key=lambda i: len(words & {w.lower() for w in _WORD.findall(sentences[i])}))
    excerpt = sentences[best]
    for following in sentences[best + 1 :]:
        if len(excerpt) + 1 + len(following) > EXCERPT_CHARS:
            break
        excerpt = f"{excerpt} {following}"
    if len(excerpt) > EXCERPT_CHARS:
        excerpt = excerpt[:EXCERPT_CHARS].rsplit(" ", 1)[0] + "…"
    return excerpt
