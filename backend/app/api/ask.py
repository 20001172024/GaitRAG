"""POST /ask: evidence-grounded, cited answer to a question (optionally about a recording)."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.api.schemas import NormComparisonOut
from app.api.search import Mode, get_retriever
from app.config import get_settings
from app.constants import DISCLAIMER
from app.db.models import Paper
from app.db.session import get_session
from app.rag.generator import generate_answer, sources_from_hits
from app.rag.llm import LLMClient, LLMError, build_llm
from app.rag.norms import compare_features
from app.rag.query_builder import build_feature_queries
from app.rag.retriever import Hit, Retriever

router = APIRouter(tags=["generation"])

FEATURE_HITS_PER_QUERY = 2


class AskRequest(BaseModel):
    """A question, optionally with the recording's gait features and predicted speed."""

    question: str = Field(min_length=3, max_length=1000)
    features: dict[str, float] | None = None
    prediction: Literal["slow", "normal", "fast"] | None = None
    mode: Mode | None = None
    rerank: bool | None = None
    tables: bool | None = None


class CitationOut(BaseModel):
    """A source cited in the answer."""

    label: str = Field(description='As written in the answer, e.g. "[P006, p.5, Table 2]"')
    paper_id: str
    title: str
    page: int
    table: str | None
    snippet: str
    source_url: str | None


class AskResponse(BaseModel):
    """Response of POST /ask."""

    answer: str
    citations: list[CitationOut]
    insufficient_evidence: bool
    reason: str = Field(description="answered | no_sources | low_relevance | model_abstained | ...")
    norms_comparison: list[NormComparisonOut]
    model: str
    latency_ms: float
    timings_ms: dict[str, float]
    disclaimer: str = DISCLAIMER


@lru_cache(maxsize=1)
def get_llm() -> LLMClient:
    """The configured LLM client (raises LLMError if no API key is set)."""
    return build_llm(get_settings())


def _llm_or_503() -> LLMClient:
    try:
        return get_llm()
    except LLMError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


def interleave(lists: list[list[Hit]], limit: int) -> list[Hit]:
    """Round-robin merge (first list first), dropping duplicate chunks, up to `limit`."""
    out, seen = [], set()
    for i in range(max((len(x) for x in lists), default=0)):
        for hits in lists:
            if i < len(hits) and hits[i].chunk.chunk_id not in seen:
                seen.add(hits[i].chunk.chunk_id)
                out.append(hits[i])
                if len(out) == limit:
                    return out
    return out


@router.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    # The LLM is resolved first: with no API key the request fails fast with 503,
    # before the retriever (which loads the embedding models) is built.
    llm: Annotated[LLMClient, Depends(_llm_or_503)],
    session: Annotated[Session, Depends(get_session)],
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> AskResponse:
    """Answer with inline citations, or report insufficient evidence."""
    start = time.perf_counter()
    settings = retriever.settings
    opts = {"mode": body.mode, "rerank": body.rerank, "tables": body.tables}
    timings: dict[str, float] = {}

    try:
        t = time.perf_counter()
        hit_lists = [
            retriever.retrieve(
                session, body.question, top_k=settings.max_context_chunks, **opts
            ).hits
        ]
        for fq in build_feature_queries(body.features or {}, body.prediction):
            hit_lists.append(
                retriever.retrieve(session, fq.query, top_k=FEATURE_HITS_PER_QUERY, **opts).hits
            )
        hits = interleave(hit_lists, settings.max_context_chunks)
        timings["retrieval_ms"] = (time.perf_counter() - t) * 1000
        comparisons = compare_features(session, body.features or {})
        papers = {p.paper_id: p for p in session.scalars(select(Paper))}
    except OperationalError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable") from exc

    t = time.perf_counter()
    try:
        result = generate_answer(
            llm,
            body.question,
            hits,
            comparisons,
            features=body.features,
            speed=body.prediction,
            min_rerank_score=settings.min_rerank_score,
        )
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc
    timings["generation_ms"] = (time.perf_counter() - t) * 1000

    by_label = {s.label: s for s in sources_from_hits(hits)}
    comp_by_label = {c.citation: c for c in comparisons}
    citations = []
    for label in result.citations:
        if label in by_label:
            s = by_label[label]
            pid, page, table, snippet = s.paper_id, s.page, s.table_no, s.texts[0][:300]
        else:
            c = comp_by_label[label]
            pid, page, table = c.paper_id, c.page, c.table_no
            snippet = f"{c.population or 'group'}: {c.norm_text}"
        paper = papers.get(pid)
        citations.append(
            CitationOut(
                label=label,
                paper_id=pid,
                title=paper.title if paper else pid,
                page=page,
                table=table,
                snippet=snippet,
                source_url=paper.source_url if paper else None,
            )
        )

    return AskResponse(
        answer=result.answer,
        citations=citations,
        insufficient_evidence=result.insufficient_evidence,
        reason=result.reason,
        norms_comparison=[NormComparisonOut.from_comparison(c) for c in comparisons],
        model=llm.model,
        latency_ms=(time.perf_counter() - start) * 1000,
        timings_ms=timings,
    )
