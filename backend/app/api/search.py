"""POST /search: retrieval only (no generation).

Used for debugging, for showing retrieved evidence transparently in the demo,
and by the M6 retrieval evaluation. `mode`, `rerank` and `top_k` can be set per
request so ablations run against one server.
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.config import get_settings
from app.constants import DISCLAIMER
from app.db.session import get_session
from app.rag.embeddings import get_embedder
from app.rag.query_builder import build_feature_queries
from app.rag.reranker import get_reranker
from app.rag.retriever import Retriever

router = APIRouter(tags=["retrieval"])

Mode = Literal["vector", "bm25", "hybrid"]


class SearchRequest(BaseModel):
    """Either a free-text `query`, or `features` (+ optional `prediction`)."""

    query: str | None = Field(default=None, min_length=3, max_length=500)
    features: dict[str, float] | None = None
    prediction: Literal["slow", "normal", "fast"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    mode: Mode | None = None
    rerank: bool | None = None

    @model_validator(mode="after")
    def _one_input(self) -> SearchRequest:
        if (self.query is None) == (self.features is None):
            raise ValueError("Provide exactly one of `query` or `features`")
        return self


class HitOut(BaseModel):
    """A retrieved chunk with everything needed to cite it."""

    paper_id: str
    title: str
    page: int
    section: str | None
    text: str
    score: float
    rerank_score: float | None
    ranks: dict[str, int] = Field(description="Rank in each retriever, e.g. {'bm25': 1}")


class QueryResult(BaseModel):
    """Hits for one query (one per feature when searching by features)."""

    feature: str | None
    query: str
    hits: list[HitOut]


class SearchResponse(BaseModel):
    """Response of POST /search."""

    results: list[QueryResult]
    mode: Mode
    rerank: bool
    latency_ms: float
    timings_ms: dict[str, float] = Field(description="Per-stage time, summed over queries")
    disclaimer: str = DISCLAIMER


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """The shared retriever: models load once, the BM25 index is built on first use."""
    settings = get_settings()
    reranker = get_reranker(settings.reranker_model) if settings.rerank else None
    return Retriever(settings, get_embedder(settings.embedding_model), reranker)


@router.post("/search", response_model=SearchResponse)
def search(
    body: SearchRequest,
    session: Annotated[Session, Depends(get_session)],
    retriever: Annotated[Retriever, Depends(get_retriever)],
) -> SearchResponse:
    """Retrieve evidence chunks for a question or for a recording's gait features."""
    start = time.perf_counter()
    mode = body.mode or retriever.settings.retrieval_mode
    rerank = retriever.settings.rerank if body.rerank is None else body.rerank
    if rerank and retriever.reranker is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Reranking is disabled on this server")

    if body.query is not None:
        queries = [(None, body.query)]
    else:
        queries = [
            (q.feature, q.query) for q in build_feature_queries(body.features, body.prediction)
        ]
        if not queries:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "No known features given")

    results, timings = [], {}
    try:
        for feature, query in queries:
            r = retriever.retrieve(session, query, top_k=body.top_k, mode=mode, rerank=rerank)
            for stage, ms in r.timings_ms.items():
                timings[stage] = timings.get(stage, 0.0) + ms
            hits = [
                HitOut(
                    paper_id=h.chunk.paper_id,
                    title=h.chunk.title,
                    page=h.chunk.page,
                    section=h.chunk.section,
                    text=h.chunk.text,
                    score=h.score,
                    rerank_score=h.rerank_score,
                    ranks=h.ranks,
                )
                for h in r.hits
            ]
            results.append(QueryResult(feature=feature, query=query, hits=hits))
    except OperationalError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable") from exc

    return SearchResponse(
        results=results,
        mode=mode,
        rerank=rerank,
        latency_ms=(time.perf_counter() - start) * 1000,
        timings_ms=timings,
    )
