"""POST /ask: evidence-grounded, cited answer to a question (optionally about a recording)."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.api.schemas import NormComparisonOut
from app.api.search import Mode, get_retriever
from app.config import get_settings
from app.constants import DISCLAIMER
from app.db.session import get_session
from app.rag.llm import LLMClient, LLMError, build_llm
from app.rag.pipeline import answer_question
from app.rag.retriever import Retriever

router = APIRouter(tags=["generation"])


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
    try:
        result = answer_question(
            session,
            retriever,
            llm,
            body.question,
            body.features,
            body.prediction,
            mode=body.mode,
            rerank=body.rerank,
            tables=body.tables,
        )
    except OperationalError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable") from exc
    except LLMError as exc:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, str(exc)) from exc

    g = result.generated
    return AskResponse(
        answer=g.answer,
        citations=[CitationOut(**vars(c)) for c in result.citations],
        insufficient_evidence=g.insufficient_evidence,
        reason=g.reason,
        norms_comparison=[NormComparisonOut.from_comparison(c) for c in result.comparisons],
        model=llm.model,
        latency_ms=(time.perf_counter() - start) * 1000,
        timings_ms=result.timings_ms,
    )
