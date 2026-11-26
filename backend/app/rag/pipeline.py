"""The full question-answering pipeline, shared by POST /ask and the evaluation scripts.

Keeping one implementation means evaluation numbers describe exactly what the API does.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Paper
from app.rag.generator import GeneratedAnswer, generate_answer, sources_from_hits
from app.rag.llm import LLMClient
from app.rag.norms import NormComparison, compare_features
from app.rag.query_builder import build_feature_queries
from app.rag.retriever import Hit, Retriever

FEATURE_HITS_PER_QUERY = 2


@dataclass(frozen=True)
class CitedSource:
    """A cited source resolved to paper metadata."""

    label: str
    paper_id: str
    title: str
    page: int
    table: str | None
    snippet: str
    source_url: str | None


@dataclass
class PipelineResult:
    """Everything produced for one question."""

    generated: GeneratedAnswer
    hits: list[Hit]
    comparisons: list[NormComparison]
    citations: list[CitedSource]
    timings_ms: dict[str, float] = field(default_factory=dict)


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


def retrieve_context(
    session: Session,
    retriever: Retriever,
    question: str,
    features: dict[str, float] | None = None,
    prediction: str | None = None,
    **opts: object,
) -> list[Hit]:
    """Hits for the question plus each feature query, merged and capped."""
    limit = retriever.settings.max_context_chunks
    lists = [retriever.retrieve(session, question, top_k=limit, **opts).hits]
    for fq in build_feature_queries(features or {}, prediction):
        lists.append(
            retriever.retrieve(session, fq.query, top_k=FEATURE_HITS_PER_QUERY, **opts).hits
        )
    return interleave(lists, limit)


def answer_question(
    session: Session,
    retriever: Retriever,
    llm: LLMClient,
    question: str,
    features: dict[str, float] | None = None,
    prediction: str | None = None,
    use_norms: bool = True,
    **opts: object,
) -> PipelineResult:
    """Retrieve, compare with norms, generate and resolve citations.

    `opts` are retrieval overrides (mode, rerank, tables). `use_norms=False` skips the
    approved-norms comparison (ablation). Database and LLM errors propagate.
    """
    timings: dict[str, float] = {}
    t = time.perf_counter()
    hits = retrieve_context(session, retriever, question, features, prediction, **opts)
    timings["retrieval_ms"] = (time.perf_counter() - t) * 1000
    comparisons = compare_features(session, features or {}) if use_norms else []
    papers = {p.paper_id: p for p in session.scalars(select(Paper))}

    t = time.perf_counter()
    generated = generate_answer(
        llm,
        question,
        hits,
        comparisons,
        features=features,
        speed=prediction,
        min_rerank_score=retriever.settings.min_rerank_score,
    )
    timings["generation_ms"] = (time.perf_counter() - t) * 1000

    by_label = {s.label: s for s in sources_from_hits(hits)}
    comp_by_label = {c.citation: c for c in comparisons}
    citations = []
    for label in generated.citations:
        if label in by_label:
            s = by_label[label]
            pid, page, table, snippet = s.paper_id, s.page, s.table_no, s.texts[0][:300]
        else:
            c = comp_by_label[label]
            pid, page, table = c.paper_id, c.page, c.table_no
            snippet = f"{c.population or 'group'}: {c.norm_text}"
        paper = papers.get(pid)
        citations.append(
            CitedSource(
                label=label,
                paper_id=pid,
                title=paper.title if paper else pid,
                page=page,
                table=table,
                snippet=snippet,
                source_url=paper.source_url if paper else None,
            )
        )
    return PipelineResult(generated, hits, comparisons, citations, timings)
