"""Cited answer generation with citation validation and insufficient-evidence mode.

Flow:
1. Gate: no sources at all, or (when calibrated) a best reranker score below
   `min_rerank_score` -> "insufficient evidence" without calling the LLM.
2. Ask the LLM with labelled sources and code-computed norm comparisons.
3. Validate: every [P..., p...] label in the answer must be one we provided,
   and there must be at least one. Otherwise retry once with specific feedback.
4. Still invalid -> "insufficient evidence". Unverified citations never reach
   the user. Uncited sentences are counted (an evaluation signal), not blocked:
   whether a sentence is a factual claim cannot be judged reliably by rules.
"""

from __future__ import annotations

import re
from collections import OrderedDict
from dataclasses import dataclass, field

from app.rag.chunking import split_sentences
from app.rag.llm import LLMClient, Message
from app.rag.norms import NormComparison, citation_label
from app.rag.prompts import INSUFFICIENT_MARKER, SYSTEM_PROMPT, Source, build_user_prompt
from app.rag.retriever import Hit

CITATION = re.compile(r"\[P\d{3}, p\.\d+(?:, Table [A-Za-z]?\d+|, Table [IVX]+)?\]")
INSUFFICIENT_ANSWER = (
    "Insufficient evidence: the indexed papers do not contain enough information to "
    "answer this question."
)


@dataclass
class GeneratedAnswer:
    """Result of generation (validated)."""

    answer: str
    citations: list[str]  # labels, in order of first use
    insufficient_evidence: bool
    reason: str  # "answered" | "no_sources" | "low_relevance" | "model_abstained" | ...
    attempts: int = 0
    uncited_sentences: int = 0
    invalid_citations: list[str] = field(default_factory=list)


def sources_from_hits(hits: list[Hit]) -> list[Source]:
    """Group hits by citation label (same paper/page/table), keeping rank order."""
    groups: OrderedDict[str, list[Hit]] = OrderedDict()
    for h in hits:
        c = h.chunk
        groups.setdefault(citation_label(c.paper_id, c.page, c.table_no), []).append(h)
    return [
        Source(
            label=label,
            paper_id=hs[0].chunk.paper_id,
            title=hs[0].chunk.title,
            page=hs[0].chunk.page,
            table_no=hs[0].chunk.table_no,
            section=hs[0].chunk.section,
            texts=tuple(h.chunk.text for h in hs),
        )
        for label, hs in groups.items()
    ]


def check_citations(answer: str, allowed: set[str]) -> tuple[list[str], list[str], int]:
    """Return (valid labels in order, invalid labels, number of uncited sentences)."""
    found = CITATION.findall(answer)
    valid = list(dict.fromkeys(c for c in found if c in allowed))
    invalid = list(dict.fromkeys(c for c in found if c not in allowed))
    # Labels that look like citations but are malformed ("[P6, page 5]") count as invalid.
    loose = re.findall(r"\[P\d+[^\]]*\]", answer)
    invalid += [c for c in loose if not CITATION.fullmatch(c) and c not in invalid]
    uncited = sum(1 for s in split_sentences(answer) if not CITATION.search(s))
    return valid, invalid, uncited


def _is_abstention(text: str) -> bool:
    return INSUFFICIENT_MARKER in text


def generate_answer(
    llm: LLMClient,
    question: str,
    hits: list[Hit],
    comparisons: list[NormComparison],
    features: dict[str, float] | None = None,
    speed: str | None = None,
    min_rerank_score: float | None = None,
) -> GeneratedAnswer:
    """Generate a validated, cited answer (or an explicit insufficient-evidence result)."""
    if not hits and not comparisons:
        return GeneratedAnswer(INSUFFICIENT_ANSWER, [], True, "no_sources")
    best = max((h.rerank_score for h in hits if h.rerank_score is not None), default=None)
    if min_rerank_score is not None and not comparisons and best is not None:
        if best < min_rerank_score:
            return GeneratedAnswer(INSUFFICIENT_ANSWER, [], True, "low_relevance")

    sources = sources_from_hits(hits)
    allowed = {s.label for s in sources} | {c.citation for c in comparisons}
    messages = [
        Message("system", SYSTEM_PROMPT),
        Message("user", build_user_prompt(question, sources, comparisons, features, speed)),
    ]

    invalid: list[str] = []
    for attempt in (1, 2):
        reply = llm.complete(messages).strip()
        if _is_abstention(reply):
            return GeneratedAnswer(INSUFFICIENT_ANSWER, [], True, "model_abstained", attempt)
        valid, invalid, uncited = check_citations(reply, allowed)
        if valid and not invalid:
            return GeneratedAnswer(reply, valid, False, "answered", attempt, uncited)
        problem = (
            f"These citations are not among the provided labels: {', '.join(invalid)}."
            if invalid
            else "The answer has no citations."
        )
        messages += [
            Message("assistant", reply),
            Message(
                "user",
                f"{problem} Rewrite the answer citing only these exact labels: "
                f"{', '.join(sorted(allowed))}. If they do not support an answer, reply "
                f"{INSUFFICIENT_MARKER}.",
            ),
        ]
    return GeneratedAnswer(
        INSUFFICIENT_ANSWER, [], True, "citation_check_failed", 2, invalid_citations=invalid
    )
