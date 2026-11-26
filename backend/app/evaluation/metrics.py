"""Evaluation metrics (pure functions, unit-tested).

Retrieval metrics work at paper level over the ranked *chunks*: a question's
gold papers are the papers that contain its answer, and a chunk counts as a hit
if it comes from one of them.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def recall_at_k(ranked_papers: Sequence[str], gold: Sequence[str], k: int) -> float:
    """Share of gold papers appearing among the top-k ranked chunks' papers."""
    if not gold:
        raise ValueError("recall is undefined without gold papers")
    return len(set(ranked_papers[:k]) & set(gold)) / len(set(gold))


def reciprocal_rank(ranked_papers: Sequence[str], gold: Sequence[str]) -> float:
    """1 / rank of the first chunk from a gold paper (0 if none)."""
    gold_set = set(gold)
    for rank, paper in enumerate(ranked_papers, start=1):
        if paper in gold_set:
            return 1.0 / rank
    return 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """p-th percentile with linear interpolation (p in 0-100)."""
    if not values:
        raise ValueError("no values")
    return float(np.percentile(np.asarray(values, dtype=float), p))


def abstention_scores(abstained: Sequence[bool], unanswerable: Sequence[bool]) -> dict[str, float]:
    """Precision/recall/F1 of abstaining, with unanswerable questions as the positive class.

    Precision: of the questions the system refused, how many were truly unanswerable.
    Recall: of the unanswerable questions, how many the system refused.
    """
    tp = sum(a and u for a, u in zip(abstained, unanswerable, strict=True))
    fp = sum(a and not u for a, u in zip(abstained, unanswerable, strict=True))
    fn = sum(u and not a for a, u in zip(abstained, unanswerable, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def best_gate_threshold(
    best_scores: Sequence[float], unanswerable: Sequence[bool]
) -> dict[str, float] | None:
    """Threshold t for "abstain if best reranker score < t" that maximizes abstention F1.

    Candidates are midpoints between sorted distinct scores. With few questions this
    overfits; report it as indicative and prefer a held-out split when the set grows.
    """
    if not any(unanswerable) or all(unanswerable):
        return None
    scores = sorted(set(best_scores))
    candidates = [(a + b) / 2 for a, b in zip(scores, scores[1:], strict=False)]
    candidates = [scores[0] - 1e-6, *candidates, scores[-1] + 1e-6]
    best: dict[str, float] | None = None
    for t in candidates:
        s = abstention_scores([x < t for x in best_scores], unanswerable)
        if best is None or s["f1"] > best["f1"]:
            best = {"threshold": t, **s}
    return best
