"""Cross-encoder reranking.

A bi-encoder (the embedding model) embeds query and passage separately, so it
cannot compare their words directly. A cross-encoder reads the pair together and
scores relevance with full attention between them: more accurate, but one model
pass per pair. So it only rescores the few fused candidates, never the corpus.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Protocol


class Reranker(Protocol):
    """Scores (query, passage) pairs; higher = more relevant."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        """One relevance score per passage."""
        ...


class CrossEncoderReranker:
    """sentence-transformers CrossEncoder, e.g. BAAI/bge-reranker-base (CPU)."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", max_length: int = 512):
        from sentence_transformers import CrossEncoder  # heavy import, load lazily

        self.model = CrossEncoder(model_name, device="cpu", max_length=max_length)

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance logits for each (query, passage) pair."""
        if not passages:
            return []
        return [float(s) for s in self.model.predict([(query, p) for p in passages])]


class KeywordOverlapReranker:
    """Model-free stand-in for tests: fraction of query words found in the passage."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Share of distinct query words present in each passage."""
        words = set(re.findall(r"[a-z0-9]+", query.lower()))
        return [
            len(words & set(re.findall(r"[a-z0-9]+", p.lower()))) / max(1, len(words))
            for p in passages
        ]


@lru_cache(maxsize=2)
def get_reranker(model_name: str) -> Reranker:
    """Load (once) the configured cross-encoder."""
    return CrossEncoderReranker(model_name)
