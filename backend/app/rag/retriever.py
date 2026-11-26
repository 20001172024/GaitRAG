"""Hybrid retrieval: BM25 + pgvector, fused with RRF, optionally reranked.

Every stage is switchable (mode, rerank, tables) so the ablation study can
compare vector-only, BM25-only, hybrid, hybrid + reranker, with and without
table chunks, on the same code path.

- Vector: cosine distance on normalized bge embeddings via the HNSW index.
  pgvector's HNSW search only explores `hnsw.ef_search` candidates (default
  40), so a LIMIT above that silently returns fewer rows. It is raised per query.
- BM25: in memory (rank_bm25). Postgres `ts_rank` is not BM25 (no IDF, no
  term-frequency saturation); with ~1k chunks the index builds in about a second.
  Documents are indexed with the same "title > section" header used for embeddings.
- Fusion: Reciprocal Rank Fusion, score = sum over rankings of 1 / (k + rank),
  k = 60 (Cormack, Clarke & Buettcher, SIGIR 2009). BM25 scores and cosine
  similarities are on different scales; RRF uses ranks only, so it needs no
  score normalization or weight tuning.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.config import Settings
from app.db.models import Chunk, Paper
from app.rag.embeddings import Embedder, passage_text
from app.rag.reranker import Reranker

# Small English stop-word list: frequent words that carry no topic.
_STOPWORDS = frozenset(
    """a an and are as at be been but by can did do does for from had has have how in into
    is it its may more most no not of on or our such than that the their them then there
    these they this those to was we were what when where which while who why will with
    within without would you your""".split()
)
_TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


def tokenize(text_: str) -> list[str]:
    """Lowercase word tokens for BM25. Keeps numbers ("1.05") and compounds ("dual-task")."""
    return [t for t in _TOKEN.findall(text_.lower()) if t not in _STOPWORDS]


def rrf_fuse(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion of ranked ID lists (best first). Returns (id, score), best first."""
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda p: (-p[1], p[0]))


@dataclass(frozen=True)
class ChunkInfo:
    """What a hit needs to be shown and cited."""

    chunk_id: int
    paper_id: str
    title: str
    page: int
    section: str | None
    text: str
    kind: str = "text"  # "text" | "table"
    table_no: str | None = None


@dataclass
class Hit:
    """One retrieved chunk with provenance of its score."""

    chunk: ChunkInfo
    score: float  # final ordering score (RRF, cosine, BM25 or reranker, by mode)
    ranks: dict[str, int] = field(default_factory=dict)  # e.g. {"vector": 3, "bm25": 1}
    rerank_score: float | None = None


@dataclass
class RetrievalResult:
    """Hits plus timing per stage, for latency reporting."""

    hits: list[Hit]
    timings_ms: dict[str, float]


class BM25Index:
    """In-memory BM25 over all chunks."""

    def __init__(self, chunks: list[ChunkInfo]):
        self.chunks = chunks
        docs = [tokenize(passage_text(c.title, c.section, c.text)) for c in chunks]
        self.bm25 = BM25Okapi(docs) if docs else None

    def search(self, query: str, k: int, tables: bool = True) -> list[tuple[int, float]]:
        """Top-k (chunk_id, score) with a positive score; `tables=False` skips table chunks.

        IDF is computed over the whole index either way, so switching tables off
        changes which documents can be returned, not how terms are weighted.
        """
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        allowed = [i for i in range(len(scores)) if tables or self.chunks[i].kind == "text"]
        order = sorted(allowed, key=lambda i: -scores[i])[:k]
        return [(self.chunks[i].chunk_id, float(scores[i])) for i in order if scores[i] > 0]


def load_chunks(session: Session) -> dict[int, ChunkInfo]:
    """All chunks with their paper title, keyed by chunk id."""
    rows = session.execute(
        select(
            Chunk.id,
            Chunk.paper_id,
            Paper.title,
            Chunk.page,
            Chunk.section,
            Chunk.text,
            Chunk.kind,
            Chunk.table_no,
        )
        .join(Paper, Paper.paper_id == Chunk.paper_id)
        .order_by(Chunk.id)
    )
    return {
        r.id: ChunkInfo(r.id, r.paper_id, r.title, r.page, r.section, r.text, r.kind, r.table_no)
        for r in rows
    }


def vector_search(
    session: Session, query_vec: list[float], k: int, ef_search: int, tables: bool = True
) -> list[tuple[int, float]]:
    """Top-k (chunk_id, cosine similarity) from pgvector; `tables=False` skips table chunks."""
    # SET LOCAL lasts for the current transaction only. ef_search must be >= k.
    session.execute(text(f"SET LOCAL hnsw.ef_search = {int(max(ef_search, k))}"))
    distance = Chunk.embedding.cosine_distance(query_vec)
    query = select(Chunk.id, distance.label("d")).order_by(distance).limit(k)
    if not tables:
        # HNSW finds neighbours first and filters after, so a WHERE clause can shrink the
        # result below k. Iterative scan (pgvector >= 0.8) keeps searching until k pass.
        session.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))
        query = query.where(Chunk.kind == "text")
    rows = session.execute(query)
    return [(r.id, 1.0 - float(r.d)) for r in rows]


class Retriever:
    """Configurable retriever. Build once (it caches the BM25 index); reuse per request."""

    def __init__(self, settings: Settings, embedder: Embedder, reranker: Reranker | None):
        self.settings = settings
        self.embedder = embedder
        self.reranker = reranker
        self._chunks: dict[int, ChunkInfo] | None = None
        self._bm25: BM25Index | None = None

    def refresh(self, session: Session) -> None:
        """(Re)load chunks and rebuild BM25, e.g. after ingestion."""
        self._chunks = load_chunks(session)
        self._bm25 = BM25Index(list(self._chunks.values()))

    def retrieve(
        self,
        session: Session,
        query: str,
        top_k: int | None = None,
        mode: str | None = None,
        rerank: bool | None = None,
        tables: bool | None = None,
    ) -> RetrievalResult:
        """Retrieve chunks for `query`. `mode`/`rerank`/`tables`/`top_k` override settings."""
        s = self.settings
        mode = mode or s.retrieval_mode
        rerank = s.rerank if rerank is None else rerank
        tables = s.retrieve_tables if tables is None else tables
        top_k = top_k or s.top_k
        if mode not in ("vector", "bm25", "hybrid"):
            raise ValueError(f"unknown retrieval mode {mode!r}")
        if rerank and self.reranker is None:
            raise ValueError("reranking requested but no reranker is configured")
        if self._chunks is None:
            self.refresh(session)
        assert self._chunks is not None and self._bm25 is not None

        timings: dict[str, float] = {}
        rankings: dict[str, list[tuple[int, float]]] = {}
        if mode in ("vector", "hybrid"):
            t = time.perf_counter()
            vec = self.embedder.embed_query(query).tolist()
            rankings["vector"] = vector_search(
                session, vec, s.vector_candidates, s.hnsw_ef_search, tables
            )
            timings["vector_ms"] = (time.perf_counter() - t) * 1000
        if mode in ("bm25", "hybrid"):
            t = time.perf_counter()
            rankings["bm25"] = self._bm25.search(query, s.bm25_candidates, tables)
            timings["bm25_ms"] = (time.perf_counter() - t) * 1000

        ranks = {
            name: {cid: r for r, (cid, _) in enumerate(lst, start=1)}
            for name, lst in rankings.items()
        }
        if mode == "hybrid":
            fused = rrf_fuse([[cid for cid, _ in lst] for lst in rankings.values()], s.rrf_k)
        else:
            fused = rankings[mode]

        n_keep = s.rerank_candidates if rerank else top_k
        hits = [
            Hit(
                chunk=self._chunks[cid],
                score=score,
                ranks={n: r[cid] for n, r in ranks.items() if cid in r},
            )
            for cid, score in fused[:n_keep]
            if cid in self._chunks
        ]

        if rerank and hits:
            t = time.perf_counter()
            passages = [passage_text(h.chunk.title, h.chunk.section, h.chunk.text) for h in hits]
            for hit, rs in zip(hits, self.reranker.score(query, passages), strict=True):
                hit.rerank_score = rs
                hit.score = rs
            hits.sort(key=lambda h: -h.score)
            timings["rerank_ms"] = (time.perf_counter() - t) * 1000

        return RetrievalResult(hits[:top_k], timings)
