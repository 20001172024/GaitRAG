"""Tests for BM25, RRF, the hybrid retriever and the query builder."""

import pytest

from app.config import Settings
from app.db.models import Chunk, Paper
from app.rag.embeddings import HashingEmbedder, passage_text
from app.rag.reranker import KeywordOverlapReranker
from app.rag.retriever import BM25Index, ChunkInfo, Retriever, rrf_fuse, tokenize


def test_tokenize_keeps_numbers_and_compounds_drops_stopwords() -> None:
    assert tokenize("The CV of stride time was 1.05 in dual-task walking.") == [
        "cv",
        "stride",
        "time",
        "1.05",
        "dual-task",
        "walking",
    ]


def test_rrf_rewards_agreement_between_rankings() -> None:
    fused = rrf_fuse([[1, 2, 3], [4, 2, 5]], k=60)
    ids = [doc for doc, _ in fused]
    assert ids[0] == 2  # 2nd in both rankings beats 1st in only one
    assert fused[0][1] == pytest.approx(2 / 62)
    assert set(ids) == {1, 2, 3, 4, 5}


def test_rrf_single_ranking_preserves_order() -> None:
    assert [d for d, _ in rrf_fuse([[5, 9, 7]])] == [5, 9, 7]


def _info(cid: int, text: str, title: str = "Paper") -> ChunkInfo:
    return ChunkInfo(cid, "P001", title, 1, "Results", text)


def test_bm25_ranks_matching_chunk_first_and_skips_zero_scores() -> None:
    index = BM25Index(
        [
            _info(1, "Plantar pressure insole sensors were calibrated."),
            _info(2, "Stride time variability increased at slow speed."),
            _info(3, "Participants were recruited from a clinic."),
        ]
    )
    hits = index.search("stride time variability", k=5)
    assert hits[0][0] == 2
    assert all(score > 0 for _, score in hits)


CORPUS = [
    ("P001", "Gait variability", "Stride time variability (CV) rose when walking speed decreased."),
    ("P001", "Gait variability", "Variability was measured over forty strides per subject."),
    ("P002", "Insole design", "The insole has sixteen piezoresistive pressure sensors."),
    ("P002", "Insole design", "Heel strike was detected when heel pressure exceeded a threshold."),
    ("P003", "Cadence norms", "Cadence in healthy adults was about 110 steps per minute."),
]


@pytest.fixture
def seeded(db_session):
    embedder = HashingEmbedder()
    for pid in {p for p, _, _ in CORPUS}:
        title = next(t for p, t, _ in CORPUS if p == pid)
        db_session.add(
            Paper(paper_id=pid, title=title, licence="CC-BY", source_url="u", pdf_url="u")
        )
    db_session.flush()
    for i, (pid, title, body) in enumerate(CORPUS):
        emb = embedder.embed_passages([passage_text(title, "Results", body)])[0]
        db_session.add(
            Chunk(
                paper_id=pid,
                chunk_index=i,
                page=i + 1,
                section="Results",
                text=body,
                token_count=len(body.split()),
                content_hash=str(i),
                embedding=emb.tolist(),
            )
        )
    db_session.flush()
    return db_session, embedder


def _retriever(embedder, **settings) -> Retriever:
    return Retriever(Settings(**settings), embedder, KeywordOverlapReranker())


@pytest.mark.parametrize("mode", ["vector", "bm25", "hybrid"])
def test_every_mode_finds_the_relevant_chunk(seeded, mode: str) -> None:
    session, embedder = seeded
    result = _retriever(embedder, retrieval_mode=mode, rerank=False).retrieve(
        session, "how does stride time variability change with walking speed", top_k=3
    )
    assert result.hits[0].chunk.text.startswith("Stride time variability")
    assert result.hits[0].chunk.paper_id == "P001"
    assert result.hits[0].chunk.page == 1


def test_hybrid_records_ranks_from_both_retrievers(seeded) -> None:
    session, embedder = seeded
    result = _retriever(embedder, retrieval_mode="hybrid", rerank=False).retrieve(
        session, "cadence steps per minute healthy adults", top_k=2
    )
    top = result.hits[0]
    assert top.chunk.paper_id == "P003"
    assert set(top.ranks) == {"vector", "bm25"}
    assert {"vector_ms", "bm25_ms"} <= set(result.timings_ms)


def test_rerank_reorders_and_respects_top_k(seeded) -> None:
    session, embedder = seeded
    result = _retriever(embedder, rerank=True, rerank_candidates=5).retrieve(
        session, "heel strike threshold pressure", top_k=2
    )
    assert len(result.hits) == 2
    assert result.hits[0].rerank_score is not None
    assert result.hits[0].rerank_score >= result.hits[1].rerank_score
    assert "rerank_ms" in result.timings_ms


def test_vector_limit_above_default_ef_search_returns_all_rows(db_session) -> None:
    """With the HNSW index, LIMIT 50 returns only 40 rows at the default ef_search=40."""
    from sqlalchemy import select, text

    from app.rag.retriever import vector_search

    embedder = HashingEmbedder()
    db_session.add(Paper(paper_id="P001", title="t", licence="CC-BY", source_url="u", pdf_url="u"))
    db_session.flush()
    for i in range(60):
        body = f"walking trial number {i} with stride and cadence {i % 7}"
        db_session.add(
            Chunk(
                paper_id="P001",
                chunk_index=i,
                page=1,
                text=body,
                token_count=9,
                content_hash=str(i),
                embedding=embedder.embed_passages([body])[0].tolist(),
            )
        )
    db_session.flush()
    db_session.execute(text("SET LOCAL enable_seqscan = off"))  # force the HNSW index
    query = embedder.embed_query("walking stride cadence").tolist()

    raw = db_session.execute(
        select(Chunk.id).order_by(Chunk.embedding.cosine_distance(query)).limit(50)
    ).all()
    assert len(raw) == 40  # the trap: plain LIMIT 50 at default ef_search
    # vector_search raises ef_search to at least k, even if configured lower.
    assert len(vector_search(db_session, query, k=50, ef_search=10)) == 50


def test_overrides_are_validated(seeded) -> None:
    session, embedder = seeded
    with pytest.raises(ValueError, match="mode"):
        _retriever(embedder).retrieve(session, "x", mode="nope")
    with pytest.raises(ValueError, match="reranker"):
        Retriever(Settings(), embedder, None).retrieve(session, "x", rerank=True)
