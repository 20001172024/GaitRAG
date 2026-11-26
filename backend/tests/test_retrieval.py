"""Tests for BM25, RRF, the hybrid retriever and the query builder."""

import pytest

from app.config import Settings
from app.db.models import Chunk, Paper
from app.rag.embeddings import HashingEmbedder
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


def _add_table_chunk(session, embedder) -> None:
    body = "Table 3 Reference values\nCadence (steps/min) — Young: 112.0 ± 8.0; Old: 104.5 ± 9.9"
    session.add(
        Chunk(
            paper_id="P003",
            chunk_index=99,
            page=6,
            kind="table",
            table_no="3",
            text=body,
            token_count=20,
            content_hash="t",
            embedding=embedder.embed_passages([body])[0].tolist(),
        )
    )
    session.flush()


@pytest.mark.parametrize("mode", ["vector", "bm25"])
def test_table_chunks_can_be_switched_off(seeded, mode: str) -> None:
    session, embedder = seeded
    _add_table_chunk(session, embedder)
    retriever = _retriever(embedder, retrieval_mode=mode, rerank=False)
    query = "cadence steps per minute reference values young old"
    with_tables = retriever.retrieve(session, query, top_k=5, tables=True).hits
    assert with_tables[0].chunk.kind == "table" and with_tables[0].chunk.table_no == "3"
    without = retriever.retrieve(session, query, top_k=5, tables=False).hits
    assert without and all(h.chunk.kind == "text" for h in without)


def test_filtered_vector_search_still_returns_k_rows(db_session) -> None:
    """With mostly table chunks near the query, the text-only filter must still fill k."""
    from sqlalchemy import text

    from app.rag.retriever import vector_search

    embedder = HashingEmbedder()
    db_session.add(Paper(paper_id="P001", title="t", licence="CC-BY", source_url="u", pdf_url="u"))
    db_session.flush()
    # 60 table rows crowd the query; 40 text rows are eligible for k = 20. The test checks
    # that filtering still fills k (iterative scan), not that approximate HNSW search has
    # perfect recall: with exactly k eligible rows a single HNSW miss returned k - 1.
    for i in range(100):
        kind = "table" if i < 60 else "text"
        body = f"cadence table row {i}" if kind == "table" else f"walking prose {i} cadence"
        db_session.add(
            Chunk(
                paper_id="P001",
                chunk_index=i,
                page=1,
                kind=kind,
                text=body,
                token_count=4,
                content_hash=str(i),
                embedding=embedder.embed_passages([body])[0].tolist(),
            )
        )
    db_session.flush()
    db_session.execute(text("SET LOCAL enable_seqscan = off"))
    q = embedder.embed_query("cadence table row").tolist()
    hits = vector_search(db_session, q, k=20, ef_search=20, tables=False)
    assert len(hits) == 20
