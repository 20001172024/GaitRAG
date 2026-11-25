"""Database schema tests (need a pgvector Postgres; see the db_engine fixture)."""

import numpy as np
from sqlalchemy import select, text

from app.db.models import EMBEDDING_DIM, Chunk, Paper


def _unit(v: np.ndarray) -> list[float]:
    return (v / np.linalg.norm(v)).tolist()


def _paper(pid: str = "P001") -> Paper:
    return Paper(
        paper_id=pid,
        title="A test paper",
        licence="CC-BY-4.0",
        source_url="https://example.org",
        pdf_url="https://example.org/p.pdf",
    )


def _chunk(pid: str, idx: int, body: str, emb: list[float]) -> Chunk:
    return Chunk(
        paper_id=pid,
        chunk_index=idx,
        page=1,
        text=body,
        token_count=5,
        content_hash=f"h{idx}",
        embedding=emb,
    )


def test_extension_and_tables_exist(db_session) -> None:
    ext = db_session.execute(text("SELECT extname FROM pg_extension WHERE extname='vector'"))
    assert ext.scalar() == "vector"
    tables = db_session.execute(
        text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
    ).scalars()
    assert {"papers", "chunks", "paper_tables", "norms"} <= set(tables)


def test_cosine_search_orders_by_similarity(db_session) -> None:
    rng = np.random.default_rng(0)
    query = rng.normal(size=EMBEDDING_DIM)
    near = query + 0.1 * rng.normal(size=EMBEDDING_DIM)
    far = rng.normal(size=EMBEDDING_DIM)
    db_session.add(_paper())
    db_session.add_all(
        [_chunk("P001", 0, "far text", _unit(far)), _chunk("P001", 1, "near text", _unit(near))]
    )
    db_session.flush()
    rows = db_session.scalars(
        select(Chunk).order_by(Chunk.embedding.cosine_distance(_unit(query))).limit(2)
    ).all()
    assert [r.text for r in rows] == ["near text", "far text"]


def test_tsv_is_generated_for_keyword_search(db_session) -> None:
    db_session.add(_paper())
    db_session.add(_chunk("P001", 0, "Stride time variability in walking", [0.0] * EMBEDDING_DIM))
    db_session.flush()
    hit = db_session.execute(
        text("SELECT count(*) FROM chunks WHERE tsv @@ plainto_tsquery('english', 'strides')")
    ).scalar()
    assert hit == 1  # stemming: "strides" matches "Stride"


def test_deleting_paper_cascades_to_chunks(db_session) -> None:
    paper = _paper()
    db_session.add(paper)
    db_session.add(_chunk("P001", 0, "x", [0.0] * EMBEDDING_DIM))
    db_session.flush()
    db_session.delete(paper)
    db_session.flush()
    assert db_session.scalar(select(Chunk).limit(1)) is None
