"""Tests for GET /papers (needs TEST_DATABASE_URL)."""

from fastapi.testclient import TestClient

from app.db.models import EMBEDDING_DIM, Chunk, Paper
from app.db.session import get_session
from app.main import app


def test_papers_lists_attribution_and_chunk_counts(db_session) -> None:
    db_session.add_all(
        [
            Paper(
                paper_id="P002",
                title="Second",
                authors="Roe R",
                year=2021,
                licence="CC-BY",
                source_url="https://example.org/2",
                pdf_url="x",
            ),
            Paper(
                paper_id="P001",
                title="First",
                authors="Doe J",
                year=2020,
                licence="CC-BY",
                source_url="https://example.org/1",
                pdf_url="x",
            ),
        ]
    )
    db_session.flush()
    db_session.add(
        Chunk(
            paper_id="P001",
            chunk_index=0,
            page=1,
            text="t",
            token_count=1,
            content_hash="h",
            embedding=[0.0] * EMBEDDING_DIM,
        )
    )
    db_session.flush()

    app.dependency_overrides[get_session] = lambda: db_session
    try:
        body = TestClient(app).get("/papers").json()
    finally:
        app.dependency_overrides.clear()

    assert [(p["paper_id"], p["n_chunks"]) for p in body] == [("P001", 1), ("P002", 0)]
    assert body[0]["licence"] == "CC-BY"
    assert body[0]["source_url"] == "https://example.org/1"
    assert "pdf_url" not in body[0]
