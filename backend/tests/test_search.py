"""Tests for POST /search (needs TEST_DATABASE_URL)."""

import pytest
from fastapi.testclient import TestClient

from app.api.search import get_retriever
from app.config import Settings
from app.constants import DISCLAIMER
from app.db.session import get_session
from app.main import app
from app.rag.reranker import KeywordOverlapReranker
from app.rag.retriever import Retriever


@pytest.fixture
def client(seeded):
    session, embedder = seeded
    retriever = Retriever(Settings(top_k=3), embedder, KeywordOverlapReranker())
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_retriever] = lambda: retriever
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_search_by_query_returns_citable_hits(client) -> None:
    r = client.post("/search", json={"query": "stride time variability and walking speed"})
    assert r.status_code == 200
    body = r.json()
    hit = body["results"][0]["hits"][0]
    assert (hit["paper_id"], hit["page"], hit["title"]) == ("P001", 1, "Gait variability")
    assert body["mode"] == "hybrid" and body["rerank"] is True and body["tables"] is True
    assert hit["kind"] == "text" and hit["table_no"] is None
    assert body["disclaimer"] == DISCLAIMER
    assert len(body["results"][0]["hits"]) <= 3


def test_search_by_features_runs_one_query_per_feature(client) -> None:
    r = client.post(
        "/search",
        json={
            "features": {"cadence_steps_per_min": 110.0, "stride_time_cv_pct": 2.0},
            "prediction": "normal",
            "mode": "bm25",
            "rerank": False,
            "top_k": 1,
        },
    )
    body = r.json()
    assert [res["feature"] for res in body["results"]] == ["cadence", "stride_time_variability"]
    assert body["results"][0]["hits"][0]["paper_id"] == "P003"
    assert body["mode"] == "bm25" and body["rerank"] is False
    assert "bm25_ms" in body["timings_ms"] and "vector_ms" not in body["timings_ms"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": "cadence", "features": {"cadence_steps_per_min": 1.0}},
        {"query": "x"},
        {"query": "cadence", "mode": "semantic"},
        {"features": {"unknown_feature": 1.0}},
    ],
)
def test_invalid_requests_are_rejected(client, payload) -> None:
    assert client.post("/search", json=payload).status_code == 422


def test_rerank_unavailable_returns_400(seeded) -> None:
    session, embedder = seeded
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_retriever] = lambda: Retriever(
        Settings(rerank=False), embedder, None
    )
    try:
        r = TestClient(app).post("/search", json={"query": "cadence", "rerank": True})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 400
