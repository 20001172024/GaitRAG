"""Tests for POST /ask with a scripted LLM (needs TEST_DATABASE_URL)."""

import pytest
from fastapi.testclient import TestClient

from app.api.ask import _llm_or_503, interleave
from app.api.search import get_retriever
from app.config import Settings
from app.constants import DISCLAIMER
from app.db.models import Norm
from app.db.session import get_session
from app.main import app
from app.rag.llm import FakeLLM, LLMError
from app.rag.prompts import INSUFFICIENT_MARKER
from app.rag.reranker import KeywordOverlapReranker
from app.rag.retriever import ChunkInfo, Hit, Retriever


def _client(seeded, llm):
    session, embedder = seeded
    retriever = Retriever(Settings(max_context_chunks=4), embedder, KeywordOverlapReranker())
    app.dependency_overrides[get_session] = lambda: session
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[_llm_or_503] = lambda: llm
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    yield
    app.dependency_overrides.clear()


def test_answer_with_citations(seeded) -> None:
    llm = FakeLLM(
        lambda m: "Stride time variability rose when walking speed decreased [P001, p.1]."
    )
    r = _client(seeded, llm).post("/ask", json={"question": "Does variability depend on speed?"})
    assert r.status_code == 200
    body = r.json()
    assert body["insufficient_evidence"] is False and body["reason"] == "answered"
    (cite,) = body["citations"]
    assert (cite["label"], cite["paper_id"], cite["page"], cite["title"]) == (
        "[P001, p.1]",
        "P001",
        1,
        "Gait variability",
    )
    assert cite["snippet"].startswith("Stride time variability")
    assert body["disclaimer"] == DISCLAIMER and body["model"] == "fake"
    assert {"retrieval_ms", "generation_ms"} <= set(body["timings_ms"])


def test_feature_questions_get_norm_comparisons_and_can_cite_them(seeded) -> None:
    session, _ = seeded
    session.add(
        Norm(
            paper_id="P003",
            page=6,
            table_no="4",
            metric="cadence",
            population="20–29 | M",
            value_text="109.88 (6.46)",
            mean=109.88,
            sd=6.46,
            units="steps/min",
        )
    )
    session.flush()
    llm = FakeLLM(lambda m: "Your cadence is within 1 SD of the group [P003, p.6, Table 4].")
    r = _client(seeded, llm).post(
        "/ask",
        json={
            "question": "Is my cadence typical?",
            "features": {"cadence_steps_per_min": 112.0},
            "prediction": "normal",
        },
    )
    body = r.json()
    assert body["norms_comparison"][0]["citation"] == "[P003, p.6, Table 4]"
    assert body["norms_comparison"][0]["within"] is True
    assert body["citations"][0]["snippet"] == "20–29 | M: 109.88 (6.46)"
    prompt = llm.calls[0][1].content
    assert "Norm comparisons" in prompt and "cadence" in prompt.lower()


def test_insufficient_evidence_is_reported(seeded) -> None:
    llm = FakeLLM(lambda m: INSUFFICIENT_MARKER)
    body = (
        _client(seeded, llm)
        .post("/ask", json={"question": "What is the capital of France?"})
        .json()
    )
    assert body["insufficient_evidence"] is True and body["citations"] == []
    assert body["answer"].startswith("Insufficient evidence")


def test_llm_failure_returns_502(seeded) -> None:
    def fail(messages):
        raise LLMError("LLM returned HTTP 429")

    r = _client(seeded, FakeLLM(fail)).post("/ask", json={"question": "cadence norms?"})
    assert r.status_code == 502


def test_missing_api_key_returns_503(monkeypatch) -> None:
    from app.api import ask as ask_module

    ask_module.get_llm.cache_clear()
    monkeypatch.setattr(ask_module, "get_settings", lambda: Settings(openai_api_key=None))
    r = TestClient(app).post("/ask", json={"question": "cadence norms?"})
    assert r.status_code == 503 and "No API key" in r.json()["detail"]
    ask_module.get_llm.cache_clear()


def test_interleave_round_robin_dedupes_and_caps() -> None:
    def h(i: int) -> Hit:
        return Hit(ChunkInfo(i, "P001", "t", 1, None, str(i)), score=1.0)

    merged = interleave([[h(1), h(2), h(3)], [h(9), h(1)], [h(7)]], limit=4)
    assert [x.chunk.chunk_id for x in merged] == [1, 9, 7, 2]
