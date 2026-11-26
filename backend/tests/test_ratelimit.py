"""Tests for the /ask rate limiter."""

from fastapi.testclient import TestClient

from app.api import ratelimit
from app.api.ratelimit import RateLimiter, client_key


def test_per_client_limit_uses_a_sliding_hour() -> None:
    rl = RateLimiter(per_client_per_hour=2, global_per_day=0)
    assert rl.check("a", now=0) is None
    assert rl.check("a", now=10) is None
    assert "last hour" in rl.check("a", now=20)
    assert rl.check("b", now=20) is None  # other clients unaffected
    assert rl.check("a", now=3601) is None  # first request has left the window


def test_global_daily_budget() -> None:
    rl = RateLimiter(per_client_per_hour=0, global_per_day=2)
    assert rl.check("a", now=0) is None and rl.check("b", now=1) is None
    assert "daily" in rl.check("c", now=2)
    assert rl.check("c", now=86401) is None


def test_refused_requests_do_not_consume_budget() -> None:
    rl = RateLimiter(per_client_per_hour=1, global_per_day=2)
    rl.check("a", now=0)
    rl.check("a", now=1)  # refused
    assert rl.check("b", now=2) is None  # global budget still has room


def test_client_key_prefers_forwarded_for() -> None:
    from starlette.requests import Request

    scope = {
        "type": "http",
        "headers": [(b"x-forwarded-for", b"203.0.113.7, 10.0.0.1")],
        "client": ("10.0.0.1", 1),
    }
    assert client_key(Request(scope)) == "203.0.113.7"


def test_ask_returns_429_when_limited(monkeypatch) -> None:
    from app.api.ask import _llm_or_503
    from app.main import app
    from app.rag.llm import FakeLLM

    monkeypatch.setattr(ratelimit, "_limiter", RateLimiter(per_client_per_hour=0, global_per_day=1))
    ratelimit.get_limiter().check("someone-else")  # use up the daily budget
    app.dependency_overrides[_llm_or_503] = lambda: FakeLLM(lambda m: "x")
    try:
        r = TestClient(app).post("/ask", json={"question": "cadence norms?"})
    finally:
        app.dependency_overrides.clear()
    assert r.status_code == 429 and "daily" in r.json()["detail"]
