"""Tests for the /health endpoint and app wiring."""

from fastapi.testclient import TestClient

from app import __version__
from app.config import Settings
from app.constants import DISCLAIMER
from app.main import app

client = TestClient(app)


def test_health_reports_db_up(monkeypatch) -> None:
    monkeypatch.setattr("app.api.health.ping", lambda engine: True)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__, "db": "ok"}


def test_health_stays_ok_when_db_is_down(monkeypatch) -> None:
    monkeypatch.setattr("app.api.health.ping", lambda engine: False)
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["db"] == "down"


def test_openapi_description_includes_disclaimer() -> None:
    schema = client.get("/openapi.json").json()
    assert DISCLAIMER in schema["info"]["description"]


def test_cors_origin_list_splits_and_strips() -> None:
    settings = Settings(cors_origins=" http://a.com , http://b.com,, ")
    assert settings.cors_origin_list == ["http://a.com", "http://b.com"]
