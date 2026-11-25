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


def test_built_frontend_is_served_when_configured(tmp_path, monkeypatch) -> None:
    from app.config import get_settings
    from app.main import create_app

    (tmp_path / "index.html").write_text("<html>GaitRAG UI</html>")
    monkeypatch.setenv("FRONTEND_DIST", str(tmp_path))
    get_settings.cache_clear()
    try:
        c = TestClient(create_app())
        assert "GaitRAG UI" in c.get("/").text
        assert c.get("/samples").status_code == 200  # API routes still win
    finally:
        get_settings.cache_clear()
