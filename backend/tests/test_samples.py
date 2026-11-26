"""Tests for the demo sample endpoints."""

from fastapi.testclient import TestClient

from app.main import app
from app.ml.schema import read_recording

client = TestClient(app)


def test_list_samples() -> None:
    names = {s["name"] for s in client.get("/samples").json()}
    assert names == {"synthetic_slow", "synthetic_normal", "synthetic_fast"}


def test_download_sample_is_a_valid_recording() -> None:
    r = client.get("/samples/synthetic_normal.csv")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert len(read_recording(r.content)) > 0


def test_unknown_or_traversal_names_are_404() -> None:
    assert client.get("/samples/nope.csv").status_code == 404
    assert client.get("/samples/..%2F..%2Fbackend%2Fpyproject.csv").status_code == 404
