"""Tests for POST /analyze."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.constants import DISCLAIMER
from app.main import app
from tests.conftest import make_trial

SAMPLES = Path(__file__).resolve().parents[2] / "data" / "samples"

client = TestClient(app)


def _post_csv(content: bytes, name: str = "trial.csv"):
    return client.post("/analyze", files={"file": (name, content, "text/csv")})


@pytest.fixture
def override_settings():
    """Temporarily replace app settings."""

    def _apply(**kwargs) -> None:
        app.dependency_overrides[get_settings] = lambda: Settings(**kwargs)

    yield _apply
    app.dependency_overrides.clear()


@pytest.mark.parametrize("speed", ["slow", "normal", "fast"])
def test_sample_recordings_are_analyzed(speed: str) -> None:
    response = _post_csv((SAMPLES / f"synthetic_{speed}.csv").read_bytes())
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in {"slow", "normal", "fast"}
    assert 0 <= body["confidence"] <= 1
    assert body["features"]["cadence_steps_per_min"] > 0
    assert body["norms_comparison"] == []
    assert body["model_info"]["trained_on"] == "synthetic"
    assert body["disclaimer"] == DISCLAIMER


def test_missing_columns_return_422() -> None:
    df, _ = make_trial("normal")
    response = _post_csv(df.drop(columns=["fsr_r_toe"]).to_csv(index=False).encode())
    assert response.status_code == 422
    assert "fsr_r_toe" in response.json()["detail"]


def test_no_walking_returns_422() -> None:
    df, _ = make_trial("normal")
    df[df.filter(like="fsr_").columns] = 0.0
    response = _post_csv(df.to_csv(index=False).encode())
    assert response.status_code == 422
    assert "strides" in response.json()["detail"]


def test_garbage_file_returns_422() -> None:
    assert _post_csv(b"\x00\x01not a csv").status_code == 422


def test_oversized_upload_returns_413(override_settings) -> None:
    override_settings(max_upload_mb=0.01)  # ~10 KB
    content = (SAMPLES / "synthetic_normal.csv").read_bytes()
    assert _post_csv(content).status_code == 413


def test_missing_model_returns_503(override_settings, tmp_path: Path) -> None:
    override_settings(speed_model_path=tmp_path / "nope.joblib")
    content = (SAMPLES / "synthetic_normal.csv").read_bytes()
    assert _post_csv(content).status_code == 503


def test_analyze_fills_norms_comparison_from_approved_norms(db_session) -> None:
    from app.db.models import Norm, Paper
    from app.db.session import get_session

    db_session.add(Paper(paper_id="P003", title="t", licence="CC-BY", source_url="u", pdf_url="u"))
    db_session.flush()
    db_session.add(
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
    db_session.flush()
    app.dependency_overrides[get_session] = lambda: db_session
    try:
        body = _post_csv((SAMPLES / "synthetic_normal.csv").read_bytes()).json()
    finally:
        app.dependency_overrides.clear()
    (comp,) = body["norms_comparison"]
    assert comp["feature"] == "cadence_steps_per_min" and comp["citation"] == "[P003, p.6, Table 4]"
    assert comp["z_score"] == pytest.approx((comp["value"] - 109.88) / 6.46)
