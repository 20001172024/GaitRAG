"""POST /analyze: classify walking speed and extract gait features from one recording.

Privacy: the upload is read into memory, processed, and discarded. Nothing is
written to disk or logged.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.api.schemas import NormComparisonOut
from app.config import Settings, get_settings
from app.constants import DISCLAIMER
from app.db.session import get_session
from app.ml.classifier import DEFAULT_MODEL_PATH, ModelBundle, load_bundle, predict_recording
from app.ml.gait_events import GaitEventError
from app.ml.schema import SchemaError, read_recording
from app.rag.norms import compare_features

router = APIRouter(tags=["analysis"])

READ_CHUNK_BYTES = 1024 * 1024


class GaitFeaturesOut(BaseModel):
    """Interpretable gait features. Units are in the field names."""

    cadence_steps_per_min: float
    stride_time_s: float
    stride_time_cv_pct: float = Field(description="Coefficient of variation of stride time")
    stance_pct_left: float
    stance_pct_right: float
    swing_pct_left: float
    swing_pct_right: float
    stance_swing_ratio: float
    asymmetry_stance_time_pct: float = Field(description="Symmetry index, 0 = symmetric")
    asymmetry_peak_pressure_pct: float = Field(description="Symmetry index, 0 = symmetric")
    n_strides_left: int
    n_strides_right: int


class ModelInfo(BaseModel):
    """Provenance of the classifier that produced the prediction."""

    trained_on: str = Field(description='"synthetic" for the public demo model')
    normalization: str
    sklearn_version: str


class AnalyzeResponse(BaseModel):
    """Response of POST /analyze."""

    prediction: Literal["slow", "normal", "fast"]
    confidence: float = Field(ge=0, le=1, description="Probability of the predicted class")
    probabilities: dict[str, float]
    features: GaitFeaturesOut
    norms_comparison: list[NormComparisonOut] = Field(
        default_factory=list,
        description="Comparison with approved published norms (empty if none are approved "
        "for these features, or the database is unavailable).",
    )
    model_info: ModelInfo
    disclaimer: str = DISCLAIMER


@lru_cache(maxsize=4)
def _load_cached(path: str) -> ModelBundle:
    return load_bundle(path)


def get_model(settings: Annotated[Settings, Depends(get_settings)]) -> ModelBundle:
    """Load (once) the speed classifier configured in settings."""
    path = settings.speed_model_path or DEFAULT_MODEL_PATH
    try:
        return _load_cached(str(path))
    except FileNotFoundError as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Speed model not found. Train it with scripts/train_model.py.",
        ) from exc


def _read_limited(file: UploadFile, max_bytes: int) -> bytes:
    """Read an upload, refusing files above `max_bytes` without loading all of it."""
    chunks, size = [], 0
    while chunk := file.file.read(READ_CHUNK_BYTES):
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(
                status.HTTP_413_CONTENT_TOO_LARGE,
                f"File exceeds {max_bytes // (1024 * 1024)} MB",
            )
        chunks.append(chunk)
    return b"".join(chunks)


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze(
    file: Annotated[UploadFile, File(description="One walking trial as CSV (40 Hz)")],
    model: Annotated[ModelBundle, Depends(get_model)],
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[Session, Depends(get_session)],
) -> AnalyzeResponse:
    """Classify walking speed, return gait features and their comparison with norms.

    Plain `def` (not `async`): the work is CPU-bound, so FastAPI runs it in a
    thread pool instead of blocking the event loop.
    """
    content = _read_limited(file, int(settings.max_upload_mb * 1024 * 1024))
    try:
        recording = read_recording(content)
        pred = predict_recording(model, recording)
    except (SchemaError, GaitEventError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    features = pred.gait.to_dict()
    try:
        comparisons = compare_features(session, features)
    except SQLAlchemyError:
        comparisons = []  # analysis still works without the paper database

    return AnalyzeResponse(
        prediction=pred.label,
        confidence=pred.confidence,
        probabilities=pred.probabilities,
        features=GaitFeaturesOut(**features),
        norms_comparison=[NormComparisonOut.from_comparison(c) for c in comparisons],
        model_info=ModelInfo(
            trained_on=model.trained_on,
            normalization=model.normalization,
            sklearn_version=model.sklearn_version,
        ),
    )
