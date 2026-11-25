"""Walking-speed classifier: model bundle persistence and single-recording inference."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier

from app.ml.features import GaitFeatures, extract_gait_features, model_features
from app.ml.normalization import zscore_by_group

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "speed_rf.joblib"


@dataclass
class ModelBundle:
    """Everything needed to reproduce inference: model + how its inputs were built."""

    model: RandomForestClassifier
    feature_names: list[str]
    normalization: str  # group column used by zscore_by_group at training time
    trained_on: str  # e.g. "synthetic"; surfaced in API responses
    sklearn_version: str = field(default_factory=lambda: sklearn.__version__)
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Prediction:
    """Classifier output for one recording."""

    label: str
    confidence: float
    probabilities: dict[str, float]
    gait: GaitFeatures


def save_bundle(bundle: ModelBundle, path: Path) -> None:
    """Persist a model bundle with joblib (compressed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: Path = DEFAULT_MODEL_PATH) -> ModelBundle:
    """Load a model bundle, warning if the scikit-learn version differs from training."""
    bundle: ModelBundle = joblib.load(path)
    if bundle.sklearn_version != sklearn.__version__:
        warnings.warn(
            f"Model trained with scikit-learn {bundle.sklearn_version}, "
            f"running {sklearn.__version__}; predictions may differ.",
            stacklevel=2,
        )
    return bundle


def feature_matrix(rows: list[dict[str, float]], feature_names: list[str]) -> np.ndarray:
    """Stack feature dicts into a matrix in a fixed column order."""
    return pd.DataFrame(rows)[feature_names].to_numpy(dtype=float)


def predict_recording(bundle: ModelBundle, raw: pd.DataFrame) -> Prediction:
    """Classify walking speed for one validated recording.

    Raises:
        GaitEventError: if too few strides are detected.
    """
    gait = extract_gait_features(raw)
    normalized = zscore_by_group(raw, bundle.normalization)
    x = feature_matrix([model_features(raw, normalized, gait)], bundle.feature_names)
    proba = bundle.model.predict_proba(x)[0]
    classes = [str(c) for c in bundle.model.classes_]
    best = int(np.argmax(proba))
    return Prediction(
        label=classes[best],
        confidence=float(proba[best]),
        probabilities={c: float(p) for c, p in zip(classes, proba, strict=True)},
        gait=gait,
    )
