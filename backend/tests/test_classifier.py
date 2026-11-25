"""Tests for the speed classifier bundle and inference."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from app.ml.classifier import (
    DEFAULT_MODEL_PATH,
    ModelBundle,
    feature_matrix,
    load_bundle,
    predict_recording,
    save_bundle,
)
from app.ml.features import model_features
from app.ml.normalization import zscore_by_group
from tests.conftest import make_trial


@pytest.fixture(scope="module")
def tiny_bundle() -> ModelBundle:
    """A small model trained on a handful of synthetic trials (fast, test-only)."""
    rows, labels = [], []
    for seed in range(4):
        for speed in ("slow", "normal", "fast"):
            df, _ = make_trial(speed, seed=seed)
            rows.append(model_features(df, zscore_by_group(df, "trial_id")))
            labels.append(speed)
    names = list(rows[0])
    model = RandomForestClassifier(n_estimators=20, random_state=0)
    model.fit(feature_matrix(rows, names), labels)
    return ModelBundle(model, names, normalization="trial_id", trained_on="synthetic")


def test_prediction_has_valid_probabilities(tiny_bundle: ModelBundle) -> None:
    df, _ = make_trial("normal", seed=99)
    pred = predict_recording(tiny_bundle, df)
    assert pred.label in {"slow", "normal", "fast"}
    assert sum(pred.probabilities.values()) == pytest.approx(1.0)
    assert pred.confidence == max(pred.probabilities.values())
    assert pred.gait.n_strides_left >= 3


def test_bundle_round_trips(tiny_bundle: ModelBundle, tmp_path: Path) -> None:
    path = tmp_path / "m.joblib"
    save_bundle(tiny_bundle, path)
    loaded = load_bundle(path)
    assert loaded.feature_names == tiny_bundle.feature_names
    df, _ = make_trial("fast", seed=3)
    x = feature_matrix([model_features(df, zscore_by_group(df, "trial_id"))], loaded.feature_names)
    assert np.allclose(loaded.model.predict_proba(x), tiny_bundle.model.predict_proba(x))


def test_feature_matrix_uses_given_column_order() -> None:
    x = feature_matrix([{"a": 1.0, "b": 2.0}], ["b", "a"])
    assert x.tolist() == [[2.0, 1.0]]


def test_committed_model_matches_current_feature_code() -> None:
    """Fails if feature extraction changed without retraining (scripts/train_model.py)."""
    bundle = load_bundle(DEFAULT_MODEL_PATH)
    df, _ = make_trial("normal")
    current = list(model_features(df, zscore_by_group(df, bundle.normalization)))
    assert bundle.feature_names == current
    assert bundle.trained_on == "synthetic"
