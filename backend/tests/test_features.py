"""Tests for gait and model feature extraction."""

import math

import numpy as np
import pytest

from app.ml.features import (
    extract_gait_features,
    imu_frequency_features,
    model_features,
    symmetry_index,
)
from app.ml.normalization import zscore_by_group
from tests.conftest import make_trial


def test_symmetry_index() -> None:
    assert symmetry_index(10.0, 10.0) == 0.0
    assert symmetry_index(11.0, 9.0) == pytest.approx(20.0)
    assert symmetry_index(0.0, 0.0) == 0.0


@pytest.mark.parametrize("speed", ["slow", "normal", "fast"])
def test_gait_features_match_ground_truth(speed: str) -> None:
    df, truth = make_trial(speed, seed=5)
    g = extract_gait_features(df)
    assert g.cadence_steps_per_min == pytest.approx(truth.cadence_steps_per_min, rel=0.03)
    assert g.stride_time_s == pytest.approx(truth.stride_time_s, rel=0.03)
    assert g.stance_pct_left == pytest.approx(truth.stance_frac_left * 100, abs=3.0)
    assert g.stance_pct_left + g.swing_pct_left == pytest.approx(100.0)
    # Generated CV is a few %; detection noise should not dominate it.
    assert 0.5 * truth.stride_cv * 100 < g.stride_time_cv_pct < 2.0 * truth.stride_cv * 100


def test_pressure_asymmetry_tracks_generated_asymmetry() -> None:
    df, truth = make_trial("normal", seed=5)
    g = extract_gait_features(df)
    # Right foot load was scaled by (1 - a); SI of L vs R(1-a) is ~ a/(1-a/2) * 100.
    expected = truth.pressure_asymmetry / (1 - truth.pressure_asymmetry / 2) * 100
    assert g.asymmetry_peak_pressure_pct == pytest.approx(expected, abs=3.0)


def test_dominant_frequency_is_stride_frequency() -> None:
    df, truth = make_trial("normal", seed=2)
    f = imu_frequency_features(df, imu_id=1)
    stride_hz = 1.0 / truth.stride_time_s
    # Foot IMU signals are harmonics of stride frequency; the peak is one of them.
    harmonics = [stride_hz, 2 * stride_hz]
    assert min(abs(f["imu1_acc_dominant_freq_hz"] - h) for h in harmonics) < 0.2


def test_band_powers_are_fractions() -> None:
    df, _ = make_trial("fast", seed=2)
    f = imu_frequency_features(df, imu_id=3)
    bands = [v for k, v in f.items() if "_power_" in k]
    assert all(0.0 <= b <= 1.0 for b in bands)
    assert sum(bands) == pytest.approx(1.0, abs=0.05)


def test_model_features_are_finite_and_stably_ordered() -> None:
    df, _ = make_trial("slow", seed=1)
    norm = zscore_by_group(df, "trial_id")
    a = model_features(df, norm)
    b = model_features(df, norm)
    assert list(a) == list(b)
    assert all(math.isfinite(v) for v in a.values())
    assert len(a) == 10 + 30 * 6 + 4 * 6  # gait + channel stats + IMU frequency


def test_faster_walking_gives_higher_cadence() -> None:
    cadences = [
        extract_gait_features(make_trial(s, seed=9)[0]).cadence_steps_per_min
        for s in ("slow", "normal", "fast")
    ]
    assert np.all(np.diff(cadences) > 0)
