"""Tests for heel-strike / toe-off detection."""

import numpy as np
import pandas as pd
import pytest

from app.ml.gait_events import (
    GaitEventError,
    _remove_short_runs,
    detect_foot_strides,
    detect_strides,
)
from app.ml.schema import ALL_COLUMNS
from tests.conftest import make_trial


def _trapezoid_recording(
    stride_s: float, stance_frac: float, ramp_s: float = 0.1, duration_s: float = 20.0
):
    """Pressure with linear 100 ms loading/unloading ramps and known stride/stance times.

    Ramps (not an instant on/off step) match how real FSR load builds up, and are what
    threshold interpolation assumes. Contact is true while load > 0.
    """
    t = np.arange(0, duration_s, 0.025)
    t_in_stride = t % stride_s
    stance_s = stance_frac * stride_s
    ramp = np.minimum(t_in_stride, stance_s - t_in_stride) / ramp_s
    on = 500.0 * np.clip(ramp, 0.0, 1.0)
    df = pd.DataFrame(0.0, index=range(len(t)), columns=ALL_COLUMNS)
    df["timestamp_ms"] = t * 1000
    df["subject_id"], df["trial_id"] = "X", "X"
    for side in ("l", "r"):
        df[f"fsr_{side}_heel"] = on
    return df


def test_stride_and_stance_are_recovered_to_sub_sample_accuracy() -> None:
    df = _trapezoid_recording(stride_s=1.1, stance_frac=0.6)
    foot = detect_foot_strides(df, "l")
    assert foot.n_strides >= 15
    # The 10% threshold is crossed 10 ms into each 100 ms ramp, so measured stance
    # is 20 ms shorter than true contact. Interpolation keeps error well under one
    # 25 ms sample.
    assert np.allclose(foot.stride_times_s, 1.1, atol=0.005)
    assert np.allclose(foot.stance_times_s, 0.66 - 0.02, atol=0.005)


@pytest.mark.parametrize("speed", ["slow", "normal", "fast"])
def test_synthetic_trial_matches_ground_truth(speed: str) -> None:
    df, truth = make_trial(speed, seed=11)
    left, right = detect_strides(df)
    assert abs(left.stride_times_s.mean() - truth.stride_time_s) < 0.03
    stance_frac = (left.stance_times_s / left.stride_times_s).mean()
    assert abs(stance_frac - truth.stance_frac_left) < 0.03
    assert right.n_strides >= 10


def test_short_blips_are_removed() -> None:
    mask = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1], dtype=bool)
    cleaned = _remove_short_runs(mask, min_len=2)
    # Lone 1 at index 4 and lone 0 at index 13 are flipped; edge runs are kept.
    assert cleaned.tolist() == [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]


def test_standing_still_raises() -> None:
    df, _ = make_trial("normal")
    fsr = df.filter(like="fsr_").columns
    df[fsr] = 300.0 + np.random.default_rng(0).normal(0, 1, size=(len(df), len(fsr)))
    with pytest.raises(GaitEventError, match="valid strides"):
        detect_strides(df)
