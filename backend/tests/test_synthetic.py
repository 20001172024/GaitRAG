"""Tests for the SYNTHETIC recording generator."""

import numpy as np

from app.ml.schema import ALL_COLUMNS, SAMPLING_RATE_HZ, validate_recording
from tests.conftest import make_trial


def test_trial_matches_schema_and_rate() -> None:
    df, _ = make_trial("slow")
    assert list(df.columns) == ALL_COLUMNS
    validate_recording(df)  # does not raise
    dt = np.diff(df["timestamp_ms"].to_numpy())
    assert np.allclose(dt, 1000 / SAMPLING_RATE_HZ, atol=1)


def test_same_seed_is_reproducible() -> None:
    a, _ = make_trial("fast", seed=7)
    b, _ = make_trial("fast", seed=7)
    assert a.equals(b)


def test_pressure_is_non_negative() -> None:
    df, _ = make_trial("normal")
    fsr = df.filter(like="fsr_")
    assert (fsr >= 0).all().all()


def test_faster_speeds_have_higher_cadence() -> None:
    cadences = [make_trial(s, seed=3)[1].cadence_steps_per_min for s in ("slow", "normal", "fast")]
    assert cadences == sorted(cadences)
