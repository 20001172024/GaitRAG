"""Tests for subject-wise z-score normalization."""

import numpy as np
import pandas as pd
import pytest

from app.ml.normalization import zscore_by_group


def _two_subjects() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["A"] * 4 + ["B"] * 4,
            "trial_id": ["A1", "A1", "A2", "A2", "B1", "B1", "B2", "B2"],
            "x": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0],
            "flat": [5.0] * 8,
        }
    )


def test_each_subject_has_zero_mean_unit_std() -> None:
    out = zscore_by_group(_two_subjects(), "subject_id", ["x"])
    stats = out.groupby("subject_id")["x"].agg(["mean", lambda s: s.std(ddof=0)])
    assert np.allclose(stats.iloc[:, 0], 0.0)
    assert np.allclose(stats.iloc[:, 1], 1.0)


def test_scale_difference_between_subjects_is_removed() -> None:
    out = zscore_by_group(_two_subjects(), "subject_id", ["x"])
    a = out.loc[out.subject_id == "A", "x"].to_numpy()
    b = out.loc[out.subject_id == "B", "x"].to_numpy()
    assert np.allclose(a, b)  # B is A x 100, so identical after normalization


def test_per_trial_grouping_normalizes_each_trial() -> None:
    out = zscore_by_group(_two_subjects(), "trial_id", ["x"])
    assert np.allclose(out.groupby("trial_id")["x"].mean(), 0.0)


def test_constant_channel_becomes_zero_not_nan() -> None:
    out = zscore_by_group(_two_subjects(), "subject_id", ["flat"])
    assert (out["flat"] == 0.0).all()


def test_input_is_not_modified() -> None:
    df = _two_subjects()
    zscore_by_group(df, "subject_id", ["x"])
    assert df["x"].iloc[0] == 1.0


def test_unknown_group_is_rejected() -> None:
    with pytest.raises(ValueError):
        zscore_by_group(_two_subjects(), "x", ["x"])
