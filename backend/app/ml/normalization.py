"""Subject-wise z-score normalization of raw sensor channels.

Why: people differ in body weight, sensor fit and IMU mounting, which shifts
and scales every signal. Z-scoring each channel within a subject removes those
between-person offsets so the classifier learns speed, not identity.

Grouping key:
- `"subject_id"`: research setting. A subject's statistics come from all of
  their trials (used for the subject-wise evaluation that mirrors the
  original study).
- `"trial_id"`: deployment setting. A demo upload is the only data we have from
  that person, so it is normalized against itself. The deployed model is
  trained this way too, so training and serving see the same feature distribution.

Timing features (cadence, stride time, stance %) come from raw pressure in
`app.ml.gait_events`, so they are unaffected by this step.
"""

from __future__ import annotations

import pandas as pd

from app.ml.schema import SENSOR_COLUMNS

NORMALIZATION_GROUPS = ("subject_id", "trial_id")


def zscore_by_group(
    df: pd.DataFrame, group_col: str, columns: list[str] | None = None
) -> pd.DataFrame:
    """Z-score `columns` within each group of `group_col`.

    Args:
        df: one or more recordings in the project schema.
        group_col: `"subject_id"` or `"trial_id"`.
        columns: channels to normalize (default: all sensor channels).

    Returns:
        A copy of `df` with normalized channels. Constant channels become 0.
    """
    if group_col not in NORMALIZATION_GROUPS:
        raise ValueError(f"group_col must be one of {NORMALIZATION_GROUPS}")
    columns = columns or SENSOR_COLUMNS
    out = df.copy()
    grouped = df.groupby(group_col)[columns]
    mean = grouped.transform("mean")
    std = grouped.transform("std", ddof=0)
    out[columns] = (df[columns] - mean) / std.where(std > 0, 1.0)
    return out
