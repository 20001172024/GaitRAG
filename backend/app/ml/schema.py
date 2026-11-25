"""CSV schema for insole gait recordings and validation of uploaded files.

Schema (to be confirmed against the real data):
    timestamp_ms, subject_id, trial_id,
    fsr_{l,r}_{heel,meta,toe},
    imu{1..4}_{ax,ay,az,gx,gy,gz}
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd

SAMPLING_RATE_HZ: float = 40.0
"""Sampling rate of the real recordings (confirmed by the project owner)."""

SIDES = ("l", "r")
FSR_SITES = ("heel", "meta", "toe")
IMU_IDS = (1, 2, 3, 4)
IMU_AXES = ("ax", "ay", "az", "gx", "gy", "gz")

ID_COLUMNS = ["timestamp_ms", "subject_id", "trial_id"]
FSR_COLUMNS = [f"fsr_{side}_{site}" for side in SIDES for site in FSR_SITES]
IMU_COLUMNS = [f"imu{i}_{axis}" for i in IMU_IDS for axis in IMU_AXES]
SENSOR_COLUMNS = FSR_COLUMNS + IMU_COLUMNS
ALL_COLUMNS = ID_COLUMNS + SENSOR_COLUMNS

MIN_DURATION_S: float = 10.0
"""Shortest recording we accept: enough for several strides at slow speed."""

SAMPLING_RATE_TOLERANCE: float = 0.10
"""Allowed relative deviation of the measured sampling rate from 40 Hz."""


class SchemaError(ValueError):
    """Raised when an uploaded recording does not match the expected schema."""


def fsr_columns(side: str) -> list[str]:
    """Return the three FSR column names for one foot (`"l"` or `"r"`)."""
    return [f"fsr_{side}_{site}" for site in FSR_SITES]


def read_recording(data: bytes | str | io.IOBase) -> pd.DataFrame:
    """Parse CSV content into a DataFrame and validate it.

    Args:
        data: raw CSV bytes, a path, or a file-like object.

    Returns:
        The validated recording, sorted by timestamp.

    Raises:
        SchemaError: if the file is not a valid single-trial recording.
    """
    if isinstance(data, bytes):
        data = io.BytesIO(data)
    try:
        df = pd.read_csv(data)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise SchemaError(f"Could not parse CSV: {exc}") from exc
    validate_recording(df)
    return df.sort_values("timestamp_ms").reset_index(drop=True)


def validate_recording(df: pd.DataFrame) -> None:
    """Check columns, types, single trial, duration and sampling rate.

    Raises:
        SchemaError: describing the first problem found.
    """
    missing = [c for c in ALL_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing columns: {', '.join(missing)}")

    numeric = ["timestamp_ms", *SENSOR_COLUMNS]
    non_numeric = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise SchemaError(f"Non-numeric values in columns: {', '.join(non_numeric)}")
    if df[numeric].isna().any().any():
        raise SchemaError("Recording contains missing (NaN) sensor values")

    if df["trial_id"].nunique() != 1 or df["subject_id"].nunique() != 1:
        raise SchemaError("Upload must contain exactly one subject and one trial")

    ts = np.sort(df["timestamp_ms"].to_numpy(dtype=float))
    if len(ts) < 2:
        raise SchemaError("Recording is too short")
    duration_s = (ts[-1] - ts[0]) / 1000.0
    if duration_s < MIN_DURATION_S:
        raise SchemaError(
            f"Recording is {duration_s:.1f} s long; at least {MIN_DURATION_S:.0f} s needed"
        )

    measured_hz = 1000.0 / float(np.median(np.diff(ts)))
    if abs(measured_hz - SAMPLING_RATE_HZ) > SAMPLING_RATE_TOLERANCE * SAMPLING_RATE_HZ:
        raise SchemaError(
            f"Sampling rate looks like {measured_hz:.1f} Hz; expected {SAMPLING_RATE_HZ:.0f} Hz"
        )
