"""Tests for CSV schema validation."""

import pytest

from app.ml.schema import ALL_COLUMNS, SchemaError, read_recording, validate_recording


def test_valid_recording_round_trips(normal_trial) -> None:
    df, _ = normal_trial
    parsed = read_recording(df.to_csv(index=False).encode())
    assert list(parsed.columns) == ALL_COLUMNS
    assert len(parsed) == len(df)


def test_missing_column_is_rejected(normal_trial) -> None:
    df, _ = normal_trial
    with pytest.raises(SchemaError, match="Missing columns: fsr_l_toe"):
        validate_recording(df.drop(columns=["fsr_l_toe"]))


def test_non_numeric_sensor_is_rejected(normal_trial) -> None:
    df, _ = normal_trial
    df = df.astype({"imu1_ax": str})
    df.loc[0, "imu1_ax"] = "oops"
    with pytest.raises(SchemaError, match="Non-numeric"):
        validate_recording(df)


def test_nan_is_rejected(normal_trial) -> None:
    df, _ = normal_trial
    df.loc[5, "fsr_r_heel"] = float("nan")
    with pytest.raises(SchemaError, match="missing"):
        validate_recording(df)


def test_multiple_trials_are_rejected(normal_trial) -> None:
    df, _ = normal_trial
    df.loc[: len(df) // 2, "trial_id"] = "other"
    with pytest.raises(SchemaError, match="exactly one"):
        validate_recording(df)


def test_short_recording_is_rejected(normal_trial) -> None:
    df, _ = normal_trial
    with pytest.raises(SchemaError, match="at least"):
        validate_recording(df.iloc[:200])  # 5 s at 40 Hz


def test_wrong_sampling_rate_is_rejected(normal_trial) -> None:
    df, _ = normal_trial
    df = df.iloc[::2]  # 20 Hz
    with pytest.raises(SchemaError, match="Sampling rate"):
        validate_recording(df)


def test_unparseable_bytes_are_rejected() -> None:
    with pytest.raises(SchemaError):
        read_recording(b"")
