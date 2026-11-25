"""Feature extraction: interpretable gait features + statistical/FFT model features.

Two outputs:
- `GaitFeatures`: the interpretable values shown to users and compared with
  published norms later (cadence, stride time and variability, stance/swing,
  left-right asymmetry). Computed from raw pressure.
- `model_features`: the classifier's input vector = gait features + per-channel
  statistics of the *normalized* signals + frequency features of each IMU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import signal, stats

from app.ml.gait_events import FootStrides, detect_strides
from app.ml.schema import IMU_IDS, SAMPLING_RATE_HZ, SENSOR_COLUMNS

FFT_BANDS_HZ = ((0.3, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 20.0))
"""Relative power bands. Walking stride/step frequencies sit roughly in 0.5-3 Hz."""

WELCH_WINDOW_S = 6.4
"""256 samples at 40 Hz: several strides per window, ~0.16 Hz frequency resolution."""


@dataclass(frozen=True)
class GaitFeatures:
    """Interpretable spatiotemporal gait features of one recording."""

    cadence_steps_per_min: float
    stride_time_s: float
    stride_time_cv_pct: float
    stance_pct_left: float
    stance_pct_right: float
    swing_pct_left: float
    swing_pct_right: float
    stance_swing_ratio: float
    asymmetry_stance_time_pct: float
    asymmetry_peak_pressure_pct: float
    n_strides_left: int
    n_strides_right: int

    def to_dict(self) -> dict[str, float]:
        """Return the features as a plain dict."""
        return asdict(self)


def symmetry_index(left: float, right: float) -> float:
    """Symmetry index in %: |L - R| / mean(L, R) x 100. 0 = perfectly symmetric."""
    mean = 0.5 * (left + right)
    return 0.0 if mean == 0 else abs(left - right) / mean * 100.0


def _cv_pct(x: np.ndarray) -> float:
    return float(np.std(x, ddof=1) / np.mean(x) * 100.0)


def gait_features_from_strides(left: FootStrides, right: FootStrides) -> GaitFeatures:
    """Compute interpretable features from detected strides of both feet."""
    stride_time = float(np.mean(np.concatenate([left.stride_times_s, right.stride_times_s])))
    stance_l = float(np.mean(left.stance_times_s / left.stride_times_s) * 100.0)
    stance_r = float(np.mean(right.stance_times_s / right.stride_times_s) * 100.0)
    mean_stance = 0.5 * (stance_l + stance_r)
    return GaitFeatures(
        # One stride = two steps.
        cadence_steps_per_min=120.0 / stride_time,
        stride_time_s=stride_time,
        # CV per foot, then averaged: pooling both feet would mix their means.
        stride_time_cv_pct=0.5 * (_cv_pct(left.stride_times_s) + _cv_pct(right.stride_times_s)),
        stance_pct_left=stance_l,
        stance_pct_right=stance_r,
        swing_pct_left=100.0 - stance_l,
        swing_pct_right=100.0 - stance_r,
        stance_swing_ratio=mean_stance / (100.0 - mean_stance),
        asymmetry_stance_time_pct=symmetry_index(
            float(np.mean(left.stance_times_s)), float(np.mean(right.stance_times_s))
        ),
        asymmetry_peak_pressure_pct=symmetry_index(
            float(np.mean(left.peak_loads)), float(np.mean(right.peak_loads))
        ),
        n_strides_left=left.n_strides,
        n_strides_right=right.n_strides,
    )


def extract_gait_features(df: pd.DataFrame) -> GaitFeatures:
    """Detect strides in a raw recording and compute interpretable gait features.

    Raises:
        GaitEventError: if too few strides are detected.
    """
    return gait_features_from_strides(*detect_strides(df))


def _channel_stats(normalized: pd.DataFrame) -> dict[str, float]:
    x = normalized[SENSOR_COLUMNS].to_numpy(dtype=float)
    funcs = {
        "mean": np.mean(x, axis=0),
        "std": np.std(x, axis=0),
        "min": np.min(x, axis=0),
        "max": np.max(x, axis=0),
        "skew": np.nan_to_num(stats.skew(x, axis=0)),
        "kurtosis": np.nan_to_num(stats.kurtosis(x, axis=0)),
    }
    return {
        f"{col}__{name}": float(values[i])
        for name, values in funcs.items()
        for i, col in enumerate(SENSOR_COLUMNS)
    }


def imu_frequency_features(raw: pd.DataFrame, imu_id: int) -> dict[str, float]:
    """Dominant frequency and relative band powers of one IMU's acceleration magnitude."""
    acc = raw[[f"imu{imu_id}_ax", f"imu{imu_id}_ay", f"imu{imu_id}_az"]].to_numpy(dtype=float)
    mag = signal.detrend(np.linalg.norm(acc, axis=1))
    nperseg = min(len(mag), round(WELCH_WINDOW_S * SAMPLING_RATE_HZ))
    freqs, power = signal.welch(mag, fs=SAMPLING_RATE_HZ, nperseg=nperseg)

    in_range = (freqs >= FFT_BANDS_HZ[0][0]) & (freqs <= FFT_BANDS_HZ[-1][1])
    total = float(power[in_range].sum()) or 1.0
    walk = (freqs >= 0.3) & (freqs <= 5.0)
    out = {f"imu{imu_id}_acc_dominant_freq_hz": float(freqs[walk][np.argmax(power[walk])])}
    for lo, hi in FFT_BANDS_HZ:
        band = (freqs >= lo) & (freqs < hi)
        out[f"imu{imu_id}_acc_power_{lo:g}_{hi:g}hz"] = float(power[band].sum()) / total
    return out


def model_features(
    raw: pd.DataFrame, normalized: pd.DataFrame, gait: GaitFeatures | None = None
) -> dict[str, float]:
    """Build the classifier's feature vector for one recording.

    Args:
        raw: the recording as uploaded (used for gait timing and FFT).
        normalized: the same recording after `zscore_by_group` (used for statistics).
        gait: precomputed gait features, to avoid detecting strides twice.

    Returns:
        Ordered mapping of feature name -> value (order is stable across calls).
    """
    gait = gait or extract_gait_features(raw)
    features = {
        f"gait__{k}": float(v) for k, v in gait.to_dict().items() if not k.startswith("n_strides")
    }
    features.update(_channel_stats(normalized))
    for imu_id in IMU_IDS:
        features.update(imu_frequency_features(raw, imu_id))
    return features
