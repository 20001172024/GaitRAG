"""Detect heel strikes and toe-offs from insole pressure, then segment strides.

Method (per foot):
1. Total foot load = heel + meta + toe FSR. Using the sum (not heel only)
   means forefoot strikers still register contact.
2. Contact threshold = 10% of the 95th-percentile load. Relative, so it adapts
   to body weight and sensor gain; the 95th percentile ignores single spikes.
3. Contact/swing runs shorter than 100 ms are treated as noise and merged.
4. Heel strike = rising threshold crossing, toe-off = falling crossing. Crossing
   times are linearly interpolated between samples: at 40 Hz one sample is
   25 ms, which would otherwise quantize stride times and inflate their CV.
5. A stride is heel strike -> next heel strike of the same foot, kept only if
   it contains exactly one toe-off and has a plausible duration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.ml.schema import fsr_columns

THRESHOLD_FRACTION = 0.10
MIN_PHASE_S = 0.10
STRIDE_RANGE_S = (0.6, 2.5)  # plausible stride times (cadence ~48-200 steps/min)
MIN_STRIDES_PER_FOOT = 3


class GaitEventError(ValueError):
    """Raised when too few valid strides are found to compute gait features."""


@dataclass(frozen=True)
class FootStrides:
    """Valid strides of one foot. All arrays have one entry per stride."""

    side: str
    heel_strikes_s: np.ndarray  # start time of each stride
    stride_times_s: np.ndarray
    stance_times_s: np.ndarray
    peak_loads: np.ndarray  # max total foot load during stance

    @property
    def n_strides(self) -> int:
        """Number of valid strides."""
        return len(self.stride_times_s)


def _remove_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    """Flip runs of identical values shorter than `min_len` samples (interior runs only)."""
    mask = mask.copy()
    changes = np.flatnonzero(np.diff(mask.astype(np.int8))) + 1
    bounds = np.concatenate([[0], changes, [len(mask)]])
    for start, end in zip(bounds[1:-2], bounds[2:-1], strict=True):
        if end - start < min_len:
            mask[start:end] = ~mask[start]
    return mask


def _crossing_times(t: np.ndarray, load: np.ndarray, idx: np.ndarray, thr: float) -> np.ndarray:
    """Interpolated time at which `load` crosses `thr` between samples idx-1 and idx."""
    v0, v1 = load[idx - 1], load[idx]
    frac = np.clip((thr - v0) / np.where(v1 == v0, 1.0, v1 - v0), 0.0, 1.0)
    return t[idx - 1] + frac * (t[idx] - t[idx - 1])


def detect_foot_strides(df: pd.DataFrame, side: str) -> FootStrides:
    """Detect valid strides for one foot.

    Args:
        df: validated recording (see `app.ml.schema`), raw (un-normalized) values.
        side: `"l"` or `"r"`.

    Returns:
        The foot's valid strides.
    """
    t = df["timestamp_ms"].to_numpy(dtype=float) / 1000.0
    load = df[fsr_columns(side)].sum(axis=1).to_numpy(dtype=float)
    thr = THRESHOLD_FRACTION * float(np.percentile(load, 95))

    dt = float(np.median(np.diff(t)))
    contact = _remove_short_runs(load > thr, max(1, round(MIN_PHASE_S / dt)))

    edges = np.diff(contact.astype(np.int8))
    rise_idx = np.flatnonzero(edges == 1) + 1
    fall_idx = np.flatnonzero(edges == -1) + 1
    heel_strikes = _crossing_times(t, load, rise_idx, thr)
    toe_offs = _crossing_times(t, load, fall_idx, thr)

    strides, stances, peaks, starts = [], [], [], []
    for hs, next_hs, i0, i1 in zip(
        heel_strikes[:-1], heel_strikes[1:], rise_idx[:-1], rise_idx[1:], strict=True
    ):
        stride = next_hs - hs
        tos = toe_offs[(toe_offs > hs) & (toe_offs < next_hs)]
        if len(tos) != 1 or not STRIDE_RANGE_S[0] <= stride <= STRIDE_RANGE_S[1]:
            continue
        starts.append(hs)
        strides.append(stride)
        stances.append(tos[0] - hs)
        peaks.append(load[i0:i1].max())

    return FootStrides(
        side=side,
        heel_strikes_s=np.array(starts),
        stride_times_s=np.array(strides),
        stance_times_s=np.array(stances),
        peak_loads=np.array(peaks),
    )


def detect_strides(df: pd.DataFrame) -> tuple[FootStrides, FootStrides]:
    """Detect strides on both feet.

    Returns:
        (left, right) strides.

    Raises:
        GaitEventError: if either foot has fewer than `MIN_STRIDES_PER_FOOT` valid strides.
    """
    left, right = detect_foot_strides(df, "l"), detect_foot_strides(df, "r")
    for foot in (left, right):
        if foot.n_strides < MIN_STRIDES_PER_FOOT:
            raise GaitEventError(
                f"Only {foot.n_strides} valid strides found on the {foot.side.upper()} foot; "
                f"need at least {MIN_STRIDES_PER_FOOT}. Is this a walking recording?"
            )
    return left, right
