"""SYNTHETIC gait recordings that mimic the real insole CSV schema.

Nothing here is real participant data, and none of the parameters below are
published norms. They are made-up values chosen so the three speed classes
overlap a little, like real data does, and the pipeline can be exercised end to end.

Model of one recording:
- Each foot alternates stance and swing. Stride times are drawn around a
  speed-dependent mean with some variability (CV).
- During stance, pressure rolls from heel -> metatarsals -> toe (three bumps).
- IMU signals are harmonics of each foot's stride phase plus noise, with a
  per-subject mounting gain/offset so subject-wise normalization matters.
- IMU placement assumed: imu1 = left foot, imu2 = right foot,
  imu3 = left shank, imu4 = right shank (configurable later).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.ml.schema import ALL_COLUMNS, FSR_SITES, IMU_AXES, SAMPLING_RATE_HZ

SPEEDS = ("slow", "normal", "fast")

# Made-up generation parameters (NOT published norms).
_CADENCE_MEAN = {"slow": 92.0, "normal": 108.0, "fast": 124.0}  # steps/min
_CADENCE_TRIAL_SD = 4.0
_STANCE_FRAC = {"slow": 0.66, "normal": 0.62, "fast": 0.58}
_STRIDE_CV = {"slow": 0.035, "normal": 0.025, "fast": 0.020}
_ACC_AMP = {"slow": 2.0, "normal": 3.5, "fast": 5.0}  # m/s^2
_GYRO_AMP = {"slow": 150.0, "normal": 250.0, "fast": 350.0}  # deg/s
_HEEL_IMPACT = {"slow": 0.9, "normal": 1.0, "fast": 1.2}

# Pressure bump (centre, width) in normalized stance phase [0, 1].
_FSR_BUMPS = {"heel": (0.12, 0.09), "meta": (0.50, 0.18), "toe": (0.85, 0.09)}
_FSR_BASE_AMP = 600.0  # arbitrary ADC-like units
_FSR_NOISE_SD = 8.0


@dataclass(frozen=True)
class SubjectParams:
    """Per-subject traits that stay fixed across that subject's trials."""

    subject_id: str
    weight_factor: float  # scales pressure
    cadence_offset: float  # steps/min added to every trial
    pressure_asymmetry: float  # right foot load = left * (1 - a)
    imu_gain: np.ndarray  # shape (4, 6): mounting/sensitivity gain
    imu_offset: np.ndarray  # shape (4, 6): mounting bias


@dataclass(frozen=True)
class TrialTruth:
    """Ground-truth values used to generate a trial (for tests and sanity checks)."""

    speed: str
    cadence_steps_per_min: float
    stride_time_s: float
    stance_frac_left: float
    stance_frac_right: float
    stride_cv: float
    pressure_asymmetry: float


def make_subject(index: int, rng: np.random.Generator) -> SubjectParams:
    """Draw random traits for one synthetic subject."""
    return SubjectParams(
        subject_id=f"SYN{index:02d}",
        weight_factor=float(rng.uniform(0.7, 1.3)),
        cadence_offset=float(rng.normal(0.0, 5.0)),
        pressure_asymmetry=float(rng.uniform(0.0, 0.12)),
        imu_gain=rng.uniform(0.8, 1.2, size=(4, 6)),
        imu_offset=rng.normal(0.0, 0.3, size=(4, 6)),
    )


def _heel_strikes(stride_times: np.ndarray, start: float) -> np.ndarray:
    return start + np.concatenate([[0.0], np.cumsum(stride_times)])


def _phase(t: np.ndarray, strikes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (stride index, phase in [0,1)) of each time point."""
    k = np.searchsorted(strikes, t, side="right") - 1
    k = np.clip(k, 0, len(strikes) - 2)
    phase = (t - strikes[k]) / (strikes[k + 1] - strikes[k])
    return k, phase


def _foot_pressure(
    phase: np.ndarray,
    stance_frac: float,
    amp: float,
    heel_impact: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    in_stance = phase < stance_frac
    p = np.where(in_stance, phase / stance_frac, 0.0)
    out = {}
    for site in FSR_SITES:
        centre, width = _FSR_BUMPS[site]
        scale = amp * (heel_impact if site == "heel" else 1.0)
        bump = scale * np.exp(-0.5 * ((p - centre) / width) ** 2) * in_stance
        noisy = bump + 5.0 + rng.normal(0.0, _FSR_NOISE_SD, size=phase.shape)
        out[site] = np.clip(noisy, 0.0, None)
    return out


def _imu(
    phase_cum: np.ndarray, acc_amp: float, gyro_amp: float, rng: np.random.Generator
) -> np.ndarray:
    """Six IMU channels (ax, ay, az, gx, gy, gz) driven by a cumulative stride phase."""
    w = 2 * np.pi * phase_cum
    chans = np.stack(
        [
            acc_amp * (np.sin(w) + 0.5 * np.sin(2 * w + 0.3)),
            0.4 * acc_amp * np.sin(w + 1.0),
            9.81 + 0.8 * acc_amp * np.cos(2 * w),
            gyro_amp * np.sin(w + 0.5),
            0.3 * gyro_amp * np.sin(2 * w),
            0.2 * gyro_amp * np.cos(w),
        ]
    )
    noise_sd = np.array([0.3, 0.3, 0.3, 10.0, 10.0, 10.0])[:, None]
    return chans + rng.normal(0.0, 1.0, size=chans.shape) * noise_sd


def generate_trial(
    subject: SubjectParams,
    speed: str,
    trial_id: str,
    rng: np.random.Generator,
    duration_s: float = 30.0,
) -> tuple[pd.DataFrame, TrialTruth]:
    """Generate one SYNTHETIC trial.

    Args:
        subject: subject traits from `make_subject`.
        speed: one of `SPEEDS`.
        trial_id: value written to the `trial_id` column.
        rng: NumPy random generator (pass a seeded one for reproducibility).
        duration_s: recording length in seconds.

    Returns:
        (recording DataFrame in the project CSV schema, ground truth used).
    """
    if speed not in SPEEDS:
        raise ValueError(f"speed must be one of {SPEEDS}")

    cadence = _CADENCE_MEAN[speed] + subject.cadence_offset + rng.normal(0.0, _CADENCE_TRIAL_SD)
    cadence = float(np.clip(cadence, 70.0, 150.0))
    stride_mean = 120.0 / cadence  # one stride = two steps
    stride_cv = float(_STRIDE_CV[speed] * rng.uniform(0.8, 1.2))
    stance_l = float(_STANCE_FRAC[speed] + rng.normal(0.0, 0.015))
    stance_r = float(stance_l * (1 - subject.pressure_asymmetry / 3))

    t = np.arange(0.0, duration_s, 1.0 / SAMPLING_RATE_HZ)
    n_strides = int(duration_s / stride_mean) + 6
    strides = rng.normal(stride_mean, stride_cv * stride_mean, size=n_strides)
    strides = np.clip(strides, 0.5 * stride_mean, 1.5 * stride_mean)

    start = -2 * stride_mean - rng.uniform(0.0, stride_mean)
    hs_left = _heel_strikes(strides, start)
    # Right foot strikes about half a stride after the left.
    half = 0.5 * np.diff(hs_left) * rng.normal(1.0, 0.01, size=len(hs_left) - 1)
    hs_right = hs_left[:-1] + half

    k_l, ph_l = _phase(t, hs_left)
    k_r, ph_r = _phase(t, hs_right)

    amp = _FSR_BASE_AMP * subject.weight_factor
    impact = _HEEL_IMPACT[speed]
    left = _foot_pressure(ph_l, stance_l, amp, impact, rng)
    right = _foot_pressure(ph_r, stance_r, amp * (1 - subject.pressure_asymmetry), impact, rng)

    cols: dict[str, np.ndarray] = {}
    for site in FSR_SITES:
        cols[f"fsr_l_{site}"] = left[site]
        cols[f"fsr_r_{site}"] = right[site]

    acc, gyro = _ACC_AMP[speed], _GYRO_AMP[speed]
    placements = [
        (k_l + ph_l, 1.0),
        (k_r + ph_r, 1.0),
        (k_l + ph_l, 0.6),
        (k_r + ph_r, 0.6),
    ]
    for i, (phase_cum, scale) in enumerate(placements):
        sig = _imu(phase_cum, acc * scale, gyro * scale, rng)
        sig = sig * subject.imu_gain[i][:, None] + subject.imu_offset[i][:, None]
        for j, axis in enumerate(IMU_AXES):
            cols[f"imu{i + 1}_{axis}"] = sig[j]

    df = pd.DataFrame(cols)
    df.insert(0, "trial_id", trial_id)
    df.insert(0, "subject_id", subject.subject_id)
    df.insert(0, "timestamp_ms", np.round(t * 1000).astype(int))
    df = df[ALL_COLUMNS].round(3)

    truth = TrialTruth(
        speed=speed,
        cadence_steps_per_min=cadence,
        stride_time_s=stride_mean,
        stance_frac_left=stance_l,
        stance_frac_right=stance_r,
        stride_cv=stride_cv,
        pressure_asymmetry=subject.pressure_asymmetry,
    )
    return df, truth
