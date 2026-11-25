"""Turn extracted gait features into retrieval queries.

- One query per feature, not one combined query: a mixed query lets the dominant
  topic crowd out the others, while per-feature queries give each feature its
  own evidence.
- The user's numeric values are left out: "112.4" does not help BM25 or
  embeddings find normative passages.
- The predicted speed class is *not* put in the query text. An early version
  added "at preferred (self-selected) walking speed", and that phrase pulled
  every query towards one paper whose title is about walking speed (P008),
  pushing the reference-value tables out of the top results. The speed class is
  kept on the query object instead, for the generation step to choose which
  norms to compare against.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureQuery:
    """A retrieval query for one gait feature."""

    feature: str  # e.g. "cadence"
    query: str
    speed: str | None  # predicted speed class, for selecting comparable norms later


# feature -> (feature keys that trigger it, query text)
_TEMPLATES: list[tuple[str, tuple[str, ...], str]] = [
    (
        "cadence",
        ("cadence_steps_per_min",),
        "normative cadence values in steps per minute in healthy adults",
    ),
    (
        "stride_time",
        ("stride_time_s",),
        "normative stride time (gait cycle duration) values in healthy adults",
    ),
    (
        "stride_time_variability",
        ("stride_time_cv_pct",),
        "stride time variability coefficient of variation in healthy adults",
    ),
    (
        "stance_swing",
        ("stance_pct_left", "stance_pct_right", "stance_swing_ratio"),
        "stance and swing phase as a percentage of the gait cycle in healthy adults",
    ),
    (
        "asymmetry",
        ("asymmetry_stance_time_pct", "asymmetry_peak_pressure_pct"),
        "left-right gait asymmetry symmetry index of stance time and plantar pressure",
    ),
]


def build_feature_queries(features: dict[str, float], prediction: str | None) -> list[FeatureQuery]:
    """One query per feature group present in `features` (keys as in GaitFeatures).

    Args:
        features: interpretable gait features, e.g. from POST /analyze.
        prediction: predicted speed class ("slow" | "normal" | "fast") or None.
    """
    return [
        FeatureQuery(name, text, prediction)
        for name, keys, text in _TEMPLATES
        if any(k in features for k in keys)
    ]
