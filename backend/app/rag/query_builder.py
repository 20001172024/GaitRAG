"""Turn extracted gait features into retrieval queries.

One query per feature, not one combined query: a mixed query lets the dominant
topic crowd out the others, while per-feature queries give each feature its
own evidence. The user's numeric values are left out on purpose: "112.4" does
not help BM25 or embeddings find normative passages. The predicted speed class
is included because normative values depend on walking speed.
"""

from __future__ import annotations

from dataclasses import dataclass

_SPEED_PHRASE = {
    "slow": "slow walking speed",
    "normal": "preferred (self-selected) walking speed",
    "fast": "fast walking speed",
}


@dataclass(frozen=True)
class FeatureQuery:
    """A retrieval query for one gait feature."""

    feature: str  # e.g. "cadence"
    query: str


# feature -> (features that trigger it, query template)
_TEMPLATES: list[tuple[str, tuple[str, ...], str]] = [
    (
        "cadence",
        ("cadence_steps_per_min",),
        "Reference values for cadence in steps per minute in healthy adults at {speed}",
    ),
    (
        "stride_time",
        ("stride_time_s",),
        "Normal stride time (gait cycle duration) in healthy adults at {speed}",
    ),
    (
        "stride_time_variability",
        ("stride_time_cv_pct",),
        "Stride time variability (coefficient of variation) in healthy adults "
        "and how it changes with walking speed",
    ),
    (
        "stance_swing",
        ("stance_pct_left", "stance_pct_right", "stance_swing_ratio"),
        "Stance and swing phase as a percentage of the gait cycle in healthy adults at {speed}",
    ),
    (
        "asymmetry",
        ("asymmetry_stance_time_pct", "asymmetry_peak_pressure_pct"),
        "Left-right gait asymmetry and symmetry index of stance time and plantar "
        "pressure in healthy walking",
    ),
]


def build_feature_queries(features: dict[str, float], prediction: str | None) -> list[FeatureQuery]:
    """One query per feature group present in `features` (keys as in GaitFeatures).

    Args:
        features: interpretable gait features, e.g. from POST /analyze.
        prediction: predicted speed class ("slow" | "normal" | "fast") or None.
    """
    speed = _SPEED_PHRASE.get(prediction or "", "different walking speeds")
    return [
        FeatureQuery(name, template.format(speed=speed))
        for name, keys, template in _TEMPLATES
        if any(k in features for k in keys)
    ]
