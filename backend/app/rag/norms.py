"""Compare a recording's gait features with approved published norms.

All arithmetic happens here, in code, not in the LLM: language models make
arithmetic and unit mistakes, and these numbers are shown as facts. The LLM
only explains comparisons it is given.

Only rows from the `norms` table are used, i.e. values a human approved in
data/papers/norms_review.csv. A row whose units cannot be converted to the
feature's units is skipped rather than guessed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Norm

_EPS = 1e-9  # a value exactly on mean ± SD counts as within (float rounding)

# GaitFeatures field -> (norm metric, feature units)
FEATURE_METRICS: dict[str, tuple[str, str]] = {
    "cadence_steps_per_min": ("cadence", "steps/min"),
    "stride_time_s": ("stride_time", "s"),
    "stride_time_cv_pct": ("stride_time_cv", "%"),
    "stance_pct_left": ("stance_pct", "%"),
    "stance_pct_right": ("stance_pct", "%"),
    "swing_pct_left": ("swing_pct", "%"),
    "swing_pct_right": ("swing_pct", "%"),
    # Symmetry indices are only comparable for the same quantity: a published step-length
    # SI must not be compared with our peak-pressure SI. The extractor proposes a generic
    # "symmetry_index"; the reviewer sets the specific metric when approving a row.
    "asymmetry_stance_time_pct": ("symmetry_index_stance_time", "%"),
    "asymmetry_peak_pressure_pct": ("symmetry_index_peak_pressure", "%"),
}


@dataclass(frozen=True)
class NormComparison:
    """One feature value compared with one published norm row."""

    feature: str
    value: float
    units: str
    paper_id: str
    page: int
    table_no: str | None
    population: str | None
    norm_text: str  # verbatim cell, e.g. "109.88 (6.46)"
    norm_mean: float | None  # converted to `units`
    norm_sd: float | None
    norm_low: float | None
    norm_high: float | None
    z_score: float | None  # (value - mean) / sd
    within: bool | None  # within mean ± 1 SD, or inside [low, high]

    @property
    def citation(self) -> str:
        """Citation label, e.g. "[P003, p.6, Table 4]"."""
        return citation_label(self.paper_id, self.page, self.table_no)


def citation_label(paper_id: str, page: int, table_no: str | None = None) -> str:
    """The inline citation format: [P012, p.4] or [P012, p.4, Table 2]."""
    return f"[{paper_id}, p.{page}, Table {table_no}]" if table_no else f"[{paper_id}, p.{page}]"


def conversion_factor(norm_units: str | None, feature_units: str) -> float | None:
    """Multiplier from a norm's units to the feature's units, or None if unknown."""
    u = re.sub(r"\s+", "", (norm_units or "").lower()).replace("−", "-")
    if feature_units == "steps/min":
        if "min" in u:
            return 1.0
        if re.search(r"/s(ec)?$|s-1$|persec", u):
            return 60.0
        return None
    if feature_units == "s":
        if u.startswith("ms") or "millisec" in u:
            return 0.001
        if u in ("s", "sec", "second", "seconds"):
            return 1.0
        return None
    if feature_units == "%":
        return 1.0 if "%" in u else None
    return None


def compare(feature: str, value: float, norm: Norm) -> NormComparison | None:
    """Compare one feature value with one norm row; None if units don't convert."""
    metric, units = FEATURE_METRICS[feature]
    if norm.metric != metric:
        return None
    factor = conversion_factor(norm.units, units)
    if factor is None:
        return None

    def conv(x: float | None) -> float | None:
        return None if x is None else x * factor

    mean, sd, low, high = conv(norm.mean), conv(norm.sd), conv(norm.low), conv(norm.high)
    z = (value - mean) / sd if mean is not None and sd else None
    if z is not None:
        within: bool | None = abs(z) <= 1.0 + _EPS
    elif low is not None and high is not None:
        within = low <= value <= high
    else:
        within = None
    return NormComparison(
        feature=feature,
        value=value,
        units=units,
        paper_id=norm.paper_id,
        page=norm.page,
        table_no=norm.table_no,
        population=norm.population,
        norm_text=norm.value_text,
        norm_mean=mean,
        norm_sd=sd,
        norm_low=low,
        norm_high=high,
        z_score=z,
        within=within,
    )


def compare_features(session: Session, features: dict[str, float]) -> list[NormComparison]:
    """Compare every known feature with every approved norm of its metric."""
    wanted = {FEATURE_METRICS[f][0] for f in features if f in FEATURE_METRICS}
    if not wanted:
        return []
    norms = session.scalars(
        select(Norm).where(Norm.metric.in_(wanted)).order_by(Norm.paper_id, Norm.page, Norm.id)
    ).all()
    out = []
    for feature, value in features.items():
        if feature not in FEATURE_METRICS:
            continue
        for norm in norms:
            c = compare(feature, value, norm)
            if c is not None:
                out.append(c)
    return out
