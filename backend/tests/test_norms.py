"""Tests for comparing gait features with approved norms."""

import pytest

from app.db.models import Norm, Paper
from app.rag.norms import citation_label, compare, compare_features, conversion_factor


def _norm(**kw) -> Norm:
    base = dict(
        paper_id="P003",
        page=6,
        table_no="4",
        metric="cadence",
        population="20–29 | M",
        value_text="109.88 (6.46)",
        mean=109.88,
        sd=6.46,
        low=None,
        high=None,
        units="steps min−1",
    )
    return Norm(**(base | kw))


@pytest.mark.parametrize(
    ("units", "feature_units", "factor"),
    [
        ("steps/min", "steps/min", 1.0),
        ("steps min−1", "steps/min", 1.0),
        ("steps/sec", "steps/min", 60.0),
        ("ms", "s", 0.001),
        ("s", "s", 1.0),
        ("%", "%", 1.0),
        ("m", "%", None),
        (None, "s", None),
    ],
)
def test_conversion_factor(units, feature_units, factor) -> None:
    assert conversion_factor(units, feature_units) == factor


def test_z_score_and_within_one_sd() -> None:
    c = compare("cadence_steps_per_min", 116.34, _norm())
    assert c.z_score == pytest.approx(1.0)
    assert c.within is True
    assert compare("cadence_steps_per_min", 125.0, _norm()).within is False


def test_units_are_converted_before_comparing() -> None:
    norm = _norm(
        metric="stride_time", value_text="1123.7 ± 122.4", mean=1123.7, sd=122.4, units="ms"
    )
    c = compare("stride_time_s", 1.0, norm)
    assert c.norm_mean == pytest.approx(1.1237)
    assert c.z_score == pytest.approx((1.0 - 1.1237) / 0.1224)


def test_range_norms_use_low_high() -> None:
    norm = _norm(metric="stance_pct", mean=None, sd=None, low=58.0, high=64.0, units="%")
    assert compare("stance_pct_left", 61.0, norm).within is True
    assert compare("stance_pct_left", 66.0, norm).within is False


def test_unconvertible_units_and_other_metrics_are_skipped() -> None:
    assert compare("cadence_steps_per_min", 110, _norm(units="n")) is None
    assert compare("cadence_steps_per_min", 110, _norm(metric="stride_time")) is None


def test_citation_label() -> None:
    assert citation_label("P003", 6, "4") == "[P003, p.6, Table 4]"
    assert citation_label("P008", 3) == "[P008, p.3]"
    assert compare("cadence_steps_per_min", 110, _norm()).citation == "[P003, p.6, Table 4]"


def test_compare_features_uses_approved_norms_only(db_session) -> None:
    db_session.add(Paper(paper_id="P003", title="t", licence="CC-BY", source_url="u", pdf_url="u"))
    db_session.flush()
    db_session.add_all([_norm(), _norm(population="20–29 | W", mean=114.91, sd=9.21)])
    db_session.flush()
    comps = compare_features(
        db_session, {"cadence_steps_per_min": 112.0, "stride_time_s": 1.0, "unknown": 3.0}
    )
    assert [(c.feature, c.population) for c in comps] == [
        ("cadence_steps_per_min", "20–29 | M"),
        ("cadence_steps_per_min", "20–29 | W"),
    ]


def test_symmetry_norms_only_compare_like_with_like() -> None:
    stance = _norm(
        metric="symmetry_index_stance_time", value_text="0.7 ± 0.1", mean=0.7, sd=0.1, units="%"
    )
    assert compare("asymmetry_stance_time_pct", 3.2, stance) is not None
    assert compare("asymmetry_peak_pressure_pct", 11.1, stance) is None
    generic = _norm(metric="symmetry_index", value_text="0.4 ± 0.1", mean=0.4, sd=0.1, units="%")
    assert compare("asymmetry_stance_time_pct", 3.2, generic) is None
