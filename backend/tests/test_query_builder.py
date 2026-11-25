"""Tests for turning gait features into retrieval queries."""

from app.rag.query_builder import build_feature_queries


def test_feature_queries_one_per_feature_without_numbers_or_speed() -> None:
    features = {
        "cadence_steps_per_min": 112.4,
        "stride_time_s": 1.07,
        "stride_time_cv_pct": 2.1,
        "stance_pct_left": 61.0,
        "asymmetry_peak_pressure_pct": 6.4,
    }
    queries = build_feature_queries(features, "fast")
    assert [q.feature for q in queries] == [
        "cadence",
        "stride_time",
        "stride_time_variability",
        "stance_swing",
        "asymmetry",
    ]
    assert not any(any(ch.isdigit() for ch in q.query) for q in queries)
    # Speed is kept on the query object, not in the text (it biased retrieval).
    assert all(q.speed == "fast" and "speed" not in q.query for q in queries)


def test_feature_queries_skip_missing_features() -> None:
    queries = build_feature_queries({"cadence_steps_per_min": 100.0}, None)
    assert [(q.feature, q.speed) for q in queries] == [("cadence", None)]
