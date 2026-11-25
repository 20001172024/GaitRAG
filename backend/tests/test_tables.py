"""Tests for candidate norm extraction from tables."""

import pytest

from app.rag.parsing import RawTable
from app.rag.tables import extract_candidates, match_metric, parse_value, units_of


@pytest.mark.parametrize(
    ("cell", "kind", "mean", "sd", "low", "high"),
    [
        ("1.04 ± 0.12", "mean_sd", 1.04, 0.12, None, None),
        ("1.20±0.11∗#", "mean_sd", 1.20, 0.11, None, None),
        ("111.87**(9.27)", "mean_paren", 111.87, 9.27, None, None),
        ("1.05 (0.90–1.20)", "mean_range", 1.05, None, 0.90, 1.20),
        ("(32.90–51.84)", "range", None, None, 32.90, 51.84),
        ("112.3", "plain", 112.3, None, None, None),
    ],
)
def test_parse_value(cell, kind, mean, sd, low, high) -> None:
    v = parse_value(cell)
    assert (v.kind, v.mean, v.sd, v.low, v.high) == (kind, mean, sd, low, high)


def test_parse_value_keeps_leading_group_label() -> None:
    assert parse_value("LWS 1.04±0.12").cell_label == "LWS"


@pytest.mark.parametrize("cell", ["< 0.001", "n.s.", "Yes", ""])
def test_non_values_are_ignored(cell: str) -> None:
    assert parse_value(cell) is None


@pytest.mark.parametrize(
    ("label", "metric"),
    [
        ("Cadence (steps/min)", "cadence"),
        ("stride time (cycle time) [s]", "stride_time"),
        ("Stride time variability (CV %)", "stride_time_cv"),
        ("stance time symmetry [%]", "symmetry_index"),
        ("relative double stance time [%]", "double_support"),
        ("1. Velocity [cm/s]", "gait_speed"),
        ("Speed 1.6 km/h", None),  # a test condition, not a value
        ("Body height (cm)", None),
    ],
)
def test_match_metric(label: str, metric: str | None) -> None:
    assert match_metric(label) == metric


def test_units_of() -> None:
    assert units_of("cadence [steps/min]") == "steps/min"
    assert units_of("Stance phase (%)") == "%"
    assert units_of("Cadence") is None


def _table(cells: list[list[str | None]]) -> RawTable:
    return RawTable(page=4, table_no="2", caption="Table 2. Gait at three speeds", cells=cells)


def test_metric_in_rows_group_in_columns() -> None:
    table = _table(
        [
            ["Variable", "Slow (n = 20)", "Fast (n = 20)", "p-value"],
            ["Cadence (steps/min)", "98.1 ± 7.2", "121.4 ± 8.0", "< 0.001"],
            ["Stance phase (%)", "63.2 ± 1.5", "59.8 ± 1.2", "0.002"],
            ["Body mass (kg)", "70.2 ± 9.1", "71.0 ± 8.4", "0.80"],
        ]
    )
    cands = extract_candidates("P001", table)
    assert [(c.metric, c.value_text, c.population, c.units) for c in cands] == [
        ("cadence", "98.1 ± 7.2", "Slow (n = 20)", "steps/min"),
        ("cadence", "121.4 ± 8.0", "Fast (n = 20)", "steps/min"),
        ("stance_pct", "63.2 ± 1.5", "Slow (n = 20)", "%"),
        ("stance_pct", "59.8 ± 1.2", "Fast (n = 20)", "%"),
    ]
    assert all(c.page == 4 and c.table_no == "2" for c in cands)


def test_metric_in_columns_group_in_rows() -> None:
    table = _table(
        [
            ["Group", "Cadence (steps/min)", "ICC"],
            ["Young adults", "112.0 (8.0)", "0.93"],
            ["Older adults", "104.5 (9.9)", "0.91"],
        ]
    )
    cands = extract_candidates("P002", table)
    assert [(c.metric, c.mean, c.sd, c.population) for c in cands] == [
        ("cadence", 112.0, 8.0, "Young adults"),
        ("cadence", 104.5, 9.9, "Older adults"),
    ]


def test_plain_numbers_need_a_mean_header_and_merged_cells_are_skipped() -> None:
    table = _table(
        [
            ["Variable", "Adjusted R2", "Mean", "Merged"],
            ["Cadence (steps/min)", "0.398", "110.2", "1.0 ± 0.1 2.0 ± 0.2"],
        ]
    )
    cands = extract_candidates("P003", table)
    assert [(c.column_header, c.value_text) for c in cands] == [("Mean", "110.2")]


def test_candidate_ids_are_stable() -> None:
    table = _table([["Variable", "Mean"], ["Cadence", "110.2"]])
    assert extract_candidates("P001", table)[0].candidate_id == (
        extract_candidates("P001", table)[0].candidate_id
    )


def test_known_header_rows_keep_value_like_headers() -> None:
    # "20–29" looks like a range value; with n_header_rows it stays a header.
    table = RawTable(
        6,
        "4",
        "Table 4",
        [
            ["parameter", "sex", "20–29", "30–39"],
            ["cadence (steps/min)", "M", "109.88 (6.46)", "110.07 (8.93)"],
            ["cadence (steps/min)", "W", "114.91 (9.21)", "116.40 (8.18)"],
        ],
        source="xml",
        n_header_rows=1,
    )
    cands = extract_candidates("P003", table)
    assert [c.population for c in cands] == ["20–29 | M", "30–39 | M", "20–29 | W", "30–39 | W"]
    assert len({c.candidate_id for c in cands}) == 4  # same row label, distinct rows


def test_section_rows_label_the_rows_below() -> None:
    table = RawTable(
        11,
        "3",
        "Table 3",
        [
            ["", "Total", "Female"],
            ["STRIDE TIME", "STRIDE TIME", "STRIDE TIME"],
            ["Mean value (ms)", "1123.7 ± 122.4", "1095.5 ± 109.8"],
            ["CoV (%)", "2.2 ± 1.1", "2.2 ± 1.1"],
        ],
        source="xml",
        n_header_rows=1,
    )
    cands = extract_candidates("P001", table)
    assert [(c.metric, c.units, c.value_text) for c in cands] == [
        ("stride_time", "ms", "1123.7 ± 122.4"),
        ("stride_time", "ms", "1095.5 ± 109.8"),
        ("stride_time_cv", "%", "2.2 ± 1.1"),
        ("stride_time_cv", "%", "2.2 ± 1.1"),
    ]
    assert cands[0].row_label == "STRIDE TIME Mean value (ms)"


def test_unicode_minus_is_parsed() -> None:
    assert parse_value("−0.42 ± 0.10").mean == -0.42
