"""Tests for JATS XML tables and mapping them to PDF pages."""

from pathlib import Path

from app.rag.jats import (
    caption_pages,
    expand_spans,
    find_caption_page,
    locate_table_page,
    merge_tables,
    parse_jats_tables,
)
from app.rag.parsing import RawTable

JATS = """<?xml version="1.0"?>
<article><body>
<table-wrap id="t1"><label>Table 4.</label>
  <caption><title>Normative reference data by age group</title></caption>
  <table>
    <thead>
      <tr><th rowspan="2">parameter</th><th rowspan="2">sex</th><th colspan="2">age</th></tr>
      <tr><th>20&#8211;29</th><th>30&#8211;39</th></tr>
    </thead>
    <tbody>
      <tr><td rowspan="2">cadence (steps/min)</td><td>M</td><td>109.88 (6.46)</td><td>110.07 (8.93)</td></tr>
      <tr><td>W</td><td><bold>114.91</bold> (9.21)</td><td>116.40 (8.18)</td></tr>
    </tbody>
  </table>
</table-wrap>
<table-wrap id="t2"><label>Table 5</label><caption><p>Image only</p></caption>
  <graphic href="t5.jpg"/></table-wrap>
</body></article>"""


def test_expand_spans_fills_rowspan_and_colspan() -> None:
    rows = [[("a", 2, 1), ("b", 1, 2)], [("c", 1, 1), ("d", 1, 1)]]
    assert expand_spans(rows) == [["a", "b", "b"], ["a", "c", "d"]]


def test_parse_jats_tables(tmp_path: Path) -> None:
    path = tmp_path / "a.xml"
    path.write_text(JATS)
    tables = parse_jats_tables(path)
    assert len(tables) == 1  # image-only table skipped
    t = tables[0]
    assert t.table_no == "4"
    assert t.caption == "Table 4. Normative reference data by age group"
    assert t.n_header_rows == 2
    assert t.source == "xml"
    assert t.cells[2] == ["cadence (steps/min)", "M", "109.88 (6.46)", "110.07 (8.93)"]
    assert t.cells[3] == ["cadence (steps/min)", "W", "114.91 (9.21)", "116.40 (8.18)"]


def test_caption_pages_ignores_body_mentions() -> None:
    pages = [
        "As Table 4 shows, cadence rose.",
        "TABLE 2 | Characteristics\nTable 4. Normative data",
        "Table 4. Continued",
        "Table 3 Gait parameters at fast speed",
    ]
    assert caption_pages(pages) == {"2": 2, "4": 2, "3": 4}


def test_find_caption_page_matches_text_ignoring_spacing() -> None:
    pages = ["intro", "Table 1. Normative  reference\ndata by age-group (n=292)"]
    assert find_caption_page("Table 9 Normative reference data by age group", pages) == 2
    assert find_caption_page("Table 9 Short", pages) is None  # too short to trust


def _xml_table(no: str, caption: str) -> RawTable:
    cells = [["x", "a"], ["cadence", "109.88 (6.46)"], ["speed", "1.17 (0.10)"]]
    return RawTable(0, no, caption, cells, "xml", 1)


def test_locate_prefers_verified_caption_page() -> None:
    table = _xml_table("4", "Table 4. Normative reference data by age group")
    pages = ["Table 4. Normative reference data", "values 109.88 (6.46) and 1.17 (0.10)"]
    # Caption line is on page 1, but the table's numbers are on page 2.
    assert locate_table_page(table, pages, {"4": 1}) == 2


def test_locate_returns_none_when_numbers_are_nowhere() -> None:
    table = _xml_table("4", "Table 4. Normative reference data by age group")
    assert locate_table_page(table, ["nothing", "here"], {}) is None


def test_merge_prefers_xml_and_keeps_missing_numbers_from_pdf() -> None:
    xml = [RawTable(3, "1", "c", [["a"]], "xml")]
    pdf = [
        RawTable(3, "1", "c", [["garbled"]]),
        RawTable(5, "2", "c", [["image-only in XML"]]),
        RawTable(6, None, None, [["uncaptioned"]]),
    ]
    assert [(t.table_no, t.source) for t in merge_tables(xml, pdf)] == [("1", "xml"), ("2", "pdf")]
    assert merge_tables([], pdf) == pdf
