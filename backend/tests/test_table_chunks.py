"""Tests for writing tables out as retrievable text chunks."""

from app.rag.parsing import RawTable
from app.rag.table_chunks import table_lines, table_to_chunks


def words(text: str) -> int:
    return len(text.split())


TABLE = RawTable(
    page=11,
    table_no="3",
    caption="Table 3 Quantitative reference values",
    cells=[
        ["", "Total", "Female (n = 437)"],
        ["STRIDE TIME", "STRIDE TIME", "STRIDE TIME"],
        ["Mean value (ms)", "1123.7 ± 122.4", "1095.5 ± 109.8"],
        ["CoV (%)", "2.2 ± 1.1", None],
    ],
    source="xml",
    n_header_rows=1,
)


def test_rows_are_written_with_headers_and_section() -> None:
    assert table_lines(TABLE) == [
        "STRIDE TIME Mean value (ms) — Total: 1123.7 ± 122.4; Female (n = 437): 1095.5 ± 109.8",
        "STRIDE TIME CoV (%) — Total: 2.2 ± 1.1",
    ]


def test_chunks_lead_with_caption_and_keep_page_and_number() -> None:
    (chunk,) = table_to_chunks(TABLE, words)
    assert chunk.text.startswith("Table 3 Quantitative reference values\n")
    assert (chunk.page, chunk.table_no) == (11, "3")


def test_large_tables_split_by_rows_and_repeat_caption() -> None:
    rows = [["", "Mean"]] + [[f"variable {i}", f"{i}.0 ± 0.1"] for i in range(60)]
    big = RawTable(2, "1", "Table 1 Big", rows, "xml", 1)
    chunks = table_to_chunks(big, words, max_tokens=40)
    assert len(chunks) > 1
    assert all(c.text.startswith("Table 1 Big\n") for c in chunks)
    assert all(c.token_count <= 40 for c in chunks)
    assert sum(c.text.count("variable") for c in chunks) == 60  # no row lost or repeated


def test_uncaptioned_table_gets_a_generic_caption() -> None:
    t = RawTable(1, None, None, [["", "Mean"], ["Cadence", "110.2"]])
    assert table_to_chunks(t, words)[0].text.splitlines()[0] == "Table"
