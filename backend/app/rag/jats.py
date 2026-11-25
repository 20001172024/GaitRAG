"""Structured tables from PMC JATS XML, mapped to their PDF page.

Why: PDF table extraction fails on rotated tables and tables drawn with dot
leaders, which hit two key reference-value papers. The JATS XML from the same
PMC record has every table as explicit rows and cells. It has no page numbers,
so each table is located in the PDF by its caption line ("Table 4. ..."), or by
its caption text when the numbering differs between XML and PDF.
"""

from __future__ import annotations

import re
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import pymupdf

from app.rag.parsing import RawTable

_LABEL_NO = re.compile(r"(?:table\s*)?([A-Z]?\d+|[IVX]+)", re.IGNORECASE)


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", "".join(el.itertext()))).strip()


def expand_spans(rows: list[list[tuple[str, int, int]]]) -> list[list[str | None]]:
    """Turn rows of (text, rowspan, colspan) cells into a rectangular grid.

    A spanned value is repeated in every cell it covers, so each data cell can
    see its row label and its column header directly.
    """
    grid: dict[tuple[int, int], str] = {}
    for r, row in enumerate(rows):
        c = 0
        for text, rowspan, colspan in row:
            while (r, c) in grid:  # skip cells filled by a rowspan from above
                c += 1
            for dr in range(rowspan):
                for dc in range(colspan):
                    grid[(r + dr, c + dc)] = text
            c += colspan
    n_rows = max((r for r, _ in grid), default=-1) + 1
    n_cols = max((c for _, c in grid), default=-1) + 1
    return [[grid.get((r, c)) or None for c in range(n_cols)] for r in range(n_rows)]


def _span(cell: ET.Element, attr: str) -> int:
    try:
        return max(1, int(cell.get(attr, "1")))
    except ValueError:
        return 1


def parse_jats_tables(xml_path: Path) -> list[RawTable]:
    """Every <table-wrap> that has real cells, with page = 0 (not yet located).

    Tables published only as images (<graphic> without <table>) are skipped.
    """
    root = ET.parse(xml_path).getroot()
    tables = []
    for wrap in root.iter("table-wrap"):
        table = wrap.find(".//table")
        if table is None:
            continue
        label = _text(wrap.find("label"))
        match = _LABEL_NO.search(label)
        table_no = match.group(1) if match else None
        caption = " ".join(p for p in (label, _text(wrap.find("caption"))) if p) or None
        rows = [
            [
                (_text(c), _span(c, "rowspan"), _span(c, "colspan"))
                for c in tr
                if c.tag in ("td", "th")
            ]
            for tr in table.iter("tr")
        ]
        cells = expand_spans(rows)
        n_header = len(table.findall("./thead/tr")) or None
        if cells:
            tables.append(RawTable(0, table_no, caption, cells, "xml", n_header))
    return tables


def _squash(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKC", text).lower())


# "Table 4. ...", "TABLE 4 | ...", "Table 4: ...", "Table 2 Gait ...", bare "Table 4".
# Body text such as "Table 4 shows ..." (lowercase next word) does not match.
_CAPTION_LINE = re.compile(r"^\s*table\s+([A-Z]?\d+|[IVX]+)\s*(?:[.:|]|$|\s+[A-Z(])", re.I)
_CAPTION_PROBE_CHARS = 40


def caption_pages(page_texts: list[str]) -> dict[str, int]:
    """Map table number -> first page (1-based) with a caption line for it."""
    pages: dict[str, int] = {}
    for page_no, text in enumerate(page_texts, start=1):
        for line in text.splitlines():
            m = _CAPTION_LINE.match(line)
            if m:
                pages.setdefault(m.group(1).upper(), page_no)
    return pages


def find_caption_page(caption: str | None, page_texts: list[str]) -> int | None:
    """Fallback: first page containing the start of the caption title.

    Used when table numbers differ between XML and PDF. Comparison ignores case,
    spacing and punctuation.
    """
    title = re.sub(r"^\s*table\s+\S+\s*", "", caption or "", flags=re.I)
    probe = _squash(title)[:_CAPTION_PROBE_CHARS]
    if len(probe) < 15:  # too short to be distinctive
        return None
    for page_no, text in enumerate(page_texts, start=1):
        if probe in _squash(text):
            return page_no
    return None


_DECIMAL = re.compile(r"\d+\.\d+")
SAMPLE_VALUES = 6


def _value_hits(table: RawTable, page_text: str) -> tuple[int, int]:
    """(found, sampled): how many of the table's decimal numbers appear on the page."""
    body = table.cells[table.n_header_rows or 1 :]
    numbers = [m for row in body for c in row[1:] if c for m in _DECIMAL.findall(c)]
    sample = list(dict.fromkeys(numbers))[:SAMPLE_VALUES]
    squashed = re.sub(r"\s", "", page_text)
    return sum(1 for n in sample if n in squashed), len(sample)


def locate_table_page(
    table: RawTable, page_texts: list[str], by_number: dict[str, int]
) -> int | None:
    """Pick the PDF page for an XML table, checked against the table's own numbers.

    Candidates, in order: the page with a "Table N" caption line, then the page
    containing the caption text. The first candidate on which at least half the
    sampled numbers appear wins. Otherwise the page with the most matching
    numbers (at least 2) is used. If nothing matches, the table is not placed:
    a citation to a guessed page would be wrong.
    """
    candidates = [
        by_number.get((table.table_no or "").upper()),
        find_caption_page(table.caption, page_texts),
    ]
    for page in candidates:
        if page is None:
            continue
        found, sampled = _value_hits(table, page_texts[page - 1])
        if sampled == 0 or found * 2 >= sampled:  # no decimals to check: trust the caption
            return page
    scores = [_value_hits(table, text)[0] for text in page_texts]
    best = max(range(len(scores)), key=lambda i: scores[i], default=None)
    return best + 1 if best is not None and scores[best] >= 2 else None


def jats_tables_with_pages(xml_path: Path, pdf_path: Path) -> list[RawTable]:
    """XML tables with their verified PDF page. Tables that cannot be located are dropped."""
    with pymupdf.open(pdf_path) as doc:
        page_texts = [unicodedata.normalize("NFKC", p.get_text()) for p in doc]
    by_number = caption_pages(page_texts)
    out = []
    for t in parse_jats_tables(xml_path):
        page = locate_table_page(t, page_texts, by_number)
        if page is not None:
            out.append(RawTable(page, t.table_no, t.caption, t.cells, "xml", t.n_header_rows))
    return out


def merge_tables(xml_tables: list[RawTable], pdf_tables: list[RawTable]) -> list[RawTable]:
    """Prefer XML tables; keep PDF tables only for numbers the XML does not have.

    Uncaptioned PDF tables are dropped when XML exists: they are usually
    mis-detections or garbled rotated tables.
    """
    if not xml_tables:
        return pdf_tables
    covered = {t.table_no for t in xml_tables}
    extra = [t for t in pdf_tables if t.table_no and t.table_no not in covered]
    return xml_tables + extra
