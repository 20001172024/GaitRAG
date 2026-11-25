"""PDF -> clean per-page text with section labels.

Decisions (checked against the corpus PDFs):
- Text comes page by page so every passage keeps its page number for citations.
- PyMuPDF's default (content-stream) order keeps two-column layouts in
  reading order; sorting blocks by position would interleave the columns.
- Running headers/footers ("Frontiers in ... | Article 353", "Page 2 of 6")
  are lines near a page edge that repeat on >= half the pages, compared with
  digits masked so changing page numbers still match.
- NFKC normalization turns ligatures (U+FB01 "fi") into plain letters; otherwise
  keyword search would miss "findings".
- Words hyphenated at line ends are re-joined when the next line starts lowercase.
- Section headings are matched against standard section names. Font cues
  (bold/size) proved unreliable across publishers.
- Text inside detected tables is removed from the prose (it becomes a soup
  of numbers that hurts both embeddings and keyword search). Tables are
  extracted separately, as cells, with the same detector (PyMuPDF find_tables).
- References and back-matter (funding, author contributions, conflicts of
  interest...) are dropped: they repeat topic words without content and would
  pollute keyword search.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf

EDGE_LINES = 4  # lines at the top/bottom of a page checked for running headers
REPEAT_FRACTION = 0.5

_SECTIONS = {
    "abstract": "Abstract",
    "background": "Background",
    "introduction": "Introduction",
    "methods": "Methods",
    "method": "Methods",
    "materials and methods": "Methods",
    "material and methods": "Methods",
    "methods and materials": "Methods",
    "results": "Results",
    "results and discussion": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "limitations": "Discussion",
}
_REFERENCES = ("references", "bibliography", "literature cited")
"""Terminal: everything after this heading is dropped."""
_BACK_MATTER = (
    "acknowledgement",
    "acknowledgment",
    "funding",
    "author contribution",
    "authors' contribution",
    "competing interest",
    "conflict of interest",
    "conflicts of interest",
    "data availability",
    "ethics statement",
    "supplementary material",
    "abbreviations",
)
_NUMBERING = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[IVX]+\.)\s+")
# Publisher-specific top-level headings, e.g. "2. The Pressure-Sensitive Foot Insole".
# Single-level numbers only: "2.1. Sample Size" is a subsection and is ignored.
# No commas: they mark reference entries ("12. Selles RW, Formanoy MA"), not headings.
_NUMBERED_TOP = re.compile(r"^(\d{1,2})\.?\s+([A-Z][A-Za-z][A-Za-z &()/:\-–]*)$")


@dataclass(frozen=True)
class Segment:
    """Contiguous text on one page under one section heading."""

    page: int  # 1-based
    section: str | None
    text: str


def _mask_digits(line: str) -> str:
    return re.sub(r"\d+", "#", line)


def _heading(line: str) -> str | None:
    """Return the canonical section name if `line` is a section heading, else None.

    Returns "__references__" for the reference list and "__back__" for other
    back-matter headings.
    """
    if len(line) > 60 or line.endswith((".", ",", ";")):
        return None
    key = _NUMBERING.sub("", line).strip().rstrip(":").lower()
    if key in _SECTIONS:
        return _SECTIONS[key]
    if key in _REFERENCES:
        return "__references__"
    if any(key.startswith(b) for b in _BACK_MATTER) and len(key.split()) <= 5:
        return "__back__"
    numbered = _NUMBERED_TOP.match(line)
    if numbered and len(numbered.group(2).split()) <= 8:
        return numbered.group(2).strip()
    return None


def _repeated_edge_lines(pages: list[list[str]]) -> set[str]:
    counts: Counter[str] = Counter()
    for lines in pages:
        edges = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        counts.update({_mask_digits(line) for line in edges})
    threshold = max(2, REPEAT_FRACTION * len(pages))
    return {line for line, c in counts.items() if c >= threshold}


def _is_page_number(line: str) -> bool:
    return bool(re.fullmatch(r"(page\s*)?\d{1,4}(\s*of\s*\d{1,4})?", line, re.IGNORECASE))


def join_lines(lines: list[str]) -> str:
    """Join wrapped lines into prose, re-joining words hyphenated at line ends."""
    out = ""
    for line in lines:
        if not out:
            out = line
        elif out.endswith("-") and len(out) > 1 and out[-2].isalpha() and line[:1].islower():
            out = out[:-1] + line
        else:
            out += " " + line
    return re.sub(r"\s+", " ", out).strip()


def clean_pages(raw_pages: list[str]) -> list[Segment]:
    """Turn raw page texts into section-labelled segments (pure function, testable)."""
    pages = [
        [ln.strip() for ln in unicodedata.normalize("NFKC", text).splitlines() if ln.strip()]
        for text in raw_pages
    ]
    repeated = _repeated_edge_lines(pages)

    segments: list[Segment] = []
    section: str | None = None
    in_back_matter = False

    def emit(page_no: int, buffer: list[str]) -> None:
        text = join_lines(buffer)
        if text and not in_back_matter:
            segments.append(Segment(page_no, section, text))

    for page_no, lines in enumerate(pages, start=1):
        buffer: list[str] = []
        for i, line in enumerate(lines):
            at_edge = i < EDGE_LINES or i >= len(lines) - EDGE_LINES
            if (at_edge and _mask_digits(line) in repeated) or _is_page_number(line):
                continue
            heading = _heading(line)
            if heading is None:
                buffer.append(line)
                continue
            emit(page_no, buffer)
            buffer = []
            if heading == "__references__":
                return segments
            in_back_matter = heading == "__back__"
            if not in_back_matter:
                section = heading
        emit(page_no, buffer)
    return segments


@dataclass(frozen=True)
class RawTable:
    """A table found on a PDF page."""

    page: int  # 1-based
    table_no: str | None  # from the caption, e.g. "2"
    caption: str | None
    cells: list[list[str | None]]


@dataclass(frozen=True)
class ParsedPDF:
    """Everything extracted from one PDF."""

    segments: list[Segment]
    tables: list[RawTable]
    n_pages: int


_CAPTION = re.compile(r"^\s*Table\s+([0-9]+|[IVX]+)\b[.:]?\s*(.*)", re.IGNORECASE | re.DOTALL)
CAPTION_MAX_GAP_PT = 60.0  # how far above/below a table its caption may be


def _find_caption(blocks: list[tuple], bbox: pymupdf.Rect) -> tuple[str | None, str | None]:
    """Nearest "Table N ..." text block above or below the table.

    Captions above win ties: most journals put table captions above the table.
    """
    best: tuple[float, str, str] | None = None
    for b in blocks:
        match = _CAPTION.match(b[4])
        if not match:
            continue
        r = pymupdf.Rect(b[:4])
        if r.y1 <= bbox.y0 + 2:  # caption above the table
            score = bbox.y0 - r.y1
        elif r.y0 >= bbox.y1 - 2:  # caption below: penalize so "above" is preferred
            score = (r.y0 - bbox.y1) + CAPTION_MAX_GAP_PT
        else:
            continue
        if score <= 2 * CAPTION_MAX_GAP_PT and (best is None or score < best[0]):
            text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", b[4])).strip()
            best = (score, match.group(1), text)
    return (best[1], best[2]) if best else (None, None)


def _page_text_and_tables(page: pymupdf.Page, page_no: int) -> tuple[str, list[RawTable]]:
    found = page.find_tables().tables
    boxes = [pymupdf.Rect(t.bbox) for t in found]
    blocks = [b for b in page.get_text("blocks") if b[6] == 0]  # text blocks, stream order

    def inside_table(b: tuple) -> bool:
        r = pymupdf.Rect(b[:4])
        area = r.get_area()
        return area > 0 and any(r.intersect(x).get_area() > 0.5 * area for x in boxes)

    text = "\n".join(b[4] for b in blocks if not inside_table(b))
    tables = []
    for t, box in zip(found, boxes, strict=True):
        table_no, caption = _find_caption(blocks, box)
        cells = [
            [unicodedata.normalize("NFKC", c).strip() if c else None for c in row]
            for row in t.extract()
        ]
        tables.append(RawTable(page_no, table_no, caption, cells))
    return text, tables


def parse_pdf(path: Path) -> ParsedPDF:
    """Extract clean, section-labelled prose and raw tables from a PDF."""
    raw_pages, tables = [], []
    with pymupdf.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            text, page_tables = _page_text_and_tables(page, page_no)
            raw_pages.append(text)
            tables.extend(page_tables)
    return ParsedPDF(clean_pages(raw_pages), tables, len(raw_pages))
