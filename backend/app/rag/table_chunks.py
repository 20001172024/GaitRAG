"""Tables as retrievable text chunks (kind = "table").

Why: normative values live in tables, which are kept out of the prose chunks
(flattened cells are number soup). Writing each table out row by row, under its
caption, lets retrieval find "cadence reference values in older adults" in
P001 Table 3 and cite it as [P001, p.11, Table 3].

Format, one line per row:
    Table 3 Quantitative reference values ... (caption)
    STRIDE TIME Mean value (ms) — Total: 1123.7 ± 122.4; Female (n = 437): 1095.5 ± 109.8
Large tables are split into row groups under MAX_TOKENS, each repeating the caption.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.rag.chunking import MAX_TOKENS, TokenCounter
from app.rag.parsing import RawTable
from app.rag.tables import column_headers, section_title


@dataclass(frozen=True)
class TableChunk:
    """A table (or part of one) as text."""

    page: int
    table_no: str | None
    text: str
    token_count: int


def table_lines(table: RawTable) -> list[str]:
    """One line of text per data row: "label — header: value; header: value"."""
    headers, first_data = column_headers(table.cells, table.n_header_rows)
    lines, section = [], None
    for row in table.cells[first_data:]:
        title = section_title(row)
        if title is not None:
            section = title
            continue
        label = (row[0] or "").strip()
        if section and label:
            label = f"{section} {label}"
        pairs = [
            f"{headers[j]}: {cell}" if j < len(headers) and headers[j] else cell
            for j, cell in enumerate(row[1:], start=1)
            if cell and cell != row[0]
        ]
        if pairs:
            lines.append(f"{label} — {'; '.join(pairs)}" if label else "; ".join(pairs))
    return lines


def _fit_lines(lines: list[str], count_tokens: TokenCounter, budget: int) -> list[str]:
    """Split any line over `budget` at its "; " pairs, repeating the row label.

    A very wide row can exceed the embedding model's input limit on its own and
    would be truncated at embedding time.
    """
    out = []
    for line in lines:
        if count_tokens(line) <= budget:
            out.append(line)
            continue
        label, sep, rest = line.partition(" — ")
        prefix = f"{label}{sep}" if sep else ""
        pairs = (rest if sep else line).split("; ")
        part: list[str] = []
        for pair in pairs:
            candidate = prefix + "; ".join([*part, pair])
            if part and count_tokens(candidate) > budget:
                out.append(prefix + "; ".join(part))
                part = []
            part.append(pair)
        if part:
            out.append(prefix + "; ".join(part))
    return out


def table_to_chunks(
    table: RawTable, count_tokens: TokenCounter, max_tokens: int = MAX_TOKENS
) -> list[TableChunk]:
    """Split a table's lines into chunks of at most ~max_tokens, each led by the caption."""
    caption = table.caption or (f"Table {table.table_no}" if table.table_no else "Table")
    chunks: list[TableChunk] = []
    current: list[str] = []

    def emit() -> None:
        text = "\n".join([caption, *current])
        chunks.append(TableChunk(table.page, table.table_no, text, count_tokens(text)))

    budget = max_tokens - count_tokens(caption)
    used = 0
    for line in _fit_lines(table_lines(table), count_tokens, budget):
        n = count_tokens(line)
        if current and used + n > budget:
            emit()
            current, used = [], 0
        current.append(line)
        used += n
    if current:
        emit()
    return chunks
