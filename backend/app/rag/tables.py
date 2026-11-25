"""Turn raw paper tables into *candidate* normative values for human review.

Why rules + human review (not an LLM): these values are shown to users as
published norms, so a wrongly parsed number is a factual error. Rules are
predictable, every candidate keeps the verbatim cell text it came from, and
only rows a reviewer approves reach the `norms` table.

Supported layouts:
- metric in the row label, one column per group/condition
  ("cadence [steps/min]" | "1.68 ± 0.14" | ...);
- metric in the column header, one row per group/condition
  ("Speed 1.6 km/h" | "54.5 (8.2)" under "Cadence").
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from app.rag.parsing import RawTable

# Order matters: more specific patterns first.
METRIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (m, re.compile(p, re.IGNORECASE))
    for m, p in [
        ("symmetry_index", r"symmetry|asymmetry"),
        ("stride_time_cv", r"stride[- ]?time.*(variab|\bcv\b|\bcov\b|coefficient)"),
        ("stride_time_cv", r"(\bcv\b|\bcov\b|coefficient of variation).*stride[- ]?time"),
        ("double_support", r"double[- ]?(limb )?(support|stance)"),
        ("single_support", r"single[- ]?(limb )?support"),
        ("stride_time", r"stride[- ]?(time|duration)|(gait )?cycle (time|duration)"),
        ("step_time", r"step[- ]?(time|duration)"),
        ("cadence", r"cadence"),
        ("stance", r"stance"),
        ("swing", r"swing"),
        ("gait_speed", r"(gait|walking) (speed|velocity)|^\s*\d*\.?\s*(speed|velocity)\b"),
    ]
]

# Columns holding test statistics or measurement reliability, not normative values.
_STAT_COLUMN = re.compile(
    r"p[- ]?val|^p$|\bp\s*[<=]|\bicc\b|\bsem\b|\bmdc\b|\bloa\b|limits of agreement|"
    r"correlat|^r$|\bbeta\b|β|\b(95\s*%\s*)?ci\b|ic\s*95|confidence|cohen|effect size|"
    r"\bdiff|\bbias\b|\bt[- ]?test|\banova\b|odds ratio|\bor\b|r2|r²|adjusted|coef|"
    r"standard error|\bauc\b|\broc\b|sensitiv|specific|\bn to\b|raw|norm\b",
    re.IGNORECASE,
)
_NUM = r"-?\d+(?:\.\d+)?"
_MARKERS = r"[*∗#†‡§¶a-d]*"
_VALUE_PATTERNS = [
    ("mean_sd", re.compile(rf"({_NUM})\s*(?:±|\+/-|\+-)\s*({_NUM}){_MARKERS}")),
    ("mean_range", re.compile(rf"({_NUM})\s*[(\[]\s*({_NUM})\s*[–—-]\s*({_NUM})\s*[)\]]")),
    ("mean_paren", re.compile(rf"({_NUM}){_MARKERS}\s*\(\s*({_NUM})\s*\)")),
    ("range", re.compile(rf"^[(\[]?\s*({_NUM})\s*[–—]\s*({_NUM})\s*[)\]]?$")),
    ("plain", re.compile(rf"^({_NUM}){_MARKERS}$")),
]
_MEAN_HEADER = re.compile(r"mean|median|average|μ", re.IGNORECASE)
# "Speed 1.6 km/h" or "at 4 km/h" is a test condition, not a measured value.
_CONDITION = re.compile(r"\d+(?:\.\d+)?\s*(?:km/h|m/s|kmh)", re.IGNORECASE)
_MULTI_VALUE = re.compile(rf"{_NUM}\s*±.*{_NUM}\s*±")

RELEVANT_METRICS = frozenset(
    {
        "cadence",
        "stride_time",
        "stride_time_cv",
        "stance_pct",
        "swing_pct",
        "symmetry_index",
        "gait_speed",
    }
)
"""Metrics that GaitRAG features are compared against; others are not extracted."""

_UNITS = re.compile(r"[(\[]\s*([^()\[\]]{1,20}?)\s*[)\]]\s*\.?$")


@dataclass(frozen=True)
class ParsedValue:
    """Numbers parsed from one cell."""

    kind: str  # mean_sd | mean_range | mean_paren | range | plain
    mean: float | None = None
    sd: float | None = None
    low: float | None = None
    high: float | None = None
    cell_label: str | None = None  # e.g. "LWS" in "LWS 1.04±0.12"


@dataclass(frozen=True)
class NormCandidate:
    """One candidate normative value, awaiting human review."""

    candidate_id: str
    paper_id: str
    page: int
    table_no: str | None
    caption: str | None
    row_label: str
    column_header: str
    metric: str
    units: str | None
    value_text: str  # verbatim cell
    kind: str
    mean: float | None
    sd: float | None
    low: float | None
    high: float | None
    population: str


def match_metric(label: str) -> str | None:
    """Canonical metric name for a row label / column header, if recognized."""
    if _CONDITION.search(label) and not re.search("cadence|stride|stance|swing", label, re.I):
        return None
    for metric, pattern in METRIC_PATTERNS:
        if pattern.search(label):
            return metric
    return None


def parse_value(cell: str) -> ParsedValue | None:
    """Parse a table cell like "1.04 ± 0.12", "54.5 (8.2)", "95–115" or "112.3"."""
    text = cell.strip()
    label = None
    lead = re.match(r"^([A-Za-z][A-Za-z ]{0,15}?)\s+(?=-?\d)", text)
    if lead:
        label, text = lead.group(1), text[lead.end() :]
    for kind, pattern in _VALUE_PATTERNS:
        m = pattern.search(text) if kind in ("mean_sd", "mean_range", "mean_paren") else None
        m = m or (pattern.match(text) if kind in ("range", "plain") else None)
        if not m:
            continue
        nums = [float(g) for g in m.groups()]
        if kind == "mean_sd" or kind == "mean_paren":
            return ParsedValue(kind, mean=nums[0], sd=nums[1], cell_label=label)
        if kind == "mean_range":
            return ParsedValue(kind, mean=nums[0], low=nums[1], high=nums[2], cell_label=label)
        if kind == "range":
            return ParsedValue(kind, low=nums[0], high=nums[1], cell_label=label)
        return ParsedValue(kind, mean=nums[0], cell_label=label)
    return None


def units_of(label: str) -> str | None:
    """Units in trailing parentheses/brackets: "cadence [steps/min]" -> "steps/min"."""
    m = _UNITS.search(label)
    return m.group(1).strip() if m else None


def refine_metric(metric: str, units: str | None) -> str:
    """Split stance/swing into % of cycle vs. seconds using the units."""
    if metric in ("stance", "swing", "double_support", "single_support"):
        if units and "%" in units:
            return f"{metric}_pct"
        if units and re.fullmatch(r"m?s(ec)?", units.strip().lower()):
            return f"{metric}_time"
    return metric


def _is_data_row(row: list[str | None]) -> bool:
    return any(c and parse_value(c) for c in row[1:])


def _column_headers(cells: list[list[str | None]]) -> tuple[list[str], int]:
    """Join the header rows (rows before the first data row) per column."""
    first_data = next((i for i, row in enumerate(cells) if _is_data_row(row)), len(cells))
    n_cols = max((len(r) for r in cells), default=0)
    headers = []
    for j in range(n_cols):
        parts = [r[j] for r in cells[:first_data] if j < len(r) and r[j] and r[j] not in "="]
        headers.append(" ".join(dict.fromkeys(parts)))  # dedupe repeated spanning headers
    return headers, first_data


def _candidate_id(*parts: object) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12]


def extract_candidates(paper_id: str, table: RawTable) -> list[NormCandidate]:
    """Extract candidate norm values from one table (both orientations)."""
    headers, first_data = _column_headers(table.cells)
    out: list[NormCandidate] = []
    for row in table.cells[first_data:]:
        row_label = (row[0] or "").strip() if row else ""
        row_metric = match_metric(row_label) if row_label else None
        for j, cell in enumerate(row[1:], start=1):
            header = headers[j] if j < len(headers) else ""
            if not cell or _STAT_COLUMN.search(header) or _MULTI_VALUE.search(cell):
                continue
            value = parse_value(cell)
            if value is None or (value.kind == "plain" and not _MEAN_HEADER.search(header)):
                continue
            metric = row_metric or match_metric(header)
            if metric is None:
                continue
            # Metric from the row -> group from the column, and vice versa.
            units = units_of(row_label) if row_metric else units_of(header)
            group = header if row_metric else row_label
            if re.search(r"\bcv\b|coefficient of variation", header, re.IGNORECASE):
                metric = "stride_time_cv" if metric == "stride_time" else f"{metric}_cv"
            population = " | ".join(p for p in (group, value.cell_label) if p)
            metric = refine_metric(metric, units)
            if metric not in RELEVANT_METRICS:
                continue
            out.append(
                NormCandidate(
                    candidate_id=_candidate_id(paper_id, table.page, table.table_no, row_label, j),
                    paper_id=paper_id,
                    page=table.page,
                    table_no=table.table_no,
                    caption=table.caption,
                    row_label=row_label,
                    column_header=header,
                    metric=metric,
                    units=units,
                    value_text=cell,
                    kind=value.kind,
                    mean=value.mean,
                    sd=value.sd,
                    low=value.low,
                    high=value.high,
                    population=population,
                )
            )
    return out
