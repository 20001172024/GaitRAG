"""Human review of extracted norm candidates (data/papers/norms_review.csv).

Workflow:
1. Ingestion writes every candidate to the review CSV with an empty `decision`,
   sorted by paper, page and table, with automatic `flags` (plausibility hints).
2. A reviewer sets `decision` to `approve` or `reject`, and may correct
   metric / population / units / mean / sd / low / high or add `notes`.
3. Re-running ingestion merges new candidates in by `candidate_id`. Edits are kept
   for every row with a decision or a note; untouched rows are refreshed.
4. Approved rows are loaded into the `norms` table, only if their `value_text`
   still appears exactly in a stored table cell on that page.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.db.models import Norm, PaperTable
from app.rag.manifest import MANIFEST_PATH
from app.rag.tables import NormCandidate

REVIEW_PATH = MANIFEST_PATH.parent / "norms_review.csv"

REVIEW_COLUMNS = [
    "candidate_id",
    "decision",
    "flags",
    "paper_id",
    "page",
    "table_no",
    "metric",
    "population",
    "units",
    "value_text",
    "mean",
    "sd",
    "low",
    "high",
    "kind",
    "row_label",
    "column_header",
    "caption",
    "notes",
]
REVIEWER_FIELDS = ("decision", "metric", "population", "units", "mean", "sd", "low", "high")
REVIEWER_FIELDS += ("notes",)
DECISIONS = ("", "approve", "reject")


def _fmt(v: object) -> str:
    return "" if v is None else str(v)


# Plausibility bounds per (metric, unit family) for *flagging only*. These are
# rough physical limits chosen by the project, not published norms: a value
# outside them is not wrong, it is just worth a second look.
_BOUNDS: dict[tuple[str, str], tuple[float, float]] = {
    ("cadence", "per_min"): (40, 180),
    ("cadence", "per_s"): (0.7, 3.0),
    ("stride_time", "s"): (0.5, 3.0),
    ("stride_time", "ms"): (500, 3000),
    ("stride_time_cv", "%"): (0, 20),
    ("stance_pct", "%"): (40, 80),
    ("swing_pct", "%"): (20, 60),
    ("gait_speed", "m/s"): (0.1, 3.0),
    ("gait_speed", "cm/s"): (10, 300),
    ("gait_speed", "km/h"): (0.4, 11),
    ("symmetry_index", "%"): (0, 50),
    ("symmetry_index_stance_time", "%"): (0, 50),
    ("symmetry_index_peak_pressure", "%"): (0, 50),
}
_NON_NORMATIVE = re.compile(
    r"reliab|regress|correlat|\bicc\b|agreement|validat|session|retest|repeatab|accuracy",
    re.IGNORECASE,
)


def _unit_family(metric: str, units: str) -> str | None:
    u = units.lower().replace(" ", "").replace("−", "-")
    if metric == "cadence":
        if re.search(r"/s(ec)?$|s-1|s−1$|persec", u) and "min" not in u:
            return "per_s"
        return "per_min" if "min" in u else None
    if metric == "gait_speed":
        for fam in ("cm/s", "km/h", "m/s"):
            if fam in u:
                return fam
        return None
    if metric == "stride_time":
        return "ms" if u.startswith("ms") or "millisec" in u else "s" if u.startswith("s") else None
    return "%" if "%" in u else None


def plausibility_flags(row: dict[str, str]) -> str:
    """Semicolon-separated hints for the reviewer (never a decision)."""
    flags = []
    metric, units = row.get("metric", ""), row.get("units", "")
    values = {k: _float(row.get(k, "")) for k in ("mean", "sd", "low", "high")}
    if not units:
        flags.append("no units")
    family = _unit_family(metric, units) if units else None
    if units and family is None:
        flags.append(f"unusual units for {metric}")
    if family == "per_s":
        flags.append("cadence in steps/s (convert?)")
    bounds = _BOUNDS.get((metric, family)) if family else None
    if bounds:
        for key in ("mean", "low", "high"):
            v = values[key]
            if v is not None and not bounds[0] <= v <= bounds[1]:
                flags.append(f"{key} {v:g} outside plausible {bounds[0]:g}-{bounds[1]:g}")
                break
    if values["mean"] is not None and values["sd"] is not None and values["sd"] > values["mean"]:
        flags.append("sd > mean")
    if metric == "symmetry_index":
        flags.append("generic symmetry index: set metric to the specific quantity to use it")
    if not row.get("population", "").strip():
        flags.append("no population")
    context = " ".join(row.get(k, "") for k in ("caption", "column_header", "row_label"))
    if _NON_NORMATIVE.search(context):
        flags.append("reliability/regression/validation table: is this a normative value?")
    return "; ".join(flags)


def candidate_row(c: NormCandidate) -> dict[str, str]:
    """A fresh (undecided) review row for a candidate."""
    row = {col: _fmt(getattr(c, col, None)) for col in REVIEW_COLUMNS}
    row["decision"] = ""
    row["notes"] = ""
    return row


def read_review(path: Path = REVIEW_PATH) -> list[dict[str, str]]:
    """Read the review CSV (empty list if it does not exist yet)."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge_review(
    candidates: list[NormCandidate], existing: list[dict[str, str]]
) -> list[dict[str, str]]:
    """Merge freshly extracted candidates with existing review rows.

    Reviewer fields are kept for known candidate IDs that have a decision or a
    note; untouched rows are refreshed from extraction. Decided rows that were not
    re-extracted are kept (so a decision is never silently lost); undecided rows
    that disappeared are dropped.
    """
    by_id = {r["candidate_id"]: r for r in existing}
    merged, seen = [], set()
    for c in candidates:
        row = candidate_row(c)
        previous = by_id.get(c.candidate_id)
        # Keep edits only for rows a reviewer has touched (decision or note). Untouched
        # rows take the fresh extraction, so fixes to the extractor reach them.
        if previous and (previous.get("decision") or previous.get("notes")):
            for field in REVIEWER_FIELDS:
                row[field] = previous.get(field, "")
        merged.append(row)
        seen.add(c.candidate_id)
    merged += [r for r in existing if r["candidate_id"] not in seen and r.get("decision")]
    return merged


def write_review(rows: list[dict[str, str]], path: Path = REVIEW_PATH) -> None:
    """Write the review CSV, sorted by paper, page and table."""
    rows = sorted(rows, key=lambda r: (r["paper_id"], int(r["page"] or 0), r["table_no"]))
    for r in rows:  # recomputed on every write, so flags follow reviewer edits
        r["flags"] = plausibility_flags(r)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _float(v: str) -> float | None:
    return float(v) if v.strip() else None


def load_approved_norms(session: Session, rows: list[dict[str, str]]) -> tuple[int, list[str]]:
    """Replace the `norms` table with the approved review rows.

    Returns:
        (number loaded, list of problems). A row is rejected if its decision is
        invalid or its value_text is not found verbatim in that page's tables.
    """
    problems: list[str] = []
    session.execute(delete(Norm))
    loaded = 0
    for r in rows:
        decision = r.get("decision", "").strip().lower()
        if decision not in DECISIONS:
            problems.append(f"{r['candidate_id']}: unknown decision {r['decision']!r}")
            continue
        if decision != "approve":
            continue
        page = int(r["page"])
        cells = session.scalars(
            select(PaperTable.cells).where(
                PaperTable.paper_id == r["paper_id"], PaperTable.page == page
            )
        ).all()
        found = any(r["value_text"] == cell for table in cells for row in table for cell in row)
        if not found:
            problems.append(
                f"{r['candidate_id']}: value {r['value_text']!r} not found in "
                f"{r['paper_id']} p.{page} tables"
            )
            continue
        session.add(
            Norm(
                paper_id=r["paper_id"],
                page=page,
                table_no=r["table_no"] or None,
                metric=r["metric"],
                population=r["population"] or None,
                value_text=r["value_text"],
                mean=_float(r["mean"]),
                sd=_float(r["sd"]),
                low=_float(r["low"]),
                high=_float(r["high"]),
                units=r["units"] or None,
            )
        )
        loaded += 1
    return loaded, problems
