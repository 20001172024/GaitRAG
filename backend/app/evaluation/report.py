"""Write evaluation results as Markdown + CSV in eval/results/."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[3] / "eval" / "results"

PLACEHOLDER_BANNER = (
    "> **PLACEHOLDER QUESTIONS.** These numbers come from `eval/questions.example.jsonl`, "
    "a handful of example questions used only to test the scripts. They are **not** "
    "results and must not be reported.\n"
)


def write_report(
    name: str,
    title: str,
    summary_rows: list[dict],
    detail_rows: list[dict],
    notes: list[str],
    placeholder: bool,
    out_dir: Path = RESULTS_DIR,
) -> tuple[Path, Path]:
    """Write `<name>.md` (summary table + notes) and `<name>.csv` (per-question rows).

    Placeholder runs get a PLACEHOLDER_ prefix (git-ignored) and a warning banner.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"PLACEHOLDER_{name}" if placeholder else name
    md_path, csv_path = out_dir / f"{stem}.md", out_dir / f"{stem}.csv"

    lines = [f"# {title}", ""]
    if placeholder:
        lines += [PLACEHOLDER_BANNER]
    lines += [f"Generated {datetime.now(UTC).isoformat(timespec='seconds')}", ""]
    if summary_rows:
        cols = list(summary_rows[0])
        lines += ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for row in summary_rows:
            cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row.values()]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [f"- {n}" for n in notes]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if detail_rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(detail_rows[0]))
            writer.writeheader()
            writer.writerows(detail_rows)
    return md_path, csv_path
