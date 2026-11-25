"""Evaluation questions (eval/questions.jsonl), written by the project owner.

One JSON object per line:
{"id": "q001", "question": "...", "gold_paper_ids": ["P012"], "gold_answer": "...",
 "type": "norm_lookup", "features": {...}?, "prediction": "normal"?}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

QUESTION_TYPES = ("norm_lookup", "method_comparison", "feature_interpretation", "unanswerable")


class QuestionError(ValueError):
    """Invalid evaluation question file."""


@dataclass(frozen=True)
class EvalQuestion:
    """One evaluation question."""

    id: str
    question: str
    gold_paper_ids: tuple[str, ...]
    gold_answer: str
    type: str
    features: dict[str, float] | None = None
    prediction: str | None = None

    @property
    def answerable(self) -> bool:
        """False for questions the corpus cannot answer (abstention tests)."""
        return self.type != "unanswerable"


def is_placeholder(path: Path) -> bool:
    """Example/placeholder files must never produce reportable results."""
    return "example" in path.name or "placeholder" in path.name


def load_questions(path: Path, known_papers: set[str] | None = None) -> list[EvalQuestion]:
    """Read and validate a questions JSONL file.

    Raises:
        QuestionError: on a malformed line, duplicate id, unknown type, a gold paper
            not in the manifest, or gold papers on an unanswerable question.
    """
    questions, seen = [], set()
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise QuestionError(f"line {n}: invalid JSON ({exc.msg})") from exc
        try:
            q = EvalQuestion(
                id=str(raw["id"]),
                question=str(raw["question"]),
                gold_paper_ids=tuple(raw.get("gold_paper_ids", [])),
                gold_answer=str(raw.get("gold_answer", "")),
                type=str(raw["type"]),
                features=raw.get("features"),
                prediction=raw.get("prediction"),
            )
        except KeyError as exc:
            raise QuestionError(f"line {n}: missing field {exc}") from exc
        if q.id in seen:
            raise QuestionError(f"line {n}: duplicate id {q.id}")
        seen.add(q.id)
        if q.type not in QUESTION_TYPES:
            raise QuestionError(f"{q.id}: unknown type {q.type!r} (use {QUESTION_TYPES})")
        if q.answerable and not q.gold_paper_ids:
            raise QuestionError(f"{q.id}: answerable questions need gold_paper_ids")
        if not q.answerable and q.gold_paper_ids:
            raise QuestionError(f"{q.id}: unanswerable questions must have no gold papers")
        unknown = set(q.gold_paper_ids) - (known_papers or set(q.gold_paper_ids))
        if unknown:
            raise QuestionError(f"{q.id}: gold papers not in the manifest: {sorted(unknown)}")
        questions.append(q)
    if not questions:
        raise QuestionError(f"{path} has no questions")
    return questions
