"""Tests for evaluation question loading, metrics and reports."""

import json
from pathlib import Path

import pytest

from app.evaluation.metrics import (
    abstention_scores,
    best_gate_threshold,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.questions import QuestionError, is_placeholder, load_questions
from app.evaluation.report import write_report


def test_recall_at_k_counts_gold_papers_in_top_k_chunks() -> None:
    ranked = ["P001", "P001", "P008", "P003", "P009"]
    assert recall_at_k(ranked, ["P008", "P003"], k=3) == 0.5
    assert recall_at_k(ranked, ["P008", "P003"], k=5) == 1.0
    with pytest.raises(ValueError):
        recall_at_k(ranked, [], k=5)


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["P001", "P008", "P003"], ["P003", "P008"]) == 0.5
    assert reciprocal_rank(["P001"], ["P009"]) == 0.0


def test_percentile_interpolates() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([10.0], 95) == 10.0


def test_abstention_scores_use_unanswerable_as_positive_class() -> None:
    s = abstention_scores(
        abstained=[True, True, False, False], unanswerable=[True, False, True, False]
    )
    assert (s["precision"], s["recall"], s["tp"], s["fp"], s["fn"]) == (0.5, 0.5, 1, 1, 1)
    assert abstention_scores([False], [False])["f1"] == 0.0


def test_best_gate_threshold_separates_low_scores() -> None:
    best = best_gate_threshold([-5.0, -4.0, 2.0, 3.0], [True, True, False, False])
    assert -4.0 < best["threshold"] < 2.0 and best["f1"] == 1.0
    assert best_gate_threshold([1.0, 2.0], [False, False]) is None


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


GOOD = {"id": "q1", "question": "Cadence norms?", "gold_paper_ids": ["P003"], "type": "norm_lookup"}


def test_load_questions(tmp_path: Path) -> None:
    rows = [GOOD, {"id": "q2", "question": "Capital of France?", "type": "unanswerable"}]
    qs = load_questions(_write(tmp_path / "questions.jsonl", rows), known_papers={"P003"})
    assert [(q.id, q.answerable) for q in qs] == [("q1", True), ("q2", False)]


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (GOOD | {"type": "trivia"}, "unknown type"),
        (GOOD | {"gold_paper_ids": []}, "need gold_paper_ids"),
        (GOOD | {"gold_paper_ids": ["P999"]}, "not in the manifest"),
        (GOOD | {"type": "unanswerable"}, "must have no gold papers"),
        ({"id": "q1", "type": "norm_lookup"}, "missing field"),
    ],
)
def test_invalid_questions_are_rejected(tmp_path: Path, row: dict, message: str) -> None:
    with pytest.raises(QuestionError, match=message):
        load_questions(_write(tmp_path / "q.jsonl", [row]), known_papers={"P003"})


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(QuestionError, match="duplicate"):
        load_questions(_write(tmp_path / "q.jsonl", [GOOD, GOOD]), known_papers={"P003"})


def test_placeholder_files_are_detected() -> None:
    assert is_placeholder(Path("eval/questions.example.jsonl"))
    assert is_placeholder(Path("eval/questions.draft.jsonl"))
    assert not is_placeholder(Path("eval/questions.jsonl"))


def test_placeholder_reports_are_prefixed_and_bannered(tmp_path: Path) -> None:
    md, csv_path = write_report(
        "retrieval",
        "Retrieval",
        [{"config": "hybrid", "recall@5": 0.5}],
        [{"id": "q1"}],
        [],
        True,
        tmp_path,
    )
    assert md.name == "PLACEHOLDER_retrieval.md" and csv_path.name == "PLACEHOLDER_retrieval.csv"
    text = md.read_text()
    assert "NOT REPORTABLE" in text and "must not be reported" in " ".join(text.split())
    assert "| hybrid | 0.500 |" in text
