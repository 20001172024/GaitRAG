"""Shared setup for the evaluation scripts (run from the repo root)."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.evaluation.questions import EvalQuestion, is_placeholder, load_questions
from app.rag.manifest import load_manifest

EVAL_DIR = Path(__file__).resolve().parent
DEFAULT_QUESTIONS = EVAL_DIR / "questions.jsonl"


def add_question_args(parser: argparse.ArgumentParser) -> None:
    """--questions and --allow-placeholder."""
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--allow-placeholder",
        action="store_true",
        help="run on a placeholder/example file (output is marked PLACEHOLDER, never results)",
    )


def questions_from_args(args: argparse.Namespace) -> tuple[list[EvalQuestion], bool]:
    """Load and validate questions; refuse placeholder files unless explicitly allowed."""
    path: Path = args.questions
    if not path.exists():
        raise SystemExit(
            f"{path} not found. The evaluation set is written by the project owner; "
            "to test the scripts use --questions eval/questions.example.jsonl --allow-placeholder"
        )
    placeholder = is_placeholder(path)
    if placeholder and not args.allow_placeholder:
        raise SystemExit(f"{path} is a placeholder file; add --allow-placeholder to test with it")
    known = {e.paper_id for e in load_manifest()}
    return load_questions(path, known_papers=known), placeholder
