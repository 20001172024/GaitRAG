"""Generation evaluation: abstention, citations and latency of the full /ask pipeline.

    python eval/run_generation_eval.py                         # configured LLM
    LLM_PROVIDER=huggingface LLM_MODEL=... python eval/run_generation_eval.py
    python eval/run_generation_eval.py --no-norms              # ablation: no norms lookup
    python eval/run_generation_eval.py --score-citations eval/results/<run>_citation_check.csv

Measures (per run):
- abstention precision / recall / F1 on `unanswerable` questions (positive class);
- answer rate on answerable questions, citations per answer, uncited sentences;
- cited-paper precision: share of cited papers that are gold papers for the question;
- latency p50 / p95 of the whole pipeline.

Citation accuracy (does the cited source support the sentence?) needs a person:
each run writes `<run>_citation_check.csv`; fill `supported` with yes/no and score it
with --score-citations.

RAGAS (faithfulness, answer relevancy, context precision) is not integrated yet:
it needs an LLM judge and will be added once an API key is available.
"""

from __future__ import annotations

import argparse
import csv
import re
import statistics
import time
from pathlib import Path

from _common import add_question_args, questions_from_args
from sqlalchemy.orm import Session

from app.api.search import get_retriever
from app.config import get_settings
from app.db.session import get_engine
from app.evaluation.metrics import abstention_scores, percentile
from app.evaluation.report import RESULTS_DIR, write_report
from app.rag.chunking import split_sentences
from app.rag.generator import CITATION
from app.rag.llm import FakeLLM, LLMError, build_llm
from app.rag.pipeline import answer_question


def stand_in_llm() -> FakeLLM:
    """For testing the script only: cites the first source label it is given."""

    def reply(messages):
        m = re.search(r"^(\[P\d{3}, p\.\d+(?:, Table [^\]]+)?\])", messages[1].content, re.M)
        return (
            f"(stand-in, not a model) First source {m.group(1)}." if m else "INSUFFICIENT_EVIDENCE"
        )

    return FakeLLM(reply, model="stand-in")


def score_citations(path: Path) -> None:
    """Citation accuracy from a filled citation-check sheet."""
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    marked = [r for r in rows if r["supported"].strip().lower() in ("yes", "no")]
    if not marked:
        raise SystemExit(f"No rows marked yes/no in {path}")
    yes = sum(r["supported"].strip().lower() == "yes" for r in marked)
    print(f"Citation accuracy: {yes}/{len(marked)} = {yes / len(marked):.3f} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_question_args(parser)
    parser.add_argument("--no-norms", action="store_true", help="ablation: skip norms lookup")
    parser.add_argument("--stand-in-llm", action="store_true", help="test the script, no API")
    parser.add_argument("--score-citations", type=Path, help="score a filled citation sheet")
    args = parser.parse_args()
    if args.score_citations:
        return score_citations(args.score_citations)

    questions, placeholder = questions_from_args(args)
    if args.stand_in_llm:
        llm, placeholder = stand_in_llm(), True  # never results
    else:
        try:
            llm = build_llm(get_settings())
        except LLMError as exc:
            raise SystemExit(f"{exc}. Set the API key, or use --stand-in-llm to test.") from exc

    retriever = get_retriever()
    details, sheet, latencies = [], [], []
    with Session(get_engine()) as session:
        for q in questions:
            t = time.perf_counter()
            result = answer_question(
                session,
                retriever,
                llm,
                q.question,
                q.features,
                q.prediction,
                use_norms=not args.no_norms,
            )
            latencies.append((time.perf_counter() - t) * 1000)
            g = result.generated
            cited = {c.paper_id for c in result.citations}
            details.append(
                {
                    "id": q.id,
                    "type": q.type,
                    "abstained": g.insufficient_evidence,
                    "reason": g.reason,
                    "attempts": g.attempts,
                    "n_citations": len(g.citations),
                    "uncited_sentences": g.uncited_sentences,
                    "cited_gold": len(cited & set(q.gold_paper_ids)),
                    "cited_papers": " ".join(sorted(cited)),
                    "latency_ms": round(latencies[-1]),
                    "answer": g.answer,
                }
            )
            by_label = {c.label: c for c in result.citations}
            for sentence in split_sentences(g.answer) if not g.insufficient_evidence else []:
                for label in dict.fromkeys(CITATION.findall(sentence)):
                    src = by_label.get(label)
                    sheet.append(
                        {
                            "id": q.id,
                            "sentence": sentence,
                            "label": label,
                            "snippet": src.snippet if src else "",
                            "supported": "",
                        }
                    )
            print(f"{q.id}: {g.reason} ({len(g.citations)} citations)")

    answered = [d for d in details if d["type"] != "unanswerable"]
    abst = abstention_scores(
        [d["abstained"] for d in details], [d["type"] == "unanswerable" for d in details]
    )
    answered_ok = [d for d in answered if not d["abstained"]]
    n_cited = sum(len(d["cited_papers"].split()) for d in answered_ok)
    summary = [
        {
            "model": llm.model,
            "norms": not args.no_norms,
            "abstention P": abst["precision"],
            "abstention R": abst["recall"],
            "abstention F1": abst["f1"],
            "answer rate": len(answered_ok) / len(answered) if answered else 0.0,
            "citations/answer": (
                statistics.mean(d["n_citations"] for d in answered_ok) if answered_ok else 0.0
            ),
            "cited-paper precision": (
                sum(d["cited_gold"] for d in answered_ok) / n_cited if n_cited else 0.0
            ),
            "uncited sentences": sum(d["uncited_sentences"] for d in answered_ok),
            "p50 ms": percentile(latencies, 50),
            "p95 ms": percentile(latencies, 95),
        }
    ]
    name = f"generation_{llm.model.split('/')[-1]}" + ("_no-norms" if args.no_norms else "")
    notes = [
        f"{len(questions)} questions ({len(questions) - len(answered)} unanswerable).",
        "Every citation shown to users is checked against the provided sources; invalid "
        "citations after one retry turn into an insufficient-evidence answer.",
        f"Citation accuracy needs a manual check: fill `supported` in {name}_citation_check.csv "
        "and run with --score-citations.",
        "RAGAS metrics are not included yet (they need an LLM judge).",
    ]
    md, _ = write_report(name, "Generation evaluation", summary, details, notes, placeholder)
    prefix = "PLACEHOLDER_" if placeholder else ""
    sheet_path = RESULTS_DIR / f"{prefix}{name}_citation_check.csv"
    if sheet:
        with sheet_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sheet[0]))
            writer.writeheader()
            writer.writerows(sheet)
    print(f"Wrote {md}" + (f" and {sheet_path}" if sheet else ""))


if __name__ == "__main__":
    main()
