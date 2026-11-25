"""Retrieval evaluation: Recall@5, Recall@10 and MRR for each ablation configuration.

    python eval/run_retrieval_eval.py                       # eval/questions.jsonl
    python eval/run_retrieval_eval.py --reranker BAAI/bge-reranker-base
    python eval/run_retrieval_eval.py --questions eval/questions.example.jsonl --allow-placeholder

Configurations (the ablation table):
  1 vector                  pgvector only
  2 bm25                    BM25 only
  3 hybrid                  RRF of vector + BM25
  4 hybrid+rerank           + cross-encoder reranking
  5 hybrid+rerank+tables    + table chunks (normative tables as retrievable text)

Retrieval metrics use answerable questions only. The reranker scores of every
question (unanswerable included) calibrate the insufficient-evidence gate
(MIN_RERANK_SCORE).
"""

from __future__ import annotations

import argparse
import statistics
import time

from _common import add_question_args, questions_from_args
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import get_engine
from app.evaluation.metrics import best_gate_threshold, recall_at_k, reciprocal_rank
from app.evaluation.report import write_report
from app.rag.embeddings import get_embedder
from app.rag.reranker import get_reranker
from app.rag.retriever import Retriever

CONFIGS = [
    ("1 vector", {"mode": "vector", "rerank": False, "tables": False}),
    ("2 bm25", {"mode": "bm25", "rerank": False, "tables": False}),
    ("3 hybrid", {"mode": "hybrid", "rerank": False, "tables": False}),
    ("4 hybrid+rerank", {"mode": "hybrid", "rerank": True, "tables": False}),
    ("5 hybrid+rerank+tables", {"mode": "hybrid", "rerank": True, "tables": True}),
]
K = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_question_args(parser)
    parser.add_argument("--reranker", default=None, help="cross-encoder model (default: settings)")
    args = parser.parse_args()
    questions, placeholder = questions_from_args(args)

    settings = get_settings()
    reranker_name = args.reranker or settings.reranker_model
    retriever = Retriever(
        settings.model_copy(update={"top_k": K}),
        get_embedder(settings.embedding_model),
        get_reranker(reranker_name),
    )
    answerable = [q for q in questions if q.answerable]
    summary, details, gate_scores = [], [], {}

    with Session(get_engine()) as session:
        retriever.retrieve(session, "warm up", top_k=1)  # load models and BM25 first
        for name, opts in CONFIGS:
            r5, r10, rr, ms = [], [], [], []
            best_scores = []
            for q in questions:
                t = time.perf_counter()
                hits = retriever.retrieve(session, q.question, top_k=K, **opts).hits
                ms.append((time.perf_counter() - t) * 1000)
                papers = [h.chunk.paper_id for h in hits]
                if opts["rerank"]:
                    best_scores.append(max((h.rerank_score for h in hits), default=float("-inf")))
                row = {"config": name, "id": q.id, "type": q.type, "top_papers": " ".join(papers)}
                if q.answerable:
                    r5.append(recall_at_k(papers, q.gold_paper_ids, 5))
                    r10.append(recall_at_k(papers, q.gold_paper_ids, 10))
                    rr.append(reciprocal_rank(papers, q.gold_paper_ids))
                    row |= {"recall@5": r5[-1], "recall@10": r10[-1], "rr": rr[-1]}
                details.append(row)
            if best_scores:
                gate_scores[name] = best_scores
            summary.append(
                {
                    "config": name,
                    "questions": len(answerable),
                    "Recall@5": statistics.mean(r5),
                    "Recall@10": statistics.mean(r10),
                    "MRR": statistics.mean(rr),
                    "median ms": statistics.median(ms),
                }
            )
            print(f"{name:26} R@5={summary[-1]['Recall@5']:.3f} MRR={summary[-1]['MRR']:.3f}")

    notes = [
        f"{len(answerable)} answerable and {len(questions) - len(answerable)} unanswerable "
        f"questions; reranker: `{reranker_name}`; embedding model: `{settings.embedding_model}`.",
        "Recall@k: share of a question's gold papers among the top-k retrieved chunks. "
        "MRR: 1 / rank of the first chunk from a gold paper.",
        "Latency is per query on the machine that ran the script.",
    ]
    for name, scores in gate_scores.items():
        gate = best_gate_threshold(scores, [not q.answerable for q in questions])
        if gate:
            notes.append(
                f"Gate calibration ({name}): MIN_RERANK_SCORE = {gate['threshold']:.3f} gives "
                f"abstention precision {gate['precision']:.2f}, recall {gate['recall']:.2f}. "
                "Fitted on this question set; indicative only."
            )
    suffix = reranker_name.split("/")[-1]
    md, _ = write_report(
        f"retrieval_ablation_{suffix}",
        "Retrieval ablation",
        summary,
        details,
        notes,
        placeholder,
    )
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
