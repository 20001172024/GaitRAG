"""Ingest corpus PDFs into Postgres and refresh the norms review file.

Requires the backend package (`pip install -e backend`), a migrated database
(`cd backend && alembic upgrade head`) and downloaded PDFs
(`python scripts/download_papers.py`).

    python scripts/ingest_papers.py                 # all papers in the manifest
    python scripts/ingest_papers.py --papers P001   # just one
    python scripts/ingest_papers.py --force         # re-ingest unchanged PDFs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.session import get_engine
from app.rag.embeddings import get_embedder
from app.rag.ingest import ingest_paper
from app.rag.manifest import MANIFEST_PATH, load_manifest
from app.rag.norms_review import load_approved_norms, merge_review, read_review, write_review


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest papers into the database.")
    parser.add_argument("--papers", help="comma-separated paper IDs (default: all)")
    parser.add_argument("--force", action="store_true", help="re-ingest unchanged PDFs")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    entries = load_manifest(args.manifest)
    if args.papers:
        wanted = set(args.papers.split(","))
        entries = [e for e in entries if e.paper_id in wanted]
    missing = [e.paper_id for e in entries if not e.pdf_path.exists()]
    if missing:
        raise SystemExit(f"PDFs missing for {missing}; run scripts/download_papers.py first")

    embedder = get_embedder(get_settings().embedding_model)
    candidates = []
    with Session(get_engine()) as session:
        for entry in entries:
            start = time.perf_counter()
            result = ingest_paper(session, entry, embedder, force=args.force)
            session.commit()
            candidates += result.candidates
            print(
                f"{result.status:9} {entry.paper_id}: {result.n_chunks} chunks, "
                f"{result.n_tables} tables, {len(result.candidates)} norm candidates "
                f"({time.perf_counter() - start:.1f}s)"
            )

        # Merge only the papers processed in this run; keep other papers' rows as they are.
        done = {e.paper_id for e in entries}
        existing = read_review()
        untouched = [r for r in existing if r["paper_id"] not in done]
        rows = merge_review(candidates, [r for r in existing if r["paper_id"] in done]) + untouched
        write_review(rows)
        undecided = sum(1 for r in rows if not r["decision"])
        print(f"norms_review.csv: {len(rows)} rows, {undecided} awaiting review")

        loaded, problems = load_approved_norms(session, rows)
        session.commit()
        print(f"norms table: {loaded} approved values loaded")
        for p in problems:
            print(f"  PROBLEM {p}")


if __name__ == "__main__":
    main()
