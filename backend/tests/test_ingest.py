"""Tests for the ingestion pipeline and norms review (DB tests need TEST_DATABASE_URL)."""

from pathlib import Path

import pymupdf
import pytest
from sqlalchemy import func, select

from app.db.models import Chunk, Norm, Paper, PaperTable
from app.rag.embeddings import HashingEmbedder
from app.rag.ingest import ingest_paper
from app.rag.manifest import parse_row
from app.rag.norms_review import candidate_row, load_approved_norms, merge_review
from app.rag.tables import NormCandidate


def _make_pdf(path: Path) -> None:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72), "Introduction\nWalking speed affects cadence and stride time in healthy adults."
    )
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Results\nCadence was clearly higher at the fast walking speed in all groups."
        "\nReferences\n1. Smith J. Gait. 2010.",
    )
    doc.save(path)


@pytest.fixture
def entry(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("app.rag.manifest.PDF_DIR", tmp_path)
    e = parse_row(
        {
            "paper_id": "P001",
            "title": "Test paper",
            "authors": "Doe J",
            "year": "2020",
            "licence": "CC BY",
            "source_url": "https://example.org",
            "pdf_url": "https://example.org/p.pdf",
        }
    )
    _make_pdf(e.pdf_path)
    return e


def test_ingest_stores_chunks_with_pages_and_is_idempotent(db_session, entry) -> None:
    embedder = HashingEmbedder()
    first = ingest_paper(db_session, entry, embedder)
    db_session.flush()
    assert first.status == "ingested"
    chunks = db_session.scalars(select(Chunk).order_by(Chunk.chunk_index)).all()
    assert [(c.page, c.section) for c in chunks] == [(1, "Introduction"), (2, "Results")]
    assert "References" not in " ".join(c.text for c in chunks)
    assert db_session.get(Paper, "P001").pdf_sha256

    second = ingest_paper(db_session, entry, embedder)
    db_session.flush()
    assert second.status == "unchanged"
    assert db_session.scalar(select(func.count()).select_from(Chunk)) == 2

    forced = ingest_paper(db_session, entry, embedder, force=True)
    db_session.flush()
    assert forced.status == "ingested"
    assert db_session.scalar(select(func.count()).select_from(Chunk)) == 2  # replaced, not added


def _candidate(cid: str = "abc", value: str = "110.2 ± 8.1") -> NormCandidate:
    return NormCandidate(
        candidate_id=cid,
        paper_id="P001",
        page=2,
        table_no="1",
        caption="Table 1",
        row_label="Cadence (steps/min)",
        column_header="Young",
        metric="cadence",
        units="steps/min",
        value_text=value,
        kind="mean_sd",
        mean=110.2,
        sd=8.1,
        low=None,
        high=None,
        population="Young",
    )


def test_merge_keeps_reviewer_edits_and_decided_rows() -> None:
    reviewed = candidate_row(_candidate("abc")) | {"decision": "approve", "population": "Edited"}
    stale_decided = candidate_row(_candidate("old")) | {"decision": "reject"}
    stale_open = candidate_row(_candidate("gone"))
    merged = merge_review(
        [_candidate("abc"), _candidate("new")], [reviewed, stale_decided, stale_open]
    )
    by_id = {r["candidate_id"]: r for r in merged}
    assert by_id["abc"]["decision"] == "approve" and by_id["abc"]["population"] == "Edited"
    assert by_id["new"]["decision"] == ""
    assert "old" in by_id and "gone" not in by_id


def test_only_verified_approved_rows_are_loaded(db_session) -> None:
    db_session.add(Paper(paper_id="P001", title="t", licence="CC-BY", source_url="u", pdf_url="u"))
    db_session.add(
        PaperTable(
            paper_id="P001", page=2, table_no="1", caption=None, cells=[["Cadence", "110.2 ± 8.1"]]
        )
    )
    db_session.flush()
    rows = [
        candidate_row(_candidate("ok")) | {"decision": "approve"},
        candidate_row(_candidate("typo", "110.2 ± 8.7")) | {"decision": "approve"},
        candidate_row(_candidate("no")) | {"decision": "reject"},
        candidate_row(_candidate("open")),
    ]
    loaded, problems = load_approved_norms(db_session, rows)
    db_session.flush()
    assert loaded == 1
    assert len(problems) == 1 and "typo" in problems[0]
    norm = db_session.scalar(select(Norm))
    assert (norm.metric, norm.value_text, norm.mean, norm.sd) == (
        "cadence",
        "110.2 ± 8.1",
        110.2,
        8.1,
    )
