"""Ingestion pipeline: PDF -> chunks + embeddings + raw tables in Postgres.

Re-runnable: a paper whose PDF hash matches the last ingest is skipped; otherwise
its chunks and tables are replaced inside one transaction, so a failure leaves
the previous version intact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Paper, PaperTable
from app.rag.chunking import chunk_segments
from app.rag.embeddings import Embedder, passage_text
from app.rag.jats import jats_tables_with_pages, merge_tables
from app.rag.manifest import ManifestEntry
from app.rag.parsing import RawTable, parse_pdf
from app.rag.table_chunks import table_to_chunks
from app.rag.tables import NormCandidate, extract_candidates

PIPELINE_VERSION = "3"
"""Bump when parsing, chunking or table handling changes, to force re-ingestion.
History: 1 = text chunks; 2 = + table chunks; 3 = wide table rows split."""


@dataclass(frozen=True)
class IngestResult:
    """Outcome for one paper."""

    paper_id: str
    status: str  # "ingested" | "unchanged"
    n_chunks: int
    n_tables: int
    candidates: list[NormCandidate]


def sha256_of(path: Path) -> str:
    """SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _upsert_paper(session: Session, entry: ManifestEntry) -> Paper:
    paper = session.get(Paper, entry.paper_id) or Paper(paper_id=entry.paper_id)
    paper.doi = entry.doi or None
    paper.title = entry.title
    paper.authors = entry.authors
    paper.year = entry.year
    paper.licence = entry.licence
    paper.source_url = entry.source_url
    paper.pdf_url = entry.pdf_url
    session.add(paper)
    return paper


def _stored_tables(session: Session, paper_id: str) -> list[RawTable]:
    rows = session.scalars(select(PaperTable).where(PaperTable.paper_id == paper_id))
    return [
        RawTable(t.page, t.table_no, t.caption, t.cells, t.source, t.n_header_rows) for t in rows
    ]


def ingest_paper(
    session: Session, entry: ManifestEntry, embedder: Embedder, force: bool = False
) -> IngestResult:
    """Parse, chunk, embed and store one paper (caller commits the session).

    Args:
        session: database session.
        entry: the paper's manifest row; its PDF must already be downloaded.
        embedder: embedding model (also provides the token counter for chunking).
        force: re-ingest even if the PDF is unchanged.
    """
    # Fingerprint (stored in papers.pdf_sha256): PDF hash + XML hash + pipeline version,
    # so a changed source file *or* a change to how papers are processed re-ingests.
    parts = [sha256_of(entry.pdf_path), PIPELINE_VERSION]
    if entry.xml_path.exists():
        parts.append(sha256_of(entry.xml_path))
    pdf_hash = hashlib.sha256("|".join(parts).encode()).hexdigest()
    paper = _upsert_paper(session, entry)
    n_existing = session.scalar(
        select(func.count()).select_from(Chunk).where(Chunk.paper_id == entry.paper_id)
    )
    if not force and paper.pdf_sha256 == pdf_hash and n_existing:
        tables = _stored_tables(session, entry.paper_id)
        candidates = [c for t in tables for c in extract_candidates(entry.paper_id, t)]
        return IngestResult(entry.paper_id, "unchanged", n_existing, len(tables), candidates)

    parsed = parse_pdf(entry.pdf_path)
    xml_tables = (
        jats_tables_with_pages(entry.xml_path, entry.pdf_path) if entry.xml_path.exists() else []
    )
    tables = merge_tables(xml_tables, parsed.tables)
    drafts = chunk_segments(parsed.segments, embedder.count_tokens)
    table_drafts = [c for t in tables for c in table_to_chunks(t, embedder.count_tokens)]
    vectors = embedder.embed_passages(
        [passage_text(entry.title, d.section, d.text) for d in drafts]
        + [passage_text(entry.title, None, t.text) for t in table_drafts]
    )

    session.execute(delete(Chunk).where(Chunk.paper_id == entry.paper_id))
    session.execute(delete(PaperTable).where(PaperTable.paper_id == entry.paper_id))
    session.add_all(
        Chunk(
            paper_id=entry.paper_id,
            chunk_index=i,
            page=d.page,
            section=d.section,
            text=d.text,
            token_count=d.token_count,
            content_hash=hashlib.sha256(d.text.encode()).hexdigest(),
            embedding=vec.tolist(),
        )
        for i, (d, vec) in enumerate(zip(drafts, vectors[: len(drafts)], strict=True))
    )
    session.add_all(
        Chunk(
            paper_id=entry.paper_id,
            chunk_index=len(drafts) + i,
            page=t.page,
            section=None,
            kind="table",
            table_no=t.table_no,
            text=t.text,
            token_count=t.token_count,
            content_hash=hashlib.sha256(t.text.encode()).hexdigest(),
            embedding=vec.tolist(),
        )
        for i, (t, vec) in enumerate(zip(table_drafts, vectors[len(drafts) :], strict=True))
    )
    session.add_all(
        PaperTable(
            paper_id=entry.paper_id,
            page=t.page,
            table_no=t.table_no,
            caption=t.caption,
            cells=t.cells,
            source=t.source,
            n_header_rows=t.n_header_rows,
        )
        for t in tables
    )
    paper.pdf_sha256 = pdf_hash
    paper.n_pages = parsed.n_pages
    paper.ingested_at = datetime.now(UTC)

    candidates = [c for t in tables for c in extract_candidates(entry.paper_id, t)]
    n_chunks = len(drafts) + len(table_drafts)
    return IngestResult(entry.paper_id, "ingested", n_chunks, len(tables), candidates)
