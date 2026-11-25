"""ORM models: papers, text chunks with embeddings, raw tables and approved norms."""

from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Computed,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

EMBEDDING_DIM = 384
"""Output size of BAAI/bge-small-en-v1.5. Changing the model needs a migration."""


class Base(DeclarativeBase):
    """Declarative base for all models."""


class Paper(Base):
    """One corpus paper (mirrors a row of data/papers/manifest.csv)."""

    __tablename__ = "papers"

    paper_id: Mapped[str] = mapped_column(String(16), primary_key=True)  # e.g. "P001"
    doi: Mapped[str | None] = mapped_column(String(255), unique=True)
    title: Mapped[str] = mapped_column(Text)
    authors: Mapped[str | None] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(Integer)
    licence: Mapped[str] = mapped_column(String(32))
    source_url: Mapped[str] = mapped_column(Text)
    pdf_url: Mapped[str] = mapped_column(Text)
    pdf_sha256: Mapped[str | None] = mapped_column(String(64))
    n_pages: Mapped[int | None] = mapped_column(Integer)
    ingested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    # passive_deletes: let Postgres' ON DELETE CASCADE remove children.
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="paper", cascade="all, delete-orphan", passive_deletes=True
    )
    tables: Mapped[list[PaperTable]] = relationship(
        cascade="all, delete-orphan", passive_deletes=True
    )
    norms: Mapped[list[Norm]] = relationship(cascade="all, delete-orphan", passive_deletes=True)


class Chunk(Base):
    """A retrievable passage. Never spans a page break, so its citation page is exact."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("paper_id", "chunk_index", name="uq_chunks_paper_index"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id", ondelete="CASCADE"))
    chunk_index: Mapped[int] = mapped_column(Integer)
    page: Mapped[int] = mapped_column(Integer)  # 1-based, as printed in citations
    section: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64))
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))
    # Kept in sync by Postgres; used for keyword search (M3).
    tsv: Mapped[str] = mapped_column(
        TSVECTOR, Computed("to_tsvector('english', text)", persisted=True)
    )

    paper: Mapped[Paper] = relationship(back_populates="chunks")


class PaperTable(Base):
    """A table extracted from a PDF page, stored as raw cells."""

    __tablename__ = "paper_tables"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id", ondelete="CASCADE"))
    page: Mapped[int] = mapped_column(Integer)
    table_no: Mapped[str | None] = mapped_column(String(16))  # from caption, e.g. "2"
    caption: Mapped[str | None] = mapped_column(Text)
    cells: Mapped[list[list[str | None]]] = mapped_column(JSONB)


class Norm(Base):
    """A published normative value, approved by a human reviewer."""

    __tablename__ = "norms"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.paper_id", ondelete="CASCADE"))
    page: Mapped[int] = mapped_column(Integer)
    table_no: Mapped[str | None] = mapped_column(String(16))
    metric: Mapped[str] = mapped_column(String(64))  # canonical, e.g. "cadence"
    population: Mapped[str | None] = mapped_column(Text)
    value_text: Mapped[str] = mapped_column(Text)  # verbatim cell, e.g. "112.3 ± 8.1"
    mean: Mapped[float | None]
    sd: Mapped[float | None]
    low: Mapped[float | None]
    high: Mapped[float | None]
    units: Mapped[str | None] = mapped_column(String(32))
    approved_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
