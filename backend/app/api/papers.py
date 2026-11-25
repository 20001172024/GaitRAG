"""GET /papers: the corpus with attribution and licence for every paper."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.db.models import Chunk, Paper
from app.db.session import get_session

router = APIRouter(tags=["papers"])


class PaperOut(BaseModel):
    """A corpus paper. Title, authors, licence and source URL are the CC BY attribution."""

    paper_id: str
    title: str
    authors: str | None
    year: int | None
    doi: str | None
    licence: str
    source_url: str
    n_pages: int | None
    n_chunks: int


@router.get("/papers", response_model=list[PaperOut])
def list_papers(session: Annotated[Session, Depends(get_session)]) -> list[PaperOut]:
    """List all ingested papers, ordered by ID."""
    n_chunks = select(Chunk.paper_id, func.count().label("n")).group_by(Chunk.paper_id).subquery()
    query = (
        select(Paper, func.coalesce(n_chunks.c.n, 0))
        .outerjoin(n_chunks, n_chunks.c.paper_id == Paper.paper_id)
        .order_by(Paper.paper_id)
    )
    try:
        rows = session.execute(query).all()
    except OperationalError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable") from exc
    return [
        PaperOut(
            paper_id=p.paper_id,
            title=p.title,
            authors=p.authors,
            year=p.year,
            doi=p.doi,
            licence=p.licence,
            source_url=p.source_url,
            n_pages=p.n_pages,
            n_chunks=n,
        )
        for p, n in rows
    ]
