"""Health check router."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from app import __version__
from app.db.session import get_engine, ping

router = APIRouter(tags=["health"])


class HealthResponse(BaseModel):
    """Liveness plus database reachability.

    `status` stays "ok" when the database is down: the API process is alive and
    /analyze still works without it. `db` reports the database separately.
    """

    status: Literal["ok"]
    version: str
    db: Literal["ok", "down"]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Report that the API is up and whether the database answers."""
    return HealthResponse(
        status="ok", version=__version__, db="ok" if ping(get_engine()) else "down"
    )
