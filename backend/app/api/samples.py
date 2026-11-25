"""GET /samples: the committed SYNTHETIC demo recordings, for one-click "try a sample"."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Repo root / data / samples. In Docker the folder is mounted at /data/samples,
# which is the same relative location (see docker-compose.yml).
SAMPLES_DIR = Path(__file__).resolve().parents[3] / "data" / "samples"

router = APIRouter(tags=["samples"])


class SampleOut(BaseModel):
    """A sample recording."""

    name: str  # e.g. "synthetic_normal"
    url: str
    synthetic: bool = True


def _available() -> dict[str, Path]:
    """Sample name -> file. Only files found here can be served (no path traversal)."""
    return {p.stem: p for p in sorted(SAMPLES_DIR.glob("*.csv"))}


@router.get("/samples", response_model=list[SampleOut])
def list_samples() -> list[SampleOut]:
    """List the demo recordings (all SYNTHETIC)."""
    return [SampleOut(name=name, url=f"/samples/{name}.csv") for name in _available()]


@router.get("/samples/{name}.csv", response_class=FileResponse)
def get_sample(name: str) -> FileResponse:
    """Download one demo recording as CSV."""
    path = _available().get(name)
    if path is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown sample")
    return FileResponse(path, media_type="text/csv", filename=path.name)
