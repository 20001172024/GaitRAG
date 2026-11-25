"""Paper manifest (data/papers/manifest.csv): reading and licence validation.

Only licences that allow redistribution are accepted, because the public repo
and demo serve text from these papers. CC BY requires attribution, which is
why title, authors, licence and source URL are mandatory.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

MANIFEST_PATH = Path(__file__).resolve().parents[3] / "data" / "papers" / "manifest.csv"
PDF_DIR = MANIFEST_PATH.parent / "pdfs"

MANIFEST_COLUMNS = [
    "paper_id",
    "pmcid",
    "doi",
    "title",
    "authors",
    "year",
    "licence",
    "source_url",
    "pdf_url",
    "pdf_md5",
    "xml_url",
    "xml_md5",
]

ALLOWED_LICENCES = frozenset({"CC-BY", "CC-BY-SA", "CC0"})
"""Licences that permit redistribution. NC/ND variants are deliberately excluded."""

_PAPER_ID = re.compile(r"^P\d{3}$")


class ManifestError(ValueError):
    """Raised for an invalid manifest row."""


@dataclass(frozen=True)
class ManifestEntry:
    """One paper in the corpus."""

    paper_id: str
    pmcid: str
    doi: str
    title: str
    authors: str
    year: int
    licence: str
    source_url: str
    pdf_url: str
    pdf_md5: str
    xml_url: str = ""  # optional JATS XML (structured tables), same source as the PDF
    xml_md5: str = ""

    @property
    def pdf_path(self) -> Path:
        """Local path of the downloaded PDF."""
        return PDF_DIR / f"{self.paper_id}.pdf"

    @property
    def xml_path(self) -> Path:
        """Local path of the downloaded JATS XML (next to the PDF, also git-ignored)."""
        return PDF_DIR / f"{self.paper_id}.xml"


def normalize_licence(raw: str) -> str:
    """Map spellings like "cc by", "CC BY 4.0" or "CC-BY-SA" to a canonical form."""
    s = re.sub(r"[\s_]+", "-", raw.strip().upper())
    s = re.sub(r"-\d(\.\d)?$", "", s)  # drop version suffix
    return s


def parse_row(row: dict[str, str]) -> ManifestEntry:
    """Validate one CSV row.

    Raises:
        ManifestError: on a bad ID, a missing attribution field, or a licence
            that does not allow redistribution.
    """
    pid = row.get("paper_id", "").strip()
    if not _PAPER_ID.match(pid):
        raise ManifestError(f"Bad paper_id {pid!r} (expected e.g. P001)")
    for field in ("title", "authors", "source_url", "pdf_url"):
        if not row.get(field, "").strip():
            raise ManifestError(f"{pid}: missing {field} (needed for attribution/download)")
    licence = normalize_licence(row.get("licence", ""))
    if licence not in ALLOWED_LICENCES:
        raise ManifestError(f"{pid}: licence {row.get('licence')!r} does not allow redistribution")
    try:
        year = int(row.get("year", ""))
    except ValueError as exc:
        raise ManifestError(f"{pid}: bad year {row.get('year')!r}") from exc
    return ManifestEntry(
        paper_id=pid,
        pmcid=row.get("pmcid", "").strip(),
        doi=row.get("doi", "").strip(),
        title=row["title"].strip(),
        authors=row["authors"].strip(),
        year=year,
        licence=licence,
        source_url=row["source_url"].strip(),
        pdf_url=row["pdf_url"].strip(),
        pdf_md5=row.get("pdf_md5", "").strip().lower(),
        xml_url=(row.get("xml_url") or "").strip(),
        xml_md5=(row.get("xml_md5") or "").strip().lower(),
    )


def load_manifest(path: Path = MANIFEST_PATH) -> list[ManifestEntry]:
    """Read and validate every row; duplicate IDs or DOIs are an error."""
    with path.open(newline="", encoding="utf-8") as f:
        entries = [parse_row(row) for row in csv.DictReader(f)]
    for key in ("paper_id", "doi"):
        values = [getattr(e, key) for e in entries if getattr(e, key)]
        dupes = {v for v in values if values.count(v) > 1}
        if dupes:
            raise ManifestError(f"Duplicate {key}: {', '.join(sorted(dupes))}")
    return entries
