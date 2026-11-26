"""Tests for manifest validation and the PDF downloader."""

import csv
import hashlib
import importlib.util
from pathlib import Path

import httpx
import pytest

from app.rag.manifest import (
    MANIFEST_COLUMNS,
    ManifestError,
    load_manifest,
    normalize_licence,
    parse_row,
)

ROOT = Path(__file__).resolve().parents[2]


def _row(**overrides) -> dict[str, str]:
    row = {
        "paper_id": "P001",
        "pmcid": "PMC1",
        "doi": "10.1/x",
        "title": "A paper",
        "authors": "Doe J",
        "year": "2020",
        "licence": "CC BY",
        "source_url": "https://example.org/a",
        "pdf_url": "https://example.org/a.pdf",
        "pdf_md5": "",
    }
    return row | overrides


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("cc by", "CC-BY"), ("CC BY 4.0", "CC-BY"), ("CC-BY-SA-4.0", "CC-BY-SA"), ("cc0", "CC0")],
)
def test_normalize_licence(raw: str, expected: str) -> None:
    assert normalize_licence(raw) == expected


@pytest.mark.parametrize("licence", ["CC BY-NC", "CC BY-ND 4.0", "All rights reserved", ""])
def test_non_redistributable_licences_are_rejected(licence: str) -> None:
    with pytest.raises(ManifestError, match="redistribution"):
        parse_row(_row(licence=licence))


def test_missing_attribution_is_rejected() -> None:
    with pytest.raises(ManifestError, match="authors"):
        parse_row(_row(authors=" "))


def test_bad_paper_id_is_rejected() -> None:
    with pytest.raises(ManifestError, match="paper_id"):
        parse_row(_row(paper_id="paper1"))


def _write(path: Path, rows: list[dict]) -> Path:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_duplicate_doi_is_rejected(tmp_path: Path) -> None:
    path = _write(tmp_path / "m.csv", [_row(), _row(paper_id="P002")])
    with pytest.raises(ManifestError, match="Duplicate doi"):
        load_manifest(path)


def test_committed_manifest_is_valid() -> None:
    load_manifest(ROOT / "data" / "papers" / "manifest.csv")  # does not raise


def _downloader():
    spec = importlib.util.spec_from_file_location("dl", ROOT / "scripts" / "download_papers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _entry(tmp_path: Path, monkeypatch, md5: str):
    monkeypatch.setattr("app.rag.manifest.PDF_DIR", tmp_path)
    return parse_row(_row(pdf_md5=md5))


def _client(body: bytes) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, content=body))
    )


def test_download_verifies_md5_and_skips_existing(tmp_path: Path, monkeypatch) -> None:
    dl = _downloader()
    body = b"%PDF-1.7 fake"
    entry = _entry(tmp_path, monkeypatch, hashlib.md5(body).hexdigest())
    assert dl.download(entry, _client(body)) == "downloaded"
    assert entry.pdf_path.read_bytes() == body
    assert dl.download(entry, _client(body)) == "skipped"


def test_download_rejects_md5_mismatch(tmp_path: Path, monkeypatch) -> None:
    dl = _downloader()
    entry = _entry(tmp_path, monkeypatch, "0" * 32)
    with pytest.raises(ValueError, match="MD5"):
        dl.download(entry, _client(b"%PDF-1.7 fake"))
    assert not entry.pdf_path.exists()


def test_download_rejects_non_pdf(tmp_path: Path, monkeypatch) -> None:
    dl = _downloader()
    entry = _entry(tmp_path, monkeypatch, "")
    with pytest.raises(ValueError, match="unexpected content"):
        dl.download(entry, _client(b"<html>challenge</html>"))


def test_download_fetches_xml_when_listed(tmp_path: Path, monkeypatch) -> None:
    dl = _downloader()
    monkeypatch.setattr("app.rag.manifest.PDF_DIR", tmp_path)
    entry = parse_row(_row(xml_url="https://example.org/a.xml"))
    bodies = {"/a.pdf": b"%PDF-1.7 fake", "/a.xml": b'<?xml version="1.0"?><article/>'}
    client = httpx.Client(
        transport=httpx.MockTransport(lambda req: httpx.Response(200, content=bodies[req.url.path]))
    )
    assert dl.download(entry, client) == "downloaded"
    assert entry.xml_path.read_bytes().startswith(b"<?xml")
    assert dl.download(entry, client) == "skipped"
