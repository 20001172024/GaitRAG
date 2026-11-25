"""Download corpus PDFs (and JATS XML) listed in data/papers/manifest.csv.

PDFs come from the PMC Open Access dataset (pmc-oa-opendata on AWS S3), NCBI's
official distribution of open-access articles. The XML gives clean, structured
tables. Each file is checked against the MD5 from the manifest. Existing valid
files are skipped. The pdfs/ folder is git-ignored: we redistribute the paper
list, not the files.

    python scripts/download_papers.py
"""

from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import httpx

from app.rag.manifest import MANIFEST_PATH, ManifestEntry, load_manifest

USER_AGENT = "gaitrag-corpus-downloader/0.1 (research; open-access papers only)"
PDF_MAGIC = (b"%PDF-",)
XML_MAGIC = (b"<?xml", b"<!DOCTYPE", b"<article")  # PMC JATS files start with a DOCTYPE
DELAY_S = 1.0  # be polite to the server between downloads


def md5_of(path: Path) -> str:
    """MD5 hex digest of a file."""
    h = hashlib.md5()  # noqa: S324 - integrity check against PMC's published MD5, not security
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, path: Path, md5: str, magic: tuple[bytes, ...], client: httpx.Client) -> str:
    """Download `url` to `path` unless a valid copy exists.

    Returns "skipped" or "downloaded". Raises ValueError if the file does not start
    with `magic` (e.g. an HTML challenge page) or fails the MD5 check.
    """
    if path.exists() and (not md5 or md5_of(path) == md5):
        return "skipped"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with client.stream("GET", url) as r:
        r.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    if not tmp.read_bytes()[:64].lstrip().startswith(magic):
        tmp.unlink()
        raise ValueError(f"{path.name}: unexpected content (expected {magic[0]!r}...)")
    if md5 and md5_of(tmp) != md5:
        tmp.unlink()
        raise ValueError(f"{path.name}: MD5 mismatch")
    tmp.replace(path)
    return "downloaded"


def download(entry: ManifestEntry, client: httpx.Client) -> str:
    """Download a paper's PDF and, if listed, its JATS XML.

    Returns "downloaded" if anything was fetched, else "skipped".
    """
    statuses = [fetch(entry.pdf_url, entry.pdf_path, entry.pdf_md5, PDF_MAGIC, client)]
    if entry.xml_url:
        statuses.append(fetch(entry.xml_url, entry.xml_path, entry.xml_md5, XML_MAGIC, client))
    return "downloaded" if "downloaded" in statuses else "skipped"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download corpus PDFs from the manifest.")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()

    entries = load_manifest(args.manifest)
    failures = 0
    with httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=60, follow_redirects=True) as c:
        for entry in entries:
            try:
                status = download(entry, c)
            except (httpx.HTTPError, ValueError) as exc:
                failures += 1
                print(f"FAIL {entry.paper_id}: {exc}")
                continue
            print(f"{status:10} {entry.paper_id} {entry.title[:70]}")
            if status == "downloaded":
                time.sleep(DELAY_S)
    print(f"{len(entries) - failures}/{len(entries)} PDFs available")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
