"""Deploy GaitRAG to a Hugging Face Space (Docker SDK).

Used by .github/workflows/deploy.yml; can also be run by hand:

    HF_TOKEN=hf_... DATABASE_URL=postgresql://... OPENAI_API_KEY=sk-... \\
        python scripts/deploy_space.py --space <hf-user>/gaitrag
    python scripts/deploy_space.py --space <hf-user>/gaitrag --dry-run   # list files only

What it does:
1. Stages exactly the files git tracks (so .env, PDFs and results never leak),
   replacing README.md with a Space README (HF reads its YAML header for config).
2. Creates the Space if needed (public, Docker SDK) and uploads the files; the
   Space then builds the root Dockerfile.
3. Copies secrets from environment variables into the Space settings:
   DATABASE_URL (required), OPENAI_API_KEY / HF_API_TOKEN (optional).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE_PREFIXES = (".github/", "eval/results/")
SECRET_KEYS = ("DATABASE_URL", "OPENAI_API_KEY", "HF_API_TOKEN")
VARIABLE_KEYS = ("LLM_PROVIDER", "LLM_MODEL", "ASK_LIMIT_PER_CLIENT_PER_HOUR", "ASK_LIMIT_PER_DAY")

SPACE_README = """---
title: GaitRAG
emoji: 🚶
colorFrom: green
colorTo: gray
sdk: docker
app_port: 8000
pinned: false
short_description: Evidence-grounded gait analysis with cited answers
---

# GaitRAG

Upload an insole walking recording (or click a synthetic sample) to get a walking-speed
classification, gait features compared with published norms, and an explanation in which
every claim is cited to an open-access paper, page and table.

> **Disclaimer:** GaitRAG is a research and education tool. It is not a medical device and
> does not provide diagnosis or clinical advice. Results are based on a model and published
> literature and may be incomplete or wrong.

The speed model is trained on **synthetic** data. Source code and documentation:
{source_url}
"""


def tracked_files() -> list[str]:
    """Files under version control, minus paths that are not needed in the Space."""
    out = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    return [f for f in out if not f.startswith(EXCLUDE_PREFIXES)]


def stage(files: list[str], dest: Path, source_url: str) -> None:
    """Copy files into `dest` and write the Space README."""
    for f in files:
        target = dest / f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, target)
    (dest / "README.md").write_text(SPACE_README.format(source_url=source_url), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy to a Hugging Face Space.")
    parser.add_argument("--space", required=True, help="e.g. <hf-user>/gaitrag")
    parser.add_argument("--source-url", default="https://github.com/20001172024/GaitRAG")
    parser.add_argument("--dry-run", action="store_true", help="list files, upload nothing")
    args = parser.parse_args()

    files = tracked_files()
    if args.dry_run:
        print("\n".join(files))
        print(f"\n{len(files)} files would be uploaded to spaces/{args.space}")
        missing = [k for k in ("HF_TOKEN", "DATABASE_URL") if not os.environ.get(k)]
        if missing:
            print(f"Missing for a real deploy: {', '.join(missing)}")
        return

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    if not os.environ.get("DATABASE_URL"):
        raise SystemExit("DATABASE_URL is not set (the Space needs a Postgres with pgvector)")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage(files, Path(tmp), args.source_url)
        api.upload_folder(
            repo_id=args.space,
            repo_type="space",
            folder_path=tmp,
            commit_message="Deploy from GitHub",
            delete_patterns=["*"],  # mirror: remove files deleted from the repo
        )
    for key in SECRET_KEYS:
        if os.environ.get(key):
            api.add_space_secret(args.space, key, os.environ[key])
            print(f"secret set: {key}")
    for key in VARIABLE_KEYS:
        if os.environ.get(key):
            api.add_space_variable(args.space, key, os.environ[key])
            print(f"variable set: {key}")
    print(f"Deployed: https://huggingface.co/spaces/{args.space}")


if __name__ == "__main__":
    main()
