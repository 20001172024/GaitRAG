#!/bin/sh
# gaitrag serve   - apply migrations, start the API (serves the frontend too)
# gaitrag ingest  - apply migrations, download papers, build the index
# anything else   - run that command
set -e
case "$1" in
  serve)
    alembic upgrade head
    # One worker: each worker loads its own copy of the models (~1 GB RAM).
    exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
    ;;
  ingest)
    alembic upgrade head
    python /app/scripts/download_papers.py
    exec python /app/scripts/ingest_papers.py
    ;;
  *)
    exec "$@"
    ;;
esac
