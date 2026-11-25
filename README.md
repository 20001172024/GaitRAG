# GaitRAG

Evidence-grounded gait analysis assistant. Upload a walking recording (insole IMU + pressure CSV), get a walking-speed classification, interpretable gait features, and an explanation compared against published norms, with every claim cited to a paper, page and table.

> **Disclaimer:** GaitRAG is a research and education tool. It is not a medical device and does not provide diagnosis or clinical advice. Results are based on a model and published literature and may be incomplete or wrong.

**Status:** M0 (project setup). See [`CLAUDE.md`](CLAUDE.md) for the full brief and milestones.

## Quick start (Docker)

```bash
cp .env.example .env
docker compose up --build
```

- Backend: http://localhost:8000/health (API docs at `/docs`)
- Frontend: http://localhost:5173
- Postgres + pgvector: `localhost:5432`

## Local development

Backend (Python 3.11):

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
uvicorn app.main:app --reload
pytest
```

Frontend (Node 22):

```bash
cd frontend
npm install
npm run dev
npm test
```

Pre-commit hooks (ruff, black, prettier):

```bash
pip install pre-commit
pre-commit install
```

## Data policy

- No real participant data is committed. The public demo uses synthetic recordings from `scripts/make_synthetic_gait.py`.
- Only open-access, redistributable papers are listed in `data/papers/manifest.csv`. PDFs are downloaded locally and git-ignored.
