# GaitRAG

Evidence-grounded gait analysis assistant. Upload a walking recording (insole IMU + pressure CSV), get a walking-speed classification, interpretable gait features, and an explanation compared against published norms, with every claim cited to a paper, page and table.

> **Disclaimer:** GaitRAG is a research and education tool. It is not a medical device and does not provide diagnosis or clinical advice. Results are based on a model and published literature and may be incomplete or wrong.

**Status:** M1 (synthetic data + speed classifier). See [`CLAUDE.md`](CLAUDE.md) for the full brief and milestones.

## Quick start (Docker)

```bash
cp .env.example .env
docker compose up --build
```

- Backend: http://localhost:8000/health (API docs at `/docs`)
- Frontend: http://localhost:5173
- Postgres + pgvector: `localhost:5432`

Try the analyzer with a sample recording:

```bash
curl -F file=@data/samples/synthetic_normal.csv http://localhost:8000/analyze
```

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

## Speed classifier (M1)

Pipeline for one recording (40 Hz CSV):

1. **Validate** columns, types, single trial, at least 10 s, and the sampling rate.
2. **Detect gait events** from insole pressure: total foot load against a relative threshold, with threshold crossings interpolated between samples for timing finer than 25 ms.
3. **Interpretable features:** cadence, stride time and its CV, stance/swing %, and left-right symmetry index of stance time and peak pressure.
4. **Model features:** the gait features, plus per-channel statistics of z-scored signals, plus Welch band powers of each IMU's acceleration magnitude.
5. **Random Forest** (300 trees) predicts slow, normal or fast.

Rebuild the model:

```bash
python scripts/make_synthetic_gait.py --out data/synthetic
python scripts/train_model.py --data data/synthetic
```

Evaluation uses `GroupKFold` by subject, so no person appears in both training and test folds. Results are in [`backend/models/training_report.md`](backend/models/training_report.md).

> **These results are on SYNTHETIC data** and only show that the pipeline runs. They say nothing about real-world accuracy.

### Known limitations

- **Normalization at inference.** In the research setting, signals are z-scored per subject using all of that subject's trials. A demo upload is the only data we have from that person, so it is normalized against itself, and the deployed model is trained the same way to match. On synthetic data this costs accuracy compared with subject-wise normalization (see the report).
- **Synthetic data.** The generator's parameters are invented, not published norms. The CSV schema and IMU placement still need to be confirmed against the real dataset.
- **Thresholds.** Gait-event thresholds were tuned on synthetic signals and need checking on real recordings.

## Data policy

- No real participant data is committed. The public demo uses synthetic recordings from `scripts/make_synthetic_gait.py`.
- Only open-access, redistributable papers are listed in `data/papers/manifest.csv`. PDFs are downloaded locally and git-ignored.
