# GaitRAG

Evidence-grounded gait analysis assistant. Upload a walking recording (insole IMU + pressure CSV), get a walking-speed classification, interpretable gait features, and an explanation compared against published norms, with every claim cited to a paper, page and table.

> **Disclaimer:** GaitRAG is a research and education tool. It is not a medical device and does not provide diagnosis or clinical advice. Results are based on a model and published literature and may be incomplete or wrong.

**Status:** M3 (retrieval). See [`CLAUDE.md`](CLAUDE.md) for the full brief and milestones.

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

## Paper corpus (M2)

```bash
python scripts/download_papers.py   # PDFs + JATS XML -> data/papers/pdfs/ (git-ignored), MD5-checked
python scripts/ingest_papers.py     # parse, chunk, embed -> Postgres; refresh norms_review.csv
```

- **Sources.** Open-access papers listed in [`data/papers/manifest.csv`](data/papers/manifest.csv) with DOI, authors, licence and source URL. Only CC-BY, CC-BY-SA and CC0 are accepted. PDFs come from the [PMC Open Access dataset](https://pmc.ncbi.nlm.nih.gov/tools/pmcaws/) and are checked against its published MD5.
- **Parsing.** PyMuPDF page by page, so every passage keeps its page number. The parser removes running headers and footers, normalizes ligatures, re-joins hyphenated words and labels standard sections. References, back matter and text inside tables are dropped.
- **Chunking.** About 300 tokens per chunk (bge tokenizer), split at sentence boundaries, with up to 50 tokens of whole-sentence overlap. Chunks never cross a page or section break, so a citation's page is exact.
- **Embeddings.** `BAAI/bge-small-en-v1.5`, 384 dimensions, normalized. Each chunk is embedded with a "title > section" header. Queries use bge's instruction prefix. They are stored in pgvector with an HNSW cosine index, plus a generated `tsvector` column for keyword search (M3).
- **Tables.** Tables are taken from PMC's JATS XML, which gives exact cells and merged-cell spans and was more accurate than reading PDF layout. Each XML table is placed on the PDF page that contains its caption *and* its own numbers, so table citations keep exact pages. PDF table detection (PyMuPDF) is the fallback for tables the XML lacks.
- **Tables to norms.** Rules propose candidate values (cadence, stride time and its CV, stance/swing %, symmetry, gait speed) into [`data/papers/norms_review.csv`](data/papers/norms_review.csv). **A person approves each value.** Only approved rows whose text still appears verbatim in the source table are loaded into the `norms` table.

Set `decision` to `approve` or `reject` in the review CSV (you may correct metric, population, units or the parsed numbers), then re-run `ingest_papers.py`. Unchanged PDFs are skipped. Your edits are kept for every row that has a decision or a note; untouched rows are refreshed from extraction.

The `flags` column gives automatic hints: no or unusual units, a value outside rough plausibility bounds, SD larger than the mean, a missing population, or a reliability, regression or validation table. **The bounds are sanity limits chosen for this project, not published norms.** A flag is not a decision; it only shows where to look first.

### Known limitations (M2)

- Papers without JATS XML rely on PDF table detection, which handles rotated tables and dot-leader layouts poorly.
- Section detection relies on standard section names. Unusual layouts fall back to the previous section label.

## Retrieval (M3)

```
query ─► vector (pgvector, cosine, top 50) ─┐
     └─► BM25 (rank_bm25, top 50)        ───┴─► RRF (k = 60) ─► cross-encoder rerank (top 30) ─► top 8
```

- **Hybrid search.** Embeddings capture paraphrases, while BM25 catches exact terms such as "CoV" or "PODOSmart". The two rankings are merged with Reciprocal Rank Fusion, which uses ranks only, so there are no score scales to normalize.
- **Vector search detail.** pgvector's HNSW search examines only `hnsw.ef_search` candidates (default 40), so `LIMIT 50` silently returned 40 rows. The retriever raises it per query, and a test covers this.
- **Feature queries.** One query per gait feature, including the predicted speed class and leaving out the user's numbers.
- **Ablations.** Set `RETRIEVAL_MODE=vector|bm25|hybrid` and `RERANK=true|false`, or override per request:

```bash
curl -X POST localhost:8000/search -H 'Content-Type: application/json' \
  -d '{"query": "stride time variability at slow speed", "mode": "hybrid", "rerank": true}'
```

Latency, measured on the 4-core development sandbox (CPU only, median per query, 921 chunks): vector 22 ms, BM25 2 ms, hybrid 23 ms, hybrid + `bge-reranker-base` reranking 30 candidates about 6.9 s. These are development measurements; the end-to-end p50/p95 numbers come from `eval/run_latency_eval.py` in M6.

## Data policy

- No real participant data is committed. The public demo uses synthetic recordings from `scripts/make_synthetic_gait.py`.
- Only open-access, redistributable papers are listed in `data/papers/manifest.csv`. PDFs are downloaded locally and git-ignored.
