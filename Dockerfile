# Production image: API + built frontend in one container (same origin, no CORS).
# Build:  docker build -t gaitrag .
# Run:    see docker-compose.prod.yml

# --- 1. Frontend build ---------------------------------------------------------
FROM node:22-slim AS frontend
WORKDIR /src
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
# Empty base URL: the API is served from the same origin.
ENV VITE_API_BASE_URL=""
RUN npm run build

# --- 2. Python dependencies + model weights ----------------------------------
FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
RUN python -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH
# CPU-only torch first; the default wheel bundles CUDA (several GB).
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY backend/pyproject.toml /tmp/backend/pyproject.toml
COPY backend/app /tmp/backend/app
# Install runtime dependencies only, then drop the package itself: the app runs
# from its source folder (PYTHONPATH) so file paths relative to the code work.
RUN pip install /tmp/backend && pip uninstall -y gaitrag-backend
# Bake the embedding model and reranker into the image (no download at startup).
ENV HF_HOME=/opt/hf
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')"

# --- 3. Runtime ---------------------------------------------------------------
FROM python:3.11-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app/backend \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    FRONTEND_DIST=/app/frontend/dist \
    APP_ENV=production
RUN useradd --create-home --uid 10001 gaitrag
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/hf /opt/hf
WORKDIR /app
COPY backend/app backend/app
COPY backend/models backend/models
COPY backend/alembic.ini backend/alembic.ini
COPY scripts scripts
COPY data/samples data/samples
COPY data/papers/manifest.csv data/papers/norms_review.csv data/papers/
COPY --from=frontend /src/dist frontend/dist
COPY docker/entrypoint.sh /usr/local/bin/gaitrag
RUN mkdir -p data/papers/pdfs && chown -R gaitrag data/papers && chmod +x /usr/local/bin/gaitrag
USER gaitrag
WORKDIR /app/backend
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
ENTRYPOINT ["gaitrag"]
CMD ["serve"]
