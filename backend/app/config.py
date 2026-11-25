"""Application settings loaded from environment variables (and `.env`)."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration for the backend.

    Values come from environment variables; a local `.env` file is read if present.
    Field names map to upper-case env vars (e.g. `database_url` -> `DATABASE_URL`).
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    # Comma-separated list of allowed frontend origins.
    cors_origins: str = "http://localhost:5173"

    database_url: str = "postgresql://gaitrag:change-me@localhost:5432/gaitrag"

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    # Default is the small MiniLM cross-encoder: on a 4-core CPU it reranked 30 pairs in
    # ~0.7 s vs ~5.3 s for BAAI/bge-reranker-base. Both are compared in the M6 ablation.
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    # Retrieval (M3). Each stage can be switched off for the ablation study.
    retrieval_mode: Literal["vector", "bm25", "hybrid"] = "hybrid"
    rerank: bool = True
    vector_candidates: int = 50  # top-k from pgvector
    bm25_candidates: int = 50  # top-k from BM25
    rrf_k: int = 60  # Reciprocal Rank Fusion constant (Cormack et al., 2009)
    rerank_candidates: int = 30  # fused hits passed to the cross-encoder
    top_k: int = 8  # hits returned
    hnsw_ef_search: int = 100  # must be >= vector_candidates, see app.rag.retriever

    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"

    # Speed classifier. None = the committed model at backend/models/speed_rf.joblib.
    speed_model_path: Path | None = None
    max_upload_mb: float = 10.0

    @property
    def cors_origin_list(self) -> list[str]:
        """Return `cors_origins` split into a clean list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (use as a FastAPI dependency)."""
    return Settings()
