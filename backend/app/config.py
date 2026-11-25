"""Application settings loaded from environment variables (and `.env`)."""

from functools import lru_cache
from pathlib import Path

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
    reranker_model: str = "BAAI/bge-reranker-base"

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
