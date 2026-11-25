"""Shared test fixtures: SYNTHETIC recordings with known ground truth."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.ml.synthetic import TrialTruth, generate_trial, make_subject


def make_trial(speed: str = "normal", seed: int = 0) -> tuple[pd.DataFrame, TrialTruth]:
    """Generate one synthetic trial deterministically."""
    rng = np.random.default_rng(seed)
    subject = make_subject(1, rng)
    return generate_trial(subject, speed, f"T_{speed}", rng)


@pytest.fixture
def normal_trial() -> tuple[pd.DataFrame, TrialTruth]:
    return make_trial("normal")


@pytest.fixture(scope="session")
def db_engine():
    """Engine on a migrated test database. Skips unless TEST_DATABASE_URL is set.

    The schema is dropped and re-created, so never point this at real data.
    """
    import os

    from alembic import command
    from alembic.config import Config
    from sqlalchemy import text

    from app.db.session import get_engine

    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    engine = get_engine(url)
    with engine.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    cfg = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    cfg.attributes["database_url"] = url
    command.upgrade(cfg, "head")
    return engine


@pytest.fixture
def db_session(db_engine):
    """A session whose changes are rolled back after each test."""
    from sqlalchemy.orm import Session

    with db_engine.connect() as conn:
        trans = conn.begin()
        with Session(bind=conn, join_transaction_mode="create_savepoint") as session:
            yield session
        trans.rollback()
