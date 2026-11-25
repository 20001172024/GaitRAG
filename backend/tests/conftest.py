"""Shared test fixtures: SYNTHETIC recordings with known ground truth."""

from __future__ import annotations

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
