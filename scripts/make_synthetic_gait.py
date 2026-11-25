"""Generate SYNTHETIC gait recordings in the project CSV schema.

Requires the backend package: `pip install -e backend`.

Examples:
    # Training set (git-ignored): 23 subjects x 12 trials, like the real dataset's shape
    python scripts/make_synthetic_gait.py --out data/synthetic

    # One committed sample recording per speed class for the demo
    python scripts/make_synthetic_gait.py --samples --out data/samples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from app.ml.synthetic import SPEEDS, generate_trial, make_subject


def make_dataset(out: Path, n_subjects: int, trials_per_subject: int, seed: int) -> None:
    """Write one CSV per trial plus `labels.csv` (trial_id, subject_id, speed)."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    labels = []
    for s in range(1, n_subjects + 1):
        subject = make_subject(s, rng)
        for k in range(trials_per_subject):
            speed = SPEEDS[k % len(SPEEDS)]  # balanced classes
            trial_id = f"{subject.subject_id}_T{k + 1:02d}"
            df, _ = generate_trial(subject, speed, trial_id, rng)
            df.to_csv(out / f"{trial_id}.csv", index=False)
            labels.append({"trial_id": trial_id, "subject_id": subject.subject_id, "speed": speed})
    pd.DataFrame(labels).to_csv(out / "labels.csv", index=False)
    print(f"Wrote {len(labels)} SYNTHETIC trials to {out}")


def make_samples(out: Path, seed: int) -> None:
    """Write one sample recording per speed class from a held-out synthetic subject."""
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    subject = make_subject(99, rng)
    for speed in SPEEDS:
        trial_id = f"{subject.subject_id}_{speed}"
        df, _ = generate_trial(subject, speed, trial_id, rng)
        path = out / f"synthetic_{speed}.csv"
        df.to_csv(path, index=False)
        print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--subjects", type=int, default=23)
    parser.add_argument("--trials", type=int, default=12, help="trials per subject")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", action="store_true", help="write demo samples only")
    args = parser.parse_args()

    if args.samples:
        make_samples(args.out, args.seed)
    else:
        make_dataset(args.out, args.subjects, args.trials, args.seed)


if __name__ == "__main__":
    main()
