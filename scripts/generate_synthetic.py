"""
Generate a synthetic tabular dataset with known ground-truth issues for agent eval.

Includes: injected nulls, outliers, mixed dtypes, a decoy leaky column, and
moderate class imbalance.

Usage:
    uv run python scripts/generate_synthetic.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FILENAME = "synthetic_ground_truth.csv"
DEFAULT_SEED = 42


def generate_synthetic_dataset(
    output_dir: Path,
    *,
    root: Path | None = None,
    n_rows: int = 2000,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Write synthetic CSV and return manifest-style metadata."""
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / DEFAULT_FILENAME

    age = rng.integers(18, 80, size=n_rows)
    income = rng.normal(55_000, 18_000, size=n_rows).clip(15_000, 200_000)
    tenure_months = rng.integers(0, 120, size=n_rows)
    usage_gb = rng.gamma(shape=2.5, scale=12.0, size=n_rows)

    segment = rng.choice(["basic", "standard", "premium"], size=n_rows, p=[0.45, 0.35, 0.20])
    region = rng.choice(["north", "south", "east", "west"], size=n_rows)

    # Mixed-type column: mostly numeric strings, some invalid tokens
    score_raw = rng.normal(650, 80, size=n_rows).clip(300, 850)
    score_str = score_raw.round(0).astype(int).astype(str)
    bad_idx = rng.choice(n_rows, size=40, replace=False)
    score_str[bad_idx] = rng.choice(["N/A", "unknown", "—"], size=len(bad_idx))

    # Datetime-like strings
    base = pd.Timestamp("2022-01-01")
    signup_offsets = rng.integers(0, 900, size=n_rows)
    signup_date = (base + pd.to_timedelta(signup_offsets, unit="D")).strftime("%Y-%m-%d")

    # True latent signal for target
    logit = (
        -2.2
        + 0.018 * (income / 1000)
        + 0.012 * tenure_months
        + 0.04 * usage_gb
        + np.where(segment == "premium", 0.6, 0.0)
        + np.where(segment == "basic", -0.35, 0.0)
    )
    churn_prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, churn_prob)

    # Decoy leaky column: copy of target with tiny noise (should be flagged)
    leaky_score = churn + rng.normal(0, 0.02, size=n_rows)

    df = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(n_rows)],
            "age": age,
            "annual_income": income.round(2),
            "tenure_months": tenure_months,
            "monthly_usage_gb": usage_gb.round(2),
            "segment": segment,
            "region": region,
            "credit_score_text": score_str,
            "signup_date": signup_date,
            "leaky_churn_copy": leaky_score.round(4),
            "churn": churn,
        }
    )

    # Inject nulls
    null_cols = {
        "annual_income": 0.08,
        "tenure_months": 0.05,
        "credit_score_text": 0.10,
        "signup_date": 0.03,
    }
    for col, frac in null_cols.items():
        idx = rng.choice(n_rows, size=int(n_rows * frac), replace=False)
        df.loc[idx, col] = np.nan

    # All-null decoy column (should be dropped)
    df["all_null_feature"] = np.nan

    # Single-category column
    df["legacy_flag"] = "legacy"

    # Outliers in usage
    outlier_idx = rng.choice(n_rows, size=25, replace=False)
    df.loc[outlier_idx, "monthly_usage_gb"] = rng.uniform(250, 400, size=len(outlier_idx))

    # Duplicate rows
    dup_rows = df.sample(20, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    df.to_csv(dest, index=False)

    ground_truth = {
        "leaky_columns": ["leaky_churn_copy"],
        "drop_columns": ["all_null_feature", "customer_id"],
        "high_null_columns": list(null_cols.keys()),
        "single_category_columns": ["legacy_flag"],
        "expected_problem_type": "classification",
        "expected_target": "churn",
    }
    sidecar = output_dir / "synthetic_ground_truth.meta.json"
    sidecar.write_text(json.dumps(ground_truth, indent=2), encoding="utf-8")

    project_root = root or output_dir.parents[1]

    return {
        "id": "synthetic_ground_truth",
        "path": str(dest.relative_to(project_root)).replace("\\", "/"),
        "filename": DEFAULT_FILENAME,
        "target_column": "churn",
        "problem_type": "classification",
        "description": (
            "Controlled synthetic set with nulls, outliers, mixed dtypes, "
            "a decoy leaky column, and known correct pipeline decisions."
        ),
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": list(df.columns),
        "ground_truth_path": str(sidecar.relative_to(project_root)).replace("\\", "/"),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    raw_dir = root / "data" / "raw"
    meta = generate_synthetic_dataset(raw_dir)
    print(f"Wrote {meta['path']} ({meta['n_rows']} rows, {meta['n_cols']} cols)")
    print(f"Ground truth: {meta['ground_truth_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
