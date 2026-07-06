"""
Download public CSV fixtures for Day-2 profiler tests.

Usage:
    uv run python scripts/download_datasets.py
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd
from sklearn.datasets import fetch_california_housing

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

TITANIC_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
WINE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
)


def download_titanic(dest: Path) -> dict:
    urlretrieve(TITANIC_URL, dest)
    df = pd.read_csv(dest)
    return {
        "id": "titanic",
        "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
        "target_column": "survived",
        "problem_type": "classification",
        "n_rows": len(df),
        "n_cols": len(df.columns),
    }


def download_wine_quality(dest: Path) -> dict:
    urlretrieve(WINE_URL, dest)
    # UCI file uses semicolons; normalize to comma-separated for consistency
    df = pd.read_csv(dest, sep=";")
    df.to_csv(dest, index=False)
    return {
        "id": "wine_quality_red",
        "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
        "target_column": "quality",
        "problem_type": "classification",
        "n_rows": len(df),
        "n_cols": len(df.columns),
    }


def write_california_housing(dest: Path) -> dict:
    data = fetch_california_housing(as_frame=True)
    df = data.frame
    df.to_csv(dest, index=False)
    return {
        "id": "california_housing",
        "path": str(dest.relative_to(ROOT)).replace("\\", "/"),
        "target_column": "MedHouseVal",
        "problem_type": "regression",
        "n_rows": len(df),
        "n_cols": len(df.columns),
    }


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    entries = [
        download_titanic(RAW_DIR / "titanic.csv"),
        download_wine_quality(RAW_DIR / "wine_quality_red.csv"),
        write_california_housing(RAW_DIR / "california_housing.csv"),
    ]

    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")

    for entry in entries:
        print(f"Ready {entry['path']} ({entry['n_rows']} rows, target={entry['target_column']})")
    print(f"Wrote {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
