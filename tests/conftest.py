"""Shared fixtures for profiler integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.generate_synthetic import generate_synthetic_dataset

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"

SYNTHETIC_CSV = RAW_DIR / "synthetic_ground_truth.csv"
TITANIC_CSV = RAW_DIR / "titanic.csv"
WINE_CSV = RAW_DIR / "wine_quality_red.csv"
CALIFORNIA_CSV = RAW_DIR / "california_housing.csv"


@pytest.fixture(scope="session")
def synthetic_csv() -> Path:
    if not SYNTHETIC_CSV.exists():
        generate_synthetic_dataset(RAW_DIR, root=ROOT)
    return SYNTHETIC_CSV


@pytest.fixture(scope="session")
def titanic_csv() -> Path:
    if not TITANIC_CSV.exists():
        pytest.skip("Run: uv run python scripts/download_datasets.py")
    return TITANIC_CSV


@pytest.fixture(scope="session")
def wine_csv() -> Path:
    if not WINE_CSV.exists():
        pytest.skip("Run: uv run python scripts/download_datasets.py")
    return WINE_CSV


@pytest.fixture(scope="session")
def california_csv() -> Path:
    if not CALIFORNIA_CSV.exists():
        pytest.skip("Run: uv run python scripts/download_datasets.py")
    return CALIFORNIA_CSV
