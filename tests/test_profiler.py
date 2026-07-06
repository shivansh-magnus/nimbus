"""Day-2 profiler tool tests — pure functions on 2-3+ sample CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from automl_agents.tools.profiler import (
    compute_correlation_flags,
    compute_missingness,
    detect_outliers_iqr,
    detect_problem_type,
    infer_dtypes,
    load_csv,
    profile_dataframe,
    profile_dataset,
)


def _column_map(report):
    return {c.column: c for c in report.columns}


# --- synthetic ground-truth dataset ---


def test_profile_dataset_returns_valid_eda_report(synthetic_csv: Path):
    report = profile_dataset(synthetic_csv, "churn")
    assert report.n_rows > 0
    assert report.n_cols == 13
    assert report.problem_type == "classification"
    assert len(report.columns) == report.n_cols


def test_missingness_fractions_on_synthetic(synthetic_csv: Path):
    df = load_csv(synthetic_csv)
    missing = compute_missingness(df)

    assert missing["all_null_feature"] == pytest.approx(1.0)
    assert 0.05 <= missing["annual_income"] <= 0.12
    assert 0.03 <= missing["tenure_months"] <= 0.08
    assert 0.07 <= missing["credit_score_text"] <= 0.13


def test_cardinality_single_category_on_synthetic(synthetic_csv: Path):
    report = profile_dataset(synthetic_csv, "churn")
    cols = _column_map(report)
    assert cols["legacy_flag"].cardinality == 1
    assert cols["customer_id"].cardinality == report.n_rows - 20  # 20 duplicate rows


def test_infer_dtypes_on_mixed_column(synthetic_csv: Path):
    df = load_csv(synthetic_csv)
    dtypes = infer_dtypes(df)
    assert dtypes["credit_score_text"] == "mixed_numeric"
    assert dtypes["signup_date"] == "datetime_string"
    assert dtypes["annual_income"] == "float64"


def test_target_balance_sums_to_one_on_synthetic(synthetic_csv: Path):
    report = profile_dataset(synthetic_csv, "churn")
    assert report.target_balance is not None
    total = sum(c.proportion for c in report.target_balance)
    assert total == pytest.approx(1.0)
    assert len(report.target_balance) == 2


def test_correlation_flags_leaky_column_on_synthetic(synthetic_csv: Path):
    df = load_csv(synthetic_csv)
    flags = compute_correlation_flags(df, "churn", threshold=0.9)
    leaky_pairs = {
        (p.col_a, p.col_b)
        for p in flags
        if "leaky_churn_copy" in (p.col_a, p.col_b) and "churn" in (p.col_a, p.col_b)
    }
    assert leaky_pairs, "expected leaky_churn_copy vs churn above threshold"


def test_outlier_iqr_detects_usage_spikes_on_synthetic(synthetic_csv: Path):
    df = load_csv(synthetic_csv)
    result = detect_outliers_iqr(df, "monthly_usage_gb")
    assert result["n_outliers"] >= 20
    assert result["outlier_fraction"] > 0.005


# --- titanic (classification, mixed types, missingness) ---


def test_titanic_classification_and_missingness(titanic_csv: Path):
    report = profile_dataset(titanic_csv, "survived")
    cols = _column_map(report)

    assert report.problem_type == "classification"
    assert cols["age"].missing_fraction > 0.1
    assert cols["deck"].missing_fraction > 0.5
    assert cols["embarked"].dtype == "object"
    assert report.target_balance is not None
    assert len(report.target_balance) == 2


# --- wine quality (ordinal-ish classification) ---


def test_wine_quality_detected_as_classification(wine_csv: Path):
    report = profile_dataset(wine_csv, "quality")
    assert report.problem_type == "classification"
    assert report.target_balance is not None
    assert len(report.target_balance) >= 3


# --- california housing (regression) ---


def test_california_housing_detected_as_regression(california_csv: Path):
    report = profile_dataset(california_csv, "MedHouseVal")
    assert report.problem_type == "regression"
    assert report.target_balance is None


def test_profile_dataframe_matches_profile_dataset(synthetic_csv: Path):
    df = load_csv(synthetic_csv)
    from_path = profile_dataset(synthetic_csv, "churn")
    from_df = profile_dataframe(df, "churn")
    assert from_path.model_dump() == from_df.model_dump()


def test_load_csv_rejects_empty_file(tmp_path: Path):
    empty = tmp_path / "empty.csv"
    empty.write_text("a,b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_csv(empty)


def test_detect_problem_type_requires_target():
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(KeyError):
        detect_problem_type(df, "missing")
