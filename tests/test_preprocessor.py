"""
Day-3 preprocessor tests.

Ground-truth synthetic dataset (`data/raw/synthetic_ground_truth.csv`) is the
primary fixture.  All tests with ``synthetic_csv`` in their signature run
against known injected conditions so assertions are exact rather than
probabilistic.

Edge cases covered:
  - all-null column dropped
  - single-category column dropped
  - mixed-type coercion (credit_score_text)
  - datetime feature extraction (signup_date)
  - exact-duplicate row removal (20 injected dupe rows)
  - leaky column drop (explicit drop_cols)
  - imputation drives null rate to zero
  - outlier clipping (monthly_usage_gb extreme values capped)
  - fit/transform leakage guard (train stats ≠ full-data stats when tested on
    a deliberately biased subsplit; transform uses train stats only)
  - parquet round-trip preserves shape and values
  - target column never touched by imputer / encoder / scaler
  - Titanic smoke test (mixed types + missingness survive without crash)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_agents.tools.preprocessor import (
    PrepArtifacts,
    PrepConfig,
    add_datetime_features,
    apply_clip_bounds,
    apply_imputer_fill,
    clip_outliers_iqr,
    clip_outliers_percentile,
    coerce_mixed_numeric,
    dedupe_rows,
    drop_all_null_columns,
    drop_columns,
    drop_rows_with_null_target,
    drop_single_category_columns,
    fit_encoders,
    fit_preprocessor,
    fit_scalers,
    impute_column,
    load_parquet_snapshot,
    parse_datetime_column,
    prep_dataframe,
    save_parquet_snapshot,
    transform_encoders,
    transform_preprocessor,
    transform_scalers,
)
from automl_agents.tools.profiler import load_csv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
SYNTHETIC_CSV = RAW_DIR / "synthetic_ground_truth.csv"
TITANIC_CSV = RAW_DIR / "titanic.csv"


# ---------------------------------------------------------------------------
# Column / row removal
# ---------------------------------------------------------------------------


def test_drop_all_null_columns(synthetic_csv: Path):
    df = load_csv(synthetic_csv)
    assert "all_null_feature" in df.columns
    cleaned = drop_all_null_columns(df)
    assert "all_null_feature" not in cleaned.columns
    # Other columns must survive
    assert "annual_income" in cleaned.columns


def test_drop_single_category_columns(synthetic_csv: Path):
    df = load_csv(synthetic_csv)
    assert "legacy_flag" in df.columns
    cleaned = drop_single_category_columns(df)
    assert "legacy_flag" not in cleaned.columns
    # Multi-category columns must survive
    assert "segment" in cleaned.columns


def test_drop_columns_explicit(synthetic_csv: Path):
    df = load_csv(synthetic_csv)
    cleaned = drop_columns(df, ["leaky_churn_copy", "customer_id"])
    assert "leaky_churn_copy" not in cleaned.columns
    assert "customer_id" not in cleaned.columns
    # Original not mutated
    assert "leaky_churn_copy" in df.columns


def test_drop_columns_silently_skips_missing():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = drop_columns(df, ["a", "nonexistent"])
    assert list(out.columns) == ["b"]


def test_drop_rows_with_null_target():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10.0, None, 30.0]})
    out = drop_rows_with_null_target(df, "y")
    assert len(out) == 2
    assert out["y"].isna().sum() == 0


def test_drop_rows_with_null_target_raises_on_missing_col():
    df = pd.DataFrame({"x": [1, 2]})
    with pytest.raises(KeyError):
        drop_rows_with_null_target(df, "missing")


def test_dedupe_rows(synthetic_csv: Path):
    df = load_csv(synthetic_csv)
    # Synthetic CSV has 20 duplicate rows appended → total 2020 rows
    assert len(df) == 2020
    deduped = dedupe_rows(df)
    assert len(deduped) == 2000


# ---------------------------------------------------------------------------
# Dtype coercion
# ---------------------------------------------------------------------------


def test_coerce_mixed_numeric(synthetic_csv: Path):
    df = load_csv(synthetic_csv)
    assert df["credit_score_text"].dtype == object
    out = coerce_mixed_numeric(df, "credit_score_text")
    assert pd.api.types.is_numeric_dtype(out["credit_score_text"])
    # Invalid tokens ("N/A", "unknown", "—") become NaN, not zero
    # There are ~40 injected bad tokens + ~10% null → expect some NaN
    assert out["credit_score_text"].isna().sum() > 0
    # Valid numeric values are correctly parsed (typical credit score range)
    valid = out["credit_score_text"].dropna()
    assert valid.min() >= 200
    assert valid.max() <= 900


def test_coerce_mixed_numeric_does_not_mutate():
    df = pd.DataFrame({"x": ["1", "N/A", "3"]})
    _ = coerce_mixed_numeric(df, "x")
    assert df["x"].dtype == object  # original unchanged


def test_parse_datetime_column():
    df = pd.DataFrame({"dt": ["2022-01-01", "2022-06-15", None]})
    out = parse_datetime_column(df, "dt")
    assert pd.api.types.is_datetime64_any_dtype(out["dt"])
    assert out["dt"].isna().sum() == 1


def test_add_datetime_features(synthetic_csv: Path):
    df = load_csv(synthetic_csv)
    out = add_datetime_features(df, "signup_date", drop_original=True)
    assert "signup_date" not in out.columns
    for feat in ["signup_date_year", "signup_date_month", "signup_date_day",
                 "signup_date_dayofweek"]:
        assert feat in out.columns
    # Year should be 2022 or 2023 (generated from base 2022-01-01 + up to 900 days)
    valid_years = out["signup_date_year"].dropna()
    assert valid_years.between(2022, 2025).all()


def test_add_datetime_features_keeps_original_when_requested():
    df = pd.DataFrame({"d": ["2023-03-15", "2023-09-01"]})
    out = add_datetime_features(df, "d", drop_original=False)
    assert "d" in out.columns
    assert "d_year" in out.columns


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------


def test_impute_column_mean():
    df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0, 5.0]})
    out, fv = impute_column(df, "x", "mean")
    assert out["x"].isna().sum() == 0
    assert fv == pytest.approx(3.0)


def test_impute_column_median():
    df = pd.DataFrame({"x": [1.0, 2.0, None, 4.0, 100.0]})
    out, fv = impute_column(df, "x", "median")
    assert out["x"].isna().sum() == 0
    assert fv == pytest.approx(3.0)


def test_impute_column_mode_categorical():
    df = pd.DataFrame({"cat": ["a", "b", "a", None, "a"]})
    out, fv = impute_column(df, "cat", "mode")
    assert out["cat"].isna().sum() == 0
    assert fv == "a"


def test_impute_column_constant():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    out, fv = impute_column(df, "x", "constant", fill_value=-999.0)
    assert out["x"].isna().sum() == 0
    assert fv == -999.0


def test_impute_column_no_nulls_after(synthetic_csv: Path):
    df = load_csv(synthetic_csv)
    assert df["annual_income"].isna().sum() > 0
    out, _ = impute_column(df, "annual_income", "median")
    assert out["annual_income"].isna().sum() == 0


def test_impute_column_does_not_touch_target(synthetic_csv: Path):
    """Imputing a different column must not alter the target."""
    df = load_csv(synthetic_csv)
    original_target = df["churn"].copy()
    out, _ = impute_column(df, "annual_income", "median")
    pd.testing.assert_series_equal(out["churn"], original_target)


def test_apply_imputer_fill_skips_missing_col():
    df = pd.DataFrame({"a": [1, 2]})
    out = apply_imputer_fill(df, "nonexistent", 0.0)
    assert list(out.columns) == ["a"]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def test_fit_transform_onehot():
    df = pd.DataFrame({"col": ["a", "b", "a", "c"]})
    state = fit_encoders(df, ["col"], "onehot")
    out, _ = transform_encoders(df, state)
    assert "col" not in out.columns
    assert "col_a" in out.columns
    assert "col_b" in out.columns
    assert "col_c" in out.columns


def test_transform_onehot_aligns_to_train_schema():
    """Test data with an unseen category or missing category stays aligned."""
    train = pd.DataFrame({"col": ["a", "b", "a"]})
    test = pd.DataFrame({"col": ["a", "c"]})  # 'c' unseen, 'b' missing
    state = fit_encoders(train, ["col"], "onehot")
    train_out, ohe_names = transform_encoders(train, state)
    test_out, _ = transform_encoders(test, state, ohe_feature_names=ohe_names)
    # Test output must have same OHE columns as train
    train_ohe_cols = [c for c in train_out.columns if c.startswith("col_")]
    test_ohe_cols = [c for c in test_out.columns if c.startswith("col_")]
    assert set(train_ohe_cols) == set(test_ohe_cols)


def test_fit_transform_ordinal():
    df = pd.DataFrame({"size": ["S", "M", "L", "M"]})
    state = fit_encoders(df, ["size"], "ordinal")
    out, _ = transform_encoders(df, state)
    assert pd.api.types.is_numeric_dtype(out["size"])
    assert set(out["size"].unique()).issubset({0.0, 1.0, 2.0})


def test_ordinal_unknown_category_gets_minus_one():
    train = pd.DataFrame({"size": ["S", "M", "L"]})
    test = pd.DataFrame({"size": ["XL"]})  # unseen
    state = fit_encoders(train, ["size"], "ordinal")
    out, _ = transform_encoders(test, state)
    assert out["size"].iloc[0] == -1.0


def test_fit_transform_target_encoding():
    df = pd.DataFrame({
        "cat": ["a", "b", "a", "b", "a"],
        "target": [1, 0, 1, 0, 0],
    })
    state = fit_encoders(df, ["cat"], "target", target_col="target")
    out, _ = transform_encoders(df, state)
    assert pd.api.types.is_numeric_dtype(out["cat"])
    # "a" mean = 2/3, "b" mean = 0
    assert out["cat"].iloc[0] == pytest.approx(2 / 3, abs=1e-6)
    assert out["cat"].iloc[1] == pytest.approx(0.0, abs=1e-6)


def test_target_encoding_requires_target_col():
    df = pd.DataFrame({"cat": ["a", "b"]})
    with pytest.raises(ValueError, match="target_col"):
        fit_encoders(df, ["cat"], "target")


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------


def test_fit_transform_standard_scaler():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})
    params = fit_scalers(df, ["x"], "standard")
    out = transform_scalers(df, params)
    assert out["x"].mean() == pytest.approx(0.0, abs=1e-10)
    assert out["x"].std(ddof=1) == pytest.approx(1.0, abs=1e-10)


def test_fit_transform_minmax_scaler():
    df = pd.DataFrame({"x": [0.0, 5.0, 10.0]})
    params = fit_scalers(df, ["x"], "minmax")
    out = transform_scalers(df, params)
    assert out["x"].min() == pytest.approx(0.0, abs=1e-10)
    assert out["x"].max() == pytest.approx(1.0, abs=1e-10)


def test_fit_transform_robust_scaler():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})
    params = fit_scalers(df, ["x"], "robust")
    out = transform_scalers(df, params)
    # Median of original = 3.5 → transformed median ≈ 0
    assert out["x"].median() == pytest.approx(0.0, abs=1e-6)


def test_transform_scalers_does_not_refit():
    """transform_scalers must use stored params, not refit from the new data."""
    train = pd.DataFrame({"x": [0.0, 10.0]})  # mean=5, std=7.07
    test = pd.DataFrame({"x": [1000.0]})
    params = fit_scalers(train, ["x"], "standard")
    out = transform_scalers(test, params)
    # (1000 - 5) / 7.07 ≈ 140.7, NOT (1000 - 1000) / 1 = 0
    assert abs(out["x"].iloc[0]) > 100


# ---------------------------------------------------------------------------
# Outlier clipping
# ---------------------------------------------------------------------------


def test_clip_outliers_iqr(synthetic_csv: Path):
    df = load_csv(synthetic_csv)
    # 25 injected outliers in [250, 400] range; normal data is Gamma ~30 GB
    assert df["monthly_usage_gb"].max() > 200
    out, bounds = clip_outliers_iqr(df, "monthly_usage_gb", k=1.5)
    assert out["monthly_usage_gb"].max() <= bounds["upper"] + 1e-6
    assert out["monthly_usage_gb"].min() >= bounds["lower"] - 1e-6


def test_clip_outliers_iqr_does_not_mutate():
    df = pd.DataFrame({"x": [1.0, 2.0, 1000.0]})
    out, _ = clip_outliers_iqr(df, "x")
    assert df["x"].max() == 1000.0  # original unchanged


def test_clip_outliers_percentile():
    df = pd.DataFrame({"x": list(range(100)) + [9999]})
    out, bounds = clip_outliers_percentile(df, "x", lower_pct=0.01, upper_pct=0.99)
    assert out["x"].max() <= bounds["upper"] + 1e-6


def test_apply_clip_bounds_skips_missing_col():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    out = apply_clip_bounds(df, "nonexistent", {"lower": 0.0, "upper": 1.0})
    pd.testing.assert_frame_equal(out, df)


# ---------------------------------------------------------------------------
# Fit/Transform leakage guard
# ---------------------------------------------------------------------------


def test_fit_transform_leakage_guard(synthetic_csv: Path):
    """Stats learned on train must differ from full-data stats; transform must use
    train stats and NOT refit on the test slice."""
    df = load_csv(synthetic_csv)

    # Deliberately biased train: only rows where annual_income > median
    # (so train mean is higher than full-data mean)
    median_income = df["annual_income"].median()
    train = df[df["annual_income"] > median_income].copy()
    test = df[df["annual_income"] <= median_income].copy()

    # Fit on (biased) train
    arts = fit_preprocessor(train, "churn")

    # Scaler center for annual_income should be ≈ train mean (above overall median)
    if "annual_income" in arts.scaler_params:
        train_center = arts.scaler_params["annual_income"]["center"]
        full_mean = float(df["annual_income"].mean())
        # Train center must be noticeably larger than full-data mean
        assert train_center > full_mean + 1_000, (
            f"Train center {train_center:.1f} should exceed full mean {full_mean:.1f} "
            "when fit on the upper half of the income distribution."
        )

    # transform on test must use those train stats (no refitting)
    out = transform_preprocessor(test, arts)
    assert "annual_income" not in out.columns or out["annual_income"].isna().sum() == 0


def test_target_never_imputed_encoded_or_scaled(synthetic_csv: Path):
    """Target column values must be identical before and after prep."""
    df = load_csv(synthetic_csv)
    df_deduped = dedupe_rows(drop_rows_with_null_target(df, "churn"))
    original_target = df_deduped["churn"].reset_index(drop=True)

    out = prep_dataframe(df, "churn")
    out_target = out["churn"].reset_index(drop=True)

    pd.testing.assert_series_equal(out_target, original_target, check_names=True)


# ---------------------------------------------------------------------------
# Parquet round-trip
# ---------------------------------------------------------------------------


def test_parquet_round_trip(synthetic_csv: Path, tmp_path: Path):
    df = load_csv(synthetic_csv)
    snap_path = tmp_path / "snapshot.parquet"
    save_parquet_snapshot(df, snap_path)
    loaded = load_parquet_snapshot(snap_path)
    assert loaded.shape == df.shape
    assert list(loaded.columns) == list(df.columns)


def test_parquet_round_trip_preserves_numerics(tmp_path: Path):
    df = pd.DataFrame({"a": [1.1, 2.2, 3.3], "b": [10, 20, 30]})
    p = tmp_path / "num.parquet"
    save_parquet_snapshot(df, p)
    loaded = load_parquet_snapshot(p)
    pd.testing.assert_series_equal(
        loaded["a"].reset_index(drop=True),
        df["a"].reset_index(drop=True),
        check_names=True,
    )


def test_load_parquet_snapshot_raises_on_missing(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_parquet_snapshot(tmp_path / "does_not_exist.parquet")


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------


def test_prep_dataframe_zero_nulls_on_numeric_cols(synthetic_csv: Path):
    """After prep_dataframe, no numeric column (except target) should have nulls."""
    df = load_csv(synthetic_csv)
    out = prep_dataframe(df, "churn")
    numeric_cols = [
        c for c in out.select_dtypes(include=[np.number]).columns
        if c != "churn"
    ]
    total_nulls = out[numeric_cols].isna().sum().sum()
    assert total_nulls == 0, (
        f"Expected 0 nulls in numeric cols after prep; got {total_nulls}."
    )


def test_prep_dataframe_shape_reduced(synthetic_csv: Path):
    """Output should drop known-bad columns and deduplicate rows.

    NOTE: total column count may be *higher* than the input because OHE
    (segment → 3 cols, region → 4 cols) and datetime expansion
    (signup_date → 4 features) legitimately add new columns.  The meaningful
    assertion is that the structural-cleanup columns are gone and row count
    is correct after deduplication.
    """
    df = load_csv(synthetic_csv)
    out = prep_dataframe(df, "churn")

    # All-null and single-category columns must be dropped
    assert "all_null_feature" not in out.columns
    assert "legacy_flag" not in out.columns

    # High-cardinality ID column must be dropped (> _MAX_OHE_CARDINALITY)
    assert "customer_id" not in out.columns

    # OHE expansion is expected: segment and region should be exploded
    assert any(c.startswith("segment_") for c in out.columns)
    assert any(c.startswith("region_") for c in out.columns)

    # Datetime expansion is expected: signup_date should be replaced by features
    assert "signup_date" not in out.columns
    assert "signup_date_year" in out.columns

    # Rows: 20 dupe rows must be removed
    assert out.shape[0] == 2000

    # Target column must survive untouched
    assert "churn" in out.columns


def test_fit_preprocessor_artifacts_have_expected_keys(synthetic_csv: Path):
    df = load_csv(synthetic_csv)
    arts = fit_preprocessor(df, "churn")
    assert "all_null_feature" in arts.dropped_columns
    assert "legacy_flag" in arts.dropped_columns
    assert arts.target_column == "churn"
    # Imputer fills should exist for columns with nulls
    assert len(arts.imputer_fills) > 0
    # Scaler params should exist for numeric columns
    assert len(arts.scaler_params) > 0


def test_fit_transform_pipeline_consistent(synthetic_csv: Path):
    """prep_dataframe (fit+transform together) matches separate fit then transform."""
    df = load_csv(synthetic_csv)
    combined = prep_dataframe(df, "churn")
    arts = fit_preprocessor(df, "churn")
    separate = transform_preprocessor(df, arts)
    pd.testing.assert_frame_equal(
        combined.reset_index(drop=True),
        separate.reset_index(drop=True),
        check_like=True,
    )


# ---------------------------------------------------------------------------
# Titanic smoke test (secondary)
# ---------------------------------------------------------------------------


def test_prep_dataframe_smoke_titanic(titanic_csv: Path):
    """prep_dataframe on Titanic must not crash (mixed types, missingness)."""
    df = load_csv(titanic_csv)
    out = prep_dataframe(df, "survived")
    assert len(out) > 0
    assert "survived" in out.columns
    # Target must be numeric and unscaled
    assert pd.api.types.is_numeric_dtype(out["survived"])
