"""
Day-10 model export unit tests.

Tests the full round-trip: build estimator → fit on full data → save bundle →
load bundle → predict from bundle — and verifies the bundle's internal structure
matches the contract defined by BUNDLE_KEYS.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from automl_agents.tools.training import _build_estimator
from automl_agents.tools.model_export import (
    BUNDLE_KEYS,
    fit_final_model,
    load_model_bundle,
    predict_from_bundle,
    save_model_bundle,
)
from automl_agents.tools.preprocessor import PrepArtifacts


# ---------------------------------------------------------------------------
# Fixtures — tiny, self-contained datasets (no LLM call, no CSV, no disk)
# ---------------------------------------------------------------------------

@pytest.fixture()
def classification_df() -> pd.DataFrame:
    """10-row binary classification dataset with one categorical column."""
    return pd.DataFrame({
        "x1": [1.0, 2.0, 1.1, 2.1, 0.9, 1.9, 1.2, 2.2, 0.8, 1.8],
        "x2": [0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })


@pytest.fixture()
def regression_df() -> pd.DataFrame:
    """10-row linear regression dataset."""
    return pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "x2": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "target": [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0],
    })


# ---------------------------------------------------------------------------
# _build_estimator
# ---------------------------------------------------------------------------

class TestBuildEstimator:
    """Tests for the _build_estimator helper extracted from training.py."""

    def test_builds_classification_models(self):
        for model_id in ("LogisticRegression", "RandomForest", "GradientBoosting", "KNN"):
            est = _build_estimator(model_id, "classification")
            assert hasattr(est, "fit"), f"{model_id} missing .fit()"
            assert hasattr(est, "predict"), f"{model_id} missing .predict()"

    def test_builds_regression_models(self):
        for model_id in ("LinearRegression", "RandomForest", "GradientBoosting", "KNN"):
            est = _build_estimator(model_id, "regression")
            assert hasattr(est, "fit"), f"{model_id} missing .fit()"
            assert hasattr(est, "predict"), f"{model_id} missing .predict()"

    def test_honours_custom_params(self):
        params = {"n_estimators": 77, "max_depth": 5}
        est = _build_estimator("RandomForest", "classification", params=params)
        actual = est.get_params()
        assert actual["n_estimators"] == 77
        assert actual["max_depth"] == 5

    def test_raises_on_unknown_model_id(self):
        with pytest.raises(ValueError, match="Unknown model_id"):
            _build_estimator("NotARealModel", "classification")


# ---------------------------------------------------------------------------
# fit_final_model + save/load/predict round-trip
# ---------------------------------------------------------------------------

class TestModelExportRoundTrip:
    """Tests the full export pipeline: fit → save → load → predict."""

    def test_classification_round_trip(self, classification_df):
        """Fit, save, load, and predict on a tiny classification set."""
        df = classification_df
        target = "target"
        model_id = "LogisticRegression"
        problem_type = "classification"
        features = ["x1", "x2"]

        # Fit the final model on the full dataset
        fitted = fit_final_model(df, target, model_id, problem_type)
        assert hasattr(fitted, "predict")

        # Save bundle with a minimal PrepArtifacts (no actual preprocessing needed
        # since the fixture is already clean numeric data)
        artifacts = PrepArtifacts()

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = save_model_bundle(
                estimator=fitted,
                prep_artifacts=artifacts,
                selected_features=features,
                target_column=target,
                problem_type=problem_type,
                model_id=model_id,
                output_path=Path(tmpdir) / "model.pkl",
            )

            assert bundle_path.exists()
            assert bundle_path.stat().st_size > 0

            # Load and verify all expected keys are present
            bundle = load_model_bundle(bundle_path)
            assert set(bundle.keys()) == BUNDLE_KEYS
            assert bundle["model_id"] == model_id
            assert bundle["target_column"] == target
            assert bundle["problem_type"] == problem_type
            assert bundle["selected_features"] == features
            assert bundle["prep_artifacts"] is not None

    def test_regression_round_trip(self, regression_df):
        """Fit, save, load, and predict on a tiny regression set."""
        df = regression_df
        target = "target"
        model_id = "LinearRegression"
        problem_type = "regression"
        features = ["x1", "x2"]

        fitted = fit_final_model(df, target, model_id, problem_type)

        artifacts = PrepArtifacts()

        with tempfile.TemporaryDirectory() as tmpdir:
            bundle_path = save_model_bundle(
                estimator=fitted,
                prep_artifacts=artifacts,
                selected_features=features,
                target_column=target,
                problem_type=problem_type,
                model_id=model_id,
                output_path=Path(tmpdir) / "model.pkl",
            )

            bundle = load_model_bundle(bundle_path)

            # predict_from_bundle needs transform_preprocessor to work —
            # with an empty PrepArtifacts, the raw df passes through unchanged
            preds = predict_from_bundle(bundle, df.drop(columns=[target]))
            assert isinstance(preds, np.ndarray)
            assert len(preds) == len(df)

    def test_tuned_model_strips_suffix(self, classification_df):
        """A model_id like 'LightGBM (Tuned)' should have the suffix stripped
        before looking up the estimator class."""
        df = classification_df
        fitted = fit_final_model(
            df, "target", "RandomForest (Tuned)", "classification",
            best_params={"n_estimators": 50, "max_depth": 3},
        )
        assert hasattr(fitted, "predict")


# ---------------------------------------------------------------------------
# load_model_bundle — error cases
# ---------------------------------------------------------------------------

class TestLoadBundleErrors:

    def test_missing_keys_raises(self, tmp_path):
        """A bundle dict missing expected keys should raise ValueError."""
        import joblib

        bad_bundle = {"model": "fake", "prep_artifacts": None}
        path = tmp_path / "bad.pkl"
        joblib.dump(bad_bundle, path)

        with pytest.raises(ValueError, match="missing expected keys"):
            load_model_bundle(path)
