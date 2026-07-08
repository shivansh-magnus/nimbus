"""
Unit and integration tests for Day 8 features:
1. Optuna Hyperparameter Tuning
2. Tenacity LLM Call Backoff & Resilience
3. MLflow Local Experiment Tracking
"""

from __future__ import annotations

import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest
import mlflow

from automl_agents.tools.training import tune_model
from automl_agents.llm_client import llm_retry_decorator
from automl_agents.graph import graph


# ===========================================================================
# 1. Optuna Tuning Tests
# ===========================================================================

def test_optuna_tune_model_classification():
    # Generate simple synthetic binary classification dataset
    np.random.seed(42)
    X = np.random.randn(100, 4)
    # Target correlates with feature 0
    y = (X[:, 0] > 0).astype(int)
    
    df = pd.DataFrame(X, columns=["feat1", "feat2", "feat3", "feat4"])
    df["target"] = y

    # Run tune_model on LogisticRegression with 3 trials
    tuned_result = tune_model(
        df,
        target="target",
        model_id="LogisticRegression",
        problem_type="classification",
        metric="f1",
        cv=3,
        n_trials=3,
    )

    assert tuned_result is not None
    assert tuned_result["model_id"] == "LogisticRegression (Tuned)"
    assert "C" in tuned_result["best_params"]
    assert "mean_scores" in tuned_result
    assert "accuracy" in tuned_result["mean_scores"]
    assert "f1" in tuned_result["mean_scores"]
    assert 0.0 <= tuned_result["mean_scores"]["f1"] <= 1.0


def test_optuna_tune_model_regression():
    # Generate simple synthetic regression dataset
    np.random.seed(42)
    X = np.random.randn(100, 3)
    y = X[:, 0] * 2.5 + X[:, 1] * -1.2 + np.random.randn(100) * 0.1
    
    df = pd.DataFrame(X, columns=["feat1", "feat2", "feat3"])
    df["target"] = y

    # Run tune_model on RandomForest with 2 trials
    tuned_result = tune_model(
        df,
        target="target",
        model_id="RandomForest",
        problem_type="regression",
        metric="r2",
        cv=2,
        n_trials=2,
    )

    assert tuned_result is not None
    assert tuned_result["model_id"] == "RandomForest (Tuned)"
    assert "n_estimators" in tuned_result["best_params"]
    assert "max_depth" in tuned_result["best_params"]
    assert "mean_scores" in tuned_result
    assert "r2" in tuned_result["mean_scores"]


# ===========================================================================
# 2. Tenacity Retry Decorator Tests
# ===========================================================================

def test_llm_retry_decorator_success_after_failures():
    call_count = 0

    @llm_retry_decorator
    def mock_llm_call():
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise ValueError("Simulated transient rate limit error (429)")
        return "Success"

    # We patch the wait time in tenacity to run the test instantly
    with patch("tenacity.nap.time.sleep", return_value=None):
        result = mock_llm_call()

    assert result == "Success"
    assert call_count == 3


def test_llm_retry_decorator_fail_completely():
    call_count = 0

    @llm_retry_decorator
    def mock_llm_call_always_fails():
        nonlocal call_count
        call_count += 1
        raise RuntimeError("Persistent server error")

    with patch("tenacity.nap.time.sleep", return_value=None):
        with pytest.raises(RuntimeError):
            mock_llm_call_always_fails()

    # stop_after_attempt(5) means it should attempt exactly 5 times
    assert call_count == 5


# ===========================================================================
# 3. MLflow Local File-Store Tracking Smoke Test
# ===========================================================================

def test_mlflow_logging_integration():
    # Use synthetic ground truth dataset
    csv_path = Path("data/raw/synthetic_ground_truth.csv")
    assert csv_path.exists(), "Synthetic dataset must be generated first."

    # Setup a minimized initial state to run quickly
    # Skip profiler and data_prep by feeding ready sliced snapshot so it runs fast
    df = pd.read_csv(csv_path)
    # Drop known leak so it runs tuning and saves to mlflow
    df_clean = df.drop(columns=["customer_id", "legacy_flag", "all_null_feature", "leaky_churn_copy"])
    
    # Save a temporary parquet snapshot in runs
    temp_dir = Path("runs/test_mlflow_smoke_run")
    temp_dir.mkdir(parents=True, exist_ok=True)
    cleaned_parquet = temp_dir / "cleaned.parquet"
    df_clean.to_parquet(cleaned_parquet)

    from automl_agents.schemas import EDAReport, ColumnProfile

    eda_report = EDAReport(
        n_rows=len(df_clean),
        n_cols=len(df_clean.columns),
        columns=[
            ColumnProfile(column=c, dtype=str(df_clean[c].dtype), missing_fraction=0.0, cardinality=len(df_clean[c].unique()))
            for c in df_clean.columns
        ],
        problem_type="classification",
    )

    initial_state = {
        "dataset_path": str(csv_path),
        "target_column": "churn",
        "eda_report": eda_report,
        "cleaned_data_path": str(cleaned_parquet),
        "prep_plan": {
            "drop_cols": ["customer_id", "legacy_flag", "all_null_feature", "leaky_churn_copy"],
            "datetime_cols": ["signup_date"],
            "scale_strategy": "standard",
            "iqr_k": 1.5,
        },
        "selected_features": ["monthly_usage_gb", "tenure_months", "annual_income"],
        "selection_rationale": "Manual subset for smoke test",
        "model_results": [],
        "best_model_id": None,
        "report_path": None,
        "stage_log": [],
        "retry_count": {"data_prep": 0},
        "token_usage": [],
    }

    context = {
        "run_id": "test_mlflow_smoke_run_id",
        "llm_provider": "gemini",
        "model_name": "gemini-3.1-flash-lite",
        "max_retries": 2,
        "token_budget": None,
    }

    # Run the graph starting from selector to reporter
    # We mock the reporter's LLM response so it runs fast and doesn't hit limits
    from automl_agents.nodes.reporter import ReportExecutiveSummary
    mock_summary = ReportExecutiveSummary(
        executive_summary="Smoke test executive summary",
        recommendations=["Deploy this model!"],
    )
    
    # Also mock trainer metric selection LLM call
    from automl_agents.nodes.trainer import TrainerMetricSelection
    mock_metric = TrainerMetricSelection(
        metric="f1",
        rationale="F1 is best for classification smoke test",
    )

    def custom_get_llm(provider=None, model=None, temperature=0.0):
        # Return a mock structured LLM returning appropriate structure
        mock_llm = MagicMock()
        mock_structured = MagicMock()
        mock_llm.with_structured_output.return_value = mock_structured
        
        def invoke_side_effect(messages, **kwargs):
            # Check structure requested
            schema = mock_llm.with_structured_output.call_args[0][0]
            if schema == ReportExecutiveSummary:
                return {"parsed": mock_summary, "raw": MagicMock()}
            elif schema == TrainerMetricSelection:
                return {"parsed": mock_metric, "raw": MagicMock()}
            return {"parsed": MagicMock(), "raw": MagicMock()}

        mock_structured.invoke.side_effect = invoke_side_effect
        return mock_llm

    # Execute graph with patched get_llm
    with patch("automl_agents.nodes.trainer.get_llm", side_effect=custom_get_llm), \
         patch("automl_agents.nodes.reporter.get_llm", side_effect=custom_get_llm):
        final_state = graph.invoke(initial_state, context=context)

    # Verify report was written
    report_file = Path(final_state["report_path"])
    assert report_file.exists()

    # Query MLflow local client
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("nimbus-automl")
    assert experiment is not None

    # Get runs for this experiment
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) > 0

    # Find the run we just logged
    smoke_run = next((r for r in runs if r.info.run_name == "test_mlflow_smoke_run_id"), None)
    assert smoke_run is not None, "MLflow run was not created/found!"

    # Assert logged params and metrics
    params = smoke_run.data.params
    metrics = smoke_run.data.metrics

    assert params.get("target_column") == "churn"
    assert params.get("best_model_id") is not None
    assert metrics.get("prep_retries") == 0.0

    # Cleanup artifacts and run dir
    import shutil
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
