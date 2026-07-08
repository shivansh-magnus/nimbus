"""
Unit and integration tests for Day 7 features (Conditional Routing & Bounded Retry Loops).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from automl_agents.schemas import EDAReport, ColumnProfile, CorrelationPair
from automl_agents.graph.pipeline import (
    route_after_profiler,
    route_after_trainer,
    route_after_supervisor,
)
from automl_agents.nodes.trainer import trainer_node
from automl_agents.nodes.supervisor import retry_supervisor_node, SupervisorDecision


# ===========================================================================
# 1. Routing Function Tests
# ===========================================================================

def test_route_after_profiler():
    # Classification
    state_cls = {
        "eda_report": EDAReport(
            n_rows=100,
            n_cols=5,
            columns=[],
            problem_type="classification",
        )
    }
    assert route_after_profiler(state_cls) == "classification_prep"

    # Regression
    state_reg = {
        "eda_report": EDAReport(
            n_rows=100,
            n_cols=5,
            columns=[],
            problem_type="regression",
        )
    }
    assert route_after_profiler(state_reg) == "regression_prep"

    # Default if report is missing
    assert route_after_profiler({}) == "classification_prep"


def test_route_after_trainer():
    # No validation errors -> reporter
    state_ok = {"validation_errors": []}
    assert route_after_trainer(state_ok) == "reporter"

    state_none = {"validation_errors": None}
    assert route_after_trainer(state_none) == "reporter"

    # Validation errors, retries < 2 -> retry_supervisor
    state_error_1 = {"validation_errors": ["Leakage detected"], "retry_count": {"data_prep": 0}}
    assert route_after_trainer(state_error_1) == "retry_supervisor"

    state_error_2 = {"validation_errors": ["Leakage detected"], "retry_count": {"data_prep": 1}}
    assert route_after_trainer(state_error_2) == "retry_supervisor"

    # Validation errors, retries >= 2 -> reporter (bounded loop safety check)
    state_error_3 = {"validation_errors": ["Leakage detected"], "retry_count": {"data_prep": 2}}
    assert route_after_trainer(state_error_3) == "reporter"


def test_route_after_supervisor():
    # Classification
    state_cls = {
        "eda_report": EDAReport(
            n_rows=100,
            n_cols=5,
            columns=[],
            problem_type="classification",
        )
    }
    assert route_after_supervisor(state_cls) == "classification_prep"

    # Regression
    state_reg = {
        "eda_report": EDAReport(
            n_rows=100,
            n_cols=5,
            columns=[],
            problem_type="regression",
        )
    }
    assert route_after_supervisor(state_reg) == "regression_prep"


# ===========================================================================
# 2. Target Leakage Validation Checks Tests (Trainer Node)
# ===========================================================================

@patch("automl_agents.nodes.trainer.load_parquet_snapshot")
@patch("automl_agents.nodes.trainer.run_model_battery")
@patch("automl_agents.nodes.trainer.get_llm")
@patch("automl_agents.nodes.trainer.record_token_usage")
def test_trainer_node_leakage_detection(
    mock_record_token, mock_get_llm, mock_run_battery, mock_load_parquet
):
    # Setup mock LLM structured output
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    
    # Mock LLM returned metric selection
    mock_parsed_metric = MagicMock()
    mock_parsed_metric.metric = "f1"
    mock_parsed_metric.rationale = "Test rationale"
    mock_structured.invoke.return_value = {
        "parsed": mock_parsed_metric,
        "raw": MagicMock(),
    }

    # Setup state
    eda_report = EDAReport(
        n_rows=100,
        n_cols=3,
        columns=[
            ColumnProfile(column="col_leak", dtype="float64", missing_fraction=0.0, cardinality=100),
            ColumnProfile(column="col_normal", dtype="float64", missing_fraction=0.0, cardinality=100),
            ColumnProfile(column="target", dtype="int64", missing_fraction=0.0, cardinality=2),
        ],
        problem_type="classification",
    )

    state = {
        "cleaned_data_path": "dummy_path",
        "selected_features": ["col_leak", "col_normal"],
        "target_column": "target",
        "eda_report": eda_report,
    }

    # Context mockup
    mock_runtime = MagicMock()
    mock_runtime.context = {
        "llm_provider": "gemini",
        "model_name": "gemini-3.1-flash-lite",
    }

    # Mock dataset: col_leak correlates at 1.0 with target, col_normal correlates at 0.1
    # Target is binary [0, 1]
    df_mock = pd.DataFrame({
        "col_leak": [0.0, 1.0, 0.0, 1.0],
        "col_normal": [0.2, 0.9, 0.4, 0.1],
        "target": [0, 1, 0, 1],
    })
    mock_load_parquet.return_value = df_mock

    # Mock model battery results: one model (LightGBM) gets perfect 1.0 F1 score
    mock_run_battery.return_value = [
        {
            "model_id": "LightGBM",
            "mean_scores": {"accuracy": 1.0, "f1": 1.0},
            "std_scores": {"accuracy": 0.0, "f1": 0.0},
        },
        {
            "model_id": "RandomForest",
            "mean_scores": {"accuracy": 0.8, "f1": 0.8},
            "std_scores": {"accuracy": 0.05, "f1": 0.05},
        }
    ]

    # Run the trainer node
    res = trainer_node(state, mock_runtime)

    # Assertions
    assert "validation_errors" in res
    errors = res["validation_errors"]
    assert errors is not None
    assert len(errors) == 3  # 1 for correlation, 2 for score checks (accuracy and f1)
    
    # Check correlation error message
    assert any("col_leak" in err and "correlates" in err for err in errors)
    # Check score check error message
    assert any("LightGBM" in err and "perfect score" in err for err in errors)


# ===========================================================================
# 3. Retry Supervisor Node Tests
# ===========================================================================

@patch("automl_agents.nodes.supervisor.get_llm")
@patch("automl_agents.nodes.supervisor.record_token_usage")
def test_retry_supervisor_node(mock_record_token, mock_get_llm):
    # Setup mock LLM structured output returning supervisor decision
    mock_llm = MagicMock()
    mock_get_llm.return_value = mock_llm
    mock_structured = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured
    
    mock_parsed_decision = SupervisorDecision(
        columns_to_drop=["col_leak"],
        explanation="Detected target leakage from perfect scores and high correlation in col_leak.",
    )
    mock_structured.invoke.return_value = {
        "parsed": mock_parsed_decision,
        "raw": MagicMock(),
    }

    # Setup state
    eda_report = EDAReport(
        n_rows=100,
        n_cols=3,
        columns=[
            ColumnProfile(column="col_leak", dtype="float64", missing_fraction=0.0, cardinality=100),
            ColumnProfile(column="col_normal", dtype="float64", missing_fraction=0.0, cardinality=100),
            ColumnProfile(column="target", dtype="int64", missing_fraction=0.0, cardinality=2),
        ],
        problem_type="classification",
        concerns=["Initial concern"],
    )

    state = {
        "target_column": "target",
        "selected_features": ["col_leak", "col_normal"],
        "validation_errors": ["Model achieved perfect score", "Col leak correlates at 1.0"],
        "eda_report": eda_report,
        "retry_count": {"data_prep": 0},
    }

    mock_runtime = MagicMock()
    mock_runtime.context = {
        "llm_provider": "gemini",
        "model_name": "gemini-3.1-flash-lite",
    }

    # Run supervisor node
    res = retry_supervisor_node(state, mock_runtime)

    # Assertions
    assert res["validation_errors"] == []  # Cleared
    assert res["retry_count"]["data_prep"] == 1  # Incremented
    
    # Verify eda_report concerns updated
    updated_report = res["eda_report"]
    assert updated_report is not None
    assert "Leaky feature 'col_leak' detected. Drop this column." in updated_report.concerns
    assert "Supervisor explanation: Detected target leakage from perfect scores and high correlation in col_leak." in updated_report.concerns
    assert "Initial concern" in updated_report.concerns
