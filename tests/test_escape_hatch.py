"""
tests/test_escape_hatch.py

Tests for the custom transform sandboxed execution sandbox and agent evals.
"""

import pytest
import pandas as pd
import numpy as np
from automl_agents.tools.custom_transform import run_custom_transform_sandboxed
from automl_agents.tools.eval import run_agent_evals


def test_custom_transform_success():
    """Asserts that custom code successfully executes and modifies the DataFrame."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [10, 20, 30]})
    code = "df['C'] = df['A'] * 10 + df['B']"
    
    df_out = run_custom_transform_sandboxed(code, df)
    assert "C" in df_out.columns
    assert list(df_out["C"]) == [20, 40, 60]


def test_custom_transform_exposes_pandas_and_numpy():
    """Asserts that sandboxed custom code can use common dataframe helpers."""
    df = pd.DataFrame({
        "date": ["2024-01-15", "2025-06-01", "2026-12-31"],
        "value": [1, 2, 3],
    })
    code = (
        "df['date'] = pd.to_datetime(df['date']); "
        "df['year'] = df['date'].dt.year; "
        "df['bucket'] = np.where(df['value'] >= 2, 'high', 'low')"
    )

    df_out = run_custom_transform_sandboxed(code, df)

    assert list(df_out["year"]) == [2024, 2025, 2026]
    assert list(df_out["bucket"]) == ["low", "high", "high"]


def test_custom_transform_syntax_error():
    """Asserts that code with syntax errors raises a RuntimeError."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    code = "df['B'] = df['A'] +++ (invalid syntax"
    
    with pytest.raises(RuntimeError) as exc_info:
        run_custom_transform_sandboxed(code, df)
    
    assert "SyntaxError" in str(exc_info.value) or "Custom code execution error" in str(exc_info.value)


def test_custom_transform_timeout():
    """Asserts that long-running/looping code triggers timeout execution safety."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    # Infinitely loop
    code = "import time\nwhile True:\n    time.sleep(0.1)"
    
    with pytest.raises(RuntimeError) as exc_info:
        run_custom_transform_sandboxed(code, df, timeout_sec=2)
        
    assert "timed out" in str(exc_info.value)


def test_custom_transform_network_isolation():
    """Asserts that sandbox code fails when trying to import blocked libraries like socket."""
    df = pd.DataFrame({"A": [1, 2, 3]})
    code = "import socket\nsocket.socket()"
    
    with pytest.raises(RuntimeError) as exc_info:
        run_custom_transform_sandboxed(code, df)
        
    assert "TypeError" in str(exc_info.value) or "NoneType" in str(exc_info.value) or "ModuleNotFoundError" in str(exc_info.value)


def test_agent_evals_calculates_correctly():
    """Asserts that run_agent_evals outputs correct metrics for a mock run state."""
    # State with all checks passing
    state_pass = {
        "prep_plan": {
            "drop_cols": ["leaky_churn_copy", "some_id"],
            "impute": {"annual_income": "median", "tenure_months": "mean"}
        },
        "eda_report": {
            "problem_type": "classification"
        }
    }
    res_pass = run_agent_evals(state_pass)
    assert res_pass["pass_rate"] == 100.0
    assert res_pass["results"]["leakage_detection"]["pass"] is True
    assert res_pass["results"]["imputation_validity"]["pass"] is True
    assert res_pass["results"]["problem_type_inference"]["pass"] is True

    # State with partial failure
    state_fail = {
        "prep_plan": {
            "drop_cols": ["some_id"],  # missed leaky_churn_copy
            "impute": {}
        },
        "eda_report": {
            "problem_type": "regression"  # wrong type
        }
    }
    res_fail = run_agent_evals(state_fail)
    assert res_fail["pass_rate"] == 0.0
    assert res_fail["results"]["leakage_detection"]["pass"] is False
