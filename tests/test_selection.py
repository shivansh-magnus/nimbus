"""
Day-4 feature selection unit tests.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from automl_agents.tools.profiler import load_csv
from automl_agents.tools.selection import (
    variance_threshold_pruning,
    correlation_pruning,
    mutual_info_pruning,
    rf_importance_pruning,
    run_selection,
)


def test_variance_threshold_pruning():
    # Feature x1 has high variance, x2 has zero variance (constant)
    # Target column is ignored, object columns should survive.
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x2": [1.0, 1.0, 1.0, 1.0, 1.0],
        "cat": ["a", "b", "a", "b", "a"],
        "target": [0, 1, 0, 1, 0]
    })
    
    # 0.0 threshold should drop x2 but keep x1 and cat
    selected = variance_threshold_pruning(df, "target", threshold=0.0)
    assert "x1" in selected
    assert "cat" in selected
    assert "x2" not in selected


def test_correlation_pruning():
    # x1 and x2 are perfectly correlated.
    # Correlation pruning should drop x2 (the later one in dataframe sequence).
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x2": [2.0, 4.0, 6.0, 8.0, 10.0],
        "x3": [5.0, 1.0, 4.0, 2.0, 9.0],
        "target": [0, 1, 0, 1, 0]
    })
    
    selected = correlation_pruning(df, "target", threshold=0.95)
    assert "x1" in selected
    assert "x3" in selected
    assert "x2" not in selected


def test_mutual_info_pruning():
    # x1 has perfect signal with target. x2 is random noise.
    rng = np.random.default_rng(42)
    x1 = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    target = x1.astype(int)
    x2 = rng.normal(size=10)
    
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "target": target
    })
    
    # Select top 1 feature (k=1)
    selected = mutual_info_pruning(df, "target", problem_type="classification", k=1)
    assert selected == ["x1"]

    # Select top 50% (k=0.5 of 2 features = 1 feature)
    selected_frac = mutual_info_pruning(df, "target", problem_type="classification", k=0.5)
    assert selected_frac == ["x1"]


def test_rf_importance_pruning():
    rng = np.random.default_rng(42)
    x1 = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
    target = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    x2 = rng.normal(size=10)
    
    df = pd.DataFrame({
        "x1": x1,
        "x2": x2,
        "target": target
    })
    
    selected = rf_importance_pruning(df, "target", problem_type="classification", k=1)
    assert selected == ["x1"]


def test_run_selection_orchestration():
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "x2": [1.0, 1.0, 1.0, 1.0, 1.0],
        "target": [0, 1, 0, 1, 0]
    })
    
    # Run selection with variance method
    selected = run_selection(df, "target", method="variance", threshold=0.0)
    assert selected == ["x1"]

    # Run selection with 'none' method
    selected_none = run_selection(df, "target", method="none")
    assert "x1" in selected_none
    assert "x2" in selected_none
    
    # Check invalid method raises ValueError
    with pytest.raises(ValueError, match="Unknown feature selection method"):
        run_selection(df, "target", method="invalid_method")
