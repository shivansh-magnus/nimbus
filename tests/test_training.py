"""
Day-4 training and model battery unit tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from automl_agents.tools.training import run_model_battery


def test_run_model_battery_classification():
    # Construct a simple classification problem
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 1.1, 2.1, 0.9, 1.9, 1.2, 2.2, 0.8, 1.8],
        "x2": [0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7, 0.2, 0.8],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })
    
    # Run a 2-fold CV classifier battery
    results = run_model_battery(df, "target", problem_type="classification", cv=2)
    
    assert len(results) > 0
    for res in results:
        assert "model_id" in res
        assert "scores" in res
        assert "mean_scores" in res
        assert "std_scores" in res
        
        # Classification evaluation should produce accuracy and f1
        assert "accuracy" in res["mean_scores"]
        assert "f1" in res["mean_scores"]
        assert len(res["scores"]["accuracy"]) == 2


def test_run_model_battery_regression():
    # Construct a simple regression problem
    df = pd.DataFrame({
        "x1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "x2": [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0],
        "target": [1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5, 15.0]
    })
    
    # Run a 2-fold CV regressor battery
    results = run_model_battery(df, "target", problem_type="regression", cv=2)
    
    assert len(results) > 0
    for res in results:
        assert "model_id" in res
        assert "scores" in res
        assert "mean_scores" in res
        assert "std_scores" in res
        
        # Regression evaluation should produce mae, rmse, r2
        assert "mae" in res["mean_scores"]
        assert "rmse" in res["mean_scores"]
        assert "r2" in res["mean_scores"]
        assert len(res["scores"]["mae"]) == 2


def test_run_model_battery_raises_on_empty_features():
    df = pd.DataFrame({"target": [1, 2, 3]})
    with pytest.raises(ValueError, match="No features available to train the model battery"):
        run_model_battery(df, "target", problem_type="classification", cv=2)
