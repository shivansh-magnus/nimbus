"""
Integration test for the Day-5 AutoML pipeline graph.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from automl_agents.graph import graph


def test_pipeline_integration():
    # Use synthetic ground truth dataset
    csv_path = Path("data/raw/synthetic_ground_truth.csv")
    assert csv_path.exists(), "Synthetic dataset must be generated first."

    # Setup initial state
    initial_state = {
        "dataset_path": str(csv_path),
        "target_column": "churn",
        "eda_report": None,
        "cleaned_data_path": None,
        "prep_plan": None,
        "selected_features": [],
        "selection_rationale": "",
        "model_results": [],
        "best_model_id": None,
        "report_path": None,
        "stage_log": [],
        "retry_count": {},
        "token_usage": [],
    }

    # Context configuration
    context = {
        "run_id": "test_integration_run",
        "llm_provider": "gemini",
        "model_name": "gemini-3.1-flash-lite",
        "max_retries": 2,
        "token_budget": None,
    }

    # Invoke graph
    final_state = graph.invoke(initial_state, context=context)
    import pprint
    print("\n\n" + "="*60)
    print("FINAL STATE RESULTS (EXCLUDING DETAILS):")
    print("="*60)
    pprint.pprint({k: v for k, v in final_state.items() if k not in ["eda_report", "prep_plan", "model_results"]})
    print("="*60 + "\n\n")


    # Assertions on state outputs
    assert final_state["eda_report"] is not None
    assert final_state["eda_report"].problem_type == "classification"
    assert final_state["eda_report"].n_rows > 0

    assert final_state["cleaned_data_path"] is not None
    assert Path(final_state["cleaned_data_path"]).exists()

    assert final_state["prep_plan"] is not None

    assert len(final_state["selected_features"]) > 0
    assert final_state["selection_rationale"] != ""

    assert len(final_state["model_results"]) > 0
    assert final_state["best_model_id"] is not None

    assert final_state["report_path"] is not None
    assert Path(final_state["report_path"]).exists()

    # Stage log checks
    assert len(final_state["stage_log"]) == 5
    stages = [entry["stage"] for entry in final_state["stage_log"]]
    assert stages == ["profiler", "data_prep", "selector", "trainer", "reporter"]
    
    for entry in final_state["stage_log"]:
        assert entry["status"] == "ok"
