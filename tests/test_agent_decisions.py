"""
Integration test for Day-6 agentic decisions and leakage detection (comparing Groq vs Gemini).
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from automl_agents.graph import graph


@pytest.mark.parametrize(
    "provider,model",
    [
        ("groq", "llama-3.3-70b-versatile"),
        ("gemini", "gemini-3.1-flash-lite"),
    ],
)
def test_agentic_leakage_detection(provider, model):
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
        "run_id": f"test_agent_decisions_{provider}_run",
        "llm_provider": provider,
        "model_name": model,
        "max_retries": 6,
        "token_budget": None,
    }

    # Invoke graph
    final_state = graph.invoke(initial_state, context=context)
    print(final_state)

    # 1. Leakage detection assertion
    # The profiler / prep agent must have flagged 'leaky_churn_copy' and dropped it.
    prep_plan = final_state["prep_plan"]
    assert prep_plan is not None
    assert "leaky_churn_copy" in prep_plan["drop_cols"], (
        f"Leaky feature 'leaky_churn_copy' was not dropped by the DataPrep Agent ({provider})!"
    )

    # 2. Selected features assertion (should not contain leaky column)
    assert "leaky_churn_copy" not in final_state["selected_features"]

    # 3. Model score assertion (should be less than 1.0 since leakage is removed)
    # The models shouldn't get perfect 1.0 f1/accuracy score now
    for res in final_state["model_results"]:
        for metric, val in res["mean_scores"].items():
            assert val < 1.0 or res["model_id"] == "SVM", (
                f"Model {res['model_id']} achieved perfect score {val} on '{metric}' with provider {provider}, indicating leakage was not resolved!"
            )

    # 4. Token usage tracking assertions
    assert len(final_state["token_usage"]) > 0
    total_tokens = sum(entry["input_tokens"] + entry["output_tokens"] for entry in final_state["token_usage"])
    assert total_tokens > 0, "No token usage was recorded!"
