#!/usr/bin/env python
"""
scripts/run_pipeline.py

CLI entrypoint to run the Day-5 AutoML StateGraph pipeline end-to-end.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path

# Insert project root into python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from automl_agents.graph import graph

# Set up clean console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Day-5 AutoML pipeline graph.")
    parser.add_argument(
        "--csv",
        type=str,
        default="data/raw/synthetic_ground_truth.csv",
        help="Path to raw CSV dataset",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="churn",
        help="Target column name",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        print(f"Error: Dataset {csv_path} not found.")
        return 1

    # Find the previous run matching the standard run_YYYYMMDD_HHMMSS nomenclature
    import re
    previous_run = "None"
    runs_dir = Path("runs")
    if runs_dir.exists():
        pattern = re.compile(r"^run_\d{8}_\d{6}$")
        run_folders = []
        for path in runs_dir.iterdir():
            if path.is_dir() and pattern.match(path.name):
                run_folders.append(path.name)
        if run_folders:
            run_folders.sort()
            previous_run = run_folders[-1]

    # Inferred run ID using timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"

    print(f"=== Starting AutoML StateGraph Run ===")
    print(f"Run ID:        {run_id}")
    print(f"Previous Run:  {previous_run}")
    print(f"Dataset Path:  {csv_path}")
    print(f"Target Column: {args.target}")
    print("======================================\n")

    # Initial state
    initial_state = {
        "dataset_path": str(csv_path),
        "target_column": args.target,
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

    # Context configuration (read-only)
    context = {
        "run_id": run_id,
        "llm_provider": "gemini",
        "model_name": "gemini-3.1-flash-lite",
        "max_retries": 2,
        "token_budget": None,
    }

    try:
        final_state = graph.invoke(initial_state, context=context)

        print("\n================ Run Complete ================")
        print(f"Cleaned Data Path: {final_state.get('cleaned_data_path')}")
        print(f"Selected Features: {final_state.get('selected_features')}")
        print(f"Best Model ID:     {final_state.get('best_model_id')}")
        print(f"Report Path:       {final_state.get('report_path')}")
        print("==============================================\n")
        return 0

    except Exception as e:
        print(f"\nExecution failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
