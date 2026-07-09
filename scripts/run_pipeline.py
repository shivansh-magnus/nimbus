#!/usr/bin/env python
"""
scripts/run_pipeline.py

CLI entrypoint to run the AutoML StateGraph pipeline end-to-end.

Day-10: the core logic is extracted into run_pipeline() so the Typer CLI
(src/automl_agents/cli.py) and the MCP server (src/automl_agents/mcp_server.py)
can both import and call it directly.  The argparse-based main() below
continues to work identically to before -- no breaking changes.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from pathlib import Path
from typing import Literal

# Insert project root into python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from automl_agents.graph import graph  # noqa: E402
from automl_agents.schemas import RunConfig  # noqa: E402

# Set up clean console logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


def run_pipeline(
    csv_path: str | Path,
    target: str,
    provider: Literal["gemini", "groq", "ollama"] = "gemini",
    model_name: str = "gemini-3.1-flash-lite",
    max_retries: int = 2,
    token_budget: int | None = None,
) -> dict:
    """Run the full AutoML StateGraph pipeline on a single CSV.

    This is the parameterised core of the former main() -- callable from
    the CLI, MCP server, or any test that needs a full end-to-end run.

    Returns the final LangGraph state dict (all pipeline outputs).
    Raises RuntimeError if the dataset file does not exist.
    """
    import re

    csv_path = Path(csv_path).resolve()
    if not csv_path.exists():
        raise RuntimeError(f"Dataset not found: {csv_path}")

    # Find the previous run matching the standard run_YYYYMMDD_HHMMSS nomenclature
    previous_run = "None"
    runs_dir = Path("runs")
    if runs_dir.exists():
        pattern = re.compile(r"^run_\d{8}_\d{6}$")
        run_folders = [
            p.name
            for p in runs_dir.iterdir()
            if p.is_dir() and pattern.match(p.name)
        ]
        if run_folders:
            run_folders.sort()
            previous_run = run_folders[-1]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"

    print(f"=== Starting AutoML StateGraph Run ===")
    print(f"Run ID:        {run_id}")
    print(f"Previous Run:  {previous_run}")
    print(f"Dataset Path:  {csv_path}")
    print(f"Target Column: {target}")
    print("======================================\n")

    initial_state = {
        "dataset_path": str(csv_path),
        "target_column": target,
        "eda_report": None,
        "cleaned_data_path": None,
        "prep_plan": None,
        "selected_features": [],
        "selection_rationale": "",
        "model_results": [],
        "best_model_id": None,
        "model_path": None,
        "report_path": None,
        "stage_log": [],
        "retry_count": {},
        "token_usage": [],
        "validation_errors": None,
    }

    context: RunConfig = {
        "run_id": run_id,
        "llm_provider": provider,
        "model_name": model_name,
        "max_retries": max_retries,
        "token_budget": token_budget,
    }

    final_state = graph.invoke(initial_state, context=context)

    print("\n================ Run Complete ================")
    print(f"Cleaned Data Path: {final_state.get('cleaned_data_path')}")
    print(f"Selected Features: {final_state.get('selected_features')}")
    print(f"Best Model ID:     {final_state.get('best_model_id')}")
    print(f"Model Bundle:      {final_state.get('model_path')}")
    print(f"Report Path:       {final_state.get('report_path')}")
    print("==============================================\n")

    return final_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the AutoML pipeline graph.")
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
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        help="LLM provider: gemini | groq | ollama",
    )
    args = parser.parse_args()

    try:
        run_pipeline(
            csv_path=args.csv,
            target=args.target,
            provider=args.provider,
        )
        return 0
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nExecution failed: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
