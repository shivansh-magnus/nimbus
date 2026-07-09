"""
scripts/stress_test.py

AutoML Pipeline Stress Tester. Runs the end-to-end StateGraph pipeline
against all local datasets and logs successes/failures.
"""

import os
import sys
import json
import time
import datetime
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from automl_agents.graph.pipeline import graph
from automl_agents.tools.eval import run_agent_evals


def validate_graph_completion(final_state):
    """Return an error message if a graph run ended without required outputs."""
    failed_stage = next(
        (
            entry
            for entry in final_state.get("stage_log", [])
            if entry.get("status") == "failed"
        ),
        None,
    )
    if failed_stage:
        stage = failed_stage.get("stage", "unknown")
        message = failed_stage.get("message", "No failure message recorded.")
        return f"Stage '{stage}' failed: {message}"

    if not final_state.get("best_model_id"):
        return "Graph completed without a best_model_id."

    if not final_state.get("model_results"):
        return "Graph completed without model_results."

    report_path = final_state.get("report_path")
    if not report_path:
        return "Graph completed without a report_path."
    if not Path(report_path).exists():
        return f"Graph reported missing report file: {report_path}"

    return None


def run_stress_test() -> dict:
    """Run the end-to-end StateGraph pipeline across all local datasets.

    Returns a summary dict with keys:
      - "results": list of per-dataset result dicts
      - "report_path": absolute path to the markdown summary file
      - "all_passed": bool, True iff every dataset succeeded

    Day-10: extracted from main() so the Typer CLI and MCP server can both
    call this function directly.
    """
    print("==================================================")
    print("          AutoML Pipeline Stress Tester           ")
    print("==================================================")

    # 1. Define datasets to test
    # Load manifest and add synthetic ground truth
    manifest_path = Path("data/raw/manifest.json")
    datasets = []
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            datasets = json.load(f)
    
    # Add synthetic ground truth if not present
    has_synthetic = any(d["id"] == "synthetic" for d in datasets)
    if not has_synthetic:
        datasets.append({
            "id": "synthetic",
            "path": "data/raw/synthetic_ground_truth.csv",
            "target_column": "churn",
            "problem_type": "classification"
        })

    # Create stress test run folder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stress_runs_dir = Path("runs") / f"stress_test_{timestamp}"
    stress_runs_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # 2. Iterate and execute graph on each dataset
    for ds in datasets:
        ds_id = ds["id"]
        csv_path = Path(ds["path"])
        target = ds["target_column"]
        
        print(f"\n---> Testing Dataset: '{ds_id}' ({csv_path})")
        if not csv_path.exists():
            print(f"     [ERROR] File not found: {csv_path}")
            results.append({
                "id": ds_id,
                "status": "Failed",
                "best_model": "N/A",
                "time_sec": 0,
                "error": f"Dataset file {csv_path} not found"
            })
            continue

        # Load and sample if the dataset is large to ensure test execution speed
        df_raw = pd.read_csv(csv_path)
        temp_csv_path = None
        
        if len(df_raw) > 2500:
            print(f"     Dataset has {len(df_raw)} rows. Sampling 2000 rows for fast testing...")
            df_sampled = df_raw.sample(n=2000, random_state=42)
            temp_csv_path = stress_runs_dir / f"temp_{ds_id}.csv"
            df_sampled.to_csv(temp_csv_path, index=False)
            run_csv_path = temp_csv_path
        else:
            run_csv_path = csv_path

        run_id = f"stress_{ds_id}_{timestamp}"
        
        # Build state
        initial_state = {
            "dataset_path": str(run_csv_path.resolve()),
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

        context = {
            "run_id": run_id,
            "llm_provider": "gemini",
            "model_name": "gemini-3.1-flash-lite",
            "max_retries": 2,
            "token_budget": None,
        }

        start_time = time.time()
        try:
            final_state = graph.invoke(initial_state, context=context)
            elapsed = time.time() - start_time

            completion_error = validate_graph_completion(final_state)
            if completion_error:
                print(f"     [FAILED] Completed in {elapsed:.1f}s but outputs are invalid: {completion_error}")
                results.append({
                    "id": ds_id,
                    "status": "Failed",
                    "best_model": "N/A",
                    "time_sec": elapsed,
                    "error": completion_error,
                    "eval": None
                })
                continue

            best_model = final_state["best_model_id"]
            
            # Run evaluation if synthetic
            eval_results = None
            if ds_id == "synthetic" or ds_id == "telco_churn":
                eval_results = run_agent_evals(final_state)
                if eval_results["pass_rate"] < 100.0:
                    error = f"Agent eval pass rate was {eval_results['pass_rate']:.1f}%."
                    print(f"     [FAILED] Completed in {elapsed:.1f}s but evals failed: {error}")
                    results.append({
                        "id": ds_id,
                        "status": "Failed",
                        "best_model": "N/A",
                        "time_sec": elapsed,
                        "error": error,
                        "eval": eval_results
                    })
                    continue

            print(f"     [SUCCESS] Completed in {elapsed:.1f}s. Best Model: {best_model}")
            
            results.append({
                "id": ds_id,
                "status": "Success",
                "best_model": best_model,
                "time_sec": elapsed,
                "error": None,
                "eval": eval_results
            })
        except Exception as err:
            elapsed = time.time() - start_time
            print(f"     [FAILED] Exception during graph execution: {err}")
            results.append({
                "id": ds_id,
                "status": "Failed",
                "best_model": "N/A",
                "time_sec": elapsed,
                "error": str(err),
                "eval": None
            })
        finally:
            # Clean up temp sampled file
            if temp_csv_path and temp_csv_path.exists():
                temp_csv_path.unlink()

    # 3. Print Console Report Table
    print("\n" + "=" * 60)
    print("                  STRESS TEST RESULTS SUMMARY             ")
    print("=" * 60)
    print(f"{'Dataset ID':<20} | {'Status':<10} | {'Best Model':<25} | {'Time (s)':<8}")
    print("-" * 60)
    for r in results:
        print(f"{r['id']:<20} | {r['status']:<10} | {r['best_model']:<25} | {r['time_sec']:<8.1f}")
    print("=" * 60)

    # 4. Generate Markdown Report
    md_content = [
        f"# Stress Test Run Report — {timestamp}\n",
        "This report logs the execution stability of the AutoML StateGraph pipeline across all local datasets.\n",
        "## Execution Summary\n",
        "| Dataset ID | Status | Best Model | Execution Time | Error |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ]
    for r in results:
        err_val = r["error"] if r["error"] else "-"
        md_content.append(f"| `{r['id']}` | **{r['status']}** | `{r['best_model']}` | {r['time_sec']:.1f}s | {err_val} |")

    # Appending synthetic evaluations details if present
    for r in results:
        if r.get("eval"):
            ev = r["eval"]
            md_content.append(f"\n## Agent Reasoning Evals ({r['id']} Dataset)")
            md_content.append(f"**Pass Rate**: {ev['pass_rate']:.1f}%\n")
            md_content.append("| Metric Check | Status | Details |")
            md_content.append("| :--- | :--- | :--- |")
            for metric, res in ev["results"].items():
                status_icon = "✅ PASS" if res["pass"] else "❌ FAIL"
                md_content.append(f"| {metric} | {status_icon} | {res['detail']} |")

    report_path_obj = stress_runs_dir / "stress_test_report.md"
    with open(report_path_obj, "w", encoding="utf-8") as f:
        f.write("\n".join(md_content))
    print(f"\nWritten detailed markdown report to: {report_path_obj.resolve()}\n")

    return {
        "results": results,
        "report_path": str(report_path_obj.resolve()),
        "all_passed": all(r["status"] == "Success" for r in results),
    }


def main():
    run_stress_test()


if __name__ == "__main__":
    main()
