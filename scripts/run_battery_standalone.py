#!/usr/bin/env python
"""
scripts/run_battery_standalone.py

A standalone verification script that loads a dataset, applies Day 3 prep,
applies Day 4 feature selection, and runs the Day 4 model battery end-to-end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Insert project root into python path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from automl_agents.tools.profiler import load_csv, detect_problem_type
from automl_agents.tools.preprocessor import prep_dataframe
from automl_agents.tools.selection import run_selection
from automl_agents.tools.training import run_model_battery


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Day 4 standalone verification pipeline.")
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
        "--selection-method",
        type=str,
        default="rf_importance",
        choices=["variance", "correlation", "mutual_info", "rf_importance", "none"],
        help="Feature selection method to apply",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Number of cross-validation folds",
    )
    args = parser.parse_args()

    csv_path = ROOT / args.csv
    if not csv_path.exists():
        print(f"Error: Dataset {csv_path} not found.")
        return 1

    print(f"=== Standalone Pipeline Run ===")
    print(f"Dataset: {csv_path.name}")
    print(f"Target:  {args.target}")
    
    # 1. Load Data
    print("\n[Step 1] Loading CSV...")
    df = load_csv(csv_path)
    problem_type = detect_problem_type(df, args.target)
    print(f"Loaded shape: {df.shape} | Detected problem type: {problem_type}")

    # 2. Preprocess Data
    print("\n[Step 2] Preprocessing Data...")
    df_prepped = prep_dataframe(df, args.target)
    print(f"Preprocessed shape: {df_prepped.shape}")

    # 3. Feature Selection
    print(f"\n[Step 3] Running Feature Selection (method: {args.selection_method})...")
    selected_features = run_selection(
        df_prepped,
        args.target,
        method=args.selection_method,
        problem_type=problem_type,
        k=0.6,  # keep top 60% of features
    )
    print(f"Selected {len(selected_features)} features: {selected_features}")
    
    # Slice dataframe to selected features + target
    df_selected = df_prepped[selected_features + [args.target]].copy()

    # 4. Model Battery CV
    print(f"\n[Step 4] Running Model Battery CV (folds={args.cv})...")
    results = run_model_battery(df_selected, args.target, problem_type=problem_type, cv=args.cv)

    print("\n=== Results Summary ===")
    for res in results:
        print(f"Model: {res['model_id']}")
        for metric, val in res['mean_scores'].items():
            std_val = res['std_scores'][metric]
            print(f"  {metric}: {val:.4f} (std: {std_val:.4f})")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
