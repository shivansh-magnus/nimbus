"""
Day-6 LangGraph node for agentic model battery training using LLM metric selection.
"""

from __future__ import annotations

import logging
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime

from automl_agents.schemas import PipelineState, RunConfig, StageLogEntry
from automl_agents.tools.preprocessor import load_parquet_snapshot
from automl_agents.tools.training import run_model_battery, tune_model
from automl_agents.llm_client import get_llm, llm_retry_decorator
from automl_agents.llm_util import record_token_usage

logger = logging.getLogger(__name__)


class TrainerMetricSelection(BaseModel):
    """Structured LLM output for selecting the optimization metric."""

    metric: str = Field(
        description="The chosen optimization metric. Must be one of: 'accuracy', 'f1', 'r2', 'rmse', 'mae'.",
    )
    rationale: str = Field(
        description="Detailed analytical explanation of why this metric is selected based on target balance or problem type.",
    )


def trainer_node(state: PipelineState, runtime: Runtime[RunConfig]) -> dict:
    """Agentic trainer node: queries the LLM for a metric, then runs the battery and ranks models."""
    logger.info("Starting model battery training...")
    cleaned_path = state["cleaned_data_path"]
    selected_features = state["selected_features"]
    target_column = state["target_column"]
    eda_report = state["eda_report"]

    if not cleaned_path or not eda_report or not selected_features:
        if not cleaned_path:
            missing = "cleaned_data_path (data_prep likely failed)"
        elif not eda_report:
            missing = "eda_report (profiler likely failed)"
        else:
            missing = "selected_features (selector likely failed or selected zero features)"
        log_entry: StageLogEntry = {
            "stage": "trainer",
            "status": "failed",
            "message": f"Training skipped: {missing}.",
        }
        return {"stage_log": [log_entry]}

    # Get runtime config for LLM
    context = runtime.context
    provider = context.get("llm_provider", "gemini")
    model = context.get("model_name")

    try:
        # Step 1: Query LLM to select optimization metric
        problem_type = eda_report.problem_type
        target_balance_summary = "N/A"
        if eda_report.target_balance:
            target_balance_summary = ", ".join(
                f"Class {item.label}: {item.proportion:.2%}"
                for item in eda_report.target_balance
            )

        system_prompt = (
            "You are a Lead Data Scientist selecting an optimization metric for evaluation.\n"
            "Choose from the following metrics:\n"
            "- For classification: 'accuracy' (for balanced datasets), 'f1' (highly recommended for imbalanced datasets).\n"
            "- For regression: 'r2' (maximize), 'rmse' (minimize), 'mae' (minimize).\n"
            "Return only the metric and a concise rationale."
        )
        user_prompt = (
            f"Problem Type: {problem_type}\n"
            f"Target Column: {target_column}\n"
            f"Class Balance/Proportions: {target_balance_summary}"
        )

        llm = get_llm(provider=provider, model=model)
        structured_llm = llm.with_structured_output(TrainerMetricSelection, include_raw=True)

        logger.info(f"Querying Trainer Agent using provider={provider}, model={model}...")
        response = llm_retry_decorator(structured_llm.invoke)([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        selection = response["parsed"]
        raw_msg = response["raw"]

        # Validate metric choice
        chosen_metric = selection.metric.lower().strip()
        allowed_metrics = ["accuracy", "f1", "r2", "rmse", "mae"]
        if chosen_metric not in allowed_metrics:
            logger.warning(f"LLM returned invalid metric '{chosen_metric}'; defaulting to 'f1' or 'r2'.")
            chosen_metric = "f1" if problem_type == "classification" else "r2"

        logger.info(f"Trainer Agent selected metric: {chosen_metric}. Rationale: {selection.rationale}")

        # Step 2: Load and slice data
        df = load_parquet_snapshot(cleaned_path)
        keep_cols = selected_features + [target_column]
        df_sliced = df[keep_cols].copy()

        # Step 3: Run model battery
        cv_folds = 3
        results = run_model_battery(
            df_sliced,
            target_column,
            problem_type=problem_type,
            cv=cv_folds,
        )

        if not results:
            raise ValueError("Model battery returned no results.")

        # Step 4.5: Validation Checks for Target Leakage
        validation_errors = []

        # A. Correlation Check: Flag features correlating with target >= 0.99
        for col in selected_features:
            if col in df.columns and col != target_column:
                try:
                    import pandas as pd
                    c_feat = pd.to_numeric(df[col], errors="coerce")
                    c_targ = pd.to_numeric(df[target_column], errors="coerce")
                    corr = float(c_feat.corr(c_targ))
                    if not pd.isna(corr) and abs(corr) >= 0.99:
                        validation_errors.append(
                            f"Feature '{col}' correlates with target '{target_column}' at {corr:.4f} -- potential target leakage."
                        )
                except Exception as corr_e:
                    logger.warning(f"Correlation check failed for column {col}: {corr_e}")

        # B. Score Check: Flag perfect cross-validation scores (excluding SVM)
        for res in results:
            model_id = res.get("model_id")
            if model_id and (model_id.startswith("SVM") or model_id.startswith("SVR")):
                continue
            mean_scores = res.get("mean_scores", {})
            for metric, val in mean_scores.items():
                if problem_type == "classification" and metric in ["accuracy", "f1"]:
                    if val >= 1.0:
                        validation_errors.append(
                            f"Model '{model_id}' achieved perfect score {val:.4f} on '{metric}' -- potential target leakage."
                        )
                elif problem_type == "regression" and metric == "r2":
                    if val >= 1.0:
                        validation_errors.append(
                            f"Model '{model_id}' achieved perfect score {val:.4f} on '{metric}' -- potential target leakage."
                        )

        if validation_errors:
            logger.warning(f"Trainer validation flagged target leakage: {validation_errors}")

        # Run Optuna tuning ONLY if no leakage was detected
        if not validation_errors:
            is_smaller_better = chosen_metric in ["rmse", "mae"]
            
            def get_model_metric_score(r):
                score = r.get("mean_scores", {}).get(chosen_metric)
                if score is None:
                    return float("inf") if is_smaller_better else float("-inf")
                return score

            sorted_results = sorted(
                results,
                key=get_model_metric_score,
                reverse=not is_smaller_better,
            )
            
            top_candidates = []
            for r in sorted_results:
                m_id = r["model_id"]
                if m_id not in top_candidates:
                    top_candidates.append(m_id)
                if len(top_candidates) == 2:
                    break

            logger.info(f"Top 2 models selected for hyperparameter tuning: {top_candidates}")
            for tc_model_id in top_candidates:
                try:
                    logger.info(f"Tuning {tc_model_id} via Optuna...")
                    tuned_res = tune_model(
                        df_sliced,
                        target_column,
                        tc_model_id,
                        problem_type,
                        chosen_metric,
                        cv=cv_folds,
                        n_trials=10,
                    )
                    if tuned_res:
                        results.append(tuned_res)
                except Exception as tune_e:
                    logger.error(f"Failed to tune {tc_model_id}: {tune_e}", exc_info=True)

        # Step 4: Identify best model based on the selected metric (baseline + tuned candidates)
        best_model_id = None
        is_smaller_better = chosen_metric in ["rmse", "mae"]
        best_score = float("inf") if is_smaller_better else float("-inf")

        for res in results:
            mean_scores = res.get("mean_scores", {})
            score = mean_scores.get(chosen_metric)
            if score is not None:
                if is_smaller_better:
                    if score < best_score:
                        best_score = score
                        best_model_id = res["model_id"]
                else:
                    if score > best_score:
                        best_score = score
                        best_model_id = res["model_id"]

        if best_model_id is None:
            best_model_id = results[0]["model_id"]

        # Step 5: Record token usage
        token_entry = record_token_usage("trainer", provider, model or "default", raw_msg)

        log_entry: StageLogEntry = {
            "stage": "trainer",
            "status": "ok",
            "message": f"Trained CV model battery and tuned top candidates. Best model: {best_model_id} (using LLM-chosen metric '{chosen_metric}'). Rationale: {selection.rationale}",
        }

        return {
            "model_results": results,
            "best_model_id": best_model_id,
            "validation_errors": validation_errors if validation_errors else None,
            "stage_log": [log_entry],
            "token_usage": [token_entry],
        }

    except Exception as e:
        logger.error(f"Error during model training node: {e}", exc_info=True)
        log_entry: StageLogEntry = {
            "stage": "trainer",
            "status": "failed",
            "message": f"Training failed: {str(e)}",
        }
        return {
            "stage_log": [log_entry],
        }
