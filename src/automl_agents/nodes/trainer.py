"""
Day-6 LangGraph node for agentic model battery training using LLM metric selection.
"""

from __future__ import annotations

import logging
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime

from automl_agents.schemas import PipelineState, RunConfig, StageLogEntry
from automl_agents.tools.preprocessor import load_parquet_snapshot
from automl_agents.tools.training import run_model_battery
from automl_agents.llm_client import get_llm
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

    if not cleaned_path:
        raise ValueError("Missing 'cleaned_data_path' in PipelineState.")
    if not eda_report:
        raise ValueError("Missing 'eda_report' in PipelineState.")
    if not selected_features:
        raise ValueError("No features selected for training.")

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
        response = structured_llm.invoke([
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

        # Step 4: Identify best model based on the selected metric
        # Larger is better for accuracy, f1, r2. Smaller is better for rmse, mae.
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

        # Fallback if preferred metric not found
        if best_model_id is None:
            best_model_id = results[0]["model_id"]
            best_score = results[0].get("mean_scores", {}).get(chosen_metric, 0.0)

        logger.info(f"Best model: {best_model_id} with validation {chosen_metric} = {best_score:.4f}")

        # Step 5: Record token usage
        token_entry = record_token_usage("trainer", provider, model or "default", raw_msg)

        log_entry: StageLogEntry = {
            "stage": "trainer",
            "status": "ok",
            "message": f"Trained CV model battery. Best model: {best_model_id} (using LLM-chosen metric '{chosen_metric}'). Rationale: {selection.rationale}",
        }

        return {
            "model_results": results,
            "best_model_id": best_model_id,
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
