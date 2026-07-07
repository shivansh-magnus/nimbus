"""
Day-6 LangGraph node for generating execution report with LLM summary and token usage.
"""

from __future__ import annotations

import logging
from pathlib import Path
from pydantic import BaseModel, Field
from langgraph.runtime import Runtime

from automl_agents.schemas import PipelineState, RunConfig, StageLogEntry
from automl_agents.llm_client import get_llm
from automl_agents.llm_util import record_token_usage

logger = logging.getLogger(__name__)


class ReportExecutiveSummary(BaseModel):
    """Structured LLM output for the report's narrative review and recommendations."""

    executive_summary: str = Field(
        description="High-level narrative review summarizing the dataset characteristics, key preprocessing/leakage decisions, and trained model results.",
    )
    recommendations: list[str] = Field(
        description="Recommended next steps for production deployment or future iteration.",
    )


def reporter_node(state: PipelineState, runtime: Runtime[RunConfig]) -> dict:
    """Agentic reporter node: queries the LLM for a narrative summary, then generates report.md."""
    logger.info("Generating markdown report...")
    eda_report = state["eda_report"]
    selected_features = state["selected_features"]
    model_results = state["model_results"]
    best_model_id = state["best_model_id"]
    prep_plan = state["prep_plan"]
    stage_log = state["stage_log"]
    token_usage = list(state["token_usage"])

    # Get runtime config for LLM
    context = runtime.context
    provider = context.get("llm_provider", "gemini")
    model = context.get("model_name")
    run_id = context.get("run_id", "default_run")

    try:
        # Step 1: Query LLM for Executive Summary & Recommendations
        system_prompt = (
            "You are a Lead Data Scientist writing an executive run report for an AutoML pipeline.\n"
            "Analyze the results of the dataset profiling, the preprocessing cleaning plan, the selected features, "
            "and the candidate model validation scores.\n"
            "Highlight any major actions taken (e.g. dropping target leakage features) and describe the performance "
            "of the best model."
        )

        user_prompt = (
            f"Dataset Columns: {eda_report.n_cols if eda_report else 'N/A'}\n"
            f"Preprocessed Drop Columns: {prep_plan.get('drop_cols', []) if prep_plan else 'N/A'}\n"
            f"Selected Features: {selected_features}\n"
            f"Best Model ID: {best_model_id}\n"
            f"Model Scores:\n"
        )
        if model_results:
            for res in model_results:
                user_prompt += f"- {res['model_id']}: mean_scores={res['mean_scores']}\n"

        llm = get_llm(provider=provider, model=model)
        structured_llm = llm.with_structured_output(ReportExecutiveSummary, include_raw=True)

        logger.info(f"Querying Reporter Agent using provider={provider}, model={model}...")
        response = structured_llm.invoke([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])

        summary_data = response["parsed"]
        raw_msg = response["raw"]

        # Record reporter stage token usage
        reporter_token_entry = record_token_usage("reporter", provider, model or "default", raw_msg)
        token_usage.append(reporter_token_entry)

        # Step 2: Build Markdown Report
        report_content = []
        report_content.append(f"# AutoML Pipeline Run Report: `{run_id}`\n")

        # LLM Executive Summary
        report_content.append("## Executive Summary")
        report_content.append(summary_data.executive_summary)
        report_content.append("\n### Recommendations")
        for rec in summary_data.recommendations:
            report_content.append(f"- {rec}")
        report_content.append("\n")

        # 1. Dataset Overview
        if eda_report:
            report_content.append("## 1. Dataset Overview")
            report_content.append(f"- **Problem Type**: {eda_report.problem_type}")
            report_content.append(f"- **Rows**: {eda_report.n_rows}")
            report_content.append(f"- **Columns**: {eda_report.n_cols}")
            if eda_report.target_balance:
                report_content.append("\n### Class Proportions")
                for item in eda_report.target_balance:
                    report_content.append(f"  - `{item.label}`: {item.proportion:.2%}")
            report_content.append("\n")
        else:
            report_content.append("## 1. Dataset Overview\nNo dataset profiling information available.\n")

        # 2. Preprocessing & Cleaning Plan
        if prep_plan:
            report_content.append("## 2. Preprocessing & Cleaning Plan")
            report_content.append(f"- **Drop Columns**: {prep_plan.get('drop_cols', [])}")
            report_content.append(f"- **Datetime Columns**: {prep_plan.get('datetime_cols', [])}")
            report_content.append(f"- **Mixed Numeric Columns**: {prep_plan.get('mixed_numeric_cols', [])}")
            report_content.append(f"- **Scaling Strategy**: {prep_plan.get('scale_strategy', 'standard')}")
            report_content.append("\n")

        # 3. Feature Selection
        if selected_features:
            report_content.append("## 3. Feature Selection")
            report_content.append(f"- **Selected {len(selected_features)} features**: {', '.join(selected_features)}")
            report_content.append(f"- **Selection Rationale**: {state.get('selection_rationale', 'N/A')}")
            report_content.append("\n")

        # 4. Model Battery Results
        if model_results:
            report_content.append("## 4. Model Battery Results")
            report_content.append(f"### Best Model: **{best_model_id}**\n")
            report_content.append("| Model ID | Metric | Mean Validation Score | Std Dev |")
            report_content.append("| --- | --- | --- | --- |")
            for res in model_results:
                model_name = res["model_id"]
                for metric, val in res["mean_scores"].items():
                    std_val = res["std_scores"].get(metric, 0.0)
                    report_content.append(f"| {model_name} | {metric} | {val:.4f} | {std_val:.4f} |")
            report_content.append("\n")

        # 5. Token Usage Metrics
        report_content.append("## 5. Token Usage & Costs")
        report_content.append("| Stage | Provider | Model | Input Tokens | Output Tokens | Total Tokens |")
        report_content.append("| --- | --- | --- | --- | --- | --- |")
        total_in, total_out = 0, 0
        for token_entry in token_usage:
            stage_name = token_entry["stage"]
            prov = token_entry["provider"]
            m_name = token_entry["model"]
            in_t = token_entry["input_tokens"]
            out_t = token_entry["output_tokens"]
            tot = in_t + out_t
            total_in += in_t
            total_out += out_t
            report_content.append(f"| {stage_name} | {prov} | {m_name} | {in_t} | {out_t} | {tot} |")
        report_content.append(f"| **Total** | | | **{total_in}** | **{total_out}** | **{total_in + total_out}** |")
        report_content.append("\n")

        # 6. Execution Logs
        report_content.append("## 6. Execution Steps Log")
        report_content.append("| Stage | Status | Message |")
        report_content.append("| --- | --- | --- |")
        for entry in stage_log:
            report_content.append(f"| {entry['stage']} | {entry['status']} | {entry['message']} |")
        report_content.append(f"| reporter | ok | Report generated successfully. |")
        report_content.append("\n")

        # Write to file
        runs_dir = Path("runs") / run_id
        runs_dir.mkdir(parents=True, exist_ok=True)
        report_path = runs_dir / "report.md"

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_content))

        logger.info(f"Report written to {report_path}")

        log_entry: StageLogEntry = {
            "stage": "reporter",
            "status": "ok",
            "message": f"Report generated successfully and saved to {report_path}.",
        }

        return {
            "report_path": str(report_path.resolve()),
            "stage_log": [log_entry],
            "token_usage": [reporter_token_entry],
        }

    except Exception as e:
        logger.error(f"Error during report generation: {e}", exc_info=True)
        log_entry: StageLogEntry = {
            "stage": "reporter",
            "status": "failed",
            "message": f"Report generation failed: {str(e)}",
        }
        return {
            "stage_log": [log_entry],
        }
