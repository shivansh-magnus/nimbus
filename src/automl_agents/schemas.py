"""
Core state schemas for the pipeline graph.

Day-1 rule (see roadmap §2.4): the raw dataframe NEVER lives in this state.
Only paths to parquet snapshots and statistical summaries do. Agents reason
over EDAReport / small samples, never over full tabular data pushed through
the graph.
"""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field


class EDAReport(BaseModel):
    """Structured output of the Profiler node. Filled in on Day 6."""

    n_rows: int
    n_cols: int
    dtypes: dict[str, str]
    missingness: dict[str, float] = Field(
        description="column -> fraction missing, 0.0-1.0"
    )
    cardinality: dict[str, int] = Field(
        description="column -> number of unique values"
    )
    problem_type: Literal["classification", "regression"]
    target_balance: dict[str, float] | None = Field(
        default=None, description="class label -> proportion, classification only"
    )
    correlations_flagged: list[tuple[str, str, float]] = Field(
        default_factory=list,
        description="(col_a, col_b, corr) pairs above a suspicion threshold",
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="LLM-written narrative flags, e.g. 'col X looks like a leaked target'",
    )


class StageLogEntry(TypedDict):
    stage: str
    status: Literal["ok", "retried", "failed"]
    message: str


class TokenUsageEntry(TypedDict):
    stage: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int


class PipelineState(TypedDict):
    # --- input ---
    dataset_path: str  # path to the ORIGINAL uploaded csv, read-only
    target_column: str

    # --- profiler stage ---
    eda_report: EDAReport | None

    # --- data prep stage (cleaning + feature engineering, merged) ---
    cleaned_data_path: str | None  # parquet snapshot, not a dataframe
    prep_plan: dict | None  # column -> chosen strategy, LLM structured output

    # --- feature selection stage ---
    selected_features: list[str]
    selection_rationale: str

    # --- training stage ---
    model_results: list[dict]  # per-candidate CV scores
    best_model_id: str | None

    # --- reporting stage ---
    report_path: str | None

    # --- control / observability (use operator.add reducers: nodes append, never overwrite) ---
    stage_log: Annotated[list[StageLogEntry], operator.add]
    retry_count: dict[str, int]
    token_usage: Annotated[list[TokenUsageEntry], operator.add]


class RunConfig(TypedDict):
    """Read-only run configuration, injected via LangGraph's context_schema.
    Never put this inside PipelineState -- it doesn't change during a run."""

    run_id: str
    llm_provider: Literal["gemini", "groq", "ollama"]
    max_retries: int
    token_budget: int | None
