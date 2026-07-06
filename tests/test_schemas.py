"""Regression guards for Gemini-safe structured-output schemas."""

import json

from automl_agents.schemas import (
    ClassProportion,
    ColumnProfile,
    CorrelationPair,
    EDAReport,
)


def test_eda_report_json_schema_has_no_gemini_blockers():
    schema = json.dumps(EDAReport.model_json_schema())
    assert "prefixItems" not in schema
    assert "additionalProperties" not in schema


def test_eda_report_instantiation_with_list_of_records():
    report = EDAReport(
        n_rows=100,
        n_cols=2,
        columns=[
            ColumnProfile(column="a", dtype="int64", missing_fraction=0.0, cardinality=100),
            ColumnProfile(column="b", dtype="object", missing_fraction=0.1, cardinality=5),
        ],
        problem_type="classification",
        target_balance=[
            ClassProportion(label="0", proportion=0.6),
            ClassProportion(label="1", proportion=0.4),
        ],
        correlations_flagged=[
            CorrelationPair(col_a="a", col_b="b", corr=0.12),
        ],
        concerns=["example concern"],
    )
    assert report.n_rows == 100
    assert report.dtypes_by_column() == {"a": "int64", "b": "object"}
    assert report.missingness_by_column()["b"] == 0.1
    assert report.target_balance_by_label() == {"0": 0.6, "1": 0.4}


def test_eda_report_regression_problem_has_no_target_balance():
    report = EDAReport(
        n_rows=50,
        n_cols=1,
        columns=[
            ColumnProfile(column="y", dtype="float64", missing_fraction=0.0, cardinality=50),
        ],
        problem_type="regression",
        target_balance=None,
    )
    assert report.target_balance is None
    assert report.target_balance_by_label() is None
