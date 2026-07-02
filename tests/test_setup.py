"""Day-1 smoke tests: nothing agent-specific yet, just prove the environment
is sound before Day 2's tool-library work begins."""

import importlib

CORE_DEPS = [
    "langgraph",
    "langchain",
    "langchain_google_genai",
    "langchain_groq",
    "pandas",
    "pyarrow",
    "sklearn",
    "xgboost",
    "lightgbm",
    "optuna",
    "mlflow",
    "pydantic",
    "feature_engine",
]


def test_all_core_dependencies_importable():
    failures = []
    for module_name in CORE_DEPS:
        try:
            importlib.import_module(module_name)
        except ImportError as e:
            failures.append(f"{module_name}: {e}")
    assert not failures, f"Failed imports: {failures}"


def test_schemas_importable_and_instantiable():
    from automl_agents.schemas import EDAReport, PipelineState  # noqa: F401

    report = EDAReport(
        n_rows=100,
        n_cols=5,
        dtypes={"a": "int64"},
        missingness={"a": 0.0},
        cardinality={"a": 100},
        problem_type="classification",
    )
    assert report.n_rows == 100


def test_llm_client_raises_clear_error_without_key(monkeypatch):
    from automl_agents.llm_client import get_llm

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    try:
        get_llm(provider="gemini")
        raise AssertionError("expected RuntimeError for missing API key")
    except RuntimeError as e:
        assert "GOOGLE_API_KEY" in str(e)
