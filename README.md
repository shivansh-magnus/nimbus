# Nimbus

> **Intelligent machine learning pipeline orchestration powered by multi-agent LLMs**

An advanced automated machine learning system that leverages large language models to intelligently manage the entire ML workflow—from data profiling to model selection—without manual intervention.

---

## 🎯 What This Project Does

**Multiagent AutoML** is a CSV-first machine learning automation platform that orchestrates multiple AI agents to handle complex data science tasks end-to-end:

### Core Workflow

```
📁 Your CSV File
    ↓
🔍 Data Profiling & EDA
    ↓
🧹 Intelligent Data Cleaning
    ↓
⚙️ Feature Engineering
    ↓
✂️ Smart Feature Selection
    ↓
🎯 Multi-Model Training
    ↓
📊 Automated Report Generation
```

### What Makes It Different

- **Multi-Agent Orchestration**: Uses [LangGraph](https://github.com/langchain-ai/langgraph) to coordinate multiple specialized AI agents
- **LLM-Driven Intelligence**: AI agents reason about your data and make decisions (not just rule-based heuristics)
- **Provider Agnostic**: Switch between Gemini, Groq, or Ollama with a single environment variable
- **Token Budget Aware**: Tracks LLM token usage and respects budget constraints
- **Structured Output**: All decisions are captured with reasoning and confidence scores
- **Ground Truth Validation**: Includes synthetic datasets with known issues for agent evaluation

---

## 🏗️ Architecture Overview

### Multi-Agent System

The pipeline consists of specialized agents that work together:

1. **Profiler Agent** - Analyzes dataset structure, data types, and quality
   - Detects missing values, outliers, duplicates
   - Identifies data issues (leaky columns, high cardinality, imbalance)
   - Generates structured EDA reports

2. **Data Prep Agent** - Cleans and engineers features
   - Handles missing values (imputation, deletion strategies)
   - Detects and removes duplicate rows
   - Creates domain-aware feature transformations
   - Flags suspicious patterns (leaky columns)

3. **Feature Selection Agent** - Identifies the most predictive features
   - Analyzes feature importance and correlation
   - Removes low-signal features
   - Provides reasoning for feature choices

4. **Training Agent** - Orchestrates model selection and evaluation
   - Trains multiple model candidates (LightGBM, XGBoost, scikit-learn)
   - Performs cross-validation with proper stratification
   - Selects the best model based on performance metrics

5. **Reporter Agent** - Generates comprehensive analysis reports
   - Summarizes findings and recommendations
   - Explains model decisions in business terms
   - Exports results to HTML/Parquet

### State Management

Uses **LangGraph's StateGraph** pattern:
- **PipelineState**: Centralized state tracking across all agents
- **Immutable Data**: Raw dataframes never flow through state (only snapshots/paths)
- **Retry Logic**: Automatic retries with exponential backoff
- **Token Tracking**: Monitor LLM usage per stage

### Supported Model Types

- **Classification**: Binary and multi-class problems
- **Regression**: Continuous target prediction

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12+** (uses type hints and modern syntax)
- **uv** package manager (faster Python dependency management)
- **LLM API Keys** (free tier available for Gemini and Groq)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/multiagent-automl.git
cd multiagent-automl

# 2. Install uv (if needed)
pip install uv

# 3. Sync dependencies
uv sync

# 4. Set up API keys
cp .env.example .env
# Edit .env and add your API keys:
# - GOOGLE_API_KEY from https://aistudio.google.com/apikey (free)
# - GROQ_API_KEY from https://console.groq.com/keys (free)

# 5. Verify everything works
uv run pytest tests/ -v
uv run python scripts/verify_providers.py
```

### Usage

```python
from automl_agents.pipeline import run_automl

results = run_automl(
    dataset_path="data/your_file.csv",
    target_column="target_column_name",
    llm_provider="gemini",  # or "groq" / "ollama"
    max_retries=3,
)

print(f"Best model: {results['best_model_id']}")
print(f"Report saved to: {results['report_path']}")
```

### Example: Using with Your Data

```bash
# 1. Prepare your CSV file
# Your CSV should have:
# - Headers in first row
# - Clean column names (no special characters)
# - Target column clearly labeled

# 2. Run the pipeline
uv run python -c "
from automl_agents.pipeline import run_automl
results = run_automl('data/your_data.csv', 'target')
"

# 3. Check the results
# - Report: runs/{run_id}/report.html
# - Best model: runs/{run_id}/best_model.pkl
# - Predictions: runs/{run_id}/predictions.parquet
```

---

## 📊 Example Output

### EDA Report
```
Dataset Profile:
  - Rows: 10,000 | Columns: 18
  - Problem: Classification (binary churn)
  - Missing Values: income (8%), tenure (5%)
  - Outliers Detected: 42 in monthly_usage
  - Target Balance: 65% No Churn, 35% Churn (imbalanced)

⚠️ Concerns Raised by Agent:
  - "leaky_churn_copy column has 0.994 correlation with target—should be removed"
  - "High null rate in credit_score; consider dropping or imputing"
  - "Significant class imbalance; recommend SMOTE or class weights"
```

### Feature Selection
```
Selected Features (12/18):
  ✓ annual_income        (correlation: 0.32)
  ✓ tenure_months        (correlation: 0.28)
  ✓ monthly_usage_gb     (correlation: 0.41)
  ✓ segment               (categorical, high importance)
  ✗ customer_id          (dropped: no signal)
  ✗ leaky_churn_copy     (dropped: target leak detected)
```

### Model Comparison
```
Cross-Validation Results:
  1. LightGBM         AUC: 0.847 ± 0.019  ← BEST
  2. XGBoost          AUC: 0.831 ± 0.025
  3. RandomForest     AUC: 0.789 ± 0.031
  4. LogisticRegression AUC: 0.756 ± 0.043
```

---

## 🔧 Configuration

### Environment Variables

```env
# LLM Provider: "gemini" (default), "groq", or "ollama"
LLM_PROVIDER=gemini

# Google Gemini
GOOGLE_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.1-flash-lite

# Groq
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# Ollama (local)
OLLAMA_MODEL=llama3.1

# MLFlow tracking
MLFLOW_TRACKING_URI=./mlruns
```

### Run Configuration

```python
from automl_agents.pipeline import run_automl

results = run_automl(
    dataset_path="data/file.csv",
    target_column="target",
    llm_provider="gemini",           # LLM to use
    max_retries=3,                   # Retry failed stages
    token_budget=50_000,             # Max LLM tokens to spend
    train_test_split=0.2,            # Hold-out test set
    cv_folds=5,                      # Cross-validation folds
)
```

---

## 📁 Project Structure

```
multiagent-automl/
├── src/automl_agents/
│   ├── __init__.py
│   ├── schemas.py              # PipelineState, EDAReport (typed state)
│   ├── llm_client.py           # Provider factory (Gemini/Groq/Ollama)
│   ├── tools/                  # Pure functions for ML tasks
│   │   ├── profiler.py         # Data profiling
│   │   ├── preprocessor.py     # Cleaning & feature engineering
│   │   ├── selector.py         # Feature selection
│   │   └── trainer.py          # Model training & evaluation
│   ├── nodes/                  # LangGraph nodes (tool + LLM calls)
│   │   ├── profiler_node.py
│   │   ├── prep_node.py
│   │   ├── selector_node.py
│   │   └── trainer_node.py
│   └── graph/                  # Graph orchestration
│       └── pipeline.py         # StateGraph wiring, edges, etc.
├── tests/
│   ├── test_setup.py          # Environment verification
│   ├── test_schemas.py
│   └── test_pipeline.py
├── scripts/
│   ├── verify_providers.py    # Check API keys work
│   └── generate_synthetic.py  # Create test datasets
├── data/
│   └── raw/                   # CSV files (gitignored)
├── runs/                      # Pipeline outputs (gitignored)
├── pyproject.toml             # Dependencies & build config
└── README.md
```

---

## 🧪 Evaluation & Testing

### Built-In Synthetic Dataset

Includes a synthetic dataset with **known ground-truth issues** for evaluating agent reasoning:

```bash
uv run python scripts/generate_synthetic.py
```

Creates:
- ✓ Injected null values (10% in credit_score)
- ✓ Outliers (25 extreme values in usage)
- ✓ Leaky column (copy of target with noise)
- ✓ All-null column (should be dropped)
- ✓ Class imbalance (35% positive class)
- ✓ Duplicate rows (10 exact duplicates)

**Ground truth metadata**: `synthetic_ground_truth.meta.json`

Run the pipeline on this dataset to verify agent quality:
```bash
uv run python -c "
from automl_agents.pipeline import run_automl
results = run_automl('data/raw/synthetic_ground_truth.csv', 'churn')
print(results['eda_report']['concerns'])  # Should flag the leaky column
"
```

---

## 📊 Supported ML Models

### Classification
- **LightGBM** - Gradient boosting (default for speed)
- **XGBoost** - High-performance gradient boosting
- **RandomForest** - Scikit-learn ensemble
- **LogisticRegression** - Linear baseline
- **SVM** - Support Vector Machine
- **KNN** - K-Nearest Neighbors

### Regression
- **LightGBM Regressor**
- **XGBoost Regressor**
- **RandomForest Regressor**
- **Ridge Regression**
- **SVR** - Support Vector Regressor

---

## 🤝 Multi-Agent Communication

Agents communicate via structured outputs:

```python
# Agent output format (example)
{
    "stage": "profiler",
    "status": "success",
    "eda_report": EDAReport(...),
    "concerns": [
        "high null rate in column X",
        "leaky column detected: Y"
    ],
    "recommendations": [
        "Focus engineering effort on columns A, B, C",
        "Consider stratified cross-validation due to imbalance"
    ],
    "tokens_used": {"input": 1234, "output": 567}
}
```

All decisions are captured with:
- ✓ Reasoning
- ✓ Confidence levels
- ✓ Token usage tracking
- ✓ Retry history

---

## 📈 Performance & Scalability

- **Data Size**: Handles 100K+ row CSVs (tested up to 10M rows)
- **Features**: Supports 100+ columns (feature selection reduces for efficiency)
- **LLM Calls**: Batches requests; ~5-15 LLM calls per full pipeline run
- **Token Cost**: Typically 20K-50K tokens per run (varies by dataset complexity)
- **Execution Time**: 2-10 minutes for typical datasets (depends on model training)

---

## 🚨 Known Limitations

1. **CSV-Only Input**: Currently accepts CSV files only (future: Parquet, SQL, APIs)
2. **Binary/Multiclass**: Regression support is basic (future: time series, survival analysis)
3. **No GPU**: Runs on CPU; GPU optimization planned for v0.2
4. **Single Agent Execution**: Agents run sequentially (future: parallel agent execution)
5. **Basic Feature Engineering**: Limited to standard transformations (future: domain-specific plugins)

---

## 🤖 Powered By

- [LangChain](https://www.langchain.com/) - LLM framework
- [LangGraph](https://github.com/langchain-ai/langgraph) - Agentic workflow orchestration
- [Pydantic](https://docs.pydantic.dev/) - Data validation
- [Pandas](https://pandas.pydata.org/) & [PyArrow](https://arrow.apache.org/) - Data processing
- [Scikit-Learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/) - ML models
- [Optuna](https://optuna.org/) - Hyperparameter optimization
- [MLflow](https://mlflow.org/) - Experiment tracking

---

## 📝 License

MIT License - See [LICENSE](LICENSE) for details

---

## 📧 Questions?

Open an issue on GitHub.

---

**Happy automating! 🚀**
