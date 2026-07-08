# Day 8 — Next Steps (Tuning, Tracking, Resilience)

Day 8 focuses on taking our baseline models and enhancing them with hyperparameter tuning, logging experiment tracking metadata, and strengthening API reliability when querying LLM models under rate limits.

---

## 1. Core Concepts Explained

### Hyperparameter Tuning via Optuna
Optuna is a next-generation hyperparameter optimization framework. Instead of a grid or random search, Optuna uses modern Bayesian optimization algorithms (primarily Tree-structured Parzen Estimators - TPE) to intelligently sample hyperparameter spaces based on historical trials.
- **Dynamic Sampling**: We focus on tuning the top-2 model candidates from our initial CV battery (e.g. LightGBM, XGBoost, or RandomForest).
- **Leakage Prevention**: Tuning is performed strictly inside the training split folds during cross-validation to prevent information leakage to validation metrics.

### Experiment Tracking with MLflow
MLflow provides a unified tracking system for logging parameters, metrics, model snapshots, and run artifacts.
- **Local File Store**: Runs are stored locally (using `./mlruns` or configured database URIs).
- **Run Metadata**: We log the pipeline config, dataset features, selected preprocessing strategies, model parameters, CV fold scores, execution timings, and token consumption statistics.

### API Rate Limit Handling via Tenacity
External LLM APIs (Gemini, Groq) enforce rate limits (RPM and TPM thresholds). A high-volume agent pipeline can trigger 429 errors easily.
- **Tenacity Decorator**: Decorating API queries with retry logic using exponential backoff with jitter prevents agent crashes during transient failures.

---

## 2. "Before vs. After" Code Architectures

### Hyperparameter Optimization Integration

#### Before: Model Battery with Fixed Hyperparameters
```python
# Fixed parameters are used for training:
models = {
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100)
}
for name, model in models.items():
    model.fit(X_train, y_train)
```

#### After: Target Tuning of Top Candidates via Optuna
```python
import optuna

def objective(trial, X, y, model_name, problem_type):
    # 1. Define hyperparameter search spaces based on model name
    if model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        }
        model = LGBMClassifier(**params, random_state=42, verbose=-1)
    # 2. Compute cross-validation score
    scores = cross_val_score(model, X, y, cv=3, scoring="f1")
    return float(scores.mean())

# Identify top-2 candidates and tune
for top_model in top_2_candidates:
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X, y, top_model), n_trials=10)
    best_params = study.best_params
```

---

### MLflow Logging Integration

#### Before: Ad-hoc State Logging
```python
# No unified registry for comparing parameters/metrics:
results = run_model_battery(...)
print(f"Best score: {results[0]['mean_score']}")
```

#### After: MLflow Run Context Tracking
```python
import mlflow

mlflow.set_experiment("AutomL-Pipeline")
with mlflow.start_run(run_name=run_id):
    # 1. Log Run Configurations
    mlflow.log_params({
        "provider": provider,
        "model_name": model_name,
        "features_count": len(selected_features)
    })
    # 2. Log Preprocessing Plan Choices
    mlflow.log_dict(prep_plan, "preprocessor_config.json")
    # 3. Log Performance Metrics
    for res in results:
        mlflow.log_metric(f"cv_{res['model_id']}_f1_mean", res["mean_scores"]["f1"])
```

---

## 3. Interview Prep Q&A

### Q1: Why tune hyperparameters inside cross-validation folds instead of on the full training split?
**A**: Tuning hyperparameters on the full training split and assessing parameters using validation scores causes **hyperparameter leakage** (or validation leakage). The hyperparameters themselves adapt to fit the validation data, resulting in overly optimistic validation metrics. Performing the tuning loop inside the cross-validation folds ensures that hyperparameter optimization is validated on clean out-of-fold partitions, producing an unbiased generalisation score.

### Q2: How does MLflow help in multi-agent orchestration?
**A**: When multiple agents operate sequentially or concurrently (e.g. data preprocessing choice, feature selection strategy, scoring metric selection), tracking which agent made which decision and how it affected final scores is extremely complex. MLflow isolates runs, log decisions (as parameters/artifacts), and results (as metrics). This makes it easy to audit the decision-making process of LLMs and track down exactly which agent choices correlate with model improvements or validation failures.

### Q3: How do we handle API rate limits (429 errors) safely in production pipelines?
**A**:
1. **Exponential Backoff**: We wrap the API invocation in a retry loop using the `tenacity` library, multiplying wait times exponentially between retries.
2. **Jitter**: We add random noise (jitter) to the wait interval to prevent concurrent failing queries from calling the API at the exact same millisecond (preventing thundering herd problems).
3. **Graceful Fallbacks**: If the rate limit is exceeded beyond maximum retries, the code fails gracefully and falls back to deterministic defaults (e.g. standard prep settings) without crashing the runtime graph.

---

# Implementation Plan — Day 8 (Tuning, Tracking, Resilience)

This plan details the implementation of hyperparameter tuning via Optuna, local experiment tracking via MLflow, and API resilience via Tenacity retry decorators.

---

## User Review Required

> [!IMPORTANT]
> - **Optuna Tuning**: Tuning will be done on the top-2 model candidates from the baseline battery. It runs inside cross-validation training folds to ensure no data leakage. The results will be appended to the state's `model_results` list under the ID `{model_id} (Tuned)`.
> - **MLflow Integration**: Experiment logging is integrated directly within `reporter_node` (the final step of the graph), ensuring that any graph run (CLI, tests, or staging) automatically records metrics and artifacts.
> - **API Retry Decorator**: All agent nodes will wrap their structured LLM invoke calls with a `tenacity`-based exponential backoff retry wrapper to handle transient 429 rate limit exceptions.

---

## Open Questions

None.

---

## Proposed Changes

### 1. Tools Update

#### [MODIFY] [training.py](file:///c:/Users/dwive/OneDrive/Desktop/nimbus/src/automl_agents/tools/training.py)
- Import `optuna`.
- Implement `tune_model`:
  - Accepts the dataset, target column, model ID, problem type, optimization metric, CV folds, and number of trials.
  - Suppresses Optuna warning logs to keep the console clean.
  - Defines hyperparameter search spaces for: `LogisticRegression`, `LinearRegression`, `RandomForest`, `GradientBoosting`, `XGBoost`, `LightGBM`, `SVM`/`SVR`, and `KNN`.
  - Runs the Optuna optimization study (direction: `minimize` for `rmse`/`mae`, otherwise `maximize`) utilizing TPE sampling.
  - Evaluates models inside the cross-validation folds to prevent leakage.
  - Re-evaluates the best found parameters to capture the full score dictionary.
  - Returns a dictionary formatted like the baseline model results, with `model_id` set to `"{model_id} (Tuned)"`.

---

### 2. Node & Client Updates

#### [MODIFY] [llm_client.py](file:///c:/Users/dwive/OneDrive/Desktop/nimbus/src/automl_agents/llm_client.py)
- Import `retry`, `stop_after_attempt`, and `wait_exponential` from `tenacity`.
- Define and export a reusable `llm_retry_decorator`:
  ```python
  llm_retry_decorator = retry(
      stop=stop_after_attempt(5),
      wait=wait_exponential(multiplier=2, min=4, max=30),
      reraise=True,
  )
  ```

#### [MODIFY] Node LLM Call Wrappers
- Wrap the `.invoke` call of the structured LLM inside each node file with `llm_retry_decorator` to introduce rate-limit recovery:
  - **[MODIFY] [profiler.py](file:///c:/Users/dwive/OneDrive/Desktop/nimbus/src/automl_agents/nodes/profiler.py)**
  - **[MODIFY] [prep.py](file:///c:/Users/dwive/OneDrive/Desktop/nimbus/src/automl_agents/nodes/prep.py)**
  - **[MODIFY] [selector.py](file:///c:/Users/dwive/OneDrive/Desktop/nimbus/src/automl_agents/nodes/selector.py)**
  - **[MODIFY] [trainer.py](file:///c:/Users/dwive/OneDrive/Desktop/nimbus/src/automl_agents/nodes/trainer.py)**
  - **[MODIFY] [supervisor.py](file:///c:/Users/dwive/OneDrive/Desktop/nimbus/src/automl_agents/nodes/supervisor.py)**
  - **[MODIFY] [reporter.py](file:///c:/Users/dwive/OneDrive/Desktop/nimbus/src/automl_agents/nodes/reporter.py)**

---

### 3. Hyperparameter Tuning Integration

#### [MODIFY] [trainer.py](file:///c:/Users/dwive/OneDrive/Desktop/nimbus/src/automl_agents/nodes/trainer.py)
- Inside `trainer_node`:
  - After executing `run_model_battery` and finding `best_model_id`:
    - Rank the models based on the LLM-selected metric.
    - Identify the top-2 model candidates.
    - Loop over the top-2 candidates and invoke `tune_model` for each (using a default of 10 trials).
    - If a tuned model performs better than or equal to its baseline, or simply for logging, append/insert the tuned result dictionary into `model_results`.
    - Re-evaluate the best model selection considering the new tuned candidates.

---

### 4. MLflow Tracking Integration

#### [MODIFY] [reporter.py](file:///c:/Users/dwive/OneDrive/Desktop/nimbus/src/automl_agents/nodes/reporter.py)
- Import `mlflow`.
- Inside `reporter_node` (after the report markdown is written to disk):
  - Set the experiment to `"nimbus-automl"`.
  - Start an active MLflow run with name `run_id`.
  - Log parameters:
    - Target column, dataset path, problem type.
    - Preprocessing strategy choices (scaling strategy, drop list, imputation choices, etc. from `prep_plan`).
    - Best model ID.
    - Candidate models trained.
  - Log metrics:
    - Total LLM tokens consumed (prompt + completion tokens).
    - Number of preprocessing retry cycles.
    - Scores (e.g. CV mean and std) for all trained model battery and tuned candidates.
  - Log artifacts:
    - The generated `report.md`.
    - The stage-cleaned parquet snapshot from `cleaned_data_path`.

---

## Verification Plan

### Automated Tests
- Run the full pytest suite: `uv run pytest`.
- Write new unit and integration tests under `tests/test_mlflow_logging.py` verifying:
  1. **Optuna Tuning**: Asserts that `tune_model` successfully optimizes hyperparams and returns formatted metrics.
  2. **Tenacity Backoff**: Asserts that rate limits/transient exceptions on LLM client calls are caught and retried by the decorator.
  3. **MLflow Log Entry**: Asserts that invoking the pipeline compiles and creates an MLflow run local entry, logging the expected parameters, metrics, and artifact paths.

### Manual Verification
- Execute `uv run python scripts/run_pipeline.py` and inspect `./mlruns` directory or run `mlflow ui` locally to visually audit the tracked experiments and parameters.

