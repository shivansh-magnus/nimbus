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
