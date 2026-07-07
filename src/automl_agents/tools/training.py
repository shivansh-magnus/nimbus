"""
Day-4 deterministic model battery and training tools.

Supports stratified/k-fold CV across a battery of standard classification
and regression algorithms. Preprocessed inputs are assumed to be numeric.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, SVR


# Suppress warnings from models (like ConvergenceWarning in LogisticRegression)
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# Import gradient boosting packages safely
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    XGBClassifier, XGBRegressor = None, None

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    LGBMClassifier, LGBMRegressor = None, None


def _get_classification_models() -> dict:
    """Return dictionary of classification model constructors/instances."""
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42, n_estimators=100, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42, n_estimators=100),
        "SVM": CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
        "KNN": KNeighborsClassifier(),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=-1)
    if LGBMClassifier is not None:
        models["LightGBM"] = LGBMClassifier(random_state=42, verbose=-1, n_jobs=-1)
    return models


def _get_regression_models() -> dict:
    """Return dictionary of regression model constructors/instances."""
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(random_state=42, n_estimators=100, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=42, n_estimators=100),
        "SVR": SVR(),
        "KNN": KNeighborsRegressor(),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = XGBRegressor(random_state=42, n_jobs=-1)
    if LGBMRegressor is not None:
        models["LightGBM"] = LGBMRegressor(random_state=42, verbose=-1, n_jobs=-1)
    return models


def run_model_battery(
    df: pd.DataFrame,
    target: str,
    problem_type: Literal["classification", "regression"],
    cv: int = 5,
) -> list[dict]:
    """Run CV across the model battery; return score metrics for each model.

    Target column is dropped from features. Remaining features are assumed to
    be numeric. Any residual missing values (which shouldn't be present after
    preprocessor.py, but could exist) are temporarily imputed using SimpleImputer.
    """
    features = [c for c in df.columns if c != target]
    if not features:
        raise ValueError("No features available to train the model battery.")

    # Convert features to numeric, impute any NaN placeholders
    X = df[features].copy()
    X_numeric = X.select_dtypes(include=[np.number])
    if X_numeric.shape[1] < X.shape[1]:
        logger.warning("Non-numeric features detected in training battery; keeping numeric features only.")
        X = X_numeric

    y = df[target]

    # Ensure clean numeric data for models
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Initialize CV splitters
    if problem_type == "classification":
        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        models = _get_classification_models()
        # Determine average parameter for f1_score based on classes
        unique_classes = np.unique(y)
        f1_avg = "binary" if len(unique_classes) <= 2 else "macro"
    else:
        splitter = KFold(n_splits=cv, shuffle=True, random_state=42)
        models = _get_regression_models()

    results = []

    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        
        fold_scores = []
        try:
            for train_idx, val_idx in splitter.split(X_imputed, y):
                X_train, X_val = X_imputed.iloc[train_idx], X_imputed.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Train model
                # Make a clean clone/new instance of the estimator
                from sklearn.base import clone
                estimator = clone(model)
                estimator.fit(X_train, y_train)

                # Predict & Evaluate
                y_pred = estimator.predict(X_val)

                if problem_type == "classification":
                    acc = float(accuracy_score(y_val, y_pred))
                    f1 = float(f1_score(y_val, y_pred, average=f1_avg))
                    fold_scores.append({"accuracy": acc, "f1": f1})
                else:
                    mae = float(mean_absolute_error(y_val, y_pred))
                    mse = float(mean_squared_error(y_val, y_pred))
                    rmse = float(np.sqrt(mse))
                    r2 = float(r2_score(y_val, y_pred))
                    fold_scores.append({"mae": mae, "rmse": rmse, "r2": r2})

            # Aggregate scores across folds
            metrics = list(fold_scores[0].keys())
            model_metrics = {"scores": {}, "mean_scores": {}, "std_scores": {}}

            for m in metrics:
                vals = [f[m] for f in fold_scores]
                model_metrics["scores"][m] = vals
                model_metrics["mean_scores"][m] = float(np.mean(vals))
                model_metrics["std_scores"][m] = float(np.std(vals))

            results.append({
                "model_id": model_name,
                "scores": model_metrics["scores"],
                "mean_scores": model_metrics["mean_scores"],
                "std_scores": model_metrics["std_scores"],
            })

        except Exception as e:
            logger.error(f"Failed to run model battery for {model_name}: {e}", exc_info=True)
            # Do not crash the entire run, continue to the next model

    return results
