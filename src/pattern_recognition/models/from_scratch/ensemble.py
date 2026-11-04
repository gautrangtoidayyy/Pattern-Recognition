"""Reusable model and experiment helpers for the ensemble notebook."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from pattern_recognition.config import SEED


def make_ensemble_models(seed: int = SEED) -> dict[str, BaseEstimator]:
    """Create the AdaBoost models compared in the ensemble notebook."""
    return {
        "AdaBoost (stump)": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=200,
            learning_rate=0.5,
            random_state=seed,
        ),
        "AdaBoost (depth=3)": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=200,
            learning_rate=0.5,
            random_state=seed,
        ),
        "AdaBoost (LogReg)": AdaBoostClassifier(
            estimator=LogisticRegression(
                max_iter=500,
                solver="lbfgs",
                random_state=seed,
            ),
            n_estimators=50,
            learning_rate=0.5,
            random_state=seed,
        ),
        "AdaBoost (GaussianNB)": AdaBoostClassifier(
            estimator=GaussianNB(),
            n_estimators=50,
            learning_rate=0.5,
            random_state=seed,
        ),
    }


def benchmark(
    models: Mapping[str, BaseEstimator],
    X_tr: Any,
    X_eval: Any,
    y_tr: Any,
    y_eval: Any,
    cv: int | None = 5,
) -> tuple[pd.DataFrame, dict[str, BaseEstimator]]:
    """Fit each model on training data and report metrics on a held-out split."""
    rows = []
    fitted_models = {}

    for name, model in models.items():
        fitted = clone(model)

        start = time.perf_counter()
        fitted.fit(X_tr, y_tr)
        train_seconds = time.perf_counter() - start

        start = time.perf_counter()
        predictions = fitted.predict(X_eval)
        predict_seconds = time.perf_counter() - start

        row = {
            "Mô hình": name,
            "Accuracy": accuracy_score(y_eval, predictions),
            "Macro F1": f1_score(y_eval, predictions, average="macro"),
            "Precision": precision_score(
                y_eval, predictions, average="macro", zero_division=0
            ),
            "Recall": recall_score(
                y_eval, predictions, average="macro", zero_division=0
            ),
            "Train (s)": train_seconds,
            "Predict (s)": predict_seconds,
        }

        if cv is not None:
            scores = cross_val_score(clone(model), X_tr, y_tr, cv=cv)
            row["CV mean"] = scores.mean()
            row["CV std"] = scores.std()

        rows.append(row)
        fitted_models[name] = fitted

    results = pd.DataFrame(rows).set_index("Mô hình")
    sort_by = "CV mean" if cv is not None else "Accuracy"
    return results.sort_values(sort_by, ascending=False).round(4), fitted_models


def evaluate_fitted(
    fitted_models: Mapping[str, BaseEstimator],
    X: Any,
    y: Any,
) -> pd.DataFrame:
    """Evaluate already-fitted models on a split without fitting them again."""
    rows = []

    for name, model in fitted_models.items():
        start = time.perf_counter()
        predictions = model.predict(X)
        predict_seconds = time.perf_counter() - start

        rows.append(
            {
                "Mô hình": name,
                "Accuracy": accuracy_score(y, predictions),
                "Macro F1": f1_score(y, predictions, average="macro"),
                "Precision": precision_score(
                    y, predictions, average="macro", zero_division=0
                ),
                "Recall": recall_score(
                    y, predictions, average="macro", zero_division=0
                ),
                "Predict (s)": predict_seconds,
            }
        )

    results = pd.DataFrame(rows).set_index("Mô hình")
    return results.sort_values("Accuracy", ascending=False).round(4)


def sweep_n_estimators(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    values: Sequence[int] = (1, 5, 10, 25, 50, 100, 200),
    seed: int = SEED,
) -> tuple[list[int], dict[str, list[float]], float]:
    """Measure validation accuracy against the number of AdaBoost estimators."""
    configs = {
        "AdaBoost (stump)": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            learning_rate=0.5,
            random_state=seed,
        ),
        "AdaBoost (depth=3)": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            learning_rate=0.5,
            random_state=seed,
        ),
        "AdaBoost (LogReg)": AdaBoostClassifier(
            estimator=LogisticRegression(
                max_iter=500, solver="lbfgs", random_state=seed
            ),
            learning_rate=0.5,
            random_state=seed,
        ),
        "AdaBoost (GaussianNB)": AdaBoostClassifier(
            estimator=GaussianNB(),
            learning_rate=0.5,
            random_state=seed,
        ),
    }

    estimator_counts = list(values)
    scores_by_model = {name: [] for name in configs}

    baseline = DecisionTreeClassifier(max_depth=1, random_state=seed)
    baseline.fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_val, baseline.predict(X_val))

    for name, base_model in configs.items():
        for count in estimator_counts:
            model = clone(base_model).set_params(n_estimators=count)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            scores_by_model[name].append(accuracy_score(y_val, predictions))

    return estimator_counts, scores_by_model, baseline_accuracy


def sweep_learning_rate(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    values: Sequence[float] = (0.01, 0.05, 0.1, 0.3, 0.5, 1.0),
    seed: int = SEED,
) -> tuple[list[float], dict[str, list[float]]]:
    """Measure validation accuracy against AdaBoost's learning rate."""
    configs = {
        "AdaBoost (stump)": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1),
            n_estimators=100,
            random_state=seed,
        ),
        "AdaBoost (depth=3)": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=100,
            random_state=seed,
        ),
        "AdaBoost (LogReg)": AdaBoostClassifier(
            estimator=LogisticRegression(
                max_iter=500, solver="lbfgs", random_state=seed
            ),
            n_estimators=50,
            random_state=seed,
        ),
        "AdaBoost (GaussianNB)": AdaBoostClassifier(
            estimator=GaussianNB(),
            n_estimators=50,
            random_state=seed,
        ),
    }

    learning_rates = list(values)
    scores_by_model = {name: [] for name in configs}

    baseline = DecisionTreeClassifier(max_depth=1, random_state=seed)
    baseline.fit(X_train, y_train)
    baseline_accuracy = accuracy_score(y_val, baseline.predict(X_val))

    for name, base_model in configs.items():
        for learning_rate in learning_rates:
            model = clone(base_model).set_params(learning_rate=learning_rate)
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            scores_by_model[name].append(accuracy_score(y_val, predictions))

    return learning_rates, scores_by_model


def fit_random_forest_importances(
    X: Any,
    y: Any,
    n_estimators: int = 100,
    seed: int = SEED,
) -> tuple[RandomForestClassifier, np.ndarray]:
    """Fit the notebook's Random Forest and return its pixel importances."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model, model.feature_importances_


def most_confused_pair(
    y_true: Any,
    y_pred: Any,
    labels: Sequence[Any],
) -> tuple[Any, Any, int, int, int]:
    """Return the most-confused unordered class pair and both directions."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    off_diagonal = matrix.copy()
    np.fill_diagonal(off_diagonal, 0)
    pair_counts = off_diagonal + off_diagonal.T
    np.fill_diagonal(pair_counts, 0)

    i, j = np.unravel_index(np.argmax(pair_counts), pair_counts.shape)
    return (
        labels[i],
        labels[j],
        int(pair_counts[i, j]),
        int(off_diagonal[i, j]),
        int(off_diagonal[j, i]),
    )


def build_metrics_dataframe(
    results_by_split: Mapping[str, pd.DataFrame],
    sample_counts: Mapping[str, int],
    seed: int,
    model_params: Mapping[str, str],
) -> pd.DataFrame:
    """Build the long-form metrics table used by the ensemble notebook."""
    test_results = results_by_split["test"]
    rows = []

    for model_name in test_results.index:
        train_seconds = float(test_results.loc[model_name, "Train (s)"])

        for split_name, results in results_by_split.items():
            row = results.loc[model_name]
            rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "accuracy": round(float(row["Accuracy"]), 5),
                    "precision_macro": round(float(row["Precision"]), 5),
                    "recall_macro": round(float(row["Recall"]), 5),
                    "f1_macro": round(float(row["Macro F1"]), 5),
                    "train_seconds": round(train_seconds, 4),
                    "predict_seconds": round(float(row["Predict (s)"]), 4),
                    "n_samples": int(sample_counts[split_name]),
                    "seed": int(seed),
                    "params": model_params.get(model_name, ""),
                }
            )

    return pd.DataFrame(rows)
