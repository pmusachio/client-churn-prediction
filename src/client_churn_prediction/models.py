"""
Training, evaluation, hyperparameter tuning, and batch prediction entrypoints.

Design follows the end-to-end ML workflow described in Aurélien Géron's
*Hands-On Machine Learning with Scikit-Learn and PyTorch* (2025), Ch. 2 & 6:
  - Stratified train/test split
  - ColumnTransformer preprocessing pipeline
  - Multi-model cross-validated comparison
  - RandomizedSearchCV for hyperparameter tuning
  - Business-oriented ranking metrics (precision@k, lift@k)
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import load_config, resolve_project_path
from .data import load_training_frame, read_csv
from .features import model_matrix, prepare_features


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _one_hot_encoder():
    from sklearn.preprocessing import OneHotEncoder

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _column_types(X):
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numeric = [c for c in X.columns if c not in categorical]
    return numeric, categorical


def _preprocessor(X):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    numeric, categorical = _column_types(X)
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", _one_hot_encoder()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def _build_pipeline(X, estimator):
    from sklearn.pipeline import Pipeline

    return Pipeline([("preprocess", _preprocessor(X)), ("model", estimator)])


# ---------------------------------------------------------------------------
# Classifier catalogue
# ---------------------------------------------------------------------------

def _get_classifier(name: str, config: dict):
    """Return an sklearn classifier by name with config-driven defaults."""
    modeling = config.get("modeling", {})
    random_state = int(modeling.get("random_state", 42))
    class_weight = modeling.get("class_weight", "balanced")

    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            max_iter=int(modeling.get("max_iter", 1000)),
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=int(modeling.get("n_estimators", 200)),
            max_depth=int(modeling.get("max_depth", 10)) if modeling.get("max_depth") else None,
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state,
        )
    if name == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=int(modeling.get("n_estimators", 200)),
            max_depth=int(modeling.get("max_depth", 4)),
            random_state=random_state,
        )
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(
            n_estimators=int(modeling.get("n_estimators", 200)),
            class_weight=class_weight,
            n_jobs=-1,
            random_state=random_state,
        )
    # default fallback
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, class_weight=class_weight, random_state=random_state)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def _classification_metrics(y_true, y_pred, y_proba, config: dict) -> dict:
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    labels = sorted(set(y_true.dropna().tolist()))
    average = "binary" if len(labels) == 2 else "macro"
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
    }
    if y_proba is not None and len(labels) == 2:
        positive = config.get("data", {}).get("positive_label", labels[-1])
        scores = y_proba[:, 1] if (hasattr(y_proba, "shape") and len(y_proba.shape) == 2) else y_proba
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
            metrics["average_precision"] = float(average_precision_score(y_true, scores))
        except ValueError:
            pass
        for k in config.get("evaluation", {}).get("top_k", []):
            k = min(int(k), len(scores))
            if k <= 0:
                continue
            order = np.argsort(scores)[::-1][:k]
            positive_mask = (y_true.reset_index(drop=True).iloc[order] == positive).astype(int)
            base_rate = float((y_true == positive).mean())
            precision_at_k = float(positive_mask.mean())
            metrics[f"precision_at_{k}"] = precision_at_k
            metrics[f"lift_at_{k}"] = round(precision_at_k / base_rate, 3) if base_rate else None
    return metrics


def _regression_metrics(y_true, y_pred) -> dict:
    import numpy as np
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denominator = np.where(y_true == 0, np.nan, y_true)
    rmspe = float(np.sqrt(np.nanmean(((y_true - y_pred) / denominator) ** 2)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "r2": float(r2_score(y_true, y_pred)),
        "rmspe": rmspe,
    }


# ---------------------------------------------------------------------------
# Cross-validation comparison (book Ch. 2 & 3 best practice)
# ---------------------------------------------------------------------------

def compare_models(config_path=None, models: list[str] | None = None) -> list[dict]:
    """
    Train and cross-validate multiple classifiers, returning a sorted comparison table.

    Follows Géron's recommendation to use StratifiedKFold cross-validation to
    compare models on the same folds before committing to one.
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate

    config = load_config(config_path)
    df = load_training_frame(config)
    X, y, _ = model_matrix(df, config, training=True)
    if y is None:
        raise ValueError("Target column not available in training data.")

    candidate_names = models or ["logistic_regression", "random_forest", "gradient_boosting", "extra_trees"]
    cv = StratifiedKFold(
        n_splits=int(config.get("modeling", {}).get("search", {}).get("cv_folds", 5)),
        shuffle=True,
        random_state=int(config.get("modeling", {}).get("random_state", 42)),
    )
    results = []
    for name in candidate_names:
        estimator = _get_classifier(name, config)
        pipeline = _build_pipeline(X, estimator)
        scores = cross_validate(
            pipeline, X, y,
            cv=cv,
            scoring=["roc_auc", "average_precision", "f1"],
            n_jobs=-1,
        )
        results.append({
            "model": name,
            "roc_auc_mean": round(float(scores["test_roc_auc"].mean()), 4),
            "roc_auc_std": round(float(scores["test_roc_auc"].std()), 4),
            "avg_precision_mean": round(float(scores["test_average_precision"].mean()), 4),
            "f1_mean": round(float(scores["test_f1"].mean()), 4),
        })
    results.sort(key=lambda r: r["roc_auc_mean"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Hyperparameter tuning (book Ch. 2: RandomizedSearchCV)
# ---------------------------------------------------------------------------

def tune_model(config_path=None) -> dict:
    """
    Run RandomizedSearchCV on the best model family (GradientBoosting by default).

    Géron recommends RandomizedSearchCV over GridSearchCV when the search space
    is large — it explores more combinations in the same compute budget.
    """
    from scipy.stats import randint, uniform
    from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

    config = load_config(config_path)
    modeling = config.get("modeling", {})
    search_cfg = modeling.get("search", {})
    df = load_training_frame(config)
    X, y, _ = model_matrix(df, config, training=True)

    from sklearn.ensemble import GradientBoostingClassifier

    pipeline = _build_pipeline(X, GradientBoostingClassifier(random_state=int(modeling.get("random_state", 42))))
    param_dist = {
        "model__n_estimators": randint(100, 500),
        "model__max_depth": randint(2, 7),
        "model__learning_rate": uniform(0.01, 0.3),
        "model__subsample": uniform(0.6, 0.4),
        "model__min_samples_leaf": randint(1, 20),
    }
    cv = StratifiedKFold(
        n_splits=int(search_cfg.get("cv_folds", 5)),
        shuffle=True,
        random_state=int(modeling.get("random_state", 42)),
    )
    search = RandomizedSearchCV(
        pipeline,
        param_dist,
        n_iter=int(search_cfg.get("n_iter", 30)),
        scoring=search_cfg.get("scoring", "roc_auc"),
        cv=cv,
        n_jobs=-1,
        random_state=int(modeling.get("random_state", 42)),
        verbose=1,
    )
    search.fit(X, y)
    return {
        "best_params": search.best_params_,
        "best_roc_auc": round(float(search.best_score_), 4),
    }


# ---------------------------------------------------------------------------
# Supervised training (main pipeline entry-point)
# ---------------------------------------------------------------------------

def train_supervised(config: dict) -> dict:
    import joblib
    from sklearn.model_selection import train_test_split

    df = load_training_frame(config)
    X, y, _ = model_matrix(df, config, training=True)
    if y is None:
        raise ValueError("Target column is not available in the training data.")

    modeling = config.get("modeling", {})
    problem_type = config.get("project", {}).get("problem_type", "binary_classification")
    stratify = y if problem_type in {"binary_classification", "multiclass_classification", "ranking"} else None

    # Stratified split preserves class ratio in both sets (Géron Ch. 2)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=float(modeling.get("test_size", 0.2)),
        random_state=int(modeling.get("random_state", 42)),
        stratify=stratify,
    )

    classifier_name = modeling.get("classifier", "gradient_boosting")
    estimator = _get_classifier(classifier_name, config)
    model = _build_pipeline(X_train, estimator)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_valid)
    y_proba = model.predict_proba(X_valid) if hasattr(model, "predict_proba") else None
    metrics = (
        _regression_metrics(y_valid, y_pred)
        if problem_type == "regression"
        else _classification_metrics(y_valid, y_pred, y_proba, config)
    )
    metrics["classifier"] = classifier_name
    metrics["train_size"] = len(X_train)
    metrics["valid_size"] = len(X_valid)

    model_path = resolve_project_path(config, modeling.get("model_file", "models/model.joblib"))
    metrics_path = resolve_project_path(config, "reports/metrics.json")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return {"model_path": str(model_path), "metrics_path": str(metrics_path), "metrics": metrics}


# ---------------------------------------------------------------------------
# Clustering training
# ---------------------------------------------------------------------------

def train_clustering(config: dict) -> dict:
    import joblib
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    from sklearn.pipeline import Pipeline

    df = load_training_frame(config)
    X, _, prepared = model_matrix(df, config, training=True)
    preprocessor = _preprocessor(X)
    X_matrix = preprocessor.fit_transform(X)
    cluster_range = config.get("modeling", {}).get("cluster_range", [3, 4, 5, 6, 7])
    sample_size = int(config.get("modeling", {}).get("silhouette_sample_size", 10000))
    results = []
    best = None
    for k in cluster_range:
        model = KMeans(n_clusters=int(k), n_init="auto", random_state=int(config.get("modeling", {}).get("random_state", 42)))
        labels = model.fit_predict(X_matrix)
        if len(labels) > sample_size:
            rng = np.random.default_rng(int(config.get("modeling", {}).get("random_state", 42)))
            idx = rng.choice(len(labels), size=sample_size, replace=False)
            sil = silhouette_score(X_matrix[idx], labels[idx])
        else:
            sil = silhouette_score(X_matrix, labels)
        m = {
            "k": int(k),
            "silhouette": float(sil),
            "calinski_harabasz": float(calinski_harabasz_score(X_matrix, labels)),
            "davies_bouldin": float(davies_bouldin_score(X_matrix, labels)),
        }
        results.append(m)
        if best is None or m["silhouette"] > best["silhouette"]:
            best = m

    selected_model = KMeans(n_clusters=best["k"], n_init="auto", random_state=int(config.get("modeling", {}).get("random_state", 42)))
    pipeline = Pipeline([("preprocess", preprocessor), ("model", selected_model)])
    labels = pipeline.fit_predict(X)
    prepared = prepared.copy()
    prepared["cluster"] = labels

    model_path = resolve_project_path(config, config.get("modeling", {}).get("model_file", "models/model.joblib"))
    metrics_path = resolve_project_path(config, "reports/metrics.json")
    clusters_path = resolve_project_path(config, "reports/cluster_assignments.csv")
    for path in [model_path, metrics_path, clusters_path]:
        path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metrics_path.write_text(json.dumps({"selected": best, "trials": results}, indent=2), encoding="utf-8")
    prepared.to_csv(clusters_path, index=False)
    return {"model_path": str(model_path), "metrics_path": str(metrics_path), "clusters_path": str(clusters_path), "metrics": best}


# ---------------------------------------------------------------------------
# Entry-points
# ---------------------------------------------------------------------------

def train(config_path=None) -> dict:
    config = load_config(config_path)
    problem_type = config.get("project", {}).get("problem_type")
    if problem_type == "clustering":
        return train_clustering(config)
    if problem_type in {"ab_testing", "price_elasticity"}:
        from .analysis import run_analysis
        return run_analysis(config_path)
    return train_supervised(config)


def predict(config_path=None, input_path=None, output_path=None):
    import joblib
    import pandas as pd

    config = load_config(config_path)
    model_path = resolve_project_path(config, config.get("modeling", {}).get("model_file", "models/model.joblib"))
    model = joblib.load(model_path)
    raw = read_csv(input_path)
    prepared = prepare_features(raw, config, training=False)
    X, _, _ = model_matrix(prepared, config, training=False)
    result = raw.copy()
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            result["churn_score"] = proba[:, 1]
        else:
            for idx, klass in enumerate(model.classes_):
                result[f"score_{klass}"] = proba[:, idx]
    result["prediction"] = model.predict(X)
    output = Path(output_path) if output_path else resolve_project_path(config, "data/processed/predictions.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return output
