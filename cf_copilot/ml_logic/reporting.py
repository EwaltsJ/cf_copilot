from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


def build_run_summary(
    model: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    metrics: dict,
    cutoff_date,
    split_column: str = "reference_date",
) -> dict:
    """
    Build a structured summary of the training run for auditability and experiment comparison.

    Args:
        model: fitted sklearn Pipeline
        X_train: train features
        X_test: test features
        y_train: train labels
        y_test: test labels
        metrics: scalar evaluation metrics
        cutoff_date: time split cutoff
        split_column: name of temporal split column

    Returns:
        Dict suitable for JSON serialization
    """
    classifier = model.named_steps.get("classifier")
    classes = [str(c) for c in model.classes_] if hasattr(model, "classes_") else []

    summary = {
        "model_summary": {
            "pipeline_type": type(model).__name__,
            "classifier_type": type(classifier).__name__ if classifier is not None else None,
            "classifier_params": classifier.get_params() if classifier is not None else {},
        },
        "data_summary": {
            "n_train_rows": int(len(X_train)),
            "n_test_rows": int(len(X_test)),
            "n_features_train": int(X_train.shape[1]),
            "n_features_test": int(X_test.shape[1]),
            "feature_names": list(X_train.columns) if hasattr(X_train, "columns") else [],
            "split_column": split_column,
            "cutoff_date": str(cutoff_date),
        },
        "target_summary": {
            "classes": classes,
            "n_classes": int(len(classes)),
            "train_class_distribution": {
                str(k): int(v) for k, v in y_train.value_counts().sort_index().items()
            },
            "test_class_distribution": {
                str(k): int(v) for k, v in y_test.value_counts().sort_index().items()
            },
        },
        "metric_summary": metrics,
        "run_metadata": {
            "artifact_version": "v1",
        },
    }

    return summary


def make_json_serializable(obj):
    """
    Recursively convert numpy / pandas objects into JSON-serializable Python types.
    """
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        return float(obj)

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()

    if pd.isna(obj) and not isinstance(obj, str):
        return None

    return obj


def build_json_artifacts(
    run_summary: dict,
    backtest_summary: dict | None = None,
    forecast_summary: dict | None = None,
    forecast_backtest_summary: dict | None = None,
) -> dict:
    json_artifacts = {"run_summary.json": run_summary}

    if backtest_summary:
        json_artifacts["backtest_summary.json"] = backtest_summary

    if forecast_summary:
        json_artifacts["forecast_holdout_summary.json"] = forecast_summary

    if forecast_backtest_summary:
        json_artifacts["forecast_backtest_summary.json"] = forecast_backtest_summary

    return json_artifacts
