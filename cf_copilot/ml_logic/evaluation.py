import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow

from colorama import Fore, Style

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, log_loss, confusion_matrix, accuracy_score, top_k_accuracy_score, ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from sklearn.base import clone

from cf_copilot.ml_logic.encoders import preprocess
from cf_copilot.cashflow_prediction.evaluation import evaluate_forecast_holdout, simulate_forecast_backtest


def evaluate_training_run(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_df: pd.DataFrame,
    big_df: pd.DataFrame,
    log_backtests_to_mlflow: bool = True,
) -> dict:
    """
    Run all evaluation steps for a trained pipeline and return one structured result.

    Returns:
        {
            "metrics": dict,
            "figures": dict,
            "artifacts": dict,
            "forecast_summary": dict,
            "backtest_summary": dict,
            "forecast_backtest_summary": dict,
        }
    """
    # 1. Holdout ML evaluation
    metrics, figures, artifacts = evaluate_model(pipeline, X_test, y_test)

    # 2. Holdout forecast evaluation
    forecast_metrics, forecast_summary = evaluate_forecast_holdout(pipeline, test_df)
    metrics.update(forecast_metrics)

    # 3. Rolling backtest - ML metrics
    backtest_pipeline = clone(pipeline)
    backtest_summary = simulate_past_performance(
        backtest_pipeline,
        big_df,
        log_to_mlflow=log_backtests_to_mlflow,
    )

    # 4. Rolling backtest - Forecast metrics
    forecast_backtest_pipeline = clone(pipeline)
    forecast_backtest_summary = simulate_forecast_backtest(
        forecast_backtest_pipeline,
        big_df,
        log_to_mlflow=log_backtests_to_mlflow,
    )

    return {
        "metrics": metrics,
        "figures": figures,
        "artifacts": artifacts,
        "forecast_summary": forecast_summary,
        "backtest_summary": backtest_summary,
        "forecast_backtest_summary": forecast_backtest_summary,
    }


def show_calibration_curves(probas: np.ndarray, classes: np.ndarray, y_test: pd.Series):
    """Plot one-vs-rest calibration curves for each target bucket.

    Args:
        probas: predicted class probabilities of shape (n_samples, n_classes).
        pipeline: fitted estimator with a classes_ attribute.
        y_test: true class labels.
    """
    n_classes = len(classes)
    n_cols = 4
    n_rows = math.ceil(n_classes / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, bucket in enumerate(classes):
        ax = axes[i]
        y_binary = (y_test == bucket).astype(int)

        # Skip empty/degenerate cases safely
        if y_binary.nunique() < 2:
            ax.text(0.5, 0.5, f"Not enough class variation\nfor {bucket}", ha="center", va="center")
            ax.set_title(f"Bucket {bucket}")
            ax.set_axis_off()
            continue

        prob_true, prob_pred = calibration_curve(y_binary, probas[:, i], n_bins=10, strategy="uniform")
        ax.plot(prob_pred, prob_true, marker="o")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_title(f"Bucket {bucket}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed frequency")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    for j in range(n_classes, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Calibration Curves by Payment Bucket", fontsize=14)
    fig.tight_layout()

    return fig


def build_confusion_matrix_figure(y_true: pd.Series, y_pred: np.ndarray, classes: np.ndarray):
    """Build a confusion matrix figure."""
    cm = confusion_matrix(y_true, y_pred, labels=classes)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()

    return fig


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> tuple[dict, dict, dict]:
    """Evaluate trained model performance on the dataset.

    Args:
        model: a fitted sklearn Pipeline.
        X: feature DataFrame.
        y: true target labels.

    Returns:
        metrics: dict of scalar metrics
        figures: dict of matplotlib figures to persist/log
    """
    print(Fore.BLUE + f"\nEvaluating model on {len(X)} rows..." + Style.RESET_ALL)

    if model is None:
        print("❌ No model to evaluate")
        return {}, {}, {}

    probas = model.predict_proba(X)
    preds = model.predict(X)

    logloss = log_loss(y, probas, labels=model.classes_)
    top_1_accuracy = accuracy_score(y, preds)
    top_2_accuracy = top_k_accuracy_score(y_true=y, y_score=probas, k=2, labels=model.classes_)

    metrics = {
        "log_loss": float(logloss),
        "top_1_accuracy": float(top_1_accuracy),
        "top_2_accuracy": float(top_2_accuracy),
    }

    report_text = classification_report(y, preds)
    print(f"✅ Log loss: {logloss:.4f}")
    print(f"✅ Top-1 accuracy: {top_1_accuracy:.4f}")
    print(f"✅ Top-2 accuracy: {top_2_accuracy:.4f}")
    print("\nClassification report:")
    print(report_text)
    print("Confusion matrix:")
    print(confusion_matrix(y, preds))

    calibration_fig = show_calibration_curves(probas=probas, classes=model.classes_, y_test=y)

    confusion_matrix_fig = build_confusion_matrix_figure(y_true=y, y_pred=preds, classes=model.classes_)

    figures = {
        "calibration_curves": calibration_fig,
        "confusion_matrix": confusion_matrix_fig,
    }

    artifacts = {
        "classification_report.txt": report_text,
    }

    return metrics, figures, artifacts


def _log_ml_backtest_metrics(summary: dict) -> None:
    """Log aggregate ML backtest metrics to MLflow."""
    if not summary or "aggregate" not in summary:
        return

    aggregate = summary["aggregate"]

    mlflow.log_metrics({
        "backtest_avg_log_loss": aggregate["avg_log_loss"],
        "backtest_avg_top_1": aggregate["avg_top_1_accuracy"],
        "backtest_avg_top_2": aggregate["avg_top_2_accuracy"],
    })


def simulate_past_performance(pipeline, df, log_to_mlflow: bool = False):
    """Run a walk-forward backtest of the pipeline over historical cutoffs.

    Args:
        pipeline: an unfitted sklearn Pipeline.
        df: full DataFrame with 'reference_date' column for temporal splitting.

    Returns:
        dict with:
            - per_cutoff_results
            - aggregate metrics
    """
    reference_dates = df["reference_date"].sort_values().unique()
    cutoffs = np.percentile(reference_dates.astype("datetime64[ns]").astype(np.int64), [40, 50, 60, 70, 80])
    cutoffs = pd.to_datetime(cutoffs)

    results = []

    for cutoff in cutoffs:
        train_df = df[df["reference_date"] <= cutoff]
        test_df = df[
            (df["reference_date"] > cutoff) &
            (df["reference_date"] <= cutoff + pd.Timedelta(weeks=6))
        ]

        if len(test_df) == 0:
            continue

        X_train, y_train = preprocess(train_df)
        X_test, y_test = preprocess(test_df)

        pipeline.fit(X_train, y_train)
        probas = pipeline.predict_proba(X_test)
        preds = pipeline.predict(X_test)

        fold_log_loss = log_loss(y_test, probas, labels=pipeline.classes_)
        fold_top_1 = accuracy_score(y_test, preds)
        fold_top_2 = top_k_accuracy_score(y_true=y_test, y_score=probas, k=2, labels=pipeline.classes_)

        result = {
            "cutoff": cutoff.date().isoformat(),
            "log_loss": float(fold_log_loss),
            "top_1_accuracy": float(fold_top_1),
            "top_2_accuracy": float(fold_top_2),
        }

        results.append(result)

        print(
            f"Cutoff {cutoff.date()} -> "
            f"log_loss: {fold_log_loss:.4f}, "
            f"top_1: {fold_top_1:.4f}, "
            f"top_2: {fold_top_2:.4f}"
        )

    if not results:
        return {}

    avg_log_loss = float(np.mean([r["log_loss"] for r in results]))
    avg_top_1 = float(np.mean([r["top_1_accuracy"] for r in results]))
    avg_top_2 = float(np.mean([r["top_2_accuracy"] for r in results]))

    summary = {
        "per_cutoff": results,
        "aggregate": {
            "avg_log_loss": avg_log_loss,
            "avg_top_1_accuracy": avg_top_1,
            "avg_top_2_accuracy": avg_top_2,
        }
    }

    print("\nModel backtest summary")
    print(summary["aggregate"])

    if log_to_mlflow:
        _log_ml_backtest_metrics(summary)

    return summary
