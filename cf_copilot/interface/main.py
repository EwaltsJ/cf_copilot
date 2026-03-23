import pandas as pd

from cf_copilot.ml_logic.data import (
    load_cashflow_data,
    data_cleaning,
    build_sliding_window_snapshots,
    upload_historical_data
)
from cf_copilot.ml_logic.encoders import preprocess
from cf_copilot.ml_logic.model import initialize_model, train_model
from cf_copilot.ml_logic.registry import save_model, load_model, predict, mlflow_run, mlflow_transition_model, save_results
from cf_copilot.ml_logic.evaluation import evaluate_training_run
from cf_copilot.ml_logic.reporting import build_run_summary, build_json_artifacts


@mlflow_run
def train():
    """Full training pipeline: load → clean → augment → train → evaluate → save."""

    # 1. Load & clean
    df = load_cashflow_data()
    model_df = data_cleaning(df)

    # 2. Build augmented dataset with sliding windows + feature engineering
    big_df = build_sliding_window_snapshots(model_df)

    # 3. Time-based train/test split
    big_df = big_df.sort_values("invoice_sent").reset_index(drop=True)
    cutoff_date = big_df["invoice_sent"].quantile(0.8)

    train_df = big_df[big_df["reference_date"] <= cutoff_date]
    test_df = big_df[big_df["reference_date"] > cutoff_date]

    X_train, y_train = preprocess(train_df)
    X_test, y_test = preprocess(test_df)

    # 4. Initialize & train
    pipeline = initialize_model()
    pipeline = train_model(pipeline, X_train, y_train)

    # 5. Evaluate holdout & rolling backtests
    evaluation_results = evaluate_training_run(pipeline=pipeline, X_test=X_test, y_test=y_test,
                                               test_df=test_df, big_df=big_df, log_backtests_to_mlflow=True
    )

    metrics = evaluation_results["metrics"]
    figures = evaluation_results["figures"]
    artifacts = evaluation_results["artifacts"]
    forecast_summary = evaluation_results["forecast_summary"]
    backtest_summary = evaluation_results["backtest_summary"]
    forecast_backtest_summary = evaluation_results["forecast_backtest_summary"]

    # 6. Build run summary & combine all summaries as JSON artifacts
    run_summary = build_run_summary(model=pipeline, X_train=X_train, X_test=X_test,
        y_train=y_train, y_test=y_test, metrics=metrics, cutoff_date=cutoff_date,
        split_column="reference_date",
    )

    json_artifacts = build_json_artifacts(run_summary=run_summary, backtest_summary=backtest_summary,
        forecast_summary=forecast_summary, forecast_backtest_summary=forecast_backtest_summary,
    )

    # 7. Save results
    save_results(metrics=metrics, figures=figures, artifacts=artifacts, json_artifacts=json_artifacts)

    # 8. Save model
    save_model(pipeline)

    # 9. Move latest model to staging
    mlflow_transition_model(current_stage="None", new_stage="Staging")

    # 10. Seed / refresh historical data
    upload_historical_data()
    return pipeline


def pred(X_new: pd.DataFrame = None):
    """Load the latest model and make predictions.

    Args:
        X_new: DataFrame with the same feature columns used during training.
              If None, loads test data from raw_data/model_df.csv as a demo.
    """
    pipeline = load_model()

    if pipeline is None:
        print("❌ No model found. Run train() first.")
        return None

    if X_new is None:
        print("⚠️  No input data provided. Run with a DataFrame of features.")
        return None

    results = predict(pipeline, X_new)

    return results


if __name__ == "__main__":
    train()
