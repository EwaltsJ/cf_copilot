import pandas as pd
from sklearn.metrics import log_loss

from cf_copilot.ml_logic.data import load_cashflow_data, data_cleaning, build_sliding_window_snapshots
from cf_copilot.ml_logic.encoders import preprocess
from cf_copilot.ml_logic.model import initialize_model, train_model, evaluate_model
from cf_copilot.ml_logic.registry import save_model, load_model, predict
from cf_copilot.ml_logic.registry import mlflow_run, mlflow_transition_model,save_results

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

    # 5. Evaluate
    metrics = evaluate_model(pipeline, X_test, y_test)

    # Save results on the hard drive using taxifare.ml_logic.registry
    save_results(metrics=dict(metrics))

    # 6. Save
    save_model(pipeline)

    # The latest model should be moved to staging
    mlflow_transition_model(current_stage="None", new_stage="Staging")

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
