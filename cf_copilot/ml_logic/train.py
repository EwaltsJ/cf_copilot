# cf_copilot/ml_logic/train.py

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from cf_copilot.utils import (
    load_cashflow_data,
    data_cleaning,
    build_sliding_window_snapshots,
    preprocess,
)

from cf_copilot.ml_logic.registry import save_model, save_results

def train() -> None:
    # 1) Load raw data
    df_raw = load_cashflow_data()

    # 2) Clean data and standardize columns
    model_df, demo_df = data_cleaning(df_raw)

    # 3) Build sliding window snapshots and engineer features (utils does this)
    big_df = build_sliding_window_snapshots(model_df)

    # 4) Preprocess -> X, y
    X, y = preprocess(big_df)

    # 5) Simple train/test split for now (you can replace with your CV later)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6) Define feature groups (must match columns in X_train)
    numeric_features = [
        "invoice_age_days",
        "days_until_due",
        "days_past_due",
        "invoice_month",
        "due_month",
        "customer_avg_delay",
        "late_payment_ratio",
        "prev_transaction_count",
        "customer_risk_score",
        "days_since_last_invoice",
        "invoice_amount",
    ]

    categorical_features = [
        "invoice_currency",
        "document_type",
        "cust_payment_terms",
    ]

    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="-1")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf),
        ]
    )

    # 7) Fit and basic eval
    pipeline.fit(X_train, y_train)

    probas = pipeline.predict_proba(X_test)
    preds = pipeline.predict(X_test)
    ll = log_loss(y_test, probas)

    print(f"Validation log_loss: {ll:.4f}")

    # 8) Save model and metadata
    save_model(pipeline)

    params = clf.get_params()
    metrics = {
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_test)),
        "log_loss": float(ll),
        "classes": list(pipeline.classes_),
    }
    save_results(params, metrics)


if __name__ == "__main__":
    train()
