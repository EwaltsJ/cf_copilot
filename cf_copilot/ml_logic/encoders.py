import pandas as pd

NUMERIC_FEATURES = [
    "invoice_age_days", "days_until_due", "days_past_due",
    "invoice_month", "due_month",
    "customer_avg_delay", "late_payment_ratio",
    "prev_transaction_count", "customer_risk_score",
    "days_since_last_invoice", "open_amount",
]

CATEGORICAL_FEATURES = [
    "invoice_currency", "document_type", "cust_payment_terms",
]

def preprocess(df: pd.DataFrame, inference: bool = False) -> tuple:
    """Preprocess a DataFrame for model training or inference.

    Imputes missing values in customer history features and splits into
    feature matrix and (optionally) target vector. Identifier columns,
    date columns, and leaky columns are excluded.

    Args:
        df: pandas DataFrame containing invoice-level records.
        inference: if True, skips dropping rows without a target and
                   returns y as None. Set this when predicting on new
                   invoices that have no 'week_bucket' column yet.

    Returns:
        A tuple (X, y) where X is a DataFrame of feature columns and y is
        a Series of 'week_bucket' target labels (or None during inference).
    """
    df = df.copy()

    if not inference:
        df = df.dropna(subset=["week_bucket"])

    df["customer_avg_delay"] = df["customer_avg_delay"].fillna(0)
    df["late_payment_ratio"] = df["late_payment_ratio"].fillna(0)
    df["days_since_last_invoice"] = df["days_since_last_invoice"].fillna(-1)

    drop_cols = [
        "doc_id","cust_number", "due_in_date", "invoice_sent",
        "reference_date", "week_bucket", "days_to_payment",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["week_bucket"] if "week_bucket" in df.columns and not inference else None

    return X, y
