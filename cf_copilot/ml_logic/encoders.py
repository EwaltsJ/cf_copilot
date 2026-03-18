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


def preprocess(df: pd.DataFrame) -> tuple:
    """Preprocess a DataFrame for model training or inference.

    Removes rows without targets, imputes missing values in customer
    history features, and splits into feature matrix and target vector.
    Identifier columns, date columns, and leaky columns are excluded.

    Args:
        df: pandas DataFrame containing invoice-level records with at least
            'week_bucket', 'customer_avg_delay', 'late_payment_ratio',
            and 'days_since_last_invoice'.

    Returns:
        A tuple (X, y) where X is a DataFrame of feature columns and y is
        a Series of 'week_bucket' target labels.
    """
    df = df.copy()
    df = df.dropna(subset=["week_bucket"])

    df["customer_avg_delay"] = df["customer_avg_delay"].fillna(0)
    df["late_payment_ratio"] = df["late_payment_ratio"].fillna(0)
    df["days_since_last_invoice"] = df["days_since_last_invoice"].fillna(-1)

    drop_cols = [
        "cust_number", "due_in_date", "invoice_sent",
        "reference_date", "week_bucket", "days_to_payment",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["week_bucket"]

    return X, y
