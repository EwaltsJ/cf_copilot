import pandas as pd

NUMERIC_FEATURES = [
    "business_year", "invoice_age_days", "days_until_due", "pay_terms_days",
    "invoice_month", "due_month", "days_past_due", "customer_avg_delay",
    "late_payment_ratio", "prev_transaction_count", "days_since_last_invoice",
    "customer_risk_score", "invoice_amount", "invoice_amount_log",
]

CATEGORICAL_FEATURES = [
    "invoice_currency", "document_type", "invoice_size_cat", "invoice_siz_cat_q"
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
        "doc_id","cust_number", "due_in_date", "invoice_sent", "total_open_amount",
        "open_amount", "reference_date", "week_bucket", "days_to_payment",
        "invoice_paid", "cust_payment_terms"
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["week_bucket"] if "week_bucket" in df.columns and not inference else None

    return X, y
