import numpy as np
import pandas as pd
from pathlib import Path
import kagglehub
from kagglehub import KaggleDatasetAdapter

def load_cashflow_data(csv_name: str = "dataset.csv") -> pd.DataFrame:
    """Load invoice dataset from local raw_data folder, or download from Kaggle.

    Args:
        csv_name: filename of the CSV to load.

    Returns:
        A pandas DataFrame with the raw invoice data.
    """
    base_dir = Path(__file__).resolve().parents[2]  # repo root (cf_copilot/)
    raw_data_dir = base_dir / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    local_path = raw_data_dir / csv_name

    if local_path.is_file():
        print(f"Loading local file: {local_path}")
        return pd.read_csv(local_path)

    print("Local file not found, downloading from Kaggle via kagglehub...")
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "pradumn203/payment-date-prediction-for-invoices-dataset",
        "dataset.csv",
    )

    df.to_csv(local_path, index=False)
    print(f"Saved local copy to {local_path}")

    return df


def data_cleaning(df: pd.DataFrame) -> tuple:
    """Clean raw data and split into model and demo DataFrames.

    Deduplicates rows, parses dates, renames misspelled columns, selects
    relevant columns, and splits into a model_df (paid invoices only)
    and demo_df (all invoices).

    Args:
        df: raw DataFrame from load_cashflow_data().

    Returns:
        A tuple (model_df, demo_df).
    """
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()

    # Drop invalid invoice ids
    df = df.dropna(subset=["invoice_id"]).copy()
    df["invoice_id"] = pd.to_numeric(df["invoice_id"], errors="coerce")
    df = df[df["invoice_id"].notna()].copy()

    # Parse date columns
    def parse_yyyymmdd_float(s):
        return pd.to_datetime(
            pd.to_numeric(s, errors="coerce").astype("Int64").astype("string"),
            format="%Y%m%d",
            errors="coerce",
        )

    df["due_in_date"] = parse_yyyymmdd_float(df["due_in_date"])
    df["baseline_create_date"] = parse_yyyymmdd_float(df["baseline_create_date"])

    # Cast IDs
    df["doc_id"] = df["doc_id"].astype("int64")
    df["invoice_id"] = df["invoice_id"].astype("int64")

    # Standardize categorical columns
    cat_cols = ["business_code", "invoice_currency", "cust_payment_terms", "name_customer"]
    for c in cat_cols:
        df[c] = df[c].astype(str)

    # Select and rename columns
    df = df[['cust_number', 'buisness_year', 'due_in_date', 'invoice_currency',
             'document type', 'total_open_amount', 'baseline_create_date',
             'cust_payment_terms', 'clear_date']]

    df.rename(columns={
        'buisness_year': 'business_year',
        'clear_date': 'invoice_paid',
        'document type': 'document_type',
        'baseline_create_date': 'invoice_sent',
    }, inplace=True)

    df = df.sort_values("invoice_sent").reset_index(drop=True)

    model_df = df[df["invoice_paid"].notnull()]
    demo_df = df

    # Save processed frames
    base_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = base_dir / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)

    model_df.to_csv(raw_data_dir / "model_df.csv", index=False)
    demo_df.to_csv(raw_data_dir / "demo_df.csv", index=False)

    print(f"Saved model_df ({len(model_df)} rows) and demo_df ({len(demo_df)} rows)")

    return model_df, demo_df


def engineer_features(snapshot: pd.DataFrame, df_full: pd.DataFrame,
                      current_date) -> pd.DataFrame:
    """Engineer all features for a given reference date.

    Adds invoice timing features, customer behaviour features (from full
    history), and amount-based features to the snapshot.

    Args:
        snapshot: DataFrame of open invoices at current_date.
        df_full: full invoice DataFrame (for historical calculations).
        current_date: the reference date for the snapshot.

    Returns:
        The snapshot DataFrame with all engineered features added.
    """
    # A) Invoice timing features
    open_invoice = snapshot["invoice_paid"].isna() | (snapshot["invoice_paid"] > current_date)

    snapshot["invoice_age_days"] = np.where(
        open_invoice,
        (current_date - snapshot["invoice_sent"]).dt.days,
        np.nan,
    )
    snapshot["days_until_due"] = np.where(
        open_invoice,
        (snapshot["due_in_date"] - current_date).dt.days,
        np.nan,
    )
    snapshot["pay_terms_days"] = (snapshot["due_in_date"] - snapshot["invoice_sent"]).dt.days
    snapshot["invoice_month"] = snapshot["invoice_sent"].dt.month
    snapshot["due_month"] = snapshot["due_in_date"].dt.month
    snapshot["days_past_due"] = (current_date - snapshot["due_in_date"]).dt.days

    # B) Customer behaviour features
    historical = df_full[df_full["invoice_paid"] <= current_date].copy()
    historical["delay"] = (historical["invoice_paid"] - historical["due_in_date"]).dt.days.clip(lower=0)
    avg_delay = historical.groupby("cust_number")["delay"].mean().rename("customer_avg_delay")

    historical["is_late"] = (historical["invoice_paid"] > historical["due_in_date"]).astype(int)
    late_ratio = historical.groupby("cust_number")["is_late"].mean().rename("late_payment_ratio")

    before_current = df_full[df_full["invoice_sent"] < current_date]
    prev_counts = before_current.groupby("cust_number").size().rename("prev_transaction_count")

    last_invoice = (
        before_current.groupby("cust_number")["invoice_sent"]
        .max()
        .rename("last_invoice_date")
    )

    customer_features = pd.concat([avg_delay, late_ratio, prev_counts, last_invoice], axis=1)
    snapshot = snapshot.merge(customer_features, on="cust_number", how="left")

    snapshot["customer_risk_score"] = (
        0.7 * snapshot["late_payment_ratio"].fillna(0)
        + 0.3 * snapshot["customer_avg_delay"].fillna(0)
    )

    snapshot["days_since_last_invoice"] = (current_date - snapshot["last_invoice_date"]).dt.days
    snapshot.drop(columns=["last_invoice_date"], inplace=True)
    snapshot["prev_transaction_count"] = snapshot["prev_transaction_count"].fillna(0).astype(int)

    # C) Invoice characteristics
    snapshot["invoice_amount"] = snapshot["total_open_amount"]
    snapshot["invoice_amount_log"] = np.log1p(snapshot["invoice_amount"].clip(lower=0))
    snapshot["open_amount"] = snapshot["total_open_amount"]

    size_bins_fixed = [-np.inf, 10000, 100000, np.inf]
    size_labels = ["small", "medium", "large"]
    snapshot["invoice_size_cat"] = pd.cut(
        snapshot["invoice_amount"], bins=size_bins_fixed, labels=size_labels,
    )

    q1, q2 = df_full["total_open_amount"].quantile([0.33, 0.66])
    size_bins_quant = [-np.inf, q1, q2, np.inf]
    snapshot["invoice_size_cat_q"] = pd.cut(
        snapshot["invoice_amount"], bins=size_bins_quant, labels=size_labels,
    )

    return snapshot


def build_sliding_window_snapshots(df: pd.DataFrame) -> pd.DataFrame:
    """Build augmented dataset by sliding a weekly window over invoice history.

    For each weekly snapshot, selects all invoices sent but not yet paid,
    engineers features, and computes the target 'week_bucket' (1-7).

    Args:
        df: cleaned DataFrame with 'invoice_sent' and 'invoice_paid' columns.

    Returns:
        Concatenated DataFrame of all weekly snapshots with target column.
    """
    all_snapshots = []

    start_date = df["invoice_sent"].min()
    end_date = df["invoice_paid"].max() - pd.Timedelta(weeks=6)
    stride = pd.Timedelta(weeks=1)

    current = start_date
    while current <= end_date:
        snapshot = df[
            (df["invoice_sent"] <= current) &
            (df["invoice_paid"] > current)
        ].copy()

        snapshot = engineer_features(snapshot, df, current)

        snapshot["reference_date"] = current
        snapshot["days_to_payment"] = (snapshot["invoice_paid"] - current).dt.days
        snapshot["week_bucket"] = np.ceil(snapshot["days_to_payment"] / 7).astype(int)
        snapshot["week_bucket"] = snapshot["week_bucket"].clip(lower=1)
        snapshot.loc[snapshot["week_bucket"] > 7, "week_bucket"] = 7

        all_snapshots.append(snapshot)
        current += stride

    big_df = pd.concat(all_snapshots, ignore_index=True)
    print(f"Original rows: {len(df)}")
    print(f"Augmented rows: {len(big_df)}")
    print(big_df["week_bucket"].value_counts().sort_index())

    return big_df
