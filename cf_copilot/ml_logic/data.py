import numpy as np
import pandas as pd
from pathlib import Path
import kagglehub
from kagglehub import KaggleDatasetAdapter
import io
import shutil
from google.cloud import storage as gcs_storage
from cf_copilot.params import (
    ENV,
    GCS_BUCKET_NAME,
    GCS_HISTORICAL_DATA_PATH,
    LOCAL_HISTORICAL_DATA_PATH,
)

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


def data_cleaning(df: pd.DataFrame, predict: bool=False) -> tuple:
    """Clean raw data and split into model and demo DataFrames.

    Deduplicates rows, parses dates, renames misspelled columns, selects
    relevant columns, and splits into a model_df (paid invoices only)
    and demo_df (all invoices).

    Args:
        df: raw DataFrame from load_cashflow_data().

    Returns:
        A tuple (model_df, demo_df).
    """
    # to remove warning of SettingWithCopyWarning:
    df = df.copy()

    # drop duplicates and name stripping
    df = df.drop_duplicates()
    df.columns = df.columns.str.strip()

    if "cust_number" in df.columns:
        df["cust_number"] = df["cust_number"].astype(str)
    # Drop invalid invoice ids
    df = df.dropna(subset=["invoice_id"]).copy()
    df["invoice_id"] = pd.to_numeric(df["invoice_id"], errors="coerce")
    df = df[df["invoice_id"].notna()].copy()

    # Parse date columns
    df["due_in_date"] = pd.to_datetime(df["due_in_date"], format="%Y%m%d", errors="coerce")
    df["baseline_create_date"] = pd.to_datetime(df["baseline_create_date"], format="%Y%m%d", errors="coerce")
    df["clear_date"] = pd.to_datetime(df["clear_date"], errors="coerce")
    # Cast IDs
    df["doc_id"] = df["doc_id"].astype("int64")

    # Standardize categorical columns
    cat_cols = ["business_code", "name_customer", "invoice_currency", "document type", "cust_payment_terms"]
    for c in cat_cols:
        df[c] = df[c].astype(str)

    # Select and rename columns
    df = df[['doc_id','cust_number', 'buisness_year', 'due_in_date', 'invoice_currency',
             'document type', 'total_open_amount', 'baseline_create_date',
             'cust_payment_terms', 'clear_date']]

    df.rename(columns={
        'buisness_year': 'business_year',
        'clear_date': 'invoice_paid',
        'document type': 'document_type',
        'baseline_create_date': 'invoice_sent',
    }, inplace=True)

    # sort values
    df = df.sort_values("invoice_sent").reset_index(drop=True)
    # data conversion
    df["business_year"] = df["business_year"].round().astype("int64")

    # converting cad to usd
    cad_to_usd_by_year = {
        2018 : 0.771,
        2019 : 0.754,
        2020 : 0.745
    }
    def convert_cad_to_usd_amount(row):
        if row['invoice_currency'].strip().upper() == "USD":
            return row["total_open_amount"]
        elif row['invoice_currency'].strip().upper() == "CAD":
            year = int(row['business_year'])
            rate = cad_to_usd_by_year.get(year,0.75)
            return row["total_open_amount"] * rate
        else:
            return row["total_open_amount"]

    df["total_open_amount"] = df.apply(convert_cad_to_usd_amount,axis=1)

    if not predict:
        df = df[df["invoice_paid"].notnull()]
    # Save processed frames
    base_dir = Path(__file__).resolve().parents[2]
    raw_data_dir = base_dir / "raw_data"
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    if not predict:
        df.to_csv(raw_data_dir / "df.csv", index=False)
    #demo_df.to_csv(raw_data_dir / "demo_df.csv", index=False)

    #print(f"Saved df ({len(df)} rows) and demo_df ({len(demo_df)} rows)")
    return df


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

    # Sin/cos cyclic encoding for month
    snapshot["invoice_month_sin"] = np.sin(2 * np.pi * snapshot["invoice_month"] / 12)
    snapshot["invoice_month_cos"] = np.cos(2 * np.pi * snapshot["invoice_month"] / 12)
    snapshot["due_month_sin"]     = np.sin(2 * np.pi * snapshot["due_month"] / 12)
    snapshot["due_month_cos"]     = np.cos(2 * np.pi * snapshot["due_month"] / 12)

    # B) Customer behaviour features
    historical = df_full[df_full["invoice_paid"] <= current_date].copy()
    historical["delay"] = (historical["invoice_paid"] - historical["due_in_date"]).dt.days.clip(lower=0)

    # Existing aggregates
    avg_delay = historical.groupby("cust_number")["delay"].mean().rename("customer_avg_delay")
    historical["is_late"] = (historical["invoice_paid"] > historical["due_in_date"]).astype(int)
    late_ratio = historical.groupby("cust_number")["is_late"].mean().rename("late_payment_ratio")

    # NEW: variability and extremes of delay
    delay_std = historical.groupby("cust_number")["delay"].std().fillna(0).rename("customer_delay_std")
    max_delay = historical.groupby("cust_number")["delay"].max().fillna(0).rename("customer_max_delay")

    before_current = df_full[df_full["invoice_sent"] < current_date]
    prev_counts = before_current.groupby("cust_number").size().rename("prev_transaction_count")

    last_invoice = (
        before_current.groupby("cust_number")["invoice_sent"]
        .max()
        .rename("last_invoice_date")
    )

    # Add new columns into customer_features
    customer_features = pd.concat(
        [avg_delay, late_ratio, prev_counts, last_invoice, delay_std, max_delay],
        axis=1
    )
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
        snapshot["week_bucket"] = np.floor(snapshot["days_to_payment"] / 7).astype(int)
        snapshot["week_bucket"] = snapshot["week_bucket"].clip(lower=0, upper=6)

        all_snapshots.append(snapshot)
        current += stride

    big_df = pd.concat(all_snapshots, ignore_index=True)
    print(f"Original rows: {len(df)}")
    print(f"Augmented rows: {len(big_df)}")
    print(big_df["week_bucket"].value_counts().sort_index())

    return big_df

def upload_historical_data(local_csv_path: str = None) -> None:
    if local_csv_path is None:
        base_dir = Path(__file__).resolve().parents[2]
        local_csv_path = base_dir / "raw_data" / "df.csv"

    local_path = Path(local_csv_path)
    if not local_path.is_file():
        raise FileNotFoundError(
            f"Source file not found at {local_path}. "
            "Run data_cleaning() first to generate df.csv."
        )
    if ENV != 'production':
        dest = Path(LOCAL_HISTORICAL_DATA_PATH)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local_path, dest)
        print(f"✅ Historical data copied locally to {dest}")
        return

    client = gcs_storage.Client()
    blob = client.bucket(GCS_BUCKET_NAME).blob(GCS_HISTORICAL_DATA_PATH)
    blob.upload_from_filename(str(local_path), content_type="text/csv")
    print(f"✅ Uploaded {local_path} → gs://{GCS_BUCKET_NAME}/{GCS_HISTORICAL_DATA_PATH}")



def load_historical_data() -> pd.DataFrame:
    date_cols = ["invoice_sent", "due_in_date", "invoice_paid"]

    if ENV == 'production':
        client = gcs_storage.Client()
        blob = client.bucket(GCS_BUCKET_NAME).blob(GCS_HISTORICAL_DATA_PATH)
        data = blob.download_as_bytes()
        df = pd.read_csv(io.BytesIO(data), parse_dates=date_cols)
        print(f"✅ Historical data loaded from GCS ({df.shape[0]} rows)")
        return df

    local_path = Path(LOCAL_HISTORICAL_DATA_PATH)
    if not local_path.is_file():
        # Fallback: initialize from df.csv on first use
        base_dir = Path(__file__).resolve().parents[2]
        model_df_path = base_dir / "raw_data" / "df.csv"
        if model_df_path.is_file():
            df = pd.read_csv(model_df_path, parse_dates=date_cols)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(local_path, index=False)
            print(f"✅ Initialized historical data from {model_df_path} → {local_path}")
            if "cust_number" in df.columns:
                df["cust_number"] = df["cust_number"].astype(str)
            return df

        # If even df.csv is missing, last‑resort error
        raise FileNotFoundError(
            f"No historical data at {local_path} and no model_df at {model_df_path}. "
            "Run data_cleaning() or upload_historical_data() first."
        )

    df = pd.read_csv(local_path, parse_dates=date_cols)
    if "cust_number" in df.columns:
        df["cust_number"] = df["cust_number"].astype(str)
    #print(f"✅ Historical data loaded locally ({df.shape[0]} rows) from {local_path}")
    return df


def append_to_historical_data(new_df: pd.DataFrame) -> None:
    try:
        historical_df = load_historical_data()
    except FileNotFoundError:
        print("⚠️  No historical data found — initializing from new data.")
        historical_df = pd.DataFrame()

    combined = pd.concat([historical_df, new_df], ignore_index=True)

    if "doc_id" in combined.columns:
        before = len(combined)
        combined = combined.drop_duplicates(subset=["doc_id"], keep="last")
        dropped = before - len(combined)
        if dropped:
            print(f"⚠️  Dropped {dropped} duplicate doc_id rows")
    else:
        print("⚠️ doc_idcolumn missing — deduplication will not work")

    print(f"✅ Historical data updated: {len(combined)} total rows")

    if ENV == 'production':
        client = gcs_storage.Client()
        blob = client.bucket(GCS_BUCKET_NAME).blob(GCS_HISTORICAL_DATA_PATH)
        blob.upload_from_string(
            combined.to_csv(index=False).encode("utf-8"),
            content_type="text/csv",
        )
        print(f"✅ Written back to gs://{GCS_BUCKET_NAME}/{GCS_HISTORICAL_DATA_PATH}")
        return

    local_path = Path(LOCAL_HISTORICAL_DATA_PATH)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(local_path, index=False)
    #print(f"✅ Written back locally to {local_path}")
