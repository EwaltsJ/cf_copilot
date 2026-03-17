import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import classification_report, log_loss, confusion_matrix
from sklearn.calibration import calibration_curve


def engineer_features(snapshot, df_full, current_date):
    """Engineer all features for a given reference date."""

    # ------------------------------------------------------------------
    # A) Invoice timing features (row-level, per invoice)
    # ------------------------------------------------------------------

    # 0. Is invoice Open?
    open_invoice = snapshot["invoice_paid"].isna() | (snapshot["invoice_paid"]\
        >current_date)
    # 1. Invoice age
    snapshot["invoice_age_days"] = np.where(
        open_invoice,
        (current_date - snapshot["invoice_sent"]).dt.days,
        np.nan
    )

    # 2. Days until due date
    snapshot["days_until_due"] = np.where(
        open_invoice,
        (snapshot["due_in_date"] - current_date).dt.days,
        np.nan,
    )
    # 3. Pay Terms
    snapshot["pay_terms_days"] = (
        (snapshot["due_in_date"] - snapshot["invoice_sent"])
        .dt.days
    )

    # 3. Invoice month
    snapshot["invoice_month"] = snapshot["invoice_sent"].dt.month

    # 4. Due month
    snapshot["due_month"] = snapshot["due_in_date"].dt.month

    # 5. Days past due
    # days past due
    snapshot["days_past_due"] = (current_date - snapshot["due_in_date"]).dt.days

    # ------------------------------------------------------------------
    # B) Customer behaviour features (customer-level, using full history)
    # ------------------------------------------------------------------

    # Behaviour specific to one customer. Needs to be calculated on whole DF,
    # since we need historical data for it

    # Get all invoices that are paid before current_date
    historical = df_full[df_full["invoice_paid"] <= current_date].copy()

    # Calculate delay for all paid invoices
    historical["delay"] = (historical["invoice_paid"] - historical["due_in_date"]).dt.days.clip(lower=0)
    # Calculate avg delay for each customer
    avg_delay = historical.groupby("cust_number")["delay"].mean().rename("customer_avg_delay")

    # 7. Customer late payment ratio
    # Caluclate boolean field, whether or not an invoice was paid late
    historical["is_late"] = (historical["invoice_paid"] > historical["due_in_date"]).astype(int)
    # calculate ratio of late-payhments per customer
    late_ratio = historical.groupby("cust_number")["is_late"].mean().rename("late_payment_ratio")

    # 8. Number of previous transactions
    before_current = df_full[df_full["invoice_sent"] < current_date]
    prev_counts = before_current.groupby("cust_number").size().rename("prev_transaction_count")

    # 9. Days since last invoice
    last_invoice = (
        before_current.groupby("cust_number")["invoice_sent"]
        .max()
        .rename("last_invoice_date")
    )

    # Merge all customer-level features
    customer_features = pd.concat([avg_delay, late_ratio, prev_counts, last_invoice], axis=1)
    snapshot = snapshot.merge(customer_features, on="cust_number", how="left")

    # 10. Customer risk score
    snapshot["customer_risk_score"] = (
        0.7 * snapshot["late_payment_ratio"].fillna(0)
        + 0.3 * snapshot["customer_avg_delay"].fillna(0)
    )

    # 11. cont. Convert last invoice date to days
    snapshot["days_since_last_invoice"] = (current_date - snapshot["last_invoice_date"]).dt.days
    snapshot.drop(columns=["last_invoice_date"], inplace=True)

    # Fill NaN for new customers
    snapshot["prev_transaction_count"] = snapshot["prev_transaction_count"].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # C) Invoice characteristics (amount-based features)
    # ------------------------------------------------------------------

    # Base amount
    snapshot["invoice_amount"] = snapshot["total_open_amount"]

    # 12. Log-transform
    snapshot["invoice_amount_log"] = np.log1p(
        snapshot["invoice_amount"].clip(lower=0)
    )

    # Size categories: first fixed bins
    size_bins_fixed = [-np.inf, 10000, 100000, np.inf]
    size_labels = ["small", "medium", "large"]

    snapshot["invoice_size_cat"] = pd.cut(
        snapshot["invoice_amount"],
        bins=size_bins_fixed,
        labels=size_labels,
    )

    # 13. Optional: quantile-based bins computed on full history (df_full)
    q1, q2 = df_full["total_open_amount"].quantile([0.33, 0.66])
    size_bins_quant = [-np.inf, q1, q2, np.inf]

    snapshot["invoice_size_cat_q"] = pd.cut(
        snapshot["invoice_amount"],
        bins=size_bins_quant,
        labels=size_labels,
    )

    return snapshot

def data_cleaning(df):
    """Clean the data and create csvs."""

    # 1. Drop duplicate rows
    df = df.drop_duplicates()

    # 2. strip column names, and standardize the cases
    df.columns = df.columns.str.strip()

    # 3. Drop invalid invoice ids rows
    df = df.dropna(subset=["invoice_id"]).copy()
    df["invoice_id"] = pd.to_numeric(df["invoice_id"], errors="coerce")
    # 3.1 even if failed to parse,just drop them
    df = df[df["invoice_id"].notna()].copy()

    # 4. fixing date formats
    def parse_yyyymmdd_float(s):
        return pd.to_datetime(
            pd.to_numeric(s, errors="coerce").astype("Int64").astype("string"),
            format="%Y%m%d",
            errors="coerce",
        )

    df["due_in_date"] = parse_yyyymmdd_float(df["due_in_date"])
    df["baseline_create_date"] = parse_yyyymmdd_float(df["baseline_create_date"])

    # 5. Now cast IDs
    df["doc_id"] = df["doc_id"].astype("int64")
    df["invoice_id"] = df["invoice_id"].astype("int64")

    # 6. standardize cat_cols
    cat_cols = ["business_code", "invoice_currency", "cust_payment_terms",\
        "name_customer"]
    for c in cat_cols:
        df[c] = df[c].astype(str)

    # 7. drop unncessary columns after analysis
    #drop_cols=['area_business',"posting_id","invoice_id","document_create_date","isOpen",'document type','document_create_date.1']
    #df=df.drop(columns=drop_cols)
    df = df[['cust_number','buisness_year','due_in_date', 'invoice_currency', 'document type',\
        'total_open_amount','baseline_create_date', 'cust_payment_terms', 'clear_date']]

    # 8. Renaming misspelled columns
    df.rename(columns={'buisness_year': 'business_year', 'clear_date':\
        'invoice_paid', 'document type': 'document_type', \
        'baseline_create_date': 'invoice_sent'}, inplace=True)

    # 9. sort the df by date
    df = df.sort_values("invoice_sent").reset_index(drop=True)

    # 10. create two dfs, one for training one for testing.
    model_df = df[df["invoice_paid"].notnull()] # training, full 40k
    demo_df = df # testing, full 50k

    # 11. save the models in raw_data folder
    def save_processed_frames():
        base_dir = Path.cwd().parent          # one level up from notebook
        raw_data_dir = base_dir / "raw_data"
        raw_data_dir.mkdir(parents=True, exist_ok=True)

        model_path = raw_data_dir / "model_df.csv"
        demo_path = raw_data_dir / "demo_df.csv"

        model_df.to_csv(model_path, index=False)
        demo_df.to_csv(demo_path, index=False)

        print(f"Saved model_df to {model_path}")
        print(f"Saved demo_df to {demo_path}")

    save_processed_frames()

    return model_df,demo_df


def preprocess(df):
    """Preprocess a DataFrame for model training or inference.

    Cleans the input data by removing rows without targets, imputing missing
    values in customer history features, and splitting into feature matrix
    and target vector. Identifier columns, date columns, and leaky columns
    are excluded from the feature set.

    Args:
        df: pandas DataFrame containing invoice-level records with at least
            the columns 'week_bucket', 'customer_avg_delay',
            'late_payment_ratio', and 'days_since_last_invoice'.

    Returns:
        A tuple (X, y) where X is a DataFrame of feature columns and y is
        a Series of 'week_bucket' target labels.
    """
    df = df.dropna(subset=["week_bucket"])

    df["customer_avg_delay"] = df["customer_avg_delay"].fillna(0)
    df["late_payment_ratio"] = df["late_payment_ratio"].fillna(0)
    df["days_since_last_invoice"] = df["days_since_last_invoice"].fillna(-1)

    drop_cols = [
        "cust_number", "due_in_date", "invoice_sent", "reference_date", "week_bucket", "days_to_payment"
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols]
    y = df["week_bucket"]

    return X, y

def evaluate_model(probas, preds, y_test):
    """Print standard classification metrics for a multi-class model.

    Outputs log loss, a per-class precision/recall/F1 classification report,
    and the confusion matrix to stdout.

    Args:
        probas: array-like of shape (n_samples, n_classes) with predicted
            class probabilities.
        preds: array-like of shape (n_samples,) with predicted class labels.
        y_test: array-like of shape (n_samples,) with true class labels.

    Returns:
        None
    """
    print(f"Log loss: {log_loss(y_test, probas):.4f}")
    print(classification_report(y_test, preds))
    print(confusion_matrix(y_test, preds))
    return None


def show_calibration_curves(probas, pipeline, y_test):
    """Plot one-vs-rest calibration curves for each target bucket.

    For every class in the pipeline, a subplot compares the mean predicted
    probability against the observed frequency across 10 bins, with a
    diagonal reference line representing perfect calibration.

    Args:
        probas: array-like of shape (n_samples, n_classes) with predicted
            class probabilities.
        pipeline: fitted estimator exposing a `classes_` attribute that
            lists the target buckets.
        y_test: array-like of shape (n_samples,) with true class labels.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for i, bucket in enumerate(pipeline.classes_):
        ax = axes.flat[i]
        y_binary = (y_test == bucket).astype(int)
        prob_true, prob_pred = calibration_curve(y_binary, probas[:, i], n_bins=10)
        ax.plot(prob_pred, prob_true, marker="o")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_title(f"Bucket {bucket}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.show()


def simulate_past_performance(pipeline, df):
    """Run a walk-forward backtest of the pipeline over historical cutoffs.

    Selects five temporal cutoff points (at the 40th through 80th
    percentiles of reference dates). For each cutoff, the model is trained
    on all data up to that date and evaluated on the following six weeks.
    Log loss is printed per cutoff along with the mean and standard
    deviation across all folds.

    Args:
        pipeline: an unfitted scikit-learn estimator (or pipeline) with
            fit and predict_proba methods.
        df: pandas DataFrame containing the full dataset, including a
            'reference_date' column used for temporal splitting.
    """
    reference_dates = df["reference_date"].sort_values().unique()
    cutoffs = np.percentile(reference_dates.astype(int), [40, 50, 60, 70, 80])
    cutoffs = pd.to_datetime(cutoffs)

    scores = []
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
        score = log_loss(y_test, probas, labels=pipeline.classes_)
        scores.append(score)
        print(f"Cutoff {cutoff.date()} -> log_loss: {score:.4f}")

    print(f"\nAverage: {np.mean(scores):.4f} (std: {np.std(scores):.4f})")

def build_sliding_window_snapshots(df):
    """Build an augmented dataset by sliding a weekly window over invoice history.

    For each weekly snapshot, the function selects all invoices that have been
    sent but not yet paid as of the current window date. It then engineers
    additional features via the provided callback and computes the target
    variable ('week_bucket') indicating how many weeks until payment, capped
    at 7 (meaning 7+ weeks out).

    The window starts at the earliest invoice send date and advances in
    one-week increments until six weeks before the latest payment date,
    ensuring every snapshot has a fully observable target.

    Args:
        invoices: pandas DataFrame with at least the columns 'invoice_sent'
            and 'invoice_paid' (both datetime). May contain additional
            columns consumed by the feature engineering callback.

    Returns:
        A pandas DataFrame combining all weekly snapshots, with added columns
        'reference_date', 'days_to_payment', and 'week_bucket' (int, 1–7).
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
