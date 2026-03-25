import pandas as pd
import numpy as np

from cf_copilot.ml_logic.registry import predict

CLASS_NAMES = [1, 2, 3, 4, 5, 6, 7]


def get_priority_invoices(invoices_df: pd.DataFrame, pipeline, current_date) -> pd.DataFrame:
    """
    End-to-end collections priority scoring & ranking from raw invoices.

    Args:
        invoices_df: raw invoices DataFrame
        pipeline: fitted sklearn Pipeline.

    Returns:
        DataFrame with top 10 most risky invoices ranked on collections
        priority score in descending order including collections_rank,
        doc_id, total_open_amount, days_overdue and risk_category.
    """
    results = predict(pipeline, invoices_df)

    risk_df = _build_risk_table(results["probabilities"], invoices_df, current_date)
    risk_df = _add_computed_risk_score(risk_df)
    risk_df = _add_collections_priority_score(risk_df)

    return _collection_ranking(risk_df)


def _build_risk_table(probas: np.ndarray, invoices_df: pd.DataFrame, current_date) -> pd.DataFrame:
    """
    Combine model probabilities with invoice details.
    """
    risk_df = pd.DataFrame(probas, columns=CLASS_NAMES)

    invoices_df = invoices_df.copy()
    invoices_df["due_in_date"] = pd.to_datetime(invoices_df["due_in_date"], format="%Y%m%d", errors="coerce")
    invoices_df["days_until_due"] = (invoices_df["due_in_date"] - current_date).dt.days
    invoices_info = ["doc_id", "cust_number", "total_open_amount", "days_until_due"]

    df = pd.concat(
        [invoices_df[invoices_info].reset_index(drop=True),
         risk_df.reset_index(drop=True)],
        axis=1
    )
    df = df.dropna(subset=[5])

    return df

def _add_computed_risk_score(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute risk score per invoice = sum of probabilities of payment in week
    buckets 5 - 7.
    """
    risk_df["risk_score"] = risk_df[5] + risk_df[6] + risk_df[7]
    risk_df["paid_soon_score"] = risk_df[1] + risk_df[2]

    # Risk score should always be between 0 and 1
    assert ((risk_df["risk_score"] >= 0) & (risk_df["risk_score"] <= 1)).all()

    # Risk score formula check for first row
    row0 = risk_df.iloc[0]
    calc_risk = row0[5] + row0[6] + row0[7]
    assert abs(row0["risk_score"] - calc_risk) < 1e-7

    # Assign risk categories based on risk scoring thresholds
    def assign_risk_buckets(risk_score: float) -> str:
        if risk_score < 0.20:
            return "Low"
        elif risk_score < 0.50:
            return "Medium"
        return "High"

    risk_df["risk_category"] = risk_df["risk_score"].apply(assign_risk_buckets)

    return risk_df


def _add_collections_priority_score(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute collections priority score per invoice = total_open_amount *
    risk_score * overdue_multiplier.
    """
    risk_df["is_overdue"] = (risk_df["days_until_due"] < 0).astype(int)
    risk_df["days_overdue"] = np.where(
        risk_df["days_until_due"] < 0,
        -risk_df["days_until_due"],
        0
    )

    def compute_overdue_multiplier(days_overdue: int) -> float:
        if days_overdue == 0:
            return 1.0
        elif days_overdue < 7:
            return 1.2
        elif days_overdue < 30:
            return 1.5
        return 2.0

    risk_df["overdue_multiplier"] = risk_df["days_overdue"].apply(compute_overdue_multiplier)

    risk_df["priority_score"] = round((risk_df["total_open_amount"] * risk_df["risk_score"] * risk_df["overdue_multiplier"]),2)

    # Priority score should never be negative
    assert (risk_df["priority_score"] >= 0).all()

    return risk_df


def _collection_ranking(risk_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks the invoices based on collection priority score and prepares the
    top 10 most riskiest invoices for follow-up actions.
    """
    risk_df = risk_df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    risk_df["collections_rank"] = risk_df.index + 1

    risk_df["pred_class"] = risk_df[CLASS_NAMES].idxmax(axis=1)
    risk_df["pred_class_proba"] = risk_df[CLASS_NAMES].max(axis=1)

    cols_to_show = [
        "collections_rank",
        "doc_id",
        "cust_number",
        "total_open_amount",
        "days_overdue",
        "risk_category"
    ]

    return risk_df[cols_to_show].head(10)
