import pandas as pd
import numpy as np

from cf_copilot.cashflow_prediction.registry import WEEK_CLASSES

def forecast_check(
    pred_cash_df: pd.DataFrame
) -> dict:
    """
    Check whether forecast math is internally consistent.
    """
    total_invoice_amount = float(pred_cash_df["total_open_amount"].sum())

    total_expected_cash = float(
        sum((pred_cash_df["total_open_amount"] * pred_cash_df[f"p_{w}"]).sum() for w in WEEK_CLASSES)
    )

    if total_expected_cash - total_invoice_amount > 0:
        raise ValueError(
            "Expected cash exceeds total invoice amount. This should not happen."
        )

    return {
        "total_invoice_amount": round(total_invoice_amount, 2),
        "total_expected_cash": round(total_expected_cash, 2),
    }

def build_actual_weekly_cf(
    invoices_df: pd.DataFrame,
    reference_date
) -> pd.DataFrame:
    """
    Build actual realized cash per week bucket for evaluation.
    Only counts payments that happen within the forecast horizon.
    """
    df = invoices_df.copy()

    # Create week buckets 1-6 based on actual payment days from reference date
    df["reference_date"] = reference_date
    df["days_to_payment"] = (df["invoice_paid"] - reference_date).dt.days
    df["week_bucket"] = np.ceil(df["days_to_payment"] / 7).astype(int)
    df["week_bucket"] = df["week_bucket"].clip(lower=1)
    df.loc[df["week_bucket"] > 6, "week_bucket"] = 7
    df = df[df["week_bucket"].between(1,6)].copy()

    # Aggregate actual cash flow per week bucket 1-6
    actual_cf = (
        df.groupby("week_bucket", as_index=False)["total_open_amount"].sum().rename(columns={"total_open_amount": "actual_cash"})
    )

    # Ensure all weeks 1-6 exist
    all_weeks = pd.DataFrame({"week_bucket": range(1,7)})
    actual_cf = all_weeks.merge(actual_cf, on="week_bucket", how="left")

    actual_cf["actual_cash"] = actual_cf["actual_cash"].fillna(0)

    return actual_cf

def compare_forecast_vs_actual(
    weekly_forecast_df: pd.DataFrame,
    actual_weekly_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare forecast and actual cash flow per week bucket.
    """
    comparison = weekly_forecast_df.merge(actual_weekly_df, on="week_bucket", how="left")

    comparison["abs_error"] = round((comparison["forecast_cash"] - comparison["actual_cash"]),2)

    comparison["perc_error"] = np.where(
        comparison["actual_cash"] == 0,
        np.nan,  # or 0.0 if you prefer
        (comparison["forecast_cash"] - comparison["actual_cash"]) / comparison["actual_cash"]
    )
    comparison["perc_error"] = comparison["perc_error"].round(2)

    return comparison

def compute_forecast_metrics(comparison_df: pd.DataFrame) -> dict:
    """
    Compute weekly forecast metrics:
    - MAE (Mean Absolute Error)
    - MAPE (Mean Absolute Percentage Error, excluding zero-actual weeks)
    """
    df = comparison_df.copy()

    # Calculate MAE
    mae = df["abs_error"].mean()

    # Calculate MAPE excluding weeks with actual cash == 0
    excluded_zero_df = df[df["actual_cash"] > 0].copy()

    if len(excluded_zero_df) > 0:
        mape = (excluded_zero_df["abs_error"] / excluded_zero_df["actual_cash"]).mean()
    else:
        mape = np.nan # no valid weeks

    # Totals useful for sanity checks and reporting
    total_forecast = df["forecast_cash"].sum()
    total_actual = df["actual_cash"].sum()

    return {
        "MAE (weekly)": round(float(mae),2),
        "MAPE (weekly)": round(float(mape),2),
        "Total actual cf": round(float(total_actual),2),
        "Total forecast cf": round(float(total_forecast),2),
        "Total cf difference": round(float((total_forecast - total_actual)),2)
    }
