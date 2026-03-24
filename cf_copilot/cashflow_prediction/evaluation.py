import pandas as pd
import numpy as np
import mlflow
from sklearn.pipeline import Pipeline

from colorama import Fore, Style

from cf_copilot.cashflow_prediction.registry import WEEK_CLASSES
from cf_copilot.ml_logic.encoders import preprocess


def build_actual_weekly_cf(invoices_df: pd.DataFrame, reference_date) -> pd.DataFrame:
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


def compare_forecast_vs_actual(weekly_forecast_df: pd.DataFrame, actual_weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare forecast and actual cash flow per week bucket.
    """
    comparison = weekly_forecast_df.merge(actual_weekly_df, on="week_bucket", how="left")

    comparison["abs_error"] = (comparison["forecast_cash"] - comparison["actual_cash"]).abs().round(2)

    comparison["perc_error"] = np.where(
        comparison["actual_cash"] == 0,
        np.nan,
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
        mape = np.nan

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


def evaluate_forecast_holdout(
    model: Pipeline,
    test_df: pd.DataFrame,
    verbose: bool = True
) -> tuple[dict, dict]:
    """
    Evaluate forecast quality on the holdout set by reference_date.

    Args:
        model: fitted sklearn Pipeline
        test_df: holdout snapshot dataframe
        verbose: whether to print evaluation output

    Returns:
        forecast_metrics: scalar metrics to merge into MLflow metrics
        forecast_summary: JSON-serializable artifact with per-reference-date detail
    """
    if verbose:
        print(Fore.BLUE + f"\nEvaluating forecast on {len(test_df)} rows..." + Style.RESET_ALL)

    if model is None:
        if verbose:
            print("❌ No model to evaluate forecast")
        return {}, {}

    if len(test_df) == 0:
        if verbose:
            print("❌ No holdout rows available for forecast evaluation")
        return {}, {}

    per_reference_results = []
    class_to_index = {int(c): i for i, c in enumerate(model.classes_)}

    for reference_date, snapshot_df in test_df.groupby("reference_date"):
        snapshot_df = snapshot_df.copy()
        if len(snapshot_df) == 0:
            continue

        X_snapshot, _ = preprocess(snapshot_df)
        probas = model.predict_proba(X_snapshot)

        pred_cash_df = snapshot_df.copy()

        for w in WEEK_CLASSES:
            pred_cash_df[f"p_{w}"] = (
                probas[:, class_to_index[int(w)]]
                if int(w) in class_to_index
                else 0.0
            )

        weekly_forecast_df = pd.DataFrame([
            {
                "week_bucket": int(w),
                "forecast_cash": round(
                    float((pred_cash_df["total_open_amount"] * pred_cash_df[f"p_{w}"]).sum()),
                    2,
                ),
            }
            for w in WEEK_CLASSES
        ])

        total_invoice_amount = round(float(pred_cash_df["total_open_amount"].sum()), 2)
        total_expected_cash = round(float(weekly_forecast_df["forecast_cash"].sum()), 2)

        if total_expected_cash - total_invoice_amount > 0:
            raise ValueError("Expected cash exceeds total invoice amount. This should not happen.")

        actual_weekly_df = build_actual_weekly_cf(
            invoices_df=snapshot_df,
            reference_date=pd.Timestamp(reference_date),
        )

        comparison_df = compare_forecast_vs_actual(
            weekly_forecast_df=weekly_forecast_df,
            actual_weekly_df=actual_weekly_df,
        )

        snapshot_metrics = compute_forecast_metrics(comparison_df)

        per_reference_results.append({
            "reference_date": str(pd.Timestamp(reference_date).date()),
            "forecast_check": {
                "total_invoice_amount": total_invoice_amount,
                "total_expected_cash": total_expected_cash,
            },
            "forecast_metrics": {
                "forecast_mae_weekly": float(snapshot_metrics["MAE (weekly)"]),
                "forecast_mape_weekly": (
                    float(snapshot_metrics["MAPE (weekly)"])
                    if pd.notna(snapshot_metrics["MAPE (weekly)"])
                    else None
                ),
                "total_actual_cf": float(snapshot_metrics["Total actual cf"]),
                "total_forecast_cf": float(snapshot_metrics["Total forecast cf"]),
                "total_cf_difference": float(snapshot_metrics["Total cf difference"]),
            },
        })

    if not per_reference_results:
        if verbose:
            print("❌ No per-reference-date forecast results available")
        return {}, {}

    mae_values = [r["forecast_metrics"]["forecast_mae_weekly"] for r in per_reference_results]
    mape_values = [
        r["forecast_metrics"]["forecast_mape_weekly"]
        for r in per_reference_results
        if r["forecast_metrics"]["forecast_mape_weekly"] is not None
    ]
    total_actual_values = [r["forecast_metrics"]["total_actual_cf"] for r in per_reference_results]
    total_forecast_values = [r["forecast_metrics"]["total_forecast_cf"] for r in per_reference_results]
    total_diff_values = [r["forecast_metrics"]["total_cf_difference"] for r in per_reference_results]

    forecast_metrics = {
        "forecast_mae_weekly": float(np.mean(mae_values)),
        "forecast_mape_weekly": float(np.mean(mape_values)) if mape_values else np.nan,
        "forecast_total_actual_cf": float(np.mean(total_actual_values)),
        "forecast_total_forecast_cf": float(np.mean(total_forecast_values)),
        "forecast_total_cf_difference": float(np.mean(total_diff_values)),
    }

    forecast_summary = {
        "per_reference_date": per_reference_results,
        "aggregate": {
            "forecast_mae_weekly": forecast_metrics["forecast_mae_weekly"],
            "forecast_mape_weekly": (
                forecast_metrics["forecast_mape_weekly"]
                if pd.notna(forecast_metrics["forecast_mape_weekly"])
                else None
            ),
            "forecast_total_actual_cf": forecast_metrics["forecast_total_actual_cf"],
            "forecast_total_forecast_cf": forecast_metrics["forecast_total_forecast_cf"],
            "forecast_total_cf_difference": forecast_metrics["forecast_total_cf_difference"]
        },
    }

    if verbose:
        print(f"✅ Forecast MAE weekly: {forecast_metrics['forecast_mae_weekly']:.2f}")

        if pd.notna(forecast_metrics["forecast_mape_weekly"]):
            print(f"✅ Forecast MAPE weekly: {forecast_metrics['forecast_mape_weekly']:.4f}")
        else:
            print("✅ Forecast MAPE weekly: None")

        print(f"✅ Forecast total actual cf: {forecast_metrics['forecast_total_actual_cf']:.2f}")
        print(f"✅ Forecast total forecast cf: {forecast_metrics['forecast_total_forecast_cf']:.2f}")
        print(f"✅ Forecast total cf difference: {forecast_metrics['forecast_total_cf_difference']:.2f}")

    return forecast_metrics, forecast_summary


def _log_forecast_backtest_metrics(summary: dict) -> None:
    """Log aggregate forecast backtest metrics to MLflow."""
    if not summary or "aggregate" not in summary:
        return

    aggregate = summary["aggregate"]

    metrics = {
        "backtest_avg_forecast_mae_weekly": aggregate["backtest_avg_forecast_mae_weekly"],
    }

    if aggregate["backtest_avg_forecast_mape_weekly"] is not None:
        metrics["backtest_avg_forecast_mape_weekly"] = aggregate["backtest_avg_forecast_mape_weekly"]

    mlflow.log_metrics(metrics)


def simulate_forecast_backtest(pipeline: Pipeline, df: pd.DataFrame, log_to_mlflow: bool = False) -> dict:
    """
    Run a walk-forward backtest for forecast/business metrics (MAE / MAPE).

    For each cutoff:
    - train on snapshots up to cutoff
    - evaluate forecast quality on the next 6-week window
    - collect per-cutoff MAE / MAPE

    Returns:
        dict with:
            - per_cutoff
            - aggregate
    """
    if pipeline is None or len(df) == 0:
        return {}

    reference_dates = df["reference_date"].sort_values().unique()
    cutoffs = np.percentile(
        reference_dates.astype("datetime64[ns]").astype(np.int64),
        [40, 50, 60, 70, 80],
    )
    cutoffs = pd.to_datetime(cutoffs)

    results = []

    for cutoff in cutoffs:
        train_df = df[df["reference_date"] <= cutoff]
        test_df = df[
            (df["reference_date"] > cutoff) &
            (df["reference_date"] <= cutoff + pd.Timedelta(weeks=6))
        ]

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        X_train, y_train = preprocess(train_df)
        pipeline.fit(X_train, y_train)

        forecast_metrics, forecast_summary = evaluate_forecast_holdout(
            model=pipeline,
            test_df=test_df,
            verbose=False,
        )

        if not forecast_metrics:
            continue

        result = {
            "cutoff": cutoff.date().isoformat(),
            "forecast_mae_weekly": float(forecast_metrics["forecast_mae_weekly"]),
            "forecast_mape_weekly": (
                float(forecast_metrics["forecast_mape_weekly"])
                if pd.notna(forecast_metrics["forecast_mape_weekly"])
                else None
            ),
        }

        results.append(result)

        print(
            f"Cutoff {cutoff.date()} -> "
            f"forecast_mae: {result['forecast_mae_weekly']:.2f}, "
            f"forecast_mape: "
            f"{result['forecast_mape_weekly']:.4f}" if result["forecast_mape_weekly"] is not None
            else f"Cutoff {cutoff.date()} -> forecast_mae: {result['forecast_mae_weekly']:.2f}, forecast_mape: None"
        )

    if not results:
        return {}

    mae_values = [r["forecast_mae_weekly"] for r in results]
    mape_values = [
        r["forecast_mape_weekly"]
        for r in results
        if r["forecast_mape_weekly"] is not None
    ]

    summary = {
        "per_cutoff": results,
        "aggregate": {
            "backtest_avg_forecast_mae_weekly": float(np.mean(mae_values)),
            "backtest_avg_forecast_mape_weekly": (
                float(np.mean(mape_values)) if mape_values else None
            ),
            "n_cutoffs": int(len(results)),
        },
    }

    print("\nForecast backtest summary")
    print(summary["aggregate"])

    if log_to_mlflow:
        _log_forecast_backtest_metrics(summary)

    return summary
