import pandas as pd
import numpy as np

CLASS_NAMES = [1, 2, 3, 4, 5, 6, 7]
WEEK_CLASSES = [1, 2, 3, 4, 5, 6]

from cf_copilot.ml_logic.registry import predict

def build_prediction_table(
    base_probas: np.ndarray,
    invoices_df: pd.DataFrame,
    # invoice_id_col: str = "invoice_id",
) -> pd.DataFrame:
    """
    Attach model probabilities to invoice metadata in a clean table.
    """
    prob_df = pd.DataFrame(base_probas, columns=[f"p_{c}" for c in CLASS_NAMES])

    result = invoices_df[["total_open_amount"]].copy()
    result = result.reset_index(drop=True)
    prob_df = prob_df.reset_index(drop=True)

    result = pd.concat([result, prob_df], axis=1)
    return result

def aggregate_weekly_forecast(
    pred_cash_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregate invoice-level expected cash into one forecast row per week.
    """
    rows = []

    for week in WEEK_CLASSES:
        cash_col = f"expected_cash_{week}"
        total_cash = round(float(pred_cash_df[cash_col].sum()), 2)

        rows.append(
            {
                "week_bucket": week,
                "forecast_cash": total_cash,
            }
        )

    return pd.DataFrame(rows)

def add_expected_cash_columns(
    pred_df: pd.DataFrame
) -> pd.DataFrame:
    """
    For each week bucket, compute expected cash = amount * probability.
    """
    result = pred_df.copy()

    for week in WEEK_CLASSES:
        prob_col = f"p_{week}"
        cash_col = f"expected_cash_{week}"
        result[cash_col] = (result["total_open_amount"] * result[prob_col]).round(2)

    return result


def predict_cashflow(
    invoices_df: pd.DataFrame,
    pipeline = None,
) -> pd.DataFrame:

    df = invoices_df
    predictions = predict(pipeline, df)

    pred_df = build_prediction_table(predictions['probabilities'], df)
    pred_cash_df = add_expected_cash_columns(pred_df)

    df = aggregate_weekly_forecast(pred_cash_df)

    return df
