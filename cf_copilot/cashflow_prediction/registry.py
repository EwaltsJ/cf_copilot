import pandas as pd
import numpy as np

from cf_copilot.ml_logic.registry import predict

WEEK_CLASSES = [1, 2, 3, 4, 5, 6]
CLASS_NAMES = [1, 2, 3, 4, 5, 6, 7]

def predict_cashflow(invoices_df: pd.DataFrame, pipeline) -> pd.DataFrame:
    """End-to-end cashflow forecast from raw invoices.

    Args:
        invoices_df: raw invoice DataFrame.
        pipeline: fitted sklearn Pipeline.

    Returns:
        DataFrame with weekly forecast_cash per bucket.
    """
    results = predict(pipeline, invoices_df)

    pred_df = _build_prediction_table(results["probabilities"], invoices_df)
    pred_df = _add_expected_cash_columns(pred_df)

    return _aggregate_weekly_forecast(pred_df)


def sharpen_probabilities(probas: np.ndarray, temperature: float = 0.7) -> np.ndarray:
    """
    Apply temperature scaling to sharpen probabilities.
    Lower temperature (<1) = sharper distribution.
    """
    probas = np.clip(probas, 1e-12, 1.0)
    probas = probas ** (1 / temperature)
    probas = probas / probas.sum(axis=1, keepdims=True)
    return probas


def _build_prediction_table(probas: np.ndarray, invoices_df: pd.DataFrame) -> pd.DataFrame:
    """Attach model probabilities to invoice amounts."""
    probas = sharpen_probabilities(probas, temperature=0.8)

    prob_df = pd.DataFrame(probas, columns=[f"p_{c}" for c in CLASS_NAMES])

    result = invoices_df[["total_open_amount"]].reset_index(drop=True)
    return pd.concat([result, prob_df], axis=1)


def _add_expected_cash_columns(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Compute expected cash = amount * probability for each week."""
    result = pred_df.copy()
    for week in WEEK_CLASSES:
        result[f"expected_cash_{week}"] = (result["total_open_amount"] * result[f"p_{week}"]).round(2)
    return result


def _aggregate_weekly_forecast(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Sum expected cash across all invoices per week bucket."""
    return pd.DataFrame([
        {
            "week_bucket": week,
            "forecast_cash": round(float(pred_df[f"expected_cash_{week}"].sum()), 2),
        }
        for week in WEEK_CLASSES
    ])
