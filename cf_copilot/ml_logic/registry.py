import os
import pickle
import time

from cf_copilot.params import LOCAL_REGISTRY_PATH

def save_model(model=None) -> None:
    """Save the fitted pipeline (preprocessor + classifier) locally.

    Args:
        model: a fitted sklearn Pipeline to persist.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", f"{timestamp}.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Pipeline saved locally to {model_path}")


def load_model():
    """Load the latest saved pipeline from disk.

    Returns:
        The most recently saved sklearn Pipeline, or None if not found.
    """
    model_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")

    if not os.path.exists(model_dir):
        print("❌ No model directory found")
        return None

    model_paths = sorted(
        [os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".pkl")]
    )

    if not model_paths:
        print("❌ No model found")
        return None

    with open(model_paths[-1], "rb") as f:
        model = pickle.load(f)

    print("✅ Pipeline loaded from disk")
    return model


def predict(model, X_new) -> dict:
    """Return predicted week buckets and probabilities.

    No separate preprocessing needed — the pipeline handles it.

    Args:
        model: a fitted sklearn Pipeline.
        X_new: DataFrame of features for new invoices.

    Returns:
        A dict with 'week_bucket' (predictions) and 'probabilities'.
    """
    preds = model.predict(X_new)
    probas = model.predict_proba(X_new)

    print(f"✅ Predictions made for {len(X_new)} invoices")
    return {"week_bucket": preds, "probabilities": probas}
