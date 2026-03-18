import os
import glob
import pickle
import time

from colorama import Fore, Style
from google.cloud import storage

from cf_copilot.params import LOCAL_REGISTRY_PATH, MODEL_TARGET, BUCKET_NAME, GCS_MODEL_PREFIX

def save_model(model=None) -> None:
    """Save the fitted pipeline to the target defined by MODEL_TARGET.

    Always saves locally first, then optionally uploads to GCS or MLflow.

    Args:
        model: a fitted sklearn Pipeline to persist.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"{timestamp}.pkl"
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", model_filename)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{GCS_MODEL_PREFIX}{model_filename}")
        blob.upload_from_filename(model_path)

        print(f"✅ Model saved to GCS: gs://{BUCKET_NAME}/{GCS_MODEL_PREFIX}{model_filename}")

def load_model(stage="Production"):
    """Load the latest saved pipeline from the target defined by MODEL_TARGET.

    Returns:
        The most recently saved sklearn Pipeline, or None if not found.
    """
    if MODEL_TARGET == "local":
        print(Fore.BLUE + "\nLoad latest model from local registry..." + Style.RESET_ALL)

        model_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")
        model_paths = glob.glob(f"{model_dir}/*.pkl")

        if not model_paths:
            print("❌ No model found")
            return None

        most_recent = sorted(model_paths)[-1]

        with open(most_recent, "rb") as f:
            model = pickle.load(f)

        print("✅ Model loaded from local disk")
        return model

    elif MODEL_TARGET == "gcs":
        print(Fore.BLUE + "\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.bucket(BUCKET_NAME).list_blobs(prefix=GCS_MODEL_PREFIX))
        pkl_blobs = [b for b in blobs if b.name.endswith(".pkl")]

        if not pkl_blobs:
            print(f"❌ No model found in gs://{BUCKET_NAME}/{GCS_MODEL_PREFIX}")
            return None

        latest_blob = max(pkl_blobs, key=lambda b: b.updated)

        local_tmp = os.path.join(LOCAL_REGISTRY_PATH, "models", "latest.pkl")
        os.makedirs(os.path.dirname(local_tmp), exist_ok=True)
        latest_blob.download_to_filename(local_tmp)

        with open(local_tmp, "rb") as f:
            model = pickle.load(f)

        print(f"✅ Model loaded from gs://{BUCKET_NAME}/{latest_blob.name}")
        return model
    else:
        print(f"❌ Unknown MODEL_TARGET: {MODEL_TARGET}")
        return None


def predict(model, X_new) -> dict:
    """Return predicted week buckets and probabilities.

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
