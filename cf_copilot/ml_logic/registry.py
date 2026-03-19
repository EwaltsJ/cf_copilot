import os
import glob
import time

import joblib
import pandas as pd
from colorama import Fore, Style
from google.cloud import storage

from cf_copilot.params import LOCAL_REGISTRY_PATH, MODEL_TARGET, GCS_BUCKET_NAME, GCS_MODEL_PREFIX
from cf_copilot.ml_logic.data import data_cleaning, engineer_features
from cf_copilot.ml_logic.encoders import preprocess


def save_model(model=None) -> None:
    """Save the fitted pipeline locally, and optionally to GCS."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_filename = f"{timestamp}.joblib"
    model_path = os.path.join(LOCAL_REGISTRY_PATH, "models", model_filename)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    joblib.dump(model, model_path, compress=3)

    print("✅ Model saved locally")

    if MODEL_TARGET == "gcs":
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(f"{GCS_MODEL_PREFIX}{model_filename}")
        blob.upload_from_filename(model_path)
        print(f"✅ Model saved to GCS: gs://{GCS_BUCKET_NAME}/{GCS_MODEL_PREFIX}{model_filename}")


def load_model():
    """Load the latest saved pipeline from local disk or GCS."""
    if MODEL_TARGET == "gcs":
        print(Fore.BLUE + "\nLoad latest model from GCS..." + Style.RESET_ALL)

        client = storage.Client()
        blobs = list(client.bucket(GCS_BUCKET_NAME).list_blobs(prefix=GCS_MODEL_PREFIX))
        model_blobs = [b for b in blobs if b.name.endswith(".joblib")]

        if not model_blobs:
            print(f"❌ No model found in gs://{GCS_BUCKET_NAME}/{GCS_MODEL_PREFIX}")
            return None

        latest_blob = max(model_blobs, key=lambda b: b.updated)
        local_tmp = os.path.join(LOCAL_REGISTRY_PATH, "models", "latest.joblib")
        os.makedirs(os.path.dirname(local_tmp), exist_ok=True)
        latest_blob.download_to_filename(local_tmp)

        model = joblib.load(local_tmp)

        print(f"✅ Model loaded from gs://{GCS_BUCKET_NAME}/{latest_blob.name}")
        return model

    # Default: local
    print(Fore.BLUE + "\nLoad latest model from local registry..." + Style.RESET_ALL)

    model_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")
    model_paths = glob.glob(f"{model_dir}/*.joblib")

    if not model_paths:
        print("❌ No model found")
        return None

    most_recent = sorted(model_paths)[-1]
    model = joblib.load(most_recent)

    print("✅ Model loaded from local disk")
    return model


def prepare_features(df: pd.DataFrame) -> tuple:
    """Clean raw invoice data and engineer features for prediction.

    Returns:
        Tuple of (X, cleaned_df) where X is the feature matrix
        and cleaned_df is the DataFrame after cleaning.
    """
    cleaned_df, _ = data_cleaning(df)
    current_date = pd.Timestamp.now()
    # TODO: Accept a historical df for customer behaviour features
    featured_df = engineer_features(cleaned_df, cleaned_df, current_date)
    X, _ = preprocess(featured_df, inference=True)
    return X, cleaned_df


def predict(model, df: pd.DataFrame) -> dict:
    """Clean, engineer features, and return predictions.

    Args:
        model: a fitted sklearn Pipeline.
        df: raw invoice DataFrame.

    Returns:
        A dict with 'week_bucket' (predictions) and 'probabilities'.
    """
    X, _ = prepare_features(df)
    preds = model.predict(X)
    probas = model.predict_proba(X)
    return {"week_bucket": preds, "probabilities": probas}
