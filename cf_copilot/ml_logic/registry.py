import os
import glob
import time
import pickle
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from colorama import Fore, Style
from google.cloud import storage

from cf_copilot.params import (
    LOCAL_REGISTRY_PATH,
    MODEL_TARGET,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MLFLOW_MODEL_NAME,
    GCS_BUCKET_NAME,
    GCS_MODEL_PREFIX,
    GCS_HISTORICAL_DATA_PATH,
    LOCAL_HISTORICAL_DATA_PATH,
)
from cf_copilot.ml_logic.data import (
    data_cleaning,
    engineer_features,
    load_historical_data,
    append_to_historical_data,
)

from cf_copilot.ml_logic.encoders import preprocess

def save_results(metrics : dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    if MODEL_TARGET == "mlflow":
        if metrics is not None:
            try:
                mlflow.log_metrics(metrics)
                print("✅ Results saved on mlflow")
            except:
                print("⚠️ Could not log to MLflow")

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Save metrics locally
    if metrics is not None:
        metrics_dir = os.path.join(LOCAL_REGISTRY_PATH, "metrics")
        os.makedirs(metrics_dir,exist_ok=True)
        metrics_path = os.path.join(LOCAL_REGISTRY_PATH, "metrics", timestamp + ".pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    print("✅ Results saved locally")

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

    # ---------- Save to MLflow ----------
    if MODEL_TARGET == "mlflow":
        try:
            print("🌐 Uploading model to MLflow...")
            #mlflow.end_run()

            #with mlflow.start_run():
            mlflow.sklearn.log_model(
                    sk_model=model_path,
                    artifact_path="model",
                    registered_model_name=MLFLOW_MODEL_NAME
                )

            print("✅ Model successfully saved to MLflow")

        except Exception as e:
            print("⚠️ Could not save model to MLflow. Falling back to local only.")
            print(e)


def load_model(stage: str = "Production"):
    """Load MLflow model (or latest local joblib fallback)."""
    if MODEL_TARGET == "mlflow":
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        try:
            model_uri = f"models:/{MLFLOW_MODEL_NAME}/{stage}"
            mlflow_model = mlflow.sklearn.load_model(model_uri)
            print("✅ Model loaded from MLflow")
            model = joblib.load(mlflow_model)
            return model
        except Exception as e:
            print("❌ No model found in MLflow registry, falling back to local")
            print(e)
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
    if MODEL_TARGET == "local":
        # ---------- Local fallback ----------
        model_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")
        if not os.path.exists(model_dir):
            print("❌ No model directory found locally")
            return None

        model_paths = sorted(glob.glob(os.path.join(model_dir, "*.joblib")))
        if not model_paths:
            print("❌ No model found locally")
            return None

        latest_model_path = model_paths[-1]
        model = joblib.load(latest_model_path)
        print(f"✅ Model loaded from {latest_model_path}")
        return model
    else:
        return None

def mlflow_transition_model(current_stage: str, new_stage: str) -> None:
    """
    Transition the latest model from the `current_stage` to the
    `new_stage` and archive the existing model in `new_stage`
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient()

    version = client.get_latest_versions(name=MLFLOW_MODEL_NAME, stages=[current_stage])

    if not version:
        print(f"\n❌ No model found with name {MLFLOW_MODEL_NAME} in stage {current_stage}")
        return None

    client.transition_model_version_stage(
        name=MLFLOW_MODEL_NAME,
        version=version[0].version,
        stage=new_stage,
        archive_existing_versions=True
    )

    print(f"✅ Model {MLFLOW_MODEL_NAME} (version {version[0].version}) transitioned from {current_stage} to {new_stage}")

    return None

def mlflow_run(func):
    """
    Generic function to log params and results to MLflow

    Args:
        - func (function): Function you want to run within the MLflow run
        - params (dict, optional): Params to add to the run in MLflow. Defaults to None.
        - context (str, optional): Param describing the context of the run. Defaults to "Train".
    """
    def wrapper(*args, **kwargs):
        mlflow.end_run()
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name=MLFLOW_EXPERIMENT)

        with mlflow.start_run():
            results = func(*args, **kwargs)

        print("✅ mlflow_run auto-log done")

        return results
    return wrapper

def prepare_features(df: pd.DataFrame) -> tuple:
    """Clean raw invoice data and engineer features for prediction.

    Returns:
        Tuple of (X, cleaned_df) where X is the feature matrix
        and cleaned_df is the DataFrame after cleaning.
    """
    cleaned_df, _ = data_cleaning(df)
    current_date = pd.Timestamp.now()
    # get historical data
    historical_df = load_historical_data()
    featured_df = engineer_features(cleaned_df, historical_df, current_date)
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
    X,cleaned_df = prepare_features(df)
    preds = model.predict(X)
    probas = model.predict_proba(X)

    # updating historical data
    cleaned_df = cleaned_df.copy()
    cleaned_df["predicted_week_bucket"] = preds

    append_to_historical_data(cleaned_df)
    return {"week_bucket": preds, "probabilities": probas}
