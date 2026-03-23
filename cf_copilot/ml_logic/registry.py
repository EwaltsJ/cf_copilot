import os
import glob
import time
import pickle
import joblib
import json
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
    GCS_MODEL_PREFIX
)
from cf_copilot.ml_logic.data import data_cleaning, engineer_features
from cf_copilot.ml_logic.encoders import preprocess
from cf_copilot.ml_logic.reporting import make_json_serializable


def save_results(
    metrics: dict,
    figures: dict | None = None,
    artifacts: dict | None = None,
    json_artifacts: dict | None = None
) -> None:
    """
    Persist evaluation results locally and optionally to MLflow.

    Local:
    - metrics as pickle
    - figures as PNG
    - text artifacts as text files
    - JSON artifacts as JSON files

    MLflow:
    - scalar metrics via mlflow.log_metrics
    - optional figures via mlflow.log_figure

    JSON artifacts are stored locally only.
    """
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    metrics_dir = os.path.join(LOCAL_REGISTRY_PATH, "metrics")
    artifacts_dir = os.path.join(metrics_dir, "artifacts", timestamp)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    # 1. Save metrics locally
    if metrics is not None:
        metrics_path = os.path.join(metrics_dir, f"{timestamp}.pickle")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)
        print(f"✅ Metrics saved locally: {metrics_path}")

    # 2. Save figures locally
    if figures:
        for figure_name, fig in figures.items():
            figure_path = os.path.join(artifacts_dir, f"{figure_name}.png")
            fig.savefig(figure_path, dpi=150, bbox_inches="tight")
            print(f"✅ Figure saved locally: {figure_path}")

    # 3. Save text artifacts locally
    if artifacts:
        for artifact_name, content in artifacts.items():
            artifact_path = os.path.join(artifacts_dir, artifact_name)
            with open(artifact_path, "w", encoding="utf-8") as file:
                file.write(content)
            print(f"✅ Artifact saved locally: {artifact_path}")

    # 4. Save JSON artifacts locally
    if json_artifacts:
        for artifact_name, content in json_artifacts.items():
            serializable_content = make_json_serializable(content)
            artifact_path = os.path.join(artifacts_dir, artifact_name)
            with open(artifact_path, "w", encoding="utf-8") as file:
                json.dump(serializable_content, file, indent=2, ensure_ascii=False)
            print(f"✅ JSON artifact saved locally: {artifact_path}")

    # 5. Log metrics and figures to MLflow
    if MODEL_TARGET == "mlflow":
        try:
            if metrics:
                mlflow.log_metrics(metrics)

            if figures:
                for figure_name, fig in figures.items():
                    mlflow.log_figure(fig, f"evaluation/{figure_name}.png")

            if artifacts:
                for artifact_name, content in artifacts.items():
                    mlflow.log_text(content, f"evaluation/{artifact_name}")

            print("✅ Results logged to MLflow")
        except Exception as e:
            print("⚠️ Could not log results to MLflow")
            print(e)

    print("✅ Results persistence complete")


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
                    sk_model=model,
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
            model = mlflow.sklearn.load_model(model_uri)
            print("✅ Model loaded from MLflow")
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
