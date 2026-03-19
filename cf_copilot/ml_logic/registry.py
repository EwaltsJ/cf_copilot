import os
import glob
import time
import logging
import requests

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from dotenv import load_dotenv
load_dotenv()

from cf_copilot.params import (
    LOCAL_REGISTRY_PATH,
    MODEL_TARGET,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MLFLOW_MODEL_NAME,
)

logging.getLogger("mlflow").setLevel(logging.INFO)

# ONLY FOR TESTING SSL ISSUES (skip in prod)
session = requests.Session()
session.verify = False
client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI, registry_uri=MLFLOW_TRACKING_URI)


def _ensure_experiment():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    exp = client.get_experiment_by_name(MLFLOW_EXPERIMENT)

    if exp is None:
        client.create_experiment(MLFLOW_EXPERIMENT)
        print(f"🆕 Created MLflow experiment: {MLFLOW_EXPERIMENT}")
    elif exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
        print(f"♻️ Restored deleted experiment: {MLFLOW_EXPERIMENT}")

    mlflow.set_experiment(MLFLOW_EXPERIMENT)


def save_model(model=None) -> None:
    """Save model locally (joblib) and optionally to MLflow."""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_dir = os.path.join(LOCAL_REGISTRY_PATH, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{timestamp}.joblib")

    # ---------- Save locally ----------
    print("💾 Saving model locally...")
    joblib.dump(model, model_path, compress=True)
    print(f"✅ Pipeline saved locally to {model_path}")

    # ---------- Save to MLflow ----------
    if MODEL_TARGET == "mlflow":
        try:
            print("🌐 Uploading model to MLflow...")
            _ensure_experiment()
            mlflow.end_run()  # safely close any previous run

            with mlflow.start_run(run_name="training_run"):
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    registered_model_name=MLFLOW_MODEL_NAME,
                    pip_requirements=[],  # skip env capture
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


def predict(model, X_new) -> dict:
    print("🔮 Generating predictions...")
    preds = model.predict(X_new)
    probas = model.predict_proba(X_new)
    print(f"✅ Predictions made for {len(X_new)} rows")
    return {"week_bucket": preds, "probabilities": probas}
