import os
import glob
import time
import pickle
import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from cf_copilot.params import (
    LOCAL_REGISTRY_PATH,
    MODEL_TARGET,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MLFLOW_MODEL_NAME,
)

def save_results(metrics : dict) -> None:
    """
    Persist params & metrics locally on the hard drive at
    "{LOCAL_REGISTRY_PATH}/metrics/{current_timestamp}.pickle"
    - (unit 03 only) if MODEL_TARGET='mlflow', also persist them on MLflow
    """
    if MODEL_TARGET == "mlflow":
        if metrics is not None:
            mlflow.log_metrics(metrics)
        print("✅ Results saved on mlflow")

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
            mlflow.end_run()

            with mlflow.start_run():
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
            model = mlflow.sklearn.load_model(model_uri)
            print("✅ Model loaded from MLflow")
            return model
        except Exception as e:
            print("❌ No model found in MLflow registry, falling back to local")
            print(e)

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

def predict(model, X_new) -> dict:
    print("🔮 Generating predictions...")
    preds = model.predict(X_new)
    probas = model.predict_proba(X_new)
    print(f"✅ Predictions made for {len(X_new)} rows")
    return {"week_bucket": preds, "probabilities": probas}

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
