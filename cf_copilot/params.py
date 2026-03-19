import os

LOCAL_REGISTRY_PATH =  os.environ.get("LOCAL_REGISTRY_PATH")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
GCS_MODEL_PREFIX = os.environ.get("GCS_MODEL_PREFIX")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
