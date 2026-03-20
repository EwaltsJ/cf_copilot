import os

#LOCAL_DATA_PATH = os.path.join(os.path.expanduser('~'), ".lewagon", "mlops", "data")
LOCAL_REGISTRY_PATH = os.environ.get("LOCAL_REGISTRY_PATH")
API_URL = os.environ.get("API_URL", "http://localhost:8080")

##################  VARIABLES  ##################
MODEL_TARGET = os.environ.get("MODEL_TARGET")
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")
EVALUATION_START_DATE = os.environ.get("EVALUATION_START_DATE")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCS_MODEL_PREFIX = os.environ.get("GCS_MODEL_PREFIX")
