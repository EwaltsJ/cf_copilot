import os
from pathlib import Path
import pandas as pd

BASE_DIR = Path(os.environ.get("BASE_DIR", Path(__file__).resolve().parents[1]))

PLAYBOOK_PATH = Path(os.environ.get("PLAYBOOK_PATH", BASE_DIR / "data" / "playbook"))
CHROMA_PATH = Path(os.environ.get("CHROMA_PATH", BASE_DIR / "data" / "chroma_db"))

LOCAL_REGISTRY_PATH = os.environ.get("LOCAL_REGISTRY_PATH", "raw_data")
LOCAL_HISTORICAL_DATA_PATH = os.path.join(LOCAL_REGISTRY_PATH, "data", "historical.csv")

GCP_PROJECT = os.environ.get("GCP_PROJECT_ID")
GCP_REGION = os.environ.get("GCP_REGION", "europe-west1")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")
GCS_MODEL_PREFIX = os.environ.get("GCS_MODEL_PREFIX", "cf_copilot")
GCS_HISTORICAL_DATA_PATH = os.environ.get("GCS_HISTORICAL_DATA_PATH", "data/historical.csv")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME")

MODEL_TARGET = os.environ.get("MODEL_TARGET", "local")
API_URL = os.environ.get("API_URL", "http://localhost:8080")
ENV = os.environ.get("ENV", "staging")

CURRENT_DATE = pd.to_datetime(os.environ.get("CURRENT_DATE", "2020-05-22"))
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
