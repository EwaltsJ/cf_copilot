import os

# 1) Where to store data and models locally
# For now we keep using your existing mlops/ folder at project root.

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOCAL_DATA_PATH = os.path.join(PROJECT_ROOT, "mlops", "data")
LOCAL_REGISTRY_PATH = os.path.join(PROJECT_ROOT, "mlops", "training_outputs")

# 2) Model target (later we can use "mlflow"; for now keep "local")
MODEL_TARGET = os.getenv("MODEL_TARGET", "local")
