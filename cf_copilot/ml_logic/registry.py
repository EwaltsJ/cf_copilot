import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sklearn.pipeline import Pipeline

from cf_copilot.ml_logic.params import LOCAL_REGISTRY_PATH, MODEL_TARGET


def _get_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _get_registry_path() -> Path:
    """
    Base registry path, e.g. <project_root>/mlops/training_outputs.
    """
    return Path(LOCAL_REGISTRY_PATH)


def save_results(params: Dict[str, Any], metrics: Dict[str, Any]) -> None:
    """
    Save params & metrics locally as pickle files.

    {LOCAL_REGISTRY_PATH}/params/{timestamp}.pickle
    {LOCAL_REGISTRY_PATH}/metrics/{timestamp}.pickle
    """
    registry_path = _get_registry_path()
    params_dir = registry_path / "params"
    metrics_dir = registry_path / "metrics"

    params_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    ts = _get_timestamp()

    with open(params_dir / f"{ts}.pickle", "wb") as f:
        pickle.dump(params, f)

    with open(metrics_dir / f"{ts}.pickle", "wb") as f:
        pickle.dump(metrics, f)


def save_model(model: Pipeline) -> None:
    """
    Save trained sklearn pipeline locally as

    {LOCAL_REGISTRY_PATH}/models/{timestamp}.pkl
    """
    registry_path = _get_registry_path()
    models_dir = registry_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    ts = _get_timestamp()
    model_path = models_dir / f"{ts}.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)


def _get_latest_model_path() -> Optional[Path]:
    registry_path = _get_registry_path()
    models_dir = registry_path / "models"

    if not models_dir.exists():
        return None

    candidates = sorted(models_dir.glob("*.pkl"))
    if not candidates:
        return None

    # Latest by filename (timestamp in name)
    return candidates[-1]


def load_model(stage: str = "Production") -> Optional[Pipeline]:
    """
    Load the latest local model. Ignore stage for now.

    Returns None if no model exists.
    """
    model_path = _get_latest_model_path()
    if model_path is None:
        return None

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model
