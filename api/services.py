import logging
import os
import pickle
import threading
import time
from typing import Any

import numpy as np
try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency at runtime
    mlflow = None

try:
    import mlflow.keras as mlflow_keras
except Exception:  # pragma: no cover - optional dependency at runtime
    mlflow_keras = None

from src.models.autoencoder import build1
from src.models.threshold import get_results
from src.utils.visualization import convert_int, get_drawn_results


logger = logging.getLogger(__name__)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///artifacts/mlflow/mlflow.db")
if mlflow is not None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    logger.info("MLflow tracking URI set to %s", MLFLOW_TRACKING_URI)

colors = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255)}
models: dict[tuple[str, str], Any] = {}
sizes: dict[str, tuple[int, int]] = {}
threshold_maps: dict[str, dict[float, float]] = {}
AVAILABLE_MODEL_NAMES: set[str] = set()
_model_load_lock = threading.Lock()


def load_metadata() -> None:
    model_dir = "artifacts/models"

    AVAILABLE_MODEL_NAMES.clear()
    sizes.clear()
    threshold_maps.clear()

    if not os.path.isdir(model_dir):
        logger.warning("Model directory does not exist: %s", model_dir)
        return

    for filename in os.listdir(model_dir):
        if filename.startswith("model_") and filename.endswith(".h5"):
            name = filename.replace("model_", "").replace(".h5", "")
            size_path = f"artifacts/sizes/sizes_{name}.pkl"
            threshold_path = f"artifacts/thresholds/thresholds_{name}.pkl"

            missing = []
            if not os.path.isfile(size_path):
                missing.append(size_path)
            if not os.path.isfile(threshold_path):
                missing.append(threshold_path)

            if missing:
                logger.warning("Skipping model metadata for %s. Missing: %s", name, ", ".join(missing))
                continue

            with open(size_path, "rb") as file:
                sizes[name] = pickle.load(file)

            with open(threshold_path, "rb") as file:
                threshold_maps[name] = pickle.load(file)

            AVAILABLE_MODEL_NAMES.add(name)

    logger.info("Loaded metadata for models: %s", sorted(AVAILABLE_MODEL_NAMES))


def _build_cache_key(name: str, version: int | None = None, stage: str = "Production") -> tuple[str, str]:
    if version is not None:
        return name, f"v{version}"
    return name, stage


def load_local_model(name: str):
    logger.info("Loading model %s from local artifacts", name)
    size = sizes[name]
    model_path = f"artifacts/models/model_{name}.h5"

    autoencoder = build1(size[0], size[0], size[1])
    autoencoder.load_weights(model_path)

    logger.info("Model %s loaded from local file", name)
    return autoencoder


def load_model(name: str, version: int | None = None, stage: str = "Production") -> None:
    cache_key = _build_cache_key(name, version, stage)
    if version is not None:
        model_uri = f"models:/{name}/{version}"
    else:
        model_uri = f"models:/{name}/{stage}"

    if mlflow_keras is None:
        logger.warning(
            "MLflow is not available, using local fallback for model %s (%s)",
            name,
            cache_key[1],
        )
        models[cache_key] = load_local_model(name)
        return

    try:
        logger.info("Loading model %s from MLflow (%s)", name, model_uri)
        model = mlflow_keras.load_model(model_uri)
        models[cache_key] = model
        logger.info("Model %s loaded from MLflow (%s)", name, model_uri)
    except Exception as exc:
        logger.warning(
            "MLflow load failed for model %s (%s), falling back. Error: %s",
            name,
            model_uri,
            str(exc),
        )
        models[cache_key] = load_local_model(name)


def get_model(name: str, version: int | None = None, stage: str = "Production"):
    cache_key = _build_cache_key(name, version, stage)
    if cache_key in models:
        logger.info("Cache hit for model %s (%s)", name, cache_key[1])
        return models[cache_key]

    logger.info("Cache miss for model %s (%s)", name, cache_key[1])
    with _model_load_lock:
        if cache_key in models:
            logger.info("Cache hit for model %s (%s)", name, cache_key[1])
            return models[cache_key]

        if name not in AVAILABLE_MODEL_NAMES:
            raise KeyError(f"Unknown model '{name}'")

        load_model(name, version=version, stage=stage)
        return models[cache_key]


def predict_images(images, model_name: str):
    return get_model(model_name).predict(images)


def get_threshold_value(model_name: str, threshold: float) -> float:
    return threshold_maps[model_name][round(threshold, 1)]


def get_model_context(model_name: str, threshold: float, redraw_color: str):
    img_size = sizes[model_name]
    threshold_value = get_threshold_value(model_name, threshold)
    color = colors[redraw_color]
    return img_size, threshold_value, color


def run_inference(
    images: np.ndarray,
    model_name: str,
    threshold_value: float,
    output_format: str,
    color: tuple,
    version: int | None = None,
    stage: str = "Production",
):
    logger.info("Running inference for model=%s version=%s stage=%s", model_name, version, stage)
    start = time.time()
    model = get_model(model_name, version=version, stage=stage)
    predicted_images = model.predict(images)
    masked_images = get_results(images, predicted_images, threshold_value)
    if output_format == "mask":
        output = convert_int(masked_images)
        logger.info("Inference completed in %.3f sec", time.time() - start)
        return output
    output, _ = get_drawn_results(images, masked_images, color)
    logger.info("Inference completed in %.3f sec", time.time() - start)
    return output


__all__ = [
    "AVAILABLE_MODEL_NAMES",
    "colors",
    "convert_int",
    "get_model",
    "get_model_context",
    "get_threshold_value",
    "load_metadata",
    "load_local_model",
    "load_model",
    "models",
    "predict_images",
    "run_inference",
    "sizes",
    "threshold_maps",
]
