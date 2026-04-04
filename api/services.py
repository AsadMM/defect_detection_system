import logging
import os
import pickle
import threading
import time
from typing import Any

import numpy as np
from src.models.autoencoder import build1
from src.models.threshold import get_results
from src.utils.visualization import convert_int, get_drawn_results


logger = logging.getLogger(__name__)
colors = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255)}
models: dict[str, Any] = {}
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


def load_model(name: str) -> None:
    logger.info("Loading model %s into memory", name)

    size = sizes[name]
    model_path = f"artifacts/models/model_{name}.h5"

    autoencoder = build1(size[0], size[0], size[1])
    autoencoder.load_weights(model_path)

    models[name] = autoencoder

    logger.info("Model %s loaded successfully", name)


def get_model(name: str):
    if name in models:
        logger.info("Cache hit for model %s", name)
        return models[name]

    logger.info("Cache miss for model %s", name)
    with _model_load_lock:
        if name in models:
            logger.info("Cache hit for model %s", name)
            return models[name]

        if name not in AVAILABLE_MODEL_NAMES:
            raise KeyError(f"Unknown model '{name}'")

        load_model(name)
        return models[name]


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
):
    logger.info("Running inference for model=%s", model_name)
    start = time.time()
    model = get_model(model_name)
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
    "load_model",
    "models",
    "predict_images",
    "run_inference",
    "sizes",
    "threshold_maps",
]
