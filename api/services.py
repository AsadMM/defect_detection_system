import logging
import os
import pickle
import time
from glob import glob
from typing import Any

import numpy as np
from src.models.autoencoder import build1
from src.utils.visualization import convert_int, get_drawn_results
from src.models.threshold import get_results


logger = logging.getLogger(__name__)
colors = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255)}
models: dict[str, Any] = {}
sizes: dict[str, tuple[int, int]] = {}
threshold_maps: dict[str, dict[float, float]] = {}


def load_artifacts() -> set[str]:
    names: set[str] = set()
    model_paths = sorted(glob(os.path.join("artifacts", "models", "model_*.h5")))
    size_paths = sorted(glob(os.path.join("artifacts", "sizes", "sizes_*.pkl")))
    threshold_paths = sorted(glob(os.path.join("artifacts", "thresholds", "thresholds_*.pkl")))

    model_map = {
        os.path.basename(path).replace("model_", "").replace(".h5", ""): path
        for path in model_paths
    }
    size_map = {
        os.path.basename(path).replace("sizes_", "").replace(".pkl", ""): path
        for path in size_paths
    }
    threshold_map = {
        os.path.basename(path).replace("thresholds_", "").replace(".pkl", ""): path
        for path in threshold_paths
    }

    all_names = sorted(set(model_map) | set(size_map) | set(threshold_map))
    for name in all_names:
        missing = []
        if name not in model_map:
            missing.append("model")
        if name not in size_map:
            missing.append("size")
        if name not in threshold_map:
            missing.append("threshold")

        if missing:
            logger.warning("Skipping artifact '%s': missing %s file(s)", name, ", ".join(missing))
            continue

        logger.info("Loading model %s", name)

        with open(size_map[name], "rb") as file:
            size = pickle.load(file)
            sizes[name] = size

        with open(threshold_map[name], "rb") as file:
            threshold_values = pickle.load(file)
            threshold_maps[name] = threshold_values

        autoencoder = build1(size[0], size[0], size[1])
        autoencoder.load_weights(model_map[name])
        models[name] = autoencoder
        names.add(name)
        logger.info("Model %s loaded successfully", name)
        logger.info(
            "Model %s artifacts paths: model=%s size=%s threshold=%s",
            name,
            model_map[name],
            size_map[name],
            threshold_map[name],
        )
        logger.info("Model %s input size: %s", name, size)

    return names


def predict_images(images, model_name: str):
    return models[model_name].predict(images)


def get_threshold_value(model_name: str, threshold: float) -> float:
    return threshold_maps[model_name][round(threshold, 1)]


def get_model_context(model_name: str, threshold: float, redraw_color: str):
    img_size = sizes[model_name]
    threshold_value = get_threshold_value(model_name, threshold)
    color = colors[redraw_color]
    return img_size, threshold_value, color


def run_inference(
    images: np.ndarray,
    model,
    model_name: str,
    threshold_value: float,
    output_format: str,
    color: tuple,
):
    logger.info("Running inference for model=%s", model_name)
    start = time.time()
    predicted_images = model.predict(images)
    masked_images = get_results(images, predicted_images, threshold_value)
    if output_format == "mask":
        output = convert_int(masked_images)
        logger.info("Inference completed in %.3f sec", time.time() - start)
        return output
    output, _ = get_drawn_results(images, masked_images, color)
    logger.info("Inference completed in %.3f sec", time.time() - start)
    return output


AVAILABLE_MODEL_NAMES = load_artifacts()


__all__ = [
    "AVAILABLE_MODEL_NAMES",
    "colors",
    "convert_int",
    "get_model_context",
    "get_threshold_value",
    "load_artifacts",
    "models",
    "predict_images",
    "run_inference",
    "sizes",
]
