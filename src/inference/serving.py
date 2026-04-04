import logging
import os
import pickle
import threading
import time
from typing import Any

import numpy as np

from src.inference.exceptions import ModelMetadataError, UnknownModelError

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


class ModelRegistryService:
    def __init__(self):
        self.models: dict[tuple[str, str], Any] = {}
        self.sizes: dict[str, tuple[int, int]] = {}
        self.threshold_maps: dict[str, dict[float, float]] = {}
        self.available_models: set[str] = set()
        self._load_lock = threading.Lock()

        if mlflow is not None:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            logger.info("MLflow tracking URI set to %s", MLFLOW_TRACKING_URI)

    def load_metadata(self) -> None:
        self.available_models.clear()
        self.sizes.clear()
        self.threshold_maps.clear()

        model_dir = "artifacts/models"
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
                    self.sizes[name] = pickle.load(file)

                with open(threshold_path, "rb") as file:
                    self.threshold_maps[name] = pickle.load(file)

                self.available_models.add(name)

        logger.info("Loaded metadata for models: %s", sorted(self.available_models))

    def _normalize_stage(self, stage: str) -> str:
        stage_map = {
            "production": "Production",
            "staging": "Staging",
            "archived": "Archived",
        }
        return stage_map.get(stage.strip().lower(), stage.strip())

    def _get_cache_key(self, name: str, version: int | None, stage: str) -> tuple[str, str]:
        if version is not None:
            return name, f"v{version}"
        normalized_stage = self._normalize_stage(stage)
        return name, normalized_stage

    def _load_local_model(self, name: str):
        logger.info("Loading %s from local artifacts", name)

        size = self.sizes[name]
        model_path = f"artifacts/models/model_{name}.h5"

        model = build1(size[0], size[0], size[1])
        model.load_weights(model_path)

        logger.info("Model %s loaded from local file", name)
        return model

    def _load_model(self, name: str, version: int | None = None, stage: str = "Production"):
        normalized_stage = self._normalize_stage(stage)
        if version is not None:
            model_uri = f"models:/{name}/{version}"
        else:
            model_uri = f"models:/{name}/{normalized_stage}"

        if mlflow_keras is None:
            logger.warning("MLflow is unavailable for %s (%s), using local fallback", name, model_uri)
            return self._load_local_model(name)

        try:
            logger.info("Loading %s from MLflow (%s)", name, model_uri)
            return mlflow_keras.load_model(model_uri)
        except Exception as exc:
            logger.warning("MLflow failed for %s (%s): %s", name, model_uri, str(exc))
            return self._load_local_model(name)

    def get_model(self, name: str, version: int | None = None, stage: str = "Production"):
        key = self._get_cache_key(name, version, stage)

        if key in self.models:
            logger.info("Cache hit for %s (%s)", name, key[1])
            return self.models[key]

        logger.info("Cache miss for %s (%s)", name, key[1])
        with self._load_lock:
            if key in self.models:
                logger.info("Cache hit for %s (%s)", name, key[1])
                return self.models[key]

            if name not in self.available_models:
                raise UnknownModelError(f"Unknown model '{name}'")

            model = self._load_model(name, version, stage)
            self.models[key] = model
            return model

    def get_threshold_value(self, model_name: str, threshold: float) -> float:
        rounded_threshold = round(threshold, 1)
        model_thresholds = self.threshold_maps.get(model_name)
        if model_thresholds is None:
            raise ModelMetadataError(f"Threshold metadata missing for model '{model_name}'")
        if rounded_threshold not in model_thresholds:
            raise ModelMetadataError(
                f"Threshold metadata missing percentile {rounded_threshold} for model '{model_name}'"
            )
        return model_thresholds[rounded_threshold]

    def get_model_context(self, model_name: str, threshold: float, redraw_color: str):
        colors = {"blue": (255, 0, 0), "green": (0, 255, 0), "red": (0, 0, 255)}
        img_size = self.sizes.get(model_name)
        if img_size is None:
            raise ModelMetadataError(f"Size metadata missing for model '{model_name}'")
        threshold_value = self.get_threshold_value(model_name, threshold)
        if redraw_color not in colors:
            raise ModelMetadataError(f"Unsupported redraw color '{redraw_color}'")
        color = colors[redraw_color]
        return img_size, threshold_value, color


model_registry = ModelRegistryService()


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


__all__ = [
    "model_registry",
    "run_inference",
]
