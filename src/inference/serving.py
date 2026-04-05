import logging
import os
import pickle
import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np
from tensorflow import keras

from src.inference.exceptions import ModelMetadataError, UnknownModelError

try:
    import mlflow
except Exception:  # pragma: no cover - optional dependency at runtime
    mlflow = None

try:
    import mlflow.keras as mlflow_keras
except Exception:  # pragma: no cover - optional dependency at runtime
    mlflow_keras = None

from src.models.threshold import get_results
from src.utils.visualization import convert_int, get_drawn_results


logger = logging.getLogger(__name__)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///artifacts/mlflow/mlflow.db")
MODEL_FILE_EXTENSION = ".keras"


def _get_positive_int_env(name: str, default: int) -> int:
    """Read a positive integer env var with safe fallback behavior."""
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


class ModelRegistryService:
    """Loads model metadata, resolves model sources, and manages an in-memory LRU model cache."""

    def __init__(self):
        """Initialize cache state and configure MLflow tracking URI (if available)."""
        self.models: OrderedDict[tuple[str, str], Any] = OrderedDict()
        self.model_cache_bytes: dict[tuple[str, str], int] = {}
        self.model_cache_total_bytes = 0
        self.max_model_cache_bytes = _get_positive_int_env("MODEL_CACHE_MAX_BYTES", 1_073_741_824)
        self.sizes: dict[str, tuple[int, int]] = {}
        self.threshold_maps: dict[str, dict[float, float]] = {}
        self.available_models: set[str] = set()
        self._load_lock = threading.Lock()

        if mlflow is not None:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            logger.info("MLflow tracking URI set to %s", MLFLOW_TRACKING_URI)
        logger.info("Model cache max size set to %.2f MB", self.max_model_cache_bytes / (1024 * 1024))

    def _estimate_model_size_bytes(self, model: Any) -> int:
        """Estimate model memory footprint in bytes for cache accounting."""
        # Approximate model memory usage from parameters.
        try:
            params = int(model.count_params())
            if params > 0:
                return params * 4  # float32 approximation
        except Exception:
            pass

        # Fallback for uncommon model wrappers.
        try:
            total = 0
            for var in model.weights:
                num_elements = var.shape.num_elements()
                dtype_size = int(getattr(var.dtype, "size", 4))
                if num_elements is not None:
                    total += int(num_elements) * dtype_size
            return int(total)
        except Exception:
            return 0

    def _evict_until_capacity(self, incoming_size: int) -> None:
        """Evict least-recently-used cached models until capacity is sufficient."""
        while self.model_cache_total_bytes + incoming_size > self.max_model_cache_bytes and self.models:
            evicted_key, _ = self.models.popitem(last=False)
            evicted_size = self.model_cache_bytes.pop(evicted_key, 0)
            self.model_cache_total_bytes -= evicted_size
            logger.info(
                "Evicted model cache entry %s (~%.2f MB). Current cache size %.2f MB",
                evicted_key,
                evicted_size / (1024 * 1024),
                self.model_cache_total_bytes / (1024 * 1024),
            )

    def _try_cache_model(self, key: tuple[str, str], model: Any) -> None:
        """Cache a loaded model when size can be estimated and budget allows."""
        size_bytes = self._estimate_model_size_bytes(model)
        if size_bytes <= 0:
            logger.warning("Model size could not be estimated for %s; model will not be cached", key)
            return

        if size_bytes > self.max_model_cache_bytes:
            logger.warning(
                "Model %s estimated at %.2f MB exceeds cache budget %.2f MB; model will not be cached",
                key,
                size_bytes / (1024 * 1024),
                self.max_model_cache_bytes / (1024 * 1024),
            )
            return

        self._evict_until_capacity(size_bytes)
        self.models[key] = model
        self.models.move_to_end(key)
        self.model_cache_bytes[key] = size_bytes
        self.model_cache_total_bytes += size_bytes
        logger.info(
            "Cached model %s (~%.2f MB). Current cache size %.2f MB",
            key,
            size_bytes / (1024 * 1024),
            self.model_cache_total_bytes / (1024 * 1024),
        )

    def load_metadata(self) -> None:
        """Load model availability, input sizes, and threshold maps from local artifacts."""
        self.available_models.clear()
        self.sizes.clear()
        self.threshold_maps.clear()

        model_dir = "artifacts/models"
        if not os.path.isdir(model_dir):
            logger.warning("Model directory does not exist: %s", model_dir)
            return

        for filename in os.listdir(model_dir):
            if filename.startswith("model_") and filename.endswith(MODEL_FILE_EXTENSION):
                name = filename[len("model_") :].rsplit(".", 1)[0]
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

    def _resolve_local_model_path(self, name: str) -> str:
        """Return expected local model file path or raise if missing."""
        candidate = f"artifacts/models/model_{name}{MODEL_FILE_EXTENSION}"
        if os.path.isfile(candidate):
            return candidate
        raise FileNotFoundError(f"No local model artifact found for '{name}' at {candidate}")

    def _normalize_stage(self, stage: str) -> str:
        """Normalize user-provided stage string to MLflow canonical stage names."""
        stage_map = {
            "production": "Production",
            "staging": "Staging",
            "archived": "Archived",
        }
        return stage_map.get(stage.strip().lower(), stage.strip())

    def _get_cache_key(self, name: str, version: int | None, stage: str) -> tuple[str, str]:
        """Build a cache key using model name and either version or normalized stage."""
        if version is not None:
            return name, f"v{version}"
        normalized_stage = self._normalize_stage(stage)
        return name, normalized_stage

    def _load_local_model(self, name: str):
        """Load model from local `artifacts/models` fallback location."""
        logger.info("Loading %s from local artifacts", name)
        model_path = self._resolve_local_model_path(name)
        model = keras.models.load_model(model_path)

        logger.info("Model %s loaded from local file", name)
        return model

    def _load_model(self, name: str, version: int | None = None, stage: str = "Production"):
        """Load model from MLflow first, then fall back to local artifacts on failure."""
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
        """Return cached model or load-and-cache it for the requested selector."""
        key = self._get_cache_key(name, version, stage)

        if key in self.models:
            self.models.move_to_end(key)
            logger.info("Cache hit for %s (%s)", name, key[1])
            return self.models[key]

        logger.info("Cache miss for %s (%s)", name, key[1])
        with self._load_lock:
            if key in self.models:
                self.models.move_to_end(key)
                logger.info("Cache hit for %s (%s)", name, key[1])
                return self.models[key]

            if name not in self.available_models:
                raise UnknownModelError(f"Unknown model '{name}'")

            model = self._load_model(name, version, stage)
            self._try_cache_model(key, model)
            return model

    def get_threshold_value(self, model_name: str, threshold: float) -> float:
        """Resolve a requested percentile threshold to the stored numeric threshold value."""
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
        """Return serving metadata tuple: input size, threshold value, and drawing color."""
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
    """Run model prediction and return either mask arrays or redrawn output images."""
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
