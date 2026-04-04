import unittest
from unittest.mock import patch
import os

from src.inference.exceptions import ModelMetadataError, UnknownModelError
from src.inference.serving import ModelRegistryService


class _FakeModel:
    def __init__(self, params: int):
        self._params = params

    def count_params(self) -> int:
        return self._params


class RegistryTests(unittest.TestCase):
    def test_stage_normalization(self):
        svc = ModelRegistryService()
        self.assertEqual(svc._normalize_stage("production"), "Production")
        self.assertEqual(svc._normalize_stage("  Staging  "), "Staging")
        self.assertEqual(svc._normalize_stage("Custom"), "Custom")

    def test_get_threshold_value_missing_model_metadata(self):
        svc = ModelRegistryService()
        with self.assertRaises(ModelMetadataError):
            svc.get_threshold_value("missing", 99.0)

    def test_get_threshold_value_missing_percentile(self):
        svc = ModelRegistryService()
        svc.threshold_maps = {"bottle": {99.0: 0.12}}
        with self.assertRaises(ModelMetadataError):
            svc.get_threshold_value("bottle", 98.0)

    def test_get_model_unknown_model(self):
        svc = ModelRegistryService()
        with self.assertRaises(UnknownModelError):
            svc.get_model("unknown_model")

    def test_get_model_context_invalid_color(self):
        svc = ModelRegistryService()
        svc.sizes = {"bottle": (128, 3)}
        svc.threshold_maps = {"bottle": {99.0: 0.12}}
        with self.assertRaises(ModelMetadataError):
            svc.get_model_context("bottle", 99.0, "purple")

    def test_model_cache_evicts_lru_when_budget_exceeded(self):
        with patch.dict(os.environ, {"MODEL_CACHE_MAX_BYTES": "4000"}):
            svc = ModelRegistryService()
        svc.available_models = {"a", "b", "c"}

        with patch.object(
            svc,
            "_load_model",
            side_effect=[_FakeModel(500), _FakeModel(500), _FakeModel(500)],
        ):
            svc.get_model("a")
            svc.get_model("b")
            svc.get_model("c")

        self.assertNotIn(("a", "Production"), svc.models)
        self.assertIn(("b", "Production"), svc.models)
        self.assertIn(("c", "Production"), svc.models)
        self.assertLessEqual(svc.model_cache_total_bytes, svc.max_model_cache_bytes)

    def test_model_larger_than_cache_budget_is_not_cached(self):
        with patch.dict(os.environ, {"MODEL_CACHE_MAX_BYTES": "1000"}):
            svc = ModelRegistryService()
        svc.available_models = {"a"}

        large_model = _FakeModel(1000)  # ~4000 bytes estimate
        with patch.object(svc, "_load_model", return_value=large_model):
            loaded = svc.get_model("a")

        self.assertIs(loaded, large_model)
        self.assertNotIn(("a", "Production"), svc.models)
        self.assertEqual(svc.model_cache_total_bytes, 0)


if __name__ == "__main__":
    unittest.main()
