import unittest

import numpy as np

from src.inference.serving import run_inference


class _EchoModel:
    def predict(self, images):
        return images


class _ZeroModel:
    def predict(self, images):
        return np.zeros_like(images)


class InferenceTests(unittest.TestCase):
    def test_run_inference_mask_output_zero_when_reconstruction_matches(self):
        images = np.random.rand(2, 8, 8, 3).astype("float32")
        model = _EchoModel()

        output = run_inference(
            images=images,
            model=model,
            model_name="bottle",
            threshold_value=0.01,
            output_format="mask",
            color=(255, 0, 0),
        )

        self.assertEqual(output.shape, (2, 8, 8))
        self.assertEqual(output.dtype, np.uint8)
        self.assertEqual(int(output.max()), 0)

    def test_run_inference_redrawn_output_shape(self):
        images = np.ones((1, 8, 8, 3), dtype="float32")
        model = _ZeroModel()

        output = run_inference(
            images=images,
            model=model,
            model_name="bottle",
            threshold_value=0.0,
            output_format="redrawn",
            color=(255, 0, 0),
        )

        self.assertEqual(output.shape, (1, 8, 8, 3))
        self.assertEqual(output.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
