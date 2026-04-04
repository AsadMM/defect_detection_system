import unittest
from io import BytesIO
from unittest.mock import patch

from fastapi import HTTPException, UploadFile

from src.inference.serving import model_registry

# Ensure model metadata exists before importing api.enums/api.routes,
# because ModelName enum is built at import time.
model_registry.load_metadata()

from api.enums import ModelName, ModelStage
from api.routes import predict_array_input, predict_image_input
from api.schemas import ArrayInput


class ApiTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _first_model_name():
        return next(iter(ModelName))

    async def test_predict_array_rejects_version_with_non_production_stage(self):
        with self.assertRaises(HTTPException) as ctx:
            await predict_array_input(
                request=ArrayInput(data=[]),
                model_name=self._first_model_name(),
                version=1,
                stage=ModelStage.staging,
            )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail["code"], "INVALID_MODEL_SELECTOR")

    async def test_predict_array_rejects_batch_size_over_limit(self):
        with patch("api.routes.MAX_ARRAY_BATCH_SIZE", 1), patch(
            "api.routes.model_registry.get_model_context",
            return_value=((2, 3), 0.1, (255, 0, 0)),
        ):
            with self.assertRaises(HTTPException) as ctx:
                await predict_array_input(
                    request=ArrayInput(data=[[0] * 12, [0] * 12]),
                    model_name=self._first_model_name(),
                )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail["code"], "BATCH_SIZE_LIMIT_EXCEEDED")

    async def test_predict_image_rejects_batch_size_over_limit(self):
        with patch("api.routes.MAX_IMAGE_BATCH_SIZE", 1), patch(
            "api.routes.model_registry.get_model_context",
            return_value=((128, 3), 0.1, (255, 0, 0)),
        ):
            files = [
                UploadFile(filename="a.png", file=BytesIO(b"x")),
                UploadFile(filename="b.png", file=BytesIO(b"y")),
            ]
            with self.assertRaises(HTTPException) as ctx:
                await predict_image_input(
                    model_name=self._first_model_name(),
                    files=files,
                )

        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail["code"], "BATCH_SIZE_LIMIT_EXCEEDED")


if __name__ == "__main__":
    unittest.main()
