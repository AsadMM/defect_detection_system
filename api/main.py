import logging
import time
import uuid

from fastapi import FastAPI
from fastapi import Request

from src.utils.logging import setup_logging

setup_logging()

from src.inference.serving import model_registry

model_registry.load_metadata()
if not model_registry.available_models:
    raise RuntimeError(
        "No model metadata found in artifacts. At least one trained model is required to start the API."
    )

from api.routes import router


logger = logging.getLogger(__name__)


tags_metadata = [
    {
        "name": "predict_array_input",
        "description": (
            "Hit the model with flattened image arrays in int8 BGR format, of specific input size. "
            "Returns output with same formats as well. Supports batch inputs"
        ),
    },
    {
        "name": "predict_image_input",
        "description": (
            "Hit the model with uploaded images (Multi-part form). Images are resized to specific model "
            "input size and the output is zip file with images. Supports batch image inputs"
        ),
    },
]

description = (
    "This API lets you infer anomaly detection models based on MVTEC-AD dataset. Models are of "
    "single-object class detection type. There are 2 endpoints which let you query the model of specific "
    "object class with either flattened image arrays in int8 BGR format or directly uploaded images "
    "(multi-part form)."
)


app = FastAPI(
    title="Autoencoder-Anomaly Detector",
    description=description,
    summary="Infer anomaly detection models based on MVTEC-AD dataset",
    version="0.0.1",
    openapi_tags=tags_metadata,
)


@app.middleware("http")
async def log_request_latency(request: Request, call_next):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        duration = time.time() - start_time
        logger.exception(
            "req_id=%s | %s %s | ERROR | time=%.3fs",
            request_id,
            request.method,
            request.url.path,
            duration,
        )
        raise

    duration = time.time() - start_time
    logger.info(
        "req_id=%s | %s %s | status=%d | time=%.3fs",
        request_id,
        request.method,
        request.url.path,
        status_code,
        duration,
    )

    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(router)
