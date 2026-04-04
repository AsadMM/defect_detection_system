from io import BytesIO
import logging
from typing import Annotated
import zipfile

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Path, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.enums import AnomalyColor, ArrayOutputFormat, ModelName, ModelStage
from api.constants import MAX_ARRAY_BATCH_SIZE, MAX_IMAGE_BATCH_SIZE, MULTI_FILE_OPENAPI_SCHEMA
from api.schemas import ArrayInput, ArrayOutputResponse, ErrorDetail, ErrorResponse
from src.inference.exceptions import ModelMetadataError, UnknownModelError
from src.inference.serving import model_registry, run_inference


router = APIRouter()
logger = logging.getLogger(__name__)


def error_detail(code: str, message: str, details: str | None = None) -> dict[str, str | None]:
    return ErrorDetail(code=code, message=message, details=details).model_dump(exclude_none=True)


def prepare_array_images(data: list[list[int]], img_size: tuple[int, int]) -> np.ndarray:
    images = np.array([np.reshape(d, (img_size[0], img_size[0], img_size[1])) for d in data])
    if np.issubdtype(images.dtype, np.number) is False:
        raise ValueError("Invalid input data type. Expected numeric pixel values.")
    if np.min(images) < 0 or np.max(images) > 255:
        raise ValueError("Invalid input data range. Pixel values must be in [0, 255].")
    return images.astype("uint8") / 255.0


def process_image(contents: bytes, img_size: int) -> np.ndarray:
    np_image = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode image")
    return cv2.resize(image, (img_size, img_size))


def encode_zip_images(filenames: list[str], output: np.ndarray, model_name: str) -> bytes:
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for filename, out in zip(filenames, output):
            success, buffer = cv2.imencode(".png", out)
            if not success:
                logger.error("Failed request for model=%s", model_name)
                raise RuntimeError(f"Failed to encode image {filename}")
            zip_file.writestr(filename, BytesIO(buffer).getvalue())
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def flatten_output_array(output: np.ndarray) -> list[list[int]]:
    return [np.ravel(out).tolist() for out in output]


@router.post(
    "/predict_array/{model_name}",
    tags=["predict_array_input"],
    response_model=ArrayOutputResponse,
    summary="Run anomaly inference on flattened array inputs",
    description=(
        "Accepts a batch of flattened BGR uint8-like arrays, reshapes each item to the model input size, "
        "runs anomaly inference, and returns either mask or redrawn outputs in flattened array form."
    ),
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid request payload, invalid selector combination, or batch-size limit exceeded",
        },
        404: {"model": ErrorResponse, "description": "Requested model is not available"},
        503: {"model": ErrorResponse, "description": "Model metadata unavailable for serving"},
    },
)
async def predict_array_input(
    request: ArrayInput,
    model_name: Annotated[ModelName, Path(title="The model for the input object class")],
    threshold: Annotated[float, Query(ge=90.0, lt=100)] = 99.0,
    version: Annotated[int | None, Query(description="Model version")] = None,
    stage: Annotated[ModelStage, Query(description="Model stage")] = ModelStage.production,
    output_format: ArrayOutputFormat = ArrayOutputFormat.mask,
    redraw_color: AnomalyColor = AnomalyColor.blue,
):
    if version is not None and stage != ModelStage.production:
        raise HTTPException(
            status_code=400,
            detail=error_detail(
                "INVALID_MODEL_SELECTOR",
                "Provide either version or stage, not both",
            ),
        )

    logger.info("Received request for model=%s", model_name.value)
    try:
        img_size, threshold_value, color = model_registry.get_model_context(
            model_name.value,
            threshold,
            redraw_color.value,
        )
    except UnknownModelError as exc:
        raise HTTPException(
            status_code=404,
            detail=error_detail("MODEL_NOT_FOUND", "Requested model is not available", str(exc)),
        ) from exc
    except ModelMetadataError as exc:
        raise HTTPException(
            status_code=503,
            detail=error_detail("MODEL_METADATA_UNAVAILABLE", "Model metadata is unavailable", str(exc)),
        ) from exc
    flat_size = img_size[0] * img_size[0] * img_size[1]
    logger.info("Batch size: %d", len(request.data))
    if len(request.data) > MAX_ARRAY_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=error_detail(
                "BATCH_SIZE_LIMIT_EXCEEDED",
                "Array batch size exceeds allowed maximum",
                f"max_batch_size={MAX_ARRAY_BATCH_SIZE}, received={len(request.data)}",
            ),
        )

    if not all(len(d) == flat_size for d in request.data):
        logger.warning("Invalid input size for model=%s", model_name.value)
        logger.error("Failed request for model=%s", model_name.value)
        raise HTTPException(
            status_code=400,
            detail=error_detail(
                "INVALID_INPUT_LENGTH",
                "Input length does not match model input size",
                f"Expected flattened size {flat_size} per item",
            ),
        )

    try:
        images = await run_in_threadpool(prepare_array_images, request.data, img_size)
    except ValueError as exc:
        logger.warning("Invalid input size for model=%s", model_name.value)
        logger.error("Failed request for model=%s", model_name.value)
        raise HTTPException(
            status_code=400,
            detail=error_detail("INVALID_INPUT_VALUES", "Input values are invalid", str(exc)),
        ) from exc

    try:
        model = await run_in_threadpool(model_registry.get_model, model_name.value, version, stage.value)
    except UnknownModelError as exc:
        raise HTTPException(
            status_code=404,
            detail=error_detail("MODEL_NOT_FOUND", "Requested model is not available", str(exc)),
        ) from exc
    except ModelMetadataError as exc:
        raise HTTPException(
            status_code=503,
            detail=error_detail("MODEL_METADATA_UNAVAILABLE", "Model metadata is unavailable", str(exc)),
        ) from exc
    output = await run_in_threadpool(
        run_inference,
        images,
        model,
        model_name.value,
        threshold_value,
        output_format.value,
        color,
    )

    flattened_output = await run_in_threadpool(flatten_output_array, output)
    return {"output": flattened_output}


@router.post(
    "/predict_image/{model_name}",
    tags=["predict_image_input"],
    openapi_extra=MULTI_FILE_OPENAPI_SCHEMA,
    summary="Run anomaly inference on uploaded image files",
    description=(
        "Accepts a batch of uploaded image files (multipart/form-data), resizes them to model input size, "
        "runs anomaly inference, and returns a ZIP containing either masks or redrawn images."
    ),
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Invalid request payload, image decode failure, invalid selector, or batch-size limit exceeded",
        },
        404: {"model": ErrorResponse, "description": "Requested model is not available"},
        500: {"model": ErrorResponse, "description": "Failed while encoding response images"},
        503: {"model": ErrorResponse, "description": "Model metadata unavailable for serving"},
    },
)
async def predict_image_input(
    model_name: Annotated[ModelName, Path(title="The model for the input object class")],
    files: list[UploadFile] = File(...),
    threshold: Annotated[float, Query(ge=90.0, lt=100)] = 99.0,
    version: Annotated[int | None, Query(description="Model version")] = None,
    stage: Annotated[ModelStage, Query(description="Model stage")] = ModelStage.production,
    output_format: ArrayOutputFormat = ArrayOutputFormat.mask,
    redraw_color: AnomalyColor = AnomalyColor.blue,
):
    if version is not None and stage != ModelStage.production:
        raise HTTPException(
            status_code=400,
            detail=error_detail(
                "INVALID_MODEL_SELECTOR",
                "Provide either version or stage, not both",
            ),
        )

    logger.info("Received request for model=%s", model_name.value)
    images = []
    filenames = []
    logger.info("Batch size: %d", len(files))
    if len(files) > MAX_IMAGE_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=error_detail(
                "BATCH_SIZE_LIMIT_EXCEEDED",
                "Image batch size exceeds allowed maximum",
                f"max_batch_size={MAX_IMAGE_BATCH_SIZE}, received={len(files)}",
            ),
        )

    try:
        img_size, threshold_value, color = model_registry.get_model_context(
            model_name.value,
            threshold,
            redraw_color.value,
        )
    except UnknownModelError as exc:
        raise HTTPException(
            status_code=404,
            detail=error_detail("MODEL_NOT_FOUND", "Requested model is not available", str(exc)),
        ) from exc
    except ModelMetadataError as exc:
        raise HTTPException(
            status_code=503,
            detail=error_detail("MODEL_METADATA_UNAVAILABLE", "Model metadata is unavailable", str(exc)),
        ) from exc

    for file in files:
        contents = await file.read()
        try:
            image = await run_in_threadpool(process_image, contents, img_size[0])
        except ValueError:
            logger.warning("Invalid input size for model=%s", model_name.value)
            logger.error("Failed request for model=%s", model_name.value)
            raise HTTPException(
                status_code=400,
                detail=error_detail(
                    "IMAGE_DECODE_FAILED",
                    "Failed to decode uploaded image",
                    f"filename={file.filename}",
                ),
            )
        images.append(image)
        filenames.append(file.filename)

    images = (np.array(images) / 255.0).astype("float32")
    try:
        model = await run_in_threadpool(model_registry.get_model, model_name.value, version, stage.value)
    except UnknownModelError as exc:
        raise HTTPException(
            status_code=404,
            detail=error_detail("MODEL_NOT_FOUND", "Requested model is not available", str(exc)),
        ) from exc
    except ModelMetadataError as exc:
        raise HTTPException(
            status_code=503,
            detail=error_detail("MODEL_METADATA_UNAVAILABLE", "Model metadata is unavailable", str(exc)),
        ) from exc
    output = await run_in_threadpool(
        run_inference,
        images,
        model,
        model_name.value,
        threshold_value,
        output_format.value,
        color,
    )
    if output_format.value == "mask":
        zip_filename = "masked_images.zip"
    else:
        zip_filename = "redrawn_images.zip"

    try:
        zip_bytes = await run_in_threadpool(encode_zip_images, filenames, output, model_name.value)
    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=error_detail("IMAGE_ENCODING_FAILED", "Failed to encode output image", str(exc)),
        ) from exc

    return Response(
        zip_bytes,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment;filename={zip_filename}"},
    )
