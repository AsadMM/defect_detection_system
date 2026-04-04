from io import BytesIO
import logging
from typing import Annotated
import zipfile

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Path, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool

from api.exceptions import ModelMetadataError, UnknownModelError
from api.enums import AnomalyColor, ArrayOutputFormat, ModelName, ModelStage
from api.constants import MUTLI_FILE_OPENAPI_SCHEMA
from api.schemas import ArrayInput
from api.services import (
    model_registry,
    run_inference,
)


router = APIRouter()
logger = logging.getLogger(__name__)


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


@router.post("/predict_array/{model_name}", tags=["predict_array_input"])
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
        raise HTTPException(status_code=400, detail="Provide either version or stage, not both")

    logger.info("Received request for model=%s", model_name.value)
    try:
        img_size, threshold_value, color = model_registry.get_model_context(
            model_name.value,
            threshold,
            redraw_color.value,
        )
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelMetadataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    flat_size = img_size[0] * img_size[0] * img_size[1]
    logger.info("Batch size: %d", len(request.data))

    if not all(len(d) == flat_size for d in request.data):
        logger.warning("Invalid input size for model=%s", model_name.value)
        logger.error("Failed request for model=%s", model_name.value)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data length in of the request. Expected flat size is {flat_size}",
        )

    try:
        images = await run_in_threadpool(prepare_array_images, request.data, img_size)
    except ValueError as exc:
        logger.warning("Invalid input size for model=%s", model_name.value)
        logger.error("Failed request for model=%s", model_name.value)
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        model = await run_in_threadpool(model_registry.get_model, model_name.value, version, stage.value)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelMetadataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    output = await run_in_threadpool(
        run_inference,
        images,
        model,
        model_name.value,
        threshold_value,
        output_format.value,
        color,
    )

    return {"output": [np.ravel(out).tolist() for out in output]}


@router.post(
    "/predict_image/{model_name}",
    tags=["predict_image_input"],
    openapi_extra=MUTLI_FILE_OPENAPI_SCHEMA,
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
        raise HTTPException(status_code=400, detail="Provide either version or stage, not both")

    logger.info("Received request for model=%s", model_name.value)
    images = []
    filenames = []
    logger.info("Batch size: %d", len(files))

    try:
        img_size, threshold_value, color = model_registry.get_model_context(
            model_name.value,
            threshold,
            redraw_color.value,
        )
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelMetadataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    for file in files:
        contents = await file.read()
        try:
            image = await run_in_threadpool(process_image, contents, img_size[0])
        except ValueError:
            logger.warning("Invalid input size for model=%s", model_name.value)
            logger.error("Failed request for model=%s", model_name.value)
            raise HTTPException(status_code=400, detail=f"Failed to decode image {file.filename}")
        images.append(image)
        filenames.append(file.filename)

    images = (np.array(images) / 255.0).astype("float32")
    try:
        model = await run_in_threadpool(model_registry.get_model, model_name.value, version, stage.value)
    except UnknownModelError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelMetadataError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
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
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return Response(
        zip_bytes,
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment;filename={zip_filename}"},
    )
