from io import BytesIO
import logging
from typing import Annotated
import zipfile

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Path, Query, Response, UploadFile

from api.enums import AnomalyColor, ArrayOutputFormat, ModelName
from api.constants import MUTLI_FILE_OPENAPI_SCHEMA
from api.schemas import ArrayInput
from api.services import (
    get_model_context,
    run_inference,
)


router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/predict_array/{model_name}", tags=["predict_array_input"])
async def predict_array_input(
    request: ArrayInput,
    model_name: Annotated[ModelName, Path(title="The model for the input object class")],
    threshold: Annotated[float, Query(ge=90.0, lt=100)] = 99.0,
    output_format: ArrayOutputFormat = ArrayOutputFormat.mask,
    redraw_color: AnomalyColor = AnomalyColor.blue,
):
    logger.info("Received request for model=%s", model_name.value)
    img_size, threshold_value, color = get_model_context(model_name.value, threshold, redraw_color.value)
    flat_size = img_size[0] * img_size[0] * img_size[1]
    logger.info("Batch size: %d", len(request.data))

    if not all(len(d) == flat_size for d in request.data):
        logger.warning("Invalid input size for model=%s", model_name.value)
        logger.error("Failed request for model=%s", model_name.value)
        raise HTTPException(
            status_code=400,
            detail=f"Invalid input data length in of the request. Expected flat size is {flat_size}",
        )

    images = np.array([np.reshape(d, (img_size[0], img_size[0], img_size[1])) for d in request.data])
    if np.max(images) > 255:
        logger.warning("Invalid input size for model=%s", model_name.value)
        logger.error("Failed request for model=%s", model_name.value)
        raise HTTPException(status_code=400, detail="Invalid input data size of int. (MAX IS 255)")

    images = images.astype("uint8") / 255.0
    output = run_inference(
        images,
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
    output_format: ArrayOutputFormat = ArrayOutputFormat.mask,
    redraw_color: AnomalyColor = AnomalyColor.blue,
):
    logger.info("Received request for model=%s", model_name.value)
    images = []
    filenames = []
    logger.info("Batch size: %d", len(files))

    img_size, threshold_value, color = get_model_context(model_name.value, threshold, redraw_color.value)

    for file in files:
        contents = await file.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Invalid input size for model=%s", model_name.value)
            logger.error("Failed request for model=%s", model_name.value)
            raise HTTPException(status_code=400, detail=f"Failed to decode image {file.filename}")
        image = cv2.resize(image, (img_size[0], img_size[0]))
        images.append(image)
        filenames.append(file.filename)

    images = np.array(images) / 255.0
    output = run_inference(
        images,
        model_name.value,
        threshold_value,
        output_format.value,
        color,
    )
    if output_format.value == "mask":
        zip_filename = "masked_images.zip"
    else:
        zip_filename = "redrawn_images.zip"

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for filename, out in zip(filenames, output):
            success, buffer = cv2.imencode(".png", out)
            if not success:
                logger.error("Failed request for model=%s", model_name.value)
                raise HTTPException(status_code=500, detail=f"Failed to encode image {filename}")
            image_file = BytesIO(buffer)
            zip_file.writestr(filename, image_file.getvalue())

    zip_buffer.seek(0)
    return Response(
        zip_buffer.getvalue(),
        media_type="application/x-zip-compressed",
        headers={"Content-Disposition": f"attachment;filename={zip_filename}"},
    )
