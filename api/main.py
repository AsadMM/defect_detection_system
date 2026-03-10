from fastapi import FastAPI

from api.routes import router


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

app.include_router(router)
