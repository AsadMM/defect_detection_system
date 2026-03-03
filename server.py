# -*- coding: utf-8 -*-
"""
Created on Saturday May 25 19:09:49 2024

@author: asadm
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Path, Query, Response
from pydantic import BaseModel
import numpy as np
from typing import List, Annotated
import cv2
from io import BytesIO
import zipfile
import pickle
from network import build1
from train import get_results, get_drawn_results, convert_int
from glob import glob
from enum import Enum
import os

tags_metadata = [
    {
        "name": "predict_array_input",
        "description": '''Hit the model with flattened image arrays in int8 BGR format, of specific input size. 
        Returns output with same formats as well. Supports batch inputs''',
    },
    {
        "name": "predict_image_input",
        "description": '''Hit the model with uploaded images (Multi-part form). Images are resized to specific model
        input size and the output is zip file with images. Supports batch image inputs''',
    },
]

description = ''' This API let's you infer anomaly detection models based on MVTEC-AD dataset. Models are
of single-object class detection type. There are 2 endpoints which let you query the model of specific object 
class with either flattened image arrays in int8 BGR format or directly uploaded images (multi-part form).
'''
# Load the app
app = FastAPI(
    title="Autoencoder-Anomaly Detector",
    description=description,
    summary="Infer anomaly detection models based on MVTEC-AD dataset",
    version="0.0.1",
    openapi_tags=tags_metadata
)

# Load the global variables including models, size files and threshold maps
colors = {'blue': (255, 0, 0), 'green': (0, 255, 0), 'red': (0, 0, 255)}
models = {}
sizes = {}
threshold_maps = {}
names = set()
# load filepaths based on patterns, for each: model, size and threshold
model_paths = glob(os.path.join('models', 'model_*.h5'))
size_paths = glob(os.path.join('sizes', 'sizes_*.pkl'))
thresholds_paths = glob(os.path.join('thresholds', 'thresholds_*.pkl'))
# read each set of files and store it in memory
for model_path, size_path, threshold_path in zip(model_paths, size_paths, thresholds_paths):
    # find the name of the object class this set of files are of
    name = model_path.split('_')[-1].replace('.h5', '')
    names.add(name)
    print("name: ", name)
    # open and store size definition of model inputs
    with open(size_path, 'rb') as file:
        size = pickle.load(file)
        sizes[name] = size
    print("sizes: ", size)
    # open and store threshold map containing threshold values based on validation data while the model was trained
    with open(threshold_path, 'rb') as file:
        threshold_map = pickle.load(file)
        threshold_maps[name] = threshold_map
    print("thresholds: ", threshold_map)
    # build the model network
    autoencoder = build1(size[0], size[0], size[1])
    # load model weights and store in memory
    autoencoder.load_weights(model_path)
    print("Loaded model for " + name + ":")
    autoencoder.summary()
    models[name] = autoencoder


# Define path enum for model
class ModelName(str, Enum):
    if 'bottle' in names:
        bottle = 'bottle'
    if 'cable' in names:
        cable = 'cable'
    if 'capsule' in names:
        capsule = 'capsule'
    if 'carpet' in names:
        carpet = 'carpet'
    if 'grid' in names:
        grid = 'grid'
    if 'hazelnut' in names:
        hazelnut = 'hazelnut'
    if 'leather' in names:
        leather = 'leather'
    if 'metal_nut' in names:
        metal_nut = 'metal_nut'
    if 'pill' in names:
        pill = 'pill'
    if 'screw' in names:
        screw = 'screw'
    if 'tile' in names:
        tile = 'tile'
    if 'toothbrush' in names:
        toothbrush = 'toothbrush'
    if 'transistor' in names:
        transistor = 'transistor'
    if 'wood' in names:
        wood = 'wood'
    if 'zipper' in names:
        zipper = 'zipper'


print('Printing model names loaded: ', [e.value for e in ModelName])


# Define class enum for predicted output format
# mask is full black image with only anomalous pixels in white
# redrawn is original image overdrawn with anomalous pixel contours
class ArrayOutputFormat(str, Enum):
    mask = 'mask'
    redrawn = 'redrawn'


# Define request models for array inputs
class ArrayInput(BaseModel):
    data: List[List[int]]


# Define class enum for color of pixels in redrawn output type
class AnomalyColor(str, Enum):
    blue = 'blue'
    green = 'green'
    red = 'red'


# Helper function to predict images from the autoencoder model
def predict_images(images, model_name):
    return models[model_name].predict(images)


@app.post("/predict_array/{model_name}", tags=["predict_array_input"])
async def predict_array_input(request: ArrayInput,
                              model_name: Annotated[ModelName, Path(title='The model for the input object class')],
                              threshold: Annotated[float, Query(ge=90.0, lt=100)] = 99.0,
                              output_format: ArrayOutputFormat = ArrayOutputFormat.mask,
                              redraw_color: AnomalyColor = AnomalyColor.blue):
    # Get the image_size, threshold_value from global dicts
    img_size = sizes[model_name.value]
    flat_size = img_size[0]*img_size[0]*img_size[1]
    threshold_value = threshold_maps[model_name.value][round(threshold, 1)]
    color = colors[redraw_color.value]
    print('Request: ', model_name.value, img_size, flat_size, threshold_value, color, output_format)
    if not all([len(d) == flat_size for d in request.data]):
        raise HTTPException(status_code=400, detail=f'''Invalid input data length in of the request. 
        Expected flat size is {flat_size}''')
    images = np.array([np.reshape(d, (size[0], size[0], size[1])) for d in request.data])
    if np.max(images) > 255:
        raise HTTPException(status_code=400, detail=f'Invalid input data size of int. (MAX IS 255)')
    else:
        images = images.astype('uint8') / 255.
    predicted_images = await predict_images(images, model_name.value)
    masked_images = get_results(images, predicted_images, threshold_value)
    if output_format.value == 'mask':
        output = convert_int(masked_images)
    else:
        output, _ = get_drawn_results(images, masked_images, color)
    return {"output": [np.ravel(out).tolist() for out in output]}


@app.post("/predict_image/{model_name}", tags=["predict_image_input"])
async def predict_image_input(model_name: Annotated[ModelName, Path(title='The model for the input object class')],
                              files: List[UploadFile] = File(...),
                              threshold: Annotated[float, Query(ge=90.0, lt=100)] = 99.0,
                              output_format: ArrayOutputFormat = ArrayOutputFormat.mask,
                              redraw_color: AnomalyColor = AnomalyColor.blue
                              ):
    images = []
    filenames = []
    # Get the image_size, threshold_value from global dicts
    img_size = sizes[model_name.value]
    flat_size = img_size[0] * img_size[0] * img_size[1]
    threshold_value = threshold_maps[model_name.value][round(threshold, 1)]
    color = colors[redraw_color.value]
    print('Request: ', model_name.value, img_size, flat_size, threshold_value, color, output_format)
    for file in files:
        contents = await file.read()
        np_image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (img_size[0], img_size[0]))
        images.append(image)
        filenames.append(file.filename)
    images = np.array(images) / 255.
    predicted_images = await predict_images(images, model_name.value)
    masked_images = get_results(images, predicted_images, threshold_value)
    if output_format.value == 'mask':
        zip_filename = 'masked_images.zip'
        output = convert_int(masked_images)
    else:
        zip_filename = 'redrawn_images.zip'
        output, _ = get_drawn_results(images, masked_images, color)

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
        for filename, out in zip(filenames,output):
            # Encode image as .jpg
            success, buffer = cv2.imencode('.png', out)
            if not success:
                raise HTTPException(status_code=500, detail=f"Failed to encode image {filename}")
            # Create a file-like object from the encoded image
            image_file = BytesIO(buffer)
            # Write the file-like object to the zip archive
            zip_file.writestr(filename, image_file.getvalue())
    zip_buffer.seek(0)
    # Grab ZIP file from in-memory, make response with correct MIME-type
    resp = Response(zip_buffer.getvalue(), media_type="application/x-zip-compressed", headers={
        'Content-Disposition': f'attachment;filename={zip_filename}'
    })
    return resp

