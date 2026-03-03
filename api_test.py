# -*- coding: utf-8 -*-
"""
Created on Sun May 26 20:51:28 2024

@author: asadm
"""

import requests
import cv2
import numpy as np
from train import get_filenames, read_images, convert_int
import os

NAME = 'hazelnut'
FORMAT = 'redrawn'
api_url = f'http://127.0.0.1:8000/predict_array/{NAME}?output_format={FORMAT}'
files = get_filenames(NAME, 'test')
print('Test files: ', files)
imgs = read_images(files, 256)
print(api_url)
imgs = convert_int(imgs)
print('imgs: ', len(imgs), imgs[0].shape)
data = [np.ravel(i).tolist() for i in imgs[8:11]]
print('data: ', len(data))
request = {"data": data}
response = requests.post(api_url, json=request)
output= response.json()["output"]
print('output: ', len(output))
if FORMAT == 'mask':
    shape = (256, 256, 1)
else:
    shape = (256, 256, 3)
output_imgs = [np.reshape(out, shape) for out in output]

print('Saving images...')
if not os.path.exists(os.path.join("api_images", NAME)):
    os.makedirs(os.path.join("api_images", NAME))
for redrawn, original, i in zip(output_imgs, imgs, range(len(imgs))):
    cv2.imwrite(os.path.join("api_images", NAME, f'redrawn_{i}.png'),
                redrawn)
    cv2.imwrite(os.path.join("api_images", NAME, f'original_{i}.png'),
                original)

