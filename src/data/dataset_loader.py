import os
from glob import glob

import cv2
import numpy as np


def get_filenames(name: str, t: str = 'train') -> list[str]:
    """
    Returns the image filepaths for the given object class.

    Parameters
    ----------
    name : str
        Object class name.
    t : str, optional
        Training or testing images. The default is 'train'.

    Returns
    -------
    list[str]
        A list of matched-paths from glob module.

    """
    if t == 'train':
        files = glob(os.path.join('data', 'mvtec', name, 'train', 'good', '*.png'))
        return files
    elif t == 'test':
        files = glob(os.path.join('data', 'mvtec', name, 'test', '*', '*.png'))
        return files


def read_defect(filepaths: list[str]) -> list[str]:
    """
    Returns the ground-truth defect for each image in the test set.

    Parameters
    ----------
    filepaths : list[str]
        Image filepaths.

    Returns
    -------
    list[str]
        Defects as stated in the folder.

    """
    defects = []
    for file in filepaths:
        defects.append(file.split(os.sep)[-2])
    return defects


def read_images(filepaths: list[str], size: int) -> np.ndarray:
    """
    Returns the resized, floating point read images.

    Parameters
    ----------
    filepaths : list[str]
        Images to read.
    size : int
        Size to be resized to.

    Returns
    -------
    array_images : numpy.ndarray
        Numpy array of the images.

    """
    images = []
    for file in filepaths:
        img = cv2.imread(file)
        if img.shape[:2] != (size, size):
            img = cv2.resize(img, (size, size))
        images.append(img)
    array_images = np.array(images)
    array_images = array_images / 255.
    return array_images
