import cv2
import numpy as np


def convert_int(images: np.ndarray) -> np.ndarray:
    """
    Returns original images with ints

    Parameters
    ----------
    images : np.ndarray
        images with floats.

    Returns
    -------
    original_images : np.ndarray
        Images with ints.
    """
    return (images * 255).astype(np.uint8)


def get_drawn_results(
    images: np.ndarray,
    masked: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns original images with drawn anomalous points

    Parameters
    ----------
    images : np.ndarray
        original images.
    masked : np.ndarray
        masked results of the predicted images.
    color : tuple[int, int, int], optional
        color of the contours in BGR. default is blue (255, 0, 0)
    thickness : int, optional
        thickness of the drawn contours. default is 1
    Returns
    -------
    redrawn_images : np.ndarray
        Images with drawn anomalous points.
    original_images : np.ndarray
        Original images (converted back to int).
    """
    redrawn_imgs = []
    images, masked = convert_int(images), convert_int(masked)
    for img, mask in zip(images, masked):
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        redrawn_imgs.append(cv2.drawContours(img.copy(), contours, -1, color, thickness))
    return np.array(redrawn_imgs), images
