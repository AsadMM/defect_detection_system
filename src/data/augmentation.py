import cv2
import numpy as np


def crop_image(image: np.ndarray, crops: tuple | list, img_size: int) -> np.ndarray:
    """
    Returns an augmented image with given crop.

    Parameters
    ----------
    image : np.ndarray
        Original image.
    crops : tuple|list
        Crop limits.
    img_size : int
        Target image size.

    Returns
    -------
    image : np.ndarray
        Cropped image.

    """
    image = image[crops[0]:, crops[1]:]
    image = cv2.resize(image, (img_size, img_size))
    return image


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Returns an augmented image with the given rotation.

    Parameters
    ----------
    img : np.ndarray
        Original image.
    angle : float
        Rotation angle.

    Returns
    -------
    img_rotated : np.ndarray
        Rotated image.

    """
    h, w = img.shape[:2]
    angle %= 360
    M_rotate = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img_rotated = cv2.warpAffine(img, M_rotate, (w, h))
    return img_rotated


def flip_image(image: np.ndarray, flip: int) -> np.ndarray:
    """
    Returns an augmented image with given flip direction.

    Parameters
    ----------
    image : np.ndarray
        Original image.
    flip : int
        Flip direction (0 is horizontal, 1 is vertical, -1 is both).

    Returns
    -------
    image : np.ndarray
        Flipped image.

    """
    image = cv2.flip(image, flip)
    return image


def augment_images(
    imgs: np.ndarray,
    aug_size: int,
    rotate_limit: tuple[int, int],
    crop_limit: int,
    img_size: int,
) -> np.ndarray:
    """
    Returns the augmented images based on the input images.

    No of images augmented images generated is based on the difference
    between the size of input images and the augmented size needed.

    Parameters
    ----------
    imgs : np.ndarray
        Input images.
    aug_size : int
        Final augment size needed.
    rotate_limit : tuple[int, int]
        Rotation angle limits.
    crop_limit : int
        Crop limit.
    img_size : int
        Target image size.

    Returns
    -------
    np.ndarray
        Numpy array of Augmented images.

    """
    aug_images = []
    j = 0
    for i in range(aug_size - imgs.shape[0]):
        # choose a random augment operation to be performed
        choice = np.random.randint(1, 4)
        if choice == 1:
            angle = np.random.randint(rotate_limit[0], rotate_limit[1] + 1)
            img = rotate_image(imgs[j], angle)
        elif choice == 2:
            crop_size = np.random.randint(crop_limit + 1)
            img = crop_image(imgs[j], (crop_size, crop_size), img_size)
        else:
            flip = np.random.randint(-1, 2)
            img = flip_image(imgs[j], flip)
        aug_images.append(img)
        # circular index
        j = (j + 1) % imgs.shape[0]
    return np.array(aug_images)
