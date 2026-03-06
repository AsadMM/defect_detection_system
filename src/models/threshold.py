import numpy as np


def get_threshold(validation: np.ndarray, predicted: np.ndarray) -> dict[float, float]:
    """
    Estimate the threshold based on the validation set

    Parameters
    ----------
    validation : np.ndarray
        Validation data set.
    predicted : np.ndarray
        Predicted images of the validation data set.

    Returns
    -------
    threshold : dict{float:float}
        Estimated threshold values
    """
    residual_maps = []
    # percentiles to calculate and store for API
    percentiles = np.arange(90, 100, 0.1)
    for val, pred in zip(validation, predicted):
        # Residual map calculation. Mean across 3 channels
        res_map = np.mean(np.power(val - pred, 2), axis=2)
        residual_maps.append(res_map)
    thresholds = np.percentile(residual_maps, percentiles)
    thresholds_dict = {round(percentile, 1): threshold for percentile, threshold in zip(percentiles, thresholds)}
    return thresholds_dict


def get_results(images: np.ndarray, predicted: np.ndarray, threshold: float) -> np.ndarray:
    """
    Returns the masked images with anomalous points.

    The images contains 0 and 1, pixels are set to 1 which is greater
    than the given threshold for anomaly detection.

    Parameters
    ----------
    images : np.ndarray
        Original test images.
    predicted : np.ndarray
        Model output images.
    threshold : float
        Threshold for anomaly detection.

    Returns
    -------
    masked_images : np.ndarray
        Numpy array of masked images with anomalous pixel set to 1.

    """
    output_images = []
    size = images.shape[1:3]
    for image, pred in zip(images, predicted):
        res_map = np.mean(np.power(image - pred, 2), axis=2)
        mask = np.zeros(size)
        mask[res_map > threshold] = 1
        output_images.append(mask)
    return np.array(output_images)
