# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:52:39 2024

@author: asadm
"""

# import the necessary packages
import os

def configure_environment():
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

configure_environment()


from src.models.autoencoder import build1
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import tensorflow as tf
import mlflow
import mlflow.keras
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
from glob import glob
import cv2
import argparse

# Configure TensorFlow to allocate GPU memory on demand instead of pre-allocating all memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Initialize the argument parser for getting the global parameters needed
# for training.
parser = argparse.ArgumentParser(description='''This program allows you to train
                                 a deep learning model for detecting anomalies based on 
                                 autoencoder. It trains the model based on MVTEC-AD dataset
                                 for the given object class. This trains a single class
                                 model rather a multi-class model''')
parser.add_argument('--name', type=str, default='hazelnut', help='object class to train model for')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs for model training')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for model training')
parser.add_argument('--img_size', type=int, default=256, help='image size for for model input & output')
parser.add_argument('--aug_to', type=int, default=2000, help='number of datapoints to augment the input size to')
parser.add_argument('--threshold_percentile', type=float, default=99.0,
                    help='threshold percentile for estimating threshold')
parser.add_argument('--filters', nargs='+', type=int, default=[32, 64, 96],
                    help="filters for the network. each filter will create an additional layer")
parser.add_argument('--latent_dim', type=int, default=100,
                    help='latent dimension for the code-space (bottleneck) of the autoencoder')


class MLflowMetricsLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if 'loss' in logs:
            mlflow.log_metric('training_loss', float(logs['loss']), step=epoch)
        if 'val_loss' in logs:
            mlflow.log_metric('validation_loss', float(logs['val_loss']), step=epoch)



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


def crop_image(image: np.ndarray, crops: tuple | list) -> np.ndarray:
    """
    Returns an augmented image with given crop.

    Parameters
    ----------
    image : np.ndarray
        Original image.
    crops : tuple|list
        Crop limits.

    Returns
    -------
    image : np.ndarray
        Cropped image.

    """
    image = image[crops[0]:, crops[1]:]
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
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


def augment_images(imgs: np.ndarray, aug_size: int) -> np.ndarray:
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
            angle = np.random.randint(ROTATE_LIMIT[0], ROTATE_LIMIT[1] + 1)
            img = rotate_image(imgs[j], angle)
        elif choice == 2:
            crop_size = np.random.randint(CROP_LIMIT + 1)
            img = crop_image(imgs[j], (crop_size, crop_size))
        else:
            flip = np.random.randint(-1, 2)
            img = flip_image(imgs[j], flip)
        aug_images.append(img)
        # circular index
        j = (j + 1) % imgs.shape[0]
    return np.array(aug_images)


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


def get_drawn_results(images: np.ndarray, masked: np.ndarray,
                      color: tuple[int, int, int] = (255, 0, 0), thickness: int = 1) -> tuple[np.ndarray, np.ndarray]:
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
        contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        redrawn_imgs.append(cv2.drawContours(img.copy(), contours, -1, color, thickness))
    return np.array(redrawn_imgs), images


if __name__ == '__main__':
    # parse the given command-line arguments
    # if not arguments just use the default values
    args = parser.parse_args()
    EPOCHS = args.epochs
    BS = args.batch_size
    NAME = args.name.lower()
    IMG_SIZE = args.img_size
    IMG_DEPTH = 3
    AUG_TO = args.aug_to
    ROTATE_LIMIT = (-45, 45)
    CROP_LIMIT = 100
    THRESH_PERCENTILE = args.threshold_percentile
    FILTERS = args.filters
    LATENT_DIM = args.latent_dim
    print(FILTERS, LATENT_DIM)
    mlflow.set_tracking_uri("sqlite:///artifacts/mlflow/mlflow.db")
    mlflow.set_experiment("autoencoder_anomaly_detection")
    mlflow.start_run()
    mlflow.log_param('dataset_name', f'mvtec/{NAME}')
    mlflow.log_param('epochs', EPOCHS)
    mlflow.log_param('batch_size', BS)
    mlflow.log_param('filters', FILTERS)
    mlflow.log_param('latent_dim', LATENT_DIM)
    mlflow.log_param('img_size', IMG_SIZE)

    print('Reading files for ' + NAME)
    train_files = get_filenames(NAME, 'train')
    imgs = read_images(train_files, IMG_SIZE)
    print('Images read:', imgs.shape)
    augmented_images = augment_images(imgs, AUG_TO)
    print('Augmented images created:', augmented_images.shape)
    training_imgs = np.vstack((imgs, augmented_images))
    print('Total training images:', training_imgs.shape)

    # Freeing up memory
    del imgs
    del augmented_images

    train_data, validation_data = train_test_split(training_imgs, test_size=0.2,
                                                   random_state=26)
    # Freeing up memory
    del training_imgs
    # construct our convolutional autoencoder
    print("building autoencoder...")
    autoencoder = build1(IMG_SIZE, IMG_SIZE, IMG_DEPTH, FILTERS, LATENT_DIM)
    autoencoder.compile(loss="mse", optimizer="adam")
    autoencoder.summary()
    # train the convolutional autoencoder
    earlyStopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    mlflowMetricsLogger = MLflowMetricsLogger()
    checkpoint = ModelCheckpoint(os.path.join('artifacts', 'checkpoints', NAME, '{epoch:02d}-{val_loss:.5f}.weights.h5'),
                                 save_best_only=True, verbose=1, save_weights_only=True)
    H = autoencoder.fit(
        train_data, train_data,
        validation_data=(validation_data, validation_data),
        epochs=EPOCHS, batch_size=BS, callbacks=[checkpoint, earlyStopping, mlflowMetricsLogger])
    # Estimate the threshold based on the validation set
    valid_predicted_imgs = autoencoder.predict(validation_data)
    thresholds_map = get_threshold(validation_data, valid_predicted_imgs)
    threshold = thresholds_map[THRESH_PERCENTILE]
    print('Estimated threshold:', threshold)

    # IGNORE THIS
    '''threshold = 0.07769
    autoencoder.load_weights(os.path.join('artifacts', 'models', 'model_hazelnut.h5'))
    autoencoder.summary()'''

    # Read test images and run through the model and finally get the masked images
    test_files = get_filenames(NAME, 'test')
    test_imgs = read_images(test_files, IMG_SIZE)
    defects = read_defect(test_files)
    test_predicted_imgs = autoencoder.predict(test_imgs)
    masked_results = get_results(test_imgs, test_predicted_imgs, threshold)
    test_results = [(np.sum(res), defect) for res, defect in zip(masked_results, defects)]
    print('Listed test results:', test_results)

    # Group the results
    # For defects the correct class is 'Predicted_1'
    # For good the correct class is 'Predicted_0'
    test_results_grouped = {}
    for res, defect in test_results:
        if test_results_grouped.get(defect) is None:
            test_results_grouped[defect] = {'Total': 0, 'Predicted_0': 0, 'Predicted_1': 0}
        test_results_grouped[defect]['Total'] += 1
        if res == 0:
            test_results_grouped[defect]['Predicted_0'] += 1
        else:
            test_results_grouped[defect]['Predicted_1'] += 1
    print('Grouped test results:', test_results_grouped)
    print('Saving images...')
    if not os.path.exists(os.path.join("comparison_images", NAME)):
        os.makedirs(os.path.join("comparison_images", NAME))
    redrawn_imgs, original_imgs = get_drawn_results(test_imgs, masked_results)
    for redrawn, original, defect, i in zip(redrawn_imgs, original_imgs, defects,
                                            range(len(original_imgs))):
        cv2.imwrite(os.path.join("comparison_images", NAME, f'redrawn_{i}.png'),
                    redrawn)
        cv2.imwrite(os.path.join("comparison_images", NAME, f'original_{defect}_{i}.png'),
                    original)

    print('Saving model, threshold map and other variables...')
    os.makedirs(os.path.join('artifacts', 'models'), exist_ok=True)
    os.makedirs(os.path.join('artifacts', 'thresholds'), exist_ok=True)
    os.makedirs(os.path.join('artifacts', 'sizes'), exist_ok=True)
    
    autoencoder.save(os.path.join('artifacts', 'models', 'model_' + NAME + '.h5'))
    mlflow.keras.log_model(autoencoder, artifact_path='model')
    # Open a file and use dump() 
    with open(os.path.join('artifacts', 'thresholds', 'thresholds_' + NAME + '.pkl'), 'wb') as file:
        # Save the thresholds value in a file
        pickle.dump(thresholds_map, file)

    with open(os.path.join('artifacts', 'sizes', 'sizes_' + NAME + '.pkl'), 'wb') as file:
        # Save the sizes value in a file
        pickle.dump([IMG_SIZE, IMG_DEPTH], file)

    # Ignore this
    '''with open('residual_' + 'hazelnut' + '.pkl', 'rb') as file:
        res_maps = pickle.load(file)
        print(res_maps)
    percentiles = np.arange(90, 100, 0.1)
    thresholds = np.percentile(res_maps, percentiles)
    thresholds_dict = {round(percentile,1): threshold for percentile, threshold in zip(percentiles, thresholds)}'''

    mlflow.end_run()
    print('Exiting...')
