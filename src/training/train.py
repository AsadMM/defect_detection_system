# -*- coding: utf-8 -*-
"""
Created on Tue May 21 15:52:39 2024

@author: asadm
"""

# import the necessary packages
import logging
import os

def configure_environment():
    os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

configure_environment()


from src.models.autoencoder import build1
from src.models.threshold import get_threshold, get_results
from src.models.evaluation import group_test_results
from src.data.dataset_loader import get_filenames, read_images, read_defect
from src.data.augmentation import augment_images
from src.training.trainer import train_autoencoder
from src.utils.visualization import get_drawn_results
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
import mlflow
import mlflow.keras
import numpy as np
import pickle
import cv2
import argparse

from src.utils.logging import setup_logging


logger = logging.getLogger(__name__)

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
parser.add_argument('--config', type=str, default=None,
                    help='path to a YAML config file with training parameters')


def load_yaml_config(config_path: str) -> dict:
    """
    Loads YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to YAML config.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required when using --config. Install with `pip install pyyaml`.") from exc

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping/object at root: {config_path}")
    return config


def apply_config_overrides(args: argparse.Namespace, config: dict, parser: argparse.ArgumentParser) -> argparse.Namespace:
    """
    Overrides parsed CLI args with values from YAML config.

    Notes
    -----
    - Values in config override values from CLI when both are provided.
    - Supports legacy `dataset` key as alias for `name`.
    """
    config_values = dict(config)
    if 'dataset' in config_values and 'name' not in config_values:
        config_values['name'] = config_values['dataset']
    config_values.pop('dataset', None)

    valid_keys = {action.dest for action in parser._actions if action.dest != 'help'}
    unknown_keys = sorted(set(config_values.keys()) - valid_keys)
    if unknown_keys:
        raise ValueError(f"Unsupported config keys: {unknown_keys}. Supported keys: {sorted(valid_keys)}")

    for key, value in config_values.items():
        setattr(args, key, value)
    return args


class MLflowMetricsLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if 'loss' in logs:
            mlflow.log_metric('training_loss', float(logs['loss']), step=epoch)
        if 'val_loss' in logs:
            mlflow.log_metric('validation_loss', float(logs['val_loss']), step=epoch)



if __name__ == '__main__':
    setup_logging()

    # parse the given command-line arguments
    # if not arguments just use the default values
    args = parser.parse_args()
    if args.config:
        config = load_yaml_config(args.config)
        args = apply_config_overrides(args, config, parser)

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
    logger.info("Model config filters=%s latent_dim=%s", FILTERS, LATENT_DIM)
    mlflow.set_tracking_uri("sqlite:///artifacts/mlflow/mlflow.db")
    mlflow.set_experiment("autoencoder_anomaly_detection")
    mlflow.start_run()
    mlflow.log_param('dataset_name', f'mvtec/{NAME}')
    mlflow.log_param('epochs', EPOCHS)
    mlflow.log_param('batch_size', BS)
    mlflow.log_param('filters', FILTERS)
    mlflow.log_param('latent_dim', LATENT_DIM)
    mlflow.log_param('img_size', IMG_SIZE)

    logger.info("Reading files for %s", NAME)
    train_files = get_filenames(NAME, 'train')
    imgs = read_images(train_files, IMG_SIZE)
    logger.info("Images read: %s", imgs.shape)
    augmented_images = augment_images(imgs, AUG_TO, ROTATE_LIMIT, CROP_LIMIT, IMG_SIZE)
    logger.info("Augmented images created: %s", augmented_images.shape)
    training_imgs = np.vstack((imgs, augmented_images))
    logger.info("Total training images: %s", training_imgs.shape)

    # Freeing up memory
    del imgs
    del augmented_images

    # construct our convolutional autoencoder
    logger.info("Building autoencoder...")
    autoencoder = build1(IMG_SIZE, IMG_SIZE, IMG_DEPTH, FILTERS, LATENT_DIM)
    autoencoder.summary()

    # train the convolutional autoencoder
    mlflowMetricsLogger = MLflowMetricsLogger()
    train_config = {
        'name': NAME,
        'epochs': EPOCHS,
        'batch_size': BS,
        'test_size': 0.2,
        'random_state': 26,
        'callbacks': [mlflowMetricsLogger],
    }
    autoencoder, H, validation_data = train_autoencoder(autoencoder, training_imgs, train_config)
    # Freeing up memory
    del training_imgs
    # Estimate the threshold based on the validation set
    valid_predicted_imgs = autoencoder.predict(validation_data)
    thresholds_map = get_threshold(validation_data, valid_predicted_imgs)
    threshold = thresholds_map[THRESH_PERCENTILE]
    logger.info("Estimated threshold: %s", threshold)

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
    logger.info("Listed test results: %s", test_results)

    # Group the results
    test_results_grouped = group_test_results(test_results)
    logger.info("Grouped test results: %s", test_results_grouped)
    logger.info("Saving images...")
    if not os.path.exists(os.path.join('artifacts', 'comparison_images', NAME)):
        os.makedirs(os.path.join('artifacts', 'comparison_images', NAME))
    redrawn_imgs, original_imgs = get_drawn_results(test_imgs, masked_results)
    for redrawn, original, defect, i in zip(redrawn_imgs, original_imgs, defects,
                                            range(len(original_imgs))):
        cv2.imwrite(os.path.join('artifacts', 'comparison_images', NAME, f'redrawn_{i}.png'),
                    redrawn)
        cv2.imwrite(os.path.join('artifacts', 'comparison_images', NAME, f'original_{defect}_{i}.png'),
                    original)

    logger.info("Saving model, threshold map and other variables...")
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
        logger.info("Residual maps: %s", res_maps)
    percentiles = np.arange(90, 100, 0.1)
    thresholds = np.percentile(res_maps, percentiles)
    thresholds_dict = {round(percentile,1): threshold for percentile, threshold in zip(percentiles, thresholds)}'''

    mlflow.end_run()
    logger.info("Exiting...")
