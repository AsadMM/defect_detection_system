# -*- coding: utf-8 -*-
"""
Created on Tue May 21 14:48:54 2024

@author: asadm
"""

# import the necessary packages
import logging

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import numpy as np


logger = logging.getLogger(__name__)


def build1(width: int, height: int, depth: int, filters: tuple | list = (32, 64, 96), latentDim: int = 100) -> Model:
    """
    Builds the autoencoder model network input image dimensions and filters.

    Parameters
    ----------
    width : int
        width of the input image.
    height : int
        height of the input image.
    depth : int
        no of channels of the input image.
    filters : tuple|list, optional
        List of filters to be applied, one conv2D layer is added for each filter.
        The default is (32, 64, 96).
    latentDim : int, optional
        The dimension of the compressed code space of the autoencoder, a dense layer
        is created based on this. The default is 100.

    Returns
    -------
    Model
        Tensorflow.keras model file with built network.

    """
    # initialize the input shape to be "channels last" along with
    # the channels dimension itself
    inputShape = (height, width, depth)
    chanDim = -1
    # define the input to the encoder
    inputs = Input(shape=inputShape)
    x = inputs
    # loop over the number of filters
    for f in filters:
        # apply a CONV => RELU => BN operation
        x = Conv2D(f, (3, 3), strides=2, padding="same",
                   activation=LeakyReLU(negative_slope=0.2))(x)
        x = BatchNormalization(axis=chanDim)(x)
    # flatten the network and then construct our latent vector
    volumeSize = K.int_shape(x)
    logger.info("Encoder volume size: %s", volumeSize)
    x = Flatten()(x)
    latent = Dense(latentDim)(x)
    # build the encoder model
    encoder = Model(inputs, latent, name="encoder")
    # start building the decoder model which will accept the
    # output of the encoder as its inputs
    latentInputs = Input(shape=(latentDim,))
    decoder_shape = volumeSize[1:]
    if any(dim is None for dim in decoder_shape):
        raise ValueError(f"Encoder output shape must be fully defined, got: {volumeSize}")
    decoder_units = int(np.prod(decoder_shape))
    if decoder_units <= 0:
        raise ValueError(f"Decoder units must be positive, got: {decoder_units}")
    x = Dense(decoder_units)(latentInputs)
    x = Reshape(tuple(int(dim) for dim in decoder_shape))(x)
    # loop over our number of filters again, but this time in
    # reverse order
    for f in filters[::-1]:
        # apply a CONV_TRANSPOSE => RELU => BN operation
        x = Conv2DTranspose(f, (3, 3), strides=2,
                            padding="same", activation=LeakyReLU(negative_slope=0.2))(x)
        x = BatchNormalization(axis=chanDim)(x)
    # apply a single CONV_TRANSPOSE layer used to recover the
    # original depth of the image
    x = Conv2DTranspose(depth, (3, 3), padding="same")(x)
    outputs = Activation("sigmoid")(x)
    # build the decoder model
    decoder = Model(latentInputs, outputs, name="decoder")
    # our autoencoder is the encoder + decoder
    autoencoder = Model(inputs, decoder(encoder(inputs)),
                        name="autoencoder")
    # return a 3-tuple of the encoder, decoder, and autoencoder
    return autoencoder


# This network is just an alternative to the first one it's not being used
def build2(width, height, depth, filters=32, latentDim=100):
    input_img = Input(shape=(height, width, depth))

    h = Conv2D(filters, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(input_img)
    h = Conv2D(filters, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(filters, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(filters, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(filters * 2, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(filters * 2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(filters * 4, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(filters * 2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(filters, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    encoded = Conv2D(latentDim, (8, 8), strides=1, activation='linear', padding='valid')(h)

    h = Conv2DTranspose(filters, (8, 8), strides=1, activation=LeakyReLU(alpha=0.2), padding='valid')(encoded)
    h = Conv2D(filters * 2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(filters * 4, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2DTranspose(filters * 2, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(filters * 2, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2DTranspose(filters, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2D(filters, (3, 3), strides=1, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2DTranspose(filters, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)
    h = Conv2DTranspose(filters, (4, 4), strides=2, activation=LeakyReLU(alpha=0.2), padding='same')(h)

    decoded = Conv2DTranspose(depth, (4, 4), strides=2, activation='sigmoid', padding='same')(h)
    return Model(input_img, decoded)
