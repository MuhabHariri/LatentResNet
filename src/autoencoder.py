import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from src.config import *
from tensorflow.keras import layers, models
from src.layers import ECALayer


def LiteAE(input_shape=INPUT_SHAPE):
    input_img = layers.Input(shape=input_shape)
    x = input_img

    # ------------------------------- Encoder -------------------------------
    for filters in ENCODER_FILTERS:
        x = layers.SeparableConv2D(filters, KERNEL_SIZE, strides=STRIDES, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
        x = layers.SpatialDropout2D(DROPOUT_RATE)(x)
        x = ECALayer(filters)(x)

    encoded = x  #Save encoded representation

    # ------------------------------- Decoder -------------------------------------
    for filters in reversed(ENCODER_FILTERS):
        x = layers.Conv2DTranspose(filters, KERNEL_SIZE, strides=STRIDES, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)
        x = layers.SpatialDropout2D(DROPOUT_RATE)(x)
        x = ECALayer(filters)(x)

    # Final reconstruction layer
    x = layers.Conv2D(OUTPUT_CHANNELS, KERNEL_SIZE, padding='same')(x)
    decoded = layers.LeakyReLU(alpha=LEAKY_RELU_ALPHA)(x)

    return models.Model(input_img, decoded, name="Autoencoder")

# Instantiate the model
autoencoder = LiteAE()

