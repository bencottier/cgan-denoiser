# -*- coding: utf-8 -*-
"""
cgan.py

Conditional Generative Adversarial Network model class.

author: Benjamin Cottier
"""
from __future__ import absolute_import, division, print_function
from config import ConfigCGAN as config
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    concatenate,
    BatchNormalization,
    LeakyReLU,
    Dropout,
    ZeroPadding2D,
    Cropping2D,
    Reshape,
    Dense,
    Flatten,
)
from tensorflow.keras.activations import relu, tanh
from data_processing import padding_power_2


def make_generator_model():
    f = config.base_number_of_filters
    k = config.kernel_size
    s = config.strides
    sz = config.raw_size
    c = config.channels
    pad = padding_power_2((sz, sz))

    # TODO add axis to MNIST data for compatibility

    inputs = Input((sz, sz, c))
    inputs_pad = ZeroPadding2D(pad)(inputs)

    # Encoder layers
    ge1 = Conv2D(f, k, s, padding="same")(inputs_pad)
    
    ge2 = LeakyReLU(config.leak)(ge1)
    ge2 = Conv2D(2*f, k, s, padding="same")(ge2)
    ge2 = BatchNormalization()(ge2)
    
    ge3 = LeakyReLU(config.leak)(ge2)
    ge3 = Conv2D(4*f, k, s, padding="same")(ge3)
    ge3 = BatchNormalization()(ge3)

    ge4 = LeakyReLU(config.leak)(ge3)
    ge4 = Conv2D(8*f, k, s, padding="same")(ge4)
    ge4 = BatchNormalization()(ge4)

    # Decoder layers with skip connections
    gd1 = LeakyReLU(0.0)(ge4)
    # TODO not sure if dimensions need specifying
    gd1 = Conv2DTranspose(4*f, k, s, padding="same")(gd1)
    gd1 = BatchNormalization()(gd1)
    gd1 = Dropout(config.dropout_rate)(gd1)
    gd1 = concatenate([gd1, ge3], axis=3)

    gd2 = LeakyReLU(0.0)(gd1)
    gd2 = Conv2DTranspose(2*f, k, s, padding="same")(gd2)
    gd2 = BatchNormalization()(gd2)
    gd2 = Dropout(config.dropout_rate)(gd2)
    gd2 = concatenate([gd2, ge2], axis=3)

    gd3 = LeakyReLU(0.0)(gd2)
    gd3 = Conv2DTranspose(f, k, s, padding="same")(gd3)
    gd3 = BatchNormalization()(gd3)
    gd3 = Dropout(config.dropout_rate)(gd3)
    gd3 = concatenate([gd3, ge1], axis=3)

    gd4 = LeakyReLU(0.0)(gd3)
    gd4 = Conv2DTranspose(c, k, s, padding="same", activation="tanh")(gd4)
    
    outputs = Cropping2D(pad)(gd4)

    model = Model(inputs=inputs, outputs=outputs)
  
    return model


def make_discriminator_model():
    f = config.base_number_of_filters
    k = config.kernel_size
    s = config.strides
    sz = config.raw_size
    c = config.channels

    inputs = Input((sz, sz, 2*c))  # 2 for real and generated samples

    d0 = Conv2D(f, k, s, padding="same")(inputs)
    d0 = LeakyReLU(config.leak)(d0)
    
    d1 = Conv2D(2*f, k, s, padding="same")(d0)
    d1 = BatchNormalization()(d1)
    d1 = LeakyReLU(config.leak)(d1)

    d2 = Conv2D(4*f, k, s, padding="same")(d1)
    d2 = BatchNormalization()(d2)
    d2 = LeakyReLU(config.leak)(d2)

    d3 = Conv2D(8*f, k, s, padding="same")(d2)
    d3 = BatchNormalization()(d3)
    d3 = LeakyReLU(config.leak)(d3)

    d4 = Flatten()(d3)

    outputs = Dense(1)(d4)

    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == "__main__":
    make_generator_model()
    make_discriminator_model()
