# -*- coding: utf-8 -*-
"""
cgan.py

Conditional Generative Adversarial Network model class.

author: Benjamin Cottier
"""
from __future__ import absolute_import, division, print_function
from config import ConfigCGAN as config
import tensorflow as tf
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    concatenate,
    BatchNormalization,
    LeakyReLU,
    Dropout
)
from tensorflow.keras.activations import relu, tanh


def make_generator_model():
    f = config.base_number_of_filters
    k = config.kernel_size
    s = config.strides

    inputs = Input((config.train_size, config.train_size, config.img_channel))

    ge1 = Conv2D(f, k, s, padding="same")(inputs)
    
    ge2 = LeakyReLU(config.leak)(ge2)
    ge2 = Conv2D(2*f, k, s, padding="same")(ge2)
    ge2 = BatchNormalization()(ge2)
    
    ge3 = LeakyReLU(config.leak)(ge2)
    ge3 = Conv2D(4*f, k, s, padding="same")(ge2)
    ge3 = BatchNormalization()(ge2)

    ge4 = LeakyReLU(config.leak)(ge3)
    ge4 = Conv2D(8*f, k, s, padding="same")(ge3)
    ge4 = BatchNormalization()(ge3)

    gd1 = relu(ge4)
    gd1 = Conv2DTranspose(8*f, k, s)(gd1)
    gd1 = BatchNormalization()(gd1)
    gd1 = Dropout(config.dropout_rate)(gd1)
    gd1 = concatenate([gd1, ge3], axis=3)

    gd2 = relu(gd1)
    gd2 = Conv2DTranspose(4*f, k, s)(gd2)
    gd2 = BatchNormalization()(gd2)
    gd2 = Dropout(config.dropout_rate)(gd2)
    gd2 = concatenate([gd2, ge2], axis=3)

    gd3 = relu(gd2)
    gd3 = Conv2DTranspose(2*f, k, s)(gd3)
    gd3 = BatchNormalization()(gd3)
    gd3 = Dropout(config.dropout_rate)(gd3)
    gd3 = concatenate([gd3, ge1], axis=3)

    gd4 = relu(gd3)
    gd4 = Conv2DTranspose(f, k, s)(gd4)
    
    model = Model(inputs=[inputs], outputs=[tanh(gd4)])
  
    return model


def make_discriminator_model():
    pass
