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

    inputs = Input((sz, sz, c), name="ginput")
    inputs_pad = ZeroPadding2D(pad, name="gpad")(inputs)

    # Encoder layers
    ge1 = Conv2D(f, k, s, padding="same", name="geconv1")(inputs_pad)
    
    ge2 = LeakyReLU(config.leak, name="geact1")(ge1)
    ge2 = Conv2D(2*f, k, s, padding="same", name="geconv2")(ge2)
    ge2 = BatchNormalization(name="gebn2")(ge2)
    
    ge3 = LeakyReLU(config.leak, name="geact2")(ge2)
    ge3 = Conv2D(4*f, k, s, padding="same", name="geconv3")(ge3)
    ge3 = BatchNormalization(name="gebn3")(ge3)

    ge4 = LeakyReLU(config.leak, name="geact3")(ge3)
    ge4 = Conv2D(8*f, k, s, padding="same", name="geconv4")(ge4)
    ge4 = BatchNormalization(name="gebn4")(ge4)

    # Decoder layers with skip connections
    gd1 = LeakyReLU(0.0, name="geact4")(ge4)
    # TODO not sure if dimensions need specifying
    gd1 = Conv2DTranspose(4*f, k, s, padding="same", name="gdconv1")(gd1)
    gd1 = BatchNormalization(name="gdbn1")(gd1)
    gd1 = Dropout(config.dropout_rate, name="gddrop1")(gd1)
    gd1 = concatenate([gd1, ge3], axis=3, name="gdcat1")

    gd2 = LeakyReLU(0.0, name="gdact1")(gd1)
    gd2 = Conv2DTranspose(2*f, k, s, padding="same", name="gdconv2")(gd2)
    gd2 = BatchNormalization(name="gdbn2")(gd2)
    gd2 = Dropout(config.dropout_rate, name="gddrop2")(gd2)
    gd2 = concatenate([gd2, ge2], axis=3, name="gdcat2")

    gd3 = LeakyReLU(0.0, name="gdact2")(gd2)
    gd3 = Conv2DTranspose(f, k, s, padding="same", name="gdconv3")(gd3)
    gd3 = BatchNormalization(name="gdbn3")(gd3)
    gd3 = Dropout(config.dropout_rate, name="gddrop3")(gd3)
    gd3 = concatenate([gd3, ge1], axis=3, name="gdcat3")

    gd4 = LeakyReLU(0.0)(gd3)
    gd4 = Conv2DTranspose(c, k, s, padding="same", activation="tanh", 
                          name="gdconvout")(gd4)
    
    outputs = Cropping2D(pad, name="gcrop")(gd4)

    model = Model(inputs=inputs, outputs=outputs, name="cond_gen")
  
    return model


def make_discriminator_model():
    f = config.base_number_of_filters
    k = config.kernel_size
    s = config.strides
    sz = config.raw_size
    c = config.channels

    inputs = Input((sz, sz, c), name="dinput")

    d0 = Conv2D(f, k, s, padding="same", name="dconv0")(inputs)
    d0 = LeakyReLU(config.leak, name="dact0")(d0)
    
    d1 = Conv2D(2*f, k, s, padding="same", name="dconv1")(d0)
    d1 = BatchNormalization(name="dbn1")(d1)
    d1 = LeakyReLU(config.leak, name="dact1")(d1)

    d2 = Conv2D(4*f, k, s, padding="same", name="dconv2")(d1)
    d2 = BatchNormalization(name="dbn2")(d2)
    d2 = LeakyReLU(config.leak, name="dact2")(d2)

    d3 = Conv2D(8*f, k, s, padding="same", name="dconv3")(d2)
    d3 = BatchNormalization(name="dbn3")(d3)
    d3 = LeakyReLU(config.leak, name="dact3")(d3)

    d4 = Flatten(name="dflatout")(d3)

    outputs = Dense(1, name="ddenseout")(d4)

    model = Model(inputs=inputs, outputs=outputs, name="cond_dsc")

    return model


if __name__ == "__main__":
    make_generator_model()
    make_discriminator_model()
