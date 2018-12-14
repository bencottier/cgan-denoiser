# -*- coding: utf-8 -*-
"""
cgan.py

Conditional Generative Adversarial Network model.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function
from config import ConfigCGAN as config
import tensorflow as tf
from data_processing import padding_power_2


def make_generator_model():
    f = config.base_number_of_filters
    k = config.kernel_size
    s = config.strides
    sz = config.train_size
    c = config.channels
    pad = padding_power_2((sz, sz))

    if sz <= 128:
        raise RuntimeError("Input size must be larger than 128 for this U-Net model")

    inputs = tf.keras.layers.Input((sz, sz, c), name="ginput")
    inputs_pad = tf.keras.layers.ZeroPadding2D(pad, name="gpad")(inputs)

    # Encoder layers
    # Input is sz x sz x c
    ge1 = tf.keras.layers.Conv2D(f, k, s, padding="same", name="geconv1")(inputs_pad)
    # Input is sz2 x sz2 x f
    ge2 = tf.keras.layers.LeakyReLU(config.leak, name="geact1")(ge1)
    ge2 = tf.keras.layers.Conv2D(2*f, k, s, padding="same", name="geconv2")(ge2)
    ge2 = tf.keras.layers.BatchNormalization(name="gebn2")(ge2)
    # Input is sz4 x sz4 x 2f
    ge3 = tf.keras.layers.LeakyReLU(config.leak, name="geact2")(ge2)
    ge3 = tf.keras.layers.Conv2D(4*f, k, s, padding="same", name="geconv3")(ge3)
    ge3 = tf.keras.layers.BatchNormalization(name="gebn3")(ge3)
    # Input is sz8 x sz8 x 4f
    ge4 = tf.keras.layers.LeakyReLU(config.leak, name="geact3")(ge3)
    ge4 = tf.keras.layers.Conv2D(8*f, k, s, padding="same", name="geconv4")(ge4)
    ge4 = tf.keras.layers.BatchNormalization(name="gebn4")(ge4)
    # Input is sz16 x sz16 x 8f
    ge5 = tf.keras.layers.LeakyReLU(config.leak, name="geact4")(ge4)
    ge5 = tf.keras.layers.Conv2D(8*f, k, s, padding="same", name="geconv5")(ge5)
    ge5 = tf.keras.layers.BatchNormalization(name="gebn5")(ge5)
    # Input is sz32 x sz32 x 8f
    ge6 = tf.keras.layers.LeakyReLU(config.leak, name="geact5")(ge5)
    ge6 = tf.keras.layers.Conv2D(8*f, k, s, padding="same", name="geconv6")(ge6)
    ge6 = tf.keras.layers.BatchNormalization(name="gebn6")(ge6)
    # Input is sz64 x sz64 x 8f
    ge7 = tf.keras.layers.LeakyReLU(config.leak, name="geact6")(ge6)
    ge7 = tf.keras.layers.Conv2D(8*f, k, s, padding="same", name="geconv7")(ge7)
    ge7 = tf.keras.layers.BatchNormalization(name="gebn7")(ge7)
    # Input is sz128 x sz128 x 8f
    ge8 = tf.keras.layers.LeakyReLU(config.leak, name="geact7")(ge7)
    ge8 = tf.keras.layers.Conv2D(8*f, k, s, padding="same", name="geconv8")(ge8)
    ge8 = tf.keras.layers.BatchNormalization(name="gebn8")(ge8)
    # Input is sz256 x sz256 x 8f

    # Decoder layers with skip connections
    gd1 = tf.keras.layers.LeakyReLU(0.0, name="geact8")(ge8)
    gd1 = tf.keras.layers.Conv2DTranspose(8*f, k, s, padding="same", name="gdconv1")(gd1)
    gd1 = tf.keras.layers.BatchNormalization(name="gdbn1")(gd1)
    gd1 = tf.keras.layers.Dropout(config.dropout_rate, name="gddrop1")(gd1)
    # Input is sz128 x sz128 x 8f
    gd1 = tf.keras.layers.concatenate([gd1, ge7], axis=3, name="gdcat1")
    gd2 = tf.keras.layers.LeakyReLU(0.0, name="gdact1")(gd1)
    gd2 = tf.keras.layers.Conv2DTranspose(8*f, k, s, padding="same", name="gdconv2")(gd2)
    gd2 = tf.keras.layers.BatchNormalization(name="gdbn2")(gd2)
    gd2 = tf.keras.layers.Dropout(config.dropout_rate, name="gddrop2")(gd2)
    # Input is sz64 x sz64 x 8f
    gd2 = tf.keras.layers.concatenate([gd2, ge6], axis=3, name="gdcat2")
    gd3 = tf.keras.layers.LeakyReLU(0.0, name="gdact2")(gd2)
    gd3 = tf.keras.layers.Conv2DTranspose(8*f, k, s, padding="same", name="gdconv3")(gd3)
    gd3 = tf.keras.layers.BatchNormalization(name="gdbn3")(gd3)
    gd3 = tf.keras.layers.Dropout(config.dropout_rate, name="gddrop3")(gd3)
    # Input is sz32 x sz32 x 8f
    gd3 = tf.keras.layers.concatenate([gd3, ge5], axis=3, name="gdcat3")
    gd4 = tf.keras.layers.LeakyReLU(0.0, name="gdact3")(gd3)
    gd4 = tf.keras.layers.Conv2DTranspose(8*f, k, s, padding="same", name="gdconv4")(gd4)
    gd4 = tf.keras.layers.BatchNormalization(name="gdbn4")(gd4)
    # Input is sz16 x sz16 x 8f
    gd4 = tf.keras.layers.concatenate([gd4, ge4], axis=3, name="gdcat4")
    gd5 = tf.keras.layers.LeakyReLU(0.0, name="gdact4")(gd4)
    gd5 = tf.keras.layers.Conv2DTranspose(4*f, k, s, padding="same", name="gdconv5")(gd5)
    gd5 = tf.keras.layers.BatchNormalization(name="gdbn5")(gd5)
    gd5 = tf.keras.layers.concatenate([gd5, ge3], axis=3, name="gdcat5")
    # Input is sz8 x sz8 x 4f
    gd6 = tf.keras.layers.LeakyReLU(0.0, name="gdact5")(gd5)
    gd6 = tf.keras.layers.Conv2DTranspose(2*f, k, s, padding="same", name="gdconv6")(gd6)
    gd6 = tf.keras.layers.BatchNormalization(name="gdbn6")(gd6)
    # Input is sz4 x sz4 x 2f
    gd6 = tf.keras.layers.concatenate([gd6, ge2], axis=3, name="gdcat6")
    gd7 = tf.keras.layers.LeakyReLU(0.0, name="gdact6")(gd6)
    gd7 = tf.keras.layers.Conv2DTranspose(f, k, s, padding="same", name="gdconv7")(gd7)
    gd7 = tf.keras.layers.BatchNormalization(name="gdbn7")(gd7)
    # Input is sz2 x sz2 x f
    gd7 = tf.keras.layers.concatenate([gd7, ge1], axis=3, name="gdcat7")
    gd8 = tf.keras.layers.LeakyReLU(0.0)(gd7)
    gd8 = tf.keras.layers.Conv2DTranspose(c, k, s, padding="same", activation="tanh", 
                          name="gdconvout")(gd8)
    # Input is sz x sz x nc
    outputs = tf.keras.layers.Cropping2D(pad, name="gcrop")(gd8)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="cond_gen")
  
    return model


def make_generator_model_small():
    f = config.base_number_of_filters
    k = config.kernel_size
    s = config.strides
    sz = config.train_size
    c = config.channels
    pad = padding_power_2((sz, sz))

    inputs = tf.keras.layers.Input((sz, sz, c), name="ginput")
    inputs_pad = tf.keras.layers.ZeroPadding2D(pad, name="gpad")(inputs)

    # Encoder layers
    ge1 = tf.keras.layers.Conv2D(f, k, s, padding="same", name="geconv1")(inputs_pad)
    
    ge2 = tf.keras.layers.LeakyReLU(config.leak, name="geact1")(ge1)
    ge2 = tf.keras.layers.Conv2D(2*f, k, s, padding="same", name="geconv2")(ge2)
    ge2 = tf.keras.layers.BatchNormalization(name="gebn2")(ge2)
    
    ge3 = tf.keras.layers.LeakyReLU(config.leak, name="geact2")(ge2)
    ge3 = tf.keras.layers.Conv2D(4*f, k, s, padding="same", name="geconv3")(ge3)
    ge3 = tf.keras.layers.BatchNormalization(name="gebn3")(ge3)

    ge4 = tf.keras.layers.LeakyReLU(config.leak, name="geact3")(ge3)
    ge4 = tf.keras.layers.Conv2D(8*f, k, s, padding="same", name="geconv4")(ge4)
    ge4 = tf.keras.layers.BatchNormalization(name="gebn4")(ge4)

    # Decoder layers with skip connections
    gd1 = tf.keras.layers.LeakyReLU(0.0, name="geact4")(ge4)
    # TODO not sure if dimensions need specifying
    gd1 = tf.keras.layers.Conv2DTranspose(4*f, k, s, padding="same", name="gdconv1")(gd1)
    gd1 = tf.keras.layers.BatchNormalization(name="gdbn1")(gd1)
    gd1 = tf.keras.layers.Dropout(config.dropout_rate, name="gddrop1")(gd1)
    gd1 = tf.keras.layers.concatenate([gd1, ge3], axis=3, name="gdcat1")

    gd2 = tf.keras.layers.LeakyReLU(0.0, name="gdact1")(gd1)
    gd2 = tf.keras.layers.Conv2DTranspose(2*f, k, s, padding="same", name="gdconv2")(gd2)
    gd2 = tf.keras.layers.BatchNormalization(name="gdbn2")(gd2)
    gd2 = tf.keras.layers.Dropout(config.dropout_rate, name="gddrop2")(gd2)
    gd2 = tf.keras.layers.concatenate([gd2, ge2], axis=3, name="gdcat2")

    gd3 = tf.keras.layers.LeakyReLU(0.0, name="gdact2")(gd2)
    gd3 = tf.keras.layers.Conv2DTranspose(f, k, s, padding="same", name="gdconv3")(gd3)
    gd3 = tf.keras.layers.BatchNormalization(name="gdbn3")(gd3)
    gd3 = tf.keras.layers.Dropout(config.dropout_rate, name="gddrop3")(gd3)
    gd3 = tf.keras.layers.concatenate([gd3, ge1], axis=3, name="gdcat3")

    gd4 = tf.keras.layers.LeakyReLU(0.0)(gd3)
    gd4 = tf.keras.layers.Conv2DTranspose(c, k, s, padding="same", activation="tanh", 
                          name="gdconvout")(gd4)
    
    outputs = tf.keras.layers.Cropping2D(pad, name="gcrop")(gd4)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="cond_gen")
  
    return model


def make_discriminator_model():
    f = config.base_number_of_filters
    k = config.kernel_size
    s = config.strides
    sz = config.train_size
    c = config.channels

    inputs = tf.keras.layers.Input((sz, sz, c), name="dinput")

    d0 = tf.keras.layers.Conv2D(f, k, s, padding="same", name="dconv0")(inputs)
    d0 = tf.keras.layers.LeakyReLU(config.leak, name="dact0")(d0)
    
    d1 = tf.keras.layers.Conv2D(2*f, k, s, padding="same", name="dconv1")(d0)
    d1 = tf.keras.layers.BatchNormalization(name="dbn1")(d1)
    d1 = tf.keras.layers.LeakyReLU(config.leak, name="dact1")(d1)

    d2 = tf.keras.layers.Conv2D(4*f, k, s, padding="same", name="dconv2")(d1)
    d2 = tf.keras.layers.BatchNormalization(name="dbn2")(d2)
    d2 = tf.keras.layers.LeakyReLU(config.leak, name="dact2")(d2)

    d3 = tf.keras.layers.Conv2D(8*f, k, s, padding="same", name="dconv3")(d2)
    d3 = tf.keras.layers.BatchNormalization(name="dbn3")(d3)
    d3 = tf.keras.layers.LeakyReLU(config.leak, name="dact3")(d3)

    d4 = tf.keras.layers.Flatten(name="dflatout")(d3)

    outputs = tf.keras.layers.Dense(1, name="ddenseout")(d4)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name="cond_dsc")

    return model


if __name__ == "__main__":
    make_generator_model()
    make_discriminator_model()
