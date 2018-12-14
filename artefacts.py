# -*- coding: utf-8 -*-
"""
artefacts.py

Functions to artificially induce artefacts in images. 

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function
from data_processing import normalise
import numpy as np


def add_gaussian_noise(data, stdev=0.1, mean=0.0, data_range=(0, 1), clip=True):
    """
    Add noise to array data, sampled from a normal/Gaussian distribution.

    Assumes the data is limited to the range [-1.0, 1.0].

    Arguments:
        data: ndarray. Data to add noise to.
        stdev: float > 0. Standard deviation of the noise distribution.
        mean: float. Mean (average) of the noise distribution.
        clamping: bool. If True, limit the resulting data to [-1.0, 1.0]
    """
    data_ = normalise(data, (-1, 1), data_range)
    noisy = data_ + np.random.normal(mean, stdev, data.shape)
    noisy = np.clip(noisy, -1, 1) if clip else noisy
    return normalise(noisy, data_range, (-1, 1))


def clamp(x, rng=(-1, 1)):
    return np.maximum(rng[0], np.minimum(x, rng[1]))


if __name__ == "__main__":
    # Test
    from scipy.misc import ascent
    import tensorflow as tf
    import matplotlib.pyplot as plt

    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    # data = ascent()
    data = train_images
    print("Data shape: ", data.shape)
    data = add_gaussian_noise(normalise(data, (-1, 1), (0, 255)), 0.2)
    # plt.imshow(data[0], cmap='gray')

    fig = plt.figure(figsize=(4,4))

    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(data[i, :, :], cmap='gray')
        plt.axis('off')

    plt.show()
