# -*- coding: utf-8 -*-
"""
artefacts.py

Functions to artificially induce artefacts in images. 

@author: Benjamin Cottier
"""
from __future__ import print_function, division
from data_processing import normalise
import numpy as np


def add_gaussian_noise(data, stdev=0.1, mean=0.0, clamping=True):
    noisy = data + np.random.normal(mean, stdev, data.shape)
    return clamp(noisy) if clamping else noisy


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
