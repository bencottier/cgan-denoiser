# -*- coding: utf-8 -*-
"""
artefacts.py

Functions to artificially induce artefacts in images. 

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function
from data_processing import normalise
import numpy as np
from scipy import fftpack
from scipy.stats import norm
import math


def add_gaussian_noise(data, stdev=0.1, mean=0.0, data_range=(0, 1), clip=True):
    """
    Add noise to array data, sampled from a normal/Gaussian distribution.

    Assumes the data is limited to the range [-1.0, 1.0].

    Arguments:
        data: ndarray. Data to add noise to.
        stdev: float > 0. Standard deviation of the noise distribution.
        mean: float. Mean (average) of the noise distribution.
        clip: bool. If True, limit the resulting data to [-1.0, 1.0]
    """
    data_ = normalise(data, (-1, 1), data_range)
    noisy = data_ + np.random.normal(mean, stdev, data.shape)
    noisy = np.clip(noisy, -1, 1) if clip else noisy
    return normalise(noisy, data_range, (-1, 1))


def add_space_noise(data, fwhm=1.4, sig=1.2, data_range=(0, 1)):
    """
    ** Requires OpenCV-Python (`cv2`) **
    Add noise to array data, approximating degradation in astronomical 
    telescope imaging. Data must be at least 3D, of shape
    (n, rows, columns, [channel])

    Based on [Schawinski et al., 2017](https://arxiv.org/abs/1702.00403)
    ([pdf](https://arxiv.org/pdf/1702.00403.pdf), 
    [code](https://github.com/SpaceML/GalaxyGAN/blob/master/roou.m)).
    
    Real sources of this degradation include the sensing element,
    the sky background, angular resolution, and atmospheric distortion.
    The approximate model used here convolves the data with a Gaussian 
    point spread function (PSF) and adds Gaussian white noise.

    Arguments:
        data: ndarray. Data to add noise to.
        fwhm: float > 0. Full-width at half maximum of the Gaussian PSF
            in arcseconds.
        sig: float > 0. White noise level.

    Returns:
        The modified data, normalised then stretched with arcsinh.
    """
    import cv2 as cv
    # Constants
    # Conversion to standard deviation: 2*sqrt(2*ln(2)) * stdev ~= 2.355 * stdev
    # https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    FWHM_TO_STDEV = 2.355
    # Pixel scale in the Sloan Digital Sky Survey
    # Slide 3: http://hosting.astro.cornell.edu/academics/courses/astro7620/docs/sdss_shan.pdf
    ARCSEC_TO_PX = 0.396
    ks = 5  # filter kernel size
    nt = 0.1  # noise threshold
    vmin = -0.1  # data value cutoff
    vmax =  4.0  # data value cutoff
    growth_rate = 10.0  # arcsinh stretch factor
    scaling = 3.0  # arcsinh stretch factor
    gaussian_stdev = (fwhm / ARCSEC_TO_PX) / FWHM_TO_STDEV

    if len(data.shape) == 1:
        raise ValueError("Data must be at least 2D")
    data_temp = normalise(data, (-1, 1), data_range).astype('float32')

    for i in range(data_temp.shape[0]):
        x = data_temp[i]
        # Add gaussian blur
        x_blur = cv.GaussianBlur(x, (ks, ks), gaussian_stdev)
        # Estimate the distribution of low-power data
        x_near_zero = x[x > -nt]
        x_near_zero = x_near_zero[x_near_zero < nt]
        m1, s1 = norm.fit(x_near_zero)
        x_blur_near_zero = x_blur[x_blur > -nt]
        x_blur_near_zero = x_blur_near_zero[x_blur_near_zero < nt]
        m2, s2 = norm.fit(x_blur_near_zero)

        # Add noise
        # VAR(X + Y) = VAR(X) + VAR(Y) if X, Y are independent
        noise_var = (sig * s1)**2 - s2**2
        noise_var = 1e-8 if noise_var <= 0 else noise_var
        noise = np.random.normal(0.0, math.sqrt(noise_var), x.shape[:2])
        x_degraded = np.zeros_like(x)
        if len(data_temp.shape) >= 4:
            for c in range(data_temp.shape[-1]):
                x_degraded[..., c] = x_blur[..., c] + noise
        else:
            x_degraded = x_blur + noise

        # Clip
        x_degraded = np.clip(x_degraded, vmin, vmax)
        # Normalise
        x_degraded = (x_degraded - vmin) / (vmax - vmin)
        # arcsinh stretch
        x_degraded = np.arcsinh(growth_rate * x_degraded) / scaling

        data_temp[i] = x_degraded

    return data_temp.astype(data.dtype)


if __name__ == "__main__":
    from scipy.misc import ascent
    import tensorflow as tf
    import matplotlib.pyplot as plt

    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

    # data = ascent()
    data = train_images
    print("Data shape: ", data.shape)
    data = add_gaussian_noise(normalise(data, (-1, 1), (0, 255)), 0.2)
    # plt.imshow(data[0], cmap='gray')

    fig = plt.figure(figsize=(5,5))

    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(data[i], cmap='gray')
        plt.axis('off')

    plt.show()

    data = ascent()
    data = 5.0 * (data - data.min()) / (data.max() - data.min()) + -0.2
    data_degraded = add_space_noise(data, 1.8, 5.0)

    plt.figure()
    plt.gray()
    plt.subplot(1, 2, 1)
    plt.imshow(data)
    plt.subplot(1, 2, 2)
    plt.imshow(data_degraded)
    plt.show()
