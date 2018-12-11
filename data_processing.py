# -*- coding: utf-8 -*-
"""
data_processing.py

Data processing for convolutional neural networks.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function
import numpy as np
import tensorflow as tf
import math


def normalise(data, new_range=(-1, 1), current_range=None, axis=None):
    """
    Normalise the values of an ndarray to a specified range.

    Arguments:
        data: ndarray. Data to normalise.
        new_range: tuple of int. Value range to normalise to.
        current_range: tuple of int. Value range to normalise from. 
            If not specified, assumes the minimum and maximum values 
            that occur in the data.
        axis: int or tuple of int. Specifies the axes to normalise over if 
            current_range is not specified. For example, if the data is 
            a batch of images, one might want to normalise each image by its 
            respective maximum and minimum value.
    """
    s = new_range[1] - new_range[0]
    if current_range is not None:
        mins = current_range[0]
        maxs = current_range[1]
    elif axis is not None:
        mins = np.nanmin(data, axis=axis, keepdims=True)
        maxs = np.nanmax(data, axis=axis, keepdims=True)   
    else:
        mins = data.min()
        maxs = data.max() 
    return s * (data - mins) / (maxs - mins) + new_range[0]


def next_power_2(n):
    """
    Compute the nearest power of 2 greater than n.

    Arguments:
        n: integer.
    """
    count = 0
    # If it is a non-zero power of 2, return it
    if n and not (n & (n - 1)): 
        return n 
    # Keep dividing n by 2 until it is 0
    while n != 0: 
        n >>= 1
        count += 1
    # Result is 2 to the power of divisions taken
    return 1 << count


def padding_power_2(shape):
    """
    Get the padding required to change the given shape to a square power 
    of 2 in each dimension.

    Arguments:
        shape: tuple of 2 ints. The original shape.
    """
    padded_size = next_power_2(max(shape))
    return ((padded_size - shape[0])//2, (padded_size - shape[1])//2)


def mse(x1, x2, norm=2):
    return tf.reduce_mean(tf.square((x1 - x2) / norm))


def rmse(x1, x2, norm=2):
    return tf.sqrt(mse(x1, x2, norm))


def psnr(x1, x2, max_diff=1):
    return 20. * tf.log(max_diff / rmse(x1, x2)) / tf.log(10.)
