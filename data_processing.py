# -*- coding: utf-8 -*-
"""
data_processing.py

Data processing for convolutional neural networks.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function
from config import ConfigCGAN as config
import numpy as np
import tensorflow as tf
import scipy.misc
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


def preprocess_train_batch(labels, inputs, **kwargs):
    labels_, labels_out = get_shaped_output(labels)
    inputs_, inputs_out = get_shaped_output(inputs)
    for i in range(labels_out.shape[0]):
        labels_out[i], inputs_out[i] = preprocess_train(labels_[i], inputs_[i], **kwargs)
    return labels_out, inputs_out


def preprocess_train(labels, inputs, new_range=(-1, 1), current_range=None, 
                     axis=None, cropping='random', hflip=True, vflip=False):  
    adjust_size = (config.adjust_size, config.adjust_size)
    train_size = (config.train_size, config.train_size)
    # Resize as configured
    labels = resize(labels, adjust_size)
    inputs = resize(inputs, adjust_size)
    # Crop to network input size for training
    if cropping:
        assert config.train_size < config.adjust_size
        crop_fns = {'topleft':crop_topleft, 
                    'center':crop_center,
                    'random':crop_randpos}
        labels, inputs = crop_fns.get(cropping)([labels, inputs], train_size)
    # Random flip half of the time
    labels, inputs = flip_random([labels, inputs], hflip, vflip)
    # Normalise
    labels = normalise(labels, new_range, current_range)
    inputs = normalise(inputs, new_range, current_range)
    return labels.astype('float32'), inputs.astype('float32')


def get_shaped_output(data):
    if len(data.shape) <= 3:
        data_ = data[np.newaxis, ...]
    else:
        data_ = data
    data_out = np.zeros((data_.shape[0],
                         config.train_size,
                         config.train_size,
                         config.channels), dtype=np.float32)
    return data_, data_out


def resize(data, size):
    if config.channels == 1:
        data_ = data.reshape(data.shape[:2])
        data_ = scipy.misc.imresize(data_, size)
        data = data_[..., np.newaxis]
    else:
        data = scipy.misc.imresize(data, size)
    return data


def crop_topleft(data_list, size):
    return crop_from_pos(data_list, (0, 0), size)


def crop_center(data_list, size):
    pos = ((data_list[0].shape[0] - size[0]) // 2, 
           (data_list[0].shape[1] - size[1]) // 2)
    return crop_from_pos(data_list, pos, size)


def crop_randpos(data_list, size):
    pos = (int(np.ceil(np.random.uniform(0.01, data_list[0].shape[0] - size[0]))),
           int(np.ceil(np.random.uniform(0.01, data_list[0].shape[1] - size[1]))))
    return crop_from_pos(data_list, pos, size)


def crop_from_pos(data_list, pos, size):
    return [data_list[i][pos[0]:(pos[0]+size[0]), pos[1]:(pos[1]+size[1])] 
            for i in range(len(data_list))]


def flip_random(data_list, hflip, vflip):
    if hflip and np.random.random() > 0.5:
        data_list = [np.fliplr(data_list[i]) for i in range(len(data_list))]
    if vflip and np.random.random() > 0.5:
        data_list = [np.fliplr(data_list[i]) for i in range(len(data_list))]
    return data_list


def mse(x1, x2, norm=2):
    return tf.reduce_mean(tf.square((x1 - x2) / norm))


def rmse(x1, x2, norm=2):
    return tf.sqrt(mse(x1, x2, norm))


def psnr(x1, x2, max_diff=1):
    return 20. * tf.log(max_diff / rmse(x1, x2)) / tf.log(10.)
