# -*- coding: utf-8 -*-
"""
data_processing.py

Data processing for convolutional neural networks.

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function
from config import Config as config
import filenames
import nibabel as nib
import numpy as np
import tensorflow as tf
from PIL import Image
from skimage.measure import compare_ssim
import math
import os


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
    Compute the nearest power of 2 greater than or equal to n.

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
    for i in range(labels.shape[0]):
        labels[i], inputs[i] = preprocess_train(labels[i], inputs[i], **kwargs)
    return labels, inputs


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


def preprocess_train(labels, inputs, new_range=(-1, 1), current_range=None, 
                     axis=None, cropping=None, hflip=0, vflip=0, 
                     max_translate=0, max_rotate=0):
    # TODO use a well-vetted data augmentation library
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
    # Flip
    if hflip or vflip:
        labels, inputs = flip([labels, inputs], hflip, vflip)
    # Translate
    if max_translate:
        labels, inputs = translate_random([labels, inputs], max_translate)
    # Rotate
    if max_rotate < 0:
        labels, inputs = rotate_rand90([labels, inputs])
    elif max_rotate > 0:
        labels, inputs = rotate_random([labels, inputs], max_rotate)
    # Normalise
    labels = normalise(labels, new_range, current_range)
    inputs = normalise(inputs, new_range, current_range)
    return labels.astype('float32'), inputs.astype('float32')


def resize(data, size):
    if config.channels == 1:
        data_ = data.reshape(data.shape[:2])
        data_ = np.array(Image.fromarray(data_, mode='F').resize(size))
        data = data_[..., np.newaxis]
    else:
        data = np.array(Image.fromarray(data, mode='F').resize())
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


def flip(data_list, hflip, vflip):
    """
    Flip each image in a list.

    Arguments:
        data_list: ndarray. List of arrays, each to be flipped.
        hflip: int. Horizontal flip (1 means random 50/50, 2 means systematic, 
            else no flip) 
        vflip: int. Vertical flip (1 means random 50/50, 2 means systematic, 
            else no flip) 
    """
    if (hflip == 1 and np.random.random() > 0.5) or hflip == 2:
        data_list = [np.fliplr(d) for d in data_list]
    if (vflip == 1 and np.random.random() > 0.5) or vflip == 2:
        data_list = [np.fliplr(d) for d in data_list]
    return data_list


def translate_random(data_list, max_translate):
    tx = np.random.randint(-max_translate, max_translate + 1)
    ty = np.random.randint(-max_translate, max_translate + 1)
    data_list = [translate(d, tx, ty) for d in data_list]
    return data_list


def translate(a, tx, ty):
    ax, ay = a.shape
    a_ = np.zeros_like(a)

    def translation_bounds(tx, ty, ax, ay):
        return max(tx, 0), min(ax + tx, ax), max(ty, 0), min(ay + ty, ay)

    x1, x2, y1, y2 = translation_bounds(-tx, -ty, ax, ay)
    x3, x4, y3, y4 = translation_bounds(tx, ty, ax, ay)

    a_[x3:x4, y3:y4] = a[x1:x2, y1:y2]
    return a_


def rotate_random(data_list, max_rotate):
    angle = np.random.uniform(-max_rotate, max_rotate)
    data_list = [rotate(d, angle) for d in data_list]
    return data_list


def rotate_rand90(data_list):
    if np.random.random() > 0.5:
        data_list = [rotate(d, 90) for d in data_list]
    return data_list


# def rotate(a, degrees):
#     return skimage.transform.rotate(a, degrees)


def imbound(im, bounds=None, center=True):
    """
    Crop/pad the image to the given bounds.

    Arguments:
        im: ndarray. Input image, at least 2D with width and height last.
        bounds: tuple. New bounds to pad/crop to.
        center: bool. When True, pad/crop about the center.
    """
    if not bounds:
        m = max(im.shape[-2:])
        shape_end = (m, m)
    else:
        shape_end = bounds
    bounds = list(im.shape[:-2])
    bounds.extend(shape_end)
    bounds = tuple(bounds)
    # Create a larger-dimension array to store the image
    cropped = np.zeros(bounds, dtype=im.dtype)

    a, b = bounds[-2:]
    c, d = im.shape[-2:]

    if center:
        # Place image values about the centre of the new larger image

        x_comp = min(a, c)
        y_comp = min(b, d)

        x_cl = (a - x_comp) // 2
        x_cr = (a + x_comp) // 2
        y_cl = (b - y_comp) // 2
        y_cr = (b + y_comp) // 2
        x_il = (c - x_comp) // 2
        x_ir = (c + x_comp) // 2
        y_il = (d - y_comp) // 2
        y_ir = (d + y_comp) // 2

        cropped[..., x_cl:x_cr, y_cl:y_cr] = im[..., x_il:x_ir, y_il:y_ir]
    else:
        # Place image values at top left corner of the new larger image
        cropped[..., :min(a, c), :min(b, d)] = im[..., :min(a, c), :min(b, d)]
    return cropped


def mse(x1, x2, norm=2.):
    return tf.reduce_mean(tf.square((x1 - x2) / norm))


def rmse(x1, x2, norm=2.):
    return tf.sqrt(mse(x1, x2, norm))


def psnr(x1, x2, max_diff=2.):
    return 20. * tf.log(1. / rmse(x1, x2, max_diff)) / tf.log(10.)


def ssim(x1, x2, max_diff=2.):
    try:
        x1 = x1.numpy()
    except: pass
    try:
        x2 = x2.numpy()
    except: pass
    x1, x2 = x1[0, :, :, 0], x2[0, :, :, 0]
    return compare_ssim(x1, x2, data_range=max_diff)
