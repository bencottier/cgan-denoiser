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
import scipy.misc
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
    labels_, labels_out = get_shaped_output(labels)
    inputs_, inputs_out = get_shaped_output(inputs)
    for i in range(labels_out.shape[0]):
        labels_out[i], inputs_out[i] = preprocess_train(labels_[i], inputs_[i], **kwargs)
    return labels_out, inputs_out


def preprocess_train(labels, inputs, new_range=(-1, 1), current_range=None, 
                     axis=None, cropping='random', hflip=1, vflip=0):  
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
    # Image flipping
    labels, inputs = flip([labels, inputs], hflip, vflip)
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
        data_list = [np.fliplr(data_list[i]) for i in range(len(data_list))]
    if (vflip == 1 and np.random.random() > 0.5) or vflip == 2:
        data_list = [np.fliplr(data_list[i]) for i in range(len(data_list))]
    return data_list


def mse(x1, x2, norm=2):
    return tf.reduce_mean(tf.square((x1 - x2) / norm))


def rmse(x1, x2, norm=2):
    return tf.sqrt(mse(x1, x2, norm))


def psnr(x1, x2, max_diff=1):
    return 20. * tf.log(max_diff / rmse(x1, x2)) / tf.log(10.)


def preprocess(data, pad_size):
    """
    Process the data into an appropriate format for CNN-based models.
    """
    # Reverse the flipping about the x-axis caused by data reading
    data = np.flipud(data)

    # Ensure it is padded to square
    data_rows, data_cols = data.shape
    # print("Original shape", (data_rows, data_cols))
    preprocessed_data = np.zeros( (pad_size, pad_size), data.dtype)
    preprocessed_data[:data_rows, :data_cols] = data
    # print("Padded shape", preprocessed_data.shape)
    # Normalise to suitable value range
    preprocessed_data = normalise(preprocessed_data)
    return preprocessed_data


def get_oasis_dataset(input_path, label_path, test_cases, max_training_cases, size):
    # Load input data
    # Get list of filenames and case IDs from path where 3D volumes are
    input_list, case_list = filenames.getSortedFileListAndCases(
        input_path, 0, "*.nii.gz", True)
    # print("Cases: {}".format(case_list))

    # Load label data
    # Get list of filenames and case IDs from path where 3D volumes are
    label_list, case_list = filenames.getSortedFileListAndCases(
        label_path, 0, "*.nii.gz", True)
    
    # Check number of inputs
    mu = len(input_list)
    if mu != len(label_list):
        print("Warning: inputs and labels don't match!")
    if mu < 1:
        print("Error: No input data found. Exiting.")
        quit()
        
    if test_cases:
        mu -= len(test_cases) #leave 1 outs
    if max_training_cases > 0:
        mu = max_training_cases
        
    print("Loading input ...")
    # Get initial input size, assumes all same size
    first = nib.load(input_list[0]).get_data().astype(np.float32)
    first = preprocess(first, size)
    # print("Input shape:", first.shape)
    input_rows, input_cols = first.shape
    
    # Use 3D array to store all inputs and labels
    train_inputs = np.ndarray((mu, input_rows, input_cols), dtype=np.float32)
    train_labels = np.ndarray((mu, input_rows, input_cols), dtype=np.float32)
    test_inputs = np.ndarray((len(test_cases), input_rows, input_cols), dtype=np.float32)
    test_labels = np.ndarray((len(test_cases), input_rows, input_cols), dtype=np.float32)
    
    # Process each 3D volume
    i = 0
    count = 0
    out_count = 0
    for input_name, label, _ in zip(input_list, label_list, case_list):
        # Load MR nifti file
        input_data = nib.load(input_name).get_data().astype(np.float32)
        input_data = preprocess(input_data, size) 
        # print("Slice shape:", input_data.shape)
        # print("Loaded", image)

        # Load label nifti file
        label = nib.load(label).get_data().astype(np.float32)
        label = preprocess(label, size) 
        # label[label > 0] = 1.0  # binary threshold labels
        # print("Slice shape:", label.shape)
        # print("Loaded", label)

        # Check for test case
        if count in test_cases:
            test_inputs[out_count] = input_data
            test_labels[out_count] = label
            count += 1
            out_count += 1
            continue

        train_inputs[i] = input_data
        train_labels[i] = label
        i += 1 
        count += 1
        
        if i == max_training_cases:
            break
    
    # labels, inputs = preprocess_train_batch(labels, inputs, 
    #                                         new_range=(-1, 1),
    #                                         current_range=None, 
    #                                         axis=None, cropping='topleft', 
    #                                         hflip=0, vflip=2)

    train_inputs = train_inputs[..., np.newaxis]
    train_labels = train_labels[..., np.newaxis]
    test_inputs = test_inputs[..., np.newaxis]
    test_labels = test_labels[..., np.newaxis]

    print("Used", i, "images for training")

    return (train_inputs, train_labels), (test_inputs, test_labels), case_list


def get_oasis_dataset_test(input_path, label_path, test_cases, max_test_cases, size):
    # Load input data
    # Get list of filenames and case IDs from path where 3D volumes are
    input_list, case_list = filenames.getSortedFileListAndCases(
        input_path, 0, "*.nii.gz", True)
    # print("Cases: {}".format(case_list))

    # Load label data
    # Get list of filenames and case IDs from path where 3D volumes are
    label_list, case_list = filenames.getSortedFileListAndCases(
        label_path, 0, "*.nii.gz", True)
    
    # Check number of inputs
    if len(input_list) != len(label_list):
        print("Warning: inputs and labels don't match!")
    if len(input_list) < 1:
        print("Error: No input data found. Exiting.")
        quit()
        
    print("Loading input ...")
    # Get initial input size, assumes all same size
    first = nib.load(input_list[0]).get_data().astype(np.float32)
    first = preprocess(first, size)
    # print("Input shape:", first.shape)
    input_rows, input_cols = first.shape
    
    # Use 3D array to store all inputs and labels
    test_inputs = np.ndarray((len(test_cases), input_rows, input_cols), dtype=np.float32)
    test_labels = np.ndarray((len(test_cases), input_rows, input_cols), dtype=np.float32)
    
    # Process each 3D volume
    count = 0
    out_count = 0
    for input_name, label, _ in zip(input_list, label_list, case_list):
        # Check for test case
        if count in test_cases:
            # Load MR nifti file
            input_data = nib.load(input_name).get_data().astype(np.float32)
            input_data = preprocess(input_data, size) 
            # print("Slice shape:", input_data.shape)
            # print("Loaded", image)

            # Load label nifti file
            label = nib.load(label).get_data().astype(np.float32)
            label = preprocess(label, size) 
            # label[label > 0] = 1.0  # binary threshold labels
            # print("Slice shape:", label.shape)
            # print("Loaded", label)

            test_inputs[out_count] = input_data
            test_labels[out_count] = label
            out_count += 1
            if out_count == max_test_cases:
                break
        count += 1

    test_inputs = test_inputs[..., np.newaxis]
    test_labels = test_labels[..., np.newaxis]

    print("Used {} images for testing".format(out_count))

    return test_inputs, test_labels
