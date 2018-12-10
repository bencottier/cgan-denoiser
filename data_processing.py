# -*- coding: utf-8 -*-
"""
data_processing.py

Data processing for convolutional neural networks.

@author: Benjamin Cottier
"""
from __future__ import print_function, division
import filenames
import numpy as np
import nibabel as nib
import math


def normalise(data, new_range=(-1, 1), current_range=None, axis=None):
    """
    Normalise the values of a numpy.ndarray to a specified range.
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
    count = 0
  
    # First n in the below condition is for the case where n is 0 
    if (n and not(n & (n - 1))): 
        return n 
      
    while( n != 0): 
        n >>= 1
        count += 1
      
    return 1<<count


def padding_power_2(shape):
    padded_size = next_power_2(max(shape))
    return ((padded_size - shape[0])//2, (padded_size - shape[1])//2)


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


def get_dataset(input_path, label_path, test_cases, max_training_cases, size):
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
    inputs = np.ndarray((mu, input_rows, input_cols), dtype=np.float32)
    labels = np.ndarray((mu, input_rows, input_cols), dtype=np.float32)
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

        inputs[i] = input_data
        labels[i] = label
        i += 1 
        count += 1
        
        if i == max_training_cases:
            break
    
    inputs = inputs[..., np.newaxis]
    labels = labels[..., np.newaxis]
    test_inputs = test_inputs[..., np.newaxis]
    test_labels = test_labels[..., np.newaxis]
    print("Used", i, "images for training")

    return (inputs, labels), (test_inputs, test_labels), case_list


def mse(x1, x2, norm=2):
    return np.mean(((x1 - x2)/norm)**2)


def rmse(x1, x2, norm=2):
    return math.sqrt(mse(x1, x2, norm))


def psnr(x1, x2, max_diff=1):
    error = rmse(x1, x2)
    psnr_out = 20 * math.log(max_diff / error, 10)
    return psnr_out
