# -*- coding: utf-8 -*-
"""
data_processing.py

MRI data preprocessing for convolutional neural networks.

@author: Benjamin Cottier
"""
from __future__ import print_function, division
import filenames
import numpy as np
import nibabel as nib


def normalise(a, r=(-1, 1)):
    """
    Normalise the values of the numpy.ndarray, a, to the range r.
    """
    return (r[1] - r[0])*(a - a.min())/(a.max() - a.min()) - r[0]


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
