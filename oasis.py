# -*- coding: utf-8 -*-
"""
oasis.py

Methods to prepare data from OASIS Brains (https://www.oasis-brains.org/)

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function
from config import Config as config
import data_processing
import artefacts
import filenames
import utils
from PIL import Image
import png
import imageio
from scipy import io as sio
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import difflib
import os
import time


MAX_UINT8 = 255
MAX_UINT16 = 65535


def string_minus(a, b):
    """
    Return a string composed of characters in string `b` 
    that are not in string `a` (NB: not vice-versa).
    """
    diff = ''
    for s in difflib.ndiff(a, b):
        if s[0] == '+': diff += s[-1]
    return diff


def get_subject_number(filename):
    # Examples
    # sub-OAS30032_ses-d3499_T1w.nii.gz
    # sub-OAS30001_ses-d0129_run-01_T1w_00100.png
    try:
        return int(filename[8:12])
    except:
        raise TypeError('Expected a string convertible to int: {}'.format(filename[8:12]))


def get_scan_paths(data_path='.', file_type='nii', scan_type='', selected_runs=(1,)):
    """
    Find the path for all scan files matching certain criteria.

    Arguments:
        data_path: string. Top-level path to search for files.
        file_type: string. File extension for the desired scans.
        scan_type: string. Type of scan, e.g. T1w, T2w, bold, FLAIR.
        selected_runs: tuple of int. If a scan has multiple runs 
            (indicated with a run number in the filename), select only 
            these run numbers.
    Returns:
        Valid scan file paths as a list of strings.
    """
    # Find MR data files
    data_files = []
    for root, dirs, files in os.walk(data_path):
        for cfile in sorted(files):
            if ".{}".format(file_type) in cfile:
                data_file = os.path.join(root, cfile)
                data_files.append(data_file)

    # Get specific data files for experiment
    exp_data_files = []
    for data_file in data_files:
        # Check for correct scan type e.g. T1-weighted, FLAIR
        if scan_type not in data_file:
            continue
        # Check if there are multiple runs and if this file is the desired run
        run_pos = data_file.find('run-')
        if run_pos != -1:
            try:
                run_num = int(data_file[run_pos + 4: run_pos + 6])
                if run_num not in selected_runs:
                    continue
            except TypeError:
                print('Expected a run number: {}'.format(
                    data_file[run_pos + 4: run_pos + 6]))
        # All clear, this is one of the files we want
        exp_data_files.append(data_file)

    return exp_data_files


def get_nifti_files(data_path, max_scans, skip, shape, 
        data_files=None, resume=None, last_subject=None):
    if data_files is None:
        data_files = get_scan_paths(data_path, 'nii', 'T1w', (1, 2))
    # Check for files meeting criteria
    if type(resume) != int or resume < 0:
        resume = 0
    else:
        print("Resuming from file #{}".format(resume))
    new_resume = None
    skipped = 0
    count = 0
    accepted_files = []
    for i in range(resume, len(data_files)):
        data_file = data_files[i]
        if count >= max_scans:
            break
        # print(nib.load(data_file).get_fdata().shape)
        if shape is None or nib.load(data_file).get_fdata().shape == shape:
            if skipped < skip:
                skipped += 1
                continue
            if last_subject is not None:
                if get_subject_number(data_file.split('/')[-1]) == last_subject:
                    skipped += 1
                    continue
            accepted_files.append(data_file)
            count += 1
    new_resume = i
    print("Found {} volumes matching criteria".format(count))
    return accepted_files, skipped, new_resume


def save_slices(volume, save_path, base_index, save_artefacts=False, category='train', fmt=None):
    # TODO artefacts option

    # Create file name (to be formatted later)
    if fmt == 'jpg':
        save_path += '.jpg'
    elif fmt == 'png':
        save_path += '.png'
    else:
        save_path += '.nii.gz'

    data = np.zeros_like(volume)

    for i, target in enumerate(volume):
        if fmt == 'jpg':
            target = data_processing.normalise(target, (0, MAX_UINT8))
        elif fmt == 'png':
            target = data_processing.normalise(target, (0, MAX_UINT16))
        data[i] = target

    for i in range(len(data)):
        save_slice(data[i], save_path.format(base_index + i), fmt)


def save_slice(slice_array, save_path, fmt):
    if fmt == 'jpg':
        img = Image.fromarray((slice_array).astype('uint8'))
        img.save(save_path)
    elif fmt == 'png':
        with open(save_path, 'wb') as f:
            z = (slice_array).astype(np.uint16)
            writer = png.Writer(width=z.shape[1], height=z.shape[0], bitdepth=16, greyscale=True)
            zlist = z.tolist()
            writer.write(f, zlist)
    else:
        nib.save(nib.Nifti1Image(slice_array, np.eye(4)), save_path)


def save_artefacts_from_images(src, dst, artefact_type, fmt='png', **kwargs):
    utils.safe_makedirs(dst)
    # Create sampling mask
    if artefact_type == 'fractal':
        sampler = artefacts.FractalRandomSampler(**kwargs)
    else:
        sampler = artefacts.OneDimCartesianRandomSampler(**kwargs)
    # Open one file to get format
    found = False
    for dirpath, dirnames, filenames in os.walk(src):
        for file in filenames[:1]:
            slice_array = imageio.imread(os.path.join(dirpath, filenames[0]))
            size = slice_array.shape[:2]
            found = True
        if found: break
    # Generate and save the mask
    sampler.generate_mask(size)
    sio.savemat(os.path.join(dst, 'mask.mat'), {'data': sampler.mask})
    # # Save parameters used, for reference
    # with open(os.path.join(dst, 'parameters.txt'), 'w') as f:
    #     for k, v in sorted(kwargs.items()):
    #         f.write("{}: {}\n".format(k, v))
    #     f.write("r_actual: {}\n".format(sampler.r_actual))
    #     f.write("r_no_tile: {}\n".format(sampler.r_no_tile))
    # # Begin processing and saving loop
    # for dirpath, dirnames, filenames in os.walk(src):
    #     for d in [os.path.join(dst, n) for n in dirnames]:
    #         utils.safe_makedirs(d)
    #     for file in filenames:
    #         # print("Processing {}".format(file))
    #         # Load into array
    #         slice_array = imageio.imread(os.path.join(dirpath, file))
    #         slice_array = slice_array.astype(np.float32)
    #         # Process
    #         max_val = MAX_UINT8 if fmt in ['jpg', 'jpeg'] else MAX_UINT16
    #         slice_array = data_processing.normalise(slice_array, (0., 1.), (0, max_val))
    #         slice_array_artefact = sampler.do_transform(slice_array)
    #         slice_array_artefact = data_processing.normalise(
    #                 slice_array_artefact, (0, MAX_UINT16))
    #         imgs = {}
    #         # Separate real and imaginary components
    #         imgs['real'] = slice_array_artefact[..., 0]
    #         imgs['imag'] = slice_array_artefact[..., 1]
    #         # Save images
    #         subpath = string_minus(src, dirpath)
    #         if subpath[0] == '/':
    #             subpath = subpath[1:]
    #         save_path = os.path.join(dst, subpath, file)
    #         for label, img in imgs.items():
    #             save_slice(img, save_path.replace('.', '_' + label + '.'), fmt)


def view_slices(data_path, slices, max_scans=24, skip=0, shape=None,
        wait=False, pause=1.0):
    print("Preparing OASIS3 data...")
    accepted_files = get_nifti_files(data_path, max_scans, skip, shape)
    print("Plotting slices")
    fig = plt.figure()
    fig.show()
    for i, data_file in enumerate(accepted_files):
        print("Scan {}: {}".format(i, data_file.split('/')[-1]))
        nimg = nib.load(data_file)
        # Get the 3D image data (series of grayscale slices)
        scan = nimg.get_data().astype(np.float32)
        scan_transposed = np.transpose(scan, (2, 0, 1))
        for s in slices:
            print("Slices {} to {}".format(s[0], s[1]))
            for t in range(s[0], s[1]):
                if t < 0:
                    t = len(scan_transposed) + t
                print("Slice {}".format(t))
                plt.imshow(scan_transposed[t], cmap='gray')
                fig.canvas.draw()
                if wait:
                    _ = raw_input("Press [enter] to continue.")
                else:
                    time.sleep(pause)
    plt.close()


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
    preprocessed_data = data_processing.normalise(preprocessed_data)
    return preprocessed_data


def get_oasis1_dataset(input_path, label_path, test_cases, max_training_cases, size):
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


def get_oasis1_dataset_test(input_path, label_path, test_cases, max_test_cases, size):
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


def prepare_oasis3_dataset(data_path, save_artefacts=False, category='train',
        slice_min=130, slice_max=170, n=256, fmt=None, **kwargs):
    print("Preparing OASIS3 data for {}...".format(category))
    accepted_files, skipped, new_resume = get_nifti_files(data_path, **kwargs)
    save_path = os.path.join(config.data_path, category)
    utils.safe_makedirs(save_path)

    for i, data_file in enumerate(accepted_files):
        name = data_file.split('/')[-1]
        print(name)
        nimg = nib.load(data_file)
        # Get the 3D image data (series of grayscale slices)
        scan = nimg.get_data().astype(np.float32)
        scan_transposed = np.transpose(scan, (2, 0, 1))
        scan_sliced = scan_transposed[slice_min:slice_max, :, :]
        scan_bounded = data_processing.imbound(scan_sliced, bounds=(n, n), center=True)
        slice_path = os.path.join(save_path, name.replace('.nii.gz', '_{:05d}'))
        save_slices(scan_bounded, slice_path, slice_min, save_artefacts, category, fmt)

    print('Finished writing')
    return accepted_files, skipped, new_resume

    
def write_split(data_path, split=[('train', 80), ('test', 20)], **kwargs):
    data_files = get_scan_paths(data_path, 'nii', 'T1w', (1, 2))
    last_subject = None
    resume = None
    for n, m in split:
        accepted_files, skipped, resume = prepare_oasis3_dataset(
            data_path, category=n, save_artefacts=False,
            max_scans=m, skip=0, data_files=data_files, resume=resume, 
            last_subject=last_subject, **kwargs)
        last_subject = get_subject_number(accepted_files[-1].split('/')[-1])


def main():
    # path = '/media/ben/ARIES/datasets/oasis3/data_full'  # raw
    src = '/home/ben/projects/honours/datasets/oasis3/exp3_png_2/ground_truth'

    dst = '/home/ben/projects/honours/datasets/oasis3/exp3_png_2/artefact_fcs_{}'
    for i, r in enumerate([0.18, 0.24, 0.36, 0.63]):
        save_artefacts_from_images(src, dst.format(i), artefact_type='fractal',
            k=1, K=0.2, r=r, two_quads=True, seed=0)

    dst = '/home/ben/projects/honours/datasets/oasis3/exp3_png_2/artefact_ocs_{}'
    for i, r in enumerate([5.0, 4.0, 3.0, 2.0]):
        save_artefacts_from_images(src, dst.format(i), artefact_type='cs', 
            r=r, r_alpha=3, axis=1, acs=3, seed=0)

    # Save data, split into categories
    # split = [('train', 160), ('train', 40), ('test', 20)]
    # write_split(path, split, 
    #     shape=(176, 256, 256), slice_min=100, slice_max=180, n=256, fmt='png')

    # Check how many valid files there are
    # get_nifti_files(path, 1000, 0, (176, 256, 256))

    # View slices in plot
    # view_slices(path, slices=[(120, 180)], max_scans=10, skip=0, 
    #         shape=(176, 256, 256), wait=True)

    # Compare scans with same subjects or runs
    # prepare_oasis3_dataset(path, category='s51', max_scans=1, skip=19, 
    #     slice_min=120, slice_max=180, fmt='jpg')
    # prepare_oasis3_dataset(path, category='s52', max_scans=1, skip=20, 
    #     slice_min=120, slice_max=180, fmt='jpg')
    # prepare_oasis3_dataset(path, category='s61', max_scans=1, skip=26, 
    #     slice_min=120, slice_max=180, fmt='jpg')


if __name__ == '__main__':
    main()
