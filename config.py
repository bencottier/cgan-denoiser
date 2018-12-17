# -*- coding: utf-8 -*-
"""
config.py

Parameters for different models

author: Ben Cottier (git: bencottier)
"""
from os.path import join

class Config:
    """
    Configuration parameters for the Conditional GAN
    """
    # Dimensions
    raw_size = None
    adjust_size = 256
    train_size = 256
    channels = 1
    base_number_of_filters = 64
    kernel_size = (3, 3)
    strides = (2, 2)

    # Fixed model parameters
    leak = 0.2
    dropout_rate = 0.5

    # Hyperparameters
    learning_rate = 2e-4
    beta1 = 0.5
    L1_lambda = 100

    # Data
    buffer_size = 164
    batch_size = 1
    max_epoch = 20
    max_training_cases = 148
    validation_split = 0.2
    test_cases = [1, 3, 4, 6, 10, 12, 37, 49, 50, 100, 120, 140, 145, 150, 151, 152]

    # Data storage
    save_per_epoch = 5
    exp_name = 'fractal_oasis1_cgan'
    data_path = join('out', exp_name, 'data')
    root_path = "/home/ben/projects/honours/datasets/oasis1/"
    input_path = root_path + "slices_artefact/"
    label_path = root_path + "slices_pad/"
    model_path = join('out', exp_name, 'model')
    results_path = join('out', exp_name, 'results')
