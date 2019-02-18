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
    raw_size = 256
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
    n_total = 1560
    n_train = 960
    n_valid = 240
    n_test = 360
    batch_size = 1
    max_epoch = 20

    # Data storage
    save_per_epoch = max_epoch
    exp_name = 'fractal_oasis3_cgan'
    data_path = '/home/Student/s4360417/honours/datasets/oasis3/exp2_jpg'
    # data_path = '/home/ben/projects/honours/datasets/oasis3/exp2_jpg'
    model_path = join('out', exp_name, 'model')
    results_path = join('out', exp_name, 'results')
