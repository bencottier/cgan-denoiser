# -*- coding: utf-8 -*-
"""
config.py

Parameters for different models

author: Ben Cottier (git: bencottier)
"""
from os.path import join

class ConfigCGAN:
    """
    Configuration parameters for the Conditional GAN
    """
    # Dimensions
    raw_size = 28
    adjust_size = 28
    train_size = 28
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
    max_epoch = 20
    L1_lambda = 100

    # Data
    buffer_size = 60000
    batch_size = 256

    # Data storage
    save_per_epoch = 5
    exp_name = 'noise_gan'
    data_path = join('out', exp_name, 'data')
    model_path = join('out', exp_name, 'model')
    results_path = join('out', exp_name, 'results')
