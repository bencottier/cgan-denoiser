"""
data.py

Methods for retrieving data at run time.

author: Ben Cottier (git: bencottier)
"""
from config import Config as config
import artefacts
from utils import imread
import nibabel as nib
import numpy as np
import os


def load_pair(path):
    for i in os.listdir(path):
        pair = imread(os.path.join(path, i))
        raw_label, raw_input = pair[:, :config.raw_size], pair[:, config.raw_size:]
        yield (raw_label, raw_input, i)


def load(path):
    for i in os.listdir(path):
        raw_label = nib.load(os.path.join(path, i)).get_data()
        raw_input = artefacts.add_turbulence(raw_label)
        raw_label = raw_label[np.newaxis, ..., np.newaxis]
        raw_input = raw_input[np.newaxis, ..., np.newaxis]
        yield (raw_label, raw_input, i)


def load_data():
    data = {}
    data["train"] = lambda: load(os.path.join(config.data_path, "train"))
    data["valid"] = lambda: load(os.path.join(config.data_path, "valid"))
    data["test"] = lambda: load(os.path.join(config.data_path, "test"))
    return data
