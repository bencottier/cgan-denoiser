"""
data.py

Methods for retrieving data at run time.

author: Ben Cottier (git: bencottier)
"""
from config import Config as config
from utils import imread
import os


def load(path):
    for i in os.listdir(path):
        pair = imread(os.path.join(path, i))
        raw_label, raw_input = pair[:, :config.raw_size], pair[:, config.raw_size:]
        yield (raw_label, raw_input, i)


def load_data():
    data = dict()
    data["train"] = lambda: load(config.data_path + "/train")
    data["test"] = lambda: load(config.data_path + "/test")
    return data
