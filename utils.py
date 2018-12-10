# -*- coding: utf-8 -*-
"""
utils.py

author: Ben Cottier (git: bencottier)
"""
from __future__ import absolute_import, division, print_function
import os, errno


def safe_makedirs(path):
    """
    Use os.makedirs with error catching
    """
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
