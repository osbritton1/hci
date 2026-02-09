import os
import numpy as np

def load_library(libname):
    try:
        loader_path = os.path.dirname(__file__)
        return np.ctypeslib.load_library(libname, loader_path)
    except OSError:
        print(f'Failed to load library with name {libname}.')
        raise