import os
import numpy as np
import ctypes

# class DoubleExcitationEntry(ctypes.Structure):
#     _fields_ = [('rank', ctypes.c_uint64), ('ijkl', ctypes.c_double), ('iljk', ctypes.c_double)]

double_excitation_entry = np.dtype([('rank', np.uint64), ('ijkl', np.double), ('iljk', np.double)])

def load_library(libname):
    try:
        loader_path = os.path.dirname(__file__)
        return np.ctypeslib.load_library(libname, loader_path)
    except OSError:
        print(f'Failed to load library with name {libname}.')
        raise