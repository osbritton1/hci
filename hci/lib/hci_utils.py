import numpy as np
from pathlib import Path

def load_library(libname: str, loader_path: Path):
    """Wrapper function around :py:func:`numpy.ctypeslib.load_library`.

    Args:
        libname (str): Name of the library being loaded. May have 'lib' as a prefix, 
            but do not include an extension as that is platform-dependent.
        loader_path (Path): Path to the library

    Returns:
        ctypes.CDLL: A ctypes library object.
    
    Raises:
        OSError: If the library was not found or is defective.
    """    
    try:
        print(loader_path)
        return np.ctypeslib.load_library(libname, loader_path)
    except OSError:
        print(f'Failed to load library with name {libname}.')
        raise

libhci = load_library("libhci", Path(__file__).parent.parent.parent / 'build/hci/lib/')

# def occslst2strs(occslst):
#     occslst = np.asarray(occslst)
#     na, nelec = occslst.shape
#     strs = np.zeros(na, dtype=np.int64)
#     for i in range(nelec):
#         strs ^= 1 << occslst[:,i]
#     return strs

# def hci_to_sci(ranks, coeffs, config_table_a, config_table_b, norb, nelec_a, nelec_b):
#     a_unique, a_indices = np.unique(ranks[:, 0], return_inverse=True)
#     b_unique, b_indices = np.unique(ranks[:, 1], return_inverse=True)
#     sci_coeffs = np.zeros((len(a_unique), len(b_unique)), dtype=np.double)
#     for i, (a_index, b_index) in enumerate(zip(a_indices, b_indices)):
#         sci_coeffs[a_index, b_index] = coeffs[i]
#     aoccs = np.zeros((len(a_unique), nelec_a), dtype=np.int64)
#     boccs = np.zeros((len(b_unique), nelec_b), dtype=np.int64)
#     for i, arank in enumerate(a_unique):
#         occ_list = np.zeros(nelec_a, dtype=np.uint64)
#         unrank(arank, occ_list, config_table_a, norb, nelec_a)
#         aoccs[i] = occ_list.astype(np.int64)
#     for i, brank in enumerate(b_unique):
#         occ_list = np.zeros(nelec_b, dtype=np.uint64)
#         unrank(brank, occ_list, config_table_b, norb, nelec_b)
#         boccs[i] = occ_list.astype(np.int64)
#     print(aoccs)
#     print(boccs)
#     return (occslst2strs(aoccs), occslst2strs(boccs)), sci_coeffs