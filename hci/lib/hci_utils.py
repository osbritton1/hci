import numpy as np
import numpy.typing as npt
import ctypes as ct
import math
import os
from typing import Any

def load_library(libname: str):
    """Wrapper function around :py:func:`numpy.ctypeslib.load_library`.

    Args:
        libname (path-like): Name of the library being loaded. May have 'lib' as a prefix, 
            but do not include an extension as that is platform-dependent.

    Returns:
        ctypes.CDLL: A ctypes library object.
    
    Raises:
        OSError: If the library was not found or is defective.
    """    
    try:
        loader_path = os.path.dirname(__file__)
        return np.ctypeslib.load_library(libname, loader_path)
    except OSError:
        print(f'Failed to load library with name {libname}.')
        raise

rank_entry = np.dtype([('arank', np.uint64), ('brank', np.uint64)])
r"""
A :py:class:`numpy.dtype` representing a configuration via the 
    ranks of its :math:`\alpha` and :math:`\beta` occupancy lists.
"""

double_exc_entry = np.dtype([('rank', np.uint64), ('ijkl', np.double), ('iljk', np.double)])
r"""
A :py:class:`numpy.dtype` containing the minimal information
necessary for computing any double excitation matrix element
involving the same four changing orbitals. Directly mirrors
:cpp:struct:`DoubleExcEntry`.
"""

mixed_exc_entry = np.dtype([('rank', np.uint64), ('ijkl', np.double)])
r"""
A :py:class:`numpy.dtype` representing a mixed excitation.
Directly mirrors :cpp:struct:`MixedExcEntry`.
"""

libhci = load_library("libhci")
libhci.get_matrix_element_by_rank.restype = ct.c_double
libhci.get_matrix_element_by_rank_test_storage.restype = ct.c_double

class ConfigInfo:
    def __init__(self, norb: int, nelec: tuple[int, int]):
        nelec_a, nelec_b = nelec
        self.occ_table_a = create_ranking_table(norb, nelec_a)
        self.virt_table_a = create_ranking_table(norb, norb-nelec_a)
        self.combmax_a = math.comb(norb, nelec_a)
        self.occ_table_b = create_ranking_table(norb, nelec_b)
        self.virt_table_b = create_ranking_table(norb, norb-nelec_b)
        self.combmax_b = math.comb(norb, nelec_b)
        self.exc_table_4o = create_ranking_table(norb, 4)
        self.exc_table_2o = create_ranking_table(norb, 2)
        self.norb = norb
        self.nelec_a = nelec_a
        self.nelec_b = nelec_b
        self.ncols_mixed = math.comb(norb, 2)

    class _ConfigInfoPtrs(ct.Structure):
        _fields_ = [('norb', ct.c_size_t),
                    ('nelec_a', ct.c_size_t),
                    ('nelec_b', ct.c_size_t),
                    ('occ_table_a', ct.c_void_p),
                    ('virt_table_a', ct.c_void_p),
                    ('combmax_a', ct.c_uint64),
                    ('occ_table_b', ct.c_void_p),
                    ('virt_table_b', ct.c_void_p),
                    ('combmax_b', ct.c_uint64),
                    ('exc_table_4o', ct.c_void_p),
                    ('exc_table_2o', ct.c_void_p),
                    ('ncols_mixed', ct.c_uint64)]
    
    @property
    def _as_parameter_(self):
        ptrs = self._ConfigInfoPtrs(ct.c_size_t(self.norb),
                                    ct.c_size_t(self.nelec_a),
                                    ct.c_size_t(self.nelec_b),
                                    self.occ_table_a.ctypes.data_as(ct.c_void_p),
                                    self.virt_table_a.ctypes.data_as(ct.c_void_p),
                                    ct.c_uint64(self.combmax_a),
                                    self.occ_table_b.ctypes.data_as(ct.c_void_p),
                                    self.virt_table_b.ctypes.data_as(ct.c_void_p),
                                    ct.c_uint64(self.combmax_b),
                                    self.exc_table_4o.ctypes.data_as(ct.c_void_p),
                                    self.exc_table_2o.ctypes.data_as(ct.c_void_p),
                                    ct.c_uint64(self.ncols_mixed))
        return ct.byref(ptrs)

class HCore:
    r"""Class storing the core Hamiltonian in both the :math:`\alpha` and :math:`\beta`
    MO bases for ease of manipulation. When passed to a :py:mod:`ctypes` function, instances
    of :py:class:`HCore` are automatically converted to a struct compatible with :cpp:struct:`HCore`.
    """    
    def __init__(self, h1e_mo_aa: npt.NDArray[np.float64], h1e_mo_bb: npt.NDArray[np.float64]):
        r"""Constructor for a :py:class:`HCore` object.

        Args:
            h1e_mo_aa: The core Hamiltonian in the :math:`\alpha` MO basis. Should be of shape :math:`(N_\text{orb}, N_\text{orb})`.
            h1e_mo_bb: The core Hamiltonian in the :math:`\beta` MO basis. Should be of shape :math:`(N_\text{orb}, N_\text{orb})`.
        """        
        self.h1e_mo_aa = h1e_mo_aa
        self.h1e_mo_bb = h1e_mo_bb
        
    class _HCorePtrs(ct.Structure):
        """Internal class to manage :py:mod:`ctypes` interoperability.""" 

        _fields_ = [('h1e_mo_aa', ct.c_void_p),
                    ('h1e_mo_bb', ct.c_void_p)]
    
    @property
    def _as_parameter_(self):
        ptrs = self._HCorePtrs(self.h1e_mo_aa.ctypes.data_as(ct.c_void_p),
                               self.h1e_mo_bb.ctypes.data_as(ct.c_void_p))
        return ct.byref(ptrs)

class ERITensor:
    def __init__(self, eri_mo_aaaa_s8: npt.NDArray[np.float64],
                 eri_mo_bbbb_s8: npt.NDArray[np.float64], 
                 eri_mo_aabb_s4: npt.NDArray[np.float64], 
                 config_info: ConfigInfo):
        self.eri_mo_aaaa_s8 = eri_mo_aaaa_s8
        self.eri_mo_bbbb_s8 = eri_mo_bbbb_s8
        self.eri_mo_aabb_s4 = eri_mo_aabb_s4
        self.ncols_aabb = math.comb(config_info.norb+1, 2)
        
    class _ERITensorPtrs(ct.Structure):
        _fields_ = [('eri_mo_aaaa_s8', ct.c_void_p),
                    ('eri_mo_bbbb_s8', ct.c_void_p),
                    ('eri_mo_aabb_s4', ct.c_void_p),
                    ('ncols_aabb', ct.c_size_t)]
        
    @property
    def _as_parameter_(self):
        ptrs = self._ERITensorPtrs(self.eri_mo_aaaa_s8.ctypes.data_as(ct.c_void_p),
                                   self.eri_mo_bbbb_s8.ctypes.data_as(ct.c_void_p),
                                   self.eri_mo_aabb_s4.ctypes.data_as(ct.c_void_p),
                                   ct.c_size_t(self.ncols_aabb))
        return ct.byref(ptrs)

def create_ranking_table(norb: int, nocc: int):
    rank_table = np.empty((nocc, norb-nocc+1), dtype=np.uint64)
    libhci.load_rank_table(rank_table.ctypes, ct.c_size_t(norb), ct.c_size_t(nocc))
    return rank_table

class ExcEntries:
    def __init__(self, eri_mo: npt.NDArray[np.float64], config_info: ConfigInfo):
        self.ndoubles_aa = math.comb(config_info.norb, 4)
        self.doubles_aa = np.empty(self.ndoubles_aa, dtype=double_exc_entry)
        self.max_mag_aa = np.empty(self.ndoubles_aa, dtype=np.float64)

        self.ndoubles_bb = math.comb(config_info.norb, 4)
        self.doubles_bb = np.empty(self.ndoubles_bb, dtype=double_exc_entry)
        self.max_mag_bb = np.empty(self.ndoubles_bb, dtype=np.float64)

        self.nmixed_ab = math.comb(config_info.norb, 2)**2
        self.mixed_ab = np.empty(self.nmixed_ab, dtype=mixed_exc_entry)
        self.max_mag_ab = np.empty(self.nmixed_ab, dtype=np.float64)
        libhci.load_exc_entries_from_eri(self, eri_mo, config_info)

    class _ExcEntriesPtrs(ct.Structure):
        _fields_ = [('doubles_aa', ct.c_void_p),
                    ('max_mag_aa', ct.c_void_p),
                    ('ndoubles_aa', ct.c_size_t),
                    ('doubles_bb', ct.c_void_p),
                    ('max_mag_bb', ct.c_void_p),
                    ('ndoubles_bb', ct.c_size_t),
                    ('mixed_ab', ct.c_void_p),
                    ('max_mag_ab', ct.c_void_p),
                    ('nmixed_ab', ct.c_size_t)]
        
    @property
    def _as_parameter_(self):
        ptrs = self._ExcEntriesPtrs(self.doubles_aa.ctypes.data_as(ct.c_void_p),
                                    self.max_mag_aa.ctypes.data_as(ct.c_void_p),
                                    ct.c_size_t(self.ndoubles_aa),
                                    self.doubles_bb.ctypes.data_as(ct.c_void_p),
                                    self.max_mag_bb.ctypes.data_as(ct.c_void_p),
                                    ct.c_size_t(self.ndoubles_bb),
                                    self.mixed_ab.ctypes.data_as(ct.c_void_p),
                                    self.max_mag_ab.ctypes.data_as(ct.c_void_p),
                                    ct.c_size_t(self.nmixed_ab))
        return ct.byref(ptrs)

    def sort_desc_by_mag(self):
        sorted_indices = np.argsort(self.max_mag_aa)[::-1]
        self.doubles_aa = self.doubles_aa[sorted_indices]
        self.max_mag_aa = self.max_mag_aa[sorted_indices]
        sorted_indices = np.argsort(self.max_mag_bb)[::-1]
        self.doubles_bb = self.doubles_bb[sorted_indices]
        self.max_mag_bb = self.max_mag_bb[sorted_indices]
        sorted_indices = np.argsort(self.max_mag_ab)[::-1]
        self.mixed_ab = self.mixed_ab[sorted_indices]
        self.max_mag_ab = self.max_mag_ab[sorted_indices]

class HCIVec(npt.NDArray[Any]):
    '''An 2D np array for HCI coefficients'''
    def __array_finalize__(self, obj: npt.NDArray[Any] | None):
        self.ranks = getattr(obj, 'ranks', None)

    # Special cases for ndarray when the array was modified (through ufunc)
    # def __array_wrap__(self, out, context=None, return_scalar=False):
    #     if out.shape == self.shape:
    #         return out
    #     elif out.shape == ():  # if ufunc returns a scalar
    #         return out[()]
    #     else:
    #         return out.view(np.ndarray)

    @classmethod
    def as_HCIVec(cls: type['HCIVec'], hcivec: 'HCIVec', ranks: npt.NDArray[np.void]) -> 'HCIVec':
        hcivec = hcivec.view(HCIVec)
        hcivec.ranks = ranks
        return hcivec
    
    @classmethod
    def as_HCIVec_if_not(cls: type['HCIVec'], hcivec: 'HCIVec', ranks: npt.NDArray[np.void]) -> 'HCIVec':
        if getattr(hcivec, 'ranks', None) is None:
            hcivec = HCIVec.as_HCIVec(hcivec, ranks)
        return hcivec

    class _HCIVecPtrs(ct.Structure):
        _fields_ = [('ranks', ct.c_void_p),
                    ('coeffs', ct.c_void_p),
                    ('len', ct.c_size_t)]

    @property
    def _as_parameter_(self):
        if self.ranks is not None:
            ptrs = self._HCIVecPtrs(self.ranks.ctypes.data_as(ct.c_void_p),
                                    self.ctypes.data_as(ct.c_void_p),
                                    ct.c_size_t(len(self)))
            return ct.byref(ptrs)
        else:
            return None

class Rank(ct.Structure):
    _fields_ = [('arank', ct.c_uint64),
                ('brank', ct.c_uint64)]
    
    @classmethod
    def from_rank_entry(cls, rank_entry: np.void):
        return cls(rank_entry['arank'], rank_entry['brank'])

def enlarge_space_doubles(hcivec: HCIVec, add_thresh: float, config_info: ConfigInfo, exc_entries: ExcEntries):
    norb = config_info.norb
    nelec_a = config_info.nelec_a
    nelec_b = config_info.nelec_b
    nexc_aa = math.comb(nelec_a, 2)*math.comb(norb-nelec_a, 2)
    nexc_bb = math.comb(nelec_b, 2)*math.comb(norb-nelec_b, 2)
    nexc_ab = nelec_a*(norb-nelec_a)*nelec_b*(norb-nelec_b)
    add_list = np.empty(len(hcivec.ranks)*(nexc_aa+nexc_bb+nexc_ab), dtype=rank_entry)
    nadd = libhci.enlarge_space_doubles(hcivec, add_list.ctypes, ct.c_double(add_thresh), config_info, exc_entries)
    return add_list, nadd

def enlarge_space_singles(hcivec: HCIVec, add_thresh: float, config_info: ConfigInfo, h1e: HCore, eri_mo: ERITensor):
    norb = config_info.norb
    nelec_a = config_info.nelec_a
    nelec_b = config_info.nelec_b
    nexc_a = nelec_a*(norb-nelec_a)
    nexc_b = nelec_b*(norb-nelec_b)
    add_list = np.empty(len(hcivec.ranks)*(nexc_a+nexc_b), dtype=rank_entry)
    nadd = libhci.enlarge_space_singles(hcivec, add_list.ctypes, ct.c_double(add_thresh), config_info, h1e, eri_mo)
    return add_list, nadd

def get_matrix_element_by_rank(rank1: Rank, rank2: Rank, config_info: ConfigInfo, h1e: HCore, eri_mo: ERITensor):
    return libhci.get_matrix_element_by_rank(rank1, rank2, config_info, h1e, eri_mo)
    
def get_matrix_element_by_rank_test_storage(rank1: Rank, rank2: Rank, config_info: ConfigInfo, 
                                            exc_entries: ExcEntries, h1e: HCore, eri_mo: ERITensor):
    return libhci.get_matrix_element_by_rank_test_storage(rank1, rank2, config_info, exc_entries, h1e, eri_mo)

def make_hdiag_slow(hcivec: HCIVec, config_info: ConfigInfo, h1e: HCore, eri_mo: ERITensor):
    hdiag = np.empty(len(hcivec.ranks), dtype=np.double)
    libhci.make_hdiag_slow(hcivec, hdiag.ctypes, config_info, h1e, eri_mo)
    return hdiag

def contract_hamiltonian_hcivec_slow(hcivec_old: HCIVec, hdiag: npt.NDArray[np.float64], config_info: ConfigInfo, h1e: HCore, eri_mo: ERITensor):
    coeffs_new = np.empty(len(hcivec_old), dtype=np.float64)
    libhci.contract_hamiltonian_hcivec_slow(hcivec_old, coeffs_new.ctypes, hdiag.ctypes, config_info, h1e, eri_mo)
    return coeffs_new

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