import numpy as np
import ctypes as ct
import math
import os

def load_library(libname):
    try:
        loader_path = os.path.dirname(__file__)
        return np.ctypeslib.load_library(libname, loader_path)
    except OSError:
        print(f'Failed to load library with name {libname}.')
        raise

rank_entry = np.dtype([('arank', np.uint64), ('brank', np.uint64)])
double_exc_entry = np.dtype([('rank', np.uint64), ('ijkl', np.double), ('iljk', np.double)])
mixed_exc_entry = np.dtype([('rank', np.uint64), ('ijkl', np.double)])

libhci = load_library("libhci")
libhci.get_matrix_element_by_rank.restype = ct.c_double
libhci.get_matrix_element_by_rank_test_storage.restype = ct.c_double

class HCore:
    def __init__(self, h1e_mo_aa, h1e_mo_bb):
        self.h1e_mo_aa = h1e_mo_aa
        self.h1e_mo_bb = h1e_mo_bb
        
    class HCorePtrs(ct.Structure):
        _fields_ = [('h1e_mo_aa', ct.c_void_p),
                    ('h1e_mo_bb', ct.c_void_p)]
    
    @property
    def _as_parameter_(self):
        ptrs = self.HCorePtrs(self.h1e_mo_aa.ctypes.data_as(ct.c_void_p),
                              self.h1e_mo_bb.ctypes.data_as(ct.c_void_p))
        return ct.byref(ptrs)

class ERITensor:
    def __init__(self, eri_mo_aaaa_s8, eri_mo_bbbb_s8, eri_mo_aabb_s4, config_info):
        self.eri_mo_aaaa_s8 = eri_mo_aaaa_s8
        self.eri_mo_bbbb_s8 = eri_mo_bbbb_s8
        self.eri_mo_aabb_s4 = eri_mo_aabb_s4
        self.ncols_aabb = math.comb(config_info.norb+1, 2)
        
    class ERITensorPtrs(ct.Structure):
        _fields_ = [('eri_mo_aaaa_s8', ct.c_void_p),
                    ('eri_mo_bbbb_s8', ct.c_void_p),
                    ('eri_mo_aabb_s4', ct.c_void_p),
                    ('ncols_aabb', ct.c_size_t)]
        
    @property
    def _as_parameter_(self):
        ptrs = self.ERITensorPtrs(self.eri_mo_aaaa_s8.ctypes.data_as(ct.c_void_p),
                                  self.eri_mo_bbbb_s8.ctypes.data_as(ct.c_void_p),
                                  self.eri_mo_aabb_s4.ctypes.data_as(ct.c_void_p),
                                  ct.c_size_t(self.ncols_aabb))
        return ct.byref(ptrs)

def create_ranking_table(norb, nocc):
    rank_table = np.empty((nocc, norb-nocc+1), dtype=np.uint64)
    libhci.load_rank_table(rank_table.ctypes, ct.c_size_t(norb), ct.c_size_t(nocc))
    return rank_table

class ConfigInfo:
    def __init__(self, norb, nelec):
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

    class ConfigInfoPtrs(ct.Structure):
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
        ptrs = self.ConfigInfoPtrs(ct.c_size_t(self.norb),
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

def load_exc_entries_from_eri(exc_entries, eri_mo, config_info):
    libhci.load_exc_entries_from_eri(exc_entries, eri_mo, config_info)

class ExcEntries:
    def __init__(self, eri_mo, config_info):
        self.ndoubles_aa = math.comb(config_info.norb, 4)
        self.doubles_aa = np.empty(self.ndoubles_aa, dtype=double_exc_entry)
        self.max_mag_aa = np.empty(self.ndoubles_aa, dtype=np.float64)

        self.ndoubles_bb = math.comb(config_info.norb, 4)
        self.doubles_bb = np.empty(self.ndoubles_bb, dtype=double_exc_entry)
        self.max_mag_bb = np.empty(self.ndoubles_bb, dtype=np.float64)

        self.nmixed_ab = math.comb(config_info.norb, 2)**2
        self.mixed_ab = np.empty(self.nmixed_ab, dtype=mixed_exc_entry)
        self.max_mag_ab = np.empty(self.nmixed_ab, dtype=np.float64)
        load_exc_entries_from_eri(self, eri_mo, config_info)

    class ExcEntriesPtrs(ct.Structure):
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
        ptrs = self.ExcEntriesPtrs(self.doubles_aa.ctypes.data_as(ct.c_void_p),
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

class HCIVec(np.ndarray):
    '''An 2D np array for HCI coefficients'''
    def __array_finalize__(self, obj):
        self.ranks = getattr(obj, 'ranks', None)

    # Special cases for ndarray when the array was modified (through ufunc)
    def __array_wrap__(self, out, context=None, return_scalar=False):
        if out.shape == self.shape:
            return out
        elif out.shape == ():  # if ufunc returns a scalar
            return out[()]
        else:
            return out.view(np.ndarray)

    def as_HCIVec(hcivec, ranks):
        hcivec = hcivec.view(HCIVec)
        hcivec.ranks = ranks
        return hcivec
    
    def as_HCIVec_if_not(hcivec, ranks):
        if getattr(hcivec, 'ranks', None) is None:
            hcivec = HCIVec.as_HCIVec(hcivec, ranks)
        return hcivec

    class HCIVecPtrs(ct.Structure):
        _fields_ = [('ranks', ct.c_void_p),
                    ('coeffs', ct.c_void_p),
                    ('len', ct.c_size_t)]

    @property
    def _as_parameter_(self):
        ptrs = self.HCIVecPtrs(self.ranks.ctypes.data_as(ct.c_void_p),
                               self.ctypes.data_as(ct.c_void_p),
                               ct.c_size_t(len(self)))
        return ct.byref(ptrs)

class Rank(ct.Structure):
    _fields_ = [('arank', ct.c_uint64),
                ('brank', ct.c_uint64)]
    
    @classmethod
    def from_rank_entry(cls, rank_entry):
        return cls(rank_entry['arank'], rank_entry['brank'])

def enlarge_space_doubles(hcivec, add_thresh, config_info, excitation_entries):
    norb = config_info.norb
    nelec_a = config_info.nelec_a
    nelec_b = config_info.nelec_b
    nexc_aa = math.comb(nelec_a, 2)*math.comb(norb-nelec_a, 2)
    nexc_bb = math.comb(nelec_b, 2)*math.comb(norb-nelec_b, 2)
    nexc_ab = nelec_a*(norb-nelec_a)*nelec_b*(norb-nelec_b)
    add_list = np.empty(len(hcivec.ranks)*(nexc_aa+nexc_bb+nexc_ab), dtype=rank_entry)
    nadd = libhci.enlarge_space_doubles(hcivec, add_list.ctypes, ct.c_double(add_thresh), config_info, excitation_entries)
    return add_list, nadd

def enlarge_space_singles(hcivec, add_thresh, config_info, h1e, eri_mo):
    norb = config_info.norb
    nelec_a = config_info.nelec_a
    nelec_b = config_info.nelec_b
    nexc_a = nelec_a*(norb-nelec_a)
    nexc_b = nelec_b*(norb-nelec_b)
    add_list = np.empty(len(hcivec.ranks)*(nexc_a+nexc_b), dtype=rank_entry)
    nadd = libhci.enlarge_space_singles(hcivec, add_list.ctypes, ct.c_double(add_thresh), config_info, h1e, eri_mo)
    return add_list, nadd

def get_matrix_element_by_rank(rank1, rank2, config_info, h1e, eri_mo):
    return libhci.get_matrix_element_by_rank(rank1, rank2, config_info, h1e, eri_mo)
    
def get_matrix_element_by_rank_test_storage(rank1, rank2, config_info, excitation_entries, h1e, eri_mo):
    return libhci.get_matrix_element_by_rank_test_storage(rank1, rank2, config_info, excitation_entries, h1e, eri_mo)

def make_hdiag_slow(hcivec, config_info, h1e, eri_mo):
    hdiag = np.empty(len(hcivec.ranks), dtype=np.double)
    libhci.make_hdiag_slow(hcivec, hdiag.ctypes, config_info, h1e, eri_mo)
    return hdiag

def contract_hamiltonian_hcivec_slow(hcivec_old, hdiag, config_info, h1e, eri_mo):
    coeffs_new = np.empty(len(hcivec_old), dtype=np.float64)
    libhci.contract_hamiltonian_hcivec_slow(hcivec_old, coeffs_new.ctypes, hdiag.ctypes, config_info, h1e, eri_mo)
    return coeffs_new

def occslst2strs(occslst):
    occslst = np.asarray(occslst)
    na, nelec = occslst.shape
    strs = np.zeros(na, dtype=np.int64)
    for i in range(nelec):
        strs ^= 1 << occslst[:,i]
    return strs

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