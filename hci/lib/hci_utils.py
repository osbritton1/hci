from hci.lib.c_utils import load_library
import numpy as np
import ctypes as ct
import math

rank_entry = np.dtype([('arank', np.uint64), ('brank', np.uint64)])
double_excitation_entry = np.dtype([('rank', np.uint64), ('ijkl', np.double), ('iljk', np.double)])
mixed_excitation_entry = np.dtype([('rank', np.uint64), ('ijkl', np.double)])

libhci = load_library("libhci")
libhci.get_matrix_element_by_rank.restype = ct.c_double
libhci.get_matrix_element_by_rank_test_storage.restype = ct.c_double
libhci.get_matrix_element_by_rank_new.restype = ct.c_double
libhci.get_matrix_element_by_rank_test_storage_new.restype = ct.c_double

class CoreHamiltonian:
    def __init__(self, h1e_mo_aa, h1e_mo_bb):
        self.h1e_mo_aa = h1e_mo_aa
        self.h1e_mo_bb = h1e_mo_bb
        self.ptrs = self.CoreHamiltonianPtrs(self.h1e_mo_aa.ctypes.data_as(ct.c_void_p),
                                             self.h1e_mo_bb.ctypes.data_as(ct.c_void_p))
        self._as_parameter_ = ct.byref(self.ptrs)
        
    class CoreHamiltonianPtrs(ct.Structure):
        _fields_ = [('h1e_mo_aa', ct.c_void_p),
                    ('h1e_mo_bb', ct.c_void_p)]

class ElectronRepulsionIntegrals:
    def __init__(self, eri_mo_aaaa_s8, eri_mo_bbbb_s8, eri_mo_aabb_s4):
        self.eri_mo_aaaa_s8 = eri_mo_aaaa_s8
        self.eri_mo_bbbb_s8 = eri_mo_bbbb_s8
        self.eri_mo_aabb_s4 = eri_mo_aabb_s4
        self.ptrs = self.ElectronRepulsionIntegralPtrs(self.eri_mo_aaaa_s8.ctypes.data_as(ct.c_void_p),
                                                       self.eri_mo_bbbb_s8.ctypes.data_as(ct.c_void_p),
                                                       self.eri_mo_aabb_s4.ctypes.data_as(ct.c_void_p))
        self._as_parameter_ = ct.byref(self.ptrs)
        
    class ElectronRepulsionIntegralPtrs(ct.Structure):
        _fields_ = [('eri_mo_aaaa_s8', ct.c_void_p),
                    ('eri_mo_bbbb_s8', ct.c_void_p),
                    ('eri_mo_aabb_s4', ct.c_void_p)]

def create_ranking_table(norb, nocc):
    rank_table = np.empty((nocc, norb-nocc+1), dtype=np.uint64)
    libhci.get_rank_table(rank_table.ctypes, ct.c_size_t(norb), ct.c_size_t(nocc))
    return rank_table

class ConfigInfo:
    def __init__(self, norb, nelec):
        nelec_a, nelec_b = nelec
        self.config_table_a = create_ranking_table(norb, nelec_a)
        self.config_table_a_complement = create_ranking_table(norb, norb-nelec_a)
        self.combmax_a = math.comb(norb, nelec_a)
        self.config_table_b = create_ranking_table(norb, nelec_b)
        self.config_table_b_complement = create_ranking_table(norb, norb-nelec_b)
        self.combmax_b = math.comb(norb, nelec_b)
        self.exc_table_4o = create_ranking_table(norb, 4)
        self.exc_table_2o = create_ranking_table(norb, 2)
        self.norb = norb
        self.nelec_a = nelec_a
        self.nelec_b = nelec_b
        self.mixed_ncols = math.comb(norb+1, 2)
        self.ptrs = self.ConfigInfoPtrs(self.config_table_a.ctypes.data_as(ct.c_void_p),
                                        self.config_table_a_complement.ctypes.data_as(ct.c_void_p),
                                        ct.c_uint64(self.combmax_a),
                                        self.config_table_b.ctypes.data_as(ct.c_void_p),
                                        self.config_table_b_complement.ctypes.data_as(ct.c_void_p),
                                        ct.c_uint64(self.combmax_b),
                                        self.exc_table_4o.ctypes.data_as(ct.c_void_p),
                                        self.exc_table_2o.ctypes.data_as(ct.c_void_p),
                                        ct.c_size_t(self.norb),
                                        ct.c_size_t(self.nelec_a),
                                        ct.c_size_t(self.nelec_b),
                                        ct.c_uint64(self.mixed_ncols))
        self._as_parameter_ = ct.byref(self.ptrs)

    class ConfigInfoPtrs(ct.Structure):
        _fields_ = [('config_table_a', ct.c_void_p),
                    ('config_table_a_complement', ct.c_void_p),
                    ('combmax_a', ct.c_uint64),
                    ('config_table_b', ct.c_void_p),
                    ('config_table_b_complement', ct.c_void_p),
                    ('combmax_b', ct.c_uint64),
                    ('exc_table_4o', ct.c_void_p),
                    ('exc_table_2o', ct.c_void_p),
                    ('norb', ct.c_size_t),
                    ('nelec_a', ct.c_size_t),
                    ('nelec_b', ct.c_size_t),
                    ('mixed_ncols', ct.c_uint64)]

# Create double excitation storage for aa and bb type excitations
def get_stored_double_exc(eri_mo_xxxx_s8, exc_table_4o, norb):
    doubles = np.empty(math.comb(norb, 4), dtype=double_excitation_entry)
    libhci.load_doubles_from_eri(doubles.ctypes, eri_mo_xxxx_s8.ctypes, 
                                 exc_table_4o.ctypes, ct.c_size_t(norb))
    return doubles

# Create double excitation storage for aa and bb type excitations
def get_stored_double_exc_new(eri_mo_xxxx_s8, config_info):
    doubles = np.empty(math.comb(config_info.norb, 4), dtype=double_excitation_entry)
    libhci.load_doubles_from_eri(doubles.ctypes, eri_mo_xxxx_s8.ctypes, 
                                 config_info.exc_table_4o.ctypes, ct.c_size_t(config_info.norb))
    return doubles

# Create double excitation storage for ab type excitations
def get_stored_mixed_exc(eri_mo_aabb_s4, exc_table_2o, norb):
    mixed = np.empty(math.comb(norb, 2)**2, dtype=mixed_excitation_entry)
    libhci.load_mixed_from_eri(mixed.ctypes, eri_mo_aabb_s4.ctypes,
                               exc_table_2o.ctypes, ct.c_size_t(norb))
    return mixed

# Create double excitation storage for ab type excitations
def get_stored_mixed_exc_new(eri_mo_aabb_s4, config_info):
    mixed = np.empty(math.comb(config_info.norb, 2)**2, dtype=mixed_excitation_entry)
    libhci.load_mixed_from_eri(mixed.ctypes, eri_mo_aabb_s4.ctypes,
                               config_info.exc_table_2o.ctypes, ct.c_size_t(config_info.norb))
    return mixed

# Get maximum magnitude of double excitation among all with same four orbitals involved
def get_max_magnitudes(doubles):
    magnitudes = np.empty(len(doubles), dtype=np.float64)
    libhci.get_max_magnitudes(doubles.ctypes, magnitudes.ctypes, ct.c_size_t(len(doubles)))
    return magnitudes

class ExcitationEntries:
    def __init__(self, eri_mo, config_info):
        self.doubles_aa = get_stored_double_exc_new(eri_mo.eri_mo_aaaa_s8, config_info)
        self.max_mag_aa = get_max_magnitudes(self.doubles_aa)
        self.ndoubles_aa = len(self.doubles_aa)
        self.doubles_bb = get_stored_double_exc_new(eri_mo.eri_mo_bbbb_s8, config_info)
        self.max_mag_bb = get_max_magnitudes(self.doubles_bb)
        self.ndoubles_bb = len(self.doubles_bb)
        self.mixed_ab = get_stored_mixed_exc_new(eri_mo.eri_mo_aabb_s4, config_info)
        self.max_mag_ab = np.abs(self.mixed_ab['ijkl'])
        self.nmixed_ab = len(self.mixed_ab)
        self.ptrs = self.ExcitationEntriesPtrs(self.doubles_aa.ctypes.data_as(ct.c_void_p),
                                               self.max_mag_aa.ctypes.data_as(ct.c_void_p),
                                               ct.c_size_t(self.ndoubles_aa),
                                               self.doubles_bb.ctypes.data_as(ct.c_void_p),
                                               self.max_mag_bb.ctypes.data_as(ct.c_void_p),
                                               ct.c_size_t(self.ndoubles_bb),
                                               self.mixed_ab.ctypes.data_as(ct.c_void_p),
                                               self.max_mag_ab.ctypes.data_as(ct.c_void_p),
                                               ct.c_size_t(self.nmixed_ab))
        self._as_parameter_ = ct.byref(self.ptrs)

    class ExcitationEntriesPtrs(ct.Structure):
        _fields_ = [('doubles_aa', ct.c_void_p),
                    ('max_mag_aa', ct.c_void_p),
                    ('ndoubles_aa', ct.c_size_t),
                    ('doubles_bb', ct.c_void_p),
                    ('max_mag_bb', ct.c_void_p),
                    ('ndoubles_bb', ct.c_size_t),
                    ('mixed_ab', ct.c_void_p),
                    ('max_mag_ab', ct.c_void_p),
                    ('nmixed_ab', ct.c_size_t)]

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
        self.ptrs = self.ExcitationEntriesPtrs(self.doubles_aa.ctypes.data_as(ct.c_void_p),
                                               self.max_mag_aa.ctypes.data_as(ct.c_void_p),
                                               ct.c_size_t(self.ndoubles_aa),
                                               self.doubles_bb.ctypes.data_as(ct.c_void_p),
                                               self.max_mag_bb.ctypes.data_as(ct.c_void_p),
                                               ct.c_size_t(self.ndoubles_bb),
                                               self.mixed_ab.ctypes.data_as(ct.c_void_p),
                                               self.max_mag_ab.ctypes.data_as(ct.c_void_p),
                                               ct.c_size_t(self.nmixed_ab))
        self._as_parameter_ = ct.byref(self.ptrs)

class HCIVector(np.ndarray):
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

    def as_HCIVector(hcivec, ranks):
        hcivec = hcivec.view(HCIVector)
        hcivec.ranks = ranks
        return hcivec
    
    def as_SCIVector_if_not(hcivec, ranks):
        if getattr(hcivec, '_strs', None) is None:
            hcivec = as_HCIVector(hcivec, ranks)
        return hcivec

    class HCIVectorPtrs(ct.Structure):
        _fields_ = [('ranks', ct.c_void_p),
                    ('coeffs', ct.c_void_p),
                    ('len', ct.c_size_t)]

    @property
    def _as_parameter_(self):
        ptrs = self.HCIVectorPtrs(self.ranks.ctypes.data_as(ct.c_void_p),
                                 self.ctypes.data_as(ct.c_void_p),
                                 ct.c_size_t(len(self)))
        return ct.byref(ptrs)

class Rank(ct.Structure):
    _fields_ = [('arank', ct.c_uint64),
                ('brank', ct.c_uint64)]
    
    @classmethod
    def from_rank_entry(cls, rank_entry):
        return cls(rank_entry['arank'], rank_entry['brank'])


def enlarge_space_doubles_new(hcivec, add_thresh, config_info, excitation_entries):
    norb = config_info.norb
    nelec_a = config_info.nelec_a
    nelec_b = config_info.nelec_b
    nexc_aa = math.comb(nelec_a, 2)*math.comb(norb-nelec_a, 2)
    nexc_bb = math.comb(nelec_b, 2)*math.comb(norb-nelec_b, 2)
    nexc_ab = nelec_a*(norb-nelec_a)*nelec_b*(norb-nelec_b)
    add_list = np.empty(len(hcivec.ranks)*(nexc_aa+nexc_bb+nexc_ab), dtype=rank_entry)
    nadd = libhci.enlarge_space_doubles_new(hcivec, add_list.ctypes, ct.c_double(add_thresh), config_info, excitation_entries)
    return add_list, nadd

def enlarge_space_singles_new(hcivec, add_thresh, config_info, h1e, eri_mo):
    norb = config_info.norb
    nelec_a = config_info.nelec_a
    nelec_b = config_info.nelec_b
    nexc_a = nelec_a*(norb-nelec_a)
    nexc_b = nelec_b*(norb-nelec_b)
    add_list = np.empty(len(hcivec.ranks)*(nexc_a+nexc_b), dtype=rank_entry)
    nadd = libhci.enlarge_space_singles_new(hcivec, add_list.ctypes, ct.c_double(add_thresh), config_info, h1e, eri_mo)
    return add_list, nadd

def get_matrix_element_by_rank_new(rank1, rank2, config_info, h1e, eri_mo):
    return libhci.get_matrix_element_by_rank_new(rank1, rank2, config_info, h1e, eri_mo)
    
def get_matrix_element_by_rank_test_storage_new(rank1, rank2, config_info, excitation_entries, h1e, eri_mo):
    return libhci.get_matrix_element_by_rank_test_storage_new(rank1, rank2, config_info, excitation_entries, h1e, eri_mo)

def make_hdiag_slow_new(hcivec, config_info, h1e, eri_mo):
    hdiag = np.empty(len(hcivec.ranks), dtype=np.double)
    libhci.make_hdiag_slow_new(hcivec, hdiag.ctypes, config_info, h1e, eri_mo)
    return hdiag

def contract_hamiltonian_hcivec_slow_new(hcivec_old, hdiag, config_info, h1e, eri_mo):
    coeffs_new = np.empty(len(hcivec_old), dtype=np.float64)
    libhci.contract_hamiltonian_hcivec_slow_new(hcivec_old, coeffs_new.ctypes, hdiag.ctypes, config_info, h1e, eri_mo)
    return coeffs_new

# OLD SPAGHETTI
# OLD SPAGHETTI
# OLD SPAGHETTI
        
# Compute ranking and unranking tables for configurations and double excitations
def get_ranking_tables(norb, nelec):
    nelec_a, nelec_b = nelec
    config_table_a = np.empty((nelec_a, norb-nelec_a+1), dtype=np.uint64)
    libhci.get_rank_table(config_table_a.ctypes.data_as(ct.c_void_p),
                          ct.c_size_t(norb),
                          ct.c_size_t(nelec_a))
    config_table_a_complement = np.empty((norb-nelec_a, nelec_a+1), dtype=np.uint64)
    libhci.get_rank_table(config_table_a_complement.ctypes.data_as(ct.c_void_p),
                          ct.c_size_t(norb),
                          ct.c_size_t(norb-nelec_a))
    config_table_b = np.empty((nelec_b, norb-nelec_b+1), dtype=np.uint64)
    libhci.get_rank_table(config_table_b.ctypes.data_as(ct.c_void_p),
                          ct.c_size_t(norb),
                          ct.c_size_t(nelec_b))
    config_table_b_complement = np.empty((norb-nelec_b, nelec_b+1), dtype=np.uint64)
    libhci.get_rank_table(config_table_b_complement.ctypes.data_as(ct.c_void_p),
                          ct.c_size_t(norb),
                          ct.c_size_t(norb-nelec_b))
    exc_table_4o = np.empty((4, norb-4+1), dtype=np.uint64)
    libhci.get_rank_table(exc_table_4o.ctypes.data_as(ct.c_void_p),
                          ct.c_size_t(norb),
                          ct.c_size_t(4))
    exc_table_2o = np.empty((2, norb-2+1), dtype=np.uint64)
    libhci.get_rank_table(exc_table_2o.ctypes.data_as(ct.c_void_p),
                          ct.c_size_t(norb),
                          ct.c_size_t(2))
    return config_table_a, config_table_a_complement, config_table_b, config_table_b_complement, exc_table_4o, exc_table_2o

def rank(occ_list, rank_table, norb, nocc):
    return libhci.rank(occ_list.ctypes.data_as(ct.c_void_p),
                       rank_table.ctypes.data_as(ct.c_void_p),
                       ct.c_size_t(norb),
                       ct.c_size_t(nocc))

def unrank(rank, occ_list, rank_table, norb, nocc):
    libhci.unrank(ct.c_uint64(rank),
                  occ_list.ctypes.data_as(ct.c_void_p),
                  rank_table.ctypes.data_as(ct.c_void_p),
                  ct.c_size_t(norb),
                  ct.c_size_t(nocc))

def rank_mixed(occ_list, rank_table, norb):
    return libhci.rank_mixed(occ_list.ctypes.data_as(ct.c_void_p),
                             rank_table.ctypes.data_as(ct.c_void_p),
                             ct.c_size_t(norb))

def unrank_mixed(rank, occ_list, rank_table, norb):
    libhci.unrank_mixed(ct.c_uint64(rank),
                        occ_list.ctypes.data_as(ct.c_void_p),
                        rank_table.ctypes.data_as(ct.c_void_p),
                        ct.c_size_t(norb))

# Expand search space using HCI selection algorithm for double excitations
def enlarge_space_doubles(ranks, coeffs, norb, nelec_a, nelec_b, thresh, 
                         config_table_a, config_table_b, exc_table_4o, exc_table_2o, 
                         doubles_aa, doubles_bb, mixed_ab,
                         max_mag_aa, max_mag_bb, max_mag_ab):
    nexc_aa = math.comb(nelec_a, 2)*math.comb(norb-nelec_a, 2)
    nexc_bb = math.comb(nelec_b, 2)*math.comb(norb-nelec_b, 2)
    nexc_ab = nelec_a*(norb-nelec_a)*nelec_b*(norb-nelec_b)
    add_list = np.empty((len(ranks)*(nexc_aa+nexc_bb+nexc_ab), 2), dtype=np.uint64)
    nadd = libhci.enlarge_space_doubles(ranks.ctypes.data_as(ct.c_void_p), coeffs.ctypes.data_as(ct.c_void_p), 
                                        ct.c_size_t(len(ranks)), add_list.ctypes.data_as(ct.c_void_p),
                                        ct.c_size_t(norb), ct.c_size_t(nelec_a), ct.c_size_t(nelec_b), ct.c_double(thresh),
                                        config_table_a.ctypes.data_as(ct.c_void_p), config_table_b.ctypes.data_as(ct.c_void_p),
                                        exc_table_4o.ctypes.data_as(ct.c_void_p), exc_table_2o.ctypes.data_as(ct.c_void_p),
                                        doubles_aa.ctypes.data_as(ct.c_void_p), ct.c_size_t(len(doubles_aa)),
                                        doubles_bb.ctypes.data_as(ct.c_void_p), ct.c_size_t(len(doubles_bb)),
                                        mixed_ab.ctypes.data_as(ct.c_void_p), ct.c_size_t(len(mixed_ab)),
                                        max_mag_aa.ctypes.data_as(ct.c_void_p), max_mag_bb.ctypes.data_as(ct.c_void_p), 
                                        max_mag_ab.ctypes.data_as(ct.c_void_p))
    return add_list, nadd
    
# Expand search space using HCI selection algorithm for single excitations
def enlarge_space_singles(ranks, coeffs, norb, nelec_a, nelec_b, thresh,
                          config_table_a, config_table_a_complement,
                          config_table_b, config_table_b_complement,
                          h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4):
    nexc_a = nelec_a*(norb-nelec_a)
    nexc_b = nelec_b*(norb-nelec_b)
    combmax_a = math.comb(norb, nelec_a)
    combmax_b = math.comb(norb, nelec_b)
    add_list = np.zeros((len(ranks)*(nexc_a+nexc_b), 2), dtype=np.uint64)
    nadd = libhci.enlarge_space_singles(ranks.ctypes.data_as(ct.c_void_p), coeffs.ctypes.data_as(ct.c_void_p), 
                                        ct.c_size_t(len(ranks)), add_list.ctypes.data_as(ct.c_void_p),
                                        ct.c_size_t(norb), ct.c_size_t(nelec_a), ct.c_size_t(nelec_b),
                                        ct.c_uint64(combmax_a), ct.c_uint64(combmax_b), ct.c_double(thresh),
                                        config_table_a.ctypes.data_as(ct.c_void_p), config_table_a_complement.ctypes.data_as(ct.c_void_p),
                                        config_table_b.ctypes.data_as(ct.c_void_p), config_table_b_complement.ctypes.data_as(ct.c_void_p),
                                        h1e_aa.ctypes.data_as(ct.c_void_p), h1e_bb.ctypes.data_as(ct.c_void_p), 
                                        eri_aaaa_s8.ctypes.data_as(ct.c_void_p), eri_bbbb_s8.ctypes.data_as(ct.c_void_p),
                                        eri_aabb_s4.ctypes.data_as(ct.c_void_p))
    return add_list, nadd

def get_matrix_element_by_rank(ranka_1, rankb_1, ranka_2, rankb_2,
                               config_table_a, config_table_b, norb, nelec_a, nelec_b,
                               h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4):
    return libhci.get_matrix_element_by_rank(ct.c_uint64(ranka_1), ct.c_uint64(rankb_1), ct.c_uint64(ranka_2), ct.c_uint64(rankb_2),
                                             config_table_a.ctypes.data_as(ct.c_void_p), config_table_b.ctypes.data_as(ct.c_void_p),
                                             ct.c_size_t(norb), ct.c_size_t(nelec_a), ct.c_size_t(nelec_b),
                                             h1e_aa.ctypes.data_as(ct.c_void_p), h1e_bb.ctypes.data_as(ct.c_void_p),
                                             eri_aaaa_s8.ctypes.data_as(ct.c_void_p), eri_bbbb_s8.ctypes.data_as(ct.c_void_p), 
                                             eri_aabb_s4.ctypes.data_as(ct.c_void_p))
    
def get_matrix_element_by_rank_test_storage(ranka_1, rankb_1, ranka_2, rankb_2,
                                            config_table_a, config_table_b, exc_table_4o, exc_table_2o,
                                            norb, nelec_a, nelec_b,
                                            ordered_doubles_aa, ordered_doubles_bb, ordered_mixed_ab,
                                            h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4):
    return libhci.get_matrix_element_by_rank_test_storage(ct.c_uint64(ranka_1), ct.c_uint64(rankb_1), ct.c_uint64(ranka_2), ct.c_uint64(rankb_2),
                                                          config_table_a.ctypes.data_as(ct.c_void_p), config_table_b.ctypes.data_as(ct.c_void_p),
                                                          exc_table_4o.ctypes.data_as(ct.c_void_p), exc_table_2o.ctypes.data_as(ct.c_void_p),
                                                          ct.c_size_t(norb), ct.c_size_t(nelec_a), ct.c_size_t(nelec_b),
                                                          ordered_doubles_aa.ctypes.data_as(ct.c_void_p), ordered_doubles_bb.ctypes.data_as(ct.c_void_p),
                                                          ordered_mixed_ab.ctypes.data_as(ct.c_void_p), h1e_aa.ctypes.data_as(ct.c_void_p), 
                                                          h1e_bb.ctypes.data_as(ct.c_void_p), eri_aaaa_s8.ctypes.data_as(ct.c_void_p),
                                                          eri_bbbb_s8.ctypes.data_as(ct.c_void_p), eri_aabb_s4.ctypes.data_as(ct.c_void_p))

def make_hdiag_slow(ranks, 
                    config_table_a, config_table_b, norb, nelec_a, nelec_b, 
                    h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4):
    hdiag = np.empty(len(ranks), dtype=np.double)
    libhci.make_hdiag_slow(ranks.ctypes.data_as(ct.c_void_p), hdiag.ctypes.data_as(ct.c_void_p), ct.c_size_t(len(ranks)),
                           config_table_a.ctypes.data_as(ct.c_void_p), config_table_b.ctypes.data_as(ct.c_void_p),
                           ct.c_size_t(norb), ct.c_size_t(nelec_a), ct.c_size_t(nelec_b),
                           h1e_aa.ctypes.data_as(ct.c_void_p), h1e_bb.ctypes.data_as(ct.c_void_p),
                           eri_aaaa_s8.ctypes.data_as(ct.c_void_p), eri_bbbb_s8.ctypes.data_as(ct.c_void_p),
                           eri_aabb_s4.ctypes.data_as(ct.c_void_p))
    return hdiag

def contract_hamiltonian_hcivec_slow(ranks, coeffs, hdiag,
                                     config_table_a, config_table_b, norb, nelec_a, nelec_b,
                                     h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4):
    coeffs_new = np.empty_like(coeffs)
    libhci.contract_hamiltonian_hcivec_slow(ranks.ctypes.data_as(ct.c_void_p), coeffs.ctypes.data_as(ct.c_void_p), 
                                            coeffs_new.ctypes.data_as(ct.c_void_p), ct.c_size_t(len(ranks)), hdiag.ctypes.data_as(ct.c_void_p),
                                            config_table_a.ctypes.data_as(ct.c_void_p), config_table_b.ctypes.data_as(ct.c_void_p),
                                            ct.c_size_t(norb), ct.c_size_t(nelec_a), ct.c_size_t(nelec_b),
                                            h1e_aa.ctypes.data_as(ct.c_void_p), h1e_bb.ctypes.data_as(ct.c_void_p),
                                            eri_aaaa_s8.ctypes.data_as(ct.c_void_p), eri_bbbb_s8.ctypes.data_as(ct.c_void_p),
                                            eri_aabb_s4.ctypes.data_as(ct.c_void_p))
    return coeffs_new

def occslst2strs(occslst):
    occslst = np.asarray(occslst)
    na, nelec = occslst.shape
    strs = np.zeros(na, dtype=np.int64)
    for i in range(nelec):
        strs ^= 1 << occslst[:,i]
    return strs

def hci_to_sci(ranks, coeffs, config_table_a, config_table_b, norb, nelec_a, nelec_b):
    a_unique, a_indices = np.unique(ranks[:, 0], return_inverse=True)
    b_unique, b_indices = np.unique(ranks[:, 1], return_inverse=True)
    sci_coeffs = np.zeros((len(a_unique), len(b_unique)), dtype=np.double)
    for i, (a_index, b_index) in enumerate(zip(a_indices, b_indices)):
        sci_coeffs[a_index, b_index] = coeffs[i]
    aoccs = np.zeros((len(a_unique), nelec_a), dtype=np.int64)
    boccs = np.zeros((len(b_unique), nelec_b), dtype=np.int64)
    for i, arank in enumerate(a_unique):
        occ_list = np.zeros(nelec_a, dtype=np.uint64)
        unrank(arank, occ_list, config_table_a, norb, nelec_a)
        aoccs[i] = occ_list.astype(np.int64)
    for i, brank in enumerate(b_unique):
        occ_list = np.zeros(nelec_b, dtype=np.uint64)
        unrank(brank, occ_list, config_table_b, norb, nelec_b)
        boccs[i] = occ_list.astype(np.int64)
    print(aoccs)
    print(boccs)
    return (occslst2strs(aoccs), occslst2strs(boccs)), sci_coeffs