from hci.lib.c_utils import load_library
import numpy as np
import ctypes
import math

double_excitation_entry = np.dtype([('rank', np.uint64), ('ijkl', np.double), ('iljk', np.double)])
mixed_excitation_entry = np.dtype([('rank', np.uint64), ('ijkl', np.double)])
hci_entry = np.dtype([('arank', np.uint64), ('brank', np.uint64), ('coeff', np.double)])

libhci = load_library("libhci")

# Compute ranking and unranking tables for configurations and double excitations
def get_ranking_tables(norb, nelec):
    nelec_a, nelec_b = nelec
    config_table_a = np.empty((nelec_a, norb-nelec_a+1), dtype=np.uint64)
    libhci.get_rank_table(config_table_a.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_size_t(norb),
                          ctypes.c_size_t(nelec_a))
    config_table_a_complement = np.empty((norb-nelec_a, nelec_a+1), dtype=np.uint64)
    libhci.get_rank_table(config_table_a_complement.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_size_t(norb),
                          ctypes.c_size_t(norb-nelec_a))
    config_table_b = np.empty((nelec_b, norb-nelec_b+1), dtype=np.uint64)
    libhci.get_rank_table(config_table_b.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_size_t(norb),
                          ctypes.c_size_t(nelec_b))
    config_table_b_complement = np.empty((norb-nelec_b, nelec_b+1), dtype=np.uint64)
    libhci.get_rank_table(config_table_b_complement.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_size_t(norb),
                          ctypes.c_size_t(norb-nelec_b))
    exc_table_4o = np.empty((4, norb-4+1), dtype=np.uint64)
    libhci.get_rank_table(exc_table_4o.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_size_t(norb),
                          ctypes.c_size_t(4))
    exc_table_2o = np.empty((2, norb-2+1), dtype=np.uint64)
    libhci.get_rank_table(exc_table_2o.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_size_t(norb),
                          ctypes.c_size_t(2))
    return config_table_a, config_table_a_complement, config_table_b, config_table_b_complement, exc_table_4o, exc_table_2o

def rank(occ_list, rank_table, norb, nocc):
    return libhci.rank(occ_list.ctypes.data_as(ctypes.c_void_p),
                       rank_table.ctypes.data_as(ctypes.c_void_p),
                       ctypes.c_size_t(norb),
                       ctypes.c_size_t(nocc))

def unrank(rank, occ_list, rank_table, norb, nocc):
    libhci.unrank(ctypes.c_uint64(rank),
                  occ_list.ctypes.data_as(ctypes.c_void_p),
                  rank_table.ctypes.data_as(ctypes.c_void_p),
                  ctypes.c_size_t(norb),
                  ctypes.c_size_t(nocc))

def rank_mixed(occ_list, rank_table, norb):
    return libhci.rank_mixed(occ_list.ctypes.data_as(ctypes.c_void_p),
                             rank_table.ctypes.data_as(ctypes.c_void_p),
                             ctypes.c_size_t(norb))

def unrank_mixed(rank, occ_list, rank_table, norb):
    libhci.unrank_mixed(ctypes.c_uint64(rank),
                        occ_list.ctypes.data_as(ctypes.c_void_p),
                        rank_table.ctypes.data_as(ctypes.c_void_p),
                        ctypes.c_size_t(norb))

# Create double excitation storage for aa and bb type excitations
def get_stored_double_exc(eri_s8, exc_table_4o, norb):
    doubles = np.empty(math.comb(norb, 4), dtype=double_excitation_entry)
    libhci.load_doubles_from_eri(doubles.ctypes.data_as(ctypes.c_void_p),
                                 eri_s8.ctypes.data_as(ctypes.c_void_p),
                                 exc_table_4o.ctypes.data_as(ctypes.c_void_p),
                                 ctypes.c_size_t(norb))
    return doubles

# Create double excitation storage for ab type excitations
def get_stored_mixed_exc(eri_s4, exc_table_2o, norb):
    mixed = np.empty(math.comb(norb, 2)**2, dtype=mixed_excitation_entry)
    libhci.load_mixed_from_eri(mixed.ctypes.data_as(ctypes.c_void_p),
                               eri_s4.ctypes.data_as(ctypes.c_void_p),
                               exc_table_2o.ctypes.data_as(ctypes.c_void_p),
                               ctypes.c_size_t(norb))
    return mixed

# Get maximum magnitude of double excitation among all with same four orbitals involved
def get_max_magnitudes(doubles, norb):
    magnitudes = np.empty(len(doubles), dtype=np.float64)
    libhci.get_max_magnitudes(doubles.ctypes.data_as(ctypes.c_void_p),
                              magnitudes.ctypes.data_as(ctypes.c_void_p),
                              ctypes.c_size_t(math.comb(norb, 4)))
    return magnitudes

# Expand search space using HCI selection algorithm for double excitations
def enlarge_space_doubles(hcivec, norb, nelec_a, nelec_b, thresh, 
                         config_table_a, config_table_b, exc_table_4o, exc_table_2o, 
                         doubles_aa, doubles_bb, mixed_ab,
                         max_mag_aa, max_mag_bb, max_mag_ab):
    nexc_aa = math.comb(nelec_a, 2)*math.comb(norb-nelec_a, 2)
    nexc_bb = math.comb(nelec_b, 2)*math.comb(norb-nelec_b, 2)
    nexc_ab = nelec_a*(norb-nelec_a)*nelec_b*(norb-nelec_b)
    add_list = np.empty((len(hcivec)*(nexc_aa+nexc_bb+nexc_ab), 2), dtype=np.uint64)
    nadd = libhci.enlarge_space_doubles(hcivec.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(len(hcivec)), add_list.ctypes.data_as(ctypes.c_void_p),
                                        ctypes.c_size_t(norb), ctypes.c_size_t(nelec_a), ctypes.c_size_t(nelec_b), ctypes.c_double(thresh),
                                        config_table_a.ctypes.data_as(ctypes.c_void_p), config_table_b.ctypes.data_as(ctypes.c_void_p),
                                        exc_table_4o.ctypes.data_as(ctypes.c_void_p), exc_table_2o.ctypes.data_as(ctypes.c_void_p),
                                        doubles_aa.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(len(doubles_aa)),
                                        doubles_bb.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(len(doubles_bb)),
                                        mixed_ab.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(len(mixed_ab)),
                                        max_mag_aa.ctypes.data_as(ctypes.c_void_p), max_mag_bb.ctypes.data_as(ctypes.c_void_p), 
                                        max_mag_ab.ctypes.data_as(ctypes.c_void_p))
    return add_list, nadd
    
# Expand search space using HCI selection algorithm for single excitations
def enlarge_space_singles(hcivec, norb, nelec_a, nelec_b, thresh,
                          config_table_a, config_table_a_complement,
                          config_table_b, config_table_b_complement,
                          h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4):
    nexc_a = nelec_a*(norb-nelec_a)
    nexc_b = nelec_b*(norb-nelec_b)
    combmax_a = math.comb(norb, nelec_a)
    combmax_b = math.comb(norb, nelec_b)
    add_list = np.zeros((len(hcivec)*(nexc_a+nexc_b), 2), dtype=np.uint64)
    nadd = libhci.enlarge_space_singles(hcivec.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(len(hcivec)), add_list.ctypes.data_as(ctypes.c_void_p),
                                        ctypes.c_size_t(norb), ctypes.c_size_t(nelec_a), ctypes.c_size_t(nelec_b),
                                        ctypes.c_uint64(combmax_a), ctypes.c_uint64(combmax_b), ctypes.c_double(thresh),
                                        config_table_a.ctypes.data_as(ctypes.c_void_p), config_table_a_complement.ctypes.data_as(ctypes.c_void_p),
                                        config_table_b.ctypes.data_as(ctypes.c_void_p), config_table_b_complement.ctypes.data_as(ctypes.c_void_p),
                                        h1e_aa.ctypes.data_as(ctypes.c_void_p), h1e_bb.ctypes.data_as(ctypes.c_void_p), 
                                        eri_aaaa_s8.ctypes.data_as(ctypes.c_void_p), eri_bbbb_s8.ctypes.data_as(ctypes.c_void_p),
                                        eri_aabb_s4.ctypes.data_as(ctypes.c_void_p))
    return add_list, nadd
    
def get_matrix_element_by_rank(ranka_1, rankb_1, ranka_2, rankb_2,
                               config_table_a, config_table_b, exc_table_4o, exc_table_2o,
                               norb, nelec_a, nelec_b,
                               ordered_doubles_aa, ordered_doubles_bb, ordered_mixed_ab,
                               h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4):
    libhci.get_matrix_element_by_rank.restype = ctypes.c_double
    return libhci.get_matrix_element_by_rank(ctypes.c_uint64(ranka_1), ctypes.c_uint64(rankb_1), ctypes.c_uint64(ranka_2), ctypes.c_uint64(rankb_2),
                                             config_table_a.ctypes.data_as(ctypes.c_void_p), config_table_b.ctypes.data_as(ctypes.c_void_p),
                                             exc_table_4o.ctypes.data_as(ctypes.c_void_p), exc_table_2o.ctypes.data_as(ctypes.c_void_p),
                                             ctypes.c_size_t(norb), ctypes.c_size_t(nelec_a), ctypes.c_size_t(nelec_b),
                                             ordered_doubles_aa.ctypes.data_as(ctypes.c_void_p), ordered_doubles_bb.ctypes.data_as(ctypes.c_void_p),
                                             ordered_mixed_ab.ctypes.data_as(ctypes.c_void_p), h1e_aa.ctypes.data_as(ctypes.c_void_p), 
                                             h1e_bb.ctypes.data_as(ctypes.c_void_p), eri_aaaa_s8.ctypes.data_as(ctypes.c_void_p),
                                             eri_bbbb_s8.ctypes.data_as(ctypes.c_void_p), eri_aabb_s4.ctypes.data_as(ctypes.c_void_p))