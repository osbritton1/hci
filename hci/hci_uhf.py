from hci import lib
from pyscf import ao2mo
from functools import reduce
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured

def c_print_doubles(doubles):
    print('{')
    for entry in doubles:
        print(f"{{{entry['rank']}, {entry['ijkl']}, {entry['iljk']}}},")
    print('}')

def c_print_mixed(mixed):
    print('{')
    for entry in mixed:
        print(f"{{{entry['rank']}, {entry['ijkl']}}},")
    print('}')

def c_print_array(array):
    print('{')
    for entry in array:
        print(f"{entry},")
    print('}')

def kernel(hcore, eri_ao, mo, norb, nelec, add_thresh, ci0=None, tol=None, lindep=None, max_cycle=None, max_space=None,
           nroots=None, davidson_only=None, max_memory=None, verbose=None, **kwargs):
    # AO to MO transformations
    mo_a, mo_b = mo
    nelec_a, nelec_b = nelec
    h1e_aa = reduce(np.dot, (mo_a.conj().T, hcore, mo_a))
    h1e_bb = reduce(np.dot, (mo_b.conj().T, hcore, mo_b))
    eri_mo_aaaa = ao2mo.restore('s8', ao2mo.full(eri_ao, mo_a), norb)
    eri_mo_bbbb = ao2mo.restore('s8', ao2mo.full(eri_ao, mo_b), norb)
    eri_mo_aabb = ao2mo.restore('s4', ao2mo.general(eri_ao, [mo_a, mo_a, mo_b, mo_b]), norb)

    # Compute ranking and unranking tables for configurations and double excitations
    config_table_a, config_table_a_complement, config_table_b, config_table_b_complement, exc_table_4o, exc_table_2o = lib.get_ranking_tables(norb, nelec)

    # Build and sort double excitation storage lists
    doubles_aa = lib.get_stored_double_exc(eri_mo_aaaa, exc_table_4o, norb)
    max_mag_aa = lib.get_max_magnitudes(doubles_aa, norb)
    sorted_indices = np.argsort(max_mag_aa)[::-1]
    doubles_aa = doubles_aa[sorted_indices]
    max_mag_aa = max_mag_aa[sorted_indices]
    
    doubles_bb = lib.get_stored_double_exc(eri_mo_bbbb, exc_table_4o, norb)
    max_mag_bb = lib.get_max_magnitudes(doubles_bb, norb)
    sorted_indices = np.argsort(max_mag_bb)[::-1]
    doubles_bb = doubles_bb[sorted_indices]
    max_mag_bb = max_mag_bb[sorted_indices]
    
    mixed_ab = lib.get_stored_mixed_exc(eri_mo_aabb, exc_table_2o, norb)
    max_mag_ab = np.abs(mixed_ab['ijkl'])
    sorted_indices = np.argsort(max_mag_ab)[::-1]
    mixed_ab = mixed_ab[sorted_indices]
    max_mag_ab = max_mag_ab[sorted_indices]

    #Initialize HCI vector if initial guess not provided
    if ci0 is None:
        ci0 = np.empty(1, dtype=lib.hci_entry)
        ci0[0] = (0, 0, 1.0)

    ci0_new = enlarge_space(hcivec, norb, nelec_a, nelec_b, thresh,
                            config_table_a, config_table_a_complement, config_table_b, config_table_b_complement, exc_table_4o, exc_table_2o,
                            doubles_aa, doubles_bb, mixed_ab, max_mag_aa, max_mag_bb, max_mag_ab,
                            h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4)

    print(ci0)
    print(ci0_new)
        
    return ci0

def enlarge_space(hcivec, norb, nelec_a, nelec_b, thresh,
                  config_table_a, config_table_a_complement, config_table_b, config_table_b_complement, exc_table_4o, exc_table_2o,
                  doubles_aa, doubles_bb, mixed_ab, max_mag_aa, max_mag_bb, max_mag_ab,
                  h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4):
    
    add_list_doubles, nadd_doubles = lib.enlarge_space_doubles(hcivec, norb, nelec_a, nelec_b, thresh, 
                                                               config_table_a, config_table_b, exc_table_4o, exc_table_2o, 
                                                               doubles_aa, doubles_bb, mixed_ab,
                                                               max_mag_aa, max_mag_bb, max_mag_ab)
    print(add_list_doubles[:nadd_doubles])
    add_list_singles, nadd_singles = lib.enlarge_space_singles(hcivec, norb, nelec_a, nelec_b, thresh,
                                                               config_table_a, config_table_a_complement,
                                                               config_table_b, config_table_b_complement,
                                                               h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4)
    print(add_list_singles[:nadd_singles])
    print()
    
    ndets_old = len(hcivec)
    already_included = structured_to_unstructured(hcivec[['arank', 'brank']])
    total_list = np.concatenate([already_included, add_list_doubles[:nadd_doubles], add_list_singles[:nadd_singles]])
    new_list, unique_rows = np.unique(total_list, return_index=True, axis=0)
    ndets_new = len(new_list)
    hcivec_new = np.zeros(ndets_new, dtype=lib.hci_entry)
    hcivec_new[['arank', 'brank']] = unstructured_to_structured(new_list)
    
    filter_from_old = unique_rows<ndets_old
    old_indices_to_copy = unique_rows[filter_from_old]
    new_locations = np.nonzero(filter_from_old)[0]
    for (old_index, new_index) in zip(old_indices_to_copy, new_locations):
        hcivec_new[new_index]['coeff'] = hcivec[old_index]['coeff']
        
    return hcivec_new