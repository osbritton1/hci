from hci import lib as hcilib
from pyscf import ao2mo, __config__
from pyscf import lib as pyscflib
from pyscf.lib import logger
from pyscf.fci import direct_spin1
from functools import reduce
import numpy as np
import math

def kernel(myhci, hcore, eri_ao, mo, norb, nelec, add_thresh, ci0=None, tol=None, lindep=None, max_cycle=None, max_space=None,
           nroots=None, davidson_only=None, max_memory=None, verbose=None, ecore=0, **kwargs):
    # Flags and logging
    log = logger.new_logger(myhci, verbose)
    if tol is None: tol = myhci.conv_tol
    if lindep is None: lindep = myhci.lindep
    if max_cycle is None: max_cycle = myhci.max_cycle
    if max_space is None: max_space = myhci.max_space
    if max_memory is None: max_memory = myhci.max_memory
    if nroots is None: nroots = myhci.nroots
    if myhci.verbose >= logger.WARN:
        myhci.check_sanity()
        
    # AO to MO transformations
    mo_a, mo_b = mo
    nelec_a, nelec_b = nelec
    h1e_aa = reduce(np.dot, (mo_a.conj().T, hcore, mo_a))
    h1e_bb = reduce(np.dot, (mo_b.conj().T, hcore, mo_b))
    eri_aaaa_s8 = ao2mo.restore('s8', ao2mo.full(eri_ao, mo_a), norb)
    eri_bbbb_s8 = ao2mo.restore('s8', ao2mo.full(eri_ao, mo_b), norb)
    eri_aabb_s4 = ao2mo.restore('s4', ao2mo.general(eri_ao, [mo_a, mo_a, mo_b, mo_b]), norb)

    # Compute ranking and unranking tables for configurations and double excitations
    config_table_a, config_table_a_complement, config_table_b, config_table_b_complement, exc_table_4o, exc_table_2o = hcilib.get_ranking_tables(norb, nelec)

    # Build and sort double excitation storage lists
    doubles_aa = hcilib.get_stored_double_exc(eri_aaaa_s8, exc_table_4o, norb)
    max_mag_aa = hcilib.get_max_magnitudes(doubles_aa, norb)
    sorted_indices = np.argsort(max_mag_aa)[::-1]
    doubles_aa = doubles_aa[sorted_indices]
    max_mag_aa = max_mag_aa[sorted_indices]
    
    doubles_bb = hcilib.get_stored_double_exc(eri_bbbb_s8, exc_table_4o, norb)
    max_mag_bb = hcilib.get_max_magnitudes(doubles_bb, norb)
    sorted_indices = np.argsort(max_mag_bb)[::-1]
    doubles_bb = doubles_bb[sorted_indices]
    max_mag_bb = max_mag_bb[sorted_indices]
    
    mixed_ab = hcilib.get_stored_mixed_exc(eri_aabb_s4, exc_table_2o, norb)
    max_mag_ab = np.abs(mixed_ab['ijkl'])
    sorted_indices = np.argsort(max_mag_ab)[::-1]
    mixed_ab = mixed_ab[sorted_indices]
    max_mag_ab = max_mag_ab[sorted_indices]

    #Initialize HCI vector if initial guess not provided
    if ci0 is None:
        ranks = np.array([[0, 0]], dtype=np.uint64)
        coeffs = [np.array([1.0], dtype=np.double)]
        ranks, coeffs = myhci.enlarge_space(ranks, coeffs, norb, nelec_a, nelec_b, add_thresh,
                                            config_table_a, config_table_a_complement, config_table_b, config_table_b_complement, exc_table_4o, exc_table_2o,
                                            doubles_aa, doubles_bb, mixed_ab, max_mag_aa, max_mag_bb, max_mag_ab,
                                            h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4)
    else:
        ranks, coeffs = ci0

    # Define Hamiltonian-vector contraction and preconditioner for use in Davidson algorithm
    def hop(c):
        hc = hcilib.contract_hamiltonian_hcivec_slow(ranks, c, hdiag,
                                                     config_table_a, config_table_b, norb, nelec_a, nelec_b,
                                                     h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4)
        return hc
    precond = lambda x, e, *args: x/(hdiag-e+myhci.level_shift)

    # Set convergence parameters
    max_len = math.comb(norb, nelec_a)*math.comb(norb, nelec_b)
    e_last = 0
    float_tol = myhci.start_tol
    tol_decay_rate = myhci.tol_decay_rate

    for icycle in range(norb):
        float_tol = max(float_tol*tol_decay_rate, tol*1e2)
        log.debug('cycle %d  ci.shape %s  float_tol %g',
                  icycle, (len(ranks),), float_tol)
        hdiag = myhci.make_hdiag_slow(ranks, 
                                      config_table_a, config_table_b, norb, nelec_a, nelec_b, 
                                      h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4)
        e, coeffs = pyscflib.davidson(hop, coeffs, precond, tol=float_tol, lindep=lindep,
                                      max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                                      max_memory=max_memory, verbose=log, **kwargs)
        if not isinstance(coeffs, list):
            coeffs = [coeffs]
        if nroots > 1:
            de, e_last = min(e)-e_last, min(e)
            log.info('cycle %d  E = %s  dE = %.8g', icycle, e+ecore, de)
        else:
            de, e_last = e-e_last, e
            print(ranks)
            print(coeffs)
            print(f'cycle {icycle}  E = {e+ecore:.15g}  dE = {de:.8g}')
            log.info('cycle %d  E = %.15g  dE = %.8g', icycle, e+ecore, de)

        if len(ranks) == max_len or abs(de) < tol*1e3:
            break

        old_length = float(len(ranks))
        ranks, coeffs = myhci.enlarge_space(ranks, coeffs, norb, nelec_a, nelec_b, add_thresh,
                                            config_table_a, config_table_a_complement, config_table_b, config_table_b_complement, exc_table_4o, exc_table_2o,
                                            doubles_aa, doubles_bb, mixed_ab, max_mag_aa, max_mag_bb, max_mag_ab,
                                            h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4)
        new_length = float(len(ranks))
        if new_length/old_length < 1.01:
            break

    log.debug('Extra CI in selected space %s', (len(ranks),))
    hdiag = myhci.make_hdiag_slow(ranks, 
                                  config_table_a, config_table_b, norb, nelec_a, nelec_b, 
                                  h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4)
    e, coeffs = pyscflib.davidson(hop, coeffs, precond, tol=float_tol, lindep=lindep,
                                  max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                                  max_memory=max_memory, verbose=log, **kwargs)
    if nroots > 1:
        for i, ei in enumerate(e+ecore):
            log.info('Selected CI state %d  E = %.15g', i, ei)
        return e+ecore, coeffs
    else:
        log.info('Selected CI  E = %.15g', e+ecore)
        return e+ecore, coeffs

def enlarge_space(myhci, ranks, coeffs, norb, nelec_a, nelec_b, add_thresh,
                  config_table_a, config_table_a_complement, config_table_b, config_table_b_complement, exc_table_4o, exc_table_2o,
                  doubles_aa, doubles_bb, mixed_ab, max_mag_aa, max_mag_bb, max_mag_ab,
                  h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4):
    add_list_doubles, nadd_doubles = hcilib.enlarge_space_doubles(ranks, coeffs[0], norb, nelec_a, nelec_b, add_thresh, 
                                                                  config_table_a, config_table_b, exc_table_4o, exc_table_2o, 
                                                                  doubles_aa, doubles_bb, mixed_ab,
                                                                  max_mag_aa, max_mag_bb, max_mag_ab)
    add_list_singles, nadd_singles = hcilib.enlarge_space_singles(ranks, coeffs[0], norb, nelec_a, nelec_b, add_thresh,
                                                                  config_table_a, config_table_a_complement,
                                                                  config_table_b, config_table_b_complement,
                                                                  h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4)
    add_list = np.unique(np.concatenate([add_list_doubles[:nadd_doubles], add_list_singles[:nadd_singles]]), axis=0)
    
    for exc_coeff in coeffs[1:]:
        add_list_doubles_exc, nadd_doubles_exc = hcilib.enlarge_space_doubles(ranks, exc_coeff, norb, nelec_a, nelec_b, add_thresh, 
                                                                              config_table_a, config_table_b, exc_table_4o, exc_table_2o, 
                                                                              doubles_aa, doubles_bb, mixed_ab,
                                                                              max_mag_aa, max_mag_bb, max_mag_ab)
        add_list_singles_exc, nadd_singles_exc = hcilib.enlarge_space_singles(ranks, exc_coeff, norb, nelec_a, nelec_b, add_thresh,
                                                                              config_table_a, config_table_a_complement,
                                                                              config_table_b, config_table_b_complement,
                                                                              h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4)
        add_list = np.unique(np.concatenate([add_list, add_list_doubles_exc[:nadd_doubles_exc], add_list_singles_exc[:nadd_singles_exc]]), axis=0)
    
    ndets_old = len(ranks)
    total_list = np.concatenate([ranks, add_list])
    ranks_new, unique_rows = np.unique(total_list, return_index=True, axis=0)
    ndets_new = len(ranks_new)
    coeffs_new = []
    for i, coeff in enumerate(coeffs):
        total_coeff = np.concatenate([coeff, np.zeros(ndets_new)])
        coeff_new = total_coeff[unique_rows]
        coeffs_new.append(coeff_new)
        
    return ranks_new, coeffs_new

class HeatBathCI(direct_spin1.FCISolver):
    ci_coeff_cutoff = getattr(__config__, 'fci_selected_ci_SCI_ci_coeff_cutoff', .5e-3)
    select_cutoff = getattr(__config__, 'fci_selected_ci_SCI_select_cutoff', .5e-3)
    conv_tol = getattr(__config__, 'fci_selected_ci_SCI_conv_tol', 1e-9)
    start_tol = getattr(__config__, 'fci_selected_ci_SCI_start_tol', 3e-4)
    tol_decay_rate = getattr(__config__, 'fci_selected_ci_SCI_tol_decay_rate', 0.3)

    _keys = {
        'ci_coeff_cutoff', 'select_cutoff', 'conv_tol', 'start_tol',
        'tol_decay_rate',
    }

    def __init__(self, mol=None):
        direct_spin1.FCISolver.__init__(self, mol)

    def dump_flags(self, verbose=None):
        direct_spin1.FCISolver.dump_flags(self, verbose)
        logger.info(self, 'ci_coeff_cutoff %g', self.ci_coeff_cutoff)
        logger.info(self, 'select_cutoff   %g', self.select_cutoff)
        logger.warn(self, '''
This is an inefficient dialect of heat-bath CI written as a senior thesis project. 
For efficient heat-bath CI programs, it is recommended to use the Dice program (https://github.com/sanshar/Dice.git).''')
        
    make_hdiag_slow = staticmethod(hcilib.make_hdiag_slow)
    enlarge_space = enlarge_space
    kernel = kernel

HCI = HeatBathCI