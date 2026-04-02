import sys
parent_module = sys.modules['.'.join(__name__.split('.')[:-1]) or '__main__']
if __name__ == '__main__' or parent_module.__name__ == '__main__':
    import lib as hcilib
    from lib import HCore, ERITensor, ConfigInfo, ExcEntries, HCIVec
else:
    import hci.lib as hcilib
    from hci.lib import HCore, ERITensor, ConfigInfo, ExcEntries, HCIVec

from pyscf import ao2mo, __config__
from pyscf import lib as pyscflib
from pyscf.lib import logger
from pyscf.fci import direct_spin1
from functools import reduce
import numpy as np

def kernel(myhci, hcore, eri_ao, mo, norb, nelec, add_thresh=None, ci0=None, tol=None, lindep=None, max_cycle=None, max_space=None,
           nroots=None, max_memory=None, verbose=None, ecore=0, **kwargs):
    
    # Flags and logging
    log = logger.new_logger(myhci, verbose)
    if add_thresh is None: add_thresh = myhci.add_thresh
    if tol is None: tol = myhci.conv_tol
    if lindep is None: lindep = myhci.lindep
    if max_cycle is None: max_cycle = myhci.max_cycle
    if max_space is None: max_space = myhci.max_space
    if max_memory is None: max_memory = myhci.max_memory
    if nroots is None: nroots = myhci.nroots
    if myhci.verbose >= logger.WARN:
        myhci.check_sanity()

    # Build ranking table
    config_info = ConfigInfo(norb, nelec)

    # AO to MO transformations
    mo_a, mo_b = mo
    h1e_mo_aa = reduce(np.dot, (mo_a.conj().T, hcore, mo_a))
    h1e_mo_bb = reduce(np.dot, (mo_b.conj().T, hcore, mo_b))
    h1e = HCore(h1e_mo_aa, h1e_mo_bb)
    eri_mo_aaaa_s8 = ao2mo.restore('s8', ao2mo.full(eri_ao, mo_a), norb)
    eri_mo_bbbb_s8 = ao2mo.restore('s8', ao2mo.full(eri_ao, mo_b), norb)
    eri_mo_aabb_s4 = ao2mo.restore('s4', ao2mo.general(eri_ao, [mo_a, mo_a, mo_b, mo_b]), norb)
    eri_mo = ERITensor(eri_mo_aaaa_s8, eri_mo_bbbb_s8, eri_mo_aabb_s4, config_info)

    # Build and sort excitation lists
    excitation_entries = ExcEntries(eri_mo, config_info)
    excitation_entries.sort_desc_by_mag()

    #Initialize HCI vector
    if isinstance(ci0, HCIVec):
        ranks = ci0.ranks
        if ci0.size == len(ranks):
            ci0 = [ci0.ravel()]
        else:
            ci0 = [c.ravel() for c in ci0]
    elif isinstance(ci0, list) and isinstance(ci0[0], HCIVec):
        ranks = ci0[0].ranks
        ci0 = [HCIVec.as_HCIVec_if_not(c.ravel(), ranks) for c in ci0]
    elif ci0 is None:
        ranks = np.zeros(1, dtype=hcilib.rank_entry)
        hf_gnd = HCIVec.as_HCIVec(np.array([1.0], dtype=np.float64), ranks)
        ci0 = [hf_gnd]
        ci0 = myhci.enlarge_space(ci0, add_thresh, config_info, excitation_entries, h1e, eri_mo)
        ranks = ci0[0].ranks
        if len(ranks) < nroots:
            raise RuntimeError(f'''
  HCI space generated from HF ground state via single and double excitations is too small
  for calculating {nroots} states.\n''')
    else:
        raise RuntimeError(f'Unsupported type {type(ci0)} for initial HCI seed ci0.')

    # Define Hamiltonian-vector contraction and preconditioner for use in Davidson algorithm
    def hop(c):
        hc = hcilib.contract_hamiltonian_hcivec_slow(HCIVec.as_HCIVec(c, ranks), hdiag, config_info, h1e, eri_mo) 
        return hc
    precond = lambda x, e, *args: x/(hdiag-e+myhci.level_shift)

    # Set convergence parameters
    max_len = config_info.combmax_a*config_info.combmax_b
    e_last = 0
    float_tol = myhci.start_tol
    tol_decay_rate = myhci.tol_decay_rate

    # Begin HCI loop
    for icycle in range(norb):
        float_tol = max(float_tol*tol_decay_rate, tol*1e2)
        log.info('HCI begin cycle %d  ndets %d/%d  float_tol %g',
                  icycle, len(ranks), max_len, float_tol)
        log.info('HCI begin davidson')
        hdiag = myhci.make_hdiag_slow(ci0[0], config_info, h1e, eri_mo)
        e, ci0 = pyscflib.davidson(hop, ci0, precond, tol=float_tol, lindep=lindep,
                                   max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                                   max_memory=max_memory, verbose=log, **kwargs)
        if nroots > 1:
            ci0 = [HCIVec.as_HCIVec(c, ranks) for c in ci0]
            de, e_last = min(e)-e_last, min(e)
            log.info('HCI end davidson  E = %s  dE_gnd = %.8g', e+ecore, de)
        else:
            ci0 = [HCIVec.as_HCIVec(ci0, ranks)]
            de, e_last = e-e_last, e
            log.info('HCI end davidson  E = %.15g  dE_gnd = %.8g', e+ecore, de)
        
        if len(ranks) == max_len:
            log.info('HCI converged to FCI ndets %d/%d', len(ranks), max_len)
            break
        if abs(de) < tol*1e3:
            log.info('HCI converged energy |dE_gnd| = %.8g < %.8g', abs(de), tol*1e3)
            break

        old_length = len(ranks)
        ci0 = myhci.enlarge_space(ci0, add_thresh, config_info, excitation_entries, h1e, eri_mo)
        ranks = ci0[0].ranks
        new_length = len(ranks)
        log.info('HCI enlarge nadd %d', new_length-old_length)
        if float(new_length)/float(old_length) < 1.01:
            log.info('HCI converged enlargement ratio %.3g < 1.01', new_length/old_length)
            break
        log.info('HCI end cycle %d', icycle)

    log.info('HCI final diagonalization ndets %d/%d', len(ranks), max_len)
    hdiag = myhci.make_hdiag_slow(ci0[0], config_info, h1e, eri_mo)
    e, c = pyscflib.davidson(hop, ci0, precond, tol=float_tol, lindep=lindep,
                             max_cycle=max_cycle, max_space=max_space, nroots=nroots,
                             max_memory=max_memory, verbose=log, **kwargs)
    if nroots > 1:
        log.info('HCI ndets = %d/%d', len(ranks), max_len)
        for i, ei in enumerate(e+ecore):
            log.info('HCI state %d  E = %.15g', i, ei)
        return e+ecore, [HCIVec.as_HCIVec(ci, ranks) for ci in c]
    else:
        log.info('HCI  E = %.15g  ndets = %d/%d', e+ecore, len(ranks), max_len)
        return e+ecore, HCIVec.as_HCIVec(c, ranks)

def enlarge_space(myhci, hcivecs, add_thresh, config_info, excitation_entries, h1e, eri_mo):
    add_list_doubles, nadd_doubles = hcilib.enlarge_space_doubles(hcivecs[0], add_thresh, config_info, excitation_entries)
    add_list_singles, nadd_singles = hcilib.enlarge_space_singles(hcivecs[0], add_thresh, config_info, h1e, eri_mo)
    add_list = np.unique(np.concatenate([add_list_doubles[:nadd_doubles], add_list_singles[:nadd_singles]]))
    for exc_vec in hcivecs[1:]:
        add_list_doubles_exc, nadd_doubles_exc = hcilib.enlarge_space_doubles(exc_vec, add_thresh, config_info, excitation_entries)
        add_list_singles_exc, nadd_singles_exc = hcilib.enlarge_space_singles(exc_vec, add_thresh, config_info, h1e, eri_mo)
        add_list = np.unique(np.concatenate([add_list, add_list_doubles_exc[:nadd_doubles_exc], add_list_singles_exc[:nadd_singles_exc]]))
    
    total_ranks = np.concatenate([hcivecs[0].ranks, add_list])
    ranks_new, unique_rows = np.unique(total_ranks, return_index=True)
    ndets_new = len(ranks_new)
    hcivecs_new = []
    for hcivec_old in hcivecs:
        hcivec_total = np.concatenate([hcivec_old, np.zeros(ndets_new)])
        hcivec_new = HCIVec.as_HCIVec(hcivec_total[unique_rows], ranks_new)
        hcivecs_new.append(hcivec_new)
        
    return hcivecs_new

class HeatBathCI(direct_spin1.FCISolver):
    add_thresh = getattr(__config__, 'fci_selected_ci_SCI_select_cutoff', .5e-3)
    conv_tol = getattr(__config__, 'fci_selected_ci_SCI_conv_tol', 1e-9)
    start_tol = getattr(__config__, 'fci_selected_ci_SCI_start_tol', 3e-4)
    tol_decay_rate = getattr(__config__, 'fci_selected_ci_SCI_tol_decay_rate', 0.3)

    _keys = {
        'add_thresh', 'conv_tol', 'start_tol',
        'tol_decay_rate',
    }

    def __init__(self, mol=None):
        direct_spin1.FCISolver.__init__(self, mol)

    def dump_flags(self, verbose=None):
        direct_spin1.FCISolver.dump_flags(self, verbose)
        logger.info(self, 'add_thresh %g', self.add_thresh)
        logger.warn(self, '''
This is an inefficient dialect of heat-bath CI written as a senior thesis project. 
For an efficient heat-bath CI program, use the Dice program (https://github.com/sanshar/Dice.git).''')
        
    make_hdiag_slow = staticmethod(hcilib.make_hdiag_slow)
    enlarge_space = enlarge_space
    kernel = kernel

HCI = HeatBathCI

if __name__ == '__main__':
    from pyscf import gto, scf, fci
    from pyscf.fci.direct_spin1 import _unpack_nelec
    from functools import reduce
    import math

    mol = gto.M(
        atom = '''
O        0.000000    0.000000    0.117790
H        0.000000    0.755453   -0.471161
H        0.000000   -0.755453   -0.471161''',
        basis = 'sto-3g',
        charge = 1,
        spin = 1  # = 2S = spin_up - spin_down
    )

    mf = scf.UHF(mol)
    mf.kernel()

    nelec = _unpack_nelec(mol.nelectron, mol.spin)
    nelec_a, nelec_b = nelec
    nao = mol.nao_nr()
    mo = mf.mo_coeff
    mo_a, mo_b = mo

    norb = mo.shape[1]
    hcore = mf.get_hcore()
    eri_ao = mf._eri
    add_thresh = 0.01
    nroots = 2

    myhci = HCI()
    e, coeff = myhci.kernel(hcore, eri_ao, mo, norb, nelec, add_thresh=add_thresh, nroots=nroots, verbose=5)

    ndets = math.comb(norb, nelec[0])*math.comb(norb, nelec[1])
    h1e_mo_aa = reduce(np.dot, (mo_a.conj().T, hcore, mo_a))
    h1e_mo_bb = reduce(np.dot, (mo_b.conj().T, hcore, mo_b))
    eri_mo_aaaa_s8 = ao2mo.restore('s8', ao2mo.full(eri_ao, mo_a), norb)
    eri_mo_bbbb_s8 = ao2mo.restore('s8', ao2mo.full(eri_ao, mo_b), norb)
    eri_mo_aabb_s4 = ao2mo.restore('s4', ao2mo.general(eri_ao, [mo_a, mo_a, mo_b, mo_b]), norb)
    addr, full_hamiltonian = fci.direct_uhf.pspace((h1e_mo_aa, h1e_mo_bb), (eri_mo_aaaa_s8, eri_mo_aabb_s4, eri_mo_bbbb_s8), norb, nelec, np=ndets)
    eigenvalues, eigenvectors = np.linalg.eigh(full_hamiltonian)
    print('FCI energies %s' % eigenvalues[eigenvalues<max(e)])