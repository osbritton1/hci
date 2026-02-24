from hci import lib
from pyscf import ao2mo
from functools import reduce
import numpy as np

def kernel(hcore, eri_ao, mo, norb, nelec, ci0=None, tol=None, lindep=None, max_cycle=None, max_space=None,
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
    config_table_a, config_table_b, exc_table_4o, exc_table_2o = lib.get_ranking_tables(norb, nelec)

    # Build and sort double excitation storage lists
    doubles_aa = lib.get_stored_double_exc(eri_mo_aaaa, exc_table_4o, norb)
    max_mag_aa = lib.get_max_magnitudes(doubles_aa, norb)
    doubles_aa = doubles_aa[np.argsort(max_mag_aa)[::-1]]
    
    doubles_bb = lib.get_stored_double_exc(eri_mo_bbbb, exc_table_4o, norb)
    max_mag_bb = lib.get_max_magnitudes(doubles_bb, norb)
    doubles_bb = doubles_bb[np.argsort(max_mag_bb)[::-1]]
    
    mixed_ab = lib.get_stored_mixed_exc(eri_mo_aabb, exc_table_2o, norb)
    max_mag_ab = np.abs(mixed_ab['ijkl'])
    mixed_ab = mixed_ab[np.argsort(max_mag_ab)[::-1]]

    #Initialize HCI vector if initial guess not provided
    if ci0 is None:
        ci0 = np.empty(1, dtype=lib.hci_entry)
        ci0[0] = (0, 0, 1.0)
        
    return ci0