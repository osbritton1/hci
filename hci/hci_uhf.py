from hci import lib
from pyscf import ao2mo
from functools import reduce
import numpy as np

libhci = lib.load_library("libhci_rank")

def kernel(hcore, eri_ao, mo, norb, nelec, ci0=None, tol=None, lindep=None, max_cycle=None, max_space=None,
           nroots=None, davidson_only=None, max_memory=None, verbose=None, **kwargs):
    mo_a, mo_b = mo
    nelec_a, nelec_b = nelec
    h1e_aa = reduce(np.dot, (mo_a.conj().T, hcore, mo_a))
    h1e_bb = reduce(np.dot, (mo_b.conj().T, hcore, mo_b))
    eri_mo_aaaa = ao2mo.restore('s8', ao2mo.full(eri_ao, mo_a), norb)
    eri_mo_bbbb = ao2mo.restore('s8', ao2mo.full(eri_ao, mo_b), norb)
    eri_mo_aabb = ao2mo.restore('s4', ao2mo.general(eri_ao, [mo_a, mo_a, mo_b, mo_b]), norb)

def 