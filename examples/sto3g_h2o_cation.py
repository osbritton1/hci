# type: ignore
from hci.hci_uhf import HCI
from pyscf import gto, scf, fci, ao2mo
from pyscf.fci.direct_spin1 import _unpack_nelec
import numpy as np
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