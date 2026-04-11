"""Pythonic interface for :ref:`c-contract` part of the :ref:`C lib`.
"""

import numpy as np
import numpy.typing as npt
import ctypes as ct
from hci.lib.hci_utils import libhci
from hci.lib.hci_rank import ConfigInfo
from hci.lib.hci_store import HCore, ERITensor, HCIVec, ExcEntries
from hci.lib.hci_store import Rank

libhci.get_matrix_element_by_rank.restype = ct.c_double
libhci.get_matrix_element_by_rank_test_storage.restype = ct.c_double

def get_matrix_element_by_rank(rank1: Rank, rank2: Rank, config_info: ConfigInfo, 
                               h1e: HCore, eri_mo: ERITensor) -> float:
    r"""Given the ranks of two configurations, calculates the matrix element between them.

    Args:
        rank1: Rank specifying the :math:`\alpha` and :math:`\beta` string of the first configuration
        rank2: Rank specifying the :math:`\alpha` and :math:`\beta` string of the second configuration
        config_info: :py:class:`ConfigInfo` instance supplying ranking tables and key parameters of the system
        h1e: The core Hamiltonian in the MO bases
        eri_mo: The electron repulsion integrals in the MO bases

    Returns:
        float: The matrix element between the configurations specified
    """    
    return libhci.get_matrix_element_by_rank(rank1, rank2, config_info, h1e, eri_mo)
    
def get_matrix_element_by_rank_test_storage(rank1: Rank, rank2: Rank, config_info: ConfigInfo, 
                                            exc_entries: ExcEntries, h1e: HCore, eri_mo: ERITensor) -> float:
    r"""Given the ranks of two configurations, calculates the matrix element between them. 
    For double excitations, uses the values stored in **exc_entries**.

    Args:
        rank1: Rank specifying the :math:`\alpha` and :math:`\beta` string of the first configuration
        rank2: Rank specifying the :math:`\alpha` and :math:`\beta` string of the second configuration
        config_info: :py:class:`ConfigInfo` instance supplying ranking tables and key parameters of the system
        exc_entries: :py:class:`ExcEntries` object storing double excitations sorted inc. by rank
        h1e: The core Hamiltonian in the MO bases
        eri_mo: The electron repulsion integrals in the MO bases

    Raises:
        ExcEntries.SortTypeError: **exc_entries** has not been sorted in increasing order by rank.

    Returns:
        float: The matrix element between the configurations specified
    """
    if exc_entries.sort_type != ExcEntries.SortType.INC_BY_RANK:
        raise ExcEntries.SortTypeError("Calling this method requires the excitation entries to be sorted increasing by rank.")
    return libhci.get_matrix_element_by_rank_test_storage(rank1, rank2, config_info, exc_entries, h1e, eri_mo)

def make_hdiag_slow(hcivec: HCIVec, config_info: ConfigInfo, h1e: HCore, eri_mo: ERITensor) -> npt.NDArray[np.float64]:
    """Assembles all diagonal matrix elements for configurations contained in **hcivec**; used for the Davidson preconditioner.

    Args:
        hcivec: Reference :py:class:`HCIVec` tagged with list of ranks describing 
            the current configuration space
        config_info: :py:class:`ConfigInfo` instance supplying ranking tables and key parameters of the system
        h1e: The core Hamiltonian in the MO bases
        eri_mo: The electron repulsion integrals in the MO bases

    Raises:
        HCIVec.RankNotSetError: rank attribute of **hcivec** is missing, which is needed to define the current
            configuration space.

    Returns:
        npt.NDArray[np.float64]: The diagonal matrix elements of the Hamiltonian in the subspace defined by **hcivec**.
    """    
    if hcivec.ranks is None:
        raise HCIVec.RankNotSetError("Rank attribute of input HCIVec was not set.")
    hdiag = np.empty(len(hcivec.ranks), dtype=np.float64)
    libhci.make_hdiag_slow(hcivec, hdiag.ctypes, config_info, h1e, eri_mo)
    return hdiag

def contract_hamiltonian_hcivec_slow(hcivec_old: HCIVec, hdiag: npt.NDArray[np.float64], config_info: ConfigInfo, 
                                     h1e: HCore, eri_mo: ERITensor) -> npt.NDArray[np.float64]:
    """Contracts **hcivec_old** with the Hamiltonian in the current configuration space
    in a naive fashion, outputting the new coefficients.

    Args:
        hcivec_old: Reference :py:class:`HCIVec` tagged with list of ranks describing 
            the current configuration space
        hdiag: The diagonal matrix elements of the Hamiltonian in the subspace defined by **hcivec_old**
        config_info: :py:class:`ConfigInfo` instance supplying ranking tables and key parameters of the system
        h1e: The core Hamiltonian in the MO bases
        eri_mo: The electron repulsion integrals in the MO bases

    Raises:
        HCIVec.RankNotSetError: rank attribute of **hcivec_old** is missing, which is needed to define the current
            configuration space.

    Returns:
        npt.NDArray[np.float64]: The coefficients after contraction with the Hamiltonian
    """   
    if hcivec_old.ranks is None:
        raise HCIVec.RankNotSetError("Rank attribute of input HCIVec was not set.")
    coeffs_new = np.empty(len(hcivec_old), dtype=np.float64)
    libhci.contract_hamiltonian_hcivec_slow(hcivec_old, coeffs_new.ctypes, hdiag.ctypes, config_info, h1e, eri_mo)
    return coeffs_new