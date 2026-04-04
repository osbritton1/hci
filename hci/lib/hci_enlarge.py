"""Pythonic interface for :ref:`c-enlarge` part of the :ref:`C lib`.
"""

import numpy as np
import numpy.typing as npt
import ctypes as ct
import math
from hci.lib.hci_utils import libhci
from hci.lib.hci_rank import ConfigInfo
from hci.lib.hci_store import HCore, ERITensor, HCIVec, ExcEntries, rank_entry

def enlarge_space_doubles(hcivec: HCIVec, add_thresh: float, config_info: ConfigInfo, 
                          exc_entries: ExcEntries) -> npt.NDArray[np.void]:
    """Given a reference :py:class:`HCIVec`, uses the HCI selection algorithm
    to generate a list of doubly excited deterinants to add to the configuration space.

    Args:
        hcivec: Reference :py:class:`HCIVec` tagged with list of ranks describing 
            the current configuration space
        add_thresh: Selection threshold for adding a new determinant. A candidate determinant is added
            if the magnitude of its matrix element with a determinant from **hcivec** times the
            coefficient of that  determinant exceeds **add_thresh**.
        config_info: :py:class:`ConfigInfo` instance supplying ranking tables and key parameters of the system
        exc_entries: :py:class:`ExcEntries` object storing double excitations sorted desc. by magnitude 

    Raises:
        HCIVec.RankNotFoundError: rank attribute of **hcivec** is missing, which is needed to define the current
            configuration space.
        ExcEntries.SortTypeError: **exc_entries** has not been sorted in descending order by magnitude.

    Returns:
        A list of ranks corresponding to configurations that passed the selection threshold
    """    
    if hcivec.ranks is None:
        raise HCIVec.RankNotSetError("Rank attribute of input HCIVec was not set.")
    if exc_entries.sort_type != ExcEntries.SortType.DESC_BY_MAG:
        raise ExcEntries.SortTypeError("Calling this method requires the excitation entries to be sorted descending by magnitude.")
    norb = config_info.norb
    nelec_a = config_info.nelec_a
    nelec_b = config_info.nelec_b
    nexc_aa = math.comb(nelec_a, 2)*math.comb(norb-nelec_a, 2)
    nexc_bb = math.comb(nelec_b, 2)*math.comb(norb-nelec_b, 2)
    nexc_ab = nelec_a*(norb-nelec_a)*nelec_b*(norb-nelec_b)
    add_list = np.empty(len(hcivec.ranks)*(nexc_aa+nexc_bb+nexc_ab), dtype=rank_entry)
    nadd = libhci.enlarge_space_doubles(hcivec, add_list.ctypes, ct.c_double(add_thresh), config_info, exc_entries)
    return add_list[:nadd]

def enlarge_space_singles(hcivec: HCIVec, add_thresh: float, config_info: ConfigInfo, 
                          h1e: HCore, eri_mo: ERITensor) -> npt.NDArray[np.void]:
    """Given a reference :py:class:`HCIVec`, uses the HCI selection algorithm
    to generate a list of singly excited deterinants to add to the configuration space.

    Args:
        hcivec: Reference :py:class:`HCIVec` tagged with list of ranks describing 
            the current configuration space
        add_thresh: Selection threshold for adding a new determinant. A candidate determinant is added
            if the magnitude of its matrix element with a determinant from **hcivec** times the
            coefficient of that  determinant exceeds **add_thresh**.
        config_info: :py:class:`ConfigInfo` instance supplying ranking tables and key parameters of the system
        h1e: The core Hamiltonian in the MO bases
        eri_mo: The electron repulsion integrals in the MO bases

    Raises:
        HCIVec.RankNotSetError: rank attribute of **hcivec** is missing, which is needed to define the current
            configuration space.

    Returns:
        A list of ranks corresponding to configurations that passed the selection threshold
    """    
    if hcivec.ranks is None:
        raise HCIVec.RankNotSetError("Rank attribute of input HCIVec was not set.")
    norb = config_info.norb
    nelec_a = config_info.nelec_a
    nelec_b = config_info.nelec_b
    nexc_a = nelec_a*(norb-nelec_a)
    nexc_b = nelec_b*(norb-nelec_b)
    add_list = np.empty(len(hcivec.ranks)*(nexc_a+nexc_b), dtype=rank_entry)
    nadd = libhci.enlarge_space_singles(hcivec, add_list.ctypes, ct.c_double(add_thresh), config_info, h1e, eri_mo)
    return add_list[:nadd]