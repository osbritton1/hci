"""Pythonic interface for :ref:`c-rank` part of the :ref:`C lib`.
"""

import numpy as np
import numpy.typing as npt
import ctypes as ct
import math
from hci.lib.hci_utils import libhci

libhci.rank_occ_a.restype = ct.c_uint64
libhci.rank_occ_b.restype = ct.c_uint64
libhci.rank_double_exc.restype = ct.c_uint64
libhci.rank_mixed_exc.restype = ct.c_uint64

def _create_ranking_table(norb: int, nocc: int) -> npt.NDArray[np.uint64]:
    r"""Creates a ranking table establishing a bijection from the set of all combinations of :math:`[0,1,\dots N_\text{orb}-1]` 
    taken :math:`N_\text{occ}` at a time to :math:`\left[0,1,\dots \binom{N_\text{orb}}{N_\text{occ}}\right].`

    Args:
        norb: Size of orbital space
        nocc: Number of occupancies; generally equal to :math:`N_\alpha` or :math:`N_\beta`

    Returns: The constructed ranking table
    """    
    rank_table = np.empty((nocc, norb-nocc+1), dtype=np.uint64)
    libhci.load_rank_table(rank_table.ctypes, ct.c_size_t(norb), ct.c_size_t(nocc))
    return rank_table

class ConfigInfo:
    r"""Manages the ranking tables needed for encoding and decoding occupancy and excitation lists.
    When passed as an argument to a :py:mod:`ctypes` function, yields a reference to a struct
    compatible with the :ref:`C lib`'s :cpp:struct:`ConfigInfo`.
    """
    def __init__(self, norb: int, nelec: tuple[int, int]):
        r"""
        Args:
            norb: Size of the orbital space
            nelec: A tuple :math:`(N_\alpha, N_\beta)`
        """        
        nelec_a, nelec_b = nelec
        self.norb: int = norb #: Total size of input orbital space :math:`N_\text{orb}`
        self.nelec_a: int = nelec_a #: Number of :math:`\alpha` electrons :math:`N_\alpha`
        self.nelec_b: int = nelec_b #: Number of :math:`\beta` electrons :math:`N_\beta`
        self.occ_table_a: npt.NDArray[np.uint64] = _create_ranking_table(norb, nelec_a) #: Ranking table for determining occupied :math:`\alpha` orbitals
        self.virt_table_a: npt.NDArray[np.uint64] = _create_ranking_table(norb, norb-nelec_a) #: Ranking table for determining virtual :math:`\alpha` orbitals
        self.combmax_a: int = math.comb(norb, nelec_a) #: Total number of :math:`\alpha` occupancy lists, :math:`\binom{N_\text{orb}}{N_\alpha}`
        self.occ_table_b: npt.NDArray[np.uint64] = _create_ranking_table(norb, nelec_b) #: Ranking table for determining occupied :math:`\beta` orbitals
        self.virt_table_b: npt.NDArray[np.uint64] = _create_ranking_table(norb, norb-nelec_b) #: Ranking table for determining virtual :math:`\beta` orbitals
        self.combmax_b: int = math.comb(norb, nelec_b) #: Total number of :math:`\beta` occupancy lists, :math:`\binom{N_\text{orb}}{N_\beta}`
        self.exc_table_4o: npt.NDArray[np.uint64] = _create_ranking_table(norb, 4) 
        r"""Ranking table for organizing double excitations of :math:`\alpha\alpha\rightarrow\alpha\alpha` or 
        :math:`\beta\beta\rightarrow\beta\beta` type"""
        self.exc_table_2o: npt.NDArray[np.uint64] = _create_ranking_table(norb, 2)
        r"""Pointer to the ranking table for organizing mixed excitations of :math:`\alpha\beta\rightarrow\alpha\beta` type"""
        self.ncols_mixed: int = math.comb(norb, 2) #: :math:`\binom{N_\text{orb}}{2}`, used for ranking and unranking mixed excitations

    class _ConfigInfoPtrs(ct.Structure):
        """Internal class to manage :py:mod:`ctypes` interoperability.""" 
        _fields_ = [('norb', ct.c_size_t),
                    ('nelec_a', ct.c_size_t),
                    ('nelec_b', ct.c_size_t),
                    ('occ_table_a', ct.c_void_p),
                    ('virt_table_a', ct.c_void_p),
                    ('combmax_a', ct.c_uint64),
                    ('occ_table_b', ct.c_void_p),
                    ('virt_table_b', ct.c_void_p),
                    ('combmax_b', ct.c_uint64),
                    ('exc_table_4o', ct.c_void_p),
                    ('exc_table_2o', ct.c_void_p),
                    ('ncols_mixed', ct.c_uint64)]
    
    @property
    def _as_parameter_(self):
        """Returns a reference to a struct of pointers to the class's data.""" 
        ptrs = self._ConfigInfoPtrs(ct.c_size_t(self.norb),
                                    ct.c_size_t(self.nelec_a),
                                    ct.c_size_t(self.nelec_b),
                                    self.occ_table_a.ctypes.data_as(ct.c_void_p),
                                    self.virt_table_a.ctypes.data_as(ct.c_void_p),
                                    ct.c_uint64(self.combmax_a),
                                    self.occ_table_b.ctypes.data_as(ct.c_void_p),
                                    self.virt_table_b.ctypes.data_as(ct.c_void_p),
                                    ct.c_uint64(self.combmax_b),
                                    self.exc_table_4o.ctypes.data_as(ct.c_void_p),
                                    self.exc_table_2o.ctypes.data_as(ct.c_void_p),
                                    ct.c_uint64(self.ncols_mixed))
        return ct.byref(ptrs)
    
    def rank_occ_a(self, occ_list_a: npt.NDArray[np.uintp]) -> int:
        r"""Ranks a given :math:`\alpha` orbital occupancy list.

        Args:
            occ_list_a: :py:class:`numpy.ndarray` of length :math:`N_\alpha`
                specifying the zero-indexed occupied :math:`\alpha` orbitals in ascending order

        Returns:
            int: The rank of the specified :math:`\alpha` occupancy list
        """        
        return libhci.rank_occ_a(occ_list_a.ctypes, self)
    
    def unrank_occ_a(self, arank: int) -> npt.NDArray[np.uintp]:
        r"""Determines which orbitals are in an :math:`\alpha` occupancy list of given rank.

        Args:
            arank: The rank of the :math:`\alpha` occupancy list of interest

        Returns:
            The :math:`\alpha` occupancy list (length :math:`N_\alpha`)
        """        
        occ_list_a = np.empty(self.nelec_a, dtype=np.uintp)
        libhci.unrank_occ_a(ct.c_uint64(arank), occ_list_a.ctypes, self)
        return occ_list_a
    
    def unrank_virt_a(self, arank: int) -> npt.NDArray[np.uintp]:
        r"""Determines all orbitals in the complement of a :math:`\alpha` occupancy list with given rank.

        Args:
            arank: The rank of the :math:`\alpha` occupancy list of interest

        Returns:
            The :math:`\alpha` virtual orbital list (length :math:`N_\text{orb}-N_\alpha`)
        """        
        virt_list_a = np.empty(self.norb-self.nelec_a, dtype=np.uintp)
        libhci.unrank_virt_a(ct.c_uint64(arank), virt_list_a.ctypes, self)
        return virt_list_a
    
    def rank_occ_b(self, occ_list_b: npt.NDArray[np.uintp]) -> int:
        r"""Ranks a given :math:`\beta` orbital occupancy list.

        Args:
            occ_list_b: :py:class:`numpy.ndarray` of length :math:`N_\beta`
                specifying the zero-indexed occupied :math:`\alpha` orbitals in ascending order

        Returns:
            The rank of the specified :math:`\beta` occupancy list
        """        
        return libhci.rank_occ_b(occ_list_b.ctypes, self)
    
    def unrank_occ_b(self, brank: int) -> npt.NDArray[np.uintp]:
        r"""Determines which orbitals are in an :math:`\beta` occupancy list of given rank.

        Args:
            brank: The rank of the :math:`\beta` occupancy list of interest

        Returns:
            The :math:`\beta` occupancy list (length :math:`N_\beta`)
        """        
        occ_list_b = np.empty(self.nelec_b, dtype=np.uintp)
        libhci.unrank_occ_b(ct.c_uint64(brank), occ_list_b.ctypes, self)
        return occ_list_b
    
    def unrank_virt_b(self, brank: int) -> npt.NDArray[np.uintp]:
        r"""Determines all orbitals in the complement of a :math:`\beta` occupancy list with given rank.

        Args:
            brank: The rank of the :math:`\beta` occupancy list of interest

        Returns:
            The :math:`\beta` virtual orbital list (length :math:`N_\text{orb}-N_\beta`)
        """        
        virt_list_b = np.empty(self.norb-self.nelec_b, dtype=np.uintp)
        libhci.unrank_virt_b(ct.c_uint64(brank), virt_list_b.ctypes, self)
        return virt_list_b
    
    def rank_double_exc(self, exc_list: npt.NDArray[np.uintp]) -> int:
        """Ranks a double excitation based on the four orbitals involved.

        Args:
            exc_list: :py:class:`numpy.ndarray` of length 4 specifying the four involved orbitals in ascending order

        Returns:
            The rank of the specified double excitation
        """        
        return libhci.rank_double_exc(exc_list.ctypes, self)
    
    def unrank_double_exc(self, exc_rank: int) -> npt.NDArray[np.uintp]:
        """Determines the four orbitals involved in a double excitation of given rank.

        Args:
            The rank of the double excitation of interest

        Returns:
            The list of four orbitals involved in the double excitation (in ascending order)
        """        
        exc_list = np.empty(4, dtype=np.uintp)
        libhci.unrank_double_exc(ct.c_uint64(exc_rank), exc_list.ctypes, self)
        return exc_list
    
    def rank_mixed_exc(self, exc_list: npt.NDArray[np.uintp]) -> int:
        r"""Calculates the rank of a mixed :math:`\alpha\beta\rightarrow\alpha\beta` excitation.

        Args:
            exc_list: :py:class:`numpy.ndarray` specifying the four involved orbitals; 
                the first two are \f$\alpha\f$ orbitals (in ascending order) and the second two are \f$\beta\f$ orbitals (also in ascending order)

        Returns:
            The rank of the specified mixed excitation
        """        
        return libhci.rank_mixed_exc(exc_list.ctypes, self)
    
    def unrank_mixed_exc(self, exc_rank_ab: int) -> npt.NDArray[np.uintp]:
        r"""Determines the occupancy list corresponding to a mixed excitation with given rank.

        Args:
            exc_rank_ab: The rank of the mixed excitation

        Returns:
            The list of the four involved orbitals; 
                the first two are :math:`\alpha` orbitals (in ascending order) and the second two are :math:`\beta` orbitals (also in ascending order)
        """        
        exc_list = np.empty(4, dtype=np.uintp)
        libhci.unrank_mixed_exc(ct.c_uint64(exc_rank_ab), exc_list.ctypes, self)
        return exc_list