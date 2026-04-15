"""Pythonic interface for :ref:`c-store` part of the :ref:`C lib`.
"""

import numpy as np
import numpy.typing as npt
from typing import Any
import ctypes as ct
import math
from enum import Enum
from hci.lib.hci_rank import ConfigInfo
from hci.lib.hci_utils import libhci
from hci.exceptions import HCIError

rank_entry = np.dtype([('arank', np.uint64), ('brank', np.uint64)])
r"""
.. _numpy-rank:

A :py:class:`numpy.dtype` representing a configuration via the 
ranks of its :math:`\alpha` and :math:`\beta` occupancy lists.
Directly mirrors the :ref:`C lib`'s :cpp:struct:`Rank`.
"""

class Rank(ct.Structure):
    """The equivalent of the :ref:`C lib`'s :cpp:struct:`Rank`.
    Used for passing individual ranks to the :ref:`C lib` directly (as
    opposed to passing in a structured :py:mod:`numpy` array).
    """
    _fields_ = [('arank', ct.c_uint64),
                ('brank', ct.c_uint64)]
    
    @classmethod
    def from_rank_entry(cls, rank_entry: np.void) -> 'Rank':
        """Converts a compatible structured scalar to a :py:class:`Rank`.

        Args:
            rank_entry (np.void): structured scalar produced by indexing a
                :py:class:`numpy.ndarray` of dtype :ref:`rank_entry <numpy-rank>`.

        Returns:
            A new :py:class:`Rank` containing the information from the structured scalar.
        """        
        return cls(rank_entry['arank'], rank_entry['brank'])

double_exc_entry = np.dtype([('rank', np.uint64), ('ijkl', np.double), ('iljk', np.double)])
r"""
A :py:class:`numpy.dtype` containing the minimal information
necessary for computing any double excitation matrix element
involving the same four changing orbitals. Directly mirrors
the :ref:`C lib`'s :cpp:struct:`DoubleExcEntry`.
"""

mixed_exc_entry = np.dtype([('rank', np.uint64), ('ijkl', np.double)])
r"""
A :py:class:`numpy.dtype` representing a mixed excitation.
Directly mirrors the :ref:`C lib`'s :cpp:struct:`MixedExcEntry`.
"""

class HCore:
    r"""Class storing the core Hamiltonian in both the :math:`\alpha` and :math:`\beta`
    MO bases for ease of manipulation. When passed as an argument to a :py:mod:`ctypes` function, 
    yields a reference to a struct compatible with the :ref:`C lib`'s :cpp:struct:`HCore`.
    """    
    def __init__(self, h1e_mo_aa: npt.NDArray[np.float64], h1e_mo_bb: npt.NDArray[np.float64]):
        r"""
        Args:
            h1e_mo_aa: The core Hamiltonian in the :math:`\alpha` MO basis. Should be of shape :math:`(N_\text{orb}, N_\text{orb})`.
            h1e_mo_bb: The core Hamiltonian in the :math:`\beta` MO basis. Should be of shape :math:`(N_\text{orb}, N_\text{orb})`.
        """        
        self.h1e_mo_aa = h1e_mo_aa
        self.h1e_mo_bb = h1e_mo_bb
        
    class _HCorePtrs(ct.Structure):
        """Internal class to manage :py:mod:`ctypes` interoperability.""" 
        _fields_ = [('h1e_mo_aa', ct.c_void_p),
                    ('h1e_mo_bb', ct.c_void_p)]
    
    @property
    def _as_parameter_(self):
        """Returns a reference to a struct of pointers to the class's data.""" 
        ptrs = self._HCorePtrs(self.h1e_mo_aa.ctypes.data_as(ct.c_void_p),
                               self.h1e_mo_bb.ctypes.data_as(ct.c_void_p))
        return ct.byref(ptrs)

class ERITensor:
    r"""Class storing the electron repulsion integrals. For a UHF starting point, 
    three MO transformed tensors are needed of :math:`[\alpha\alpha|\alpha\alpha]`,
    :math:`[\alpha\alpha|\beta\beta]`, and :math:`[\beta\beta|\beta\beta]` type.
    When passed as an argument to a :py:mod:`ctypes` function, yields a reference to 
    a struct compatible with the :ref:`C lib`'s :cpp:struct:`ERITensor`.
    """    
    def __init__(self, eri_mo_aaaa_s8: npt.NDArray[np.float64],
                 eri_mo_bbbb_s8: npt.NDArray[np.float64], 
                 eri_mo_aabb_s4: npt.NDArray[np.float64], 
                 config_info: ConfigInfo):
        r"""
        Args:
            eri_mo_aaaa_s8: Eightfold compressed ERI tensor in the :math:`\alpha` MO basis
            eri_mo_bbbb_s8: Eightfold compressed ERI tensor in the :math:`\beta` MO basis
            eri_mo_aabb_s4: Fourfold compressed ERI tensor in the mixed MO basis
            config_info: :py:class:`ConfigInfo` instance supplying ranking tables and key parameters of the system
        """        
        self.eri_mo_aaaa_s8 = eri_mo_aaaa_s8 #: Eightfold compressed ERI tensor in the :math:`\alpha` MO basis
        self.eri_mo_bbbb_s8 = eri_mo_bbbb_s8 #: Eightfold compressed ERI tensor in the :math:`\beta` MO basis
        self.eri_mo_aabb_s4 = eri_mo_aabb_s4 #: Fourfold compressed ERI tensor in the mixed MO basis
        self.ncols_aabb = math.comb(config_info.norb+1, 2) 
        r""":math:`\binom{N_\text{orb}+1}{2}`, i.e. the number of entries in the mixed ERI tensor associated 
        with a single pair of :math:`\alpha` orbitals; needed for looping constructs"""
        
    class _ERITensorPtrs(ct.Structure):
        """Internal class to manage :py:mod:`ctypes` interoperability.""" 
        _fields_ = [('eri_mo_aaaa_s8', ct.c_void_p),
                    ('eri_mo_bbbb_s8', ct.c_void_p),
                    ('eri_mo_aabb_s4', ct.c_void_p),
                    ('ncols_aabb', ct.c_size_t)]
        
    @property
    def _as_parameter_(self):
        """Returns a reference to a struct of pointers to the class's data.""" 
        ptrs = self._ERITensorPtrs(self.eri_mo_aaaa_s8.ctypes.data_as(ct.c_void_p),
                                   self.eri_mo_bbbb_s8.ctypes.data_as(ct.c_void_p),
                                   self.eri_mo_aabb_s4.ctypes.data_as(ct.c_void_p),
                                   ct.c_size_t(self.ncols_aabb))
        return ct.byref(ptrs)

class ExcEntries:
    """A class that stores all the double excitation matrix elements needed for carrying
    out the HCI algorithm. When passed as an argument to a :py:mod:`ctypes` function, yields 
    a reference to a struct compatible with the :ref:`C lib`'s :cpp:struct:`ExcEntries`.
    """
    def __init__(self, eri_mo: ERITensor, config_info: ConfigInfo):
        """
        Args:
            eri_mo: :py:class:`ERITensor` instance from which to calculate the double excitation matrix elements
            config_info: :py:class:`ConfigInfo` instance supplying ranking tables and key parameters of the system
        """        
        self.ndoubles_aa: int = math.comb(config_info.norb, 4) #: :math:`\binom{N_\text{orb}}{4}`, the number of double :math:`\alpha` excitations
        self.doubles_aa: npt.NDArray[np.void] = np.empty(self.ndoubles_aa, dtype=double_exc_entry) #: The stored double :math:`\alpha` excitation matrix elements
        self.max_mag_aa: npt.NDArray[np.float64] = np.empty(self.ndoubles_aa, dtype=np.float64)
        r"""Array specifying the max. magnitude of all double :math:`\alpha` matrix elements associated with the same four orbitals"""

        self.ndoubles_bb = math.comb(config_info.norb, 4) #: :math:`\binom{N_\text{orb}}{4}`, the number of double :math:`\beta` excitations
        self.doubles_bb: npt.NDArray[np.void] = np.empty(self.ndoubles_bb, dtype=double_exc_entry)  #: The stored double :math:`\beta` excitation matrix elements
        self.max_mag_bb: npt.NDArray[np.float64] = np.empty(self.ndoubles_bb, dtype=np.float64)
        r"""Array specifying the max. magnitude of all double :math:`\beta` matrix elements associated with the same four orbitals"""

        self.nmixed_ab = math.comb(config_info.norb, 2)**2 #: :math:`\binom{N_\text{orb}}{2}^2`, the number of mixed excitations
        self.mixed_ab: npt.NDArray[np.void] = np.empty(self.nmixed_ab, dtype=mixed_exc_entry) #: The stores mixed excitation matrix elements
        self.max_mag_ab: npt.NDArray[np.float64] = np.empty(self.nmixed_ab, dtype=np.float64) #: Array containing the magnitudes of the mixed excitation matrix elements

        libhci.load_exc_entries_from_eri(self, eri_mo, config_info)

        self.sort_type: ExcEntries.SortType = self.SortType.INC_BY_RANK #: All excitations are initially computed in order of increasing rank

    class _ExcEntriesPtrs(ct.Structure):
        """Internal class to manage :py:mod:`ctypes` interoperability.""" 
        _fields_ = [('doubles_aa', ct.c_void_p),
                    ('max_mag_aa', ct.c_void_p),
                    ('ndoubles_aa', ct.c_size_t),
                    ('doubles_bb', ct.c_void_p),
                    ('max_mag_bb', ct.c_void_p),
                    ('ndoubles_bb', ct.c_size_t),
                    ('mixed_ab', ct.c_void_p),
                    ('max_mag_ab', ct.c_void_p),
                    ('nmixed_ab', ct.c_size_t)]
        
    @property
    def _as_parameter_(self):
        """Returns a reference to a struct of pointers to the class's data.""" 
        ptrs = self._ExcEntriesPtrs(self.doubles_aa.ctypes.data_as(ct.c_void_p),
                                    self.max_mag_aa.ctypes.data_as(ct.c_void_p),
                                    ct.c_size_t(self.ndoubles_aa),
                                    self.doubles_bb.ctypes.data_as(ct.c_void_p),
                                    self.max_mag_bb.ctypes.data_as(ct.c_void_p),
                                    ct.c_size_t(self.ndoubles_bb),
                                    self.mixed_ab.ctypes.data_as(ct.c_void_p),
                                    self.max_mag_ab.ctypes.data_as(ct.c_void_p),
                                    ct.c_size_t(self.nmixed_ab))
        return ct.byref(ptrs)
    
    class SortType(Enum):
        """Flag enum to indicate how the excitations are sorted."""        
        INC_BY_RANK = 1
        DESC_BY_MAG = 2

    class SortTypeError(HCIError):
        """Error raised if the :py:class:`ExcEntries.SortType` is incorrect."""

    def sort_desc_by_mag(self):
        """Sorts the excitation lists in order of decreasing magnitude"""        
        sorted_indices = np.argsort(self.max_mag_aa)[::-1]
        self.doubles_aa = self.doubles_aa[sorted_indices]
        self.max_mag_aa = self.max_mag_aa[sorted_indices]
        sorted_indices = np.argsort(self.max_mag_bb)[::-1]
        self.doubles_bb = self.doubles_bb[sorted_indices]
        self.max_mag_bb = self.max_mag_bb[sorted_indices]
        sorted_indices = np.argsort(self.max_mag_ab)[::-1]
        self.mixed_ab = self.mixed_ab[sorted_indices]
        self.max_mag_ab = self.max_mag_ab[sorted_indices]
        self.sort_type = self.SortType.DESC_BY_MAG
    
    def sort_inc_by_rank(self):
        """Sorts the excitation lists in order of decreasing magnitude""" 
        sorted_indices = np.argsort(self.doubles_aa, order='rank')
        self.doubles_aa = self.doubles_aa[sorted_indices]
        self.max_mag_aa = self.max_mag_aa[sorted_indices]
        sorted_indices = np.argsort(self.doubles_bb, order='rank')
        self.doubles_bb = self.doubles_bb[sorted_indices]
        self.max_mag_bb = self.max_mag_bb[sorted_indices]
        sorted_indices = np.argsort(self.mixed_ab, order='rank')
        self.mixed_ab = self.mixed_ab[sorted_indices]
        self.max_mag_ab = self.max_mag_ab[sorted_indices]
        self.sort_type = self.SortType.INC_BY_RANK

class HCIVec(npt.NDArray[Any]):
    """Subclass of :py:mod:`numpy.ndarray` tagged by a list of ranks. When passed as 
    an argument to a :py:mod:`ctypes` function, yields a reference to a struct compatible 
    with the :ref:`C lib`'s :cpp:struct:`HCIVec`.
    """    
    def __new__(cls, input_array: npt.NDArray[Any], ranks: npt.NDArray[np.void] | None=None):
        hcivec = np.asarray(input_array).view(cls)
        hcivec.ranks = ranks
        return hcivec

    def __array_finalize__(self, obj: npt.NDArray[Any] | None):
        if obj is None: return
        self.ranks = getattr(obj, 'ranks', None)

    # Special cases for ndarray when the array was modified (through ufunc)
    # def __array_wrap__(self, out, context=None, return_scalar=False):
    #     if out.shape == self.shape:
    #         return out
    #     elif out.shape == ():  # if ufunc returns a scalar
    #         return out[()]
    #     else:
    #         return out.view(np.ndarray)

    @classmethod
    def as_HCIVec(cls: type['HCIVec'], coeffs: npt.NDArray[np.float64], ranks: npt.NDArray[np.void]) -> 'HCIVec':
        """Factory method to tag a preexisting list of coefficients with their ranks.

        Args:
            coeffs: The coefficients of the :py:class:`HCIVec`
            ranks: The rank list to use for tagging

        Returns:
            The :py:class:`HCIVec` formed by tagging the coefficient list with the ranks list
        """        
        hcivec = coeffs.view(HCIVec)
        hcivec.ranks = ranks
        return hcivec
    
    @classmethod
    def as_HCIVec_if_not(cls: type['HCIVec'], vec: npt.NDArray[Any], ranks: npt.NDArray[np.void]) -> 'HCIVec':
        """Factory method to tag an array with associated ranks if they are not already set.

        Args:
            vec: Input array that may or may not have ranks associated with it
            ranks: The rank list to use for tagging

        Returns:
            The :py:class:`HCIVec` formed by tagging the input array with the ranks list if no
            preexisting ranks were found
        """  
        if getattr(vec, 'ranks', None) is None:
            vec = HCIVec.as_HCIVec(vec, ranks)
        return vec.view(HCIVec)

    class _HCIVecPtrs(ct.Structure):
        """Internal class to manage :py:mod:`ctypes` interoperability.""" 
        _fields_ = [('ranks', ct.c_void_p),
                    ('coeffs', ct.c_void_p),
                    ('len', ct.c_size_t)]

    @property
    def _as_parameter_(self):
        """Returns a reference to a struct of pointers to the class's data.""" 
        if self.ranks is not None:
            ptrs = self._HCIVecPtrs(self.ranks.ctypes.data_as(ct.c_void_p),
                                    self.ctypes.data_as(ct.c_void_p),
                                    ct.c_size_t(len(self)))
            return ct.byref(ptrs)
        else:
            ptrs = self._HCIVecPtrs(None,
                                    self.ctypes.data_as(ct.c_void_p),
                                    ct.c_size_t(len(self)))
            return ct.byref(ptrs)
    
    class RankNotSetError(HCIError):
        """Error type indicating the rank attribute hasn't been set."""