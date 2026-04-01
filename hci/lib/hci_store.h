/**
 * \ingroup store
 */

#ifndef HCI_STORE_H
#define HCI_STORE_H

#include "hci_rank.h"

/**
 * Function-like macro to compute the maximum of two numbers
 * @param[in] a
 * @param[in] b
 * @returns max(a,b)
 */
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/**
 * Function-like macro to compute the maximum of three numbers
 * @param[in] a
 * @param[in] b
 * @param[in] c
 * @returns max(a,b,c)
 */
#define MAX3(a, b, c) ((a) > (b) ? ((a) > (c) ? (a) : (c)) : ((b) > (c) ? (b) : (c)))

/**
 * A structure that holds pointers to the core Hamiltonian
 * in both the \f$\alpha\f$ MO and \f$\beta\f$ MO bases.
 *
 * Neither array is expected to be compressed (i.e. both must
 * have all \f$N_\text{orb}^2\f$ entries).
 */
typedef struct {
    double *h1e_mo_aa; /**< Pointer to the core Hamiltonian in the \f$\alpha\f$ MO basis */
    double *h1e_mo_bb; /**< Pointer to the core Hamiltonian in the \f$\beta\f$ MO basis */
} HCore;

/**
 * A structure that holds pointers to the electron repulsion integrals in
 * the \f$\alpha\f$ MO and \f$\beta\f$ MO bases (in chemist's notation).
 *
 * The tag at the end of each field name indicates the expected degree of compression
 * (see https://pyscf.org/pyscf_api_docs/pyscf.ao2mo.html).
 */
typedef struct {
    double *eri_mo_aaaa_s8; /** Pointer to the eightfold compressed ERI tensor in the \f$\alpha\f$ MO basis */
    double *eri_mo_bbbb_s8; /** Pointer to the eightfold compressed ERI tensor in the \f$\beta\f$ MO basis */
    double *eri_mo_aabb_s4; /** Pointer to the fourfold compressed ERI tensor in the mixed MO basis */
    uint64_t ncols_aabb; /** \f$\binom{N_\text{orb}+1}{2}$, i.e. the number of entries in the mixed ERI tensor associated with a single pair of \f$\alpha\f$ orbitals */
} ERITensor;

/**
 * A struct that stores the minimal amount of information necessary
 * to compute a double excitation matrix element given the four changing
 * orbitals encoded by rank (referred to as i, j, k, and l in ascending order).
 *
 * [ik|lj]-[ij|lk] may be recovered by computing negative the sum of ijkl and iljk;
 * all double excitation matrix elements involving these four orbitals may be mapped
 * to one of these three values up to a sign (see 
 * get_double_exc_value_from_store(const DoubleExcEntry *exc_entry, const ExcResult *double_exc)).
 */
typedef struct {
    uint64_t rank; /**< The combinatorial rank of the list of four changing orbitals; see rank_double_exc(size_t *exc_list, const ConfigInfo *config_info) */
    double ijkl; /**< [ij|kl]-[il|kj] in chemist's notation */
    double iljk; /**< [il|jk]-[ik|jl] in chemist's notation */
} DoubleExcEntry;

/**
 * A struct to represent mixed excitations.
 *
 * Because the ERI tensor is real, only excitations with \f$i<j\f$
 * and \f$k<l\f$ need to be stored; there are no restrictions otherwise
 * because \f$i\f$ and \f$j\f$ index \f$\alpha\f$ MOs while \f$k\f$ and
 * \f$l\f$ index \f$\beta\f$ MOs.
 */
typedef struct {
    uint64_t rank; /**< The rank of the mixed excitation; see rank_mixed(size_t *occ_list, const uint64_t *exc_table_2o, size_t norb) */
    double ijkl; /**< [ij|kl] in chemist's notation; i and j are alpha orbitals, while k and l are beta orbitals */
} MixedExcEntry;

/**
 * A struct that organizes information about stored double excitation
 * matrix elements.
 *
 * Used to enlarge the configuration space according to the HCI algorithm.
 */
typedef struct {
    DoubleExcEntry *doubles_aa;
    double *max_mag_aa;
    size_t ndoubles_aa;
    DoubleExcEntry *doubles_bb;
    double *max_mag_bb;
    size_t ndoubles_bb;
    MixedExcEntry *mixed_ab;
    double *max_mag_ab;
    size_t nmixed_ab;
} ExcEntries;

size_t index_2d(size_t i, size_t j);
size_t index_4d(size_t i, size_t j, size_t k, size_t l, size_t ncols);
size_t index_8d(size_t i, size_t j, size_t k, size_t l);
void load_exc_entries_from_eri(ExcEntries *exc_entries, ERITensor *eri_mo, const ConfigInfo *config_info);

#endif