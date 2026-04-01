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

typedef struct {
    double *h1e_mo_aa;
    double *h1e_mo_bb;
} HCore;

typedef struct {
    double *eri_mo_aaaa_s8;
    double *eri_mo_bbbb_s8;
    double *eri_mo_aabb_s4;
    uint64_t ncols_aabb;
} ERITensor;

/**
 * A struct that stores the minimal amount of information necessary
 * to compute a double excitation matrix element given the four changing
 * orbitals encoded by rank (referred to as i, j, k, and l in ascending order).
 * [ik|lj]-[ij|lk] may be recovered by computing negative the sum of ijkl and iljk;
 * all double excitation matrix elements involving these four orbitals may be mapped
 * to one of these three values up to a sign.
 */
typedef struct {
    uint64_t rank; /**< The combinatorial rank of the list of four changing orbitals; see rank(const size_t *occ_list, const uint64_t *rank_table, size_t norb, size_t nocc) */
    double ijkl; /**< [ij|kl]-[il|kj] in chemist's notation */
    double iljk; /**< [il|jk]-[ik|jl] in chemist's notation */
} DoubleExcEntry;

/**
 * A struct to represent mixed ab excitations; i<j and k<l, but there are no restrictions
 * otherwise because orbitals i and j are alpha MOs, while k and l are beta MOs
 */
typedef struct {
    uint64_t rank; /**< The rank of the mixed excitation; see rank_mixed(size_t *occ_list, const uint64_t *exc_table_2o, size_t norb) */
    double ijkl; /**< [ij|kl] in chemist's notation; i and j are alpha orbitals, while k and l are beta orbitals */
} MixedExcEntry;

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

typedef struct {
    uint64_t arank;
    uint64_t brank;
} Rank;

typedef struct {
    Rank *ranks;
    double *coeffs;
    size_t len;
} HCIVec;

size_t index_2d(size_t i, size_t j);
size_t index_4d(size_t i, size_t j, size_t k, size_t l, size_t ncols);
size_t index_8d(size_t i, size_t j, size_t k, size_t l);
void get_max_magnitudes(const DoubleExcEntry *doubles, double *magnitudes, size_t ndoubles);
void load_exc_entries_from_eri(ExcEntries *exc_entries, ERITensor *eri_mo, const ConfigInfo *config_info);

#endif