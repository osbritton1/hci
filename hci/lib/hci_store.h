#ifndef HCI_STORE_H
#define HCI_STORE_H

#include <stdlib.h>
#include <stdint.h>

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
} DoubleExcitationEntry;

/**
 * A struct to represent mixed ab excitations; i<j and k<l, but there are no restrictions
 * otherwise because orbitals i and j are alpha MOs, while k and l are beta MOs
 */
typedef struct {
    uint64_t rank; /**< The rank of the mixed excitation; see rank_mixed(size_t *occ_list, const uint64_t *exc_table_2o, size_t norb) */
    double ijkl; /**< [ij|kl] in chemist's notation; i and j are alpha orbitals, while k and l are beta orbitals */
} MixedExcitationEntry;

typedef struct {
    DoubleExcitationEntry *doubles_aa;
    double *max_mag_aa;
    size_t ndoubles_aa;
    DoubleExcitationEntry *doubles_bb;
    double *max_mag_bb;
    size_t ndoubles_bb;
    MixedExcitationEntry *mixed_ab;
    double *max_mag_ab;
    size_t nmixed_ab;
} ExcitationEntries;

typedef struct {
    uint64_t arank;
    uint64_t brank;
} Rank;

typedef struct {
    Rank *ranks;
    double *coeffs;
    size_t len;
} HCIVector;

typedef struct {
    double *h1e_aa;
    double *h1e_bb;
} H1E;

typedef struct {
    double *eri_mo_aaaa_s8;
    double *eri_mo_bbbb_s8;
    double *eri_mo_aabb_s4;
} ERI_MO;

void get_max_magnitudes(const DoubleExcitationEntry *doubles, double *magnitudes, size_t ndoubles);
size_t index_2d(size_t i, size_t j);
size_t index_4d(size_t i, size_t j, size_t k, size_t l, size_t ncols);
size_t index_8d(size_t i, size_t j, size_t k, size_t l);
void load_doubles_from_eri(DoubleExcitationEntry *doubles, const double *eri_s8, const uint64_t *exc_table_4o, size_t norb);
void load_mixed_from_eri(MixedExcitationEntry *mixed, double *eri_s4, const uint64_t *exc_table_2o, size_t norb);

#endif