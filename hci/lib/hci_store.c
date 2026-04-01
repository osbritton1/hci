#include "hci_store.h"
#include <math.h>

/**
 * Helper function for computing nCr(n,2)
 * @param[in] n
 * @return nCr(n,2)
 */
static uint64_t nC2(size_t n) {
    return (n % 2 == 0) ? n/2*(n-1) : (n-1)/2*n;
}

/**
 * Given two indices i and j, computes the index in the corresponding row-major packed
 * lower triangular array
 * @param[in] i The row index
 * @param[in] j The column index
 * @return the index of the (i,j) entry in a row-major packed lower triangular array
 */
size_t index_2d(size_t i, size_t j) {
    return (i > j) ? nC2(i+1)+j : nC2(j+1)+i;
}

/**
 * Computes the index of [ij|kl] in a fourfold-compressed ERI tensor.
 * ncols should be norb*(norb+1)/2
 * @param[in] i The zero-based index of the first orbital
 * @param[in] j The zero-based index of the second orbital
 * @param[in] k The zero-based index of the third orbital
 * @param[in] l The zero-based index of the fourth orbital
 * @return the index of [ij|kl] in a fourfold-compressed ERI tensor
 */
size_t index_4d(size_t i, size_t j, size_t k, size_t l, size_t ncols) {
    return (index_2d(i, j)*ncols) + index_2d(k, l);
}

/**
 * Computes the index of [ij|kl] in an eightfold-compressed ERI tensor.
 * @param[in] i The zero-based index of the first orbital
 * @param[in] j The zero-based index of the second orbital
 * @param[in] k The zero-based index of the third orbital
 * @param[in] l The zero-based index of the fourth orbital
 * @return the index of [ij|kl] in an eightfold-compressed ERI tensor
 */
size_t index_8d(size_t i, size_t j, size_t k, size_t l) {
    return index_2d(index_2d(i, j), index_2d(k, l));
}

/**
 * Computes the maximum magnitude of all excitations associated with the same four changing orbitals.
 * @param[in] doubles A pointer to the array of double excitations of length ndoubles
 * @param[out] magnitudes A pointer to the output array of magnitudes of length ndoubles
 * @param[in] ndoubles The number of entries to be processed
 */
static void get_max_mag_doubles(const DoubleExcEntry *doubles, double *magnitudes, size_t ndoubles) {
    for (size_t i=0; i<ndoubles; i++) {
        DoubleExcEntry entry = doubles[i];
        double ijkl = entry.ijkl;
        double iljk = entry.iljk;
        magnitudes[i] = MAX3(fabs(ijkl), fabs(iljk), fabs(ijkl+iljk));
    }
}

static void get_max_mag_mixed(const MixedExcEntry *mixed, double *magnitudes, size_t nmixed) {
    for (size_t i=0; i<nmixed; i++) {
        magnitudes[i] = fabs(mixed[i].ijkl);
    }
}

/**
 * Computes and stores the minimal amount of data needed to generate
 * all possible double excitation matrix elements from a given ERI tensor.
 * @param[out] doubles A pointer to an array of DoubleExcitationEntry of length nCr(norb, 4)
 * @param[in] eri_s8 A pointer to the eightfold-compressed ERI tensor (of aaaa or bbbb type)
 * @param[in] exc_table_4o A pointer to the four-orbital excitation table
 * @param[in] norb The number of orbitals
 */
static void load_doubles_from_eri(DoubleExcEntry *doubles, const double *eri_mo_xxxx_s8, const ConfigInfo *config_info) {
    size_t norb = config_info->norb;
    for (size_t i=0; i<norb-3; i++) {
        for (size_t j=i+1; j<norb-2; j++) {
            for (size_t k=j+1; k<norb-1; k++) {
                for (size_t l=k+1; l<norb; l++) {
                    size_t occ_list[4] = {i, j, k, l};
                    size_t entry_rank = rank_double_exc(occ_list, config_info);
                    double ijkl = eri_mo_xxxx_s8[index_8d(i, j, k, l)]-eri_mo_xxxx_s8[index_8d(i, l, k, j)];
                    double iljk = eri_mo_xxxx_s8[index_8d(i, l, j, k)]-eri_mo_xxxx_s8[index_8d(i, k, j, l)];
                    DoubleExcEntry entry = {entry_rank, ijkl, iljk};
                    doubles[entry_rank] = entry;
                }
            }
        }
    }
}

/**
 * Computes and stores mixed excitations.
 * @param[out] mixed A pointer to an array of MixedExcitationEntry of length nCr(norb, 2)**2
 * @param[in] eri_s4 A pointer to the fourfold-compressed ERI tensor (of aabb type)
 * @param[in] exc_table_2o A pointer to the two-orbital excitation table
 * @param[in] norb The number of orbitals
 */
static void load_mixed_from_eri(MixedExcEntry *mixed, double *eri_mo_aabb_s4, const ConfigInfo *config_info, size_t ncols_aabb) {
    size_t norb = config_info->norb;
    for (size_t i=0; i<norb-1; i++) {
        for (size_t j=i+1; j<norb; j++) {
            double *row = eri_mo_aabb_s4+(index_2d(i, j)*ncols_aabb);
            for (size_t k=0; k<norb-1; k++) {
                for (size_t l=k+1; l<norb; l++) {
                    size_t col = index_2d(k, l);
                    size_t occ_list[4] = {i, j, k, l};
                    size_t entry_rank = rank_mixed_exc(occ_list, config_info);
                    MixedExcEntry entry = {entry_rank, row[col]};
                    mixed[entry_rank] = entry;
                }
            }
        }
    }
}

void load_exc_entries_from_eri(ExcEntries *exc_entries, ERITensor *eri_mo, const ConfigInfo *config_info) {
    load_doubles_from_eri(exc_entries->doubles_aa, eri_mo->eri_mo_aaaa_s8, config_info);
    get_max_mag_doubles(exc_entries->doubles_aa, exc_entries->max_mag_aa, exc_entries->ndoubles_aa);
    load_doubles_from_eri(exc_entries->doubles_bb, eri_mo->eri_mo_bbbb_s8, config_info);
    get_max_mag_doubles(exc_entries->doubles_bb, exc_entries->max_mag_bb, exc_entries->ndoubles_bb);
    load_mixed_from_eri(exc_entries->mixed_ab, eri_mo->eri_mo_aabb_s4, config_info, eri_mo->ncols_aabb);
    get_max_mag_mixed(exc_entries->mixed_ab, exc_entries->max_mag_ab, exc_entries->nmixed_ab);
}