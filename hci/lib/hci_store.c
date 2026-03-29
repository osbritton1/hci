#include <math.h>
#include "hci_store.h"
#include "hci_rank.h"

/**
 * Computes the maximum magnitude of all excitations associated with the same four changing orbitals.
 * @param[in] doubles A pointer to the array of double excitations of length ndoubles
 * @param[out] magnitudes A pointer to the output array of magnitudes of length ndoubles
 * @param[in] ndoubles The number of entries to be processed
 */
void get_max_magnitudes(const DoubleExcitationEntry *doubles, double *magnitudes, size_t ndoubles) {
    for (size_t i=0; i<ndoubles; i++) {
        DoubleExcitationEntry entry = doubles[i];
        double ijkl = entry.ijkl;
        double iljk = entry.iljk;
        magnitudes[i] = MAX3(fabs(ijkl), fabs(iljk), fabs(ijkl+iljk));
    }
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
 * Computes and stores the minimal amount of data needed to generate
 * all possible double excitation matrix elements from a given ERI tensor.
 * @param[out] doubles A pointer to an array of DoubleExcitationEntry of length nCr(norb, 4)
 * @param[in] eri_s8 A pointer to the eightfold-compressed ERI tensor (of aaaa or bbbb type)
 * @param[in] exc_table_4o A pointer to the four-orbital excitation table
 * @param[in] norb The number of orbitals
 */
void load_doubles_from_eri(DoubleExcitationEntry *doubles, const double *eri_s8, const uint64_t *exc_table_4o, size_t norb) {
    for (size_t i=0; i<norb-3; i++) {
        for (size_t j=i+1; j<norb-2; j++) {
            for (size_t k=j+1; k<norb-1; k++) {
                for (size_t l=k+1; l<norb; l++) {
                    size_t occ_list[4] = {i, j, k, l};
                    size_t entry_rank = rank(occ_list, exc_table_4o, norb, 4);
                    double ijkl = eri_s8[index_8d(i, j, k, l)]-eri_s8[index_8d(i, l, k, j)];
                    double iljk = eri_s8[index_8d(i, l, j, k)]-eri_s8[index_8d(i, k, j, l)];
                    DoubleExcitationEntry entry = {entry_rank, ijkl, iljk};
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
void load_mixed_from_eri(MixedExcitationEntry *mixed, double *eri_s4, const uint64_t *exc_table_2o, size_t norb) {
    size_t ncols = nC2(norb+1);
    for (size_t i=0; i<norb-1; i++) {
        for (size_t j=i+1; j<norb; j++) {
            double *row = eri_s4+(index_2d(i, j)*ncols);
            for (size_t k=0; k<norb-1; k++) {
                for (size_t l=k+1; l<norb; l++) {
                    size_t col = index_2d(k, l);
                    size_t occ_list[4] = {i, j, k, l};
                    size_t entry_rank = rank_mixed(occ_list, exc_table_2o, norb);
                    MixedExcitationEntry entry = {entry_rank, row[col]};
                    mixed[entry_rank] = entry;
                }
            }
        }
    }
}