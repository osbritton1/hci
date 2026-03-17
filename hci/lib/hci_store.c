#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include "hci_store.h"
#include "hci_rank.h"

void get_max_magnitudes(DoubleExcitationEntry *doubles, double *magnitudes, size_t ndoubles) {
    size_t i;
    for (i=0; i<ndoubles; i++) {
        DoubleExcitationEntry entry = doubles[i];
        double ijkl = entry.ijkl;
        double iljk = entry.iljk;
        magnitudes[i] = MAX3(fabs(ijkl), fabs(iljk), fabs(ijkl+iljk));
    }
}

size_t index_2d(size_t i, size_t j) {
    return (i > j) ? nC2(i+1)+j : nC2(j+1)+i;
}

size_t index_4d(size_t i, size_t j, size_t k, size_t l) {
    return index_2d(index_2d(i, j), index_2d(k, l));
}

void load_doubles_from_eri(DoubleExcitationEntry *doubles, double *eri_s8, uint64_t *index_table, size_t norb) {
    size_t i, j, k, l;
    for (i=0; i<norb-3; i++) {
        for (j=i+1; j<norb-2; j++) {
            for (k=j+1; k<norb-1; k++) {
                for (l=k+1; l<norb; l++) {
                    size_t occ_list[4] = {i, j, k, l};
                    size_t entry_rank = rank(occ_list, index_table, norb, 4);
                    double ijkl = eri_s8[index_4d(i, j, k, l)]-eri_s8[index_4d(i, l, k, j)];
                    double iljk = eri_s8[index_4d(i, l, j, k)]-eri_s8[index_4d(i, k, j, l)];
                    DoubleExcitationEntry entry = {entry_rank, ijkl, iljk};
                    doubles[entry_rank] = entry;
                }
            }
        }
    }
}

void load_mixed_from_eri(MixedExcitationEntry *mixed, double *eri_s4, uint64_t *index_table, size_t norb) {
    size_t i, j, k, l;
    size_t ncols = nC2(norb+1);
    for (i=0; i<norb-1; i++) {
        for (j=i+1; j<norb; j++) {
            double *row = eri_s4+index_2d(i, j)*ncols;
            for (k=0; k<norb-1; k++) {
                for (l=k+1; l<norb; l++) {
                    size_t col = index_2d(k, l);
                    size_t occ_list[4] = {i, j, k, l};
                    size_t entry_rank = rank_mixed(occ_list, index_table, norb);
                    MixedExcitationEntry entry = {entry_rank, row[col]};
                    mixed[entry_rank] = entry;
                }
            }
        }
    }
}