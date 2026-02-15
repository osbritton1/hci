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

static size_t index_2d(size_t i, size_t j) {
    return ((i < j) ? i*(i+1)/2+j : j*(j+1)/2+i);
}

static size_t index_4d(size_t i, size_t j, size_t k, size_t l) {
    return index_2d(index_2d(i, j), index_2d(k, l));
}

void load_from_eri(DoubleExcitationEntry *doubles, double *eri_s8, uint64_t *index_table, size_t norb) {
    size_t i, j, k, l;
    for (i=0; i<norb-3; i++) {
        for (j=i+1; j<norb-2; j++) {
            for (k=j+1; k<norb-1; k++) {
                for (l=k+1; l<norb; l++) {
                    size_t occ_list[4] = {i, j, k, l};
                    size_t entry_rank = rank(occ_list, index_table, norb, 4);
                    double ijkl = eri_s8[index_4d(i, j, k, l)];
                    double iljk = eri_s8[index_4d(i, l, j, k)];
                    DoubleExcitationEntry entry = {entry_rank, ijkl, iljk};
                    doubles[entry_rank] = entry;
                }
            }
        }
    }
}