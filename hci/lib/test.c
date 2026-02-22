#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <time.h>
#include "hci_rank.h"
#include "hci_store.h"

int test_rank_unrank() {
    size_t i, j;
    size_t norb = 8;
    size_t nocc = 4;
    size_t nrows = nocc;
    size_t ncols = norb-nocc+1;
    uint64_t rank_table[nrows*ncols];
    uint64_t rank = 57;
    size_t occ_list[nocc];

    get_rank_table(rank_table, norb, nocc);

    for (i=0; i<nrows; i++) {
        uint64_t *row = rank_table+i*ncols;
        for (j=0; j<ncols; j++) {
            printf("%"PRIu64" ", row[j]);
        }
        printf("\n");
    }

    unrank(rank, occ_list, rank_table, norb, nocc);

    for (i=0; i<nocc; i++) {
        printf("%"PRIu64" ", occ_list[i]);
    }

    return 0;
}

int test_mixed_storage() {
    size_t norb = 4;
    size_t nocc = 2;
    size_t npairs = nC2(norb+1);
    size_t ncombs = nC2(norb);
    double eri_s4[npairs*npairs];
    size_t i;
    uint64_t exc_table_2o[2*(norb-2+1)];
    MixedExcitationEntry mixed[ncombs*ncombs];

    get_rank_table(exc_table_2o, norb, 2);
    srand(time(NULL));
    for (i=0; i<npairs*npairs; i++) {
        eri_s4[i] = (double)rand() / (double)RAND_MAX;
    }

    load_mixed_from_eri(mixed, eri_s4, exc_table_2o, norb);
    return 0;
}

int main() {
    return test_mixed_storage();
}

