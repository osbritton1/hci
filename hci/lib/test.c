#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <time.h>
#include "hci_rank.h"
#include "hci_store.h"
#include "hci_enlarge.h"
#include "hci_contract.h"

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

//bool get_changing_orbitals(size_t *exc, size_t *occ, 
    //size_t *exc_min_occ, size_t *exc_int_occ, size_t *new_occ,
    //size_t exact_diffs, size_t nocc)

int test_get_changing_orbitals() {
    size_t exc[8] = {0,5,6,10,12,14,17,21};
    size_t occ[11] = {1,2,6,7,10,11,13,15,17,21,28};
    size_t exc_min_occ[4];
    size_t exc_int_occ[4];
    size_t new_occ[11];
    size_t exact_diffs = 4;
    size_t nocc = 11;
    bool condition = get_changing_orbitals(exc, occ, exc_min_occ, exc_int_occ, new_occ, exact_diffs, nocc);
    printf("%u\n", condition);
    size_t i;
    for (i=0; i<exact_diffs; i++) {
        printf("%zu ", exc_min_occ[i]);
    }
    printf("\n");
    for (i=0; i<exact_diffs; i++) {
        printf("%zu ", exc_int_occ[i]);
    }
    printf("\n");
    for (i=0; i<nocc; i++) {
        printf("%zu ", new_occ[i]);
    }
    printf("\n");
    return 0;
}

int main() {
    return test_get_changing_orbitals();
}

