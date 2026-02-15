#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include "hci_rank.h"

int main() {
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