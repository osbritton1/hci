#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stdio.h>
#include <inttypes.h>
#include "hci_rank.h"

// Command to run to generate this library: gcc -shared -o libhci.so -fPIC hci_rank.c hci_store.c hci_enlarge.c hci_contract.c -Wall -g

/**
 * Initializes the nocc by (norb-nocc+1) table needed to rank and unrank combinations using the combinatorial number system
 * The binomial coefficients needed for the encoding the position of the ith electron are stored in row i-1
 * The entry in position (i, j) is nCr(i+j, i+1); when i+j < i+1 (first column), this is defined to be 0
 * @param[in] table Pointer to the uninitialized ranking table; must be able to accommodate at least nocc rows by norb-nocc+1 columns
 * @param[in] norb Number of orbitals/positions
 * @param[in] nocc Number of occupancies; generally equal to the number of alpha or beta electrons in the system
 */
void get_rank_table(uint64_t *table, size_t norb, size_t nocc) {
    size_t ncols, i, j;
    // Edge case: no electrons
    if (nocc == 0) {
        return;
    }
    ncols = norb-nocc+1;
    // Initialize first row of ranking table
    for (j=0; j<ncols; j++) {
        table[j] = j;
    }
    // Initialize rest of ranking table using binomial coefficient recurrence relation
    for (i=1; i<nocc; i++) {
        uint64_t *prev_row = table+(i-1)*ncols;
        uint64_t *curr_row = table+i*ncols;
        curr_row[0] = 0;
        for (j=1; j<ncols; j++) {
            curr_row[j] = curr_row[j-1]+prev_row[j];
        }
    }
}

/**
 * Get the combinatorial rank corresponding to the provided list of occupied orbitals
 * Formula: \f$\sum\limits_{i=1}^{N_\text{occ}}\binom{c_i}{i}$\f
 * @param[in] occ_list Pointer to an array of length nocc specifying the zero-indexed occupied orbitals in increasing order
 * @param[in] rank_table Pointer to ranking table initialized by get_rank_table()
 * @param[in] norb Number of orbitals/positions
 * @param[in] nocc Number of occupancies; generally equal to the number of alpha or beta electrons in the system
 */
uint64_t rank(size_t *occ_list, uint64_t *rank_table, size_t norb, size_t nocc) {
    uint64_t sum = 0;
    size_t i;
    size_t occ_orbital;
    size_t ncols = norb-nocc+1;
    for (i=0; i<nocc; i++) {
        occ_orbital = occ_list[i];
        sum += rank_table[i*ncols+(occ_orbital-i)];
    }
    return sum;
}

size_t find_row_index(uint64_t rank, uint64_t *row, size_t norb, size_t nocc) {
    size_t low = 0;
    size_t high = norb-nocc;
    while (low < high) {
        size_t mid = low+(high-low+1)/2;
        if (row[mid]>rank) {
            high = mid-1;
        } else {
            low = mid;
        }
    }
    return low;
}

void unrank(uint64_t rank, size_t *occ_list, uint64_t *rank_table, size_t norb, size_t nocc) {
    size_t i;
    size_t ncols = norb-nocc+1;
    // Reverse iteration with unsigned types is a bit funky
    for (i=nocc; i-->0;) {
        uint64_t *row = rank_table+i*ncols;
        size_t row_index = find_row_index(rank, row, norb, nocc);
        rank -= row[row_index];
        occ_list[i] = row_index+i;
    }
}

uint64_t nC2(size_t n) {
    return (n % 2 == 0) ? n/2*(n-1) : (n-1)/2*n;
}

uint64_t rank_mixed(size_t *occ_list, uint64_t *rank_table, size_t norb) {
    size_t ij[2] = {occ_list[0], occ_list[1]};
    size_t kl[2] = {occ_list[2], occ_list[3]};
    uint64_t ij_rank = rank(ij, rank_table, norb, 2);
    uint64_t kl_rank = rank(kl, rank_table, norb, 2);
    uint64_t ncols = nC2(norb);
    return ncols*ij_rank + kl_rank;
}

void unrank_mixed(uint64_t rank, size_t *occ_list, uint64_t *rank_table, size_t norb) {
    uint64_t ncols = nC2(norb);
    uint64_t ij_rank = rank/ncols;
    uint64_t kl_rank = rank%ncols;
    unrank(ij_rank, occ_list, rank_table, norb, 2);
    unrank(kl_rank, occ_list+2, rank_table, norb, 2);
}