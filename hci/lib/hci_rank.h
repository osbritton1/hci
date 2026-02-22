#ifndef HCI_RANK_H
#define HCI_RANK_H

#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#define MAX_PRECOMP 64
      
bool get_consecutive_combs(uint64_t *comb_list, size_t norb, size_t nocc);
void get_rank_table(uint64_t *table, size_t norb, size_t nocc);
uint64_t rank(size_t *occ_list, uint64_t *rank_table, size_t norb, size_t nocc);
size_t find_row_index(uint64_t rank, uint64_t *row, size_t norb, size_t nocc);
void unrank(uint64_t rank, size_t *occ_list, uint64_t *rank_table, size_t norb, size_t nocc);
uint64_t nC2(size_t n);
uint64_t rank_mixed(size_t *occ_list, uint64_t *rank_table, size_t norb);
void unrank_mixed(uint64_t rank, size_t *occ_list, uint64_t *rank_table, size_t norb);

#endif