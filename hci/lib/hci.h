#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#define MAX_PRECOMP 64

bool get_consecutive_combs(uint64_t *comb_list, int norb, int nocc);
void get_rank_table(uint64_t *table, int norb, int nocc);
uint64_t rank(int *occ_list, uint64_t *rank_table, int norb, int nocc);
int find_row_index(uint64_t rank, uint64_t *row, int norb, int nocc);
void unrank(uint64_t rank, int *occ_list, uint64_t *rank_table, int norb, int nocc);