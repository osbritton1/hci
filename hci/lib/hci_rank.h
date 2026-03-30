#ifndef HCI_RANK_H
#define HCI_RANK_H

#include <stdlib.h>
#include <stdint.h>

typedef struct {
    uint64_t *config_table_a;
    uint64_t *config_table_a_complement;
    uint64_t combmax_a;
    uint64_t *config_table_b;
    uint64_t *config_table_b_complement;
    uint64_t combmax_b;
    uint64_t *exc_table_4o;
    uint64_t *exc_table_2o;
    uint64_t norb;
    uint64_t nelec_a;
    uint64_t nelec_b;
    uint64_t mixed_ncols;
} ConfigInfo;

void get_rank_table(uint64_t *table, size_t norb, size_t nocc);
uint64_t rank(const size_t *occ_list, const uint64_t *rank_table, size_t norb, size_t nocc);
size_t find_row_index(uint64_t target, const uint64_t *row, size_t norb, size_t nocc);
void unrank(uint64_t rank, size_t *occ_list, const uint64_t *rank_table, size_t norb, size_t nocc);
uint64_t nC2(size_t n);
uint64_t rank_mixed(size_t *occ_list, const uint64_t *exc_table_2o, size_t norb);
void unrank_mixed(uint64_t rank, size_t *occ_list, const uint64_t *exc_table_2o, size_t norb);

#endif