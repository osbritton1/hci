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

uint64_t rank_occ_a(const size_t *occ_list, const ConfigInfo *config_info);
void unrank_occ_a(uint64_t arank, size_t *occ_list, const ConfigInfo *config_info);
void unrank_virt_a(uint64_t arank, size_t *virt_list, const ConfigInfo *config_info);

uint64_t rank_occ_b(const size_t *occ_list, const ConfigInfo *config_info);
void unrank_occ_b(uint64_t brank, size_t *occ_list, const ConfigInfo *config_info);
void unrank_virt_b(uint64_t brank, size_t *virt_list, const ConfigInfo *config_info);

void unrank_exc_aa(uint64_t exc_rank_aa, size_t *exc_list, const ConfigInfo *config_info);
void unrank_exc_bb(uint64_t exc_rank_bb, size_t *exc_list, const ConfigInfo *config_info);
void unrank_exc_ab(uint64_t exc_rank_ab, size_t *exc_list, const ConfigInfo *config_info);

#endif