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
    size_t norb;
    size_t nelec_a;
    size_t nelec_b;
    uint64_t ncols_mixed;
} ConfigInfo;

void load_rank_table(uint64_t *table, size_t norb, size_t nocc);

uint64_t rank_occ_a(const size_t *occ_list, const ConfigInfo *config_info);
void unrank_occ_a(uint64_t arank, size_t *occ_list, const ConfigInfo *config_info);
void unrank_virt_a(uint64_t arank, size_t *virt_list, const ConfigInfo *config_info);

uint64_t rank_occ_b(const size_t *occ_list, const ConfigInfo *config_info);
void unrank_occ_b(uint64_t brank, size_t *occ_list, const ConfigInfo *config_info);
void unrank_virt_b(uint64_t brank, size_t *virt_list, const ConfigInfo *config_info);

uint64_t rank_double_exc(size_t *exc_list, const ConfigInfo *config_info);
void unrank_double_exc(uint64_t exc_rank, size_t *exc_list, const ConfigInfo *config_info);

uint64_t rank_mixed_exc(size_t *exc_list, const ConfigInfo *config_info);
void unrank_mixed_exc(uint64_t exc_rank_ab, size_t *exc_list, const ConfigInfo *config_info);

#endif