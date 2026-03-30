#ifndef HCI_CONTRACT_H
#define HCI_CONTRACT_H

#include <stdint.h>
#include <stdbool.h>
#include "hci_enlarge.h"
#include "hci_store.h"

typedef enum {
    ZERO,
    SINGLE,
    DOUBLE,
    THREE_PLUS
} DiffType;

double get_diag_value_new(const size_t *occ_a, const size_t *occ_b, const ConfigInfo *config_info,
    const H1E *h1e, const ERI_MO *eri_mo);

double get_single_exc_value_a_new(const ExcResult *single_exc, const size_t *occ_a, const size_t *occ_b,
    const ConfigInfo *config_info, const H1E *h1e, const ERI_MO *eri_mo);
double get_single_exc_value_b_new(const ExcResult *single_exc, const size_t *occ_a, const size_t *occ_b,
    const ConfigInfo *config_info, const H1E *h1e, const ERI_MO *eri_mo);

double get_double_exc_value_aa_new(const ExcResult *double_exc, const ERI_MO *eri_mo);
double get_double_exc_value_bb_new(const ExcResult *double_exc, const ERI_MO *eri_mo);
double get_double_exc_value_from_store_new(const DoubleExcitationEntry *exc_entry, const ExcResult *double_exc);

double get_mixed_exc_value_new(const ExcResult *single_exc_a, const ExcResult *single_exc_b, 
    const ConfigInfo *config_info, const ERI_MO *eri_mo);
double get_mixed_exc_value_from_store(const MixedExcitationEntry *exc_entry, 
    const ExcResult *single_exc_a, const ExcResult *single_exc_b);

DiffType get_diff_type_new(size_t *occ_1, size_t *occ_2, ExcResult *exc_result);

double get_matrix_element_by_rank_new(Rank rank1, Rank rank2, 
    const ConfigInfo *config_info, const H1E *h1e, const ERI_MO *eri_mo);

double get_matrix_element_by_partial_rank_new(uint64_t *occ_a_1, uint64_t *occ_b_1, Rank rank2,
    const ConfigInfo *config_info, const H1E *h1e, const ERI_MO *eri_mo);

double get_matrix_element_by_rank_test_storage_new(Rank rank1, Rank rank2, 
    const ConfigInfo *config_info, const ExcitationEntries *excitation_entries,
    const H1E *h1e, const ERI_MO *eri_mo);

void make_hdiag_slow_new(HCIVector *hcivec, double *hdiag,
    const ConfigInfo *config_info, const H1E *h1e, const ERI_MO *eri_mo);

void contract_hamiltonian_hcivec_slow_new(HCIVector *hcivec_old, double *coeffs_new, double *hdiag,
    const ConfigInfo *config_info, const H1E *h1e, const ERI_MO *eri_mo);

// Old spaghetti

double get_single_excitation_value_a(size_t occ_orb, size_t virt_orb, size_t norb, size_t nelec_a, size_t nelec_b, size_t *occ_a, size_t *occ_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);
double get_single_excitation_mag_a(size_t occ_orb, size_t virt_orb, size_t norb, size_t nelec_a, size_t nelec_b, size_t *occ_a, size_t *occ_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);

double get_single_excitation_value_b(size_t occ_orb, size_t virt_orb, size_t norb, size_t nelec_a, size_t nelec_b, size_t *occ_a, size_t *occ_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);
double get_single_excitation_mag_b(size_t occ_orb, size_t virt_orb, size_t norb, size_t nelec_a, size_t nelec_b, size_t *occ_a, size_t *occ_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);

double get_double_excitation_value_aa(size_t *one_min_two, size_t *two_min_one, size_t *one_min_two_indices, size_t *two_min_one_indices, double *eri_aaaa_s8);
double get_double_excitation_value_bb(size_t *one_min_two, size_t *two_min_one, size_t *one_min_two_indices, size_t *two_min_one_indices, double *eri_bbbb_s8);
double get_double_excitation_value_from_store(DoubleExcitationEntry exc_entry, size_t *exc_min_occ, size_t *exc_int_occ, size_t *old_indices, size_t *new_indices);
double get_double_excitation_mag_from_store(DoubleExcitationEntry exc_entry, size_t *exc_min_occ, size_t *exc_int_occ, size_t *old_indices, size_t *new_indices);

double get_mixed_excitation_value(size_t *occ_a_1_min_2, size_t *occ_a_2_min_1, size_t *occ_a_1_min_2_indices, size_t *occ_a_2_min_1_indices,
    size_t *occ_b_1_min_2, size_t *occ_b_2_min_1, size_t *occ_b_1_min_2_indices, size_t *occ_b_2_min_1_indices,
    size_t norb, double *eri_aabb_s4);
double get_mixed_excitation_value_from_store(MixedExcitationEntry exc_entry, size_t old_a_index, size_t new_a_index, size_t old_b_index, size_t new_b_index);

double get_diag_value(size_t *occ_a, size_t *occ_b, size_t norb, size_t nelec_a, size_t nelec_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);

DiffType get_diff_type(size_t *occ_1, size_t *occ_2, size_t *one_min_two, size_t *one_min_two_indices, size_t *two_min_one, size_t *two_min_one_indices, size_t nocc);

double get_matrix_element_by_rank(uint64_t ranka_1, uint64_t rankb_1, uint64_t ranka_2, uint64_t rankb_2,
    uint64_t *config_table_a, uint64_t *config_table_b, size_t norb, size_t nelec_a, size_t nelec_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);

double get_matrix_element_by_partial_rank(uint64_t *occ_a_1, uint64_t *occ_b_1, uint64_t ranka_2, uint64_t rankb_2,
    uint64_t *config_table_a, uint64_t *config_table_b, size_t norb, size_t nelec_a, size_t nelec_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);

double get_matrix_element_by_rank_test_storage(uint64_t ranka_1, uint64_t rankb_1, uint64_t ranka_2, uint64_t rankb_2, 
    uint64_t *config_table_a, uint64_t *config_table_b, uint64_t *exc_table_4o, uint64_t *exc_table_2o,
    size_t norb, size_t nelec_a, size_t nelec_b,
    DoubleExcitationEntry *ordered_doubles_aa, DoubleExcitationEntry *ordered_doubles_bb, MixedExcitationEntry *ordered_mixed_ab,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);

void make_hdiag_slow(uint64_t *ranks, double *hdiag, size_t hci_len,
    uint64_t *config_table_a, uint64_t *config_table_b, size_t norb, size_t nelec_a, size_t nelec_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);

void contract_hamiltonian_hcivec_slow(uint64_t *ranks, double *coeffs, double *coeffs_new, size_t hci_len, double *hdiag,
    uint64_t *config_table_a, uint64_t *config_table_b, size_t norb, size_t nelec_a, size_t nelec_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);

#endif