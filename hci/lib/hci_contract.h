#ifndef HCI_CONTRACT_H
#define HCI_CONTRACT_H

#include <stdint.h>
#include <stdbool.h>
#include "hci_store.h"

typedef struct {
    uint64_t ranka;
    uint64_t rankb;
    double coeff;
} HCIEntry;

typedef enum {
    ZERO,
    SINGLE,
    DOUBLE,
    THREE_PLUS
} DiffType;

double get_single_excitation_value_a(size_t occ_orb, size_t virt_orb, size_t norb, size_t nelec_a, size_t nelec_b, size_t *occ_a, size_t *occ_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);
double get_single_excitation_mag_a(size_t occ_orb, size_t virt_orb, size_t norb, size_t nelec_a, size_t nelec_b, size_t *occ_a, size_t *occ_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);

double get_single_excitation_value_b(size_t occ_orb, size_t virt_orb, size_t norb, size_t nelec_a, size_t nelec_b, size_t *occ_a, size_t *occ_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);
double get_single_excitation_mag_b(size_t occ_orb, size_t virt_orb, size_t norb, size_t nelec_a, size_t nelec_b, size_t *occ_a, size_t *occ_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);

double get_double_excitation_value_from_store(DoubleExcitationEntry exc_entry, size_t *exc_min_occ, size_t *exc_int_occ, size_t *old_indices, size_t *new_indices);
double get_double_excitation_mag_from_store(DoubleExcitationEntry exc_entry, size_t *exc_min_occ, size_t *exc_int_occ, size_t *old_indices, size_t *new_indices);

double get_mixed_excitation_value_from_store(MixedExcitationEntry exc_entry, size_t old_a_index, size_t new_a_index, size_t old_b_index, size_t new_b_index);

double get_diag_value(size_t *occ_a, size_t *occ_b, size_t norb, size_t nelec_a, size_t nelec_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);

DiffType get_diff_type(size_t *occ_1, size_t *occ_2, size_t *one_min_two, size_t *one_min_two_indices, size_t *two_min_one,size_t *two_min_one_indices, size_t nocc);

double get_matrix_element_by_rank(uint64_t ranka_1, uint64_t rankb_1, uint64_t ranka_2, uint64_t rankb_2, 
    uint64_t *config_table_a, uint64_t *config_table_b, uint64_t *exc_table_4o, uint64_t *exc_table_2o,
    size_t norb, size_t nelec_a, size_t nelec_b,
    DoubleExcitationEntry *ordered_doubles_aa, DoubleExcitationEntry *ordered_doubles_bb, MixedExcitationEntry *ordered_mixed_ab,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);

#endif