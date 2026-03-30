#ifndef HCI_ENLARGE_H
#define HCI_ENLARGE_H

#include <stdint.h>
#include <stdbool.h>
#include "hci_rank.h"
#include "hci_store.h"

typedef struct {
    size_t *old_orbs;
    size_t *new_orbs;
    double sign;
} ExcResult;

#define NEW_DOUBLE_EXC_RESULT() \
    {(size_t[2]){}, (size_t[2]){}, 1.0}

#define NEW_SINGLE_EXC_RESULT() \
    {(size_t[1]){}, (size_t[1]){}, 1.0}

#define SINGLE_EXC_RESULT_NOSIGN(old_orb, new_orb) \
    {(size_t[1]){old_orb}, (size_t[1]){new_orb}, 1.0}

#define SORTED(occ_orb, virt_orb) \
    (occ_orb) < (virt_orb) ? (size_t[2]){(occ_orb), (virt_orb)} : (size_t[2]){(virt_orb), (occ_orb)}

bool get_changing_orbitals_new(const size_t *exc_list, size_t exc_order, const size_t *occ_list, size_t nocc,
    ExcResult *res, size_t *new_occ_list);

size_t add_doubles_aa(const size_t *occ_a, size_t brank, const ExcitationEntries *exc_entries, double entry_thresh, 
     Rank *add_list, const ConfigInfo *config_info);
size_t add_doubles_bb(const size_t *occ_b, size_t arank, const ExcitationEntries *exc_entries, double entry_thresh, 
     Rank *add_list, const ConfigInfo *config_info);
size_t add_mixed_ab(const size_t *occ_a, const size_t *occ_b, const ExcitationEntries *exc_entries, double entry_thresh, 
     Rank *add_list, const ConfigInfo *config_info);
size_t enlarge_space_doubles_new(const HCIVector *hcivec, Rank *add_list, double thresh, 
    const ConfigInfo *config_info, const ExcitationEntries *exc_entries);

size_t add_singles_a(const size_t *occ_a, const size_t *virt_a, const size_t *occ_b, size_t brank, 
    const H1E *h1e, const ERI_MO *eri_mo, double entry_thresh, Rank *add_list, const ConfigInfo *config_info);
size_t add_singles_b(const size_t *occ_b, const size_t *virt_b, const size_t *occ_a, size_t arank, 
    const H1E *h1e, const ERI_MO *eri_mo, double entry_thresh, Rank *add_list, const ConfigInfo *config_info);
size_t enlarge_space_singles_new(const HCIVector *hcivec, Rank *add_list, double thresh,
    const ConfigInfo *config_info, const H1E *h1e, const ERI_MO *eri_mo);

// Old spaghetti
// Old spaghetti
// Old spaghetti

bool get_changing_orbitals(size_t *exc, size_t *occ, 
    size_t *exc_min_occ, size_t *exc_int_occ, size_t *new_occ,
    size_t *old_indices, size_t *new_indices, size_t exact_diffs, size_t nocc);
size_t enlarge_space_doubles(uint64_t *ranks, double *coeffs, size_t hci_len, uint64_t *add_doubles,
    size_t norb, size_t nelec_a, size_t nelec_b, double thresh,
    uint64_t *config_table_a, uint64_t *config_table_b, uint64_t *exc_table_4o, uint64_t *exc_table_2o,
    DoubleExcitationEntry *doubles_aa, size_t ndoubles_aa, 
    DoubleExcitationEntry *doubles_bb, size_t ndoubles_bb,
    MixedExcitationEntry *mixed_ab, size_t ndoubles_ab,
    double *max_mag_aa, double *max_mag_bb, double *max_mag_ab);
size_t enlarge_space_singles(uint64_t *ranks, double *coeffs, size_t hci_len, uint64_t *add_singles,
    size_t norb, size_t nelec_a, size_t nelec_b, uint64_t combmax_a, uint64_t combmax_b, double thresh,
    uint64_t *config_table_a, uint64_t *config_table_a_complement,
    uint64_t *config_table_b, uint64_t *config_table_b_complement,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4);

#endif