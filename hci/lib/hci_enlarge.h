/**
 * \file hci_enlarge.h
 * \addtogroup enlarge
 * @{
 */

#ifndef HCI_ENLARGE_H
#define HCI_ENLARGE_H

#include "hci_store.h"
#include <stdbool.h>

typedef struct {
    size_t *old_orbs;
    size_t *new_orbs;
    double sign;
} ExcResult;

typedef struct {
    uint64_t arank;
    uint64_t brank;
} Rank;

typedef struct {
    Rank *ranks;
    double *coeffs;
    size_t len;
} HCIVec;

#define NEW_DOUBLE_EXC_RESULT() \
    {(size_t[2]){}, (size_t[2]){}, 1.0}

#define NEW_SINGLE_EXC_RESULT() \
    {(size_t[1]){}, (size_t[1]){}, 1.0}

#define SINGLE_EXC_RESULT_NOSIGN(old_orb, new_orb) \
    {(size_t[1]){old_orb}, (size_t[1]){new_orb}, 1.0}

#define SORTED(occ_orb, virt_orb) \
    (occ_orb) < (virt_orb) ? (size_t[2]){(occ_orb), (virt_orb)} : (size_t[2]){(virt_orb), (occ_orb)}

bool get_changing_orbitals(const size_t *exc_list, size_t exc_order, const size_t *occ_list, size_t nocc,
    ExcResult *res, size_t *new_occ_list);

size_t add_doubles_aa(const size_t *occ_a, size_t brank, const ExcEntries *exc_entries, double entry_thresh, 
     Rank *add_list, const ConfigInfo *config_info);
size_t add_doubles_bb(const size_t *occ_b, size_t arank, const ExcEntries *exc_entries, double entry_thresh, 
     Rank *add_list, const ConfigInfo *config_info);
size_t add_mixed_ab(const size_t *occ_a, const size_t *occ_b, const ExcEntries *exc_entries, double entry_thresh, 
     Rank *add_list, const ConfigInfo *config_info);
size_t enlarge_space_doubles(const HCIVec *hcivec, Rank *add_list, double thresh, 
    const ConfigInfo *config_info, const ExcEntries *exc_entries);

size_t add_singles_a(const size_t *occ_a, const size_t *virt_a, const size_t *occ_b, size_t brank, 
    const HCore *h1e, const ERITensor *eri_mo, double entry_thresh, Rank *add_list, const ConfigInfo *config_info);
size_t add_singles_b(const size_t *occ_b, const size_t *virt_b, const size_t *occ_a, size_t arank, 
    const HCore *h1e, const ERITensor *eri_mo, double entry_thresh, Rank *add_list, const ConfigInfo *config_info);
size_t enlarge_space_singles(const HCIVec *hcivec, Rank *add_list, double thresh,
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo);

#endif

/**
 * @}
 */