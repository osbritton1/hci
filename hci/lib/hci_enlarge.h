#ifndef HCI_ENLARGE_H
#define HCI_ENLARGE_H

#include <stdint.h>
#include <stdbool.h>
#include "hci_store.h"
#include "hci_contract.h"

bool get_changing_orbitals(size_t *exc, size_t *occ, 
    size_t *exc_min_occ, size_t *exc_int_occ, size_t *new_occ,
    size_t *exc_int_occ_indices, size_t *new_indices, size_t exact_diffs, size_t nocc);
size_t enlarge_space_doubles(HCIEntry *hcivec, size_t hci_len, uint64_t *add_doubles,
    size_t norb, size_t nelec_a, size_t nelec_b, double thresh,
    uint64_t *config_table_a, uint64_t *config_table_b, uint64_t *exc_table_4o, uint64_t *exc_table_2o,
    DoubleExcitationEntry *doubles_aa, size_t ndoubles_aa, 
    DoubleExcitationEntry *doubles_bb, size_t ndoubles_bb,
    MixedExcitationEntry *mixed_ab, size_t ndoubles_ab,
    double *max_mag_aa, double *max_mag_bb, double *max_mag_ab);

#endif