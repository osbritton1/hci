/**
 * \file hci_contract.h
 * \addtogroup contract
 * @{
 */

#ifndef HCI_CONTRACT_H
#define HCI_CONTRACT_H

#include "hci_enlarge.h"

double get_diag_value(const size_t *occ_a, const size_t *occ_b, const ConfigInfo *config_info,
    const HCore *h1e, const ERITensor *eri_mo);

double get_single_exc_value_a(const ExcResult *single_exc, const size_t *occ_a, const size_t *occ_b,
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo);
double get_single_exc_value_b(const ExcResult *single_exc, const size_t *occ_a, const size_t *occ_b,
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo);

double get_double_exc_value_aa(const ExcResult *double_exc, const ERITensor *eri_mo);
double get_double_exc_value_bb(const ExcResult *double_exc, const ERITensor *eri_mo);
double get_double_exc_value_from_store(const DoubleExcEntry *exc_entry, const ExcResult *double_exc);

double get_mixed_exc_value(const ExcResult *single_exc_a, const ExcResult *single_exc_b, 
    const ERITensor *eri_mo);
double get_mixed_exc_value_from_store(const MixedExcEntry *exc_entry, 
    const ExcResult *single_exc_a, const ExcResult *single_exc_b);

double get_matrix_element_by_rank(Rank rank1, Rank rank2, 
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo);
double get_matrix_element_by_partial_rank(uint64_t *occ_a_1, uint64_t *occ_b_1, Rank rank2,
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo);
double get_matrix_element_by_rank_test_storage(Rank rank1, Rank rank2, 
    const ConfigInfo *config_info, const ExcEntries *excitation_entries,
    const HCore *h1e, const ERITensor *eri_mo);

void make_hdiag_slow(const HCIVec *hcivec, double *hdiag,
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo);

void contract_hamiltonian_hcivec_slow(HCIVec *hcivec_old, double *coeffs_new, const double *hdiag,
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo);

#endif

/**
 * @}
 */