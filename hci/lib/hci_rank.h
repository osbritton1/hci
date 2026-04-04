/**
 * \file hci_rank.h
 * \addtogroup rank
 * @{
 */

#ifndef HCI_RANK_H
#define HCI_RANK_H

#include <stdlib.h>
#include <stdint.h>

/**
 * A struct that holds pointers to all of the ranking tables
 * needed for carrying out HCI calculations, along with their
 * associated parameters.
 */
typedef struct {
    size_t norb; /**< Total size of input orbital space \f$N_\text{orb}\f$ */
    size_t nelec_a; /**< Number of \f$\alpha\f$ electrons \f$N_\alpha\f$ */
    size_t nelec_b; /**< Number of \f$\beta\f$ electrons \f$N_\beta\f$ */
    uint64_t *occ_table_a; /**< Pointer to the ranking table for determining occupied \f$\alpha\f$ orbitals */
    uint64_t *virt_table_a; /**< Pointer to the ranking table for determining virtual \f$\alpha\f$ orbitals */
    uint64_t combmax_a; /**< Total number of \f$\alpha\f$ occupancy lists, \f$\binom{N_\text{orb}}{N_\alpha}\f$ */
    uint64_t *occ_table_b; /**< Pointer to the ranking table for determining occupied \f$\beta\f$ orbitals */
    uint64_t *virt_table_b; /**< Pointer to the ranking table for determining virtual \f$\beta\f$ orbitals */
    uint64_t combmax_b; /**< Total number of \f$\beta\f$ occupancy lists, \f$\binom{N_\text{orb}}{N_\beta}\f$ */
    /** 
     * Pointer to the ranking table for organizing double excitations of \f$\alpha\alpha\rightarrow\alpha\alpha\f$ or 
     * \f$\beta\beta\rightarrow\beta\beta\f$ type
    */
    uint64_t *exc_table_4o; 
    /** 
     * Pointer to the ranking table for organizing mixed excitations of \f$\alpha\beta\rightarrow\alpha\beta\f$ type
    */
    uint64_t *exc_table_2o;
    uint64_t ncols_mixed; /**< \f$\binom{N_\text{orb}}{2}\f$, used for ranking and unranking mixed excitations */
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

/**
 * @}
 */