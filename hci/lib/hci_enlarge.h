/**
 * \file hci_enlarge.h
 * \addtogroup enlarge
 * @{
 */

#ifndef HCI_ENLARGE_H
#define HCI_ENLARGE_H

#include "hci_store.h"
#include <stdbool.h>

/**
 * Given a reference configuration and an excited configuration,
 * stores which orbitals are being substituted for new orbitals
 * and the sign of the resulting excitation
 */
typedef struct {
    size_t *old_orbs; /**< Pointer to list of orbitals present in the reference but not the excited determinant */
    size_t *new_orbs; /**< Pointer to list of orbitals present in the excited determinant but not the reference */
    double sign; /**< The sign of the excitation, computed from indices in the old and new orbital lists */
} ExcResult;

/**
 * Lightweight tuple-like struct for storing associated \f$\alpha\f$ and \f$\beta\f$ ranks.
 */
typedef struct {
    uint64_t arank; /**< The rank of the \f$\alpha\f$ occupancy list of the associated configuration */
    uint64_t brank; /**< The rank of the \f$\beta\f$ occupancy list of the associated configuration */
} Rank;

/**
 * Struct representing a weighted sum of configurations.
 *
 * Implemented as a structure of arrays to make interoperability
 * with \c NumPy and \c PySCF easier.
 */
typedef struct {
    Rank *ranks; /**< Pointer to list of ranks specifying the configurations */
    double *coeffs; /**< Pointer to list of coefficients of the corresponding ranks */
    size_t len; /**< Length of the vector */
} HCIVec;

/**
 * Macro to initialize an empty \ref ExcResult using compound literals capable of storing
 * double excitations (or single excitations).
 */
#define NEW_DOUBLE_EXC_RESULT() \
    {(size_t[2]){}, (size_t[2]){}, 1.0}

/**
 * Macro to initialize an empty \ref ExcResult using compound literals capable of storing
 * single excitations.
 */
#define NEW_SINGLE_EXC_RESULT() \
    {(size_t[1]){}, (size_t[1]){}, 1.0}

/**
 * Macro to initialize a single \ref ExcResult with specified old and new orbitals.
 */
#define SINGLE_EXC_RESULT_NOSIGN(old_orb, new_orb) \
    {(size_t[1]){old_orb}, (size_t[1]){new_orb}, 1.0}

/**
 * Macro to return a pointer to a sorted list of the two input orbitals.
 */
#define SORTED(occ_orb, virt_orb) \
    (occ_orb) < (virt_orb) ? (size_t[2]){(occ_orb), (virt_orb)} : (size_t[2]){(virt_orb), (occ_orb)}

size_t enlarge_space_doubles(const HCIVec *hcivec, Rank *add_list, double thresh, 
    const ConfigInfo *config_info, const ExcEntries *exc_entries);

size_t enlarge_space_singles(const HCIVec *hcivec, Rank *add_list, double thresh,
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo);

#endif

/**
 * @}
 */