/**
 * \file hci_enlarge.c
 * \addtogroup enlarge
 * @{
 */

#include "hci_enlarge.h"
#include "hci_contract.h"
#include <math.h>
#include <string.h>

/**
 * Given a list of excitation orbitals, attempts to excite the provided configuration;
 * on success, outputs the detailed information of the excitation and generates a new
 * occupancy list.
 *
 * To be explicit, this function returns true exactly when the the intersection of \p exc_list
 * and \p occ_list and the set difference \p exc_list\ \p occ_list are both of size \p exc_order, in which
 * case the intersection will be written to \p res->old_orbs, the difference to \p res->new_orbs
 * and the excited configuration formed by substituting \c old_orbs with \c new_orbs to \p new_occ_list.
 * This is essentially a customized merge algorithm for sorted lists.
 *
 * @param[in] exc_list Pointer to the list of orbitals involved in the excitation of length 2*\p exc_order (sorted in asc. order)
 * @param[in] exc_order The order of the excitation
 * @param[in] occ_list Pointer to the occupancy list of the reference determinant of length \p nocc (sorted in asc. order)
 * @param[in] nocc The number of orbitals occupied in the reference determinant
 * @param[out] res Pointer to the ExcResult to be written to on success
 * @param[out] new_occ_list Pointer to array where the new occupied orbitals should be stored (of length \p nocc); will be sorted in asc. order as well
 * @return True if the given excitation and determinant are compatible; when false, \p res and \p new_occ_list are garbage
 */
static bool get_changing_orbitals(const size_t *exc_list, size_t exc_order, const size_t *occ_list, size_t nocc,
    ExcResult *res, size_t *new_occ_list) {
        size_t iexc = 0; // Index of excitation list being read from
        size_t iocc = 0; // Index of occupancy list being read from
        size_t iold = 0; // Index of ExcResult old_orbs being written to
        size_t inew = 0; // Index of ExcResult new_orbs being written to
        size_t inew_occ = 0; // Index of new occupancy list being written to
        double sign = 1.0; // Sign of excitation
        while ((iexc < 2*exc_order) && (iocc < nocc)) {
            size_t exc_orb = exc_list[iexc];
            size_t occ_orb = occ_list[iocc];
            if (exc_orb < occ_orb) {
                // Too many orbitals in excitation list are not in occupied list
                // or new occupancy list overflows (intersection not big enough)
                if ((inew == exc_order) || inew_occ == nocc) {
                    return false;
                }
                (res->new_orbs)[inew] = exc_orb;
                new_occ_list[inew_occ] = exc_orb;
                sign *= ((inew_occ - inew) % 2 == 0) ? 1.0 : -1.0;
                iexc++;
                inew++;
                inew_occ++;
            } else if (exc_orb == occ_orb) {
                // Too many orbitals are in both lists (intersection too big)
                if (iold == exc_order) {
                    return false;
                }
                (res->old_orbs)[iold] = occ_orb;
                sign *= ((iocc - iold) % 2 == 0) ? 1.0 : -1.0;
                iexc++;
                iocc++;
                iold++;
            } else {
                // Occupancy list overflows (intersection not big enough)
                if (inew_occ == nocc) {
                    return false;
                }
                new_occ_list[inew_occ] = occ_orb;
                iocc++;
                inew_occ++;
            }
        }
        if (iexc == 2*exc_order) {
            // Traversed to the end of the excitation list; both set difference and intersection should be constructed by now
            if (!((inew == exc_order) && (iold == exc_order))) {
                return false;
            } else {
                size_t rem_in_occ = nocc-iocc;
                memcpy(new_occ_list+inew_occ, occ_list+iocc, rem_in_occ*sizeof(size_t));
            }
        } else {
            size_t rem_in_exc = (2*exc_order)-iexc;
            size_t rem_in_new = exc_order-inew;
            // Traversed to the end of the occupancy list; intersection should be constructed and remaining part of set difference
            // needs to be the right length
            if (!((iold == exc_order) && rem_in_exc==rem_in_new)) {
                return false;
            } else {
                sign *= (((inew_occ - inew) % 2 == 0) || (rem_in_new % 2 == 0)) ? 1.0 : -1.0;
                while (inew < exc_order) {
                    (res->new_orbs)[inew] = exc_list[iexc];
                    new_occ_list[inew_occ] = exc_list[iexc];
                    iexc++;
                    inew++;
                    inew_occ++;
                }
            }
        }
        res->sign = sign;
        return true;
}

/**
 * Given a reference \f$\alpha\f$ occupancy list, outputs all \f$\alpha\alpha\rightarrow\alpha\alpha\f$
 * excitations to \p add_list that satisfy the HCI selection criterion.
 *
 * @param[in] occ_a Pointer to list of occupied \f$\alpha\f$ orbitals
 * @param[in] brank Rank of associated \f$\beta\f$ string being left unchanged
 * @param[in] exc_entries Pointer to desc. sorted \ref ExcEntries object providing location of stored double \f$\alpha\f$ excitations
 * @param[in] entry_thresh Selection threshold; value handed in from outer loop in \ref enlarge_space_doubles
 * @param[out] add_list Pointer to \ref Rank list storing configurations to be added
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 */
static size_t add_doubles_aa(const size_t *occ_a, size_t brank, const ExcEntries *exc_entries, double entry_thresh, 
    Rank *add_list, const ConfigInfo *config_info) {
        size_t nelec_a = config_info->nelec_a;
        size_t iadd = 0;
        for (size_t iexc=0; iexc<exc_entries->ndoubles_aa; iexc++) {
            DoubleExcEntry exc_entry_aa = exc_entries->doubles_aa[iexc];
            size_t exc_aa[4];
            size_t new_occ_a[nelec_a];
            ExcResult exc_result_aa = NEW_DOUBLE_EXC_RESULT();
            // Decode excitation entry
            unrank_double_exc(exc_entry_aa.rank, exc_aa, config_info);
            // Break if excitation magnitude falls below threshold
            if (exc_entries->max_mag_aa[iexc] < entry_thresh) {
                break;
            }
            // If the excitation entry is a valid double excitation
            if (get_changing_orbitals(exc_aa, 2, occ_a, nelec_a, &exc_result_aa, new_occ_a)) {
                // Get correct matrix element and compare to threshold
                double exc_val = get_double_exc_value_from_store(&exc_entry_aa, &exc_result_aa);
                if (fabs(exc_val) >= entry_thresh) {
                    add_list[iadd].arank = rank_occ_a(new_occ_a, config_info);
                    add_list[iadd].brank = brank;
                    iadd++;
                }
            }
        }
        return iadd;
}

/**
 * Given a reference \f$\beta\f$ occupancy list, outputs all \f$\beta\beta\rightarrow\beta\beta\f$
 * excitations to \p add_list that satisfy the HCI selection criterion.
 *
 * @param[in] occ_b Pointer to list of occupied \f$\beta\f$ orbitals
 * @param[in] arank Rank of associated \f$\alpha\f$ string being left unchanged
 * @param[in] exc_entries Pointer to desc. sorted \ref ExcEntries object providing location of stored double \f$\beta\f$ excitations
 * @param[in] entry_thresh Selection threshold; value handed in from outer loop in \ref enlarge_space_doubles
 * @param[out] add_list Pointer to \ref Rank list storing configurations to be added
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 */
static size_t add_doubles_bb(const size_t *occ_b, size_t arank, const ExcEntries *exc_entries, double entry_thresh, 
     Rank *add_list, const ConfigInfo *config_info) {
        size_t nelec_b = config_info->nelec_b;
        size_t iadd = 0;
        for (size_t iexc=0; iexc<exc_entries->ndoubles_bb; iexc++) {
            DoubleExcEntry exc_entry_bb = exc_entries->doubles_bb[iexc];
            size_t exc_bb[4];
            size_t new_occ_b[nelec_b];
            ExcResult exc_result_bb = NEW_DOUBLE_EXC_RESULT();
            exc_entry_bb = exc_entries->doubles_bb[iexc];
            // Decode excitation entry
            unrank_double_exc(exc_entry_bb.rank, exc_bb, config_info);
            // Break if excitation magnitude falls below threshold
            if (exc_entries->max_mag_bb[iexc] < entry_thresh) {
                break;
            }
            // If the excitation entry is a valid double excitation 
            if (get_changing_orbitals(exc_bb, 2, occ_b, nelec_b, &exc_result_bb, new_occ_b)) {
                // Get correct matrix element and compare to threshold
                double exc_val = get_double_exc_value_from_store(&exc_entry_bb, &exc_result_bb);
                if (fabs(exc_val) >= entry_thresh) {
                    add_list[iadd].arank = arank;
                    add_list[iadd].brank = rank_occ_b(new_occ_b, config_info);
                    iadd++;
                }
            }
        }
        return iadd;
}

/**
 * Given reference \f$\alpha\f$ and \f$\beta\f$ occupancy lists, outputs all \f$\alpha\beta\rightarrow\alpha\beta\f$
 * excitations to \p add_list that satisfy the HCI selection criterion.
 *
 * @param[in] occ_a Pointer to list of occupied \f$\alpha\f$ orbitals
 * @param[in] occ_b Pointer to list of occupied \f$\beta\f$ orbitals
 * @param[in] exc_entries Pointer to desc. sorted \ref ExcEntries object providing location of stored mixed excitations
 * @param[in] entry_thresh Selection threshold; value handed in from outer loop in \ref enlarge_space_doubles
 * @param[out] add_list Pointer to \ref Rank list storing configurations to be added
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 */
static size_t add_mixed_ab(const size_t *occ_a, const size_t *occ_b, const ExcEntries *exc_entries, double entry_thresh, 
     Rank *add_list, const ConfigInfo *config_info) {
        size_t nelec_a = config_info->nelec_a;
        size_t nelec_b = config_info->nelec_b;
        size_t iadd = 0;
        for (size_t iexc=0; iexc<exc_entries->nmixed_ab; iexc++) {
            MixedExcEntry exc_entry = exc_entries->mixed_ab[iexc];
            size_t exc_ab[4];
            size_t new_occ_a[nelec_a];
            size_t new_occ_b[nelec_b];
            ExcResult single_exc_result_a = NEW_SINGLE_EXC_RESULT();
            ExcResult single_exc_result_b = NEW_SINGLE_EXC_RESULT();
            // Decode the mixed excitation
            unrank_mixed_exc(exc_entry.rank, exc_ab, config_info);
            // Break if excitation magnitude falls below threshold
            if (exc_entries->max_mag_ab[iexc] < entry_thresh) {
                break;
            }
            // If the excitation entry is a valid mixed excitation
            if (get_changing_orbitals(exc_ab, 1, occ_a, nelec_a, &single_exc_result_a, new_occ_a) &&
                get_changing_orbitals(exc_ab+2, 1, occ_b, nelec_b, &single_exc_result_b, new_occ_b)) {
                    add_list[iadd].arank = rank_occ_a(new_occ_a, config_info);
                    add_list[iadd].brank = rank_occ_b(new_occ_b, config_info);
                    iadd++;
            }
        }
        return iadd;
}

/**
 * Outputs all double and mixed excitations to \p add_list that satisfy the HCI selection criterion.
 *
 * @param[in] hcivec Pointer to a \ref HCIVec supplying configuration ranks and their coefficients
 * @param[out] add_list Pointer to \ref Rank list storing configurations to be added
 * @param[in] thresh The threshold used for adding a determinant to the configuration space; compared to the magnitude of the coefficient times the excitation matrix element
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 * @param[in] exc_entries Pointer to desc. sorted \ref ExcEntries object providing location of stored mixed excitations
 * @return The number of (not necessarily distinct) configurations written to \p add_list
 */
size_t enlarge_space_doubles(const HCIVec *hcivec, Rank *add_list, double thresh, 
    const ConfigInfo *config_info, const ExcEntries *exc_entries) {
        size_t iadd = 0;
        size_t nelec_a = config_info->nelec_a;
        size_t nelec_b = config_info->nelec_b;
        for (size_t iconfig=0; iconfig<hcivec->len; iconfig++) {
            size_t occ_a[nelec_a];
            size_t occ_b[nelec_b];
            double coeff = hcivec->coeffs[iconfig];
            uint64_t arank = hcivec->ranks[iconfig].arank;
            uint64_t brank = hcivec->ranks[iconfig].brank;
            double entry_thresh = fabs(coeff*thresh);

            // aa excitations
            unrank_occ_a(arank, occ_a, config_info);
            size_t nadd_aa = add_doubles_aa(occ_a, brank, exc_entries, entry_thresh, add_list+iadd, config_info);
            iadd += nadd_aa;

            // bb excitations
            unrank_occ_b(brank, occ_b, config_info);
            size_t nadd_bb = add_doubles_bb(occ_b, arank, exc_entries, entry_thresh, add_list+iadd, config_info);
            iadd += nadd_bb;

            // Mixed excitations
            size_t nadd_ab = add_mixed_ab(occ_a, occ_b, exc_entries, entry_thresh, add_list+iadd, config_info);
            iadd += nadd_ab;
        }
        return iadd;
}

/**
 * Given reference \f$\alpha\f$ and \f$\beta\f$ occupancy lists, outputs all single \f$\alpha\rightarrow\alpha\f$
 * excitations to \p add_list that satisfy the HCI selection criterion.
 *
 * @param[in] occ_a Pointer to list of occupied \f$\alpha\f$ orbitals
 * @param[in] virt_a Pointer to list of virtual/unoccupied \f$\alpha\f$ orbitals
 * @param[in] occ_b Pointer to list of occupied \f$\beta\f$ orbitals
 * @param[in] brank Rank of associated \f$\beta\f$ string being left unchanged
 * @param[in] h1e Pointer to \ref HCore object storing locations of the core Hamiltonian matrix elements
 * @param[in] eri_mo Pointer to \ref ERITensor object storing locations of the electron repulsion integrals
 * @param[in] entry_thresh Selection threshold; value handed in from outer loop in \ref enlarge_space_singles
 * @param[out] add_list Pointer to \ref Rank list storing configurations to be added
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 */
static size_t add_singles_a(const size_t *occ_a, const size_t *virt_a, const size_t *occ_b, size_t brank, 
    const HCore *h1e, const ERITensor *eri_mo, double entry_thresh, Rank *add_list, const ConfigInfo *config_info) {
        size_t norb = config_info->norb;
        size_t nelec_a = config_info->nelec_a;
        size_t iadd = 0;
        // Try and generate all single excitations from an occupied to a virtual orbital
        for (size_t iocc=0; iocc<nelec_a; iocc++) {
            size_t occ_orb = occ_a[iocc];
            for (size_t ivirt=0; ivirt<norb-nelec_a; ivirt++) {
                size_t virt_orb = virt_a[ivirt];
                ExcResult single_exc_a = SINGLE_EXC_RESULT_NOSIGN(occ_orb, virt_orb);
                double exc_val = get_single_exc_value_a(&single_exc_a, occ_a, occ_b, 
                    config_info, h1e, eri_mo);
                if (fabs(exc_val) >= entry_thresh) {
                    size_t new_occ_a[nelec_a];
                    size_t *exc_list = SORTED(occ_orb, virt_orb);
                    get_changing_orbitals(exc_list, 1, occ_a, nelec_a, &single_exc_a, new_occ_a);
                    add_list[iadd].arank = rank_occ_a(new_occ_a, config_info);
                    add_list[iadd].brank = brank;
                    iadd++;
                }
            }
        }
        return iadd;
}

/**
 * Given reference \f$\alpha\f$ and \f$\beta\f$ occupancy lists, outputs all single \f$\beta\rightarrow\beta\f$
 * excitations to \p add_list that satisfy the HCI selection criterion.
 *
 * @param[in] occ_b Pointer to list of occupied \f$\beta\f$ orbitals
 * @param[in] virt_b Pointer to list of virtual/unoccupied \f$\beta\f$ orbitals
 * @param[in] occ_a Pointer to list of occupied \f$\alpha\f$ orbitals
 * @param[in] arank Rank of associated \f$\alpha\f$ string being left unchanged
 * @param[in] h1e Pointer to \ref HCore object storing locations of the core Hamiltonian matrix elements
 * @param[in] eri_mo Pointer to \ref ERITensor object storing locations of the electron repulsion integrals
 * @param[in] entry_thresh Selection threshold; value handed in from outer loop in \ref enlarge_space_singles
 * @param[out] add_list Pointer to \ref Rank list storing configurations to be added
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 */
static size_t add_singles_b(const size_t *occ_b, const size_t *virt_b, const size_t *occ_a, size_t arank, 
    const HCore *h1e, const ERITensor *eri_mo, double entry_thresh, Rank *add_list, const ConfigInfo *config_info) {
        size_t norb = config_info->norb;
        size_t nelec_b = config_info->nelec_b;
        size_t iadd = 0;
        // Try and generate all single excitations from an occupied to a virtual orbital
        for (size_t iocc=0; iocc<nelec_b; iocc++) {
            size_t occ_orb = occ_b[iocc];
            for (size_t ivirt=0; ivirt<norb-nelec_b; ivirt++) {
                size_t virt_orb = virt_b[ivirt];
                ExcResult single_exc_b = SINGLE_EXC_RESULT_NOSIGN(occ_orb, virt_orb);
                double exc_val = get_single_exc_value_b(&single_exc_b, occ_a, occ_b, 
                    config_info, h1e, eri_mo);
                if (fabs(exc_val) >= entry_thresh) {
                    size_t new_occ_b[nelec_b];
                    size_t *exc_list = SORTED(occ_orb, virt_orb);
                    get_changing_orbitals(exc_list, 1, occ_b, nelec_b, &single_exc_b, new_occ_b);
                    add_list[iadd].arank = arank;
                    add_list[iadd].brank = rank_occ_b(new_occ_b, config_info);
                    iadd++;
                }
            }
        }
        return iadd;
}

/**
 * Outputs all single excitations to \p add_list that satisfy the HCI selection criterion.
 *
 * @param[in] hcivec Pointer to a \ref HCIVec supplying configuration ranks and their coefficients
 * @param[out] add_list Pointer to \ref Rank list storing configurations to be added
 * @param[in] thresh The threshold used for adding a determinant to the configuration space; compared to the magnitude of the coefficient times the excitation matrix element
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 * @param[in] h1e Pointer to \ref HCore object storing locations of the core Hamiltonian matrix elements
 * @param[in] eri_mo Pointer to \ref ERITensor object storing locations of the electron repulsion integrals
 * @return The number of (not necessarily distinct) configurations written to \p add_list
 */
size_t enlarge_space_singles(const HCIVec *hcivec, Rank *add_list, double thresh,
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo) {
        size_t iadd = 0;
        size_t norb = config_info->norb;
        size_t nelec_a = config_info->nelec_a;
        size_t nelec_b = config_info->nelec_b;
        for (size_t iconfig=0; iconfig<hcivec->len; iconfig++) {
            size_t occ_a[nelec_a];
            size_t occ_b[nelec_b];
            size_t virt_a[norb-nelec_a];
            size_t virt_b[norb-nelec_b];
            double coeff = hcivec->coeffs[iconfig];
            uint64_t arank = hcivec->ranks[iconfig].arank;
            uint64_t brank = hcivec->ranks[iconfig].brank;
            double entry_thresh = fabs(coeff*thresh);
            
            unrank_occ_a(arank, occ_a, config_info);
            unrank_virt_a(arank, virt_a, config_info);
            unrank_occ_b(brank, occ_b, config_info);
            unrank_virt_b(brank, virt_b, config_info);

            // a excitations
            size_t nadd_a = add_singles_a(occ_a, virt_a, occ_b, brank, h1e, eri_mo, entry_thresh,
                add_list+iadd, config_info);
            iadd += nadd_a;

            // b excitations
            size_t nadd_b = add_singles_b(occ_b, virt_b, occ_a, arank, h1e, eri_mo, entry_thresh,
                add_list+iadd, config_info);
            iadd += nadd_b;
        }
        return iadd;
}

/**
 * @}
 */