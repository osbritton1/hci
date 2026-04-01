/**
 * \ingroup enlarge
 */

#include "hci_enlarge.h"
#include "hci_contract.h"
#include <math.h>
#include <string.h>

bool get_changing_orbitals(const size_t *exc_list, size_t exc_order, const size_t *occ_list, size_t nocc,
    ExcResult *res, size_t *new_occ_list) {
        size_t iexc = 0;
        size_t iocc = 0;
        size_t iold = 0;
        size_t inew = 0;
        size_t inew_occ = 0;
        double sign = 1.0;
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
                // (res->new_indices)[inew] = inew_occ;
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
                // (res->old_indices)[iold] = iocc;
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
                    // sign *= ((inew_occ - inew) % 2 == 0) ? 1.0 : -1.0;
                    // (res->new_indices)[inew] = inew_occ;
                    iexc++;
                    inew++;
                    inew_occ++;
                }
            }
        }
        res->sign = sign;
        return true;
}

size_t add_doubles_aa(const size_t *occ_a, size_t brank, const ExcEntries *exc_entries, double entry_thresh, 
    Rank *add_list, const ConfigInfo *config_info) {
        size_t nelec_a = config_info->nelec_a;
        size_t iadd = 0;
        for (size_t iexc=0; iexc<exc_entries->ndoubles_aa; iexc++) {
            DoubleExcEntry exc_entry_aa = exc_entries->doubles_aa[iexc];
            size_t exc_aa[4];
            size_t new_occ_a[nelec_a];
            ExcResult exc_result_aa = NEW_DOUBLE_EXC_RESULT();

            unrank_double_exc(exc_entry_aa.rank, exc_aa, config_info);
            if (exc_entries->max_mag_aa[iexc] < entry_thresh) {
                break;
            }
            if (get_changing_orbitals(exc_aa, 2, occ_a, nelec_a, &exc_result_aa, new_occ_a)) {
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

size_t add_doubles_bb(const size_t *occ_b, size_t arank, const ExcEntries *exc_entries, double entry_thresh, 
     Rank *add_list, const ConfigInfo *config_info) {
        size_t nelec_b = config_info->nelec_b;
        size_t iadd = 0;
        for (size_t iexc=0; iexc<exc_entries->ndoubles_bb; iexc++) {
            DoubleExcEntry exc_entry_bb = exc_entries->doubles_bb[iexc];
            size_t exc_bb[4];
            size_t new_occ_b[nelec_b];
            ExcResult exc_result_bb = NEW_DOUBLE_EXC_RESULT();
            exc_entry_bb = exc_entries->doubles_bb[iexc];

            unrank_double_exc(exc_entry_bb.rank, exc_bb, config_info);
            if (exc_entries->max_mag_bb[iexc] < entry_thresh) {
                break;
            }
            if (get_changing_orbitals(exc_bb, 2, occ_b, nelec_b, &exc_result_bb, new_occ_b)) {
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

size_t add_mixed_ab(const size_t *occ_a, const size_t *occ_b, const ExcEntries *exc_entries, double entry_thresh, 
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
            
            unrank_mixed_exc(exc_entry.rank, exc_ab, config_info);
            if (exc_entries->max_mag_ab[iexc] < entry_thresh) {
                break;
            }
            if (get_changing_orbitals(exc_ab, 1, occ_a, nelec_a, &single_exc_result_a, new_occ_a) &&
                get_changing_orbitals(exc_ab+2, 1, occ_b, nelec_b, &single_exc_result_b, new_occ_b)) {
                    add_list[iadd].arank = rank_occ_a(new_occ_a, config_info);
                    add_list[iadd].brank = rank_occ_b(new_occ_b, config_info);
                    iadd++;
            }
        }
        return iadd;
}

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

size_t add_singles_a(const size_t *occ_a, const size_t *virt_a, const size_t *occ_b, size_t brank, 
    const HCore *h1e, const ERITensor *eri_mo, double entry_thresh, Rank *add_list, const ConfigInfo *config_info) {
        size_t norb = config_info->norb;
        size_t nelec_a = config_info->nelec_a;
        size_t iadd = 0;
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

size_t add_singles_b(const size_t *occ_b, const size_t *virt_b, const size_t *occ_a, size_t arank, 
    const HCore *h1e, const ERITensor *eri_mo, double entry_thresh, Rank *add_list, const ConfigInfo *config_info) {
        size_t norb = config_info->norb;
        size_t nelec_b = config_info->nelec_b;
        size_t iadd = 0;
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