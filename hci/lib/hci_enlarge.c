#include <math.h>
#include <string.h>
#include <stdbool.h>
#include "hci_enlarge.h"
#include "hci_contract.h"
#include "hci_rank.h"

bool get_changing_orbitals_new(const size_t *exc_list, size_t exc_order, const size_t *occ_list, size_t nocc,
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

size_t add_doubles_aa(const size_t *occ_a, size_t brank, const ExcitationEntries *exc_entries, double entry_thresh, 
    Rank *add_list, const ConfigInfo *config_info) {
        size_t nelec_a = config_info->nelec_a;
        size_t iadd = 0;
        for (size_t iexc=0; iexc<exc_entries->ndoubles_aa; iexc++) {
            DoubleExcitationEntry exc_entry_aa = exc_entries->doubles_aa[iexc];
            size_t exc_aa[4];
            size_t new_occ_a[nelec_a];
            ExcResult exc_result_aa = NEW_DOUBLE_EXC_RESULT();

            unrank_exc_aa(exc_entry_aa.rank, exc_aa, config_info);
            if (exc_entries->max_mag_aa[iexc] < entry_thresh) {
                break;
            }
            if (get_changing_orbitals_new(exc_aa, 2, occ_a, nelec_a, &exc_result_aa, new_occ_a)) {
                double exc_val = get_double_exc_value_from_store_new(&exc_entry_aa, &exc_result_aa);
                if (fabs(exc_val) >= entry_thresh) {
                    add_list[iadd].arank = rank_occ_a(new_occ_a, config_info);
                    add_list[iadd].brank = brank;
                    iadd++;
                }
            }
        }
        return iadd;
}

size_t add_doubles_bb(const size_t *occ_b, size_t arank, const ExcitationEntries *exc_entries, double entry_thresh, 
     Rank *add_list, const ConfigInfo *config_info) {
        size_t nelec_b = config_info->nelec_b;
        size_t iadd = 0;
        for (size_t iexc=0; iexc<exc_entries->ndoubles_bb; iexc++) {
            DoubleExcitationEntry exc_entry_bb = exc_entries->doubles_bb[iexc];
            size_t exc_bb[4];
            size_t new_occ_b[nelec_b];
            ExcResult exc_result_bb = NEW_DOUBLE_EXC_RESULT();
            exc_entry_bb = exc_entries->doubles_bb[iexc];

            unrank_exc_bb(exc_entry_bb.rank, exc_bb, config_info);
            if (exc_entries->max_mag_bb[iexc] < entry_thresh) {
                break;
            }
            if (get_changing_orbitals_new(exc_bb, 2, occ_b, nelec_b, &exc_result_bb, new_occ_b)) {
                double exc_val = get_double_exc_value_from_store_new(&exc_entry_bb, &exc_result_bb);
                if (fabs(exc_val) >= entry_thresh) {
                    add_list[iadd].arank = arank;
                    add_list[iadd].brank = rank_occ_b(new_occ_b, config_info);
                    iadd++;
                }
            }
        }
        return iadd;
}

size_t add_mixed_ab(const size_t *occ_a, const size_t *occ_b, const ExcitationEntries *exc_entries, double entry_thresh, 
     Rank *add_list, const ConfigInfo *config_info) {
        size_t nelec_a = config_info->nelec_a;
        size_t nelec_b = config_info->nelec_b;
        size_t iadd = 0;
        for (size_t iexc=0; iexc<exc_entries->nmixed_ab; iexc++) {
            MixedExcitationEntry exc_entry = exc_entries->mixed_ab[iexc];
            size_t exc_ab[4];
            size_t new_occ_a[nelec_a];
            size_t new_occ_b[nelec_b];
            ExcResult single_exc_result_a = NEW_SINGLE_EXC_RESULT();
            ExcResult single_exc_result_b = NEW_SINGLE_EXC_RESULT();
            
            unrank_exc_ab(exc_entry.rank, exc_ab, config_info);
            if (exc_entries->max_mag_ab[iexc] < entry_thresh) {
                break;
            }
            if (get_changing_orbitals_new(exc_ab, 1, occ_a, nelec_a, &single_exc_result_a, new_occ_a) &&
                get_changing_orbitals_new(exc_ab+2, 1, occ_b, nelec_b, &single_exc_result_b, new_occ_b)) {
                    add_list[iadd].arank = rank_occ_a(new_occ_a, config_info);
                    add_list[iadd].brank = rank_occ_b(new_occ_b, config_info);
                    iadd++;
            }
        }
        return iadd;
}

size_t enlarge_space_doubles_new(const HCIVector *hcivec, Rank *add_list, double thresh, 
    const ConfigInfo *config_info, const ExcitationEntries *exc_entries) {
        size_t iadd = 0;
        size_t norb = config_info->norb;
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
            unrank(arank, occ_a, config_info->config_table_a, norb, nelec_a);
            size_t nadd_aa = add_doubles_aa(occ_a, brank, exc_entries, entry_thresh, add_list+iadd, config_info);
            iadd += nadd_aa;

            // bb excitations
            unrank(brank, occ_b, config_info->config_table_b, norb, nelec_b);
            size_t nadd_bb = add_doubles_bb(occ_b, arank, exc_entries, entry_thresh, add_list+iadd, config_info);
            iadd += nadd_bb;

            // Mixed excitations
            size_t nadd_ab = add_mixed_ab(occ_a, occ_b, exc_entries, entry_thresh, add_list+iadd, config_info);
            iadd += nadd_ab;
        }
        return iadd;
}

size_t add_singles_a(const size_t *occ_a, const size_t *virt_a, const size_t *occ_b, size_t brank, 
    const H1E *h1e, const ERI_MO *eri_mo, double entry_thresh, Rank *add_list, const ConfigInfo *config_info) {
        size_t norb = config_info->norb;
        size_t nelec_a = config_info->nelec_a;
        size_t iadd = 0;
        for (size_t iocc=0; iocc<nelec_a; iocc++) {
            size_t occ_orb = occ_a[iocc];
            for (size_t ivirt=0; ivirt<norb-nelec_a; ivirt++) {
                size_t virt_orb = virt_a[ivirt];
                ExcResult single_exc_a = SINGLE_EXC_RESULT_NOSIGN(occ_orb, virt_orb);
                double exc_val = get_single_exc_value_a_new(&single_exc_a, occ_a, occ_b, 
                    config_info, h1e, eri_mo);
                if (fabs(exc_val) >= entry_thresh) {
                    size_t new_occ_a[nelec_a];
                    size_t *exc_list = SORTED(occ_orb, virt_orb);
                    get_changing_orbitals_new(exc_list, 1, occ_a, nelec_a, &single_exc_a, new_occ_a);
                    add_list[iadd].arank = rank_occ_a(new_occ_a, config_info);
                    add_list[iadd].brank = brank;
                    iadd++;
                }
            }
        }
        return iadd;
}

size_t add_singles_b(const size_t *occ_b, const size_t *virt_b, const size_t *occ_a, size_t arank, 
    const H1E *h1e, const ERI_MO *eri_mo, double entry_thresh, Rank *add_list, const ConfigInfo *config_info) {
        size_t norb = config_info->norb;
        size_t nelec_b = config_info->nelec_b;
        size_t iadd = 0;
        for (size_t iocc=0; iocc<nelec_b; iocc++) {
            size_t occ_orb = occ_b[iocc];
            for (size_t ivirt=0; ivirt<norb-nelec_b; ivirt++) {
                size_t virt_orb = virt_b[ivirt];
                ExcResult single_exc_b = SINGLE_EXC_RESULT_NOSIGN(occ_orb, virt_orb);
                double exc_val = get_single_exc_value_b_new(&single_exc_b, occ_a, occ_b, 
                    config_info, h1e, eri_mo);
                if (fabs(exc_val) >= entry_thresh) {
                    size_t new_occ_b[nelec_b];
                    size_t *exc_list = SORTED(occ_orb, virt_orb);
                    get_changing_orbitals_new(exc_list, 1, occ_b, nelec_b, &single_exc_b, new_occ_b);
                    add_list[iadd].arank = arank;
                    add_list[iadd].brank = rank_occ_b(new_occ_b, config_info);
                    iadd++;
                }
            }
        }
        return iadd;
}

size_t enlarge_space_singles_new(const HCIVector *hcivec, Rank *add_list, double thresh,
    const ConfigInfo *config_info, const H1E *h1e, const ERI_MO *eri_mo) {
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

// Old spaghetti
// Old spaghetti
// Old spaghetti

bool get_changing_orbitals(size_t *exc, size_t *occ, 
    size_t *exc_min_occ, size_t *exc_int_occ, size_t *new_occ,
    size_t *old_indices, size_t *new_indices, size_t exact_diffs, size_t nocc) {
        size_t iocc = 0;
        size_t iexc = 0;
        size_t inew = 0;
        size_t iexc_min_occ = 0;
        size_t iexc_int_occ = 0;
        size_t isegment, iocc_prev, edge, inew_indices;
        while ((iexc < 2*exact_diffs) && (iocc < nocc)) {
            size_t exc_orb = exc[iexc];
            size_t occ_orb = occ[iocc];
            if (exc_orb < occ_orb) {
                // Too many orbitals in excitation list are not in occupied list
                if (iexc_min_occ == exact_diffs) {
                    return false;
                }
                exc_min_occ[iexc_min_occ] = exc_orb;
                iexc++;
                iexc_min_occ++;
            } else if (exc_orb == occ_orb) {
                // Too many orbitals are in both lists
                if (iexc_int_occ == exact_diffs) {
                    return false;
                }
                exc_int_occ[iexc_int_occ] = exc_orb;
                old_indices[iexc_int_occ] = iocc;
                iexc++;
                iocc++;
                iexc_int_occ++;
            } else {
                iocc++;
            }
        }
        if (iexc == 2*exact_diffs) {
            // Traversed to the end of the excitation list; both set difference and intersection should be constructed by now
            if (!((iexc_min_occ == exact_diffs) && (iexc_int_occ == exact_diffs))) {
                return false;
            }
        } else if (iocc == nocc) {
            size_t remaining_in_exc = 2*exact_diffs-iexc;
            size_t remaining_in_exc_min_occ = exact_diffs-iexc_min_occ;
            // Traversed to the end of the occupancy list; intersection should be constructed and remaining part of set difference
            // needs to be the right length
            if (!((iexc_int_occ == exact_diffs) && remaining_in_exc==remaining_in_exc_min_occ)) {
                return false;
            } else {
                memcpy(exc_min_occ+iexc_min_occ, exc+iexc, remaining_in_exc*sizeof(size_t));
            }
        }
        // Now we can build our new occupancy list
        iocc_prev = 0;
        iexc_min_occ = 0;
        inew_indices = 0;
        // Deal with the fragmented pieces of occ
        for (isegment=0; isegment<exact_diffs; isegment++) {
            edge = old_indices[isegment];
            for (iocc=iocc_prev; iocc<edge; iocc++) {
                size_t occ_orb = occ[iocc];
                while (iexc_min_occ<exact_diffs)  {
                    size_t exc_orb = exc_min_occ[iexc_min_occ];
                    if (exc_orb > occ_orb) {
                        break;
                    }
                    new_occ[inew] = exc_orb;
                    new_indices[inew_indices] = inew;
                    iexc_min_occ++;
                    inew++;
                    inew_indices++;
                }
                new_occ[inew] = occ_orb;
                inew++;
            }
            iocc_prev = edge+1;
        }
        iocc = iocc_prev;
        // Now everything remaining is contiguous
        while ((iocc < nocc) && (iexc_min_occ<exact_diffs)) {
            size_t exc_orb = exc_min_occ[iexc_min_occ];
            size_t occ_orb = occ[iocc];
            if (exc_orb < occ_orb) {
                new_occ[inew] = exc_orb;
                new_indices[inew_indices] = inew;
                inew++;
                inew_indices++;
                iexc_min_occ++;
            } else {
                new_occ[inew] = occ_orb;
                inew++;
                iocc++;
            }
        }
        if (iexc_min_occ == exact_diffs) {
            memcpy(new_occ+inew, occ+iocc, (nocc-iocc)*sizeof(size_t));
        } else if (iocc == nocc) {
            memcpy(new_occ+inew, exc_min_occ+iexc_min_occ, (exact_diffs-iexc_min_occ)*sizeof(size_t));
            while (inew_indices<exact_diffs) {
                new_indices[inew_indices] = inew;
                inew_indices++;
                inew++;
            }
        }
        return true;
}

size_t enlarge_space_doubles(uint64_t *ranks, double *coeffs, size_t hci_len, uint64_t *add_doubles,
    size_t norb, size_t nelec_a, size_t nelec_b, double thresh,
    uint64_t *config_table_a, uint64_t *config_table_b, uint64_t *exc_table_4o, uint64_t *exc_table_2o,
    DoubleExcitationEntry *doubles_aa, size_t ndoubles_aa, 
    DoubleExcitationEntry *doubles_bb, size_t ndoubles_bb,
    MixedExcitationEntry *mixed_ab, size_t ndoubles_ab,
    double *max_mag_aa, double *max_mag_bb, double *max_mag_ab) {
        size_t iconfig, iexc, iadd;
        size_t occ_a[nelec_a], occ_b[nelec_b], exc_aa[4], exc_bb[4], exc_ab[4];
        iadd = 0;
        for (iconfig=0; iconfig<hci_len; iconfig++) {
            double coeff = coeffs[iconfig];
            uint64_t arank = ranks[2*iconfig], brank = ranks[2*iconfig+1];
            double entry_thresh = fabs(coeff*thresh);
            // aa excitations
            unrank(arank, occ_a, config_table_a, norb, nelec_a);
            for (iexc=0; iexc<ndoubles_aa; iexc++) {
                DoubleExcitationEntry exc_entry = doubles_aa[iexc];
                size_t exc_min_occ[2], exc_int_occ[2], old_indices[2], new_occ[nelec_a], new_indices[2];
                unrank(exc_entry.rank, exc_aa, exc_table_4o, norb, 4);
                if (max_mag_aa[iexc] < entry_thresh) {
                    break;
                }
                if (get_changing_orbitals(exc_aa, occ_a, exc_min_occ, exc_int_occ, new_occ, old_indices, new_indices, 2, nelec_a)) {
                     double excitation_mag = get_double_excitation_mag_from_store(exc_entry, exc_min_occ, exc_int_occ, old_indices, new_indices);
                     if (excitation_mag >= entry_thresh) {
                        add_doubles[iadd] = rank(new_occ, config_table_a, norb, nelec_a);
                        add_doubles[iadd+1] = brank;
                        iadd += 2;
                     }
                }
            }
            // bb excitations
            unrank(brank, occ_b, config_table_b, norb, nelec_b);
            for (iexc=0; iexc<ndoubles_bb; iexc++) {
                DoubleExcitationEntry exc_entry = doubles_bb[iexc];
                size_t exc_min_occ[2], exc_int_occ[2], old_indices[2], new_occ[nelec_b], new_indices[2];
                unrank(exc_entry.rank, exc_bb, exc_table_4o, norb, 4);
                if (max_mag_bb[iexc] < entry_thresh) {
                    break;
                }
                if (get_changing_orbitals(exc_bb, occ_b, exc_min_occ, exc_int_occ, new_occ, old_indices, new_indices, 2, nelec_b)) {
                     double excitation_mag = get_double_excitation_mag_from_store(exc_entry, exc_min_occ, exc_int_occ, old_indices, new_indices);
                     if (excitation_mag >= entry_thresh) {
                        add_doubles[iadd] = arank;
                        add_doubles[iadd+1] = rank(new_occ, config_table_b, norb, nelec_b);
                        iadd += 2;
                     }
                }
            }
            // Mixed excitations
            for (iexc=0; iexc<ndoubles_ab; iexc++) {
                MixedExcitationEntry exc_entry = mixed_ab[iexc];
                size_t new_a_orb[1], new_a_index[1], old_a_orb[1], old_a_index[1], new_b_orb[1], new_b_index[1], old_b_orb[1], old_b_index[1], new_a_occ[nelec_a], new_b_occ[nelec_b];
                unrank_mixed(exc_entry.rank, exc_ab, exc_table_2o, norb);
                if (max_mag_ab[iexc] < entry_thresh) {
                    break;
                }
                if (get_changing_orbitals(exc_ab, occ_a, new_a_orb, old_a_orb, new_a_occ, old_a_index, new_a_index, 1, nelec_a) &&
                    get_changing_orbitals(exc_ab+2, occ_b, new_b_orb, old_b_orb, new_b_occ, old_b_index, new_b_index, 1, nelec_b)) {
                        add_doubles[iadd] = rank(new_a_occ, config_table_a, norb, nelec_a);
                        add_doubles[iadd+1] = rank(new_b_occ, config_table_b, norb, nelec_b);
                        iadd += 2;
                }
            }
        }
        return iadd/2;
}

size_t enlarge_space_singles(uint64_t *ranks, double *coeffs, size_t hci_len, uint64_t *add_singles,
    size_t norb, size_t nelec_a, size_t nelec_b, uint64_t combmax_a, uint64_t combmax_b, double thresh,
    uint64_t *config_table_a, uint64_t *config_table_a_complement,
    uint64_t *config_table_b, uint64_t *config_table_b_complement,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4) {
        size_t iconfig, iocc, ivirt, iadd;
        size_t occ_a[nelec_a], virt_a[norb-nelec_a], occ_b[nelec_b], virt_b[norb-nelec_b];
        iadd = 0;
        for (iconfig=0; iconfig<hci_len; iconfig++) {
            double coeff = coeffs[iconfig];
            uint64_t arank = ranks[2*iconfig], brank = ranks[2*iconfig+1];
            double entry_thresh = fabs(coeff*thresh);
            unrank(arank, occ_a, config_table_a, norb, nelec_a);
            unrank(combmax_a-arank-1, virt_a, config_table_a_complement, norb, norb-nelec_a);
            unrank(brank, occ_b, config_table_b, norb, nelec_b);
            unrank(combmax_b-brank-1, virt_b, config_table_b_complement, norb, norb-nelec_b);

            // a excitations
            for (iocc=0; iocc<nelec_a; iocc++) {
                size_t occ_orb = occ_a[iocc];
                for (ivirt=0; ivirt<norb-nelec_a; ivirt++) {
                    size_t virt_orb = virt_a[ivirt];
                    double excitation_mag = get_single_excitation_mag_a(occ_orb, virt_orb, norb, nelec_a, nelec_b, occ_a, occ_b, 
                        h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4);
                    if (excitation_mag >= entry_thresh) {
                        size_t exc_a[2];
                        if (occ_orb < virt_orb) {
                            exc_a[0] = occ_orb;
                            exc_a[1] = virt_orb;
                        } else {
                            exc_a[0] = virt_orb;
                            exc_a[1] = occ_orb;
                        }
                        size_t exc_min_occ[1], exc_int_occ[1], new_occ[nelec_a], old_indices[1], new_indices[1];
                        get_changing_orbitals(exc_a, occ_a, exc_min_occ, exc_int_occ, new_occ, old_indices, new_indices, 1, nelec_a);
                        add_singles[iadd] = rank(new_occ, config_table_a, norb, nelec_a);
                        add_singles[iadd+1] = brank;
                        iadd += 2;
                    }
                }
            }

            // b excitations
            for (iocc=0; iocc<nelec_b; iocc++) {
                size_t occ_orb = occ_b[iocc];
                for (ivirt=0; ivirt<norb-nelec_b; ivirt++) {
                    size_t virt_orb = virt_b[ivirt];
                    double excitation_mag = get_single_excitation_mag_b(occ_orb, virt_orb, norb, nelec_a, nelec_b, occ_a, occ_b, 
                        h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4);
                    if (excitation_mag >= entry_thresh) {
                        size_t exc_b[2];
                        if (occ_orb < virt_orb) {
                            exc_b[0] = occ_orb;
                            exc_b[1] = virt_orb;
                        } else {
                            exc_b[0] = virt_orb;
                            exc_b[1] = occ_orb;
                        }
                        size_t exc_min_occ[1], exc_int_occ[1], new_occ[nelec_b], old_indices[1], new_indices[1];
                        get_changing_orbitals(exc_b, occ_b, exc_min_occ, exc_int_occ, new_occ, old_indices, new_indices, 1, nelec_b);
                        add_singles[iadd] = arank;
                        add_singles[iadd+1] = rank(new_occ, config_table_b, norb, nelec_b);
                        iadd += 2;
                    }
                }
            }
        }
        return iadd/2;
}