#include <math.h>
#include <string.h>
#include <stdbool.h>
#include "hci_rank.h"
#include "hci_store.h"
#include "hci_contract.h"

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

size_t enlarge_space_doubles(HCIEntry *hcivec, size_t hci_len, uint64_t *add_doubles,
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
            HCIEntry hci_entry = hcivec[iconfig];
            double entry_thresh = fabs(hci_entry.coeff*thresh);
            // aa excitations
            unrank(hci_entry.ranka, occ_a, config_table_a, norb, nelec_a);
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
                        add_doubles[iadd+1] = hci_entry.rankb;
                        iadd += 2;
                     }
                }
            }
            // bb excitations
            unrank(hci_entry.rankb, occ_b, config_table_b, norb, nelec_b);
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
                        add_doubles[iadd] = hci_entry.ranka;
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

size_t enlarge_space_singles(HCIEntry *hcivec, size_t hci_len, uint64_t *add_singles,
    size_t norb, size_t nelec_a, size_t nelec_b, uint64_t combmax_a, uint64_t combmax_b, double thresh,
    uint64_t *config_table_a, uint64_t *config_table_a_complement,
    uint64_t *config_table_b, uint64_t *config_table_b_complement,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4) {
        size_t iconfig, iocc, ivirt, iadd;
        size_t occ_a[nelec_a], virt_a[norb-nelec_a], occ_b[nelec_b], virt_b[norb-nelec_b];
        iadd = 0;
        for (iconfig=0; iconfig<hci_len; iconfig++) {
            HCIEntry hci_entry = hcivec[iconfig];
            double entry_thresh = fabs(hci_entry.coeff*thresh);
            unrank(hci_entry.ranka, occ_a, config_table_a, norb, nelec_a);
            unrank(combmax_a-hci_entry.ranka-1, virt_a, config_table_a_complement, norb, norb-nelec_a);
            unrank(hci_entry.rankb, occ_b, config_table_b, norb, nelec_b);
            unrank(combmax_b-hci_entry.rankb-1, virt_b, config_table_b_complement, norb, norb-nelec_b);

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
                        add_singles[iadd+1] = hci_entry.rankb;
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
                        add_singles[iadd] = hci_entry.ranka;
                        add_singles[iadd+1] = rank(new_occ, config_table_b, norb, nelec_b);
                        iadd += 2;
                    }
                }
            }
        }
        return iadd/2;
}