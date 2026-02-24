#include <math.h>
#include <string.h>
#include <stdbool.h>
#include "hci_rank.h"
#include "hci_store.h"
#include "hci_contract.h"

bool get_changing_orbitals(size_t *exc, size_t *occ, 
    size_t *exc_min_occ, size_t *exc_int_occ, size_t *new_occ,
    size_t exact_diffs, size_t nocc) {
        size_t iocc = 0;
        size_t iexc = 0;
        size_t inew = 0;
        size_t iexc_min_occ = 0;
        size_t iexc_int_occ = 0;
        size_t exc_int_occ_indices[exact_diffs];
        size_t isegment, iocc_prev, edge;
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
                exc_int_occ_indices[iexc_int_occ] = iocc;
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
        // Deal with the fragmented pieces of occ
        for (isegment=0; isegment<exact_diffs; isegment++) {
            edge = exc_int_occ_indices[isegment];
            for (iocc=iocc_prev; iocc<edge; iocc++) {
                size_t occ_orb = occ[iocc];
                while (iexc_min_occ<exact_diffs)  {
                    size_t exc_orb = exc_min_occ[iexc_min_occ];
                    if (exc_orb > occ_orb) {
                        break;
                    }
                    new_occ[inew] = exc_orb;
                    iexc_min_occ++;
                    inew++;
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
                inew++;
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
        size_t iconfig, iexc, nadd;
        size_t occ_a[nelec_a];
        size_t occ_b[nelec_b];
        size_t exc_aa[4];
        size_t exc_bb[4];
        size_t exc_ab[4];
        for (iconfig=0; iconfig<hci_len; iconfig++) {
            HCIEntry hci_entry = hcivec[iconfig];
            double entry_thresh = fabs(hci_entry.coeff*thresh);
            unrank(hci_entry.ranka, occ_a, config_table_a, norb, nelec_a);
            for (iexc=0; iexc<ndoubles_aa; iexc++) {
                DoubleExcitationEntry exc_entry = doubles_aa[iexc];
                unrank(exc_entry.rank, exc_aa, exc_table_4o, norb, nelec_a);

            }
        }
}