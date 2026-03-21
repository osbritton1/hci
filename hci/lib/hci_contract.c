#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include "hci_contract.h"
#include "hci_rank.h"
#include "hci_store.h"
#include "hci_enlarge.h"

double get_single_excitation_value_a(size_t occ_orb, size_t virt_orb, size_t norb, size_t nelec_a, size_t nelec_b, size_t *occ_a, size_t *occ_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4) {
        size_t iocc;
        size_t ncols = nC2(norb+1);
        double sum = 0.0, sign;
        double *row = eri_aabb_s4+index_2d(occ_orb, virt_orb)*ncols;
        size_t exc_a[2], exc_min_occ[1], exc_int_occ[1], new_occ[nelec_a], old_indices[1], new_indices[1];

        // Contribution from aaaa block
        for (iocc=0; iocc<nelec_a; iocc++) {
            size_t occ_orb_2 = occ_a[iocc];
            sum += eri_aaaa_s8[index_4d(occ_orb, virt_orb, occ_orb_2, occ_orb_2)]-eri_aaaa_s8[index_4d(occ_orb, occ_orb_2, occ_orb_2, virt_orb)];
        }
        // Contribution from aabb block
        for (iocc=0; iocc<nelec_b; iocc++) {
            size_t occ_orb_2 = occ_b[iocc];
            sum += row[index_2d(occ_orb_2, occ_orb_2)];
        }
        // Contribution from 1e Hamiltonian
        sum += h1e_aa[occ_orb*norb+virt_orb];

        if (occ_orb < virt_orb) {
            exc_a[0] = occ_orb;
            exc_a[1] = virt_orb;
        } else {
            exc_a[0] = virt_orb;
            exc_a[1] = occ_orb;
        }
        get_changing_orbitals(exc_a, occ_a, exc_min_occ, exc_int_occ, new_occ, old_indices, new_indices, 1, nelec_a);
        if ((old_indices[0] + new_indices[0]) % 2 == 0) {
            sign = 1.0;
        } else {
            sign = -1.0;
        }
        return sign*sum;
}

double get_single_excitation_mag_a(size_t occ_orb, size_t virt_orb, size_t norb, size_t nelec_a, size_t nelec_b, size_t *occ_a, size_t *occ_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4) {
        size_t iocc;
        size_t ncols = nC2(norb+1);
        double sum = 0.0;
        double *row = eri_aabb_s4+index_2d(occ_orb, virt_orb)*ncols;
        // Contribution from aaaa block
        for (iocc=0; iocc<nelec_a; iocc++) {
            size_t occ_orb_2 = occ_a[iocc];
            sum += eri_aaaa_s8[index_4d(occ_orb, virt_orb, occ_orb_2, occ_orb_2)]-eri_aaaa_s8[index_4d(occ_orb, occ_orb_2, occ_orb_2, virt_orb)];
        }
        // Contribution from aabb block
        for (iocc=0; iocc<nelec_b; iocc++) {
            size_t occ_orb_2 = occ_b[iocc];
            sum += row[index_2d(occ_orb_2, occ_orb_2)];
        }
        // Contribution from 1e Hamiltonian
        sum += h1e_aa[occ_orb*norb+virt_orb];
        return fabs(sum);
}

double get_single_excitation_value_b(size_t occ_orb, size_t virt_orb, size_t norb, size_t nelec_a, size_t nelec_b, size_t *occ_a, size_t *occ_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4) {
        size_t iocc;
        size_t ncols = nC2(norb+1);
        size_t col = index_2d(occ_orb, virt_orb);
        double sum = 0.0, sign;
        size_t exc_b[2], exc_min_occ[1], exc_int_occ[1], new_occ[nelec_a], old_indices[1], new_indices[1];

        // Contribution from bbbb block
        for (iocc=0; iocc<nelec_b; iocc++) {
            size_t occ_orb_2 = occ_b[iocc];
            sum += eri_bbbb_s8[index_4d(occ_orb, virt_orb, occ_orb_2, occ_orb_2)]-eri_bbbb_s8[index_4d(occ_orb, occ_orb_2, occ_orb_2, virt_orb)];
        }
        // Contribution from aabb block
        for (iocc=0; iocc<nelec_a; iocc++) {
            size_t occ_orb_2 = occ_a[iocc];
            sum += eri_aabb_s4[index_2d(occ_orb_2, occ_orb_2)*ncols+col];
        }
        // Contribution from 1e Hamiltonian
        sum += h1e_bb[occ_orb*norb+virt_orb];

        if (occ_orb < virt_orb) {
            exc_b[0] = occ_orb;
            exc_b[1] = virt_orb;
        } else {
            exc_b[0] = virt_orb;
            exc_b[1] = occ_orb;
        }
        get_changing_orbitals(exc_b, occ_b, exc_min_occ, exc_int_occ, new_occ, old_indices, new_indices, 1, nelec_b);
        if ((old_indices[0] + new_indices[0]) % 2 == 0) {
            sign = 1.0;
        } else {
            sign = -1.0;
        }
        return sign*sum;
}

double get_single_excitation_mag_b(size_t occ_orb, size_t virt_orb, size_t norb, size_t nelec_a, size_t nelec_b, size_t *occ_a, size_t *occ_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4) {
        size_t iocc;
        size_t ncols = nC2(norb+1);
        size_t col = index_2d(occ_orb, virt_orb);
        double sum = 0.0;
        // Contribution from bbbb block
        for (iocc=0; iocc<nelec_b; iocc++) {
            size_t occ_orb_2 = occ_b[iocc];
            sum += eri_bbbb_s8[index_4d(occ_orb, virt_orb, occ_orb_2, occ_orb_2)]-eri_bbbb_s8[index_4d(occ_orb, occ_orb_2, occ_orb_2, virt_orb)];
        }
        // Contribution from aabb block
        for (iocc=0; iocc<nelec_a; iocc++) {
            size_t occ_orb_2 = occ_a[iocc];
            sum += eri_aabb_s4[index_2d(occ_orb_2, occ_orb_2)*ncols+col];
        }
        // Contribution from 1e Hamiltonian
        sum += h1e_bb[occ_orb*norb+virt_orb];
        return fabs(sum);
}

double get_double_excitation_value_aa(size_t *one_min_two, size_t *two_min_one, size_t *one_min_two_indices, size_t *two_min_one_indices, double *eri_aaaa_s8) {
    double sign;
    if ((one_min_two_indices[0]+(one_min_two_indices[1]-1)+two_min_one_indices[0]+(two_min_one_indices[1]-1)) % 2 == 0) {
        sign = 1.0;
    } else {
        sign = -1.0;
    }
    return sign*(eri_aaaa_s8[index_4d(one_min_two[0], two_min_one[0], one_min_two[1], two_min_one[1])]
        -eri_aaaa_s8[index_4d(one_min_two[0], two_min_one[1], one_min_two[1], two_min_one[0])]);
}

double get_double_excitation_value_bb(size_t *one_min_two, size_t *two_min_one, size_t *one_min_two_indices, size_t *two_min_one_indices, double *eri_bbbb_s8) {
    double sign;
    if ((one_min_two_indices[0]+(one_min_two_indices[1]-1)+two_min_one_indices[0]+(two_min_one_indices[1]-1)) % 2 == 0) {
        sign = 1.0;
    } else {
        sign = -1.0;
    }
    return sign*(eri_bbbb_s8[index_4d(one_min_two[0], two_min_one[0], one_min_two[1], two_min_one[1])]
        -eri_bbbb_s8[index_4d(one_min_two[0], two_min_one[1], one_min_two[1], two_min_one[0])]);
}

double get_double_excitation_value_from_store(DoubleExcitationEntry exc_entry, size_t *exc_min_occ, size_t *exc_int_occ, size_t *old_indices, size_t *new_indices) {
    size_t orb_list[4];
    double sign;
    uint8_t designator = 0;
    if (exc_int_occ[0] <= exc_min_occ[0]) {
        orb_list[0] = exc_int_occ[0];
        orb_list[1] = exc_min_occ[0];
        orb_list[2] = exc_int_occ[1];
        orb_list[3] = exc_min_occ[1];
    } else {
        orb_list[0] = exc_min_occ[0];
        orb_list[1] = exc_int_occ[0];
        orb_list[2] = exc_min_occ[1];
        orb_list[3] = exc_int_occ[1];
    }
    if ((old_indices[0]+(old_indices[1]-1)+new_indices[0]+(new_indices[1]-1)) % 2 == 0) {
        sign = 1.0;
    } else {
        sign = -1.0;
    }
    designator += (orb_list[1] < orb_list[2]) ? 4 : 0;
    designator += (orb_list[2] < orb_list[3]) ? 2 : 0;
    designator += (orb_list[3] < orb_list[1]) ? 1 : 0;
    switch (designator) {
        case 1:
            return -exc_entry.ijkl*sign;
        case 2:
            return -exc_entry.iljk*sign;
        case 3:
            return exc_entry.iljk*sign;
        case 4:
            return (exc_entry.ijkl+exc_entry.iljk)*sign;
        case 5:
            return -(exc_entry.ijkl+exc_entry.iljk)*sign;
        case 6:
            return exc_entry.ijkl*sign;
        default:
            return nan("");
    }
}

double get_double_excitation_mag_from_store(DoubleExcitationEntry exc_entry, size_t *exc_min_occ, size_t *exc_int_occ, size_t *old_indices, size_t *new_indices) {
    size_t orb_list[4];
    if (exc_int_occ[0] <= exc_min_occ[0]) {
        orb_list[0] = exc_int_occ[0];
        orb_list[1] = exc_min_occ[0];
        orb_list[2] = exc_int_occ[1];
        orb_list[3] = exc_min_occ[1];
    } else {
        orb_list[0] = exc_min_occ[0];
        orb_list[1] = exc_int_occ[0];
        orb_list[2] = exc_min_occ[1];
        orb_list[3] = exc_int_occ[1];
    }
    if ((orb_list[1] < orb_list[2]) && (orb_list[2] < orb_list[3])) {
        return fabs(exc_entry.ijkl);
    } else if ((orb_list[2] < orb_list[3]) && (orb_list[3] < orb_list[1])) {
        return fabs(exc_entry.iljk);
    } else {
        return fabs(exc_entry.ijkl+exc_entry.iljk);
    }
}

double get_mixed_excitation_value(size_t *occ_a_1_min_2, size_t *occ_a_2_min_1, size_t *occ_a_1_min_2_indices, size_t *occ_a_2_min_1_indices,
    size_t *occ_b_1_min_2, size_t *occ_b_2_min_1, size_t *occ_b_1_min_2_indices, size_t *occ_b_2_min_1_indices,
    size_t norb, double *eri_aabb_s4) {
        double sign;
        size_t ncols = nC2(norb+1);
        if (((occ_a_1_min_2_indices[0] + occ_a_2_min_1_indices[0] + occ_b_1_min_2_indices[0] + occ_b_2_min_1_indices[0]) % 2) == 0) {
            sign = 1.0;
        } else {
            sign = -1.0;
        }
        return sign*eri_aabb_s4[index_2d(occ_a_1_min_2[0], occ_a_2_min_1[0])*ncols+index_2d(occ_b_1_min_2[0], occ_b_2_min_1[0])];
}

double get_mixed_excitation_value_from_store(MixedExcitationEntry exc_entry, size_t old_a_index, size_t new_a_index, size_t old_b_index, size_t new_b_index) {
    double sign;
    if (((old_a_index + new_a_index + old_b_index + new_b_index) % 2) == 0) {
        sign = 1.0;
    } else {
        sign = -1.0;
    }
    return sign*exc_entry.ijkl;
}

double get_diag_value(size_t *occ_a, size_t *occ_b, size_t norb, size_t nelec_a, size_t nelec_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4) {
        double sum = 0;
        size_t iocc, jocc, occ_orb, occ_orb_2;
        size_t ncols = nC2(norb+1);
        // Contribution from h1e_aa
        for (iocc=0; iocc<nelec_a; iocc++) {
            occ_orb = occ_a[iocc];
            sum += h1e_aa[occ_orb*norb+occ_orb];
        }
        // Contribution from h1e_bb
        for (iocc=0; iocc<nelec_b; iocc++) {
            occ_orb = occ_b[iocc];
            sum += h1e_bb[occ_orb*norb+occ_orb];
        }
        // Contribution from eri_aaaa_s8
        for (iocc=0; iocc<nelec_a-1; iocc++) {
            occ_orb = occ_a[iocc];
            for (jocc=iocc+1; jocc<nelec_a; jocc++) {
                occ_orb_2 = occ_a[jocc];
                sum += eri_aaaa_s8[index_4d(occ_orb, occ_orb, occ_orb_2, occ_orb_2)]-eri_aaaa_s8[index_4d(occ_orb, occ_orb_2, occ_orb_2, occ_orb)];
            }
        }
        // Contribution from eri_bbbb_s8
        for (iocc=0; iocc<nelec_b-1; iocc++) {
            occ_orb = occ_b[iocc];
            for (jocc=iocc+1; jocc<nelec_b; jocc++) {
                occ_orb_2 = occ_b[jocc];
                sum += eri_bbbb_s8[index_4d(occ_orb, occ_orb, occ_orb_2, occ_orb_2)]-eri_bbbb_s8[index_4d(occ_orb, occ_orb_2, occ_orb_2, occ_orb)];
            }
        }
        // Contribution from eri_aabb_s4
        for (iocc=0; iocc<nelec_a; iocc++) {
            occ_orb = occ_a[iocc];
            for (jocc=0; jocc<nelec_b; jocc++) {
                occ_orb_2 = occ_b[jocc];
                sum += eri_aabb_s4[index_2d(occ_orb, occ_orb)*ncols+index_2d(occ_orb_2, occ_orb_2)];
            }
        }
        return sum;
}

DiffType get_diff_type(size_t *occ_1, size_t *occ_2, size_t *one_min_two, size_t *one_min_two_indices, size_t *two_min_one, size_t *two_min_one_indices, size_t nocc) {
    size_t i_1_min_2 = 0, i_2_min_1 = 0, iocc_1 = 0, iocc_2 = 0;
    while ((iocc_1 < nocc) && (iocc_2 < nocc)) {
        size_t occ_orb_1 = occ_1[iocc_1];
        size_t occ_orb_2 = occ_2[iocc_2];
        if (occ_orb_1 < occ_orb_2) {
            if (i_1_min_2 == 2) {
                return THREE_PLUS;
            }
            one_min_two[i_1_min_2] = occ_orb_1;
            one_min_two_indices[i_1_min_2] = iocc_1;
            iocc_1++;
            i_1_min_2++;
        } else if (occ_orb_2 < occ_orb_1) {
            if (i_2_min_1 == 2) {
                return THREE_PLUS;
            }
            two_min_one[i_2_min_1] = occ_orb_2;
            two_min_one_indices[i_2_min_1] = iocc_2;
            iocc_2++;
            i_2_min_1++;
        } else {
            iocc_1++;
            iocc_2++;
        }
    }
    if ((iocc_1 == nocc) && (iocc_2 < nocc)) {
        size_t nrem = nocc-iocc_2;
        if ((i_2_min_1 + nrem) > 2) {
            return THREE_PLUS;
        } else {
            while (iocc_2 < nocc) {
                two_min_one[i_2_min_1] = occ_2[iocc_2];
                two_min_one_indices[i_2_min_1] = iocc_2;
                iocc_2++;
                i_2_min_1++;
            }
        }
    } else if ((iocc_1 < nocc) && (iocc_2 == nocc)) {
        size_t nrem = nocc-iocc_1;
        if ((i_1_min_2 + nrem) > 2) {
            return THREE_PLUS;
        } else {
            while (iocc_1 < nocc) {
                one_min_two[i_1_min_2] = occ_1[iocc_1];
                one_min_two_indices[i_1_min_2] = iocc_1;
                iocc_1++;
                i_1_min_2++;
            }
        }
    }
    assert(i_1_min_2 == i_2_min_1);
    switch (i_1_min_2) {
        case 0:
            return ZERO;
        case 1:
            return SINGLE;
        case 2:
            return DOUBLE;
        default:
            return THREE_PLUS;
    }
}

static void sort_changing_orbs(size_t *orb_list, size_t *one_min_two, size_t *two_min_one) {
    if (one_min_two[0] < two_min_one[0]) {
        orb_list[0] = one_min_two[0];
        if (two_min_one[0] < one_min_two[1]) {
            orb_list[1] = two_min_one[0];
            if (one_min_two[1] < two_min_one[1]) {
                orb_list[2] = one_min_two[1];
                orb_list[3] = two_min_one[1];
            } else {
                orb_list[2] = two_min_one[1];
                orb_list[3] = one_min_two[1];
            }
        } else {
            orb_list[1] = one_min_two[1];
            orb_list[2] = two_min_one[0];
            orb_list[3] = two_min_one[1];
        }
    } else {
        orb_list[0] = two_min_one[0];
        if (one_min_two[0] < two_min_one[1]) {
            orb_list[1] = one_min_two[0];
            if (two_min_one[1] < one_min_two[1]) {
                orb_list[2] = two_min_one[1];
                orb_list[3] = one_min_two[1];
            } else {
                orb_list[2] = one_min_two[1];
                orb_list[3] = two_min_one[1];
            }
        } else {
            orb_list[1] = two_min_one[1];
            orb_list[2] = one_min_two[0];
            orb_list[3] = one_min_two[1];
        }
    }
}

double get_matrix_element_by_rank_test_storage(uint64_t ranka_1, uint64_t rankb_1, uint64_t ranka_2, uint64_t rankb_2, 
    uint64_t *config_table_a, uint64_t *config_table_b, uint64_t *exc_table_4o, uint64_t *exc_table_2o,
    size_t norb, size_t nelec_a, size_t nelec_b,
    DoubleExcitationEntry *ordered_doubles_aa, DoubleExcitationEntry *ordered_doubles_bb, MixedExcitationEntry *ordered_mixed_ab,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4) {
        size_t occ_a_1[nelec_a], occ_b_1[nelec_b], occ_a_2[nelec_a], occ_b_2[nelec_b];
        size_t occ_a_1_min_2[2], occ_a_1_min_2_indices[2], occ_a_2_min_1[2], occ_a_2_min_1_indices[2];
        size_t occ_b_1_min_2[2], occ_b_1_min_2_indices[2], occ_b_2_min_1[2], occ_b_2_min_1_indices[2];
        unrank(ranka_1, occ_a_1, config_table_a, norb, nelec_a);
        unrank(ranka_2, occ_a_2, config_table_a, norb, nelec_a);
        DiffType res_a = get_diff_type(occ_a_1, occ_a_2, occ_a_1_min_2, occ_a_1_min_2_indices,
             occ_a_2_min_1, occ_a_2_min_1_indices, nelec_a);
        switch (res_a) {
            case ZERO: {
                unrank(rankb_1, occ_b_1, config_table_b, norb, nelec_b);
                unrank(rankb_2, occ_b_2, config_table_b, norb, nelec_b);
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, occ_b_1_min_2, occ_b_1_min_2_indices,
                    occ_b_2_min_1, occ_b_2_min_1_indices, nelec_b);
                switch (res_b) {
                    case ZERO:
                    // Diagonal
                        return get_diag_value(occ_a_1, occ_b_1, norb, nelec_a, nelec_b,
                            h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4);
                    case SINGLE:
                    // Single b
                        return get_single_excitation_value_b(occ_b_1_min_2[0], occ_b_2_min_1[0], norb, nelec_a, nelec_b, occ_a_1, occ_b_1,
                            h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4);
                    case DOUBLE: {
                    // Double bb
                        size_t exc_label[4];
                        size_t exc_rank;
                        sort_changing_orbs(exc_label, occ_b_1_min_2, occ_b_2_min_1);
                        exc_rank = rank(exc_label, exc_table_4o, norb, 4);
                        return get_double_excitation_value_from_store(ordered_doubles_bb[exc_rank], occ_b_2_min_1, occ_b_1_min_2, 
                            occ_b_1_min_2_indices, occ_b_2_min_1_indices);
                    }
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case SINGLE: {
                unrank(rankb_1, occ_b_1, config_table_b, norb, nelec_b);
                unrank(rankb_2, occ_b_2, config_table_b, norb, nelec_b);
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, occ_b_1_min_2, occ_b_1_min_2_indices,
                    occ_b_2_min_1, occ_b_2_min_1_indices, nelec_b);
                switch (res_b) {
                    case ZERO:
                    // Single a
                        return get_single_excitation_value_a(occ_a_1_min_2[0], occ_a_2_min_1[0], norb, nelec_a, nelec_b, occ_a_1, occ_b_1,
                            h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4);
                    case SINGLE: {
                    // Mixed ab
                        size_t exc_label[4];
                        if (occ_a_1_min_2[0] < occ_a_2_min_1[0]) {
                            exc_label[0] = occ_a_1_min_2[0];
                            exc_label[1] = occ_a_2_min_1[0];
                        } else {
                            exc_label[0] = occ_a_2_min_1[0];
                            exc_label[1] = occ_a_1_min_2[0];
                        }
                        if (occ_b_1_min_2[0] < occ_b_2_min_1[0]) {
                            exc_label[2] = occ_b_1_min_2[0];
                            exc_label[3] = occ_b_2_min_1[0];
                        } else {
                            exc_label[2] = occ_b_2_min_1[0];
                            exc_label[3] = occ_b_1_min_2[0];
                        }
                        size_t exc_rank = rank_mixed(exc_label, exc_table_2o, norb);
                        return get_mixed_excitation_value_from_store(ordered_mixed_ab[exc_rank], occ_a_1_min_2_indices[0], occ_a_2_min_1_indices[0],
                            occ_b_1_min_2_indices[0], occ_b_2_min_1_indices[0]);
                    }
                    case DOUBLE:
                        return 0.0;
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case DOUBLE: {
                unrank(rankb_1, occ_b_1, config_table_b, norb, nelec_b);
                unrank(rankb_2, occ_b_2, config_table_b, norb, nelec_b);
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, occ_b_1_min_2, occ_b_1_min_2_indices,
                    occ_b_2_min_1, occ_b_2_min_1_indices, nelec_b);
                switch (res_b) {
                    case ZERO: {
                    // Double aa
                        size_t exc_label[4];
                        size_t exc_rank;
                        sort_changing_orbs(exc_label, occ_a_1_min_2, occ_a_2_min_1);
                        exc_rank = rank(exc_label, exc_table_4o, norb, 4);
                        return get_double_excitation_value_from_store(ordered_doubles_aa[exc_rank], occ_a_2_min_1, occ_a_1_min_2, 
                            occ_a_1_min_2_indices, occ_a_2_min_1_indices);
                    }
                    case SINGLE:
                        return 0.0;
                    case DOUBLE:
                        return 0.0;
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case THREE_PLUS:
                return 0.0;
        }
        return 0.0;
}

double get_matrix_element_by_rank(uint64_t ranka_1, uint64_t rankb_1, uint64_t ranka_2, uint64_t rankb_2,
    uint64_t *config_table_a, uint64_t *config_table_b, size_t norb, size_t nelec_a, size_t nelec_b,
    double *h1e_aa, double *h1e_bb, double *eri_aaaa_s8, double *eri_bbbb_s8, double *eri_aabb_s4) {
        size_t occ_a_1[nelec_a], occ_b_1[nelec_b], occ_a_2[nelec_a], occ_b_2[nelec_b];
        size_t occ_a_1_min_2[2], occ_a_1_min_2_indices[2], occ_a_2_min_1[2], occ_a_2_min_1_indices[2];
        size_t occ_b_1_min_2[2], occ_b_1_min_2_indices[2], occ_b_2_min_1[2], occ_b_2_min_1_indices[2];
        unrank(ranka_1, occ_a_1, config_table_a, norb, nelec_a);
        unrank(ranka_2, occ_a_2, config_table_a, norb, nelec_a);
        DiffType res_a = get_diff_type(occ_a_1, occ_a_2, occ_a_1_min_2, occ_a_1_min_2_indices,
             occ_a_2_min_1, occ_a_2_min_1_indices, nelec_a);
        switch (res_a) {
            case ZERO: {
                unrank(rankb_1, occ_b_1, config_table_b, norb, nelec_b);
                unrank(rankb_2, occ_b_2, config_table_b, norb, nelec_b);
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, occ_b_1_min_2, occ_b_1_min_2_indices,
                    occ_b_2_min_1, occ_b_2_min_1_indices, nelec_b);
                switch (res_b) {
                    case ZERO:
                    // Diagonal
                        return get_diag_value(occ_a_1, occ_b_1, norb, nelec_a, nelec_b,
                            h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4);
                    case SINGLE:
                    // Single b
                        return get_single_excitation_value_b(occ_b_1_min_2[0], occ_b_2_min_1[0], norb, nelec_a, nelec_b, occ_a_1, occ_b_1,
                            h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4);
                    case DOUBLE: 
                    // Double bb
                        return get_double_excitation_value_bb(occ_b_1_min_2, occ_b_2_min_1, 
                            occ_b_1_min_2_indices, occ_b_2_min_1_indices, eri_bbbb_s8);
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case SINGLE: {
                unrank(rankb_1, occ_b_1, config_table_b, norb, nelec_b);
                unrank(rankb_2, occ_b_2, config_table_b, norb, nelec_b);
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, occ_b_1_min_2, occ_b_1_min_2_indices,
                    occ_b_2_min_1, occ_b_2_min_1_indices, nelec_b);
                switch (res_b) {
                    case ZERO:
                    // Single a
                        return get_single_excitation_value_a(occ_a_1_min_2[0], occ_a_2_min_1[0], norb, nelec_a, nelec_b, occ_a_1, occ_b_1,
                            h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4);
                    case SINGLE: 
                    // Mixed ab
                        return get_mixed_excitation_value(occ_a_1_min_2, occ_a_2_min_1, occ_a_1_min_2_indices, occ_a_2_min_1_indices,
                            occ_b_1_min_2, occ_b_2_min_1, occ_b_1_min_2_indices, occ_b_2_min_1_indices, norb, eri_aabb_s4);
                    case DOUBLE:
                        return 0.0;
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case DOUBLE: {
                unrank(rankb_1, occ_b_1, config_table_b, norb, nelec_b);
                unrank(rankb_2, occ_b_2, config_table_b, norb, nelec_b);
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, occ_b_1_min_2, occ_b_1_min_2_indices,
                    occ_b_2_min_1, occ_b_2_min_1_indices, nelec_b);
                switch (res_b) {
                    case ZERO: 
                        return get_double_excitation_value_aa(occ_a_1_min_2, occ_a_2_min_1, 
                            occ_a_1_min_2_indices, occ_a_2_min_1_indices, eri_aaaa_s8);
                    case SINGLE:
                        return 0.0;
                    case DOUBLE:
                        return 0.0;
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case THREE_PLUS:
                return 0.0;
        }
        return 0.0;
}