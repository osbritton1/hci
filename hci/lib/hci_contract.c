/**
 * \file hci_contract.c
 * \addtogroup contract
 * @{
 */

#include "hci_contract.h"
#include <math.h>
#include <assert.h>

typedef enum {
    ZERO,
    SINGLE,
    DOUBLE,
    THREE_PLUS
} DiffType;

typedef enum {
    IJKL,
    IJLK,
    IKJL,
    IKLJ,
    ILJK,
    ILKJ
} DESIGNATOR;

static void sort_changing_orbs(size_t *orb_list, const size_t *old_orbs, const size_t *new_orbs) {
    if (old_orbs[0] < new_orbs[0]) {
        orb_list[0] = old_orbs[0];
        if (new_orbs[0] < old_orbs[1]) {
            orb_list[1] = new_orbs[0];
            if (old_orbs[1] < new_orbs[1]) {
                orb_list[2] = old_orbs[1];
                orb_list[3] = new_orbs[1];
            } else {
                orb_list[2] = new_orbs[1];
                orb_list[3] = old_orbs[1];
            }
        } else {
            orb_list[1] = old_orbs[1];
            orb_list[2] = new_orbs[0];
            orb_list[3] = new_orbs[1];
        }
    } else {
        orb_list[0] = new_orbs[0];
        if (old_orbs[0] < new_orbs[1]) {
            orb_list[1] = old_orbs[0];
            if (new_orbs[1] < old_orbs[1]) {
                orb_list[2] = new_orbs[1];
                orb_list[3] = old_orbs[1];
            } else {
                orb_list[2] = old_orbs[1];
                orb_list[3] = new_orbs[1];
            }
        } else {
            orb_list[1] = new_orbs[1];
            orb_list[2] = old_orbs[0];
            orb_list[3] = old_orbs[1];
        }
    }
}

double get_diag_value(const size_t *occ_a, const size_t *occ_b, const ConfigInfo *config_info,
    const HCore *h1e, const ERITensor *eri_mo) {
        size_t norb = config_info->norb;
        size_t nelec_a = config_info->nelec_a;
        size_t nelec_b = config_info->nelec_b;
        size_t ncols = eri_mo->ncols_aabb;
        double sum = 0.0;

        // Contribution from h1e_aa
        for (size_t iocc=0; iocc<nelec_a; iocc++) {
            size_t occ_orb = occ_a[iocc];
            sum += h1e->h1e_mo_aa[(occ_orb*norb)+occ_orb];
        }
        // Contribution from h1e_bb
        for (size_t iocc=0; iocc<nelec_b; iocc++) {
            size_t occ_orb = occ_b[iocc];
            sum += h1e->h1e_mo_bb[(occ_orb*norb)+occ_orb];
        }
        // Contribution from eri_aaaa_s8
        for (size_t iocc=0; iocc<nelec_a-1; iocc++) {
            size_t occ_orb = occ_a[iocc];
            for (size_t jocc=iocc+1; jocc<nelec_a; jocc++) {
                size_t occ_orb_2 = occ_a[jocc];
                sum += eri_mo->eri_mo_aaaa_s8[index_8d(occ_orb, occ_orb, occ_orb_2, occ_orb_2)]
                      -eri_mo->eri_mo_aaaa_s8[index_8d(occ_orb, occ_orb_2, occ_orb_2, occ_orb)];
            }
        }
        // Contribution from eri_bbbb_s8
        for (size_t iocc=0; iocc<nelec_b-1; iocc++) {
            size_t occ_orb = occ_b[iocc];
            for (size_t jocc=iocc+1; jocc<nelec_b; jocc++) {
                size_t occ_orb_2 = occ_b[jocc];
                sum += eri_mo->eri_mo_bbbb_s8[index_8d(occ_orb, occ_orb, occ_orb_2, occ_orb_2)]
                      -eri_mo->eri_mo_bbbb_s8[index_8d(occ_orb, occ_orb_2, occ_orb_2, occ_orb)];
            }
        }
        // Contribution from eri_aabb_s4
        for (size_t iocc=0; iocc<nelec_a; iocc++) {
            size_t occ_orb = occ_a[iocc];
            double *row = eri_mo->eri_mo_aabb_s4+(index_2d(occ_orb, occ_orb)*ncols);
            for (size_t jocc=0; jocc<nelec_b; jocc++) {
                size_t occ_orb_2 = occ_b[jocc];
                sum += row[index_2d(occ_orb_2, occ_orb_2)];
            }
        }
        return sum;
}

double get_single_exc_value_a(const ExcResult *single_exc, const size_t *occ_a, const size_t *occ_b,
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo) {
        size_t norb = config_info->norb;
        size_t nelec_a = config_info->nelec_a;
        size_t nelec_b = config_info->nelec_b;
        size_t ncols = eri_mo->ncols_aabb;
        size_t old_orb = single_exc->old_orbs[0];
        size_t new_orb = single_exc->new_orbs[0];
        double sum = 0.0;
        double *row = eri_mo->eri_mo_aabb_s4+(index_2d(old_orb, new_orb)*ncols);

        // Contribution from aaaa block
        for (size_t iocc=0; iocc<nelec_a; iocc++) {
            size_t occ_orb_2 = occ_a[iocc];
            sum += eri_mo->eri_mo_aaaa_s8[index_8d(old_orb, new_orb, occ_orb_2, occ_orb_2)]
                  -eri_mo->eri_mo_aaaa_s8[index_8d(old_orb, occ_orb_2, occ_orb_2, new_orb)];
        }
        // Contribution from aabb block
        for (size_t iocc=0; iocc<nelec_b; iocc++) {
            size_t occ_orb_2 = occ_b[iocc];
            sum += row[index_2d(occ_orb_2, occ_orb_2)];
        }
        // Contribution from 1e Hamiltonian
        sum += h1e->h1e_mo_aa[(old_orb*norb)+new_orb];

        return single_exc->sign*sum;
}

double get_single_exc_value_b(const ExcResult *single_exc, const size_t *occ_a, const size_t *occ_b,
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo) {
        size_t norb = config_info->norb;
        size_t nelec_a = config_info->nelec_a;
        size_t nelec_b = config_info->nelec_b;
        size_t ncols = eri_mo->ncols_aabb;
        size_t old_orb = single_exc->old_orbs[0];
        size_t new_orb = single_exc->new_orbs[0];
        double sum = 0.0;
        size_t col = index_2d(old_orb, new_orb);

        // Contribution from bbbb block
        for (size_t iocc=0; iocc<nelec_b; iocc++) {
            size_t occ_orb_2 = occ_b[iocc];
            sum += eri_mo->eri_mo_bbbb_s8[index_8d(old_orb, new_orb, occ_orb_2, occ_orb_2)]
                  -eri_mo->eri_mo_bbbb_s8[index_8d(old_orb, occ_orb_2, occ_orb_2, new_orb)];
        }
        // Contribution from aabb block
        for (size_t iocc=0; iocc<nelec_a; iocc++) {
            size_t occ_orb_2 = occ_a[iocc];
            sum += eri_mo->eri_mo_aabb_s4[(index_2d(occ_orb_2, occ_orb_2)*ncols)+col];
        }
        // Contribution from 1e Hamiltonian
        sum += h1e->h1e_mo_bb[(old_orb*norb)+new_orb];

        return single_exc->sign*sum;
}

double get_double_exc_value_aa(const ExcResult *double_exc, const ERITensor *eri_mo) {
    size_t *old_orbs = double_exc->old_orbs;
    size_t *new_orbs = double_exc->new_orbs;
    return double_exc->sign*(eri_mo->eri_mo_aaaa_s8[index_8d(old_orbs[0], new_orbs[0], old_orbs[1], new_orbs[1])]
                            -eri_mo->eri_mo_aaaa_s8[index_8d(old_orbs[0], new_orbs[1], old_orbs[1], new_orbs[0])]);
}

double get_double_exc_value_bb(const ExcResult *double_exc, const ERITensor *eri_mo) {
    size_t *old_orbs = double_exc->old_orbs;
    size_t *new_orbs = double_exc->new_orbs;
    return double_exc->sign*(eri_mo->eri_mo_bbbb_s8[index_8d(old_orbs[0], new_orbs[0], old_orbs[1], new_orbs[1])]
                            -eri_mo->eri_mo_bbbb_s8[index_8d(old_orbs[0], new_orbs[1], old_orbs[1], new_orbs[0])]);
}

double get_double_exc_value_from_store(const DoubleExcEntry *exc_entry, const ExcResult *double_exc) {
    size_t *old_orbs = double_exc->old_orbs;
    size_t *new_orbs = double_exc->new_orbs;
    size_t orb_list[4];
    uint8_t designator = 0;
    if (old_orbs[0] <= new_orbs[0]) {
        orb_list[0] = old_orbs[0];
        orb_list[1] = new_orbs[0];
        orb_list[2] = old_orbs[1];
        orb_list[3] = new_orbs[1];
    } else {
        orb_list[0] = new_orbs[0];
        orb_list[1] = old_orbs[0];
        orb_list[2] = new_orbs[1];
        orb_list[3] = old_orbs[1];
    }
    designator += (orb_list[1] > orb_list[2]) ? 2 : 0;
    designator += (orb_list[1] > orb_list[3]) ? 2 : 0;
    designator += (orb_list[2] > orb_list[3]) ? 1 : 0;
    switch (designator) {
        case IJKL:
            return exc_entry->ijkl*double_exc->sign;
        case IJLK:
            return (exc_entry->ijkl+exc_entry->iljk)*double_exc->sign;
        case IKJL:
            return -exc_entry->iljk*double_exc->sign;
        case IKLJ:
            return -(exc_entry->ijkl+exc_entry->iljk)*double_exc->sign;
        case ILJK:
            return exc_entry->iljk*double_exc->sign;
        case ILKJ:
            return -exc_entry->ijkl*double_exc->sign;
        default:
            return nan("");
    }
}

double get_mixed_exc_value(const ExcResult *single_exc_a, const ExcResult *single_exc_b, 
    const ERITensor *eri_mo) {
        double sign = single_exc_a->sign*single_exc_b->sign;
        size_t row = index_2d(single_exc_a->old_orbs[0], single_exc_a->new_orbs[0]);
        size_t col = index_2d(single_exc_b->old_orbs[0], single_exc_b->new_orbs[0]);
        return sign*eri_mo->eri_mo_aabb_s4[(row*eri_mo->ncols_aabb)+col];
}

double get_mixed_exc_value_from_store(const MixedExcEntry *exc_entry, 
    const ExcResult *single_exc_a, const ExcResult *single_exc_b) {
        double sign = single_exc_a->sign * single_exc_b->sign;
        return sign*exc_entry->ijkl;
}

DiffType get_diff_type(const size_t *occ_1, const size_t *occ_2, size_t nocc, ExcResult *exc_result) {
    size_t iold = 0;
    size_t inew = 0;
    size_t iocc_1 = 0;
    size_t iocc_2 = 0;
    double sign = 1.0;
    while ((iocc_1 < nocc) && (iocc_2 < nocc)) {
        size_t occ_orb_1 = occ_1[iocc_1];
        size_t occ_orb_2 = occ_2[iocc_2];
        if (occ_orb_1 < occ_orb_2) {
            if (iold == 2) {
                return THREE_PLUS;
            }
            exc_result->old_orbs[iold] = occ_orb_1;
            sign *= ((iocc_1 - iold) % 2 == 0) ? 1.0 : -1.0;
            iocc_1++;
            iold++;
        } else if (occ_orb_2 < occ_orb_1) {
            if (inew == 2) {
                return THREE_PLUS;
            }
            exc_result->new_orbs[inew] = occ_orb_2;
            sign *= ((iocc_2 - inew) % 2 == 0) ? 1.0 : -1.0;
            iocc_2++;
            inew++;
        } else {
            iocc_1++;
            iocc_2++;
        }
    }
    if ((iocc_1 == nocc) && (iocc_2 < nocc)) {
        size_t nrem = nocc-iocc_2;
        if ((inew + nrem) > 2) {
            return THREE_PLUS;
        } else {
            sign *= (((iocc_2 - inew) % 2 == 0) || (nrem % 2 == 0)) ? 1.0 : -1.0;
            while (iocc_2 < nocc) {
                exc_result->new_orbs[inew] = occ_2[iocc_2];
                iocc_2++;
                inew++;
            }
        }
    } else if ((iocc_1 < nocc) && (iocc_2 == nocc)) {
        size_t nrem = nocc-iocc_1;
        if ((iold + nrem) > 2) {
            return THREE_PLUS;
        } else {
            sign *= (((iocc_1 - iold) % 2 == 0) || (nrem % 2 == 0)) ? 1.0 : -1.0;
            while (iocc_1 < nocc) {
                exc_result->old_orbs[iold] = occ_1[iocc_1];
                iocc_1++;
                iold++;
            }
        }
    }
    assert(iold == inew);
    exc_result->sign = sign;
    switch (iold) {
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

double get_matrix_element_by_rank(Rank rank1, Rank rank2, 
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo) {
        size_t nelec_a = config_info->nelec_a;
        size_t nelec_b = config_info->nelec_b;

        size_t occ_a_1[nelec_a];
        size_t occ_a_2[nelec_a];
        unrank_occ_a(rank1.arank, occ_a_1, config_info);
        unrank_occ_a(rank2.arank, occ_a_2, config_info);
        ExcResult exc_a = NEW_DOUBLE_EXC_RESULT();
        DiffType res_a = get_diff_type(occ_a_1, occ_a_2, nelec_a, &exc_a);
        switch (res_a) {
            case ZERO: {
                size_t occ_b_1[nelec_b];
                size_t occ_b_2[nelec_b];
                unrank_occ_b(rank1.brank, occ_b_1, config_info);
                unrank_occ_b(rank2.brank, occ_b_2, config_info);
                ExcResult exc_b = NEW_DOUBLE_EXC_RESULT();
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, nelec_b, &exc_b);
                switch (res_b) {
                    case ZERO:
                    // Diagonal
                        return get_diag_value(occ_a_1, occ_b_1, 
                            config_info, h1e, eri_mo);
                    case SINGLE:
                    // Single b
                        return get_single_exc_value_b(&exc_b, occ_a_1, occ_b_1, 
                            config_info, h1e, eri_mo);
                    case DOUBLE: 
                    // Double bb
                        return get_double_exc_value_bb(&exc_b, eri_mo);
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case SINGLE: {
                size_t occ_b_1[nelec_b];
                size_t occ_b_2[nelec_b];
                unrank_occ_b(rank1.brank, occ_b_1, config_info);
                unrank_occ_b(rank2.brank, occ_b_2, config_info);
                ExcResult exc_b = NEW_DOUBLE_EXC_RESULT();
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, nelec_b, &exc_b);
                switch (res_b) {
                    case ZERO:
                    // Single a
                        return get_single_exc_value_a(&exc_a, occ_a_1, occ_b_1, 
                            config_info, h1e, eri_mo);
                    case SINGLE: 
                    // Mixed ab
                        return get_mixed_exc_value(&exc_a, &exc_b, eri_mo);
                    case DOUBLE:
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case DOUBLE: {
                size_t occ_b_1[nelec_b];
                size_t occ_b_2[nelec_b];
                unrank_occ_b(rank1.brank, occ_b_1, config_info);
                unrank_occ_b(rank2.brank, occ_b_2, config_info);
                ExcResult exc_b = NEW_DOUBLE_EXC_RESULT();
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, nelec_b, &exc_b);
                switch (res_b) {
                    case ZERO: 
                        return get_double_exc_value_aa(&exc_a, eri_mo);
                    case SINGLE:
                    case DOUBLE:
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case THREE_PLUS:
                return 0.0;
        }
        return 0.0;
}

double get_matrix_element_by_partial_rank(uint64_t *occ_a_1, uint64_t *occ_b_1, Rank rank2,
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo) {
        size_t nelec_a = config_info->nelec_a;
        size_t nelec_b = config_info->nelec_b;

        size_t occ_a_2[nelec_a];
        unrank_occ_a(rank2.arank, occ_a_2, config_info);
        ExcResult exc_a = NEW_DOUBLE_EXC_RESULT();
        DiffType res_a = get_diff_type(occ_a_1, occ_a_2, nelec_a, &exc_a);
        switch (res_a) {
            case ZERO: {
                size_t occ_b_2[nelec_b];
                unrank_occ_b(rank2.brank, occ_b_2, config_info);
                ExcResult exc_b = NEW_DOUBLE_EXC_RESULT();
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, nelec_b, &exc_b);
                switch (res_b) {
                    case ZERO:
                    // Diagonal
                        return get_diag_value(occ_a_1, occ_b_1, 
                            config_info, h1e, eri_mo);
                    case SINGLE:
                    // Single b
                        return get_single_exc_value_b(&exc_b, occ_a_1, occ_b_1, 
                            config_info, h1e, eri_mo);
                    case DOUBLE: 
                    // Double bb
                        return get_double_exc_value_bb(&exc_b, eri_mo);
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case SINGLE: {
                size_t occ_b_2[nelec_b];
                unrank_occ_b(rank2.brank, occ_b_2, config_info);
                ExcResult exc_b = NEW_DOUBLE_EXC_RESULT();
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, nelec_b, &exc_b);
                switch (res_b) {
                    case ZERO:
                    // Single a
                        return get_single_exc_value_a(&exc_a, occ_a_1, occ_b_1, 
                            config_info, h1e, eri_mo);
                    case SINGLE: 
                    // Mixed ab
                        return get_mixed_exc_value(&exc_a, &exc_b, eri_mo);
                    case DOUBLE:
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case DOUBLE: {
                size_t occ_b_2[nelec_b];
                unrank_occ_b(rank2.brank, occ_b_2, config_info);
                ExcResult exc_b = NEW_DOUBLE_EXC_RESULT();
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, nelec_b, &exc_b);
                switch (res_b) {
                    case ZERO: 
                        return get_double_exc_value_aa(&exc_a, eri_mo);
                    case SINGLE:
                    case DOUBLE:
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case THREE_PLUS:
                return 0.0;
        }
        return 0.0;
}

double get_matrix_element_by_rank_test_storage(Rank rank1, Rank rank2, 
    const ConfigInfo *config_info, const ExcEntries *excitation_entries,
    const HCore *h1e, const ERITensor *eri_mo) {
        size_t nelec_a = config_info->nelec_a;
        size_t nelec_b = config_info->nelec_b;

        size_t occ_a_1[nelec_a];
        size_t occ_a_2[nelec_a];
        unrank_occ_a(rank1.arank, occ_a_1, config_info);
        unrank_occ_a(rank2.arank, occ_a_2, config_info);
        ExcResult exc_a = NEW_DOUBLE_EXC_RESULT();
        DiffType res_a = get_diff_type(occ_a_1, occ_a_2, nelec_a, &exc_a);
        switch (res_a) {
            case ZERO: {
                size_t occ_b_1[nelec_b];
                size_t occ_b_2[nelec_b];
                unrank_occ_b(rank1.brank, occ_b_1, config_info);
                unrank_occ_b(rank2.brank, occ_b_2, config_info);
                ExcResult exc_b = NEW_DOUBLE_EXC_RESULT();
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, nelec_b, &exc_b);
                switch (res_b) {
                    case ZERO:
                    // Diagonal
                        return get_diag_value(occ_a_1, occ_b_1, 
                            config_info, h1e, eri_mo);
                    case SINGLE:
                    // Single b
                        return get_single_exc_value_b(&exc_b, occ_a_1, occ_b_1, 
                            config_info, h1e, eri_mo);
                    case DOUBLE: {
                    // Double bb
                        size_t exc_label[4];
                        sort_changing_orbs(exc_label, exc_b.old_orbs, exc_b.new_orbs);
                        size_t exc_rank = rank_double_exc(exc_label, config_info);
                        return get_double_exc_value_from_store(excitation_entries->doubles_bb+exc_rank, &exc_b);
                    }
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case SINGLE: {
                size_t occ_b_1[nelec_b];
                size_t occ_b_2[nelec_b];
                unrank_occ_b(rank1.brank, occ_b_1, config_info);
                unrank_occ_b(rank2.brank, occ_b_2, config_info);
                ExcResult exc_b = NEW_DOUBLE_EXC_RESULT();
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, nelec_b, &exc_b);
                switch (res_b) {
                    case ZERO:
                    // Single a
                        return get_single_exc_value_a(&exc_a, occ_a_1, occ_b_1, 
                            config_info, h1e, eri_mo);
                    case SINGLE: {
                    // Mixed ab
                        size_t exc_label[4];
                        if (exc_a.old_orbs[0] < exc_a.new_orbs[0]) {
                            exc_label[0] = exc_a.old_orbs[0];
                            exc_label[1] = exc_a.new_orbs[0];
                        } else {
                            exc_label[0] = exc_a.new_orbs[0];
                            exc_label[1] = exc_a.old_orbs[0];
                        }
                        if (exc_b.old_orbs[0] < exc_b.new_orbs[0]) {
                            exc_label[2] = exc_b.old_orbs[0];
                            exc_label[3] = exc_b.new_orbs[0];
                        } else {
                            exc_label[2] = exc_b.new_orbs[0];
                            exc_label[3] = exc_b.old_orbs[0];
                        }
                        size_t exc_rank = rank_mixed_exc(exc_label, config_info);
                        return get_mixed_exc_value_from_store(excitation_entries->mixed_ab+exc_rank, &exc_a, &exc_b);
                    }
                    case DOUBLE:
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case DOUBLE: {
                size_t occ_b_1[nelec_b];
                size_t occ_b_2[nelec_b];
                unrank_occ_b(rank1.brank, occ_b_1, config_info);
                unrank_occ_b(rank2.brank, occ_b_2, config_info);
                ExcResult exc_b = NEW_DOUBLE_EXC_RESULT();
                DiffType res_b = get_diff_type(occ_b_1, occ_b_2, nelec_b, &exc_b);
                switch (res_b) {
                    case ZERO: {
                        size_t exc_label[4];
                        sort_changing_orbs(exc_label, exc_a.old_orbs, exc_a.new_orbs);
                        size_t exc_rank = rank_double_exc(exc_label, config_info);
                        return get_double_exc_value_from_store(excitation_entries->doubles_aa+exc_rank, &exc_a);
                    }
                    case SINGLE:
                    case DOUBLE:
                    case THREE_PLUS:
                        return 0.0;
                }
            }
            case THREE_PLUS:
                return 0.0;
        }
        return 0.0;
}

void make_hdiag_slow(HCIVec *hcivec, double *hdiag,
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo) {
        size_t nelec_a = config_info->nelec_a;
        size_t nelec_b = config_info->nelec_b;
        for (size_t i=0; i<hcivec->len; i++) {
            size_t occ_a[nelec_a];
            size_t occ_b[nelec_b];
            uint64_t arank = hcivec->ranks[i].arank;
            uint64_t brank = hcivec->ranks[i].brank;
            unrank_occ_a(arank, occ_a, config_info);
            unrank_occ_b(brank, occ_b, config_info);
            hdiag[i] = get_diag_value(occ_a, occ_b, config_info, h1e, eri_mo);
        }
}

void contract_hamiltonian_hcivec_slow(HCIVec *hcivec_old, double *coeffs_new, const double *hdiag,
    const ConfigInfo *config_info, const HCore *h1e, const ERITensor *eri_mo) {
        size_t nelec_a = config_info->nelec_a;
        size_t nelec_b = config_info->nelec_b;
        Rank *ranks = hcivec_old->ranks;
        double *coeffs = hcivec_old->coeffs;
        for (size_t i=0; i<hcivec_old->len; i++) {
            double sum = 0.0;
            uint64_t bra_arank = ranks[i].arank;
            uint64_t bra_brank = ranks[i].brank;
            uint64_t bra_occ_a[nelec_a];
            uint64_t bra_occ_b[nelec_b];
            unrank_occ_a(bra_arank, bra_occ_a, config_info);
            unrank_occ_b(bra_brank, bra_occ_b, config_info);
            for (size_t j=0; j<i; j++) {
                if (coeffs[j] == 0.0) {
                    continue;
                }
                sum += get_matrix_element_by_partial_rank(bra_occ_a, bra_occ_b, ranks[j],
                    config_info, h1e, eri_mo)*coeffs[j];
            }
            sum += hdiag[i]*coeffs[i];
            for (size_t j=i+1; j<hcivec_old->len; j++) {
                if (coeffs[j] == 0.0) {
                    continue;
                }
                sum += get_matrix_element_by_partial_rank(bra_occ_a, bra_occ_b, ranks[j],
                    config_info, h1e, eri_mo)*coeffs[j];
            }
            coeffs_new[i] = sum;
        }
}

/**
 * @}
 */