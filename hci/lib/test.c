#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <time.h>
#include "hci_rank.h"
#include "hci_store.h"
#include "hci_enlarge.h"
#include "hci_contract.h"

int test_rank_unrank() {
    size_t i, j;
    size_t norb = 8;
    size_t nocc = 4;
    size_t nrows = nocc;
    size_t ncols = norb-nocc+1;
    uint64_t rank_table[nrows*ncols];
    uint64_t rank = 57;
    size_t occ_list[nocc];

    get_rank_table(rank_table, norb, nocc);

    for (i=0; i<nrows; i++) {
        uint64_t *row = rank_table+i*ncols;
        for (j=0; j<ncols; j++) {
            printf("%"PRIu64" ", row[j]);
        }
        printf("\n");
    }

    unrank(rank, occ_list, rank_table, norb, nocc);

    for (i=0; i<nocc; i++) {
        printf("%"PRIu64" ", occ_list[i]);
    }

    return 0;
}

int test_mixed_storage() {
    size_t norb = 4;
    size_t nocc = 2;
    size_t npairs = nC2(norb+1);
    size_t ncombs = nC2(norb);
    double eri_s4[npairs*npairs];
    size_t i;
    uint64_t exc_table_2o[2*(norb-2+1)];
    MixedExcitationEntry mixed[ncombs*ncombs];

    get_rank_table(exc_table_2o, norb, 2);
    srand(time(NULL));
    for (i=0; i<npairs*npairs; i++) {
        eri_s4[i] = (double)rand() / (double)RAND_MAX;
    }

    load_mixed_from_eri(mixed, eri_s4, exc_table_2o, norb);
    return 0;
}

int test_get_changing_orbitals() {
    size_t exc[8] = {0,5,6,10,12,14,17,21};
    size_t occ[11] = {1,2,6,7,10,11,13,15,17,21,28};
    size_t exc_min_occ[4];
    size_t exc_int_occ[4];
    size_t exc_int_occ_indices[4];
    size_t new_occ[11];
    size_t new_indices[4];
    size_t exact_diffs = 4;
    size_t nocc = 11;
    bool condition = get_changing_orbitals(exc, occ, exc_min_occ, exc_int_occ, new_occ, exc_int_occ_indices, new_indices, exact_diffs, nocc);
    printf("%u\n", condition);
    size_t i;
    for (i=0; i<exact_diffs; i++) {
        printf("%zu ", exc_min_occ[i]);
    }
    printf("\n");
    for (i=0; i<exact_diffs; i++) {
        printf("%zu ", exc_int_occ[i]);
    }
    printf("\n\n");
    for (i=0; i<nocc; i++) {
        printf("%zu ", occ[i]);
    }
    printf("\n");
    for (i=0; i<exact_diffs; i++) {
        printf("%zu ", exc_int_occ_indices[i]);
    }
    printf("\n\n");
    for (i=0; i<nocc; i++) {
        printf("%zu ", new_occ[i]);
    }
    printf("\n");
    for (i=0; i<exact_diffs; i++) {
        printf("%zu ", new_indices[i]);
    }
    printf("\n");
    return 0;
}

int test_enlarge_doubles_1() {
    HCIEntry hcivec[1] = {{0, 0, 1.0}};
    size_t hci_len = 1;
    uint64_t add_doubles[18];
    size_t norb = 4;
    size_t nelec_a = 1;
    size_t nelec_b = 1;
    double thresh = 0.0;
    uint64_t config_table_a[8] = {0, 1, 2, 3, 0, 1, 2, 3};
    uint64_t config_table_b[8] = {0, 1, 2, 3, 0, 1, 2, 3};
    uint64_t exc_table_4o[4] = {0, 0, 0, 0};
    uint64_t exc_table_2o[6] = {0, 1, 2, 0, 1, 3};
    DoubleExcitationEntry doubles_aa[1] = {{0, 8.96804313e-17, 0.11841575}};
    size_t ndoubles_aa = 1;
    DoubleExcitationEntry doubles_bb[1] = {{0, 2.71145682e-17, 0.11841574}};
    size_t ndoubles_bb = 1;
    MixedExcitationEntry mixed_ab[36] = 
    {{7,  4.07364481e-01}, {35,  1.37319570e-01}, {21,  1.18415744e-01},
    {15,  1.18415744e-01}, {14,  1.18415744e-01}, {20,  1.18415744e-01},
    { 0, -8.24893642e-02}, { 5,  7.90993359e-02}, {30, -7.90993352e-02},
    {13,  5.13708578e-02}, {19,  5.13708578e-02}, { 8,  5.13708558e-02},
    { 9,  5.13708558e-02}, {28, -3.68701063e-02}, { 4,  2.24259416e-02},
    {24,  2.24259388e-02}, {29,  2.00916579e-02}, {34, -2.00916563e-02},
    {10, -6.03750919e-16}, {31, -4.44124587e-16}, {25,  3.19532996e-16},
    {11, -2.86407089e-16}, {18, -1.13397637e-16}, {12, -1.13397637e-16},
    { 1, -7.66060403e-17}, { 6, -7.24393863e-17}, {22, -4.95659735e-17},
    {16, -4.95659735e-17}, {32, -4.61179287e-17}, {33, -4.61179287e-17},
    {17,  3.54508911e-17}, {23,  3.54508911e-17}, { 2, -2.81434794e-17},
    { 3, -2.81434794e-17}, {26,  4.04473209e-18}, {27,  4.04473209e-18}};
    size_t ndoubles_ab = 36;
    double max_mag_aa[1] = {0.11841575};
    double max_mag_bb[1] = {0.11841574};
    double max_mag_ab[36] = 
    {4.07364481e-01, 1.37319570e-01, 1.18415744e-01, 1.18415744e-01,
    1.18415744e-01, 1.18415744e-01, 8.24893642e-02, 7.90993359e-02,
    7.90993352e-02, 5.13708578e-02, 5.13708578e-02, 5.13708558e-02,
    5.13708558e-02, 3.68701063e-02, 2.24259416e-02, 2.24259388e-02,
    2.00916579e-02, 2.00916563e-02, 4.81518397e-16, 4.77467532e-16,
    3.72077810e-16, 3.31373010e-16, 2.04928300e-16, 2.04928300e-16,
    1.82390092e-16, 1.82390092e-16, 1.47787966e-16, 1.47787966e-16,
    1.46391649e-16, 8.97736184e-17, 8.97736184e-17, 8.14127263e-17,
    8.14127263e-17, 5.20950292e-17, 2.22462662e-17, 2.22462662e-17};
    size_t nadd = enlarge_space_doubles(hcivec, hci_len, add_doubles,
        norb, nelec_a, nelec_b, thresh,
        config_table_a, config_table_b, exc_table_4o, exc_table_2o,
        doubles_aa, ndoubles_aa, 
        doubles_bb, ndoubles_bb,
        mixed_ab, ndoubles_ab,
        max_mag_aa, max_mag_bb, max_mag_ab);
    return 0;
}

int test_enlarge_singles() {
    HCIEntry hcivec[1] = {{0, 0, 1.0}};
    size_t hci_len = 1;
    uint64_t add_singles[44];
    size_t norb = 7;
    size_t nelec_a = 5;
    size_t nelec_b = 4;
    size_t combmax_a = 21;
    size_t combmax_b = 35;
    double thresh = 0.0;
    uint64_t config_table_a[15] = {0, 1, 2, 0, 1, 3, 0, 1, 4, 0, 1, 5, 0, 1, 6};
    uint64_t config_table_a_complement[12] = {0, 1, 2, 3, 4, 5, 0, 1, 3, 6, 10, 15};
    uint64_t config_table_b[16] = {0, 1, 2, 3, 0, 1, 3, 6, 0, 1, 4, 10, 0, 1, 5, 15};
    uint64_t config_table_b_complement[15] = {0, 1, 2, 3, 4, 0, 1, 3, 6, 10, 0, 1, 4, 10, 20};
    double h1e_aa[49] = {0};
    double h1e_bb[49] = {0};
    double eri_aaaa_s8[406] = {0};
    double eri_bbbb_s8[406] = {0};
    double eri_aabb_s4[784] = {0};
    enlarge_space_singles(hcivec, hci_len, add_singles,
        norb, nelec_a, nelec_b, combmax_a, combmax_b, thresh,
        config_table_a, config_table_a_complement,
        config_table_b, config_table_b_complement,
        h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4);
    return 0;
}

int test_rank() {
    size_t i, j;
    size_t norb = 7;
    size_t nocc = 5;
    size_t nrows = nocc;
    size_t ncols = norb-nocc+1;
    uint64_t rank_table[nrows*ncols];
    size_t occ_list[5] = {0, 1, 2, 3, 4};

    get_rank_table(rank_table, norb, nocc);

    uint64_t test_rank = rank(occ_list, rank_table, norb, nocc);

    printf("%"PRIu64" ", test_rank);

    return 0;
}

// DiffType get_exc_type(size_t *occ_1, size_t *occ_2, size_t *one_min_two, size_t *two_min_one, size_t nocc)

int test_get_diff_type() {
    size_t occ_1[5] = {2, 6, 8, 12, 15};
    size_t occ_2[5] = {3, 6, 8, 12, 16};
    size_t one_min_two[2], one_min_two_indices[2], two_min_one[2], two_min_one_indices[2];
    size_t nocc = 5, i;
    DiffType result = get_diff_type(occ_1, occ_2, one_min_two, one_min_two_indices, two_min_one, two_min_one_indices, nocc);
    printf("%u\n\n", result);
    if (result < 3) {
        for (i=0; i<result; i++) {
        printf("%zu ", one_min_two[i]);
    }
        printf("\n");
        for (i=0; i<result; i++) {
            printf("%zu ", two_min_one[i]);
        }
        printf("\n");
    } else {
        printf("Three or more differences\n");
    }
    return 0;
}



int test_get_matrix_element_by_rank() {
    size_t norb = 7;
    size_t nelec_a = 5;
    size_t nelec_b = 4;
    size_t i, j, ranka_1, rankb_1, ranka_2, rankb_2;
    size_t config_table_a[21], config_table_b[35], exc_table_4o[35], exc_table_2o[21];
    DoubleExcitationEntry ordered_doubles_aa[35], ordered_doubles_bb[35];
    MixedExcitationEntry ordered_mixed_ab[441];

    get_rank_table(config_table_a, norb, nelec_a);
    get_rank_table(config_table_b, norb, nelec_b);
    get_rank_table(exc_table_4o, norb, 4);
    get_rank_table(exc_table_2o, norb, 2);

    srand(time(NULL));
    double h1e_aa[norb*norb], h1e_bb[norb*norb], eri_aaaa_s8[406], eri_aabb_s4[784], eri_bbbb_s8[406];
    for (i=0; i<norb; i++) {
        for (j=i; j<norb; j++) {
            double rand_entry = (double)rand() / (double)RAND_MAX;
            h1e_aa[i*norb+j] = rand_entry;
            h1e_aa[j*norb+i] = rand_entry;
        }
    }
    for (i=0; i<norb; i++) {
        for (j=i; j<norb; j++) {
            double rand_entry = (double)rand() / (double)RAND_MAX;
            h1e_bb[i*norb+j] = rand_entry;
            h1e_bb[j*norb+i] = rand_entry;
        }
    }
    for (i=0; i<406; i++) {
        eri_aaaa_s8[i] = (double)rand() / (double)RAND_MAX;
        eri_bbbb_s8[i] = (double)rand() / (double)RAND_MAX;
    }
    for (i=0; i<784; i++) {
        eri_aabb_s4[i] = (double)rand() / (double)RAND_MAX;
    }

    load_doubles_from_eri(ordered_doubles_aa, eri_aaaa_s8, exc_table_4o, norb);
    load_doubles_from_eri(ordered_doubles_bb, eri_aaaa_s8, exc_table_4o, norb);
    load_mixed_from_eri(ordered_mixed_ab, eri_aabb_s4, exc_table_2o, norb);

    get_matrix_element_by_rank(0, 0, 18, 0,
        config_table_a, config_table_b, exc_table_4o, exc_table_2o,
        norb, nelec_a, nelec_b,
        ordered_doubles_aa, ordered_doubles_bb, ordered_mixed_ab,
        h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4);

    // for (ranka_1=0; ranka_1<21; ranka_1++) {
    //     for (rankb_1=0; rankb_1<35; rankb_1++) {
    //         for (ranka_2=0; ranka_2<21; ranka_2++) {
    //             for (rankb_2=0; rankb_2<35; rankb_2++) {
    //                 get_matrix_element_by_rank(ranka_1, rankb_1, ranka_2, rankb_2,
    //                     config_table_a, config_table_b, exc_table_4o, exc_table_2o,
    //                     norb, nelec_a, nelec_b,
    //                     ordered_doubles_aa, ordered_doubles_bb, ordered_mixed_ab,
    //                     h1e_aa, h1e_bb, eri_aaaa_s8, eri_bbbb_s8, eri_aabb_s4);
    //             }
    //         }
    //     }
    // }

    return 0;
}

int main() {
    return test_get_matrix_element_by_rank();
}

