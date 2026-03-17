#ifndef HCI_STORE_H
#define HCI_STORE_H

#include <stdint.h>
#include <stdlib.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX3(a, b, c) ((a) > (b) ? ((a) > (c) ? (a) : (c)) : ((b) > (c) ? (b) : (c)))

typedef struct {
    uint64_t rank;
    double ijkl;
    double iljk;
} DoubleExcitationEntry;

typedef struct {
    uint64_t rank;
    double ijkl;
} MixedExcitationEntry;

void get_max_magnitudes(DoubleExcitationEntry *doubles, double *magnitudes, size_t ndoubles);
size_t index_2d(size_t i, size_t j);
size_t index_4d(size_t i, size_t j, size_t k, size_t l);
void load_doubles_from_eri(DoubleExcitationEntry *doubles, double *eri_s8, uint64_t *index_table, size_t norb);
void load_mixed_from_eri(MixedExcitationEntry *mixed, double *eri_s4, uint64_t *index_table, size_t norb);

#endif