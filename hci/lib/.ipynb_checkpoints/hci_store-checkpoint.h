#ifndef HCI_STORE_H
#define HCI_STORE_H

#include <stdint.h>

typedef struct {
    uint64_t rank;
    double ijkl;
    double iljk;
} DoubleExcitationEntry;

void get_max_magnitudes(DoubleExcitationEntry *doubles, double *magnitudes);

#endif