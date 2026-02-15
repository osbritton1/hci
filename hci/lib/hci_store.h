#ifndef HCI_STORE_H
#define HCI_STORE_H

#include <stdint.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MAX3(a, b, c) ((a) > (b) ? ((a) > (c) ? (a) : (c)) : ((b) > (c) ? (b) : (c)))

typedef struct {
    uint64_t rank;
    double ijkl;
    double iljk;
} DoubleExcitationEntry;

#endif