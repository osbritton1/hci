#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#define MAX_PRECOMP 64

bool get_combs(uint64_t *comb_list, int norb, int nocc);
int rank(uint64_t *occ_list, int norb, int nocc);
int unrank(int rank, int norb, int nocc);