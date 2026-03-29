#include "hci_rank.h"

// Command to run to generate this library: gcc -shared -o libhci.so -fPIC hci_rank.c hci_store.c hci_enlarge.c hci_contract.c -Wall -g

/**
 * Initializes the nocc by (norb-nocc+1) table needed to rank and unrank combinations using the combinatorial number system.
 * The binomial coefficients needed for the encoding the position of the ith electron are stored in row i-1.
 * The entry in position (i, j) is nCr(i+j, i+1); when i+j < i+1 (first column), this is defined to be 0.
 * @param[out] table Pointer to the uninitialized ranking table; must be able to accommodate at least nocc rows by norb-nocc+1 columns
 * @param[in] norb Number of orbitals
 * @param[in] nocc Number of occupancies; generally equal to the number of alpha or beta electrons in the system
 */
void get_rank_table(uint64_t *table, size_t norb, size_t nocc) {
    size_t ncols = norb-nocc+1;
    // Edge case: no electrons
    if (nocc == 0) {
        return;
    }
    // Initialize first row of ranking table
    for (size_t j=0; j<ncols; j++) {
        table[j] = j;
    }
    // Initialize rest of ranking table using binomial coefficient recurrence relation
    for (size_t i=1; i<nocc; i++) {
        uint64_t *prev_row = table+((i-1)*ncols);
        uint64_t *curr_row = table+(i*ncols);
        curr_row[0] = 0;
        for (size_t j=1; j<ncols; j++) {
            curr_row[j] = curr_row[j-1]+prev_row[j];
        }
    }
}

/**
 * Get the combinatorial rank corresponding to the provided list of occupied orbitals.
 *
 * Formula: @f[ \sum\limits_{i=1}^{N_\text{occ}}\binom{c_i}{i} @f]
 * @param[in] occ_list Pointer to an array of length nocc specifying the zero-indexed occupied orbitals in increasing order
 * @param[in] rank_table Pointer to ranking table initialized by get_rank_table(uint64_t *table, size_t norb, size_t nocc)
 * @param[in] norb Number of orbitals
 * @param[in] nocc Number of occupancies; generally equal to the number of alpha or beta electrons in the system
 * @return The rank of the specified occupancy list
 */
uint64_t rank(const size_t *occ_list, const uint64_t *rank_table, size_t norb, size_t nocc) {
    uint64_t sum = 0;
    size_t ncols = norb-nocc+1;
    for (size_t i=0; i<nocc; i++) {
        size_t occ_orbital = occ_list[i];
        sum += rank_table[(i*ncols)+occ_orbital-i];
    }
    return sum;
}

/**
 * Helper function for unranking algorithm. Uses binary search to find the index of the last
 * entry in a row of the ranking table that is less than the value of target.
 * @param[in] target Target value used as a reference for the search
 * @param[in] row Pointer to the row of the ranking table used for the search
 * @param[in] norb Number of orbitals
 * @param[in] nocc Number of occupancies; generally equal to the number of alpha or beta electrons in the system 
 */
size_t find_row_index(uint64_t target, const uint64_t *row, size_t norb, size_t nocc) {
    size_t low = 0;
    size_t high = norb-nocc;
    while (low < high) {
        // Need to use upper midpoint to ensure proper termination when high=low+1 and else statement is reached
        size_t mid = low+((high-low+1)/2);
        if (row[mid]>target) {
            high = mid-1;
        } else {
            low = mid;
        }
    }
    return low;
}

/**
 * Greedy unranking algorithm for computing occupancy list given its combinatorial rank.
 * Starting from the last row in the ranking table (corresponding to the highest occupied orbital),
 * uses find_row_index(uint64_t target, uint64_t *row, size_t norb, size_t nocc) to find the column
 * index of the entry that does not exceed the value of target, updates the occupancy list accordingly,
 * then subtracts off the ranking table entry from target and repeats.
 * @param[in] rank The rank of the occupancy list to be computed
 * @param[out] occ_list A pointer to an array of length nocc to store the occupancy list
 * @param[in] rank_table Pointer to ranking table initialized by get_rank_table(uint64_t *table, size_t norb, size_t nocc)
 * @param[in] norb Number of orbitals
 * @param[in] nocc Number of occupancies; generally equal to the number of alpha or beta electrons in the system
 */
void unrank(uint64_t rank, size_t *occ_list, const uint64_t *rank_table, size_t norb, size_t nocc) {
    size_t ncols = norb-nocc+1;
    size_t target = rank;
    // Reverse iteration with unsigned types is a bit funky
    for (size_t i=nocc; i-->0;) {
        const uint64_t *row = rank_table+(i*ncols);
        size_t row_index = find_row_index(target, row, norb, nocc);
        target -= row[row_index];
        occ_list[i] = row_index+i;
    }
}

/**
 * Helper function for computing nCr(n,2)
 * @param[in] n
 * @return nCr(n,2)
 */
uint64_t nC2(size_t n) {
    return (n % 2 == 0) ? n/2*(n-1) : (n-1)/2*n;
}

/**
 * Calculates the rank of a mixed ab excitation.
 * @param[in] occ_list A pointer to an array of four numbers {i, j, k, l}
 * @param[in] exc_table_2o A pointer to the two-orbital excitation table
 * @param[in] norb Number of orbitals
 * @return The rank of the specified mixed ab excitation
 */
uint64_t rank_mixed(size_t *occ_list, const uint64_t *exc_table_2o, size_t norb) {
    size_t ij_list[2] = {occ_list[0], occ_list[1]};
    size_t kl_list[2] = {occ_list[2], occ_list[3]};
    uint64_t ij_rank = rank(ij_list, exc_table_2o, norb, 2);
    uint64_t kl_rank = rank(kl_list, exc_table_2o, norb, 2);
    uint64_t ncols = nC2(norb);
    return (ncols*ij_rank) + kl_rank;
}

/**
 * Determines the occupancy list corresponding to a mixed ab excitation with given rank.
 * @param[in] rank The rank of ab excitation
 * @param[out] occ_list A pointer to an array of length nocc to store the occupancy list
 * @param[in] exc_table_2o A pointer to the two-orbital excitation table
 * @param[in] Number of orbitals
 */
void unrank_mixed(uint64_t rank, size_t *occ_list, const uint64_t *exc_table_2o, size_t norb) {
    uint64_t ncols = nC2(norb);
    uint64_t ij_rank = rank/ncols;
    uint64_t kl_rank = rank%ncols;
    unrank(ij_rank, occ_list, exc_table_2o, norb, 2);
    unrank(kl_rank, occ_list+2, exc_table_2o, norb, 2);
}