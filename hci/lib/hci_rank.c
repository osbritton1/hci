/**
 * \file hci_rank.c
 * \addtogroup rank
 * @{
 */

#include "hci_rank.h"

// Command to run to generate this library: gcc -shared -o libhci.so -fPIC hci_rank.c hci_store.c hci_enlarge.c hci_contract.c -Wall -g

/**
 * Helper function for unranking algorithm. 
 * 
 * Uses binary search to find the index of the last entry in a row of 
 * the ranking table that is less than the value of target.
 *
 * @param[in] target Target value used as a reference for the search
 * @param[in] row Pointer to the row of the ranking table used for the search
 * @param[in] norb Size of orbital space
 * @param[in] nocc Number of occupancies; generally equal to \f$N_\alpha\f$ or \f$N_\beta\f$
 */
static size_t find_row_index(uint64_t target, const uint64_t *row, size_t norb, size_t nocc) {
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
 * Get the combinatorial rank corresponding to the provided list of occupied orbitals.
 *
 * Formula: @f[ \sum\limits_{i=1}^{N_\text{occ}}\binom{c_i}{i}, @f]
 * where \f$c_i\f$ is the \f$i\f$th orbital in the occupancy list sorted in ascending order
 *
 * @param[in] occ_list Pointer to an array of length \f$N_\text{occ}\f$ specifying the zero-indexed occupied orbitals in ascending order
 * @param[in] rank_table Pointer to ranking table initialized by \ref load_rank_table
 * @param[in] norb Size of orbital space
 * @param[in] nocc Number of occupancies; generally equal to \f$N_\alpha\f$ or \f$N_\beta\f$
 * @return The rank of the specified occupancy list
 */
static uint64_t rank(const size_t *occ_list, const uint64_t *rank_table, size_t norb, size_t nocc) {
    uint64_t sum = 0;
    size_t ncols = norb-nocc+1;
    for (size_t i=0; i<nocc; i++) {
        size_t occ_orbital = occ_list[i];
        sum += rank_table[(i*ncols)+occ_orbital-i];
    }
    return sum;
}

/**
 * Greedy unranking algorithm for computing an occupancy list given its combinatorial rank.
 *
 * Starting from the last row in the ranking table (corresponding to the highest occupied orbital),
 * uses \ref find_row_index to find the column index of the entry that does not exceed the value of target,
 * updates the occupancy list accordingly, then subtracts off the ranking table entry from target and repeats.
 *
 * @param[in] rank The rank of the occupancy list to be computed
 * @param[out] occ_list Pointer to an array of length \f$N_\text{occ}\f$ to store the occupancy list
 * @param[in] rank_table Pointer to ranking table initialized by \ref load_rank_table
 * @param[in] norb Size of orbital space
 * @param[in] nocc Number of occupancies; generally equal to \f$N_\alpha\f$ or \f$N_\beta\f$
 */
static void unrank(uint64_t rank, size_t *occ_list, const uint64_t *rank_table, size_t norb, size_t nocc) {
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
 * Initializes the \f$N_\text{occ}\f$ by \f$N_\text{orb}-N_\text{occ}+1\f$ table 
 * needed to rank and unrank combinations using the combinatorial number system.
 *
 * The binomial coefficients needed for the encoding the position of the ith electron are stored in row i-1.
 * The entry in position (i, j) is \f$\binom{i+j}{i+1}\f$; when i+j < i+1 (first column), this is defined to be 0.
 *
 * @param[out] table Pointer to the uninitialized ranking table; must accommodate \f$N_\text{occ}\cdot(N_\text{orb}-N_\text{occ}+1)\f$ entries
 * @param[in] norb Size of orbital space
 * @param[in] nocc Number of occupancies; generally equal to \f$N_\alpha\f$ or \f$N_\beta\f$
 */
void load_rank_table(uint64_t *table, size_t norb, size_t nocc) {
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
 * Ranks a given \f$\alpha\f$ orbital occupancy list.
 *
 * @param[in] occ_list Pointer to an array of length \f$N_\text{occ}\f$ specifying the zero-indexed occupied \f$\alpha\f$ orbitals in ascending order
 * @param[in] config_info Pointer to a \ref ConfigInfo struct storing the location of the necessary tables
 * @return The rank of the specified \f$\alpha\f$ occupancy list
 */
uint64_t rank_occ_a(const size_t *occ_list, const ConfigInfo *config_info) {
    return rank(occ_list, config_info->occ_table_a, config_info->norb, config_info->nelec_a);
}

/**
 * Unranks a given \f$\alpha\f$ orbital occupancy list.
 * 
 * @param[in] arank The rank of the \f$\alpha\f$ occupancy list of interest
 * @param[out] occ_list Pointer to an array of length \f$N_\text{occ}\f$ to store the \f$\alpha\f$ occupancy list
 * @param[in] config_info Pointer to a \ref ConfigInfo struct storing the location of the necessary tables
 */
void unrank_occ_a(uint64_t arank, size_t *occ_list, const ConfigInfo *config_info) {
    unrank(arank, occ_list, config_info->occ_table_a, config_info->norb, config_info->nelec_a);
}

/**
 * Determines all orbitals in the complement of a \f$\alpha\f$ occupancy list with given rank.
 * 
 * @param[in] arank The rank of the \f$\alpha\f$ occupancy list of interest
 * @param[out] virt_list Pointer to an array of length \f$N_\text{orb}-N_\text{occ}\f$ to store the \f$\alpha\f$ virtual orbital list
 * @param[in] config_info Pointer to a \ref ConfigInfo struct storing the location of the necessary tables
 */
void unrank_virt_a(uint64_t arank, size_t *virt_list, const ConfigInfo *config_info) {
    unrank(config_info->combmax_a-arank-1, virt_list, 
        config_info->virt_table_a, config_info->norb, config_info->norb-config_info->nelec_a);
}

/**
 * Ranks a given \f$\beta\f$ orbital occupancy list.
 *
 * @param[in] occ_list Pointer to an array of length \f$N_\text{occ}\f$ specifying the zero-indexed occupied \f$\beta\f$ orbitals in ascending order
 * @param[in] config_info Pointer to a \ref ConfigInfo struct storing the location of the necessary tables
 * @return The rank of the specified \f$\beta\f$ occupancy list
 */
uint64_t rank_occ_b(const size_t *occ_list, const ConfigInfo *config_info) {
    return rank(occ_list, config_info->occ_table_b, config_info->norb, config_info->nelec_b);
}

/**
 * Unranks a given \f$\beta\f$ orbital occupancy list.
 * 
 * @param[in] arank The rank of the \f$\beta\f$ occupancy list of interest
 * @param[out] occ_list Pointer to an array of length \f$N_\text{occ}\f$ to store the \f$\beta\f$ occupancy list
 * @param[in] config_info Pointer to a \ref ConfigInfo struct storing the location of the necessary tables
 */
void unrank_occ_b(uint64_t brank, size_t *occ_list, const ConfigInfo *config_info) {
    unrank(brank, occ_list, config_info->occ_table_b, config_info->norb, config_info->nelec_b);
}

/**
 * Determines all orbitals in the complement of a \f$\beta\f$ occupancy list with given rank.
 * 
 * @param[in] arank The rank of the \f$\beta\f$ occupancy list of interest
 * @param[out] virt_list Pointer to an array of length \f$N_\text{orb}-N_\text{occ}\f$ to store the \f$\beta\f$ virtual orbital list
 * @param[in] config_info Pointer to a \ref ConfigInfo struct storing the location of the necessary tables
 */
void unrank_virt_b(uint64_t brank, size_t *virt_list, const ConfigInfo *config_info) {
    unrank(config_info->combmax_b-brank-1, virt_list, 
        config_info->virt_table_b, config_info->norb, config_info->norb-config_info->nelec_b);
}

/**
 * Ranks a double excitation based on the four orbitals involved.
 *
 * @param[in] exc_list Pointer to an array of length 4 specifying the four involved orbitals in ascending order
 * @param[in] config_info Pointer to a \ref ConfigInfo struct storing the location of the necessary tables
 * @return The rank of the specified double excitation
 */
uint64_t rank_double_exc(size_t *exc_list, const ConfigInfo *config_info) {
    return rank(exc_list, config_info->exc_table_4o, config_info->norb, 4);
}

/**
 * Determines the four orbitals involved in a double excitation of given rank.
 * 
 * @param[in] exc_rank The rank of the double excitation of interest
 * @param[out] exc_list Pointer to an array of length 4 to store the excitation orbital list
 * @param[in] config_info Pointer to a \ref ConfigInfo struct storing the location of the necessary tables
 */
void unrank_double_exc(uint64_t exc_rank, size_t *exc_list, const ConfigInfo *config_info) {
    unrank(exc_rank, exc_list, config_info->exc_table_4o, config_info->norb, 4);
}

/**
 * Calculates the rank of a mixed \f$\alpha\beta\rightarrow\alpha\beta\f$ excitation.
 *
 * @param[in] exc_list Pointer to an array of the four involved orbitals
 * The first two are \f$\alpha\f$ orbitals (in ascending order) and the second two are \f$\beta\f$ orbitals (also in ascending order)
 * @param[in] config_info Pointer to a \ref ConfigInfo struct storing the location of the necessary tables
 * @return The rank of the specified mixed excitation
 */
uint64_t rank_mixed_exc(size_t *exc_list, const ConfigInfo *config_info) {
    uint64_t ij_rank = rank(exc_list, config_info->exc_table_2o, config_info->norb, 2);
    uint64_t kl_rank = rank(exc_list+2, config_info->exc_table_2o, config_info->norb, 2);
    return (ij_rank*config_info->ncols_mixed)+kl_rank;
}

/**
 * Determines the occupancy list corresponding to a mixed excitation with given rank.
 * 
 * @param[in] exc_rank_ab The rank of the mixed excitation
 * @param[out] exc_list Pointer to an array of length 4 to store the excitation orbital list
 * @param[in] config_info Pointer to a \ref ConfigInfo struct storing the location of the necessary tables
 */
void unrank_mixed_exc(uint64_t exc_rank_ab, size_t *exc_list, const ConfigInfo *config_info) {
    uint64_t ncols = config_info->ncols_mixed;
    uint64_t ij_rank = exc_rank_ab/ncols;
    uint64_t kl_rank = exc_rank_ab%ncols;
    unrank(ij_rank, exc_list, config_info->exc_table_2o, config_info->norb, 2);
    unrank(kl_rank, exc_list+2, config_info->exc_table_2o, config_info->norb, 2);
}

/**
 * @}
 */