/**
 * \file hci_contract.c
 * \addtogroup contract
 * @{
 */

#include "hci_contract.h"
#include <math.h>
#include <assert.h>
#include <stdlib.h>

/**
 * Enum to encode number of differences between two occupancy lists; more
 * than two differences results in a matrix element of zero because only
 * one- and two-electron operators are involved in the Hamiltonian.
 */
typedef enum {
    ZERO,
    SINGLE,
    DOUBLE,
    THREE_PLUS
} DiffType;

/**
 * Enum used in the determination of the correct double excitation matrix element from storage.
 */
typedef enum {
    IJKL,
    IJLK,
    IKJL
} DESIGNATOR;

/**
 * Helper method to merge two sorted two-orbital lists into a sorted four-orbital list
 * in the calculation of the rank of a double excitation.
 *
 * @param[out] orb_list Pointer to the sorted four-orbital list
 * @param[in] old_orbs Pointer to the first sorted two-orbital list
 * @param[in] new_orbs Pointer to the second sorted two-orbital list
 */
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

/**
 * Given two occupation lists, determines if they are identical or differ by a single orbital,
 * two orbitals, or three or more orbitals, writing \p occ_1 \ \p occ_2 to \c old_orbs and
 * \p occ_2 \ \p occ_1 to \c new_orbs of \p exc_result (computing the sign of the excitation
 * along the way).
 *
 * @param[in] occ_1 Pointer to the first orbital occupancy list of length \p nocc
 * @param[in] occ_2 Pointer to the second orbital occupancy list of length \p nocc
 * @param[in] nocc The number of occupied orbitals
 * @param[out] exc_result Pointer to the output \ref ExcResult 
 */
static DiffType get_diff_type(const size_t *occ_1, const size_t *occ_2, size_t nocc, ExcResult *exc_result) {
    size_t iold = 0; // Index of ExcResult old_orbs being written to
    size_t inew = 0; // Index of ExcResult new_orbs being written to
    size_t iocc_1 = 0;  // Index of first orbital occupancy list being read from
    size_t iocc_2 = 0; // Index of second orbital occupancy list being read from
    double sign = 1.0; // Sign of excitation
    while ((iocc_1 < nocc) && (iocc_2 < nocc)) {
        size_t occ_orb_1 = occ_1[iocc_1];
        size_t occ_orb_2 = occ_2[iocc_2];
        if (occ_orb_1 < occ_orb_2) {
            // Three or more orbitals found in occ_1\occ_2
            if (iold == 2) {
                return THREE_PLUS;
            }
            exc_result->old_orbs[iold] = occ_orb_1;
            sign *= ((iocc_1 - iold) % 2 == 0) ? 1.0 : -1.0;
            iocc_1++;
            iold++;
        } else if (occ_orb_2 < occ_orb_1) {
            // Three or more orbitals found in occ_2\occ_1
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
    // Determine which condition caused the while loop to exit and handle the remaining
    // part of the relevant list
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

/**
 * Calculates the expectation value of the Hamiltonian for a given configuration (i.e. the
 * on-diagonal element).
 *
 * Formula: \f{aligned}{ 
             \langle\Psi|\Psi\rangle=&\sum\limits_a [a|h_\text{core}|a] + \dfrac{1}{2}\sum\limits_{ab} [aa|bb]-[ab|ba]&\\
             =\,&\sum\limits_a (a|h_\text{core}|a) + \sum\limits_\bar{a} (\bar{a}|h_\text{core}|\bar{a})&\text{From $\verb|h1e_aa|$ and $\verb|h1e_bb|$}\\
             +\,&\sum\limits_{a<b} (aa|bb)-(ab|ba)&\text{From $\verb|eri_mo_aaaa_s8|$}\\
             +\,&\sum\limits_{\bar{a}<\bar{b}} (\bar{a}\bar{a}|\bar{b}\bar{b})-(\bar{a}\bar{b}|\bar{b}\bar{a})&\text{From $\verb|eri_mo_bbbb_s8|$}\\
             +\,&\sum\limits_{a\bar{b}} (aa|\bar{b}\bar{b})&\text{From $\verb|eri_mo_aabb_s4|$}
           \f}
 * 
 * Formula conventions:
 * - \f$[\quad|\quad]\f$ are used for spin orbital summations, while \f$(\quad|\quad)\f$ are used for spatial orbital summations
 * - \f$a\f$ and \f$b\f$ index occupied spin orbitals in \f$\Psi\f$ if \f$[\quad|\quad]\f$ are used and occupied spatial orbitals if \f$(\quad|\quad)\f$ are used
 * - When summing over spatial orbitals, \f$a\f$ and \f$b\f$ are \f$\alpha\f$ MOs while \f$\bar{a}\f$ and \f$\bar{b}\f$ are \f$\beta\f$ MOs
 *
 * @param[in] occ_a  Pointer to the list of occupied \f$\alpha\f$ orbitals
 * @param[in] occ_b  Pointer to the list of occupied \f$\beta\f$ orbitals
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 * @param[in] h1e Pointer to \ref HCore object storing locations of the core Hamiltonian matrix elements
 * @param[in] eri_mo Pointer to \ref ERITensor object storing locations of the electron repulsion integrals
 * @return The expectation value of the energy of the configuration
 */
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

/**
 * Calculates the matrix element between configurations that differ by one \f$\alpha\f$ orbital.
 *
 * Formula: \f{aligned}{ 
             \langle\Psi|\Psi_a^r\rangle=&[a|h_\text{core}|r] + \sum\limits_{b} [ar|bb]-[ab|br]&\\
             =\,&(a|h_\text{core}|r) &\text{From $\verb|h1e_aa|$}\\
             +\,&\sum\limits_{b} (ar|bb)-(ab|br)&\text{From $\verb|eri_mo_aaaa_s8|$}\\
             +\,&\sum\limits_{\bar{b}} (ar|\bar{b}\bar{b})&\text{From $\verb|eri_mo_aabb_s4|$}
           \f}
 * 
 * Formula conventions:
 * - \f$[\quad|\quad]\f$ are used for spin orbital summations, while \f$(\quad|\quad)\f$ are used for spatial orbital summations
 * - \f$\Psi_a^r\f$ differs from \f$\Psi\f$ in the substitution of \f$\alpha\f$ orbital \f$a\f$ for \f$r\f$
 * - \f$b\f$ indexes occupied spin orbitals in \f$\Psi\f$ if \f$[\quad|\quad]\f$ is used and occupied spatial orbitals if \f$(\quad|\quad)\f$ is used
 * - When summing over spatial orbitals, \f$b\f$ is an \f$\alpha\f$ MO while \f$\bar{b}\f$ is a \f$\beta\f$ MO
 * - The above formula assumes the configurations are in maximum coincidence; an extra sign factor
 * derived from the indices of the changing orbitals in the old and new sorted occupation lists is necessary,
 * which is calculated in the computation of \p single_exc
 *
 * @param[in] single_exc Pointer to \ref ExcResult describing occupied and excitation orbitals
 * @param[in] occ_a  Pointer to the list of occupied \f$\alpha\f$ orbitals
 * @param[in] occ_b  Pointer to the list of occupied \f$\beta\f$ orbitals
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 * @param[in] h1e Pointer to \ref HCore object storing locations of the core Hamiltonian matrix elements
 * @param[in] eri_mo Pointer to \ref ERITensor object storing locations of the electron repulsion integrals
 * @return The single excitation matrix element
 */
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

/**
 * Calculates the matrix element between configurations that differ by one \f$\beta\f$ orbital.
 *
 * Formula: \f{aligned}{ 
             \langle\Psi|\Psi_\bar{a}^\bar{r}\rangle=&[\bar{a}|h_\text{core}|\bar{r}] + \sum\limits_{b} [\bar{a}\bar{r}|bb]-[\bar{a}b|b\bar{r}]&\\
             =\,&(\bar{a}|h_\text{core}|\bar{r}) &\text{From $\verb|h1e_bb|$}\\
             +\,&\sum\limits_{\bar{b}} (\bar{a}\bar{r}|\bar{b}\bar{b})-(\bar{a}\bar{b}|\bar{b}\bar{r})&\text{From $\verb|eri_mo_bbbb_s8|$}\\
             +\,&\sum\limits_{b} (bb|\bar{a}\bar{r})&\text{From $\verb|eri_mo_aabb_s4|$}
           \f}
 * 
 * Formula conventions:
 * - \f$[\quad|\quad]\f$ are used for spin orbital summations, while \f$(\quad|\quad)\f$ are used for spatial orbital summations
 * - \f$\Psi_\bar{a}^\bar{r}\f$ differs from \f$\Psi\f$ in the substitution of \f$\beta\f$ orbital \f$\bar{a}\f$ for \f$\bar{r}\f$
 * - \f$b\f$ indexes occupied spin orbitals in \f$\Psi\f$ if \f$[\quad|\quad]\f$ is used and occupied spatial orbitals if \f$(\quad|\quad)\f$ is used
 * - When summing over spatial orbitals, \f$b\f$ is an \f$\alpha\f$ MO while \f$\bar{b}\f$ is a \f$\beta\f$ MO
 * - The above formula assumes the configurations are in maximum coincidence; an extra sign factor
 * derived from the indices of the changing orbitals in the old and new sorted occupation lists is necessary,
 * which is calculated in the computation of \p single_exc
 *
 * @param[in] single_exc Pointer to \ref ExcResult describing occupied and excitation orbitals
 * @param[in] occ_a  Pointer to the list of occupied \f$\alpha\f$ orbitals
 * @param[in] occ_b  Pointer to the list of occupied \f$\beta\f$ orbitals
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 * @param[in] h1e Pointer to \ref HCore object storing locations of the core Hamiltonian matrix elements
 * @param[in] eri_mo Pointer to \ref ERITensor object storing locations of the electron repulsion integrals
 * @return The single excitation matrix element
 */
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

/**
 * Calculates the matrix element between configurations that differ by two \f$\alpha\f$ orbitals.
 *
 * Formula: \f{aligned}{ 
             \langle\Psi|\Psi_{ab}^{rs}\rangle=&[ar|bs]-[as|br]&\\
             =\,&(ar|bs)-(as|br) &\text{From $\verb|eri_mo_aaaa_s8|$}\\
           \f}
 * 
 * Formula conventions:
 * - \f$[\quad|\quad]\f$ are used for spin orbitals, while \f$(\quad|\quad)\f$ are used for spatial orbitals
 * - \f$\Psi_{ab}^{rs}\f$ differs from \f$\Psi\f$ in the substitution of \f$\alpha\f$ orbitals \f$a\f$ and \f$b\f$ for \f$r\f$ and \f$s\f$
 * - The above formula assumes the configurations are in maximum coincidence; an extra sign factor
 * derived from the indices of the changing orbitals in the old and new sorted occupation lists is necessary,
 * which is calculated in the computation of \p double_exc
 *
 * @param[in] double_exc Pointer to \ref ExcResult describing occupied and excitation orbitals
 * @param[in] eri_mo Pointer to \ref ERITensor object storing locations of the electron repulsion integrals
 * @return The double excitation matrix element
 */
double get_double_exc_value_aa(const ExcResult *double_exc, const ERITensor *eri_mo) {
    size_t *old_orbs = double_exc->old_orbs;
    size_t *new_orbs = double_exc->new_orbs;
    return double_exc->sign*(eri_mo->eri_mo_aaaa_s8[index_8d(old_orbs[0], new_orbs[0], old_orbs[1], new_orbs[1])]
                            -eri_mo->eri_mo_aaaa_s8[index_8d(old_orbs[0], new_orbs[1], old_orbs[1], new_orbs[0])]);
}

/**
 * Calculates the matrix element between configurations that differ by two \f$\beta\f$ orbitals.
 *
 * Formula: \f{aligned}{ 
             \langle\Psi|\Psi_{\bar{a}\bar{b}}^{\bar{r}\bar{s}}\rangle=&[\bar{a}\bar{r}|\bar{b}\bar{s}]-[\bar{a}\bar{s}|\bar{b}\bar{r}]&\\
             =\,&(\bar{a}\bar{r}|\bar{b}\bar{s})-(\bar{a}\bar{s}|\bar{b}\bar{r}) &\text{From $\verb|eri_mo_bbbb_s8|$}\\
           \f}
 * 
 * Formula conventions:
 * - \f$[\quad|\quad]\f$ are used for spin orbitals, while \f$(\quad|\quad)\f$ are used for spatial orbitals
 * - \f$\Psi_{\bar{a}\bar{b}}^{\bar{r}\bar{s}}\f$ differs from \f$\Psi\f$ in the substitution of \f$\beta\f$ orbitals \f$\bar{a}\f$ and \f$\bar{b}\f$ for \f$\bar{r}\f$ and \f$\bar{s}\f$
 * - The above formula assumes the configurations are in maximum coincidence; an extra sign factor
 * derived from the indices of the changing orbitals in the old and new sorted occupation lists is necessary,
 * which is calculated in the computation of \p double_exc
 *
 * @param[in] double_exc Pointer to \ref ExcResult describing occupied and excitation orbitals
 * @param[in] eri_mo Pointer to \ref ERITensor object storing locations of the electron repulsion integrals
 * @return The double excitation matrix element
 */
double get_double_exc_value_bb(const ExcResult *double_exc, const ERITensor *eri_mo) {
    size_t *old_orbs = double_exc->old_orbs;
    size_t *new_orbs = double_exc->new_orbs;
    return double_exc->sign*(eri_mo->eri_mo_bbbb_s8[index_8d(old_orbs[0], new_orbs[0], old_orbs[1], new_orbs[1])]
                            -eri_mo->eri_mo_bbbb_s8[index_8d(old_orbs[0], new_orbs[1], old_orbs[1], new_orbs[0])]);
}

/**
 * Extracts a double excitation matrix element from storage in a \ref DoubleExcEntry.
 *
 * Given the list of occupied/old and excitation/new orbitals, first swaps the lowest-numbered
 * orbital \f$i\f$ into position using the symmetries of the ERI tensor, then determines the ordering
 * of the remaining orbitals \f$j\f$, \f$k\f$, and \f$l\f$, where \f$i<j<k<l\f$. Once \f$j\f$, \f$k\f$,
 * and \f$l\f$ are located, the correct stored value can be extracted. As with \ref get_double_exc_value_aa
 * and \ref get_double_exc_value_bb, an additional sign factor is necessary, handled in the computation of
 * \p exc_entry.
 *
 * @param[in] exc_entry Pointer to \ref DoubleExcEntry storing the excitation matrix element
 * @param[in] double_exc Pointer to \ref ExcResult describing occupied and excitation orbitals
 * @return The stored double excitation matrix element
 */
double get_double_exc_value_from_store(const DoubleExcEntry *exc_entry, const ExcResult *double_exc) {
    size_t *old_orbs = double_exc->old_orbs;
    size_t *new_orbs = double_exc->new_orbs;
    size_t r = 0;
    size_t b = 0;
    size_t s = 0;
    if (old_orbs[0] < new_orbs[0]) {
        r = new_orbs[0];
        b = old_orbs[1];
        s = new_orbs[1];
    } else {
        r = old_orbs[0];
        b = new_orbs[1];
        s = old_orbs[1];
    }
    if (b < r) {
        return -exc_entry->iljk*double_exc->sign;
    } else if (b < s) {
        return exc_entry->ijkl*double_exc->sign;
    } else {
        return (exc_entry->ijkl+exc_entry->iljk)*double_exc->sign;
    }
}

/**
 * Calculates the matrix element between configurations that differ by one \f$\alpha\f$ and one \f$\beta\f$ orbital.
 *
 * Formula: \f{aligned}{ 
             \langle\Psi|\Psi_{a\bar{b}}^{r\bar{s}}\rangle=&[ar|\bar{b}\bar{s}]&\\
             =\,&(ar|\bar{b}\bar{s}) &\text{From $\verb|eri_mo_aabb_s4|$}\\
           \f}
 * 
 * Formula conventions:
 * - \f$[\quad|\quad]\f$ are used for spin orbitals, while \f$(\quad|\quad)\f$ are used for spatial orbitals
 * - \f$\Psi_{\bar{a}\bar{b}}^{\bar{r}\bar{s}}\f$ differs from \f$\Psi\f$ in the substitution of \f$\alpha\f$ orbital \f$a\f$ for \f$r\f$ and \f$\beta\f$ orbitals \f$\bar{b}\f$ and \f$\bar{s}\f$
 * - The above formula assumes the configurations are in maximum coincidence; an extra sign factor
 * derived from the indices of the changing orbitals in the old and new sorted occupation lists is necessary,
 * which is calculated in the computation of \p single_exc_a and \p single_exc_b
 *
 * @param[in] single_exc_a Pointer to \ref ExcResult describing occupied and excitation \f$\alpha\f$ orbitals
 * @param[in] single_exc_b Pointer to \ref ExcResult describing occupied and excitation \f$\beta\f$ orbitals
 * @param[in] eri_mo Pointer to \ref ERITensor object storing locations of the electron repulsion integrals
 * @return The mixed excitation matrix element
 */
double get_mixed_exc_value(const ExcResult *single_exc_a, const ExcResult *single_exc_b, 
    const ERITensor *eri_mo) {
        double sign = single_exc_a->sign*single_exc_b->sign;
        size_t row = index_2d(single_exc_a->old_orbs[0], single_exc_a->new_orbs[0]);
        size_t col = index_2d(single_exc_b->old_orbs[0], single_exc_b->new_orbs[0]);
        return sign*eri_mo->eri_mo_aabb_s4[(row*eri_mo->ncols_aabb)+col];
}

/**
 * Extracts a mixed excitation matrix element from storage in a \ref MixedExcEntry.
 *
 * @param[in] exc_entry Pointer to \ref MixedExcEntry storing the excitation matrix element
 * @param[in] single_exc_a Pointer to \ref ExcResult describing occupied and excitation \f$\alpha\f$ orbitals
 * @param[in] single_exc_b Pointer to \ref ExcResult describing occupied and excitation \f$\beta\f$ orbitals
 * @return The stored mixed excitation matrix element
 */
double get_mixed_exc_value_from_store(const MixedExcEntry *exc_entry, 
    const ExcResult *single_exc_a, const ExcResult *single_exc_b) {
        double sign = single_exc_a->sign * single_exc_b->sign;
        return sign*exc_entry->ijkl;
}

/**
 * Calculates the Hamiltonian matrix element between two configurations given their ranks.
 *
 * @param[in] rank1 \ref Rank specifying the \f$\alpha\f$ and \f$\beta\f$ string of the first configuration
 * @param[in] rank2 \ref Rank specifying the \f$\alpha\f$ and \f$\beta\f$ string of the second configuration
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 * @param[in] h1e Pointer to \ref HCore object storing locations of the core Hamiltonian matrix elements
 * @param[in] eri_mo Pointer to \ref ERITensor object storing locations of the electron repulsion integrals
 * @return The matrix element of of the Hamiltonian between the two configurations
 */
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

/**
 * Calculates the Hamiltonian matrix element between two configurations given occupancy lists for the first
 * configuration and the \ref Rank associated with the second configuration.
 *
 * @param[in] occ_a_1 Pointer to \f$\alpha\f$ occupancy list of first configuration
 * @param[in] occ_b_1 Pointer to \f$\beta\f$ occupancy list of first configuration
 * @param[in] rank2 \ref Rank specifying the \f$\alpha\f$ and \f$\beta\f$ string of the second configuration
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 * @param[in] h1e Pointer to \ref HCore object storing locations of the core Hamiltonian matrix elements
 * @param[in] eri_mo Pointer to \ref ERITensor object storing locations of the electron repulsion integrals
 * @return The matrix element of of the Hamiltonian between the two configurations
 */
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

/**
 * Calculates the Hamiltonian matrix element between two configurations given their ranks;
 * uses the excitation matrix elements stored in \p exc_entries instead of \p eri_mo directly
 * for the purposes of checking the extraction logic.
 *
 * @param[in] rank1 \ref Rank specifying the \f$\alpha\f$ and \f$\beta\f$ string of the first configuration
 * @param[in] rank2 \ref Rank specifying the \f$\alpha\f$ and \f$\beta\f$ string of the second configuration
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 * @param[in] exc_entries Pointer to \ref ExcEntries object sorted inc. by rank providing location of stored excitations
 * @param[in] h1e Pointer to \ref HCore object storing locations of the core Hamiltonian matrix elements
 * @param[in] eri_mo Pointer to \ref ERITensor object storing locations of the electron repulsion integrals
 * @return The matrix element of of the Hamiltonian between the two configurations
 */
double get_matrix_element_by_rank_test_storage(Rank rank1, Rank rank2, 
    const ConfigInfo *config_info, const ExcEntries *exc_entries,
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
                        return get_double_exc_value_from_store(exc_entries->doubles_bb+exc_rank, &exc_b);
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
                        return get_mixed_exc_value_from_store(exc_entries->mixed_ab+exc_rank, &exc_a, &exc_b);
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
                        return get_double_exc_value_from_store(exc_entries->doubles_aa+exc_rank, &exc_a);
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

/**
 * Assembles all diagonal matrix elements for configurations contained
 * in \p hcivec naively; used for the Davidson preconditioner.
 *
 * @param[in] hcivec Pointer to a \ref HCIVec supplying configuration ranks (and their coefficients too)
 * @param[out] hdiag Pointer to array of length \p hcivec->len where the diagonal elements should be written to
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 * @param[in] h1e Pointer to \ref HCore object storing locations of the core Hamiltonian matrix elements
 * @param[in] eri_mo Pointer to \ref ERITensor object storing locations of the electron repulsion integrals
 */
void make_hdiag_slow(const HCIVec *hcivec, double *hdiag,
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

/**
 * Contracts \p hcivec_old with the Hamiltonian in the subspace spanned by the \p hcivec_old->ranks
 * in a naive fashion, outputting the new coefficients to \p coeffs_new.
 *
 * @param[in] hcivec_old Pointer to a \ref HCIVec supplying configuration ranks and their coefficients
 * @param[out] coeffs_new The coefficients after contraction with the Hamiltonian
 * @param[in] hdiag Pointer to the diagonal elements of the Hamiltonian
 * @param[in] config_info Pointer to \ref ConfigInfo object needed to perform unranking, control loop structure, etc.
 * @param[in] h1e Pointer to \ref HCore object storing locations of the core Hamiltonian matrix elements
 * @param[in] eri_mo Pointer to \ref ERITensor object storing locations of the electron repulsion integrals
 */
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