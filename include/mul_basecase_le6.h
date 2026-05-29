#pragma once

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Multiply {ap, an} by {bp, bn}, where 1 <= an,bn <= 6.
// The result is written to rp[0 .. an + bn - 1].
//
// This is a tiny assembly basecase intended for the scalar front-end before the
// u52/IFMA kernels become worthwhile.  The implementation may swap operands
// internally; callers only need to provide non-overlapping output.
void simd_mpn_mul_basecase_le6(plimb rp, const _limb *ap, uint64_t an,
                               const _limb *bp, uint64_t bn);

// Add {ap, an} * {bp, bn} into rp[0 .. an + bn - 1].
// Returns the carry out beyond limb an + bn.
_limb simd_mpn_addmul_basecase_le6(plimb rp, const _limb *ap, uint64_t an,
                                   const _limb *bp, uint64_t bn);

#ifdef __cplusplus
}
#endif
