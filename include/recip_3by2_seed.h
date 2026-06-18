#pragma once
// recip_3by2_seed: stripped 3/2 block reciprocal (src/avx512/recip_3by2_seed.s).
//
//   in : I = uint64_t[13], normalized 832-bit divisor (top bit of I[12] set).
//   out: one zmm (8 u52 lanes) = a reciprocal seed within +-1 of the exact
//        V = floor((2^1248-1)/I) - 2^416.  Specifically seed in {V-1, V}
//        (one-sided low estimate).  Proof: 3by2_recip.md Sec.6.
//
// = recip_3by2 with the verify tail (reverse-multiply, canonize, compare,
// inc/dec) removed: just inlined recip_mono on the top 448 bits of I + the
// bit[31..446] seed load.  Use when the consumer (e.g. OurDiv3by2) corrects a
// +-1 reciprocal in its own quotient fixup, so the exact floor is unnecessary.
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif
_vec recip_3by2_seed(const uint64_t *I);
#ifdef __cplusplus
}
#endif
