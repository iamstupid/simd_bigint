#pragma once
// recip_3by2: fused AVX-512 3/2 block reciprocal (src/avx512/recip_3by2.s).
//
//   in : I = uint64_t[13], a normalized 832-bit divisor -- 2^831 <= I < 2^832,
//        i.e. the top bit of I[12] is set.
//   out: one zmm (8 u52 lanes) = V = floor((2^1248 - 1) / I) - 2^416, the exact
//        Moeller-Granlund 3/2 block reciprocal with the implicit leading
//        B = 2^416 dropped.  This is the multiplier for dividing a 3-block
//        (1248-bit) value by the 2-block (832-bit) divisor I.
//
// Monolithic hand kernel: inlines recip_mono for the scalar 448-bit seed, then
// seed load -> decode I -> seed*I (madd52) -> alignr diagonal reduce + canonize
// -> compare vs T=[all1,~i0,~i1] -> single inc/dec.  ~192 cycles on Zen5;
// validated bit-exact against GMP over 30M+ random and adversarial inputs.
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

// I must be exactly 13 limbs and normalized (top bit of I[12] set).
// Returns V in a zmm register (8 canonical u52 lanes, lane 0 least significant).
_vec recip_3by2(const uint64_t *I);

#ifdef __cplusplus
}
#endif
