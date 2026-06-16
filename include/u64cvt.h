#pragma once
// u64 <-> u52 boundary layer and the public multiply entry point.
//
// Layout fact both directions exploit: 8 u52 digits = 416 bits = exactly 52
// bytes of the packed u64 representation, so block j of digits maps to byte
// offset 52*j with byte-aligned blocks. Within a block, digit i starts at byte
// floor(6.5*i) with a nibble shift of (i&1)*4.
//
// decode (u64 -> u52): one (masked) 64-byte load, one vpermb gathering each
// digit's 8-byte window, one variable shift, one mask -- 8 digits per ~4 ops.
// encode (u52 -> u64): fused with the signed canonicalization of the redundant
// product; the canonical vector is byte-permuted into the packed layout with
// two vpermb (byte 7 of a canonical digit is always zero and serves as the
// zero filler -- scheme ported from reference/ifma52_mul.cpp pack52_vec) and
// byte-masked stores. Exact u52 digit counts come from __builtin_clzll on the
// top limb.
#include "mul.h"

// ---- decode -----------------------------------------------------------

// digit i gathers input bytes floor(6.5 i) .. +7 of its 52-byte block
alignas(64) static const uint8_t u52_dec_perm[64] = {
     0,  1,  2,  3,  4,  5,  6,  7,
     6,  7,  8,  9, 10, 11, 12, 13,
    13, 14, 15, 16, 17, 18, 19, 20,
    19, 20, 21, 22, 23, 24, 25, 26,
    26, 27, 28, 29, 30, 31, 32, 33,
    32, 33, 34, 35, 36, 37, 38, 39,
    39, 40, 41, 42, 43, 44, 45, 46,
    45, 46, 47, 48, 49, 50, 51, 52,
};

// bit length of {ap, an}; tolerates zero high limbs
static inline uint64_t u64_bit_length(const uint64_t *ap, uint64_t an){
    while(an && ap[an - 1] == 0) --an;
    if(!an) return 0;
    return an * 64 - (uint64_t)__builtin_clzll(ap[an - 1]);
}

// unpack {ap, an} u64 limbs into n52 = ceil(bits/52) canonical u52 digits at
// r (vector-padded: full-vector stores, zero lanes beyond n52). Returns n52.
static inline uint64_t u52_from_u64(pvec r, const uint64_t *ap, uint64_t an){
    const uint64_t bits = u64_bit_length(ap, an);
    const uint64_t n52 = (bits + 51) / 52;
    const uint8_t *p = (const uint8_t *)ap;
    int64_t rem = (int64_t)(an * 8);
    const _vec perm = load_vec((cpvec)u52_dec_perm);
    const _vec sh = setr_64(0, 4, 0, 4, 0, 4, 0, 4);
    uint64_t blocks = (n52 + 7) >> 3;
    for(; rem >= 64 && blocks > 1; --blocks, p += 52, rem -= 52, ++r)
        store_vec(r, and_v(srlv(permb(perm, load_vec((cpvec)p)), sh), MASK52));
    for(; blocks; --blocks, p += 52, rem -= 52, ++r){    // at most two masked
        const _vec w = vec_fn(maskz_loadu_epi8)(
            rem >= 64 ? ~0ull : (rem > 0 ? (~0ull >> (64 - rem)) : 0), p);
        store_vec(r, and_v(srlv(permb(perm, w), sh), MASK52));
    }
    return n52;
}

// ---- public entry ------------------------------------------------------

#ifndef SIMD_MPN_FFT
#define SIMD_MPN_FFT 1
#endif
#if SIMD_MPN_FFT
#include "fft16.h"

// FFT/toom frontier. FFT cost is a function of an+bn only (~F ns per total
// limb); the toom route's per-an-limb cost converges, for an >> bn, to a
// smooth function S(bn) of bn alone (2bn x bn strip blocks). Route to FFT
// iff S(bn)*an > F*(an+bn). S measured at an = 32*bn (bench sbn_table.csv,
// T22 = 96), q8 fixed point, bn in steps of 8; beyond the table S > 2F so
// FFT wins for every shape. F = 5.0 ns (empirical median; the frontier is
// where both routes tie, so the rule's slop costs only second-order time).
static const uint16_t simd_mpn_fft_sq8[48] = {
     214,  267,  353,  438,  531,  624,  671,  773,  865, 1149,
    1161, 1237, 1229, 1315, 1374, 1400, 1465, 1539, 1559, 1791,
    1852, 1867, 1949, 1920, 2023, 1929, 2015, 2091, 2064, 2183,
    2166, 2189, 2257, 2366, 2332, 2403, 2416, 2496, 2492, 2633,
    2637, 2656, 2691, 2718, 2694, 2752, 2800, 2825,
};
#define SIMD_MPN_FFT_FQ8 1280u   /* 5.0 ns in q8 */

static inline int simd_mpn_fft_profitable(uint64_t an, uint64_t bn){
    if(2u * (an + bn) > F16_MAX_N_C) return 0;     /* FFT band cap */
    if(bn > 384) return 1;                         /* S(bn) > 2F: always */
    uint64_t s_q8 = simd_mpn_fft_sq8[(bn - 1) >> 3];
    return s_q8 * an > (uint64_t)SIMD_MPN_FFT_FQ8 * (an + bn);
}
#endif

// rp[0 .. an+bn) = {ap, an} * {bp, bn} on packed u64 limbs. Operands may come
// in either order; rp must not overlap the inputs.
static inline void simd_mpn_mul(uint64_t *rp, const uint64_t *ap, uint64_t an,
                                const uint64_t *bp, uint64_t bn){
    if(an < bn){
        const uint64_t *tp = ap; ap = bp; bp = tp;
        uint64_t tn = an; an = bn; bn = tn;
    }
    if(an <= 6)
        return simd_mpn_mul_basecase_le6(rp, ap, an, bp, bn);
#if SIMD_MPN_FFT
    if(simd_mpn_fft_profitable(an, bn) &&
       fft16_mul(rp, ap, (ptrdiff_t)an, bp, (ptrdiff_t)bn))
        return;
#endif

    scratch *sc = scratch_thread();
    SCRATCH(sc);
    // worst case ceil(64n/52) digits, +1 vector of slack for full-vector ops
    const uint64_t va = u52_vec_count((an * 16 + 12) / 13) + 1;
    const uint64_t vb = u52_vec_count((bn * 16 + 12) / 13) + 1;
    pvec a52 = SALLOC(sc, _vec, va);
    pvec b52 = SALLOC(sc, _vec, vb);
    pvec p52 = SALLOC(sc, _vec, va + vb);
    const uint64_t n52a = u52_from_u64(a52, ap, an);
    const uint64_t n52b = u52_from_u64(b52, bp, bn);
    if(!n52a || !n52b){
        memset(rp, 0, (an + bn) * 8);
        return;
    }
    const int cls = mul_u52_dispatch(p52, (cpvec)a52, (cpvec)b52, n52a, n52b, sc);
    // the encode walks ceil(out_bytes/52) blocks = 8*ceil(...) digits; lanes
    // between the product's n52a+n52b digits and that boundary must be zero
    const uint64_t need = 8 * ((8 * (an + bn) + 51) / 52);
    uint64_t *pl = (uint64_t *)(void *)p52;
    for(uint64_t i = n52a + n52b; i < need; ++i) pl[i] = 0;
    // class-1 top level: the signed chain rides the encode (the top level
    // never folds; this pass replaces it for free)
    if(cls) u64_from_u52_canonneg(rp, (cpvec)p52, an + bn);
    else    u64_from_u52_canon(rp, (cpvec)p52, an + bn);
}
