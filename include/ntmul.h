/* ntmul.h -- combined NTT multiply/square dispatch over the p30 and
 * p50 engines (docs/p50_ntt_design.md §11).
 *
 * Measured shape (balanced sweeps, Zen5, bench/nt_sweep_v1.csv):
 *   - in-cache (small totals) p50 wins outright: +12-19% mul,
 *     +10-20% sqr (conv8 + IFMA, compute-bound -- transform fill
 *     barely matters);
 *   - deep DRAM p30 wins outright (20% less spectra bandwidth +
 *     fused input f3 + the 3*2^k ladder octave) to its CRT cap
 *     (min operand 2.34M limbs / total 2^22 limbs);
 *   - BETWEEN the two, the winner alternates with transform
 *     occupancy: the engines' power-of-two steps are offset (p50
 *     steps at 11*2^k total limbs at T = 88, p30 at 2^(k+4)), and
 *     picking the FULLER truncated transform (fill = tv/M) matched
 *     the measured winner at every swept point (worst 2.3%, noise;
 *     a single-threshold rule leaves 9-14% on seam points);
 *   - p50 alone covers the band beyond p30's caps (to ~2^40 limbs).
 *
 * Entries return 1 on success, 0 only beyond the p50 band. */
#ifndef NTMUL_H
#define NTMUL_H

#include "p30.h"
#include "p50.h"

#ifndef NT_Z_LO
#define NT_Z_LO 180224u              /* 11*2^14: p50's first M-step */
#endif
#ifndef NT_Z_HI
#define NT_Z_HI (1u << 20)
#endif

static inline int nt_pick50(uint64_t an, uint64_t bn){
    uint64_t zl = an + bn;
    if(zl <= NT_Z_LO) return 1;
    if(zl >= NT_Z_HI) return 0;
    /* seam band: compare transform fills (T = 88 always fits here) */
    uint64_t zt = (an * 64 + 87) / 88 + (bn * 64 + 87) / 88 - 1;
    uint64_t tv50 = (zt + 7) / 8;
    uint64_t tv30 = (zl - 1 + 15) / 16;
    uint64_t M50 = 1, M30 = 1;
    while(M50 < tv50) M50 <<= 1;
    while(M30 < tv30) M30 <<= 1;
    return tv50 * M30 >= tv30 * M50;
}

static inline int nt_mul_r(uint64_t *rp,
                           const uint64_t *ap, ptrdiff_t an,
                           const uint64_t *bp, ptrdiff_t bn,
                           scratch *sc){
    if(an <= 0 || bn <= 0) return 0;
    if(nt_pick50((uint64_t)an, (uint64_t)bn)){
        if(p50_mul_r(rp, ap, an, bp, bn, sc)) return 1;
        return p30_mul_r(rp, ap, an, bp, bn, sc);
    }
    if(p30_mul_r(rp, ap, an, bp, bn, sc)) return 1;
    return p50_mul_r(rp, ap, an, bp, bn, sc);
}

static inline int nt_sqr_r(uint64_t *rp, const uint64_t *ap, ptrdiff_t an,
                           scratch *sc){
    if(an <= 0) return 0;
    if(nt_pick50((uint64_t)an, (uint64_t)an)){
        if(p50_sqr_r(rp, ap, an, sc)) return 1;
        return p30_sqr_r(rp, ap, an, sc);
    }
    if(p30_sqr_r(rp, ap, an, sc)) return 1;
    return p50_sqr_r(rp, ap, an, sc);
}

static inline int nt_mul(uint64_t *rp,
                         const uint64_t *ap, ptrdiff_t an,
                         const uint64_t *bp, ptrdiff_t bn){
    return nt_mul_r(rp, ap, an, bp, bn, scratch_thread());
}

static inline int nt_sqr(uint64_t *rp, const uint64_t *ap, ptrdiff_t an){
    return nt_sqr_r(rp, ap, an, scratch_thread());
}

#endif /* NTMUL_H */
