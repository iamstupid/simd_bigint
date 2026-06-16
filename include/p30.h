#pragma once
// p30.h - 5x30-bit-prime truncated NTT bigint multiplier (AVX-512).
// u64-limb I/O, 64-bit chunking: ONE limb = ONE coefficient. Band:
// total = an+bn <= 2^22 limbs and min(an,bn) <= 2,344,425 (CRT bound).
// Design + measurement history: docs/p30_ntt_design.md. This file is
// the CLEAN production engine; the experimental superset (dormant
// paths, the general standalone inverse) lives in p30_ref.h.
//
// Pipeline (W = 16 lanes, y = x^16):
//   input   u64 limbs -> 5 prime residue arrays, one fused pass (the
//           limb loads are shared across primes, fused REDC
//           (hi*R^2 + lo*R)*R^-1). Past L3 (M >= 2^17) the pass also
//           absorbs the FIRST 3 transform levels as a tiled block-
//           column network with NT stores (p30_input_f3); the
//           traversals then enter 3 levels down (p30_*_pre).
//   y-TFT   van der Hoeven TRUNCATED NTT on whole 16-lane vectors
//           (vector j = limbs 16j..16j+15): 16 scalar TFTs in
//           lockstep, FLINT twiddle convention (node twiddle
//           w2[j] = w[2j], prefix-stable bit-reversed tables as
//           Barrett (c, rec) pairs). Full subtrees run an ITERATIVE
//           DFS (radix-8 moths fire on leaf-block alignment) with
//           in-register m <= 32 bottoms; the truncation spine is
//           radix-2 (containment rule: no truncated moth variants).
//   fused   fwd(b) + pointwise + inverse in ONE traversal: each leaf
//           block transforms, convolves against the matching
//           a-spectrum block and inverts while cache-hot. The spine
//           is fully fused: nonzero inverse tails travel in a side
//           channel laid over the a-spectrum's DEAD tail [tv, M).
//   pw      16-point twisted cyclic convolution mod (x^16 - w[k]) at
//           spectral position k: parity-split Karatsuba (3 twisted-8
//           convs, 2 positions per call). The only Montgomery use;
//           its uniform R^-1 stray cancels in the final scale.
//   output  per-prime scale s = NV^-1 * R, then Garner mixed-radix
//           CRT (ascending primes, batched step-major through an L1
//           buffer), 3-stream compose, two SWAR carry chains.
//   ladder  one octave (tv in (2^15, 3*2^14], where the binary
//           working set spills L3) dispatches to a mixed 3*2^k
//           transform: full radix-3 top level (guaranteed by the
//           window), children TWISTED to cyclic form (x -> nu*x) so
//           all binary machinery runs verbatim, 2x2 Vandermonde seam
//           on the inverse; separate 3*2^20-capable prime set.
//
// Range discipline: between-pass values are LAZY [0,4p); Barrett
// (p30_bar2p) eats multiplied operands raw; the unmultiplied
// butterfly operand shrinks once per level (p30_sh2). Conv inputs
// shrink to canonical at the conv (each spectrum vector is read by
// exactly one conv, so this replaces a standalone pass for free).
//
// Every kernel choice here is microbenched (bench/p30_*_bench.c,
// p30_fuse_probe.c) and the whole engine is gated by GMP
// differentials incl. all-0xFF adversarial probes (tests/p30_test.c);
// transform math proven by scalar models (tests/p30_tft_model_test.c,
// tests/p30_tft3_model_test.c).
//
// Entry: p30_mul_r(rp, ap, an, bp, bn, scratch*) -> 1 ok / 0 out of
// band; p30_mul(...) uses the thread scratch.
#include "scratch.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned __int128 p30_u128;

#define P30_NP 5
#define P30_W 16
#define P30_MAX_LG 18                    /* M <= 2^18 vectors = 2^22 limbs */

/* ascending order is load-bearing for Garner */
static const uint32_t P30_PR[P30_NP] = { 918552577u, 935329793u, 943718401u,
                                         985661441u, 998244353u };
/* mixed-radix set: largest 5 primes with 3*2^20 | p-1 (radix-3 twist
 * roots to the top rung 3*2^16 vectors); product 2^149.45, min-operand
 * cap 2.86M limbs. NOTE p up to 1.054e9: conv wide-REDC results reach
 * 2.93p (< 3p) before the pack-side folds -- kshrp3 handles [0, 3p). */
static const uint32_t P30_MPR[P30_NP] = { 962592769u, 972029953u, 975175681u,
                                          1012924417u, 1053818881u };
#define P30_MIN_OPERAND_CAP 2344425u     /* floor(P / (2^64-1)^2) */

/* fused-input gates (overridable; tests force them low/high).
 * P30_FIN_MIN_M: use the fused input column pass (decode + first 3
 * levels, bench/p30_fuse_probe.c) when M >= this (must be >= 8).
 * P30_FIN_NT_MIN_M: non-temporal spectra stores when M >= this (NT
 * only pays once the working set is past L3 -- below that it forces
 * DRAM round trips the cache would have absorbed). */
/* mixed-radix ladder window (by binary lg; defaults = the measured
 * win region). Tests force MIN_LG low to exercise every rung. */
#ifndef P30_MIX_MIN_LG
#define P30_MIX_MIN_LG 16
#endif
#ifndef P30_MIX_MAX_LG
#define P30_MIX_MAX_LG 16
#endif

#ifndef P30_FIN_MIN_M
#define P30_FIN_MIN_M (1 << 17)      /* tuned: wins past L3 only */
#endif
#ifndef P30_FIN_NT_MIN_M
#define P30_FIN_NT_MIN_M (1 << 17)   /* past-L3 working sets only */
#endif

/* ------------------------------------------------------------------ */
/* scalar modular helpers (setup only)                                 */
/* ------------------------------------------------------------------ */
static inline uint32_t p30_mulm(uint64_t a, uint64_t b, uint32_t p){
    return (uint32_t)(a * b % p);
}
static inline uint32_t p30_powm(uint64_t a, uint64_t e, uint32_t p){
    uint64_t r = 1;
    for(; e; e >>= 1, a = a * a % p)
        if(e & 1) r = r * a % p;
    return (uint32_t)r;
}
static inline uint32_t p30_invm(uint32_t a, uint32_t p){
    return p30_powm(a, p - 2u, p);
}

/* Barrett constant pair: r = a*c - hi32(a*rec)*p in [0, 2p) for ANY
 * u32 lane a (rec = floor(c * 2^32 / p), q underestimates by <= 1) */
typedef struct { uint32_t c, rec; } p30_cc;

static inline p30_cc p30_cc_of(uint32_t c, uint32_t p){
    p30_cc t = { c, (uint32_t)(((p30_u128)c << 32) / p) };
    return t;
}

/* ------------------------------------------------------------------ */
/* plan: per-prime constants + prefix-stable twiddle tables. TWO prime */
/* SETS: binary sizes use P30_PR (largest 2^22-capable primes); mixed  */
/* 3*2^k sizes use P30_MPR (largest 3*2^20-capable primes -- needed    */
/* for the radix-3 twists; min-operand cap 2.86M, even better than     */
/* the binary set). Each set is a complete engine (twiddles, Garner).  */
/* ------------------------------------------------------------------ */
typedef struct { uint32_t c[16], rec[16]; } p30_lvec;

typedef struct {
    uint32_t p, p2, J, R1, R2;           /* J = -p^-1 mod 2^32 */
    p30_cc   inv2;                       /* (p+1)/2 */
    p30_cc  *w;                          /* BR root sequence (c, rec) */
    p30_cc  *winv;                       /* inverse roots */
    /* mixed-set extras: children are TWISTED to cyclic form by the
     * ring substitution x -> nu*x (nu^(16L) = zeta^t), which factors
     * per coefficient (vector v, lane l) as mu^(t*v) * nu_t^l. The
     * per-vector part mu^v / mu^2v lives in tw1/tw2 (built for the
     * TOP rung Ls = 2^16, smaller rungs L stride by Ls / L); the
     * per-lane part nu^l is one FIXED 16-lane vector per child per
     * rung (lv1/lv2/ilv1/ilv2, indexed by lg L; consumed via
     * p30_bar2pv -- NOTE the twist must be by x-degree, not vector
     * index: a vector-only twist breaks on the conv's x-carries). */
    p30_cc   zt, zt2;                    /* zeta, zeta^2 (cube roots) */
    p30_cc   i1z;                        /* 1/(1 - zeta) */
    p30_cc   c3;                         /* 3 (seam output scale) */
    p30_cc  *tw1, *tw2, *itw1, *itw2;    /* per-vector twist tables */
    p30_lvec *lv1, *lv2, *ilv1, *ilv2;   /* per-rung lane twists [lgL] */
    /* packed per-rung copies of the strided master rows (strided 8B
     * reads at large stride thrash; packed = sequential) */
    p30_cc  *rtw1[17], *rtw2[17], *ritw1[17], *ritw2[17];
} p30_prime;

typedef struct {
    const uint32_t *prs;                 /* the 5 prime values */
    p30_prime pr[P30_NP];
    int       lg;                        /* tables built for 2^lg      */
    int       maxlg;                     /* table allocation depth     */
    int       mixed;                     /* has radix-3 twist tables   */
    /* Garner constants (output stage) */
    uint32_t  inv[P30_NP], inv_rec[P30_NP];
    uint32_t  ppm[P30_NP][P30_NP], ppm_rec[P30_NP][P30_NP];
    uint32_t  w1c[1], w2c[2], w3c[3], w4c[4];   /* W_i chunks */
} p30_pset;

typedef struct {
    p30_pset b;                          /* binary-size engine */
    p30_pset m;                          /* mixed 3*2^k engine */
    int      init;
} p30_plan;

static inline void *p30_alloc(size_t bytes){
    void *q = NULL;
    if(posix_memalign(&q, 64, bytes)) abort();
    return q;
}

static inline uint32_t p30_primitive_root(uint32_t p){
    uint32_t f[16];
    int nf = 0;
    uint32_t n = p - 1u;
    for(uint32_t d = 2; (uint64_t)d * d <= n; d += (d == 2 ? 1 : 2))
        if(n % d == 0){ f[nf++] = d; while(n % d == 0) n /= d; }
    if(n > 1) f[nf++] = n;
    for(uint32_t g = 2; ; ++g){
        int ok = 1;
        for(int i = 0; i < nf && ok; ++i)
            if(p30_powm(g, (p - 1u) / f[i], p) == 1u) ok = 0;
        if(ok) return g;
    }
}

static void p30_kara_init(void);
static inline void p30_pset_init(p30_pset *ps, const uint32_t *prs,
                                 int maxlg, int mixed){
    ps->prs = prs;
    ps->maxlg = maxlg;
    ps->mixed = mixed;
    for(int q = 0; q < P30_NP; ++q){
        uint32_t p = prs[q], pinv = 1;
        for(int i = 0; i < 5; ++i) pinv *= 2u - p * pinv;
        p30_prime *pp = &ps->pr[q];
        pp->p = p;
        pp->p2 = 2u * p;
        pp->J = 0u - pinv;
        pp->R1 = (uint32_t)(((uint64_t)1 << 32) % p);
        pp->R2 = p30_mulm(pp->R1, pp->R1, p);
        pp->inv2 = p30_cc_of((p + 1u) >> 1, p);
        pp->w    = (p30_cc *)p30_alloc(sizeof(p30_cc) << maxlg);
        pp->winv = (p30_cc *)p30_alloc(sizeof(p30_cc) << maxlg);
        pp->w[0] = pp->winv[0] = p30_cc_of(1u, p);
        if(mixed){
            /* Omega of order F = 3*2^20 (p = c*3*2^20 + 1) */
            uint32_t g = p30_primitive_root(p);
            uint32_t Om = p30_powm(g, (p - 1u) / (3u << 20), p);
            uint32_t zt = p30_powm(Om, 1u << 20, p);     /* cube root */
            pp->zt  = p30_cc_of(zt, p);
            pp->zt2 = p30_cc_of(p30_mulm(zt, zt, p), p);
            pp->i1z = p30_cc_of(p30_invm((1u + p - zt) % p, p), p);
            pp->c3  = p30_cc_of(3u, p);
            /* twist tables for L* = 2^16: mu* = Om^(F/(3*2^16)) */
            uint32_t mu  = p30_powm(Om, 1u << 4, p);
            uint32_t mui = p30_invm(mu, p);
            size_t Ls = (size_t)1 << 16;
            pp->tw1  = (p30_cc *)p30_alloc(sizeof(p30_cc) * Ls);
            pp->tw2  = (p30_cc *)p30_alloc(sizeof(p30_cc) * Ls);
            pp->itw1 = (p30_cc *)p30_alloc(sizeof(p30_cc) * Ls);
            pp->itw2 = (p30_cc *)p30_alloc(sizeof(p30_cc) * Ls);
            uint32_t a1 = 1, a2 = 1, b1 = 1, b2 = 1;
            for(size_t v = 0; v < Ls; ++v){
                pp->tw1[v]  = p30_cc_of(a1, p);
                pp->tw2[v]  = p30_cc_of(a2, p);
                pp->itw1[v] = p30_cc_of(b1, p);
                pp->itw2[v] = p30_cc_of(b2, p);
                a1 = p30_mulm(a1, mu, p);
                a2 = p30_mulm(p30_mulm(a2, mu, p), mu, p);
                b1 = p30_mulm(b1, mui, p);
                b2 = p30_mulm(p30_mulm(b2, mui, p), mui, p);
            }
            /* per-rung lane twists: nu = Om^(2^(16-lgL)), nu^16 = mu_L */
            pp->lv1  = (p30_lvec *)p30_alloc(sizeof(p30_lvec) * 17);
            pp->lv2  = (p30_lvec *)p30_alloc(sizeof(p30_lvec) * 17);
            pp->ilv1 = (p30_lvec *)p30_alloc(sizeof(p30_lvec) * 17);
            pp->ilv2 = (p30_lvec *)p30_alloc(sizeof(p30_lvec) * 17);
            for(int lgL = 0; lgL <= 16; ++lgL){
                uint32_t nu  = p30_powm(Om, (uint32_t)1 << (16 - lgL), p);
                uint32_t nui = p30_invm(nu, p);
                uint32_t e1 = 1, e2 = 1, f1 = 1, f2 = 1;
                for(int l = 0; l < 16; ++l){
                    p30_cc t;
                    t = p30_cc_of(e1, p);
                    pp->lv1[lgL].c[l] = t.c;  pp->lv1[lgL].rec[l] = t.rec;
                    t = p30_cc_of(e2, p);
                    pp->lv2[lgL].c[l] = t.c;  pp->lv2[lgL].rec[l] = t.rec;
                    t = p30_cc_of(f1, p);
                    pp->ilv1[lgL].c[l] = t.c; pp->ilv1[lgL].rec[l] = t.rec;
                    t = p30_cc_of(f2, p);
                    pp->ilv2[lgL].c[l] = t.c; pp->ilv2[lgL].rec[l] = t.rec;
                    e1 = p30_mulm(e1, nu, p);
                    e2 = p30_mulm(p30_mulm(e2, nu, p), nu, p);
                    f1 = p30_mulm(f1, nui, p);
                    f2 = p30_mulm(p30_mulm(f2, nui, p), nui, p);
                }
            }
            for(int g = 0; g <= 16; ++g)
                pp->rtw1[g] = pp->rtw2[g] = pp->ritw1[g] = pp->ritw2[g] = NULL;
        }else{
            pp->tw1 = pp->tw2 = pp->itw1 = pp->itw2 = NULL;
            pp->lv1 = pp->lv2 = pp->ilv1 = pp->ilv2 = NULL;
        }
    }
    ps->lg = 0;
    for(int i = 1; i < P30_NP; ++i){
        uint32_t pi = prs[i];
        uint64_t pp = 1;
        for(int k = 0; k < i; ++k){
            ps->ppm[i][k] = (uint32_t)pp;
            ps->ppm_rec[i][k] = p30_cc_of((uint32_t)pp, pi).rec;
            pp = pp * (prs[k] % pi) % pi;
        }
        ps->inv[i] = p30_invm((uint32_t)pp, pi);
        ps->inv_rec[i] = p30_cc_of(ps->inv[i], pi).rec;
    }
    p30_u128 w = prs[0];
    ps->w1c[0] = (uint32_t)w;
    w *= prs[1];
    ps->w2c[0] = (uint32_t)w; ps->w2c[1] = (uint32_t)(w >> 32);
    w *= prs[2];
    ps->w3c[0] = (uint32_t)w; ps->w3c[1] = (uint32_t)(w >> 32);
    ps->w3c[2] = (uint32_t)(w >> 64);
    w *= prs[3];
    ps->w4c[0] = (uint32_t)w; ps->w4c[1] = (uint32_t)(w >> 32);
    ps->w4c[2] = (uint32_t)(w >> 64); ps->w4c[3] = (uint32_t)(w >> 96);
}

static inline void p30_plan_init(p30_plan *pl){
    if(pl->init) return;
    p30_pset_init(&pl->b, P30_PR, P30_MAX_LG, 0);
    p30_pset_init(&pl->m, P30_MPR, 16, 1);
    p30_kara_init();
    pl->init = 1;
}

/* grow the BR root tables to 2^lg entries (prefix-stable: append only).
 *   w[2^{t-1} + s] = zeta_t * w[s], zeta_t of order 2^t (tower-consistent) */
static inline void p30_pset_ensure(p30_pset *ps, int lg){
    if(lg <= ps->lg) return;
    for(int q = 0; q < P30_NP; ++q){
        p30_prime *pp = &ps->pr[q];
        uint32_t p = pp->p;
        uint32_t g = p30_primitive_root(p);
        for(int t = ps->lg + 1; t <= lg; ++t){
            uint32_t z  = p30_powm(g, (p - 1u) >> t, p);
            uint32_t zi = p30_invm(z, p);
            uint32_t blk = 1u << (t - 1);
            for(uint32_t s = 0; s < blk; ++s){
                pp->w[blk + s]    = p30_cc_of(p30_mulm(z,  pp->w[s].c,    p), p);
                pp->winv[blk + s] = p30_cc_of(p30_mulm(zi, pp->winv[s].c, p), p);
            }
        }
    }
    ps->lg = lg;
}
/* pack one rung's twist rows (lazy, per prime) */
static inline void p30_mix_rung(p30_pset *ps, int lgL){
    size_t L = (size_t)1 << lgL, st = (size_t)1 << (16 - lgL);
    for(int q = 0; q < P30_NP; ++q){
        p30_prime *pp = &ps->pr[q];
        if(pp->rtw1[lgL]) continue;
        p30_cc *a = (p30_cc *)p30_alloc(sizeof(p30_cc) * L);
        p30_cc *b = (p30_cc *)p30_alloc(sizeof(p30_cc) * L);
        p30_cc *c = (p30_cc *)p30_alloc(sizeof(p30_cc) * L);
        p30_cc *d = (p30_cc *)p30_alloc(sizeof(p30_cc) * L);
        for(size_t v = 0; v < L; ++v){
            a[v] = pp->tw1[v * st];
            b[v] = pp->tw2[v * st];
            c[v] = pp->itw1[v * st];
            d[v] = pp->itw2[v * st];
        }
        pp->rtw1[lgL] = a; pp->rtw2[lgL] = b;
        pp->ritw1[lgL] = c; pp->ritw2[lgL] = d;
    }
}

/* compat entry (benches/tests ensure the binary set) */
static inline void p30_plan_ensure(p30_plan *pl, int lg){
    p30_plan_init(pl);
    p30_pset_ensure(&pl->b, lg);
}

/* ------------------------------------------------------------------ */
/* SIMD modular primitives (stage 1: canonical [0,p) discipline)       */
/* ------------------------------------------------------------------ */
static inline __m512i p30_shrink(__m512i x, __m512i vp){
    return _mm512_min_epu32(x, _mm512_sub_epi32(x, vp));
}
static inline __m512i p30_addm(__m512i a, __m512i b, __m512i vp){
    return p30_shrink(_mm512_add_epi32(a, b), vp);
}
static inline __m512i p30_subm(__m512i a, __m512i b, __m512i vp){
    __m512i d = _mm512_sub_epi32(a, b);
    return _mm512_min_epu32(d, _mm512_add_epi32(d, vp));
}
/* a * c mod p in [0, 2p), any u32 a (broadcasts hoisted by caller) */
static inline __m512i p30_bar2p(__m512i a, __m512i vc, __m512i vrec, __m512i vp){
    __m512i pe = _mm512_mul_epu32(a, vrec);
    __m512i po = _mm512_mul_epu32(_mm512_srli_epi64(a, 32), vrec);
    __m512i q  = _mm512_mask_mov_epi32(_mm512_srli_epi64(pe, 32),
                                       (__mmask16)0xAAAA, po);
    return _mm512_sub_epi32(_mm512_mullo_epi32(a, vc),
                            _mm512_mullo_epi32(q, vp));
}
static inline __m512i p30_barm(__m512i a, __m512i vc, __m512i vrec, __m512i vp){
    return p30_shrink(p30_bar2p(a, vc, vrec, vp), vp);
}
/* Barrett with PER-LANE (c, rec): p30_bar2p assumes a BROADCAST rec
 * (its odd-lane quotient uses the even lane's rec); per-lane twiddle
 * vectors need the shifted rec plane (one extra vpsrlq). */
static inline __m512i p30_bar2pv(__m512i a, __m512i vc, __m512i vrec,
                                 __m512i vp){
    __m512i pe = _mm512_mul_epu32(a, vrec);
    __m512i po = _mm512_mul_epu32(_mm512_srli_epi64(a, 32),
                                  _mm512_srli_epi64(vrec, 32));
    __m512i q  = _mm512_mask_mov_epi32(_mm512_srli_epi64(pe, 32),
                                       (__mmask16)0xAAAA, po);
    return _mm512_sub_epi32(_mm512_mullo_epi32(a, vc),
                            _mm512_mullo_epi32(q, vp));
}

/* per-call hoisted twiddle broadcast */
typedef struct { __m512i c, rec; } p30_vcc;
static inline p30_vcc p30_vcc_of(p30_cc t){
    p30_vcc v = { _mm512_set1_epi32((int)t.c), _mm512_set1_epi32((int)t.rec) };
    return v;
}

/* ------------------------------------------------------------------ */
/* STAGE-2 range discipline: the between-pass invariant is LAZY        */
/* [0,4p). Barrett consumes multiplied operands raw (any u32); the     */
/* unmultiplied operand of a butterfly shrinks once ([0,4p)->[0,2p)).  */
/* Forward butterfly (FLINT convention, w = w2[j] = w[2j]):            */
/*   a' = sh2(a); wb = bar(b, w) in [0,2p)                             */
/*   c = a' + wb in [0,4p);  d = a' + (2p - wb) in (0,4p)              */
/* ------------------------------------------------------------------ */
static inline __m512i p30_sh2(__m512i x, __m512i vp2){
    return _mm512_min_epu32(x, _mm512_sub_epi32(x, vp2));
}

/* full-node radix-8 forward moth: 3 fused DIF levels per round-trip.
 * Twiddles (w2[x] = w[2x]): L1 w[2j]; L2 w[4j], w[4j+2];
 * L3 w[8j], w[8j+2], w[8j+4], w[8j+6]. All broadcast node constants. */
static void p30_fwd8(uint32_t *B, size_t m, size_t j,
                     const p30_prime *pp, __m512i vp, __m512i vp2){
    size_t e = m >> 3;
    p30_vcc t1  = p30_vcc_of(pp->w[2 * j]);
    p30_vcc t2a = p30_vcc_of(pp->w[4 * j]);
    p30_vcc t2b = p30_vcc_of(pp->w[4 * j + 2]);
    p30_vcc t30 = p30_vcc_of(pp->w[8 * j]);
    p30_vcc t31 = p30_vcc_of(pp->w[8 * j + 2]);
    p30_vcc t32 = p30_vcc_of(pp->w[8 * j + 4]);
    p30_vcc t33 = p30_vcc_of(pp->w[8 * j + 6]);
    for(size_t v = 0; v < e; ++v){
        __m512i x[8];
        for(int c = 0; c < 8; ++c)
            x[c] = _mm512_load_si512(B + P30_W * (v + (size_t)c * e));
#define P30_BF(lo_, hi_, T) do{                                          \
        __m512i a_  = p30_sh2(x[lo_], vp2);                              \
        __m512i wb_ = p30_bar2p(x[hi_], (T).c, (T).rec, vp);             \
        x[lo_] = _mm512_add_epi32(a_, wb_);                              \
        x[hi_] = _mm512_add_epi32(a_, _mm512_sub_epi32(vp2, wb_));       \
    }while(0)
        P30_BF(0, 4, t1); P30_BF(1, 5, t1);
        P30_BF(2, 6, t1); P30_BF(3, 7, t1);
        P30_BF(0, 2, t2a); P30_BF(1, 3, t2a);
        P30_BF(4, 6, t2b); P30_BF(5, 7, t2b);
        P30_BF(0, 1, t30); P30_BF(2, 3, t31);
        P30_BF(4, 5, t32); P30_BF(6, 7, t33);
#undef P30_BF
        for(int c = 0; c < 8; ++c)
            _mm512_store_si512(B + P30_W * (v + (size_t)c * e), x[c]);
    }
}

/* iterative full-subtree engines (defined after the leaf kernels):
 * same DFS pass order as the old recursion, flattened to a j-block
 * walk with in-register m <= 32 bottoms */
static void p30_fwd_full(uint32_t *B, size_t m, size_t j,
                         const p30_prime *pp, __m512i vp, __m512i vp2);
static void p30_fused_full(uint32_t *B, const uint32_t *A, size_t m,
                           size_t j, const p30_prime *pp, __m512i vp,
                           __m512i vp2, int sq);

/* truncated forward, node (B[0..m) vectors, j), output trunc zo.
 * Full nodes (zo == m) flow through the iterative engine; the
 * truncation spine and the m < 8 remainder levels run radix-2
 * (containment rule: no truncated radix-8/16/32 variants). */
static void p30_fwd_rec(uint32_t *B, size_t m, size_t j, size_t zo,
                        const p30_prime *pp, __m512i vp){
    if(m == 1) return;
    __m512i vp2 = _mm512_add_epi32(vp, vp);
    if(zo == m && m >= 8){
        p30_fwd_full(B, m, j, pp, vp, vp2);
        return;
    }
    size_t h = m >> 1;
    uint32_t *lo = B, *hi = B + P30_W * h;
    p30_vcc w = p30_vcc_of(pp->w[2 * j]);
    if(zo <= h){
        for(size_t v = 0; v < h; ++v){
            __m512i a  = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
            __m512i wb = p30_bar2p(_mm512_load_si512(hi + P30_W * v),
                                   w.c, w.rec, vp);
            _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(a, wb));
        }
        p30_fwd_rec(B, h, 2 * j, zo, pp, vp);
        return;
    }
    for(size_t v = 0; v < h; ++v){
        __m512i a  = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
        __m512i wb = p30_bar2p(_mm512_load_si512(hi + P30_W * v),
                               w.c, w.rec, vp);
        _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(a, wb));
        _mm512_store_si512(hi + P30_W * v,
            _mm512_add_epi32(a, _mm512_sub_epi32(vp2, wb)));
    }
    p30_fwd_rec(B, h, 2 * j, h, pp, vp);
    p30_fwd_rec(B + P30_W * h, h, 2 * j + 1, zo - h, pp, vp);
}

/* ------------------------------------------------------------------ */
/* truncated INVERSE invariant (proven in tests/p30_tft_model_test.c): */
/* entry B[0..z) = spectral, B[z..m) = m*u (scaled coefficient tails,  */
/* zeros at the top call); exit B[0..z) = m*u. The fused traversals    */
/* below implement it; the standalone general inverse it derives from  */
/* is p30_inv_rec in p30_ref.h.                                        */
/* ------------------------------------------------------------------ */
/* full-node radix-8 inverse moth: 3 fused DIT levels (L3, L2, L1).
 * Inverse butterfly: s = sh2(a) + sh2(b) in [0,4p);
 *                    d = bar(sh2(a) - sh2(b) + 2p, winv) in [0,2p).   */
static void p30_inv8(uint32_t *B, size_t m, size_t j,
                     const p30_prime *pp, __m512i vp, __m512i vp2){
    size_t e = m >> 3;
    p30_vcc t1  = p30_vcc_of(pp->winv[2 * j]);
    p30_vcc t2a = p30_vcc_of(pp->winv[4 * j]);
    p30_vcc t2b = p30_vcc_of(pp->winv[4 * j + 2]);
    p30_vcc t30 = p30_vcc_of(pp->winv[8 * j]);
    p30_vcc t31 = p30_vcc_of(pp->winv[8 * j + 2]);
    p30_vcc t32 = p30_vcc_of(pp->winv[8 * j + 4]);
    p30_vcc t33 = p30_vcc_of(pp->winv[8 * j + 6]);
    for(size_t v = 0; v < e; ++v){
        __m512i x[8];
        for(int c = 0; c < 8; ++c)
            x[c] = _mm512_load_si512(B + P30_W * (v + (size_t)c * e));
#define P30_IBF(lo_, hi_, T) do{                                         \
        __m512i a_ = p30_sh2(x[lo_], vp2);                               \
        __m512i b_ = p30_sh2(x[hi_], vp2);                               \
        x[lo_] = _mm512_add_epi32(a_, b_);                               \
        x[hi_] = p30_bar2p(_mm512_add_epi32(_mm512_sub_epi32(a_, b_),    \
                                            vp2),                       \
                           (T).c, (T).rec, vp);                          \
    }while(0)
        P30_IBF(0, 1, t30); P30_IBF(2, 3, t31);
        P30_IBF(4, 5, t32); P30_IBF(6, 7, t33);
        P30_IBF(0, 2, t2a); P30_IBF(1, 3, t2a);
        P30_IBF(4, 6, t2b); P30_IBF(5, 7, t2b);
        P30_IBF(0, 4, t1); P30_IBF(1, 5, t1);
        P30_IBF(2, 6, t1); P30_IBF(3, 7, t1);
#undef P30_IBF
        for(int c = 0; c < 8; ++c)
            _mm512_store_si512(B + P30_W * (v + (size_t)c * e), x[c]);
    }
}


/* ------------------------------------------------------------------ */
/* pointwise: 16-point twisted cyclic conv mod (x^16 - ww), ww = w[k]. */
/* fa <- fa * fb (canonical inputs/outputs). Windows are [aw | fa]     */
/* alignments (aw = fa * ww), b enters as lane broadcasts; u64 lane    */
/* sums (< 16 p^2 < 2^63.8) take one wide REDC -> stray R^-1 per       */
/* output, cancelled by the final scale constant.                      */
/* ------------------------------------------------------------------ */
static inline void p30_conv16(uint32_t *fa, const uint32_t *fb, p30_cc ww,
                              const p30_prime *pp, __m512i vp){
    const __m512i M32 = _mm512_set1_epi64(0xFFFFFFFFu);
    p30_vcc vw = p30_vcc_of(ww);
    __m512i vp2c = _mm512_add_epi32(vp, vp);
    /* fa arrives lazy [0,4p); fb must already be canonical [0,p) (the
     * caller pre-shrinks the b spectrum so its lanes can be broadcast
     * STRAIGHT FROM MEMORY: vpbroadcastd m32, one uop, no index-vector
     * register pressure -- the first-cut vpermd form spilled). */
    __m512i a = _mm512_load_si512(fa);
    a = p30_shrink(p30_sh2(a, vp2c), vp);
    __m512i aw = p30_barm(a, vw.c, vw.rec, vp);
    __m512i re = _mm512_setzero_si512(), ro = _mm512_setzero_si512();
#define P30_CONV_STEP(i) do{                                                  \
        __m512i bi_  = _mm512_set1_epi32((int)fb[i]);                         \
        __m512i win_ = (i) ? _mm512_alignr_epi32(a, aw, (16 - (i)) & 15) : a; \
        re = _mm512_add_epi64(re, _mm512_mul_epu32(win_, bi_));               \
        ro = _mm512_add_epi64(ro,                                             \
            _mm512_mul_epu32(_mm512_srli_epi64(win_, 32), bi_));              \
    }while(0)
    P30_CONV_STEP(0);  P30_CONV_STEP(1);  P30_CONV_STEP(2);  P30_CONV_STEP(3);
    P30_CONV_STEP(4);  P30_CONV_STEP(5);  P30_CONV_STEP(6);  P30_CONV_STEP(7);
    P30_CONV_STEP(8);  P30_CONV_STEP(9);  P30_CONV_STEP(10); P30_CONV_STEP(11);
    P30_CONV_STEP(12); P30_CONV_STEP(13); P30_CONV_STEP(14); P30_CONV_STEP(15);
#undef P30_CONV_STEP
    /* wide REDC: r = (t >> 32) + (m*p >> 32) + (lo32(t) != 0), r < 2^32 */
    const __m512i vJ = _mm512_set1_epi64(pp->J);
    const __m512i vP = _mm512_set1_epi64(pp->p);
    __m512i me = _mm512_mul_epu32(re, vJ), mo = _mm512_mul_epu32(ro, vJ);
    __m512i ue = _mm512_mul_epu32(me, vP), uo = _mm512_mul_epu32(mo, vP);
    __mmask8 ce = _mm512_test_epi64_mask(re, M32);
    __mmask8 co = _mm512_test_epi64_mask(ro, M32);
    __m512i se = _mm512_add_epi64(_mm512_srli_epi64(re, 32),
                                  _mm512_srli_epi64(ue, 32));
    __m512i so = _mm512_add_epi64(_mm512_srli_epi64(ro, 32),
                                  _mm512_srli_epi64(uo, 32));
    se = _mm512_mask_add_epi64(se, ce, se, _mm512_set1_epi64(1));
    so = _mm512_mask_add_epi64(so, co, so, _mm512_set1_epi64(1));
    /* r can reach hi32(16p^2) + p + 1 ~ 4.64e9 > 2^32: fold once in the
     * 64-bit lanes BEFORE the 32-bit pack (all-max inputs hit this) */
    const __m512i vP2w = _mm512_set1_epi64(pp->p2);
    se = _mm512_min_epu64(se, _mm512_sub_epi64(se, vP2w));
    so = _mm512_min_epu64(so, _mm512_sub_epi64(so, vP2w));
    /* pack even/odd output lanes, reduce (< 2.7p) to [0, p) */
    __m512i r = _mm512_mask_mov_epi32(se, (__mmask16)0xAAAA,
                                      _mm512_slli_epi64(so, 32));
    __m512i vp2 = _mm512_set1_epi32((int)pp->p2);
    r = _mm512_min_epu32(r, _mm512_sub_epi32(r, vp2));
    r = p30_shrink(r, vp);
    _mm512_store_si512(fa, r);
}


/* ==== V1: parity-split Karatsuba, 2 positions per call ============== */
/* twisted-window index vectors: W_i[l] = X[l-i] if (l&7) >= i else
 * ww*X[l-i+8]  ->  vpermt2d from (X : AW) with idx_i[l] =
 * (l&7) >= i ? l-i : 24+l-i. Shared constants (built once). */
static __m512i P30_KIDX[8];
static __m512i P30_ITLV_E, P30_ITLV_O, P30_DEIL_E, P30_DEIL_O, P30_ROT1;
static int p30_kara_ready;
static void p30_kara_init(void){
    if(p30_kara_ready) return;
    uint32_t t[16];
    for(int i = 1; i < 8; ++i){
        for(int l = 0; l < 16; ++l)
            t[l] = (uint32_t)(((l & 7) >= i) ? l - i : 24 + l - i);
        P30_KIDX[i] = _mm512_loadu_si512(t);
    }
    /* y * m: per-half rotate up by 1, wrapped lane gets ww*m */
    for(int l = 0; l < 16; ++l)
        t[l] = (uint32_t)(((l & 7) >= 1) ? l - 1 : 24 + l - 1);
    P30_ROT1 = _mm512_loadu_si512(t);
    /* deinterleave a position pair: E = even coeffs [k | k1], O = odd */
    for(int l = 0; l < 8; ++l){ t[l] = (uint32_t)(2 * l); t[8 + l] = (uint32_t)(16 + 2 * l); }
    P30_DEIL_E = _mm512_loadu_si512(t);
    for(int l = 0; l < 8; ++l){ t[l] = (uint32_t)(2 * l + 1); t[8 + l] = (uint32_t)(16 + 2 * l + 1); }
    P30_DEIL_O = _mm512_loadu_si512(t);
    /* re-interleave: position k from (CE half0, CO half0) etc. */
    for(int l = 0; l < 8; ++l){ t[2 * l] = (uint32_t)l; t[2 * l + 1] = (uint32_t)(16 + l); }
    P30_ITLV_E = _mm512_loadu_si512(t);
    for(int l = 0; l < 8; ++l){ t[2 * l] = (uint32_t)(8 + l); t[2 * l + 1] = (uint32_t)(16 + 8 + l); }
    P30_ITLV_O = _mm512_loadu_si512(t);
    p30_kara_ready = 1;
}

/* b-pair broadcast for step i: lanes 0..7 = B[i], lanes 8..15 = B[8+i]
 * (two immediate shuffles, no index registers) */
#define P30_KBCAST(B, i) \
    _mm512_shuffle_i32x4( \
        _mm512_shuffle_epi32((B), (_MM_PERM_ENUM)(((i) & 3) * 0x55)), \
        _mm512_shuffle_epi32((B), (_MM_PERM_ENUM)(((i) & 3) * 0x55)), \
        (int)((((i) >> 2) & 1) * 0x55 + 0xA0))
/* sel: lanes0,1 = src128[(i>>2)&1], lanes2,3 = src128[2 + ((i>>2)&1)]:
 * imm = b01b01 | 10+b 10+b << ...: (b) | (b)<<2 | (2+b)<<4 | (2+b)<<6
 *     = b*0x05 + 0xA0  with b = (i>>2)&1                              */

/* one twisted-8 batched conv: acc (re, ro) += conv(X via windows, B) */
#define P30_KCONV8(X, AW, B, RE, RO) do{                                     \
    RE = _mm512_add_epi64(RE, _mm512_mul_epu32((X), kb_tab[0]));         \
    RO = _mm512_add_epi64(RO,                                            \
        _mm512_mul_epu32(_mm512_srli_epi64((X), 32), kb_tab[0]));        \
    for(int i_ = 1; i_ < 8; ++i_){                                       \
        __m512i w_ = _mm512_permutex2var_epi32((X), P30_KIDX[i_], (AW));     \
        RE = _mm512_add_epi64(RE, _mm512_mul_epu32(w_, kb_tab[i_]));     \
        RO = _mm512_add_epi64(RO,                                        \
            _mm512_mul_epu32(_mm512_srli_epi64(w_, 32), kb_tab[i_]));    \
    }                                                                    \
}while(0)

static inline __m512i p30_kshrp(__m512i x, __m512i vp){    /* [0,2p)->[0,p) */
    return _mm512_min_epu32(x, _mm512_sub_epi32(x, vp));
}
static inline __m512i p30_kshrp3(__m512i x, __m512i vp, __m512i vp2){
    /* [0, ~2.65p) -> [0, p): one fold vs 2p, one vs p */
    x = _mm512_min_epu32(x, _mm512_sub_epi32(x, vp2));
    return _mm512_min_epu32(x, _mm512_sub_epi32(x, vp));
}
/* wide REDC of (re,ro) 64-bit accs (< 2^63.x) -> packed u32 in [0,2p) */
static inline __m512i p30_kredc(__m512i re, __m512i ro, __m512i vJ, __m512i vP,
                            __m512i vP2w){
    const __m512i M32 = _mm512_set1_epi64(0xFFFFFFFFu);
    __m512i me = _mm512_mul_epu32(re, vJ), mo = _mm512_mul_epu32(ro, vJ);
    __m512i ue = _mm512_mul_epu32(me, vP), uo = _mm512_mul_epu32(mo, vP);
    __mmask8 ce = _mm512_test_epi64_mask(re, M32);
    __mmask8 co = _mm512_test_epi64_mask(ro, M32);
    __m512i se = _mm512_add_epi64(_mm512_srli_epi64(re, 32),
                                  _mm512_srli_epi64(ue, 32));
    __m512i so = _mm512_add_epi64(_mm512_srli_epi64(ro, 32),
                                  _mm512_srli_epi64(uo, 32));
    se = _mm512_mask_add_epi64(se, ce, se, _mm512_set1_epi64(1));
    so = _mm512_mask_add_epi64(so, co, so, _mm512_set1_epi64(1));
    se = _mm512_min_epu64(se, _mm512_sub_epi64(se, vP2w));
    so = _mm512_min_epu64(so, _mm512_sub_epi64(so, vP2w));
    __m512i r = _mm512_mask_mov_epi32(se, (__mmask16)0xAAAA,
                                      _mm512_slli_epi64(so, 32));
    return r;          /* < 2.65p: caller folds to its target range */
}

/* fa[k], fa[k+1] *= fb (mod x^16 - ww_k / ww_k1); canonical inputs */
static void p30_conv16_kara2(uint32_t *fa, const uint32_t *fb,
                             p30_cc wwk, p30_cc wwk1,
                             const p30_prime *pp, __m512i vp){
    const __m512i vP2w = _mm512_set1_epi64(pp->p2);
    const __m512i vJ = _mm512_set1_epi64(pp->J);
    const __m512i vP = _mm512_set1_epi64(pp->p);
    __m512i vp2 = _mm512_add_epi32(vp, vp);
    /* per-pair vector twiddle constants (halves k / k1) */
    __m512i wc = _mm512_mask_mov_epi32(_mm512_set1_epi32((int)wwk.c),
                     (__mmask16)0xFF00, _mm512_set1_epi32((int)wwk1.c));
    __m512i wr = _mm512_mask_mov_epi32(_mm512_set1_epi32((int)wwk.rec),
                     (__mmask16)0xFF00, _mm512_set1_epi32((int)wwk1.rec));

    /* engine context: fa AND fb arrive LAZY [0,4p). fb is shrunk here
     * (canonical needed for the accumulator bound): each b-spectrum
     * vector is read by exactly one conv, so this replaces the old
     * standalone a-canonicalize pass at zero extra memory traffic. */
    __m512i xa0 = _mm512_load_si512(fa);
    __m512i xa1 = _mm512_load_si512(fa + 16);
    xa0 = p30_kshrp(p30_sh2(xa0, vp2), vp);
    xa1 = p30_kshrp(p30_sh2(xa1, vp2), vp);
    __m512i xb0 = _mm512_load_si512(fb);
    __m512i xb1 = _mm512_load_si512(fb + 16);
    xb0 = p30_kshrp(p30_sh2(xb0, vp2), vp);
    xb1 = p30_kshrp(p30_sh2(xb1, vp2), vp);
    /* parity deinterleave into [k | k1] halves */
    __m512i AE = _mm512_permutex2var_epi32(xa0, P30_DEIL_E, xa1);
    __m512i AO = _mm512_permutex2var_epi32(xa0, P30_DEIL_O, xa1);
    __m512i BE = _mm512_permutex2var_epi32(xb0, P30_DEIL_E, xb1);
    __m512i BO = _mm512_permutex2var_epi32(xb0, P30_DEIL_O, xb1);
    /* sums (< 2p -> 1 min to < p keeps the 8-product bound) */
    __m512i AS = p30_kshrp(_mm512_add_epi32(AE, AO), vp);
    __m512i BS = p30_kshrp(_mm512_add_epi32(BE, BO), vp);
    /* ww-scaled window companions, canonical */
    __m512i AWE = p30_kshrp(p30_bar2p(AE, wc, wr, vp), vp);
    __m512i AWO = p30_kshrp(p30_bar2p(AO, wc, wr, vp), vp);
    __m512i AWS = p30_kshrp(p30_bar2p(AS, wc, wr, vp), vp);

    /* b broadcast tables (8 each) -- registers, built by shuffles */
    __m512i kb_tab[8];
    __m512i re0 = _mm512_setzero_si512(), ro0 = _mm512_setzero_si512();
    __m512i re2 = _mm512_setzero_si512(), ro2 = _mm512_setzero_si512();
    __m512i res = _mm512_setzero_si512(), ros = _mm512_setzero_si512();
#define P30_KFILL(B) do{ \
        kb_tab[0] = P30_KBCAST(B, 0); kb_tab[1] = P30_KBCAST(B, 1);              \
        kb_tab[2] = P30_KBCAST(B, 2); kb_tab[3] = P30_KBCAST(B, 3);              \
        kb_tab[4] = P30_KBCAST(B, 4); kb_tab[5] = P30_KBCAST(B, 5);              \
        kb_tab[6] = P30_KBCAST(B, 6); kb_tab[7] = P30_KBCAST(B, 7);              \
    }while(0)
    P30_KFILL(BE); P30_KCONV8(AE, AWE, BE, re0, ro0);
    P30_KFILL(BO); P30_KCONV8(AO, AWO, BO, re2, ro2);
    P30_KFILL(BS); P30_KCONV8(AS, AWS, BS, res, ros);
#undef P30_KFILL
    /* NOTE: msum_acc - m0_acc - m2_acc is NOT exact in the raw
     * accumulator domain (the wrapped window terms use REDUCED ww*x
     * values: the identity holds only mod p, the u64 difference can go
     * negative). Combine AFTER REDC, in u32 with a +2p offset. */
    __m512i m0 = p30_kshrp3(p30_kredc(re0, ro0, vJ, vP, vP2w), vp, vp2);
    __m512i m2 = p30_kshrp3(p30_kredc(re2, ro2, vJ, vP, vP2w), vp, vp2);
    __m512i ms = p30_kshrp3(p30_kredc(res, ros, vJ, vP, vP2w), vp, vp2);
    __m512i m1 = _mm512_sub_epi32(
        _mm512_add_epi32(ms, vp2),
        _mm512_add_epi32(m0, m2));                /* (0, 3p) */
    /* c_even = m0 + y*m2: rotate m2 within halves, wrap gets ww*m2 */
    __m512i awm2 = p30_bar2p(m2, wc, wr, vp);     /* [0, 2p) */
    __m512i ym2 = _mm512_permutex2var_epi32(m2, P30_ROT1, awm2);
    __m512i CE = _mm512_add_epi32(m0, ym2);       /* < 3p */
    __m512i CO = m1;                              /* < 3p */
    /* re-interleave to per-position vectors, output lazy [0,4p) */
    _mm512_store_si512(fa,      _mm512_permutex2var_epi32(CE, P30_ITLV_E, CO));
    _mm512_store_si512(fa + 16, _mm512_permutex2var_epi32(CE, P30_ITLV_O, CO));
}

/* register-input kara2: p30_conv16_kara2 minus the fa memory round
 * trip. xa0/xa1 arrive LAZY [0,4p); outputs lazy (< 3p). */
static inline void p30_kara2r(__m512i *pa0, __m512i *pa1, const uint32_t *fb,
                              p30_cc wwk, p30_cc wwk1,
                              const p30_prime *pp, __m512i vp){
    const __m512i vP2w = _mm512_set1_epi64(pp->p2);
    const __m512i vJ = _mm512_set1_epi64(pp->J);
    const __m512i vP = _mm512_set1_epi64(pp->p);
    __m512i vp2 = _mm512_add_epi32(vp, vp);
    __m512i wc = _mm512_mask_mov_epi32(_mm512_set1_epi32((int)wwk.c),
                     (__mmask16)0xFF00, _mm512_set1_epi32((int)wwk1.c));
    __m512i wr = _mm512_mask_mov_epi32(_mm512_set1_epi32((int)wwk.rec),
                     (__mmask16)0xFF00, _mm512_set1_epi32((int)wwk1.rec));
    __m512i xa0 = p30_kshrp(p30_sh2(*pa0, vp2), vp);
    __m512i xa1 = p30_kshrp(p30_sh2(*pa1, vp2), vp);
    __m512i xb0 = _mm512_load_si512(fb);
    __m512i xb1 = _mm512_load_si512(fb + 16);
    xb0 = p30_kshrp(p30_sh2(xb0, vp2), vp);
    xb1 = p30_kshrp(p30_sh2(xb1, vp2), vp);
    __m512i AE = _mm512_permutex2var_epi32(xa0, P30_DEIL_E, xa1);
    __m512i AO = _mm512_permutex2var_epi32(xa0, P30_DEIL_O, xa1);
    __m512i BE = _mm512_permutex2var_epi32(xb0, P30_DEIL_E, xb1);
    __m512i BO = _mm512_permutex2var_epi32(xb0, P30_DEIL_O, xb1);
    __m512i AS = p30_kshrp(_mm512_add_epi32(AE, AO), vp);
    __m512i BS = p30_kshrp(_mm512_add_epi32(BE, BO), vp);
    __m512i AWE = p30_kshrp(p30_bar2p(AE, wc, wr, vp), vp);
    __m512i AWO = p30_kshrp(p30_bar2p(AO, wc, wr, vp), vp);
    __m512i AWS = p30_kshrp(p30_bar2p(AS, wc, wr, vp), vp);
    __m512i kb_tab[8];
    __m512i re0 = _mm512_setzero_si512(), ro0 = _mm512_setzero_si512();
    __m512i re2 = _mm512_setzero_si512(), ro2 = _mm512_setzero_si512();
    __m512i res = _mm512_setzero_si512(), ros = _mm512_setzero_si512();
#define P30_KFILLR(B) do{ \
        kb_tab[0] = P30_KBCAST(B, 0); kb_tab[1] = P30_KBCAST(B, 1);      \
        kb_tab[2] = P30_KBCAST(B, 2); kb_tab[3] = P30_KBCAST(B, 3);      \
        kb_tab[4] = P30_KBCAST(B, 4); kb_tab[5] = P30_KBCAST(B, 5);      \
        kb_tab[6] = P30_KBCAST(B, 6); kb_tab[7] = P30_KBCAST(B, 7);      \
    }while(0)
    P30_KFILLR(BE); P30_KCONV8(AE, AWE, BE, re0, ro0);
    P30_KFILLR(BO); P30_KCONV8(AO, AWO, BO, re2, ro2);
    P30_KFILLR(BS); P30_KCONV8(AS, AWS, BS, res, ros);
#undef P30_KFILLR
    __m512i m0 = p30_kshrp3(p30_kredc(re0, ro0, vJ, vP, vP2w), vp, vp2);
    __m512i m2 = p30_kshrp3(p30_kredc(re2, ro2, vJ, vP, vP2w), vp, vp2);
    __m512i ms = p30_kshrp3(p30_kredc(res, ros, vJ, vP, vP2w), vp, vp2);
    __m512i m1 = _mm512_sub_epi32(_mm512_add_epi32(ms, vp2),
                                  _mm512_add_epi32(m0, m2));
    __m512i awm2 = p30_bar2p(m2, wc, wr, vp);
    __m512i ym2 = _mm512_permutex2var_epi32(m2, P30_ROT1, awm2);
    __m512i CE = _mm512_add_epi32(m0, ym2);
    __m512i CO = m1;
    *pa0 = _mm512_permutex2var_epi32(CE, P30_ITLV_E, CO);
    *pa1 = _mm512_permutex2var_epi32(CE, P30_ITLV_O, CO);
}

/* ------------------------------------------------------------------ */
/* in-register bottoms (bench/p30_leaf_bench.c winners): the m <= 32   */
/* tails of full subtrees complete in one load/store round trip with   */
/* unrolled levels (the recursive radix-2 remainders were 2-4x off     */
/* kernel pace from per-node call + broadcast setup).                  */
/* ------------------------------------------------------------------ */
#define P30_BFR(lo_, hi_, T) do{                                         \
        __m512i a_  = p30_sh2(x[lo_], vp2);                              \
        __m512i wb_ = p30_bar2p(x[hi_], (T).c, (T).rec, vp);             \
        x[lo_] = _mm512_add_epi32(a_, wb_);                              \
        x[hi_] = _mm512_add_epi32(a_, _mm512_sub_epi32(vp2, wb_));       \
    }while(0)
#define P30_IBFR(lo_, hi_, T) do{                                        \
        __m512i a_ = p30_sh2(x[lo_], vp2);                               \
        __m512i b_ = p30_sh2(x[hi_], vp2);                               \
        x[lo_] = _mm512_add_epi32(a_, b_);                               \
        x[hi_] = p30_bar2p(_mm512_add_epi32(_mm512_sub_epi32(a_, b_),    \
                                            vp2),                       \
                           (T).c, (T).rec, vp);                          \
    }while(0)
/* c-output-only butterfly (truncated forward spine) */
#define P30_BFC(lo_, hi_, T) do{                                         \
        __m512i a_  = p30_sh2(x[lo_], vp2);                              \
        __m512i wb_ = p30_bar2p(x[hi_], (T).c, (T).rec, vp);             \
        x[lo_] = _mm512_add_epi32(a_, wb_);                              \
    }while(0)

/* the 4 forward levels of a full m=16 node, on a live register frame */
static inline void p30_fwd16x(__m512i x[16], size_t j, const p30_prime *pp,
                              __m512i vp, __m512i vp2){
    {   p30_vcc t = p30_vcc_of(pp->w[2*j]);
        P30_BFR(0,8,t); P30_BFR(1,9,t); P30_BFR(2,10,t); P30_BFR(3,11,t);
        P30_BFR(4,12,t); P30_BFR(5,13,t); P30_BFR(6,14,t); P30_BFR(7,15,t); }
    {   p30_vcc a = p30_vcc_of(pp->w[4*j]), b = p30_vcc_of(pp->w[4*j+2]);
        P30_BFR(0,4,a); P30_BFR(1,5,a); P30_BFR(2,6,a); P30_BFR(3,7,a);
        P30_BFR(8,12,b); P30_BFR(9,13,b); P30_BFR(10,14,b); P30_BFR(11,15,b); }
    for(int c = 0; c < 4; ++c){
        p30_vcc t = p30_vcc_of(pp->w[8*j + 2*(size_t)c]);
        P30_BFR(4*c, 4*c+2, t); P30_BFR(4*c+1, 4*c+3, t);
    }
    for(int c = 0; c < 8; ++c){
        p30_vcc t = p30_vcc_of(pp->w[16*j + 2*(size_t)c]);
        P30_BFR(2*c, 2*c+1, t);
    }
}
static inline void p30_inv16x(__m512i x[16], size_t j, const p30_prime *pp,
                              __m512i vp, __m512i vp2){
    for(int c = 0; c < 8; ++c){
        p30_vcc t = p30_vcc_of(pp->winv[16*j + 2*(size_t)c]);
        P30_IBFR(2*c, 2*c+1, t);
    }
    for(int c = 0; c < 4; ++c){
        p30_vcc t = p30_vcc_of(pp->winv[8*j + 2*(size_t)c]);
        P30_IBFR(4*c, 4*c+2, t); P30_IBFR(4*c+1, 4*c+3, t);
    }
    {   p30_vcc a = p30_vcc_of(pp->winv[4*j]), b = p30_vcc_of(pp->winv[4*j+2]);
        P30_IBFR(0,4,a); P30_IBFR(1,5,a); P30_IBFR(2,6,a); P30_IBFR(3,7,a);
        P30_IBFR(8,12,b); P30_IBFR(9,13,b); P30_IBFR(10,14,b); P30_IBFR(11,15,b); }
    {   p30_vcc t = p30_vcc_of(pp->winv[2*j]);
        P30_IBFR(0,8,t); P30_IBFR(1,9,t); P30_IBFR(2,10,t); P30_IBFR(3,11,t);
        P30_IBFR(4,12,t); P30_IBFR(5,13,t); P30_IBFR(6,14,t); P30_IBFR(7,15,t); }
}

static void p30_fwd16r(uint32_t *B, size_t j, const p30_prime *pp,
                       __m512i vp, __m512i vp2){
    __m512i x[16];
    for(int c = 0; c < 16; ++c) x[c] = _mm512_load_si512(B + P30_W * c);
    p30_fwd16x(x, j, pp, vp, vp2);
    for(int c = 0; c < 16; ++c) _mm512_store_si512(B + P30_W * c, x[c]);
}
static void p30_inv16r(uint32_t *B, size_t j, const p30_prime *pp,
                       __m512i vp, __m512i vp2){
    __m512i x[16];
    for(int c = 0; c < 16; ++c) x[c] = _mm512_load_si512(B + P30_W * c);
    p30_inv16x(x, j, pp, vp, vp2);
    for(int c = 0; c < 16; ++c) _mm512_store_si512(B + P30_W * c, x[c]);
}

/* m=32: radix-2 seam level over memory + two in-register 16s (beat
 * the moth + reg-4 decomposition in the leaf bench) */
static void p30_fwd32r(uint32_t *B, size_t j, const p30_prime *pp,
                       __m512i vp, __m512i vp2){
    p30_vcc w = p30_vcc_of(pp->w[2*j]);
    for(size_t v = 0; v < 16; ++v){
        __m512i a  = p30_sh2(_mm512_load_si512(B + P30_W * v), vp2);
        __m512i wb = p30_bar2p(_mm512_load_si512(B + P30_W * (v + 16)),
                               w.c, w.rec, vp);
        _mm512_store_si512(B + P30_W * v, _mm512_add_epi32(a, wb));
        _mm512_store_si512(B + P30_W * (v + 16),
            _mm512_add_epi32(a, _mm512_sub_epi32(vp2, wb)));
    }
    p30_fwd16r(B,       2*j,     pp, vp, vp2);
    p30_fwd16r(B + 256, 2*j + 1, pp, vp, vp2);
}
static void p30_inv32r(uint32_t *B, size_t j, const p30_prime *pp,
                       __m512i vp, __m512i vp2){
    p30_inv16r(B,       2*j,     pp, vp, vp2);
    p30_inv16r(B + 256, 2*j + 1, pp, vp, vp2);
    p30_vcc wi = p30_vcc_of(pp->winv[2*j]);
    for(size_t v = 0; v < 16; ++v){
        __m512i a = p30_sh2(_mm512_load_si512(B + P30_W * v), vp2);
        __m512i b = p30_sh2(_mm512_load_si512(B + P30_W * (v + 16)), vp2);
        _mm512_store_si512(B + P30_W * v, _mm512_add_epi32(a, b));
        _mm512_store_si512(B + P30_W * (v + 16),
            p30_bar2p(_mm512_add_epi32(_mm512_sub_epi32(a, b), vp2),
                      wi.c, wi.rec, vp));
    }
}

/* fused leaves: forward levels + kara conv + inverse levels per block */
static inline void p30_fused8l(uint32_t *B, const uint32_t *A, size_t j,
                               const p30_prime *pp, __m512i vp, __m512i vp2){
    p30_fwd8(B, 8, j, pp, vp, vp2);
    for(size_t t = 0; t < 4; ++t)
        p30_conv16_kara2(B + 32 * t, A + 32 * t,
                         pp->w[8*j + 2*t], pp->w[8*j + 2*t + 1], pp, vp);
    p30_inv8(B, 8, j, pp, vp, vp2);
}
/* m=16/32 fused leaves are SPLIT (fast bottoms + conv from L1): the
 * all-register form loses ~6% to spills inside kara (leaf bench,
 * "split" vs "reg" variants) -- holding the data frame across the
 * register-hungry conv costs more than the block's load/stores. */
static void p30_fused16l(uint32_t *B, const uint32_t *A, size_t j,
                         const p30_prime *pp, __m512i vp, __m512i vp2){
    p30_fwd16r(B, j, pp, vp, vp2);
    for(size_t t = 0; t < 8; ++t)
        p30_conv16_kara2(B + 32*t, A + 32*t,
                         pp->w[16*j + 2*t], pp->w[16*j + 2*t + 1], pp, vp);
    p30_inv16r(B, j, pp, vp, vp2);
}
static void p30_fused32l(uint32_t *B, const uint32_t *A, size_t j,
                         const p30_prime *pp, __m512i vp, __m512i vp2){
    p30_fwd32r(B, j, pp, vp, vp2);
    for(size_t t = 0; t < 16; ++t)
        p30_conv16_kara2(B + 32*t, A + 32*t,
                         pp->w[32*j + 2*t], pp->w[32*j + 2*t + 1], pp, vp);
    p30_inv32r(B, j, pp, vp, vp2);
}

/* ------------------------------------------------------------------ */
/* iterative full-subtree engines: DFS order flattened to a j-block    */
/* walk (zint-style). Leaf block size LS by lg m mod 3 (8/16/32); the  */
/* radix-8 moths above it fire on block alignment: a size-s node       */
/* STARTS at block b iff b == 0 mod (s/LS) (pre-order, largest first); */
/* it ENDS after block b iff b+1 == 0 mod (s/LS) (post-order, smallest */
/* first). Identical pass sequence to the old recursion, no calls.     */
/* ------------------------------------------------------------------ */
static inline size_t p30_leaf_ls(size_t m){     /* m > 32, power of 2 */
    int lgm = 0;
    while(((size_t)1 << lgm) < m) ++lgm;
    int r = lgm % 3;
    return r == 0 ? 8 : r == 1 ? 16 : 32;
}

static void p30_fwd_full(uint32_t *B, size_t m, size_t j,
                         const p30_prime *pp, __m512i vp, __m512i vp2){
    if(m <= 4){              /* tiny subtrees (pre-depth entry at small M) */
        if(m == 1) return;
        __m512i x[4];
        for(size_t c = 0; c < m; ++c) x[c] = _mm512_load_si512(B + P30_W * c);
        if(m == 2){
            p30_vcc t = p30_vcc_of(pp->w[2*j]); P30_BFR(0,1,t);
        }else{
            {   p30_vcc t = p30_vcc_of(pp->w[2*j]); P30_BFR(0,2,t); P30_BFR(1,3,t); }
            {   p30_vcc a = p30_vcc_of(pp->w[4*j]), b = p30_vcc_of(pp->w[4*j+2]);
                P30_BFR(0,1,a); P30_BFR(2,3,b); }
        }
        for(size_t c = 0; c < m; ++c) _mm512_store_si512(B + P30_W * c, x[c]);
        return;
    }
    if(m == 8){  p30_fwd8(B, 8, j, pp, vp, vp2);  return; }
    if(m == 16){ p30_fwd16r(B, j, pp, vp, vp2);   return; }
    if(m == 32){ p30_fwd32r(B, j, pp, vp, vp2);   return; }
    size_t LS = p30_leaf_ls(m);
    size_t nb = m / LS;
    for(size_t b = 0; b < nb; ++b){
        size_t o = b * LS;
        for(size_t s = m; s > LS; s >>= 3){          /* moths, top down */
            if(b & (s / LS - 1)) continue;
            p30_fwd8(B + P30_W * o, s, j * (m / s) + o / s, pp, vp, vp2);
        }
        uint32_t *Bb = B + P30_W * o;
        size_t jb = j * nb + b;
        if(LS == 8)       p30_fwd8(Bb, 8, jb, pp, vp, vp2);
        else if(LS == 16) p30_fwd16r(Bb, jb, pp, vp, vp2);
        else              p30_fwd32r(Bb, jb, pp, vp, vp2);
    }
}


static void p30_fused_full(uint32_t *B, const uint32_t *A, size_t m,
                           size_t j, const p30_prime *pp, __m512i vp,
                           __m512i vp2, int sq){
    if(m == 1){          /* conv16 broadcasts fb lanes: canonical copy */
        uint32_t tmp[16] __attribute__((aligned(64)));
        __m512i ax = _mm512_load_si512(sq ? B : A);
        _mm512_store_si512(tmp, p30_kshrp(p30_sh2(ax, vp2), vp));
        p30_conv16(B, tmp, pp->w[j], pp, vp);
        return;
    }
    if(m == 2){
        __m512i x[2];
        x[0] = _mm512_load_si512(B);
        x[1] = _mm512_load_si512(B + P30_W);
        {   p30_vcc t = p30_vcc_of(pp->w[2*j]); P30_BFR(0,1,t); }
        if(sq){                  /* squaring: leaf convs read the leaf */
            _mm512_store_si512(B,         x[0]);
            _mm512_store_si512(B + P30_W, x[1]);
            A = B;
        }
        p30_kara2r(&x[0], &x[1], A, pp->w[2*j], pp->w[2*j+1], pp, vp);
        {   p30_vcc t = p30_vcc_of(pp->winv[2*j]); P30_IBFR(0,1,t); }
        _mm512_store_si512(B,         x[0]);
        _mm512_store_si512(B + P30_W, x[1]);
        return;
    }
    if(m == 4){
        __m512i x[4];
        for(int c = 0; c < 4; ++c) x[c] = _mm512_load_si512(B + P30_W * c);
        {   p30_vcc t = p30_vcc_of(pp->w[2*j]); P30_BFR(0,2,t); P30_BFR(1,3,t); }
        {   p30_vcc a = p30_vcc_of(pp->w[4*j]), b = p30_vcc_of(pp->w[4*j+2]);
            P30_BFR(0,1,a); P30_BFR(2,3,b); }
        if(sq){
            for(int c = 0; c < 4; ++c)
                _mm512_store_si512(B + P30_W * c, x[c]);
            A = B;
        }
        p30_kara2r(&x[0], &x[1], A,      pp->w[4*j],   pp->w[4*j+1], pp, vp);
        p30_kara2r(&x[2], &x[3], A + 32, pp->w[4*j+2], pp->w[4*j+3], pp, vp);
        {   p30_vcc a = p30_vcc_of(pp->winv[4*j]), b = p30_vcc_of(pp->winv[4*j+2]);
            P30_IBFR(0,1,a); P30_IBFR(2,3,b); }
        {   p30_vcc t = p30_vcc_of(pp->winv[2*j]); P30_IBFR(0,2,t); P30_IBFR(1,3,t); }
        for(int c = 0; c < 4; ++c) _mm512_store_si512(B + P30_W * c, x[c]);
        return;
    }
    if(sq) A = B;                /* squaring: conv leaves against self */
    if(m == 8){  p30_fused8l(B, A, j, pp, vp, vp2);  return; }
    if(m == 16){ p30_fused16l(B, A, j, pp, vp, vp2); return; }
    if(m == 32){ p30_fused32l(B, A, j, pp, vp, vp2); return; }
    size_t LS = p30_leaf_ls(m);
    size_t nb = m / LS;
    for(size_t b = 0; b < nb; ++b){
        size_t o = b * LS;
        for(size_t s = m; s > LS; s >>= 3){
            if(b & (s / LS - 1)) continue;
            p30_fwd8(B + P30_W * o, s, j * (m / s) + o / s, pp, vp, vp2);
        }
        uint32_t *Bb = B + P30_W * o;
        const uint32_t *Ab = A + P30_W * o;
        size_t jb = j * nb + b;
        if(LS == 8)       p30_fused8l(Bb, Ab, jb, pp, vp, vp2);
        else if(LS == 16) p30_fused16l(Bb, Ab, jb, pp, vp, vp2);
        else              p30_fused32l(Bb, Ab, jb, pp, vp, vp2);
        for(size_t s = 8 * LS; s <= m; s <<= 3){
            if((b + 1) & (s / LS - 1)) continue;
            size_t os = o + LS - s;
            p30_inv8(B + P30_W * os, s, j * (m / s) + os / s, pp, vp, vp2);
        }
    }
}



/* ------------------------------------------------------------------ */
/* fused traversal with GENERAL inverse tails in a SIDE BUFFER.        */
/* Contract: entry B[0..m) = this node's b-coefficients; SB[v] (local  */
/* frame, v in [z, m)) = this node's inverse tails m*u_v; exit         */
/* B[0..z) = m*(a*b), SB consumed. The side buffer breaks the anti-    */
/* dependence that forced z > h right children to run UNFUSED (their  */
/* tails had to overwrite coefficient storage the forward still        */
/* needed): tail values now flow next to the data, so the whole spine  */
/* fuses. Derivation = the general truncated inverse (model test):     */
/*   z <= h:  child tails  h*c_v = (T_v + w*T_{v+h})*inv2, v in [z,h) */
/*            combine      m*u_v = 2*h*c_v - w*T_{v+h},    v in [0,z) */
/*   z >  h:  right tails  h*d_v = h*c_v - w*T_{v+h},      v in [zr,h)*/
/*            parent       m*u_v = h*c_v + h*d_v                       */
/* Slot frames nest exactly: a z > h right child's frame is SB + 16h,  */
/* and its tail slots coincide with the parent tails consumed in the   */
/* same loop iteration (read-then-write).                              */
/* The buffer is FREE: every spine node sits at global spectral base   */
/* g with g + z == tv, so tail slots map to global positions [tv, M)   */
/* -- the a-spectrum's DEAD tail (the conv reads only the demanded     */
/* prefix [0, tv); beyond it ra holds fwd leftovers). Since A walks    */
/* the recursion with the same offsets, SB == (uint32_t *)A: tails     */
/* ride in cache lines adjacent to the a-blocks being convolved.       */
/* ------------------------------------------------------------------ */
static void p30_fused_sb(uint32_t *B, const uint32_t *A, size_t m, size_t j,
                         size_t z, const p30_prime *pp, __m512i vp,
                         int sq){
    if(z == 0) return;
    __m512i vp2 = _mm512_add_epi32(vp, vp);
    if(z == m){
        p30_fused_full(B, A, m, j, pp, vp, vp2, sq);
        return;
    }
    uint32_t *SB = (uint32_t *)A;            /* dead beyond local z */
    size_t h = m >> 1;
    uint32_t *lo = B, *hi = B + P30_W * h;
    p30_vcc w = p30_vcc_of(pp->w[2 * j]);
    if(z <= h){
        for(size_t v = 0; v < h; ++v){       /* fwd: c half only */
            __m512i a  = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
            __m512i wb = p30_bar2p(_mm512_load_si512(hi + P30_W * v),
                                   w.c, w.rec, vp);
            _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(a, wb));
        }
        p30_vcc i2 = p30_vcc_of(pp->inv2);
        for(size_t v = z; v < h; ++v){       /* child tails, in place */
            __m512i t  = p30_sh2(_mm512_load_si512(SB + P30_W * v), vp2);
            __m512i wb = p30_bar2p(_mm512_load_si512(SB + P30_W * (v + h)),
                                   w.c, w.rec, vp);
            _mm512_store_si512(SB + P30_W * v,
                p30_bar2p(_mm512_add_epi32(t, wb), i2.c, i2.rec, vp));
        }
        p30_fused_sb(lo, A, h, 2 * j, z, pp, vp, sq);
        for(size_t v = 0; v < z; ++v){       /* m*u = 2*(h*c) - w*T   */
            __m512i a  = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
            __m512i u  = p30_sh2(_mm512_add_epi32(a, a), vp2);
            __m512i wb = p30_bar2p(_mm512_load_si512(SB + P30_W * (v + h)),
                                   w.c, w.rec, vp);
            _mm512_store_si512(lo + P30_W * v,
                _mm512_add_epi32(u, _mm512_sub_epi32(vp2, wb)));
        }
        return;
    }
    size_t zr = z - h;
    for(size_t v = 0; v < h; ++v){           /* fwd full level */
        __m512i a  = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
        __m512i wb = p30_bar2p(_mm512_load_si512(hi + P30_W * v),
                               w.c, w.rec, vp);
        _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(a, wb));
        _mm512_store_si512(hi + P30_W * v,
            _mm512_add_epi32(a, _mm512_sub_epi32(vp2, wb)));
    }
    p30_fused_full(lo, A, h, 2 * j, pp, vp, vp2, sq); /* left, fused */
    for(size_t v = zr; v < h; ++v){          /* right tails + parent out */
        __m512i c  = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
        __m512i wb = p30_bar2p(_mm512_load_si512(SB + P30_W * (v + h)),
                               w.c, w.rec, vp);
        __m512i d  = _mm512_add_epi32(c, _mm512_sub_epi32(vp2, wb));
        _mm512_store_si512(SB + P30_W * (v + h), d);
        _mm512_store_si512(lo + P30_W * v,
            _mm512_add_epi32(c, p30_sh2(d, vp2)));
    }
    p30_fused_sb(hi, A + P30_W * h, h, 2 * j + 1, zr, pp, vp, sq);
    p30_vcc wi = p30_vcc_of(pp->winv[2 * j]);
    for(size_t v = 0; v < zr; ++v){
        __m512i a = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
        __m512i b = p30_sh2(_mm512_load_si512(hi + P30_W * v), vp2);
        _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(a, b));
        _mm512_store_si512(hi + P30_W * v,
            p30_bar2p(_mm512_add_epi32(_mm512_sub_epi32(a, b), vp2),
                      wi.c, wi.rec, vp));
    }
}

/* ------------------------------------------------------------------ */
/* fused fwd(b) + pointwise + inverse traversal (depth-first: a leaf  */
/* block runs forward, conv against the matching a-spectrum block and */
/* the inverse while L1-hot; inverse combines happen on the ascent).  */
/* Node contract: entry B = this node's b-COEFFICIENTS; exit B[0..z)  */
/* = m * (a*b coefficients), B[z..m) unspecified. A = matching block  */
/* of the canonical a-spectrum.                                       */
/* Truncation-spine specializations rely on the zero-region property  */
/* propagating recursively (result positions >= z of a node are zero),*/
/* so the inverse tail-fills vanish and dead halves are never read.   */
/* ------------------------------------------------------------------ */
static void p30_fused_rec(uint32_t *B, const uint32_t *A, size_t m, size_t j,
                          size_t z, const p30_prime *pp, __m512i vp,
                          int sq){
    if(z == 0) return;
    __m512i vp2 = _mm512_add_epi32(vp, vp);
    if(m == 1){                              /* single spectral position */
        uint32_t tmp[16] __attribute__((aligned(64)));
        __m512i ax = _mm512_load_si512(sq ? B : A);
        _mm512_store_si512(tmp, p30_kshrp(p30_sh2(ax, vp2), vp));
        p30_conv16(B, tmp, pp->w[j], pp, vp);
        return;
    }
    if(z == m){
        p30_fused_full(B, A, m, j, pp, vp, vp2, sq);
        return;
    }
    size_t h = m >> 1;
    uint32_t *lo = B, *hi = B + P30_W * h;
    p30_vcc w = p30_vcc_of(pp->w[2 * j]);
    if(z <= h){
        /* fwd: c = sh2(a) + w*b into lo (right output half not wanted) */
        for(size_t v = 0; v < h; ++v){
            __m512i a  = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
            __m512i wb = p30_bar2p(_mm512_load_si512(hi + P30_W * v),
                                   w.c, w.rec, vp);
            _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(a, wb));
        }
        p30_fused_rec(lo, A, h, 2 * j, z, pp, vp, sq);
        /* inverse, z <= h with zero hi-half: m*u = 2*(h*c) */
        for(size_t v = 0; v < z; ++v){
            __m512i x = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
            _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(x, x));
        }
        return;
    }
    /* z > h. The right child's inverse tails are h*d_v = h*c_v (the
     * LEFT child's outputs, zero parent tails) -- NOT zero. They go
     * to the SIDE BUFFER so the right child runs FULLY FUSED
     * (p30_fused_sb); coefficient storage is never overwritten before
     * its forward consumes it, which is what used to force the plain
     * fwd -> conv -> general-inverse detour here. */
    size_t zr = z - h;
    for(size_t v = 0; v < h; ++v){           /* fwd full level */
        __m512i a  = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
        __m512i wb = p30_bar2p(_mm512_load_si512(hi + P30_W * v),
                               w.c, w.rec, vp);
        _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(a, wb));
        _mm512_store_si512(hi + P30_W * v,
            _mm512_add_epi32(a, _mm512_sub_epi32(vp2, wb)));
    }
    p30_fused_full(lo, A, h, 2 * j, pp, vp, vp2, sq); /* left, fused */
    {   /* seed right-child tails into the dead a-spectrum tail */
        uint32_t *SBc = (uint32_t *)(A + P30_W * h);
        for(size_t v = zr; v < h; ++v){      /* h*d = h*c, parent out */
            __m512i c = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
            _mm512_store_si512(SBc + P30_W * v, c);
            _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(c, c));
        }
    }
    p30_fused_sb(hi, A + P30_W * h, h, 2 * j + 1, zr, pp, vp, sq);
    p30_vcc wi = p30_vcc_of(pp->winv[2 * j]);
    for(size_t v = 0; v < zr; ++v){
        __m512i a = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
        __m512i b = p30_sh2(_mm512_load_si512(hi + P30_W * v), vp2);
        _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(a, b));
        _mm512_store_si512(hi + P30_W * v,
            p30_bar2p(_mm512_add_epi32(_mm512_sub_epi32(a, b), vp2),
                      wi.c, wi.rec, vp));
    }
}

/* ------------------------------------------------------------------ */
/* pre-depth entry drivers: the fused INPUT pass absorbs the first 3   */
/* transform levels (exactly one radix-8 moth depth), so the           */
/* traversals enter d levels down; d == 0 falls through to the normal  */
/* engines. The fwd passes those levels would have done are simply     */
/* skipped; everything inverse-side (tails, combines) is unchanged.    */
/* Gating guarantees M >= 8, so m >> d == M >> 3 >= 1 throughout and   */
/* m == 1 implies d == 0.                                              */
/* ------------------------------------------------------------------ */
static void p30_fwd_pre(uint32_t *B, size_t m, size_t j, size_t z, int d,
                        const p30_prime *pp, __m512i vp){
    if(d == 0){ p30_fwd_rec(B, m, j, z, pp, vp); return; }
    if(z == m){
        __m512i vp2 = _mm512_add_epi32(vp, vp);
        size_t sub = m >> d;
        for(size_t i = 0; i < ((size_t)1 << d); ++i)
            p30_fwd_full(B + P30_W * sub * i, sub, (j << d) + i,
                         pp, vp, vp2);
        return;
    }
    size_t h = m >> 1;
    if(z <= h){ p30_fwd_pre(B, h, 2 * j, z, d - 1, pp, vp); return; }
    p30_fwd_pre(B, h, 2 * j, h, d - 1, pp, vp);
    p30_fwd_pre(B + P30_W * h, h, 2 * j + 1, z - h, d - 1, pp, vp);
}

static void p30_fused_full_pre(uint32_t *B, const uint32_t *A, size_t m,
                               size_t j, int d, const p30_prime *pp,
                               __m512i vp, __m512i vp2, int sq){
    if(d == 0){ p30_fused_full(B, A, m, j, pp, vp, vp2, sq); return; }
    if(d == 3){             /* the absorbed levels == this node's moth */
        size_t e = m >> 3;
        for(size_t c = 0; c < 8; ++c)
            p30_fused_full(B + P30_W * e * c, A + P30_W * e * c,
                           e, 8 * j + c, pp, vp, vp2, sq);
        p30_inv8(B, m, j, pp, vp, vp2);
        return;
    }
    size_t h = m >> 1;      /* d = 1, 2 (spine-adjacent): radix-2 form */
    p30_fused_full_pre(B, A, h, 2 * j, d - 1, pp, vp, vp2, sq);
    p30_fused_full_pre(B + P30_W * h, A + P30_W * h, h, 2 * j + 1, d - 1,
                       pp, vp, vp2, sq);
    p30_vcc wi = p30_vcc_of(pp->winv[2 * j]);
    uint32_t *lo = B, *hi = B + P30_W * h;
    for(size_t v = 0; v < h; ++v){
        __m512i a = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
        __m512i b = p30_sh2(_mm512_load_si512(hi + P30_W * v), vp2);
        _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(a, b));
        _mm512_store_si512(hi + P30_W * v,
            p30_bar2p(_mm512_add_epi32(_mm512_sub_epi32(a, b), vp2),
                      wi.c, wi.rec, vp));
    }
}

static void p30_fused_sb_pre(uint32_t *B, const uint32_t *A, size_t m,
                             size_t j, size_t z, int d,
                             const p30_prime *pp, __m512i vp, int sq){
    if(d == 0){ p30_fused_sb(B, A, m, j, z, pp, vp, sq); return; }
    __m512i vp2 = _mm512_add_epi32(vp, vp);
    if(z == m){ p30_fused_full_pre(B, A, m, j, d, pp, vp, vp2, sq); return; }
    uint32_t *SB = (uint32_t *)A;            /* dead beyond local z */
    size_t h = m >> 1;
    uint32_t *lo = B, *hi = B + P30_W * h;
    p30_vcc w = p30_vcc_of(pp->w[2 * j]);
    if(z <= h){
        /* fwd c-pass absorbed by the input pass */
        p30_vcc i2 = p30_vcc_of(pp->inv2);
        for(size_t v = z; v < h; ++v){       /* child tails, in place */
            __m512i t  = p30_sh2(_mm512_load_si512(SB + P30_W * v), vp2);
            __m512i wb = p30_bar2p(_mm512_load_si512(SB + P30_W * (v + h)),
                                   w.c, w.rec, vp);
            _mm512_store_si512(SB + P30_W * v,
                p30_bar2p(_mm512_add_epi32(t, wb), i2.c, i2.rec, vp));
        }
        p30_fused_sb_pre(lo, A, h, 2 * j, z, d - 1, pp, vp, sq);
        for(size_t v = 0; v < z; ++v){       /* m*u = 2*(h*c) - w*T   */
            __m512i a  = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
            __m512i u  = p30_sh2(_mm512_add_epi32(a, a), vp2);
            __m512i wb = p30_bar2p(_mm512_load_si512(SB + P30_W * (v + h)),
                                   w.c, w.rec, vp);
            _mm512_store_si512(lo + P30_W * v,
                _mm512_add_epi32(u, _mm512_sub_epi32(vp2, wb)));
        }
        return;
    }
    size_t zr = z - h;                       /* fwd full level absorbed */
    p30_fused_full_pre(lo, A, h, 2 * j, d - 1, pp, vp, vp2, sq);
    for(size_t v = zr; v < h; ++v){          /* right tails + parent out */
        __m512i c  = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
        __m512i wb = p30_bar2p(_mm512_load_si512(SB + P30_W * (v + h)),
                               w.c, w.rec, vp);
        __m512i dd = _mm512_add_epi32(c, _mm512_sub_epi32(vp2, wb));
        _mm512_store_si512(SB + P30_W * (v + h), dd);
        _mm512_store_si512(lo + P30_W * v,
            _mm512_add_epi32(c, p30_sh2(dd, vp2)));
    }
    p30_fused_sb_pre(hi, A + P30_W * h, h, 2 * j + 1, zr, d - 1, pp, vp,
                     sq);
    p30_vcc wi = p30_vcc_of(pp->winv[2 * j]);
    for(size_t v = 0; v < zr; ++v){
        __m512i a = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
        __m512i b = p30_sh2(_mm512_load_si512(hi + P30_W * v), vp2);
        _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(a, b));
        _mm512_store_si512(hi + P30_W * v,
            p30_bar2p(_mm512_add_epi32(_mm512_sub_epi32(a, b), vp2),
                      wi.c, wi.rec, vp));
    }
}

static void p30_fused_pre(uint32_t *B, const uint32_t *A, size_t m, size_t j,
                          size_t z, int d, const p30_prime *pp, __m512i vp,
                          int sq){
    if(d == 0){ p30_fused_rec(B, A, m, j, z, pp, vp, sq); return; }
    __m512i vp2 = _mm512_add_epi32(vp, vp);
    if(z == m){ p30_fused_full_pre(B, A, m, j, d, pp, vp, vp2, sq); return; }
    size_t h = m >> 1;
    uint32_t *lo = B, *hi = B + P30_W * h;
    if(z <= h){                              /* fwd c-pass absorbed */
        p30_fused_pre(lo, A, h, 2 * j, z, d - 1, pp, vp, sq);
        for(size_t v = 0; v < z; ++v){       /* zero tails: m*u = 2*(h*c) */
            __m512i x = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
            _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(x, x));
        }
        return;
    }
    size_t zr = z - h;                       /* fwd full level absorbed */
    p30_fused_full_pre(lo, A, h, 2 * j, d - 1, pp, vp, vp2, sq);
    {   /* seed right-child tails into the dead a-spectrum tail */
        uint32_t *SBc = (uint32_t *)(A + P30_W * h);
        for(size_t v = zr; v < h; ++v){      /* h*d = h*c, parent out */
            __m512i c = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
            _mm512_store_si512(SBc + P30_W * v, c);
            _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(c, c));
        }
    }
    p30_fused_sb_pre(hi, A + P30_W * h, h, 2 * j + 1, zr, d - 1, pp, vp,
                     sq);
    p30_vcc wi = p30_vcc_of(pp->winv[2 * j]);
    for(size_t v = 0; v < zr; ++v){
        __m512i a = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
        __m512i b = p30_sh2(_mm512_load_si512(hi + P30_W * v), vp2);
        _mm512_store_si512(lo + P30_W * v, _mm512_add_epi32(a, b));
        _mm512_store_si512(hi + P30_W * v,
            p30_bar2p(_mm512_add_epi32(_mm512_sub_epi32(a, b), vp2),
                      wi.c, wi.rec, vp));
    }
}

/* ------------------------------------------------------------------ */
/* mixed-radix 3*2^k transform (EXPERIMENTAL, ladder sizes; math       */
/* proven in tests/p30_tft3_model_test.c, docs §15). N = 3L vectors,   */
/* tv in (2L, 3L] => the radix-3 level is ALWAYS FULL and truncation   */
/* lives entirely inside child 2. Children are TWISTED to cyclic form  */
/* (child t coefficient i scaled by mu^(t*i), mu^L = zeta), so ALL     */
/* binary machinery -- tables, traversals, convs -- applies verbatim   */
/* with j = 0 local indexing. The inverse seam (docs §15):             */
/*   a_{v+L} = (b0 - b1)/(1 - zeta);  a_v = b0 - a_{v+L};              */
/*   child-2 tails b2 = a_v + zeta^2 a_{v+L}    (v in [z2, L))         */
/* with untwist/retwist folded into the seam multiplies. Outputs are   */
/* 3L * u (seam region scaled by 3 explicitly; the 3-point combine is  */
/* left unscaled, giving the 3 for free); final scale (3L)^-1 * R.     */
/* ------------------------------------------------------------------ */
static void p30_r3fwd(uint32_t *B, size_t L, int lgL,
                      const p30_prime *pp, __m512i vp, __m512i vp2){
    const p30_cc *tw1 = pp->rtw1[lgL], *tw2 = pp->rtw2[lgL];
    p30_vcc z1 = p30_vcc_of(pp->zt), z2 = p30_vcc_of(pp->zt2);
    __m512i lc1 = _mm512_loadu_si512(pp->lv1[lgL].c);
    __m512i lr1 = _mm512_loadu_si512(pp->lv1[lgL].rec);
    __m512i lc2 = _mm512_loadu_si512(pp->lv2[lgL].c);
    __m512i lr2 = _mm512_loadu_si512(pp->lv2[lgL].rec);
    for(size_t v = 0; v < L; ++v){
        __m512i a = _mm512_load_si512(B + P30_W * v);
        __m512i b = _mm512_load_si512(B + P30_W * (v + L));
        __m512i d = _mm512_load_si512(B + P30_W * (v + 2 * L));
        /* inputs canonical [0,p) (decode pass) */
        __m512i t0 = _mm512_add_epi32(_mm512_add_epi32(a, b), d);
        __m512i y1 = _mm512_add_epi32(
            p30_sh2(_mm512_add_epi32(a, p30_bar2p(b, z1.c, z1.rec, vp)), vp2),
            p30_bar2p(d, z2.c, z2.rec, vp));
        __m512i y2 = _mm512_add_epi32(
            p30_sh2(_mm512_add_epi32(a, p30_bar2p(b, z2.c, z2.rec, vp)), vp2),
            p30_bar2p(d, z1.c, z1.rec, vp));
        p30_vcc w1 = p30_vcc_of(tw1[v]);
        p30_vcc w2 = p30_vcc_of(tw2[v]);
        _mm512_store_si512(B + P30_W * v, t0);
        _mm512_store_si512(B + P30_W * (v + L),
            p30_bar2pv(p30_bar2p(y1, w1.c, w1.rec, vp), lc1, lr1, vp));
        _mm512_store_si512(B + P30_W * (v + 2 * L),
            p30_bar2pv(p30_bar2p(y2, w2.c, w2.rec, vp), lc2, lr2, vp));
    }
}

static void p30_mfwd(uint32_t *B, size_t L, size_t z2v, int lgL,
                     const p30_prime *pp, __m512i vp){
    __m512i vp2 = _mm512_add_epi32(vp, vp);
    p30_r3fwd(B, L, lgL, pp, vp, vp2);
    p30_fwd_rec(B,                 L, 0, L,   pp, vp);
    p30_fwd_rec(B + P30_W * L,     L, 0, L,   pp, vp);
    p30_fwd_rec(B + P30_W * 2 * L, L, 0, z2v, pp, vp);
}

static void p30_mfused(uint32_t *B, const uint32_t *A, size_t L, size_t z2v,
                       int lgL, const p30_prime *pp, __m512i vp, int sq){
    __m512i vp2 = _mm512_add_epi32(vp, vp);
    const p30_cc *tw2 = pp->rtw2[lgL];
    const p30_cc *itw1 = pp->ritw1[lgL], *itw2 = pp->ritw2[lgL];
    p30_r3fwd(B, L, lgL, pp, vp, vp2);
    p30_fused_full(B,             A,             L, 0, pp, vp, vp2, sq);
    p30_fused_full(B + P30_W * L, A + P30_W * L, L, 0, pp, vp, vp2, sq);
    uint32_t *lo = B, *mid = B + P30_W * L, *hi = B + P30_W * 2 * L;
    p30_vcc zz1 = p30_vcc_of(pp->zt), zz2 = p30_vcc_of(pp->zt2);
    p30_vcc i1z = p30_vcc_of(pp->i1z), c3 = p30_vcc_of(pp->c3);
    __m512i lc2  = _mm512_loadu_si512(pp->lv2[lgL].c);
    __m512i lr2  = _mm512_loadu_si512(pp->lv2[lgL].rec);
    __m512i ilc1 = _mm512_loadu_si512(pp->ilv1[lgL].c);
    __m512i ilr1 = _mm512_loadu_si512(pp->ilv1[lgL].rec);
    __m512i ilc2 = _mm512_loadu_si512(pp->ilv2[lgL].c);
    __m512i ilr2 = _mm512_loadu_si512(pp->ilv2[lgL].rec);
    {   /* seam: 2x2 solve, child-2 tails into A's dead tail */
        uint32_t *SB2 = (uint32_t *)(A + P30_W * 2 * L);
        for(size_t v = z2v; v < L; ++v){
            p30_vcc it1 = p30_vcc_of(itw1[v]);
            p30_vcc rw2 = p30_vcc_of(tw2[v]);
            __m512i c0 = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
            __m512i c1 = p30_bar2pv(
                p30_bar2p(_mm512_load_si512(mid + P30_W * v),
                          it1.c, it1.rec, vp),
                ilc1, ilr1, vp);                            /* untwist */
            __m512i u1 = p30_bar2p(
                _mm512_add_epi32(c0, _mm512_sub_epi32(vp2, c1)),
                i1z.c, i1z.rec, vp);                        /* L*a_{v+L} */
            __m512i u0 = p30_sh2(
                _mm512_add_epi32(c0, _mm512_sub_epi32(vp2, u1)), vp2);
            __m512i t = p30_sh2(
                _mm512_add_epi32(u0, p30_bar2p(u1, zz2.c, zz2.rec, vp)),
                vp2);                                       /* L*b2_v */
            _mm512_store_si512(SB2 + P30_W * v,
                p30_bar2pv(p30_bar2p(t, rw2.c, rw2.rec, vp),
                           lc2, lr2, vp));                  /* retwist */
            _mm512_store_si512(lo + P30_W * v,
                               p30_bar2p(u0, c3.c, c3.rec, vp));
            _mm512_store_si512(mid + P30_W * v,
                               p30_bar2p(u1, c3.c, c3.rec, vp));
        }
    }
    p30_fused_sb(hi, A + P30_W * 2 * L, L, 0, z2v, pp, vp, sq);
    for(size_t v = 0; v < z2v; ++v){      /* 3-pt combine, unscaled */
        p30_vcc it1 = p30_vcc_of(itw1[v]);
        p30_vcc it2 = p30_vcc_of(itw2[v]);
        __m512i c0 = p30_sh2(_mm512_load_si512(lo + P30_W * v), vp2);
        __m512i c1 = p30_bar2pv(
            p30_bar2p(_mm512_load_si512(mid + P30_W * v),
                      it1.c, it1.rec, vp), ilc1, ilr1, vp);
        __m512i c2 = p30_bar2pv(
            p30_bar2p(_mm512_load_si512(hi + P30_W * v),
                      it2.c, it2.rec, vp), ilc2, ilr2, vp);
        __m512i s0 = _mm512_add_epi32(
            p30_sh2(_mm512_add_epi32(c0, c1), vp2), c2);
        __m512i s1 = _mm512_add_epi32(
            p30_sh2(_mm512_add_epi32(c0, p30_bar2p(c1, zz2.c, zz2.rec, vp)),
                    vp2),
            p30_bar2p(c2, zz1.c, zz1.rec, vp));
        __m512i s2 = _mm512_add_epi32(
            p30_sh2(_mm512_add_epi32(c0, p30_bar2p(c1, zz1.c, zz1.rec, vp)),
                    vp2),
            p30_bar2p(c2, zz2.c, zz2.rec, vp));
        _mm512_store_si512(lo  + P30_W * v, s0);
        _mm512_store_si512(mid + P30_W * v, s1);
        _mm512_store_si512(hi  + P30_W * v, s2);
    }
}






/* ------------------------------------------------------------------ */
/* input: u64 limbs -> 5 residue arrays in [0, p), one fused pass      */
/* (bench/p30_io_bench.c kernel + one extra shrink to canonical)       */
/* ------------------------------------------------------------------ */
static inline __m512i p30_in_redc8(__m512i x, __m512i R1, __m512i R2,
                                   __m512i J, __m512i p){
    __m512i hi = _mm512_srli_epi64(x, 32);
    __m512i t  = _mm512_add_epi64(_mm512_mul_epu32(x, R1),
                                  _mm512_mul_epu32(hi, R2));
    __m512i m  = _mm512_mul_epu32(t, J);
    return _mm512_srli_epi64(_mm512_add_epi64(t, _mm512_mul_epu32(m, p)), 32);
}
static inline __m512i p30_pack_idx(void){
    static const uint32_t idx[16] = { 0, 2, 4, 6, 8, 10, 12, 14,
                                      16, 18, 20, 22, 24, 26, 28, 30 };
    return _mm512_loadu_si512(idx);
}
/* nv = buffer size in vectors; limbs beyond n decode to zero */
static inline void p30_input(uint32_t *out[P30_NP], const uint64_t *a,
                             size_t n, size_t nv, const p30_pset *pl){
    const __m512i PK = p30_pack_idx();
    size_t i = 0;
    for(; i + P30_W <= n; i += P30_W){
        __m512i x0 = _mm512_loadu_si512(a + i);
        __m512i x1 = _mm512_loadu_si512(a + i + 8);
        for(int q = 0; q < P30_NP; ++q){
            const p30_prime *pp = &pl->pr[q];
            __m512i R1 = _mm512_set1_epi64(pp->R1);
            __m512i R2 = _mm512_set1_epi64(pp->R2);
            __m512i J  = _mm512_set1_epi64(pp->J);
            __m512i p  = _mm512_set1_epi64(pp->p);
            __m512i r0 = p30_in_redc8(x0, R1, R2, J, p);
            __m512i r1 = p30_in_redc8(x1, R1, R2, J, p);
            __m512i r  = _mm512_permutex2var_epi32(r0, PK, r1);
            r = p30_shrink(r, _mm512_set1_epi32((int)pp->p2));
            r = p30_shrink(r, _mm512_set1_epi32((int)pp->p));
            _mm512_store_si512(out[q] + i, r);
        }
    }
    for(; i < n; ++i)
        for(int q = 0; q < P30_NP; ++q)
            out[q][i] = (uint32_t)(a[i] % pl->pr[q].p);
    for(int q = 0; q < P30_NP; ++q)
        memset(out[q] + n, 0, (nv * P30_W - n) * 4);
}

/* ------------------------------------------------------------------ */
/* fused input: decode + the FIRST 3 forward levels in one tiled       */
/* block-column pass (probe bench/p30_fuse_probe.c, F2 winner: 16-     */
/* column ~1KB bursts, the limb tile stays L1/L2-hot across the 5      */
/* primes, NT stores size-gated). Matrix view: M vectors = 8 rows x    */
/* C = M/8 columns; row r = depth-3 subtree r; truncation demands      */
/* rows < zrow = ceil(tv / C) (zrow in (4, 8] for any minimal M).      */
/* Limbs beyond n decode to zero; rows >= zrow are never stored (the   */
/* pre-depth drivers never read them).                                 */
/* ------------------------------------------------------------------ */

/* truncated 3-level forward on the 8 row registers: node twiddles
 * w[0] | w[0], w[2] | w[0], w[2], w[4], w[6] (prefix-stable table) */
static inline void p30_cfwd8x(__m512i x[8], size_t z, const p30_prime *pp,
                              __m512i vp, __m512i vp2){
    p30_vcc w0 = p30_vcc_of(pp->w[0]);
    if(z > 4){
        P30_BFR(0,4,w0); P30_BFR(1,5,w0); P30_BFR(2,6,w0); P30_BFR(3,7,w0);
    }else{
        P30_BFC(0,4,w0); P30_BFC(1,5,w0); P30_BFC(2,6,w0); P30_BFC(3,7,w0);
    }
    {   size_t zl = z < 4 ? z : 4;
        if(zl > 2){ P30_BFR(0,2,w0); P30_BFR(1,3,w0); }
        else      { P30_BFC(0,2,w0); P30_BFC(1,3,w0); }
    }
    p30_vcc w2 = p30_vcc_of(pp->w[2]);
    if(z > 4){
        if(z - 4 > 2){ P30_BFR(4,6,w2); P30_BFR(5,7,w2); }
        else         { P30_BFC(4,6,w2); P30_BFC(5,7,w2); }
    }
    if(z >= 2)      P30_BFR(0,1,w0);
    else            P30_BFC(0,1,w0);
    if(z >= 4)      P30_BFR(2,3,w2);
    else if(z == 3) P30_BFC(2,3,w2);
    if(z >= 5){
        p30_vcc w4 = p30_vcc_of(pp->w[4]);
        if(z >= 6) P30_BFR(4,5,w4); else P30_BFC(4,5,w4);
    }
    if(z >= 7){
        p30_vcc w6 = p30_vcc_of(pp->w[6]);
        if(z >= 8) P30_BFR(6,7,w6); else P30_BFC(6,7,w6);
    }
}

static void p30_input_f3(uint32_t *out[P30_NP], const uint64_t *a, size_t n,
                         size_t M, size_t zrow, const p30_pset *pl, int nt){
    const __m512i PK = p30_pack_idx();
    size_t C = M >> 3;
    size_t nf = n / P30_W;                   /* full input vectors */
    size_t rem = n % P30_W;
    __mmask8 m0 = (__mmask8)((rem >= 8) ? 0xFF : ((1u << rem) - 1u));
    __mmask8 m1 = (__mmask8)((rem >= 8) ? ((1u << (rem - 8)) - 1u) : 0);
    __m512i tb[8][16];                       /* 8KB L1 staging tile */
    for(size_t ct = 0; ct < C; ct += 16){
        size_t tw = (ct + 16 < C ? 16 : C - ct);
        for(int q = 0; q < P30_NP; ++q){
            const p30_prime *pp = &pl->pr[q];
            __m512i R1 = _mm512_set1_epi64(pp->R1);
            __m512i R2 = _mm512_set1_epi64(pp->R2);
            __m512i J  = _mm512_set1_epi64(pp->J);
            __m512i p64 = _mm512_set1_epi64(pp->p);
            __m512i vp  = _mm512_set1_epi32((int)pp->p);
            __m512i vp2 = _mm512_add_epi32(vp, vp);
            /* decode row-bursts into the tile (2KB limb reads per row,
             * the 16KB limb tile stays cache-hot across the 5 primes) */
            for(size_t r = 0; r < 8; ++r){
                size_t v0 = r * C + ct;
                for(size_t t = 0; t < tw; ++t){
                    size_t v = v0 + t;
                    if(v > nf || (v == nf && !rem)){
                        tb[r][t] = _mm512_setzero_si512();
                        continue;
                    }
                    __m512i x0, x1;
                    if(v < nf){
                        x0 = _mm512_loadu_si512(a + P30_W * v);
                        x1 = _mm512_loadu_si512(a + P30_W * v + 8);
                    }else{                   /* partial last vector */
                        x0 = _mm512_maskz_loadu_epi64(m0, a + P30_W * v);
                        x1 = _mm512_maskz_loadu_epi64(m1, a + P30_W * v + 8);
                    }
                    __m512i r0 = p30_in_redc8(x0, R1, R2, J, p64);
                    __m512i r1 = p30_in_redc8(x1, R1, R2, J, p64);
                    __m512i rr = _mm512_permutex2var_epi32(r0, PK, r1);
                    rr = p30_shrink(rr, vp2);
                    tb[r][t] = p30_shrink(rr, vp);
                }
            }
            /* 3-level column network, in-tile (L1) */
            for(size_t t = 0; t < tw; ++t){
                __m512i x[8];
                for(size_t r = 0; r < 8; ++r) x[r] = tb[r][t];
                p30_cfwd8x(x, zrow, pp, vp, vp2);
                for(size_t r = 0; r < zrow; ++r) tb[r][t] = x[r];
            }
            /* row-burst stores (~1KB; NT skips the RFO when gated on) */
            for(size_t r = 0; r < zrow; ++r){
                uint32_t *dst = out[q] + P30_W * (r * C + ct);
                if(nt)
                    for(size_t t = 0; t < tw; ++t)
                        _mm512_stream_si512((void *)(dst + P30_W * t),
                                            tb[r][t]);
                else
                    for(size_t t = 0; t < tw; ++t)
                        _mm512_store_si512(dst + P30_W * t, tb[r][t]);
            }
        }
    }
    if(nt) _mm_sfence();
}


/* ------------------------------------------------------------------ */
/* output: 5 residue arrays ([0,2p) accepted) -> carried limbs.        */
/* Batched step-major Garner (bench/p30_io_bench.c winner: each digit  */
/* step sweeps a 256-coefficient L1 buffer with only its own constants */
/* live) + 3-stream compose + two SWAR carry chains. Writes zlen + 2.  */
/* ------------------------------------------------------------------ */
typedef struct { __m512i prev; unsigned cin; } p30_chain;
static inline __m512i p30_chain_step(p30_chain *c, __m512i lo, __m512i addend){
    __m512i his = _mm512_alignr_epi64(addend, c->prev, 7);
    __m512i sum = _mm512_add_epi64(lo, his);
    unsigned g  = _mm512_cmplt_epu64_mask(sum, his);
    unsigned pr = _mm512_cmpeq_epu64_mask(sum, _mm512_set1_epi64(-1));
    unsigned cn = pr + ((g << 1) | c->cin);
    unsigned cy = cn ^ pr;
    sum = _mm512_mask_add_epi64(sum, (__mmask8)cy, sum, _mm512_set1_epi64(1));
    c->cin = (cn >> 8) & 1u;
    c->prev = addend;
    return sum;
}
static inline void p30_compose8(const p30_pset *pl, __m512i d[P30_NP],
                                __m512i *L0, __m512i *L1, __m512i *L2){
    __m512i col0 = _mm512_add_epi64(d[0],
        _mm512_mul_epu32(d[1], _mm512_set1_epi64(pl->w1c[0])));
    col0 = _mm512_add_epi64(col0,
        _mm512_mul_epu32(d[2], _mm512_set1_epi64(pl->w2c[0])));
    col0 = _mm512_add_epi64(col0,
        _mm512_mul_epu32(d[3], _mm512_set1_epi64(pl->w3c[0])));
    col0 = _mm512_add_epi64(col0,
        _mm512_mul_epu32(d[4], _mm512_set1_epi64(pl->w4c[0])));
    __m512i col1 = _mm512_add_epi64(
        _mm512_mul_epu32(d[2], _mm512_set1_epi64(pl->w2c[1])),
        _mm512_mul_epu32(d[3], _mm512_set1_epi64(pl->w3c[1])));
    col1 = _mm512_add_epi64(col1,
        _mm512_mul_epu32(d[4], _mm512_set1_epi64(pl->w4c[1])));
    __m512i col2 = _mm512_add_epi64(
        _mm512_mul_epu32(d[3], _mm512_set1_epi64(pl->w3c[2])),
        _mm512_mul_epu32(d[4], _mm512_set1_epi64(pl->w4c[2])));
    __m512i col3 = _mm512_mul_epu32(d[4], _mm512_set1_epi64(pl->w4c[3]));

    __m512i l0 = _mm512_add_epi64(col0, _mm512_slli_epi64(col1, 32));
    __mmask8 k0 = _mm512_cmplt_epu64_mask(l0, col0);
    __m512i t = _mm512_add_epi64(_mm512_srli_epi64(col1, 32), col2);
    t = _mm512_mask_add_epi64(t, k0, t, _mm512_set1_epi64(1));
    __m512i l1 = _mm512_add_epi64(t, _mm512_slli_epi64(col3, 32));
    __mmask8 k1 = _mm512_cmplt_epu64_mask(l1, t);
    __m512i l2 = _mm512_srli_epi64(col3, 32);
    l2 = _mm512_mask_add_epi64(l2, k1, l2, _mm512_set1_epi64(1));
    *L0 = l0; *L1 = l1; *L2 = l2;
}
static inline __m512i p30_obar(__m512i a, uint32_t c, uint32_t rec, uint32_t p){
    return p30_bar2p(a, _mm512_set1_epi32((int)c),
                     _mm512_set1_epi32((int)rec), _mm512_set1_epi32((int)p));
}
#define P30_OB 256
static void p30_output(uint64_t *out, uint32_t *const *r,
                       size_t zlen, const p30_pset *pl){
    const __m512i ILO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i IHI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    const __m512i M32 = _mm512_set1_epi64(0xFFFFFFFFu);
    _Alignas(64) uint32_t vb[P30_NP][P30_OB];

    p30_chain c1 = { _mm512_setzero_si512(), 0 };
    p30_chain c2 = { _mm512_setzero_si512(), 0 };
    __m512i l2prev = _mm512_setzero_si512();

    size_t j = 0;
    for(; j + P30_OB <= zlen; j += P30_OB){
        for(int t = 0; t < P30_OB; t += 16){
            __m512i x = _mm512_loadu_si512(r[0] + j + t);
            _mm512_store_si512(vb[0] + t,
                p30_shrink(x, _mm512_set1_epi32((int)pl->pr[0].p)));
        }
        for(int i = 1; i < P30_NP; ++i){
            uint32_t pi = pl->pr[i].p, p2i = pl->pr[i].p2;
            __m512i vpi = _mm512_set1_epi32((int)pi);
            __m512i vp2 = _mm512_set1_epi32((int)p2i);
            for(int t = 0; t < P30_OB; t += 16){
                __m512i acc = _mm512_load_si512(vb[0] + t);
                for(int k = 1; k < i; ++k){
                    __m512i u = p30_obar(_mm512_load_si512(vb[k] + t),
                                         pl->ppm[i][k], pl->ppm_rec[i][k], pi);
                    acc = p30_shrink(_mm512_add_epi32(acc, u), vp2);
                }
                __m512i d = _mm512_sub_epi32(
                    _mm512_add_epi32(_mm512_loadu_si512(r[i] + j + t), vp2), acc);
                __m512i v = p30_obar(d, pl->inv[i], pl->inv_rec[i], pi);
                _mm512_store_si512(vb[i] + t, p30_shrink(v, vpi));
            }
        }
        for(int t = 0; t < P30_OB; t += 16){
            __m512i de[P30_NP], dd[P30_NP];
            for(int q = 0; q < P30_NP; ++q){
                __m512i v = _mm512_load_si512(vb[q] + t);
                de[q] = _mm512_and_si512(v, M32);
                dd[q] = _mm512_srli_epi64(v, 32);
            }
            __m512i L0e, L1e, L2e, L0o, L1o, L2o;
            p30_compose8(pl, de, &L0e, &L1e, &L2e);
            p30_compose8(pl, dd, &L0o, &L1o, &L2o);
            __m512i L0a = _mm512_permutex2var_epi64(L0e, ILO, L0o);
            __m512i L0b = _mm512_permutex2var_epi64(L0e, IHI, L0o);
            __m512i L1a = _mm512_permutex2var_epi64(L1e, ILO, L1o);
            __m512i L1b = _mm512_permutex2var_epi64(L1e, IHI, L1o);
            __m512i L2a = _mm512_permutex2var_epi64(L2e, ILO, L2o);
            __m512i L2b = _mm512_permutex2var_epi64(L2e, IHI, L2o);
            __m512i L2da = _mm512_alignr_epi64(L2a, l2prev, 7);
            __m512i L2db = _mm512_alignr_epi64(L2b, L2a, 7);
            l2prev = L2b;
            __m512i s;
            s = p30_chain_step(&c1, L0a, L1a);
            s = p30_chain_step(&c2, s, L2da);
            _mm512_storeu_si512(out + j + t, s);
            s = p30_chain_step(&c1, L0b, L1b);
            s = p30_chain_step(&c2, s, L2db);
            _mm512_storeu_si512(out + j + t + 8, s);
        }
    }
    /* scalar tail + flush */
    uint64_t lane7[8];
    _mm512_storeu_si512(lane7, c1.prev);
    uint64_t pend_L1 = lane7[7];
    _mm512_storeu_si512(lane7, c2.prev);
    uint64_t pend_L2_2 = lane7[7];
    _mm512_storeu_si512(lane7, l2prev);
    uint64_t pend_L2_1 = lane7[7];
    p30_u128 carry = (p30_u128)c1.cin + c2.cin;
    for(; j < zlen; ++j){
        uint64_t v[P30_NP];
        v[0] = r[0][j] % pl->prs[0];
        for(int i = 1; i < P30_NP; ++i){
            uint64_t pi = pl->prs[i], t = v[0] % pi;
            for(int k = 1; k < i; ++k)
                t = (t + v[k] * (uint64_t)pl->ppm[i][k]) % pi;
            v[i] = (r[i][j] % pi + pi - t) % pi * pl->inv[i] % pi;
        }
        p30_u128 x = v[4];
        x = x * pl->prs[3] + v[3];
        x = x * pl->prs[2] + v[2];
        x = x * pl->prs[1] + v[1];
        p30_u128 m0 = (p30_u128)(uint64_t)x * pl->prs[0] + v[0];
        p30_u128 m1 = (p30_u128)(uint64_t)(x >> 64) * pl->prs[0] + (uint64_t)(m0 >> 64);
        p30_u128 s = (p30_u128)(uint64_t)m0 + pend_L1 + pend_L2_2 + carry;
        out[j] = (uint64_t)s;
        carry = s >> 64;
        pend_L1 = (uint64_t)m1;
        pend_L2_2 = pend_L2_1;
        pend_L2_1 = (uint64_t)(m1 >> 64);
    }
    p30_u128 s = (p30_u128)pend_L1 + pend_L2_2 + carry;
    out[zlen] = (uint64_t)s;
    out[zlen + 1] = (uint64_t)((s >> 64) + pend_L2_1);
}

/* ------------------------------------------------------------------ */
/* public entry                                                        */
/* ------------------------------------------------------------------ */
static SCRATCH_TLS p30_plan p30_tls_plan;
static int p30_ladder = 1;        /* runtime ladder switch (A/B bench) */

static inline int p30_mul_r(uint64_t *rp,
                            const uint64_t *ap, ptrdiff_t an,
                            const uint64_t *bp, ptrdiff_t bn,
                            scratch *sc){
    if(an <= 0 || bn <= 0) return 0;
    uint64_t mn = (uint64_t)(an < bn ? an : bn);
    uint64_t zlen = (uint64_t)an + (uint64_t)bn - 1;
    if(mn > P30_MIN_OPERAND_CAP) return 0;        /* CRT bound */
    uint64_t tv = (zlen + P30_W - 1) / P30_W;     /* trunc in vectors */
    int lg = 0;
    while(((uint64_t)1 << lg) < tv) lg++;
    if(lg > P30_MAX_LG) return 0;                 /* transform bound */
    size_t M = (size_t)1 << lg;

    /* ladder dispatch: tv in (2*2^k, 3*2^k] -> mixed 3*2^k transform
     * (radix-3 always full; needs the 3*2^20-capable prime set).
     * GATED to the L3-crossing octave where the in-process A/B bench
     * showed +8-11% (binary M = 2^16 spills L3, mixed 3*2^14 fits);
     * elsewhere the twist overhead / missing input fusion loses
     * (bench /tmp/p30_ab.c numbers in docs §16). Tests override the
     * gates to cover every rung. p30_ladder: runtime A/B switch. */
    int mixed = (p30_ladder && lg >= P30_MIX_MIN_LG && lg <= P30_MIX_MAX_LG
                 && lg - 2 <= 16 && tv <= ((uint64_t)3 << (lg - 2)));
    size_t L = mixed ? ((size_t)1 << (lg - 2)) : 0;
    size_t NV = mixed ? 3 * L : M;

    p30_plan *pl = &p30_tls_plan;
    p30_plan_init(pl);
    const p30_pset *ps = mixed ? &pl->m : &pl->b;
    p30_pset_ensure(mixed ? &pl->m : &pl->b, mixed ? lg - 2 : lg);
    if(mixed) p30_mix_rung(&pl->m, lg - 2);

    SCRATCH(sc);
    /* 5 + 5 residue buffers (NV*16 u32 each, 64B-aligned, staggered) +
     * result staging (layout normalization + the zlen+2 spill guard).
     * The spine-tail side channel lives in the a-spectrum's dead tail
     * (global positions [tv, NV)), so it needs no allocation. */
    size_t bv = NV * P30_W + 64;                  /* u32 per buffer */
    uint32_t *blk = SALLOC(sc, uint32_t, 10 * bv);
    uint32_t *ra[P30_NP], *rb[P30_NP];
    for(int q = 0; q < P30_NP; ++q){
        ra[q] = blk + (size_t)(2 * q) * bv;
        rb[q] = blk + (size_t)(2 * q + 1) * bv;
    }
    uint64_t *st = SALLOC(sc, uint64_t, zlen + 2 + 8);

    /* fused input column pass: decode + first 3 levels in one pass
     * (size-gated, binary sizes only for now; the traversals then
     * enter 3 levels down) */
    int fin = !mixed && (M >= P30_FIN_MIN_M) && M >= 8;
    int nt  = fin && (M >= P30_FIN_NT_MIN_M);
    if(fin){
        size_t zrow = (tv + (M >> 3) - 1) / (M >> 3);
        p30_input_f3(ra, ap, (size_t)an, M, zrow, ps, nt);
        p30_input_f3(rb, bp, (size_t)bn, M, zrow, ps, nt);
    }else{
        p30_input(ra, ap, (size_t)an, NV, ps);
        p30_input(rb, bp, (size_t)bn, NV, ps);
    }

    for(int q = 0; q < P30_NP; ++q){
        const p30_prime *pp = &ps->pr[q];
        __m512i vp = _mm512_set1_epi32((int)pp->p);
        /* (no standalone a-canonicalize pass: the convs shrink their
         * fb blocks in registers -- same ops, no extra traffic) */
        if(mixed){
            p30_mfwd(ra[q], L, (size_t)tv - 2 * L, lg - 2, pp, vp);
            p30_mfused(rb[q], ra[q], L, (size_t)tv - 2 * L, lg - 2, pp,
                       vp, 0);
        }else if(fin){
            p30_fwd_pre(ra[q], M, 0, tv, 3, pp, vp);
            p30_fused_pre(rb[q], ra[q], M, 0, tv, 3, pp, vp, 0);
        }else{
            p30_fwd_rec(ra[q], M, 0, tv, pp, vp);
            p30_fused_rec(rb[q], ra[q], M, 0, tv, pp, vp, 0);
        }
        /* scale: s = NV^-1 * R (cancels the conv REDC stray R^-1) */
        uint32_t sc32 = p30_mulm(p30_invm((uint32_t)(NV % pp->p), pp->p),
                                 pp->R1, pp->p);
        p30_vcc vs = p30_vcc_of(p30_cc_of(sc32, pp->p));
        for(size_t k = 0; k < tv; ++k){
            uint32_t *v = rb[q] + P30_W * k;
            _mm512_store_si512(v,
                p30_barm(_mm512_load_si512(v), vs.c, vs.rec, vp));
        }
    }

    p30_output(st, rb, zlen, ps);
    if(st[zlen + 1] != 0) return 0;               /* must not spill */
    memcpy(rp, st, ((size_t)an + (size_t)bn) * 8);
    return 1;
}

static inline int p30_mul(uint64_t *rp,
                          const uint64_t *ap, ptrdiff_t an,
                          const uint64_t *bp, ptrdiff_t bn){
    return p30_mul_r(rp, ap, an, bp, bn, scratch_thread());
}

/* squaring: one input pass, no standalone forward -- the fused
 * traversal convolves each leaf with itself (sq = 1). The a-spectrum
 * array degenerates to side-buffer scratch (spine tails + ladder
 * seam); one buffer serves all primes. */
static inline int p30_sqr_r(uint64_t *rp, const uint64_t *ap, ptrdiff_t an,
                            scratch *sc){
    if(an <= 0) return 0;
    uint64_t zlen = 2 * (uint64_t)an - 1;
    if((uint64_t)an > P30_MIN_OPERAND_CAP) return 0;
    uint64_t tv = (zlen + P30_W - 1) / P30_W;
    int lg = 0;
    while(((uint64_t)1 << lg) < tv) lg++;
    if(lg > P30_MAX_LG) return 0;
    size_t M = (size_t)1 << lg;

    int mixed = (p30_ladder && lg >= P30_MIX_MIN_LG && lg <= P30_MIX_MAX_LG
                 && lg - 2 <= 16 && tv <= ((uint64_t)3 << (lg - 2)));
    size_t L = mixed ? ((size_t)1 << (lg - 2)) : 0;
    size_t NV = mixed ? 3 * L : M;

    p30_plan *pl = &p30_tls_plan;
    p30_plan_init(pl);
    const p30_pset *ps = mixed ? &pl->m : &pl->b;
    p30_pset_ensure(mixed ? &pl->m : &pl->b, mixed ? lg - 2 : lg);
    if(mixed) p30_mix_rung(&pl->m, lg - 2);

    SCRATCH(sc);
    size_t bv = NV * P30_W + 64;                  /* u32 per buffer */
    uint32_t *blk = SALLOC(sc, uint32_t, 6 * bv);
    uint32_t *rb[P30_NP];
    for(int q = 0; q < P30_NP; ++q)
        rb[q] = blk + (size_t)q * bv;
    uint32_t *sbuf = blk + (size_t)P30_NP * bv;
    uint64_t *st = SALLOC(sc, uint64_t, zlen + 2 + 8);

    int fin = !mixed && (M >= P30_FIN_MIN_M) && M >= 8;
    int nt  = fin && (M >= P30_FIN_NT_MIN_M);
    if(fin){
        size_t zrow = (tv + (M >> 3) - 1) / (M >> 3);
        p30_input_f3(rb, ap, (size_t)an, M, zrow, ps, nt);
    }else{
        p30_input(rb, ap, (size_t)an, NV, ps);
    }

    for(int q = 0; q < P30_NP; ++q){
        const p30_prime *pp = &ps->pr[q];
        __m512i vp = _mm512_set1_epi32((int)pp->p);
        if(mixed){
            p30_mfused(rb[q], sbuf, L, (size_t)tv - 2 * L, lg - 2, pp,
                       vp, 1);
        }else if(fin){
            p30_fused_pre(rb[q], sbuf, M, 0, tv, 3, pp, vp, 1);
        }else{
            p30_fused_rec(rb[q], sbuf, M, 0, tv, pp, vp, 1);
        }
        /* scale: s = NV^-1 * R (cancels the conv REDC stray R^-1) */
        uint32_t sc32 = p30_mulm(p30_invm((uint32_t)(NV % pp->p), pp->p),
                                 pp->R1, pp->p);
        p30_vcc vs = p30_vcc_of(p30_cc_of(sc32, pp->p));
        for(size_t k = 0; k < tv; ++k){
            uint32_t *v = rb[q] + P30_W * k;
            _mm512_store_si512(v,
                p30_barm(_mm512_load_si512(v), vs.c, vs.rec, vp));
        }
    }

    p30_output(st, rb, zlen, ps);
    if(st[zlen + 1] != 0) return 0;               /* must not spill */
    memcpy(rp, st, 2 * (size_t)an * 8);
    return 1;
}

static inline int p30_sqr(uint64_t *rp, const uint64_t *ap, ptrdiff_t an){
    return p30_sqr_r(rp, ap, an, scratch_thread());
}
