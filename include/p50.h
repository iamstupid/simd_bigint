#pragma once
// p50.h - 4x50-bit-prime truncated NTT bigint multiplier (AVX-512
// IFMA52). u64-limb I/O, TRUNK chunking with a size ladder
// T in {88, 84, 80} (largest fitting trunk wins; caps 2^24 / 2^32 /
// 2^40 limbs on min(an,bn)). Design: docs/p50_ntt_design.md; kernels
// pinned by bench/p50_arith_bench.c, p50_kernel_bench.c,
// p50_io_bench.c. Structure is the p30 skeleton (radix-agnostic):
// vdH truncated y-TFT on W = 8 lane vectors (64-bit lanes), FLINT
// twiddle convention with prefix-stable BR tables as Barrett52
// (c, rec) u64 pairs, iterative DFS full subtrees with in-register
// m <= 32 bottoms, fused fwd(b)+pw+inv traversal with the spine-tail
// side channel over the a-spectrum's dead tail.
//
// Deltas from p30: arithmetic = Barrett52 (3 madd52 + and; the p30
// lemma at R = 2^52, valid for ANY a < 2^52) and wide REDC52;
// pointwise = conv8 SCHOOLBOOK (kara LOSES under IFMA: fused
// accumulate makes products cheap, bench 1.45x); lazy [0,4p) < 2^52
// (p < 2^50 by prime choice); conv eats LAZY fb directly (32p^2 <
// 2^104 column headroom) -- no canonicalize anywhere. I/O =
// table-driven trunk extraction (vpshrdvq funnels, per-piece <= 52
// bits so 2 limbs always suffice) and the trunk-radix Garner
// (radix-2^52 IFMA compose -> regroup -> radix-2^T SWAR carries ->
// 4-or-more-trunks -> limbs funnel regroup).
//
// v1 scope: no input-stage level fusion, no 3*2^k ladder, no
// squaring path (all ported later if the sweep says p50 displaces
// p30 in-band as projected).
//
// Entry: p50_mul_r(rp, ap, an, bp, bn, scratch*) -> 1 ok / 0 out of
// band; p50_mul(...) uses the thread scratch.
#include "scratch.h"
#include <immintrin.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned __int128 p50_u128;

#define P50_NP 4
#define P50_W 8
#ifndef P50_MAX_LG
#define P50_MAX_LG 22            /* M <= 2^22 vectors = 2^25 trunks;
                                  * root tables grow lazily (128B per
                                  * vector across primes), so idle
                                  * headroom costs nothing */
#endif
#define P50_BASE_LG 12           /* eager table allocation */
#define P50_M52 ((1ull << 52) - 1)

/* ascending order is load-bearing for Garner */
static const uint64_t P50_PR[P50_NP] = {
    1025844348715009ull,         /* 933*2^40 + 1 */
    1072023837081601ull,         /* 975*2^40 + 1 */
    1086317488242689ull,         /* 247*2^42 + 1 */
    1108307720798209ull          /*  63*2^44 + 1 */
};

/* ------------------------------------------------------------------ */
/* scalar helpers (setup only)                                         */
/* ------------------------------------------------------------------ */
static inline uint64_t p50_mulm(uint64_t a, uint64_t b, uint64_t p){
    return (uint64_t)((p50_u128)a * b % p);
}
static inline uint64_t p50_powm(uint64_t a, uint64_t e, uint64_t p){
    uint64_t r = 1;
    for(; e; e >>= 1, a = p50_mulm(a, a, p))
        if(e & 1) r = p50_mulm(r, a, p);
    return r;
}
static inline uint64_t p50_invm(uint64_t a, uint64_t p){
    return p50_powm(a, p - 2, p);
}

/* Barrett52 pair: r = a*c - hi52(a*rec)*p in [0, 2p) for ANY a < 2^52 */
typedef struct { uint64_t c, rec; } p50_cc;
static inline p50_cc p50_cc_of(uint64_t c, uint64_t p){
    p50_cc t = { c, (uint64_t)(((p50_u128)c << 52) / p) };
    return t;
}

/* ------------------------------------------------------------------ */
/* per-T I/O plan (trunk ladder: one table-driven path for all T)      */
/* ------------------------------------------------------------------ */
typedef struct {
    int T, LW;                   /* trunk bits, lo width = T-52 */
    int BT, BL, nv;              /* extraction block: BT trunks/BL limbs */
    int vbase[2];
    _Alignas(64) uint64_t loidx[2][8], losh[2][8];
    _Alignas(64) uint64_t hiidx[2][8], hish[2][8];
    int OT, OL, onv;             /* regroup superblock */
    int obase[24];
    _Alignas(64) uint64_t ouidx[24][8], osh[24][8], olsh[24][8];
    unsigned char oswp[24];      /* lanes with s >= 64 (vpshrdvq mod 64) */
    uint64_t C[P50_NP];          /* 2^T mod p */
    uint64_t capt;               /* min-operand cap in TRUNKS */
} p50_tio;

typedef struct {
    uint64_t p, p2, J;           /* J = -p^-1 mod 2^52 */
    p50_cc   inv2;
    p50_cc  *w, *winv;           /* prefix-stable BR tables */
} p50_prime;

typedef struct {
    p50_prime pr[P50_NP];
    int       lg;
    p50_tio   tio[3];            /* T = 80, 84, 88 */
    /* Garner + compose constants */
    uint64_t  inv[P50_NP], inv_rec[P50_NP];
    uint64_t  ppm[P50_NP][P50_NP], ppm_rec[P50_NP][P50_NP];
    uint64_t  w1[1], w2[2], w3[3];   /* weight 52-bit words */
    int       init;
    int       alg;                   /* allocated table capacity (lg) */
} p50_plan;

static inline void *p50_alloc(size_t bytes){
    void *q = NULL;
    if(posix_memalign(&q, 64, bytes)) abort();
    return q;
}

static void p50_tio_build(p50_tio *tp, int T){
    tp->T = T;
    tp->LW = T - 52;
    int bt = 8;
    while((bt * T) % 64) bt += 8;
    tp->BT = bt; tp->BL = bt * T / 64; tp->nv = bt / 8;
    for(int v = 0; v < tp->nv; ++v){
        tp->vbase[v] = (T * 8 * v) >> 6;
        for(int l = 0; l < 8; ++l){
            long long bp = (long long)T * (8*v + l);
            tp->loidx[v][l] = (uint64_t)((bp >> 6) - tp->vbase[v]);
            tp->losh[v][l]  = (uint64_t)(bp & 63);
            bp += tp->LW;
            tp->hiidx[v][l] = (uint64_t)((bp >> 6) - tp->vbase[v]);
            tp->hish[v][l]  = (uint64_t)(bp & 63);
        }
    }
    int ol = tp->BL;
    while(ol % 8) ol += tp->BL;
    tp->OL = ol; tp->OT = ol * 64 / T; tp->onv = ol / 8;
    for(int ov = 0; ov < tp->onv; ++ov){
        tp->obase[ov] = (int)(((long long)64 * 8 * ov) / T);
        tp->oswp[ov] = 0;
        for(int l = 0; l < 8; ++l){
            long long L = 8*ov + l;
            long long u = (64*L) / T;
            long long s = 64*L - (long long)T*u;
            tp->ouidx[ov][l] = (uint64_t)(u - tp->obase[ov]);
            tp->osh[ov][l]   = (uint64_t)s;
            tp->olsh[ov][l]  = (uint64_t)(T - s);
            if(s >= 64) tp->oswp[ov] |= (unsigned char)(1u << l);
        }
    }
    for(int q = 0; q < P50_NP; ++q)
        tp->C[q] = (uint64_t)((((p50_u128)1) << T) % P50_PR[q]);
    /* cap: largest n with n * (2^T-1)^2 < prod(p) */
    {
        p50_u128 hi = 1;            /* prod / 2^128 via long division feel */
        /* prod ~ 2^199.72: compute cap = floor(prod / 2^(2T)) approx
         * exactly: prod = p0*p1*p2*p3: do 256-bit by repeated 128 ops */
        /* simpler: cap_t = floor( ((p0*p1/2^T) * (p2*p3/2^T)) ) lower
         * bound (safe underestimate) */
        p50_u128 a = (p50_u128)P50_PR[0] * P50_PR[1];
        p50_u128 b = (p50_u128)P50_PR[2] * P50_PR[3];
        p50_u128 cap = (a >> T) * (b >> T) >> 0;
        (void)hi;
        tp->capt = (cap > (p50_u128)~0ull) ? ~0ull : (uint64_t)cap;
    }
}

static uint64_t p50_primitive_root(uint64_t p){
    uint64_t f[16];
    int nf = 0;
    uint64_t n = p - 1;
    for(uint64_t d = 2; (p50_u128)d * d <= n; d += (d == 2 ? 1 : 2))
        if(n % d == 0){ f[nf++] = d; while(n % d == 0) n /= d; }
    if(n > 1) f[nf++] = n;
    for(uint64_t g = 2; ; ++g){
        int ok = 1;
        for(int i = 0; i < nf && ok; ++i)
            if(p50_powm(g, (p - 1) / f[i], p) == 1) ok = 0;
        if(ok) return g;
    }
}

static inline void p50_plan_init(p50_plan *pl){
    if(pl->init) return;
    for(int q = 0; q < P50_NP; ++q){
        uint64_t p = P50_PR[q];
        p50_prime *pp = &pl->pr[q];
        pp->p = p;
        pp->p2 = 2 * p;
        uint64_t pi = 1;
        for(int i = 0; i < 6; ++i) pi *= 2 - p * pi;
        pp->J = (0 - pi) & P50_M52;
        pp->inv2 = p50_cc_of((p + 1) >> 1, p);
        pp->w    = (p50_cc *)p50_alloc(sizeof(p50_cc) << P50_BASE_LG);
        pp->winv = (p50_cc *)p50_alloc(sizeof(p50_cc) << P50_BASE_LG);
        pp->w[0] = pp->winv[0] = p50_cc_of(1, p);
    }
    pl->lg = 0;
    pl->alg = P50_BASE_LG;
    p50_tio_build(&pl->tio[0], 80);
    p50_tio_build(&pl->tio[1], 84);
    p50_tio_build(&pl->tio[2], 88);
    for(int i = 1; i < P50_NP; ++i){
        uint64_t p = P50_PR[i], pp = 1;
        for(int k = 0; k < i; ++k){
            pl->ppm[i][k] = pp;
            pl->ppm_rec[i][k] = p50_cc_of(pp, p).rec;
            pp = p50_mulm(pp, P50_PR[k] % p, p);
        }
        pl->inv[i] = p50_invm(pp, p);
        pl->inv_rec[i] = p50_cc_of(pl->inv[i], p).rec;
    }
    pl->w1[0] = P50_PR[0] & P50_M52;
    {
        p50_u128 w = (p50_u128)P50_PR[0] * P50_PR[1];
        pl->w2[0] = (uint64_t)(w & P50_M52);
        pl->w2[1] = (uint64_t)(w >> 52);
        uint64_t a0 = (uint64_t)(w & P50_M52),
                 a1 = (uint64_t)((w >> 52) & P50_M52),
                 a2 = (uint64_t)(w >> 104);
        p50_u128 c0 = (p50_u128)a0 * P50_PR[2];
        p50_u128 c1 = (p50_u128)a1 * P50_PR[2] + (uint64_t)(c0 >> 52);
        p50_u128 c2 = (p50_u128)a2 * P50_PR[2] + (uint64_t)(c1 >> 52);
        pl->w3[0] = (uint64_t)(c0 & P50_M52);
        pl->w3[1] = (uint64_t)(c1 & P50_M52);
        pl->w3[2] = (uint64_t)(c2 & P50_M52);
    }
    pl->init = 1;
}

/* grow the BR root tables (prefix-stable, tower-consistent) */
static inline void p50_plan_ensure(p50_plan *pl, int lg){
    p50_plan_init(pl);
    if(lg <= pl->lg) return;
    if(lg > pl->alg){                /* grow buffers, keep the prefix */
        size_t keep = sizeof(p50_cc) << pl->lg;
        for(int q = 0; q < P50_NP; ++q){
            p50_prime *pp = &pl->pr[q];
            p50_cc *nw = (p50_cc *)p50_alloc(sizeof(p50_cc) << lg);
            p50_cc *ni = (p50_cc *)p50_alloc(sizeof(p50_cc) << lg);
            memcpy(nw, pp->w,    keep);
            memcpy(ni, pp->winv, keep);
            free(pp->w); free(pp->winv);
            pp->w = nw; pp->winv = ni;
        }
        pl->alg = lg;
    }
    for(int q = 0; q < P50_NP; ++q){
        p50_prime *pp = &pl->pr[q];
        uint64_t p = pp->p;
        uint64_t g = p50_primitive_root(p);
        for(int t = pl->lg + 1; t <= lg; ++t){
            uint64_t z  = p50_powm(g, (p - 1) >> t, p);
            uint64_t zi = p50_invm(z, p);
            uint64_t blk = (uint64_t)1 << (t - 1);
            for(uint64_t s = 0; s < blk; ++s){
                pp->w[blk + s]    = p50_cc_of(p50_mulm(z,  pp->w[s].c,    p), p);
                pp->winv[blk + s] = p50_cc_of(p50_mulm(zi, pp->winv[s].c, p), p);
            }
        }
    }
    pl->lg = lg;
}

/* ------------------------------------------------------------------ */
/* SIMD primitives. Lazy range discipline identical to p30, in 52-bit  */
/* lanes: between-pass values in [0,4p) < 2^52; Barrett eats any       */
/* a < 2^52; the unmultiplied butterfly operand folds once per level.  */
/* ------------------------------------------------------------------ */
static inline __m512i p50_b52(__m512i a, __m512i vc, __m512i vrec,
                              __m512i vpn){
    __m512i z = _mm512_setzero_si512();
    __m512i q = _mm512_madd52hi_epu64(z, a, vrec);
    __m512i s = _mm512_madd52lo_epu64(z, a, vc);
    s = _mm512_madd52lo_epu64(s, q, vpn);
    return _mm512_and_si512(s, _mm512_set1_epi64((long long)P50_M52));
}
static inline __m512i p50_sh2(__m512i x, __m512i vp2){
    return _mm512_min_epu64(x, _mm512_sub_epi64(x, vp2));
}
static inline __m512i p50_shp(__m512i x, __m512i vp){
    return _mm512_min_epu64(x, _mm512_sub_epi64(x, vp));
}
/* wide REDC52 of (lo, hi) columns (t < 2^104) -> < ~2.1p-ish lazy */
static inline __m512i p50_redc(__m512i lo, __m512i hi, __m512i vJ,
                               __m512i vP, __m512i vM){
    __m512i lom = _mm512_and_si512(lo, vM);
    __m512i m = _mm512_and_si512(
        _mm512_madd52lo_epu64(_mm512_setzero_si512(), lom, vJ), vM);
    __m512i r = _mm512_madd52hi_epu64(
        _mm512_add_epi64(hi, _mm512_srli_epi64(lo, 52)), m, vP);
    __mmask8 cy = _mm512_test_epi64_mask(lom, vM);
    return _mm512_mask_add_epi64(r, cy, r, _mm512_set1_epi64(1));
}

typedef struct { __m512i c, rec; } p50_vcc;
static inline p50_vcc p50_vcc_of(p50_cc t){
    p50_vcc v = { _mm512_set1_epi64((long long)t.c),
                  _mm512_set1_epi64((long long)t.rec) };
    return v;
}

/* in-register butterflies on an x[] frame (vpn = 2^52 - p hoisted) */
#define P50_BFR(lo_, hi_, T) do{                                         \
        __m512i a_  = p50_sh2(x[lo_], vp2);                              \
        __m512i wb_ = p50_b52(x[hi_], (T).c, (T).rec, vpn);              \
        x[lo_] = _mm512_add_epi64(a_, wb_);                              \
        x[hi_] = _mm512_add_epi64(a_, _mm512_sub_epi64(vp2, wb_));       \
    }while(0)
#define P50_IBFR(lo_, hi_, T) do{                                        \
        __m512i a_ = p50_sh2(x[lo_], vp2);                               \
        __m512i b_ = p50_sh2(x[hi_], vp2);                               \
        x[lo_] = _mm512_add_epi64(a_, b_);                               \
        x[hi_] = p50_b52(_mm512_add_epi64(_mm512_sub_epi64(a_, b_),      \
                                          vp2),                         \
                         (T).c, (T).rec, vpn);                           \
    }while(0)

/* ------------------------------------------------------------------ */
/* radix-8 moths (3 fused levels per memory round trip)                */
/* ------------------------------------------------------------------ */
static void p50_fwd8(uint64_t *B, size_t m, size_t j,
                     const p50_prime *pp, __m512i vpn, __m512i vp2){
    size_t e = m >> 3;
    p50_vcc t1  = p50_vcc_of(pp->w[2*j]);
    p50_vcc t2a = p50_vcc_of(pp->w[4*j]),   t2b = p50_vcc_of(pp->w[4*j+2]);
    p50_vcc t30 = p50_vcc_of(pp->w[8*j]),   t31 = p50_vcc_of(pp->w[8*j+2]);
    p50_vcc t32 = p50_vcc_of(pp->w[8*j+4]), t33 = p50_vcc_of(pp->w[8*j+6]);
    for(size_t v = 0; v < e; ++v){
        __m512i x[8];
        for(int c = 0; c < 8; ++c)
            x[c] = _mm512_load_si512(B + P50_W * (v + (size_t)c * e));
        P50_BFR(0,4,t1);  P50_BFR(1,5,t1);  P50_BFR(2,6,t1);  P50_BFR(3,7,t1);
        P50_BFR(0,2,t2a); P50_BFR(1,3,t2a); P50_BFR(4,6,t2b); P50_BFR(5,7,t2b);
        P50_BFR(0,1,t30); P50_BFR(2,3,t31); P50_BFR(4,5,t32); P50_BFR(6,7,t33);
        for(int c = 0; c < 8; ++c)
            _mm512_store_si512(B + P50_W * (v + (size_t)c * e), x[c]);
    }
}
static void p50_inv8(uint64_t *B, size_t m, size_t j,
                     const p50_prime *pp, __m512i vpn, __m512i vp2){
    size_t e = m >> 3;
    p50_vcc t1  = p50_vcc_of(pp->winv[2*j]);
    p50_vcc t2a = p50_vcc_of(pp->winv[4*j]),   t2b = p50_vcc_of(pp->winv[4*j+2]);
    p50_vcc t30 = p50_vcc_of(pp->winv[8*j]),   t31 = p50_vcc_of(pp->winv[8*j+2]);
    p50_vcc t32 = p50_vcc_of(pp->winv[8*j+4]), t33 = p50_vcc_of(pp->winv[8*j+6]);
    for(size_t v = 0; v < e; ++v){
        __m512i x[8];
        for(int c = 0; c < 8; ++c)
            x[c] = _mm512_load_si512(B + P50_W * (v + (size_t)c * e));
        P50_IBFR(0,1,t30); P50_IBFR(2,3,t31); P50_IBFR(4,5,t32); P50_IBFR(6,7,t33);
        P50_IBFR(0,2,t2a); P50_IBFR(1,3,t2a); P50_IBFR(4,6,t2b); P50_IBFR(5,7,t2b);
        P50_IBFR(0,4,t1);  P50_IBFR(1,5,t1);  P50_IBFR(2,6,t1);  P50_IBFR(3,7,t1);
        for(int c = 0; c < 8; ++c)
            _mm512_store_si512(B + P50_W * (v + (size_t)c * e), x[c]);
    }
}

/* ------------------------------------------------------------------ */
/* in-register m = 16 / 32 bottoms (p30 leaf-bench winners, ported)    */
/* ------------------------------------------------------------------ */
static inline void p50_fwd16x(__m512i x[16], size_t j, const p50_prime *pp,
                              __m512i vpn, __m512i vp2){
    {   p50_vcc t = p50_vcc_of(pp->w[2*j]);
        P50_BFR(0,8,t); P50_BFR(1,9,t); P50_BFR(2,10,t); P50_BFR(3,11,t);
        P50_BFR(4,12,t); P50_BFR(5,13,t); P50_BFR(6,14,t); P50_BFR(7,15,t); }
    {   p50_vcc a = p50_vcc_of(pp->w[4*j]), b = p50_vcc_of(pp->w[4*j+2]);
        P50_BFR(0,4,a); P50_BFR(1,5,a); P50_BFR(2,6,a); P50_BFR(3,7,a);
        P50_BFR(8,12,b); P50_BFR(9,13,b); P50_BFR(10,14,b); P50_BFR(11,15,b); }
    for(int c = 0; c < 4; ++c){
        p50_vcc t = p50_vcc_of(pp->w[8*j + 2*(size_t)c]);
        P50_BFR(4*c, 4*c+2, t); P50_BFR(4*c+1, 4*c+3, t);
    }
    for(int c = 0; c < 8; ++c){
        p50_vcc t = p50_vcc_of(pp->w[16*j + 2*(size_t)c]);
        P50_BFR(2*c, 2*c+1, t);
    }
}
static inline void p50_inv16x(__m512i x[16], size_t j, const p50_prime *pp,
                              __m512i vpn, __m512i vp2){
    for(int c = 0; c < 8; ++c){
        p50_vcc t = p50_vcc_of(pp->winv[16*j + 2*(size_t)c]);
        P50_IBFR(2*c, 2*c+1, t);
    }
    for(int c = 0; c < 4; ++c){
        p50_vcc t = p50_vcc_of(pp->winv[8*j + 2*(size_t)c]);
        P50_IBFR(4*c, 4*c+2, t); P50_IBFR(4*c+1, 4*c+3, t);
    }
    {   p50_vcc a = p50_vcc_of(pp->winv[4*j]), b = p50_vcc_of(pp->winv[4*j+2]);
        P50_IBFR(0,4,a); P50_IBFR(1,5,a); P50_IBFR(2,6,a); P50_IBFR(3,7,a);
        P50_IBFR(8,12,b); P50_IBFR(9,13,b); P50_IBFR(10,14,b); P50_IBFR(11,15,b); }
    {   p50_vcc t = p50_vcc_of(pp->winv[2*j]);
        P50_IBFR(0,8,t); P50_IBFR(1,9,t); P50_IBFR(2,10,t); P50_IBFR(3,11,t);
        P50_IBFR(4,12,t); P50_IBFR(5,13,t); P50_IBFR(6,14,t); P50_IBFR(7,15,t); }
}
static void p50_fwd16r(uint64_t *B, size_t j, const p50_prime *pp,
                       __m512i vpn, __m512i vp2){
    __m512i x[16];
    for(int c = 0; c < 16; ++c) x[c] = _mm512_load_si512(B + P50_W * c);
    p50_fwd16x(x, j, pp, vpn, vp2);
    for(int c = 0; c < 16; ++c) _mm512_store_si512(B + P50_W * c, x[c]);
}
static void p50_inv16r(uint64_t *B, size_t j, const p50_prime *pp,
                       __m512i vpn, __m512i vp2){
    __m512i x[16];
    for(int c = 0; c < 16; ++c) x[c] = _mm512_load_si512(B + P50_W * c);
    p50_inv16x(x, j, pp, vpn, vp2);
    for(int c = 0; c < 16; ++c) _mm512_store_si512(B + P50_W * c, x[c]);
}
static void p50_fwd32r(uint64_t *B, size_t j, const p50_prime *pp,
                       __m512i vpn, __m512i vp2){
    p50_vcc w = p50_vcc_of(pp->w[2*j]);
    for(size_t v = 0; v < 16; ++v){
        __m512i a  = p50_sh2(_mm512_load_si512(B + P50_W * v), vp2);
        __m512i wb = p50_b52(_mm512_load_si512(B + P50_W * (v + 16)),
                             w.c, w.rec, vpn);
        _mm512_store_si512(B + P50_W * v, _mm512_add_epi64(a, wb));
        _mm512_store_si512(B + P50_W * (v + 16),
            _mm512_add_epi64(a, _mm512_sub_epi64(vp2, wb)));
    }
    p50_fwd16r(B,             2*j,     pp, vpn, vp2);
    p50_fwd16r(B + P50_W*16,  2*j + 1, pp, vpn, vp2);
}
static void p50_inv32r(uint64_t *B, size_t j, const p50_prime *pp,
                       __m512i vpn, __m512i vp2){
    p50_inv16r(B,            2*j,     pp, vpn, vp2);
    p50_inv16r(B + P50_W*16, 2*j + 1, pp, vpn, vp2);
    p50_vcc wi = p50_vcc_of(pp->winv[2*j]);
    for(size_t v = 0; v < 16; ++v){
        __m512i a = p50_sh2(_mm512_load_si512(B + P50_W * v), vp2);
        __m512i b = p50_sh2(_mm512_load_si512(B + P50_W * (v + 16)), vp2);
        _mm512_store_si512(B + P50_W * v, _mm512_add_epi64(a, b));
        _mm512_store_si512(B + P50_W * (v + 16),
            p50_b52(_mm512_add_epi64(_mm512_sub_epi64(a, b), vp2),
                    wi.c, wi.rec, vpn));
    }
}

/* ------------------------------------------------------------------ */
/* pointwise: conv8 mod (x^8 - w[k]) SCHOOLBOOK (bench winner): one    */
/* wide REDC52, uniform R^-1 stray (cancelled by the final scale).     */
/* fa lazy [0,4p), shrunk canonical in-register for the windows. fb    */
/* (the a-spectrum) is loaded as a VECTOR, folded once to [0,2p) and   */
/* broadcast from register: the column bound is then 8 * p * 2p =      */
/* 16p^2 < 2^104. (fb at [0,4p) gives 32p^2 = 2^104.94 -- a value-     */
/* dependent accumulator overflow the random kernel-bench validation   */
/* MISSED and the GMP battery caught.)                                 */
/* ------------------------------------------------------------------ */
static inline void p50_conv8(uint64_t *fa, const uint64_t *fb, p50_cc ww,
                             const p50_prime *pp, __m512i vpn, __m512i vp2){
    __m512i vp = _mm512_set1_epi64((long long)pp->p);
    __m512i vJ = _mm512_set1_epi64((long long)pp->J);
    __m512i vM = _mm512_set1_epi64((long long)P50_M52);
    __m512i a = _mm512_load_si512(fa);
    a = p50_shp(p50_sh2(a, vp2), vp);
    __m512i aw = p50_shp(p50_b52(a, _mm512_set1_epi64((long long)ww.c),
                                 _mm512_set1_epi64((long long)ww.rec), vpn),
                         vp);
    __m512i fbv = p50_sh2(_mm512_load_si512(fb), vp2);   /* [0,2p) */
    __m512i lo = _mm512_setzero_si512(), hi = _mm512_setzero_si512();
#define P50_CSTEP(i) do{                                                 \
        __m512i b_ = _mm512_permutexvar_epi64(                           \
            _mm512_set1_epi64(i), fbv);                                  \
        __m512i w_ = (i) ? _mm512_alignr_epi64(a, aw, 8 - (i)) : a;      \
        lo = _mm512_madd52lo_epu64(lo, w_, b_);                          \
        hi = _mm512_madd52hi_epu64(hi, w_, b_);                          \
    }while(0)
    P50_CSTEP(0); P50_CSTEP(1); P50_CSTEP(2); P50_CSTEP(3);
    P50_CSTEP(4); P50_CSTEP(5); P50_CSTEP(6); P50_CSTEP(7);
#undef P50_CSTEP
    /* REDC out < hi_col + p + 1 <= 16p^2/2^52 + p ~ 4.86p, which can
     * EXCEED 2^52 at adversarial magnitudes: fold once so every
     * downstream madd52 input stays < 2^52 (4.86p -> < 2.86p; the
     * next sh2 then gives < 2p, butterfly args < 4p < 2^52). */
    __m512i r = p50_redc(lo, hi, vJ, vp, vM);
    _mm512_store_si512(fa, p50_sh2(r, vp2));
}
/* ------------------------------------------------------------------ */
/* fused leaves (split form, the p30 verdict): fwd levels + conv8 per  */
/* position + inverse levels, conv from L1                             */
/* ------------------------------------------------------------------ */
static inline void p50_fused8l(uint64_t *B, const uint64_t *A, size_t j,
                               const p50_prime *pp, __m512i vpn, __m512i vp2){
    p50_fwd8(B, 8, j, pp, vpn, vp2);
    for(size_t t = 0; t < 8; ++t)
        p50_conv8(B + P50_W * t, A + P50_W * t, pp->w[8*j + t], pp, vpn, vp2);
    p50_inv8(B, 8, j, pp, vpn, vp2);
}
static void p50_fused16l(uint64_t *B, const uint64_t *A, size_t j,
                         const p50_prime *pp, __m512i vpn, __m512i vp2){
    p50_fwd16r(B, j, pp, vpn, vp2);
    for(size_t t = 0; t < 16; ++t)
        p50_conv8(B + P50_W * t, A + P50_W * t, pp->w[16*j + t], pp, vpn, vp2);
    p50_inv16r(B, j, pp, vpn, vp2);
}
static void p50_fused32l(uint64_t *B, const uint64_t *A, size_t j,
                         const p50_prime *pp, __m512i vpn, __m512i vp2){
    p50_fwd32r(B, j, pp, vpn, vp2);
    for(size_t t = 0; t < 32; ++t)
        p50_conv8(B + P50_W * t, A + P50_W * t, pp->w[32*j + t], pp, vpn, vp2);
    p50_inv32r(B, j, pp, vpn, vp2);
}

/* ------------------------------------------------------------------ */
/* iterative full-subtree engines (DFS flattened to a leaf-block walk; */
/* p30 structure, leaf size by lg m mod 3)                             */
/* ------------------------------------------------------------------ */
static inline size_t p50_leaf_ls(size_t m){
    int lgm = 0;
    while(((size_t)1 << lgm) < m) ++lgm;
    int r = lgm % 3;
    return r == 0 ? 8 : r == 1 ? 16 : 32;
}

static void p50_fwd_full(uint64_t *B, size_t m, size_t j,
                         const p50_prime *pp, __m512i vpn, __m512i vp2){
    if(m <= 4){
        if(m == 1) return;
        __m512i x[4];
        for(size_t c = 0; c < m; ++c) x[c] = _mm512_load_si512(B + P50_W * c);
        if(m == 2){
            p50_vcc t = p50_vcc_of(pp->w[2*j]); P50_BFR(0,1,t);
        }else{
            {   p50_vcc t = p50_vcc_of(pp->w[2*j]); P50_BFR(0,2,t); P50_BFR(1,3,t); }
            {   p50_vcc a = p50_vcc_of(pp->w[4*j]), b = p50_vcc_of(pp->w[4*j+2]);
                P50_BFR(0,1,a); P50_BFR(2,3,b); }
        }
        for(size_t c = 0; c < m; ++c) _mm512_store_si512(B + P50_W * c, x[c]);
        return;
    }
    if(m == 8){  p50_fwd8(B, 8, j, pp, vpn, vp2);  return; }
    if(m == 16){ p50_fwd16r(B, j, pp, vpn, vp2);   return; }
    if(m == 32){ p50_fwd32r(B, j, pp, vpn, vp2);   return; }
    size_t LS = p50_leaf_ls(m);
    size_t nb = m / LS;
    for(size_t b = 0; b < nb; ++b){
        size_t o = b * LS;
        for(size_t s = m; s > LS; s >>= 3){
            if(b & (s / LS - 1)) continue;
            p50_fwd8(B + P50_W * o, s, j * (m / s) + o / s, pp, vpn, vp2);
        }
        uint64_t *Bb = B + P50_W * o;
        size_t jb = j * nb + b;
        if(LS == 8)       p50_fwd8(Bb, 8, jb, pp, vpn, vp2);
        else if(LS == 16) p50_fwd16r(Bb, jb, pp, vpn, vp2);
        else              p50_fwd32r(Bb, jb, pp, vpn, vp2);
    }
}

static void p50_fused_full(uint64_t *B, const uint64_t *A, size_t m,
                           size_t j, const p50_prime *pp, __m512i vpn,
                           __m512i vp2, int sq){
    if(m == 1){ p50_conv8(B, sq ? B : A, pp->w[j], pp, vpn, vp2); return; }
    if(m <= 4){
        /* fwd level(s) over memory, conv per position, inverse */
        p50_vcc w = p50_vcc_of(pp->w[2*j]);
        size_t h = m >> 1;
        for(size_t v = 0; v < h; ++v){
            __m512i a  = p50_sh2(_mm512_load_si512(B + P50_W * v), vp2);
            __m512i wb = p50_b52(_mm512_load_si512(B + P50_W * (v + h)),
                                 w.c, w.rec, vpn);
            _mm512_store_si512(B + P50_W * v, _mm512_add_epi64(a, wb));
            _mm512_store_si512(B + P50_W * (v + h),
                _mm512_add_epi64(a, _mm512_sub_epi64(vp2, wb)));
        }
        p50_fused_full(B,             A,             h, 2*j,     pp, vpn,
                       vp2, sq);
        p50_fused_full(B + P50_W * h, A + P50_W * h, h, 2*j + 1, pp, vpn,
                       vp2, sq);
        p50_vcc wi = p50_vcc_of(pp->winv[2*j]);
        for(size_t v = 0; v < h; ++v){
            __m512i a = p50_sh2(_mm512_load_si512(B + P50_W * v), vp2);
            __m512i b = p50_sh2(_mm512_load_si512(B + P50_W * (v + h)), vp2);
            _mm512_store_si512(B + P50_W * v, _mm512_add_epi64(a, b));
            _mm512_store_si512(B + P50_W * (v + h),
                p50_b52(_mm512_add_epi64(_mm512_sub_epi64(a, b), vp2),
                        wi.c, wi.rec, vpn));
        }
        return;
    }
    if(sq) A = B;                /* squaring: conv leaves against self */
    if(m == 8){  p50_fused8l(B, A, j, pp, vpn, vp2);  return; }
    if(m == 16){ p50_fused16l(B, A, j, pp, vpn, vp2); return; }
    if(m == 32){ p50_fused32l(B, A, j, pp, vpn, vp2); return; }
    size_t LS = p50_leaf_ls(m);
    size_t nb = m / LS;
    for(size_t b = 0; b < nb; ++b){
        size_t o = b * LS;
        for(size_t s = m; s > LS; s >>= 3){
            if(b & (s / LS - 1)) continue;
            p50_fwd8(B + P50_W * o, s, j * (m / s) + o / s, pp, vpn, vp2);
        }
        uint64_t *Bb = B + P50_W * o;
        const uint64_t *Ab = A + P50_W * o;
        size_t jb = j * nb + b;
        if(LS == 8)       p50_fused8l(Bb, Ab, jb, pp, vpn, vp2);
        else if(LS == 16) p50_fused16l(Bb, Ab, jb, pp, vpn, vp2);
        else              p50_fused32l(Bb, Ab, jb, pp, vpn, vp2);
        for(size_t s = 8 * LS; s <= m; s <<= 3){
            if((b + 1) & (s / LS - 1)) continue;
            size_t os = o + LS - s;
            p50_inv8(B + P50_W * os, s, j * (m / s) + os / s, pp, vpn, vp2);
        }
    }
}

/* ------------------------------------------------------------------ */
/* truncated forward spine                                             */
/* ------------------------------------------------------------------ */
static void p50_fwd_rec(uint64_t *B, size_t m, size_t j, size_t zo,
                        const p50_prime *pp, __m512i vpn, __m512i vp2){
    if(m == 1) return;
    if(zo == m){ p50_fwd_full(B, m, j, pp, vpn, vp2); return; }
    size_t h = m >> 1;
    uint64_t *lo = B, *hi = B + P50_W * h;
    p50_vcc w = p50_vcc_of(pp->w[2 * j]);
    if(zo <= h){
        for(size_t v = 0; v < h; ++v){
            __m512i a  = p50_sh2(_mm512_load_si512(lo + P50_W * v), vp2);
            __m512i wb = p50_b52(_mm512_load_si512(hi + P50_W * v),
                                 w.c, w.rec, vpn);
            _mm512_store_si512(lo + P50_W * v, _mm512_add_epi64(a, wb));
        }
        p50_fwd_rec(B, h, 2 * j, zo, pp, vpn, vp2);
        return;
    }
    for(size_t v = 0; v < h; ++v){
        __m512i a  = p50_sh2(_mm512_load_si512(lo + P50_W * v), vp2);
        __m512i wb = p50_b52(_mm512_load_si512(hi + P50_W * v),
                             w.c, w.rec, vpn);
        _mm512_store_si512(lo + P50_W * v, _mm512_add_epi64(a, wb));
        _mm512_store_si512(hi + P50_W * v,
            _mm512_add_epi64(a, _mm512_sub_epi64(vp2, wb)));
    }
    p50_fwd_rec(B, h, 2 * j, h, pp, vpn, vp2);
    p50_fwd_rec(B + P50_W * h, h, 2 * j + 1, zo - h, pp, vpn, vp2);
}

/* ------------------------------------------------------------------ */
/* fused traversal with GENERAL tails in the a-spectrum dead tail      */
/* (the p30 derivation, docs p30 §13; SB == (uint64_t *)A frames)      */
/* ------------------------------------------------------------------ */
static void p50_fused_sb(uint64_t *B, const uint64_t *A, size_t m, size_t j,
                         size_t z, const p50_prime *pp, __m512i vpn,
                         __m512i vp2, int sq){
    if(z == 0) return;
    if(z == m){ p50_fused_full(B, A, m, j, pp, vpn, vp2, sq); return; }
    uint64_t *SB = (uint64_t *)A;
    size_t h = m >> 1;
    uint64_t *lo = B, *hi = B + P50_W * h;
    p50_vcc w = p50_vcc_of(pp->w[2 * j]);
    if(z <= h){
        for(size_t v = 0; v < h; ++v){
            __m512i a  = p50_sh2(_mm512_load_si512(lo + P50_W * v), vp2);
            __m512i wb = p50_b52(_mm512_load_si512(hi + P50_W * v),
                                 w.c, w.rec, vpn);
            _mm512_store_si512(lo + P50_W * v, _mm512_add_epi64(a, wb));
        }
        p50_vcc i2 = p50_vcc_of(pp->inv2);
        for(size_t v = z; v < h; ++v){
            __m512i t  = p50_sh2(_mm512_load_si512(SB + P50_W * v), vp2);
            __m512i wb = p50_b52(_mm512_load_si512(SB + P50_W * (v + h)),
                                 w.c, w.rec, vpn);
            _mm512_store_si512(SB + P50_W * v,
                p50_b52(_mm512_add_epi64(t, wb), i2.c, i2.rec, vpn));
        }
        p50_fused_sb(lo, A, h, 2 * j, z, pp, vpn, vp2, sq);
        for(size_t v = 0; v < z; ++v){
            __m512i a  = p50_sh2(_mm512_load_si512(lo + P50_W * v), vp2);
            __m512i u  = p50_sh2(_mm512_add_epi64(a, a), vp2);
            __m512i wb = p50_b52(_mm512_load_si512(SB + P50_W * (v + h)),
                                 w.c, w.rec, vpn);
            _mm512_store_si512(lo + P50_W * v,
                _mm512_add_epi64(u, _mm512_sub_epi64(vp2, wb)));
        }
        return;
    }
    size_t zr = z - h;
    for(size_t v = 0; v < h; ++v){
        __m512i a  = p50_sh2(_mm512_load_si512(lo + P50_W * v), vp2);
        __m512i wb = p50_b52(_mm512_load_si512(hi + P50_W * v),
                             w.c, w.rec, vpn);
        _mm512_store_si512(lo + P50_W * v, _mm512_add_epi64(a, wb));
        _mm512_store_si512(hi + P50_W * v,
            _mm512_add_epi64(a, _mm512_sub_epi64(vp2, wb)));
    }
    p50_fused_full(lo, A, h, 2 * j, pp, vpn, vp2, sq);
    for(size_t v = zr; v < h; ++v){
        __m512i c  = p50_sh2(_mm512_load_si512(lo + P50_W * v), vp2);
        __m512i wb = p50_b52(_mm512_load_si512(SB + P50_W * (v + h)),
                             w.c, w.rec, vpn);
        __m512i dd = _mm512_add_epi64(c, _mm512_sub_epi64(vp2, wb));
        _mm512_store_si512(SB + P50_W * (v + h), dd);
        _mm512_store_si512(lo + P50_W * v,
            _mm512_add_epi64(c, p50_sh2(dd, vp2)));
    }
    p50_fused_sb(hi, A + P50_W * h, h, 2 * j + 1, zr, pp, vpn, vp2, sq);
    p50_vcc wi = p50_vcc_of(pp->winv[2 * j]);
    for(size_t v = 0; v < zr; ++v){
        __m512i a = p50_sh2(_mm512_load_si512(lo + P50_W * v), vp2);
        __m512i b = p50_sh2(_mm512_load_si512(hi + P50_W * v), vp2);
        _mm512_store_si512(lo + P50_W * v, _mm512_add_epi64(a, b));
        _mm512_store_si512(hi + P50_W * v,
            p50_b52(_mm512_add_epi64(_mm512_sub_epi64(a, b), vp2),
                    wi.c, wi.rec, vpn));
    }
}

static void p50_fused_rec(uint64_t *B, const uint64_t *A, size_t m, size_t j,
                          size_t z, const p50_prime *pp, __m512i vpn,
                          __m512i vp2, int sq){
    if(z == 0) return;
    if(z == m){ p50_fused_full(B, A, m, j, pp, vpn, vp2, sq); return; }
    size_t h = m >> 1;
    uint64_t *lo = B, *hi = B + P50_W * h;
    p50_vcc w = p50_vcc_of(pp->w[2 * j]);
    if(z <= h){
        for(size_t v = 0; v < h; ++v){
            __m512i a  = p50_sh2(_mm512_load_si512(lo + P50_W * v), vp2);
            __m512i wb = p50_b52(_mm512_load_si512(hi + P50_W * v),
                                 w.c, w.rec, vpn);
            _mm512_store_si512(lo + P50_W * v, _mm512_add_epi64(a, wb));
        }
        p50_fused_rec(lo, A, h, 2 * j, z, pp, vpn, vp2, sq);
        for(size_t v = 0; v < z; ++v){
            __m512i x = p50_sh2(_mm512_load_si512(lo + P50_W * v), vp2);
            _mm512_store_si512(lo + P50_W * v, _mm512_add_epi64(x, x));
        }
        return;
    }
    size_t zr = z - h;
    for(size_t v = 0; v < h; ++v){
        __m512i a  = p50_sh2(_mm512_load_si512(lo + P50_W * v), vp2);
        __m512i wb = p50_b52(_mm512_load_si512(hi + P50_W * v),
                             w.c, w.rec, vpn);
        _mm512_store_si512(lo + P50_W * v, _mm512_add_epi64(a, wb));
        _mm512_store_si512(hi + P50_W * v,
            _mm512_add_epi64(a, _mm512_sub_epi64(vp2, wb)));
    }
    p50_fused_full(lo, A, h, 2 * j, pp, vpn, vp2, sq);
    {
        uint64_t *SBc = (uint64_t *)(A + P50_W * h);
        for(size_t v = zr; v < h; ++v){
            __m512i c = p50_sh2(_mm512_load_si512(lo + P50_W * v), vp2);
            _mm512_store_si512(SBc + P50_W * v, c);
            _mm512_store_si512(lo + P50_W * v, _mm512_add_epi64(c, c));
        }
    }
    p50_fused_sb(hi, A + P50_W * h, h, 2 * j + 1, zr, pp, vpn, vp2, sq);
    p50_vcc wi = p50_vcc_of(pp->winv[2 * j]);
    for(size_t v = 0; v < zr; ++v){
        __m512i a = p50_sh2(_mm512_load_si512(lo + P50_W * v), vp2);
        __m512i b = p50_sh2(_mm512_load_si512(hi + P50_W * v), vp2);
        _mm512_store_si512(lo + P50_W * v, _mm512_add_epi64(a, b));
        _mm512_store_si512(hi + P50_W * v,
            p50_b52(_mm512_add_epi64(_mm512_sub_epi64(a, b), vp2),
                    wi.c, wi.rec, vpn));
    }
}

/* ------------------------------------------------------------------ */
/* input: limbs -> 4 residue arrays (canonical), zero-padded to padv   */
/* trunks. Full blocks via the table-driven funnel kernel; the ragged  */
/* operand tail goes through a stack-padded scalar path (the SIMD      */
/* window may not over-read caller memory).                            */
/* ------------------------------------------------------------------ */
static void p50_input(uint64_t *out[P50_NP], const uint64_t *a,
                      size_t nlimb, size_t nat, size_t padv,
                      const p50_tio *tp, const p50_plan *pl){
    const __m512i vM52 = _mm512_set1_epi64((long long)P50_M52);
    const __m512i vMLO = _mm512_set1_epi64(
        (long long)((1ull << tp->LW) - 1));
    __m512i vC[P50_NP], vJ[P50_NP], vP[P50_NP];
    for(int q = 0; q < P50_NP; ++q){
        vC[q] = _mm512_set1_epi64((long long)tp->C[q]);
        vJ[q] = _mm512_set1_epi64((long long)pl->pr[q].J);
        vP[q] = _mm512_set1_epi64((long long)pl->pr[q].p);
    }
    /* full blocks: limb window [lb, lb + BL + 16) must be readable */
    size_t t0 = 0;
    for(; t0 + (size_t)tp->BT <= nat; t0 += (size_t)tp->BT){
        size_t lb = t0 * (size_t)tp->T / 64;
        if(lb + (size_t)tp->BL + 16 > nlimb) break;
        for(int v = 0; v < tp->nv; ++v){
            const uint64_t *base = a + lb + (size_t)tp->vbase[v];
            __m512i L0 = _mm512_loadu_si512(base);
            __m512i L1 = _mm512_loadu_si512(base + 8);
            __m512i iA, sA, A, B, hivec, lovec;
            iA = _mm512_load_si512(tp->loidx[v]);
            sA = _mm512_load_si512(tp->losh[v]);
            A = _mm512_permutex2var_epi64(L0, iA, L1);
            B = _mm512_permutex2var_epi64(L0,
                    _mm512_add_epi64(iA, _mm512_set1_epi64(1)), L1);
            lovec = _mm512_and_si512(_mm512_shrdv_epi64(A, B, sA), vMLO);
            iA = _mm512_load_si512(tp->hiidx[v]);
            sA = _mm512_load_si512(tp->hish[v]);
            A = _mm512_permutex2var_epi64(L0, iA, L1);
            B = _mm512_permutex2var_epi64(L0,
                    _mm512_add_epi64(iA, _mm512_set1_epi64(1)), L1);
            hivec = _mm512_and_si512(_mm512_shrdv_epi64(A, B, sA), vM52);
            size_t ot = t0 + (size_t)8 * v;
            for(int q = 0; q < P50_NP; ++q){
                __m512i z = _mm512_setzero_si512();
                __m512i xl = _mm512_madd52lo_epu64(z, hivec, vC[q]);
                __m512i xh = _mm512_madd52hi_epu64(z, hivec, vC[q]);
                __m512i xm = _mm512_and_si512(xl, vM52);
                __m512i m = _mm512_and_si512(
                    _mm512_madd52lo_epu64(z, xm, vJ[q]), vM52);
                __m512i r = _mm512_madd52hi_epu64(xh, m, vP[q]);
                __mmask8 cy = _mm512_test_epi64_mask(xm, vM52);
                r = _mm512_mask_add_epi64(r, cy, r, _mm512_set1_epi64(1));
                r = _mm512_add_epi64(r, lovec);
                r = p50_shp(r, vP[q]);
                r = p50_shp(r, vP[q]);
                _mm512_storeu_si512(out[q] + ot, r);
            }
        }
    }
    /* ragged tail: scalar over a zero-padded stack copy */
    if(t0 < nat){
        uint64_t pad[160];
        size_t lb = t0 * (size_t)tp->T / 64;
        size_t rem = nlimb > lb ? nlimb - lb : 0;
        if(rem > 144) rem = 144;             /* tail spans < 144 limbs */
        memset(pad, 0, sizeof pad);
        memcpy(pad, a + lb, rem * 8);
        for(size_t t = t0; t < nat; ++t){
            long long bp = (long long)tp->T * (long long)(t - t0)
                         + (long long)((t0 * (size_t)tp->T) & 63);
            size_t q = (size_t)(bp >> 6);
            int s = (int)(bp & 63);
            p50_u128 v = pad[q] | ((p50_u128)pad[q + 1] << 64);
            uint64_t lo = (uint64_t)((v >> s) & ((1ull << tp->LW) - 1));
            /* hi may span into a third limb (s + T > 128) */
            int hs = s + tp->LW;
            p50_u128 vh = (v >> hs) |
                          ((p50_u128)pad[q + 2] << (128 - hs));
            uint64_t hi = (uint64_t)vh & P50_M52;
            for(int qq = 0; qq < P50_NP; ++qq){
                uint64_t p = pl->pr[qq].p;
                out[qq][t] = (uint64_t)((((p50_u128)hi << tp->LW) + lo) % p);
            }
        }
    }
    for(int q = 0; q < P50_NP; ++q)
        memset(out[q] + nat, 0, (padv - nat) * 8);
}

/* ------------------------------------------------------------------ */
/* output: 4 residue arrays ([0,2p) ok) -> carried limbs, trunk-radix  */
/* Garner (docs §3). ntr must be OT-aligned and cover zt + 2 (stream   */
/* delays); caller zeroes residues beyond the live coefficients.       */
/* ------------------------------------------------------------------ */
typedef struct { unsigned cin; } p50_cst;
static inline void p50_chainT(__m512i *lo, __m512i *hi, __m512i alo,
                              __m512i ahi, p50_cst *cs, __m512i LIM){
    __m512i slo = _mm512_add_epi64(*lo, alo);
    __mmask8 lc = _mm512_cmplt_epu64_mask(slo, alo);
    __m512i shi = _mm512_add_epi64(_mm512_add_epi64(*hi, ahi),
                                   _mm512_maskz_set1_epi64(lc, 1));
    __m512i LM1 = _mm512_sub_epi64(LIM, _mm512_set1_epi64(1));
    __mmask8 g  = _mm512_cmpge_epu64_mask(shi, LIM);
    __mmask8 plo = _mm512_cmpeq_epi64_mask(slo, _mm512_set1_epi64(-1));
    __mmask8 pr = plo & _mm512_cmpeq_epi64_mask(shi, LM1);
    unsigned cn = (unsigned)pr + (((unsigned)g << 1) | cs->cin);
    unsigned cy = cn ^ (unsigned)pr;
    slo = _mm512_mask_add_epi64(slo, (__mmask8)cy, slo,
                                _mm512_set1_epi64(1));
    __mmask8 bump = (__mmask8)cy & plo;
    shi = _mm512_mask_add_epi64(shi, bump, shi, _mm512_set1_epi64(1));
    __mmask8 outs = g | ((__mmask8)cy & pr);
    shi = _mm512_mask_sub_epi64(shi, outs, shi, LIM);
    cs->cin = (cn >> 8) & 1u;
    *lo = slo; *hi = shi;
}

static void p50_output(uint64_t *out, uint64_t *const r[P50_NP], size_t ntr,
                       const p50_tio *tp, const p50_plan *pl,
                       uint64_t *scratch){
    const int T = tp->T;
    const __m512i vM52 = _mm512_set1_epi64((long long)P50_M52);
    size_t stl = ntr + 64;
    uint64_t *B0l = scratch,     *B0h = B0l + stl;
    uint64_t *B1l = B0h + stl,   *B1h = B1l + stl;
    uint64_t *B2  = B1h + stl;
    uint64_t *Tl  = B2 + stl,    *Th = Tl + stl;
    memset(B0l, 0, 64); memset(B0h, 0, 64);
    memset(B1l, 0, 64); memset(B1h, 0, 64); memset(B2, 0, 64);

    __m512i vp[P50_NP], vpn[P50_NP], vp2[P50_NP];
    for(int q = 0; q < P50_NP; ++q){
        vp[q]  = _mm512_set1_epi64((long long)pl->pr[q].p);
        vpn[q] = _mm512_set1_epi64((long long)((1ull << 52) - pl->pr[q].p));
        vp2[q] = _mm512_set1_epi64((long long)pl->pr[q].p2);
    }
    for(size_t j = 0; j < ntr; j += 8){
        __m512i vd[P50_NP];
        vd[0] = p50_shp(_mm512_loadu_si512(r[0] + j), vp[0]);
        for(int i = 1; i < P50_NP; ++i){
            __m512i acc = vd[0];
            for(int k = 1; k < i; ++k){
                __m512i u = p50_b52(vd[k],
                        _mm512_set1_epi64((long long)pl->ppm[i][k]),
                        _mm512_set1_epi64((long long)pl->ppm_rec[i][k]),
                        vpn[i]);
                acc = _mm512_add_epi64(acc, u);
                acc = _mm512_min_epu64(acc, _mm512_sub_epi64(acc, vp2[i]));
            }
            __m512i d = _mm512_sub_epi64(
                _mm512_add_epi64(_mm512_loadu_si512(r[i] + j), vp2[i]),
                acc);
            __m512i v = p50_b52(d,
                    _mm512_set1_epi64((long long)pl->inv[i]),
                    _mm512_set1_epi64((long long)pl->inv_rec[i]), vpn[i]);
            vd[i] = p50_shp(v, vp[i]);
        }
        __m512i c0 = vd[0], c1, c2, c3;
        c0 = _mm512_madd52lo_epu64(c0, vd[1],
                 _mm512_set1_epi64((long long)pl->w1[0]));
        c1 = _mm512_madd52hi_epu64(_mm512_setzero_si512(), vd[1],
                 _mm512_set1_epi64((long long)pl->w1[0]));
        c0 = _mm512_madd52lo_epu64(c0, vd[2],
                 _mm512_set1_epi64((long long)pl->w2[0]));
        c1 = _mm512_madd52hi_epu64(c1, vd[2],
                 _mm512_set1_epi64((long long)pl->w2[0]));
        c1 = _mm512_madd52lo_epu64(c1, vd[2],
                 _mm512_set1_epi64((long long)pl->w2[1]));
        c2 = _mm512_madd52hi_epu64(_mm512_setzero_si512(), vd[2],
                 _mm512_set1_epi64((long long)pl->w2[1]));
        c0 = _mm512_madd52lo_epu64(c0, vd[3],
                 _mm512_set1_epi64((long long)pl->w3[0]));
        c1 = _mm512_madd52hi_epu64(c1, vd[3],
                 _mm512_set1_epi64((long long)pl->w3[0]));
        c1 = _mm512_madd52lo_epu64(c1, vd[3],
                 _mm512_set1_epi64((long long)pl->w3[1]));
        c2 = _mm512_madd52hi_epu64(c2, vd[3],
                 _mm512_set1_epi64((long long)pl->w3[1]));
        c2 = _mm512_madd52lo_epu64(c2, vd[3],
                 _mm512_set1_epi64((long long)pl->w3[2]));
        c3 = _mm512_madd52hi_epu64(_mm512_setzero_si512(), vd[3],
                 _mm512_set1_epi64((long long)pl->w3[2]));
        c1 = _mm512_add_epi64(c1, _mm512_srli_epi64(c0, 52));
        c0 = _mm512_and_si512(c0, vM52);
        c2 = _mm512_add_epi64(c2, _mm512_srli_epi64(c1, 52));
        c1 = _mm512_and_si512(c1, vM52);
        c3 = _mm512_add_epi64(c3, _mm512_srli_epi64(c2, 52));
        c2 = _mm512_and_si512(c2, vM52);
        const int Thw = T - 64;
        __m512i b0l = _mm512_or_si512(c0, _mm512_slli_epi64(c1, 52));
        __m512i b0h = _mm512_and_si512(_mm512_srli_epi64(c1, 12),
                          _mm512_set1_epi64((1ll << Thw) - 1));
        __m512i b1l = _mm512_or_si512(
                          _mm512_srli_epi64(c1, (unsigned)(T - 52)),
                          _mm512_slli_epi64(c2, (unsigned)(104 - T)));
        __m512i b1h = _mm512_and_si512(
                          _mm512_or_si512(
                              _mm512_srli_epi64(c2, (unsigned)(T - 40)),
                              _mm512_slli_epi64(c3, (unsigned)(92 - T))),
                          _mm512_set1_epi64((1ll << Thw) - 1));
        __m512i b2 = _mm512_srli_epi64(c3, (unsigned)(2 * T - 156));
        _mm512_storeu_si512(B0l + 8 + j, b0l);
        _mm512_storeu_si512(B0h + 8 + j, b0h);
        _mm512_storeu_si512(B1l + 8 + j, b1l);
        _mm512_storeu_si512(B1h + 8 + j, b1h);
        _mm512_storeu_si512(B2  + 8 + j, b2);
    }
    {
        __m512i LIM = _mm512_set1_epi64(1ll << (T - 64));
        p50_cst cA = {0}, cB = {0};
        for(size_t j = 0; j < ntr; j += 8){
            __m512i lo = _mm512_loadu_si512(B0l + 8 + j);
            __m512i hi = _mm512_loadu_si512(B0h + 8 + j);
            p50_chainT(&lo, &hi, _mm512_loadu_si512(B1l + 7 + j),
                                 _mm512_loadu_si512(B1h + 7 + j), &cA, LIM);
            p50_chainT(&lo, &hi, _mm512_loadu_si512(B2 + 6 + j),
                                 _mm512_setzero_si512(), &cB, LIM);
            _mm512_storeu_si512(Tl + j, lo);
            _mm512_storeu_si512(Th + j, hi);
        }
    }
    {
        size_t nl = ntr * (size_t)T / 64;
        for(size_t L0i = 0; L0i < nl; L0i += (size_t)tp->OL){
            size_t tb = L0i * 64 / (size_t)T;
            for(int ov = 0; ov < tp->onv; ++ov){
                size_t base = tb + (size_t)tp->obase[ov];
                __m512i Wl0 = _mm512_loadu_si512(Tl + base);
                __m512i Wl1 = _mm512_loadu_si512(Tl + base + 8);
                __m512i Wh0 = _mm512_loadu_si512(Th + base);
                __m512i Wh1 = _mm512_loadu_si512(Th + base + 8);
                __m512i iu = _mm512_load_si512(tp->ouidx[ov]);
                __m512i sh = _mm512_load_si512(tp->osh[ov]);
                __m512i ls = _mm512_load_si512(tp->olsh[ov]);
                __m512i la = _mm512_permutex2var_epi64(Wl0, iu, Wl1);
                __m512i ha = _mm512_permutex2var_epi64(Wh0, iu, Wh1);
                __m512i lb = _mm512_permutex2var_epi64(Wl0,
                    _mm512_add_epi64(iu, _mm512_set1_epi64(1)), Wl1);
                __mmask8 sw = (__mmask8)tp->oswp[ov];
                __m512i fa = _mm512_mask_blend_epi64(sw, la, ha);
                __m512i fb = _mm512_maskz_mov_epi64((__mmask8)~sw, ha);
                __m512i limb = _mm512_or_si512(
                    _mm512_shrdv_epi64(fa, fb, sh),
                    _mm512_sllv_epi64(lb, ls));
                _mm512_storeu_si512(out + L0i + 8 * (size_t)ov, limb);
            }
        }
    }
}


/* ------------------------------------------------------------------ */
/* public entry                                                        */
/* ------------------------------------------------------------------ */
static SCRATCH_TLS p50_plan p50_tls_plan;

static inline int p50_mul_r(uint64_t *rp,
                            const uint64_t *ap, ptrdiff_t an,
                            const uint64_t *bp, ptrdiff_t bn,
                            scratch *sc){
    if(an <= 0 || bn <= 0) return 0;
    p50_plan *pl = &p50_tls_plan;
    p50_plan_init(pl);
    uint64_t mn = (uint64_t)(an < bn ? an : bn);
    int ti = -1;
    for(int i = 2; i >= 0; --i){             /* prefer T = 88, 84, 80 */
        const p50_tio *t_ = &pl->tio[i];
        uint64_t mnt = (mn * 64 + (uint64_t)t_->T - 1) / (uint64_t)t_->T;
        if(mnt <= t_->capt){ ti = i; break; }
    }
#ifdef P50_FORCE_T
    ti = P50_FORCE_T;
#endif
    if(ti < 0) return 0;
    const p50_tio *tio = &pl->tio[ti];
    const int T = tio->T;
    size_t nat = ((size_t)an * 64 + (size_t)T - 1) / (size_t)T;
    size_t nbt = ((size_t)bn * 64 + (size_t)T - 1) / (size_t)T;
    size_t zt = nat + nbt - 1;
    size_t tv = (zt + P50_W - 1) / P50_W;
    int lg = 0;
    while(((size_t)1 << lg) < tv) lg++;
    if(lg > P50_MAX_LG) return 0;
    size_t M = (size_t)1 << lg;
    p50_plan_ensure(pl, lg);

    SCRATCH(sc);
    size_t ntp = ((zt + 2 + (size_t)tio->OT - 1) / (size_t)tio->OT)
                 * (size_t)tio->OT;          /* covers stream delays */
    size_t mv = M * P50_W;
    size_t bv = (mv > ntp ? mv : ntp) + 192;
    uint64_t *blk = SALLOC(sc, uint64_t, 8 * bv);
    uint64_t *ra[P50_NP], *rb[P50_NP];
    for(int q = 0; q < P50_NP; ++q){
        ra[q] = blk + (size_t)(2 * q) * bv;
        rb[q] = blk + (size_t)(2 * q + 1) * bv;
    }
    uint64_t *oscr = SALLOC(sc, uint64_t, 7 * (ntp + 64));
    size_t nlp = ntp * (size_t)T / 64;
    uint64_t *st = SALLOC(sc, uint64_t, nlp + 8);

    p50_input(ra, ap, (size_t)an, nat, bv - 64, tio, pl);
    p50_input(rb, bp, (size_t)bn, nbt, bv - 64, tio, pl);

    for(int q = 0; q < P50_NP; ++q){
        const p50_prime *pp = &pl->pr[q];
        __m512i vpn = _mm512_set1_epi64((long long)((1ull << 52) - pp->p));
        __m512i vp2 = _mm512_set1_epi64((long long)pp->p2);
        p50_fwd_rec(ra[q], M, 0, tv, pp, vpn, vp2);
        p50_fused_rec(rb[q], ra[q], M, 0, tv, pp, vpn, vp2, 0);
        /* scale s = M^-1 * 2^52 (cancels the conv REDC stray R^-1) */
        uint64_t sc52 = p50_mulm(p50_invm(M % pp->p, pp->p),
                                 (uint64_t)((((p50_u128)1) << 52) % pp->p),
                                 pp->p);
        p50_vcc vs = p50_vcc_of(p50_cc_of(sc52, pp->p));
        __m512i vp = _mm512_set1_epi64((long long)pp->p);
        for(size_t k = 0; k < tv; ++k){
            uint64_t *v = rb[q] + P50_W * k;
            __m512i x = p50_b52(_mm512_load_si512(v), vs.c, vs.rec, vpn);
            _mm512_store_si512(v, p50_shp(x, vp));
        }
        if(ntp > tv * P50_W)
            memset(rb[q] + tv * P50_W, 0, (ntp - tv * P50_W) * 8);
    }

    p50_output(st, rb, ntp, tio, pl, oscr);
    size_t outl = (size_t)an + (size_t)bn;
    if(outl < nlp && st[outl] != 0) return 0;     /* must not spill */
    memcpy(rp, st, outl * 8);
    return 1;
}

static inline int p50_mul(uint64_t *rp,
                          const uint64_t *ap, ptrdiff_t an,
                          const uint64_t *bp, ptrdiff_t bn){
    return p50_mul_r(rp, ap, an, bp, bn, scratch_thread());
}

/* squaring: one input pass, no standalone forward -- the fused
 * traversal convolves each leaf with itself (sq = 1). The a-spectrum
 * array degenerates to side-buffer scratch; one buffer serves all
 * primes (each prime's pass is complete before the next begins). */
static inline int p50_sqr_r(uint64_t *rp, const uint64_t *ap, ptrdiff_t an,
                            scratch *sc){
    if(an <= 0) return 0;
    p50_plan *pl = &p50_tls_plan;
    p50_plan_init(pl);
    int ti = -1;
    for(int i = 2; i >= 0; --i){             /* prefer T = 88, 84, 80 */
        const p50_tio *t_ = &pl->tio[i];
        uint64_t mnt = ((uint64_t)an * 64 + (uint64_t)t_->T - 1)
                       / (uint64_t)t_->T;
        if(mnt <= t_->capt){ ti = i; break; }
    }
#ifdef P50_FORCE_T
    ti = P50_FORCE_T;
#endif
    if(ti < 0) return 0;
    const p50_tio *tio = &pl->tio[ti];
    const int T = tio->T;
    size_t nat = ((size_t)an * 64 + (size_t)T - 1) / (size_t)T;
    size_t zt = 2 * nat - 1;
    size_t tv = (zt + P50_W - 1) / P50_W;
    int lg = 0;
    while(((size_t)1 << lg) < tv) lg++;
    if(lg > P50_MAX_LG) return 0;
    size_t M = (size_t)1 << lg;
    p50_plan_ensure(pl, lg);

    SCRATCH(sc);
    size_t ntp = ((zt + 2 + (size_t)tio->OT - 1) / (size_t)tio->OT)
                 * (size_t)tio->OT;
    size_t mv = M * P50_W;
    size_t bv = (mv > ntp ? mv : ntp) + 192;
    uint64_t *blk = SALLOC(sc, uint64_t, 5 * bv);
    uint64_t *rb[P50_NP];
    for(int q = 0; q < P50_NP; ++q)
        rb[q] = blk + (size_t)q * bv;
    uint64_t *sbuf = blk + (size_t)P50_NP * bv;
    uint64_t *oscr = SALLOC(sc, uint64_t, 7 * (ntp + 64));
    size_t nlp = ntp * (size_t)T / 64;
    uint64_t *st = SALLOC(sc, uint64_t, nlp + 8);

    p50_input(rb, ap, (size_t)an, nat, bv - 64, tio, pl);

    for(int q = 0; q < P50_NP; ++q){
        const p50_prime *pp = &pl->pr[q];
        __m512i vpn = _mm512_set1_epi64((long long)((1ull << 52) - pp->p));
        __m512i vp2 = _mm512_set1_epi64((long long)pp->p2);
        p50_fused_rec(rb[q], sbuf, M, 0, tv, pp, vpn, vp2, 1);
        /* scale s = M^-1 * 2^52 (cancels the conv REDC stray R^-1) */
        uint64_t sc52 = p50_mulm(p50_invm(M % pp->p, pp->p),
                                 (uint64_t)((((p50_u128)1) << 52) % pp->p),
                                 pp->p);
        p50_vcc vs = p50_vcc_of(p50_cc_of(sc52, pp->p));
        __m512i vp = _mm512_set1_epi64((long long)pp->p);
        for(size_t k = 0; k < tv; ++k){
            uint64_t *v = rb[q] + P50_W * k;
            __m512i x = p50_b52(_mm512_load_si512(v), vs.c, vs.rec, vpn);
            _mm512_store_si512(v, p50_shp(x, vp));
        }
        if(ntp > tv * P50_W)
            memset(rb[q] + tv * P50_W, 0, (ntp - tv * P50_W) * 8);
    }

    p50_output(st, rb, ntp, tio, pl, oscr);
    size_t outl = 2 * (size_t)an;
    if(outl < nlp && st[outl] != 0) return 0;     /* must not spill */
    memcpy(rp, st, outl * 8);
    return 1;
}

static inline int p50_sqr(uint64_t *rp, const uint64_t *ap, ptrdiff_t an){
    return p50_sqr_r(rp, ap, an, scratch_thread());
}

