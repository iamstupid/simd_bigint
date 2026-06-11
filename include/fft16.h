/* fft16.h - AVX-512 PQ FFT bigint multiply. u64-limb I/O, u16-digit
 * internals, C11 header-only. Phase 0: power-of-two transform sizes (M = 1),
 * U16 codec, N_complex in [512, 65536] (i.e. products up to 16384 u64 limbs).
 *
 * Layout: AoSoV tiles of 8 complex per zmm pair, [re0..7 | im0..7], 128 B.
 * Forward is DIF with root e^{+2*pi*i/N} (reference int_fft convention),
 * natural input, bit-reversed output. Leaves are stored WITHOUT the
 * transpose-back ("pi layout"): within each group of 8 tiles, stored tile q
 * lane t holds canonical-BR position t*8 + q of the group. The pointwise
 * twiddle table is generated directly in pi order and the inverse leaves
 * consume pi directly, so the two transposes cancel out of the transform.
 *
 * Stage structure (each choice pinned by microbenchmark, see bench/):
 *   fwd:  fused u16-decode + radix-2^2 unroll-2 at len N (full-zmm reads)
 *         -> radix-2^3 ancestor passes -> leaf
 *   leaf: 32x2 / 64 / 128 points by parity, across-stage + 8x8 transpose +
 *         lane-parallel DFT8, no transpose-back
 *   pw:   PQ pointwise; scalar head over the first 64 positions, 8-lane
 *         mirror pairs above
 *   inv:  mirrored leaves and passes; final radix-2^2-inv at len N fused
 *         with the U16 emit and a base-2^64 SWAR carry (no partial array),
 *         one carry chain per quarter-front + junction fixups.
 */
#ifndef FFT16_H
#define FFT16_H

#include <immintrin.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/* complex vector helpers                                              */
/* ------------------------------------------------------------------ */
typedef struct { __m512d re, im; } fcv;

static inline fcv f_load(const double* p){
    fcv r; r.re = _mm512_load_pd(p); r.im = _mm512_load_pd(p + 8); return r;
}
static inline void f_store(double* p, fcv x){
    _mm512_store_pd(p, x.re); _mm512_store_pd(p + 8, x.im);
}
static inline fcv f_add(fcv a, fcv b){
    fcv r; r.re = _mm512_add_pd(a.re, b.re); r.im = _mm512_add_pd(a.im, b.im); return r;
}
static inline fcv f_sub(fcv a, fcv b){
    fcv r; r.re = _mm512_sub_pd(a.re, b.re); r.im = _mm512_sub_pd(a.im, b.im); return r;
}
static inline fcv f_mul(fcv a, fcv w){            /* a * w */
    fcv r;
    r.re = _mm512_fmsub_pd(a.re, w.re, _mm512_mul_pd(a.im, w.im));
    r.im = _mm512_fmadd_pd(a.re, w.im, _mm512_mul_pd(a.im, w.re));
    return r;
}
static inline fcv f_mulc(fcv a, fcv w){           /* a * conj(w) */
    fcv r;
    r.re = _mm512_fmadd_pd(a.im, w.im, _mm512_mul_pd(a.re, w.re));
    r.im = _mm512_fmsub_pd(a.im, w.re, _mm512_mul_pd(a.re, w.im));
    return r;
}
static inline fcv f_j(fcv a){                     /* +i * a */
    fcv r; r.re = _mm512_sub_pd(_mm512_setzero_pd(), a.im); r.im = a.re; return r;
}
static inline fcv f_mj(fcv a){                    /* -i * a */
    fcv r; r.re = a.im; r.im = _mm512_sub_pd(_mm512_setzero_pd(), a.re); return r;
}
static inline fcv f_rev(fcv a){                   /* lane reverse */
    const __m512i ix = _mm512_setr_epi64(7, 6, 5, 4, 3, 2, 1, 0);
    fcv r; r.re = _mm512_permutexvar_pd(ix, a.re); r.im = _mm512_permutexvar_pd(ix, a.im);
    return r;
}

#define F16_C8 0.70710678118654752440  /* cos(pi/4) */

/* rot8[t](a) = a * e^{+2*pi*i*t/8}, t = 1, 3 (t = 0 id, t = 2 = f_j) */
static inline fcv f_rot81(fcv a){
    const __m512d c = _mm512_set1_pd(F16_C8);
    fcv r;
    r.re = _mm512_mul_pd(c, _mm512_sub_pd(a.re, a.im));
    r.im = _mm512_mul_pd(c, _mm512_add_pd(a.re, a.im));
    return r;
}
static inline fcv f_rot83(fcv a){
    const __m512d c = _mm512_set1_pd(F16_C8);
    fcv r;
    r.re = _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(c, _mm512_add_pd(a.re, a.im)));
    r.im = _mm512_mul_pd(c, _mm512_sub_pd(a.re, a.im));
    return r;
}
/* conjugate rotations for the inverse */
static inline fcv f_irot81(fcv a){
    const __m512d c = _mm512_set1_pd(F16_C8);
    fcv r;
    r.re = _mm512_mul_pd(c, _mm512_add_pd(a.re, a.im));
    r.im = _mm512_mul_pd(c, _mm512_sub_pd(a.im, a.re));
    return r;
}
static inline fcv f_irot83(fcv a){
    const __m512d c = _mm512_set1_pd(F16_C8);
    fcv r;
    r.re = _mm512_mul_pd(c, _mm512_sub_pd(a.im, a.re));
    r.im = _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(c, _mm512_add_pd(a.re, a.im)));
    return r;
}

/* ------------------------------------------------------------------ */
/* 8x8 double transpose (24 shuffles)                                  */
/* ------------------------------------------------------------------ */
static inline void f_tr8(__m512d r[8]){
    __m512d t0 = _mm512_unpacklo_pd(r[0], r[1]);
    __m512d t1 = _mm512_unpackhi_pd(r[0], r[1]);
    __m512d t2 = _mm512_unpacklo_pd(r[2], r[3]);
    __m512d t3 = _mm512_unpackhi_pd(r[2], r[3]);
    __m512d t4 = _mm512_unpacklo_pd(r[4], r[5]);
    __m512d t5 = _mm512_unpackhi_pd(r[4], r[5]);
    __m512d t6 = _mm512_unpacklo_pd(r[6], r[7]);
    __m512d t7 = _mm512_unpackhi_pd(r[6], r[7]);

    __m512d v0 = _mm512_shuffle_f64x2(t0, t2, 0x44);  /* cols 0,2 lo */
    __m512d v1 = _mm512_shuffle_f64x2(t4, t6, 0x44);
    __m512d v2 = _mm512_shuffle_f64x2(t0, t2, 0xEE);  /* cols 4,6 hi */
    __m512d v3 = _mm512_shuffle_f64x2(t4, t6, 0xEE);
    __m512d v4 = _mm512_shuffle_f64x2(t1, t3, 0x44);  /* cols 1,3 */
    __m512d v5 = _mm512_shuffle_f64x2(t5, t7, 0x44);
    __m512d v6 = _mm512_shuffle_f64x2(t1, t3, 0xEE);  /* cols 5,7 */
    __m512d v7 = _mm512_shuffle_f64x2(t5, t7, 0xEE);

    r[0] = _mm512_shuffle_f64x2(v0, v1, 0x88);
    r[2] = _mm512_shuffle_f64x2(v0, v1, 0xDD);
    r[4] = _mm512_shuffle_f64x2(v2, v3, 0x88);
    r[6] = _mm512_shuffle_f64x2(v2, v3, 0xDD);
    r[1] = _mm512_shuffle_f64x2(v4, v5, 0x88);
    r[3] = _mm512_shuffle_f64x2(v4, v5, 0xDD);
    r[5] = _mm512_shuffle_f64x2(v6, v7, 0x88);
    r[7] = _mm512_shuffle_f64x2(v6, v7, 0xDD);
}
static inline void f_tr8cv(fcv v[8]){
    __m512d re[8], im[8];
    for(int i = 0; i < 8; ++i){ re[i] = v[i].re; im[i] = v[i].im; }
    f_tr8(re); f_tr8(im);
    for(int i = 0; i < 8; ++i){ v[i].re = re[i]; v[i].im = im[i]; }
}

/* ------------------------------------------------------------------ */
/* lane-parallel 8-point DFT across 8 vectors (DIF, e^{+} root,        */
/* outputs in sequential = bit-reversed slot order). Unscaled inverse. */
/* ------------------------------------------------------------------ */
static inline void f_dft8(fcv v[8]){
    fcv e0 = f_add(v[0], v[4]), o0 = f_sub(v[0], v[4]);
    fcv e1 = f_add(v[1], v[5]), o1 = f_rot81(f_sub(v[1], v[5]));
    fcv e2 = f_add(v[2], v[6]), o2 = f_j(f_sub(v[2], v[6]));
    fcv e3 = f_add(v[3], v[7]), o3 = f_rot83(f_sub(v[3], v[7]));

    fcv ee0 = f_add(e0, e2), eo0 = f_sub(e0, e2);
    fcv ee1 = f_add(e1, e3), eo1 = f_j(f_sub(e1, e3));
    fcv oe0 = f_add(o0, o2), oo0 = f_sub(o0, o2);
    fcv oe1 = f_add(o1, o3), oo1 = f_j(f_sub(o1, o3));

    v[0] = f_add(ee0, ee1); v[1] = f_sub(ee0, ee1);
    v[2] = f_add(eo0, eo1); v[3] = f_sub(eo0, eo1);
    v[4] = f_add(oe0, oe1); v[5] = f_sub(oe0, oe1);
    v[6] = f_add(oo0, oo1); v[7] = f_sub(oo0, oo1);
}
static inline void f_idft8(fcv v[8]){
    fcv ee0 = f_add(v[0], v[1]), ee1 = f_sub(v[0], v[1]);
    fcv eo0 = f_add(v[2], v[3]), eo1 = f_sub(v[2], v[3]);
    fcv oe0 = f_add(v[4], v[5]), oe1 = f_sub(v[4], v[5]);
    fcv oo0 = f_add(v[6], v[7]), oo1 = f_sub(v[6], v[7]);

    fcv e0 = f_add(ee0, eo0), e2 = f_sub(ee0, eo0);
    fcv t  = f_mj(eo1);
    fcv e1 = f_add(ee1, t),   e3 = f_sub(ee1, t);
    fcv o0 = f_add(oe0, oo0), o2 = f_sub(oe0, oo0);
    t      = f_mj(oo1);
    fcv o1 = f_add(oe1, t),   o3 = f_sub(oe1, t);

    o1 = f_irot81(o1); o2 = f_mj(o2); o3 = f_irot83(o3);
    v[0] = f_add(e0, o0); v[4] = f_sub(e0, o0);
    v[1] = f_add(e1, o1); v[5] = f_sub(e1, o1);
    v[2] = f_add(e2, o2); v[6] = f_sub(e2, o2);
    v[3] = f_add(e3, o3); v[7] = f_sub(e3, o3);
}

/* ------------------------------------------------------------------ */
/* plan: twiddle tables, decode shuffles, workspace                    */
/* ------------------------------------------------------------------ */
#define F16_MAX_R8 4
#define F16_MAX_LG 17

/* per-transform stage shape, derived from lg(n) alone */
typedef struct {
    uint32_t leaf;              /* 32, 64 or 128 */
    uint32_t cnt;               /* number of r8 ancestor stages */
    uint32_t len[F16_MAX_R8];   /* their lengths, descending */
} f16_shape;

static inline f16_shape f16_shape_of(uint32_t n){
    unsigned lgn = (unsigned)__builtin_ctz(n);
    f16_shape s;
    uint32_t leaf_lg = 5u + ((lgn - 7u) % 3u);    /* lgn >= 9 always */
    s.leaf = 1u << leaf_lg;
    s.cnt  = (lgn - 2u - leaf_lg) / 3u;
    uint32_t len = n / 4u;
    for(uint32_t i = 0; i < s.cnt; ++i, len /= 8u) s.len[i] = len;
    return s;
}

/* full-length stage shape: for a transform whose FIRST stage is an r8
 * pass at len m (no prior fused levels). Used by the 3x2 branch-halves. */
static inline f16_shape f16_shape_of_full(uint32_t m){
    unsigned lgm = (unsigned)__builtin_ctz(m);
    f16_shape s;
    uint32_t leaf_lg = 5u + ((lgm - 5u) % 3u);
    s.leaf = 1u << leaf_lg;
    s.cnt  = (lgm - leaf_lg) / 3u;
    uint32_t len = m;
    for(uint32_t i = 0; i < s.cnt; ++i, len /= 8u) s.len[i] = len;
    return s;
}

/* Twiddle tables are shared across transform sizes and grown once:
 *  - tw22[lg]: len-2^lg r22 unroll-2 table (input + final-emit stage of a
 *    size-2^lg transform), 2^lg doubles. Depends on len only.
 *  - tw8[lg]:  len-2^lg r8 stage table, 2^lg doubles. Depends on len only.
 *  - pq: pi-ordered pointwise table. Its values are prefix-stable in the
 *    canonical-BR index (bitrev(k,b)/2^b is width-independent), so the
 *    max_n table serves every n <= max_n as a prefix. */
typedef struct {
    uint32_t pq_n;              /* pow2 branch size the PQ table is built for */
    uint32_t cap_ws;            /* workspace capacity in complex (full N)    */
    double*  tw22[F16_MAX_LG];
    double*  tw8[F16_MAX_LG];
    double*  pq;                /* compact PQ g-table: per 8-tile group, 32
                                 * doubles [re x16 | im x16]; pq_n/2 total */
    /* leaf constants (transform-size independent) */
    _Alignas(64) double l32w[2 * 16];   /* w1, w2  at len 32 (lanes)  */
    _Alignas(64) double l64w[4 * 16];   /* w1, w1*w8, w2, w4 at len 64 */
    _Alignas(64) double l128w[8 * 16];  /* {W1,W2} at len 128, t = 0..3 */
    _Alignas(64) double gh0[16];        /* head g-values: g(p>>2), p = 0..7 */
    _Alignas(64) double wt5[5 * 16];    /* PFA lane fixup w_5^{b*(l%5)} */
    _Alignas(64) double wt7[7 * 16];    /* PFA lane fixup w_7^{b*(l%7)} */
    __m512i  dec_idx[4];        /* u16 digit decode shuffles   */
    int      leaf_init;
    /* workspace */
    double*  da;  double* db;   /* 2N doubles each */
    uint64_t* pada; uint64_t* padb; uint64_t* padr;  /* N/2 limbs each */
    size_t   cap_n;             /* allocation size */
} f16_plan;

static inline void* f16_alloc(size_t bytes){
    void* p = NULL;
    if(posix_memalign(&p, 128, bytes)) abort();
    return p;
}

static inline uint32_t f16_bitrev(uint32_t x, unsigned bits){
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);
    x = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu);
    x = (x << 16) | (x >> 16);
    return bits ? (x >> (32u - bits)) : 0u;
}

#define F16_PI 3.14159265358979323846
/* root e^{+2*pi*i*k/n} */
static inline void f16_root(double* wr, double* wi, uint64_t k, uint64_t n){
    double a = (2.0 * F16_PI * (double)(k % n)) / (double)n;
    *wr = cos(a); *wi = sin(a);
}

/* write one twiddle cv: lanes l = root(scale*(j+l), len) at dst[0..7|8..15] */
static inline void f16_twcv(double* dst, uint64_t j, uint64_t scale, uint64_t len){
    for(int l = 0; l < 8; ++l)
        f16_root(dst + l, dst + 8 + l, (j + (uint64_t)l) * scale, len);
}
/* same multiplied by e^{+2*pi*i/8} */
static inline void f16_twcv8(double* dst, uint64_t j, uint64_t scale, uint64_t len){
    for(int l = 0; l < 8; ++l){
        double wr, wi;
        f16_root(&wr, &wi, (j + (uint64_t)l) * scale, len);
        dst[l]     = F16_C8 * (wr - wi);
        dst[8 + l] = F16_C8 * (wr + wi);
    }
}

/* PQ twiddle for canonical-BR position p (reference pq_twiddle port) */
static inline void f16_pqw(double* wr, double* wi, uint32_t p, uint32_t n, unsigned lgn){
    double gr, gi;
    f16_root(&gr, &gi, f16_bitrev(p >> 2, lgn - 2u), n);
    switch(p & 3u){
    case 0:  *wr = 1.0 + gr; *wi =  gi; break;
    case 1:  *wr = 1.0 - gr; *wi = -gi; break;
    case 2:  *wr = 1.0 - gi; *wi =  gr; break;
    default: *wr = 1.0 + gi; *wi = -gr; break;
    }
}

static inline void f16_leaf_init(f16_plan* pl){
    if(pl->leaf_init) return;
    f16_twcv (pl->l32w + 0,  0, 1, 32);
    f16_twcv (pl->l32w + 16, 0, 2, 32);
    f16_twcv (pl->l64w + 0,  0, 1, 64);
    f16_twcv8(pl->l64w + 16, 0, 1, 64);
    f16_twcv (pl->l64w + 32, 0, 2, 64);
    f16_twcv (pl->l64w + 48, 0, 4, 64);
    for(int t = 0; t < 4; ++t){
        f16_twcv(pl->l128w + 32 * t,      (uint64_t)(8 * t), 1, 128);
        f16_twcv(pl->l128w + 32 * t + 16, (uint64_t)(8 * t), 2, 128);
    }
    /* PFA lane-twiddle fixups: wtM[b] lane l = e^{-2pi i b (l%M) / M} */
    for(int b = 0; b < 5; ++b)
        for(int l = 0; l < 8; ++l){
            double th = -2.0 * F16_PI * (double)(b * (l % 5)) / 5.0;
            pl->wt5[16 * b + l] = cos(th); pl->wt5[16 * b + 8 + l] = sin(th);
        }
    for(int b = 0; b < 7; ++b)
        for(int l = 0; l < 8; ++l){
            double th = -2.0 * F16_PI * (double)(b * (l % 7)) / 7.0;
            pl->wt7[16 * b + l] = cos(th); pl->wt7[16 * b + 8 + l] = sin(th);
        }
    /* digit decode: parity x tile -> 8 dword slots of (2-byte digit, 2 zero) */
    for(int v = 0; v < 4; ++v){
        int parity = v & 1, tile = v >> 1;
        char idx[64];
        for(int l = 0; l < 8; ++l){
            int digit = tile * 16 + 2 * l + parity;
            idx[l * 4 + 0] = (char)(digit * 2);
            idx[l * 4 + 1] = (char)(digit * 2 + 1);
            idx[l * 4 + 2] = (char)0x40;          /* zero source */
            idx[l * 4 + 3] = (char)0x40;
        }
        for(int i = 32; i < 64; ++i) idx[i] = (char)0x40;
        memcpy(&pl->dec_idx[v], idx, 64);
    }
    pl->leaf_init = 1;
}

/* build the len-2^lg r22 unroll-2 table: per iter {w1a, w2a, w1b, w2b} */
static inline void f16_build_tw22(f16_plan* pl, unsigned lg){
    if(pl->tw22[lg]) return;
    uint32_t len = 1u << lg;
    double* tab = (double*)f16_alloc((size_t)len * 8);
    for(uint32_t i = 0; i < len / 64u; ++i){
        double* d = tab + 64u * i;
        uint64_t j = 16u * i;
        f16_twcv(d + 0,  j,     1, len);
        f16_twcv(d + 16, j,     2, len);
        f16_twcv(d + 32, j + 8, 1, len);
        f16_twcv(d + 48, j + 8, 2, len);
    }
    pl->tw22[lg] = tab;
}
/* build the len-2^lg r8 stage table: per iter {W1, W1*w8, W2, W4} */
static inline void f16_build_tw8(f16_plan* pl, unsigned lg){
    if(pl->tw8[lg]) return;
    uint32_t len = 1u << lg;
    double* tab = (double*)f16_alloc((size_t)len * 8);
    for(uint32_t i = 0; i < len / 64u; ++i){
        double* d = tab + 64u * i;
        uint64_t j = 8u * i;
        f16_twcv (d + 0,  j, 1, len);
        f16_twcv8(d + 16, j, 1, len);
        f16_twcv (d + 32, j, 2, len);
        f16_twcv (d + 48, j, 4, len);
    }
    pl->tw8[lg] = tab;
}

/* branch = pow2 transform/branch size (keys the twiddle + PQ tables);
 * nfull = full transform size in complex (keys workspace capacity; equals
 * branch for pow2 transforms, M*branch for PFA). */
static inline void f16_plan_ensure(f16_plan* pl, uint32_t branch, uint32_t nfull){
    f16_leaf_init(pl);
    unsigned lgn = (unsigned)__builtin_ctz(branch);

    if(nfull > pl->cap_ws){
        free(pl->da); free(pl->db);
        free(pl->pada); free(pl->padb); free(pl->padr);
        pl->da   = (double*)f16_alloc((size_t)nfull * 16);
        pl->db   = (double*)f16_alloc((size_t)nfull * 16);
        pl->pada = (uint64_t*)f16_alloc((size_t)nfull * 4 + 64);
        pl->padb = (uint64_t*)f16_alloc((size_t)nfull * 4 + 64);
        pl->padr = (uint64_t*)f16_alloc((size_t)nfull * 4 + 64);
        pl->cap_ws = nfull;
    }
    if(branch > pl->pq_n){
        /* compact PQ table: group g64 covers canonical positions
         * [g64*64, g64*64+64) whose quarter-indices are g64*16 + [0,16);
         * store those 16 g = root(bitrev(k)/branch) values per group. The
         * values are width-independent fractions, so building at the largest
         * branch serves every smaller one as a prefix. */
        free(pl->pq);
        pl->pq = (double*)f16_alloc((size_t)branch * 4);
        for(uint32_t g = 0; g < branch / 64u; ++g)
            for(uint32_t k = 0; k < 16; ++k)
                f16_root(pl->pq + 32u * g + k, pl->pq + 32u * g + 16u + k,
                         f16_bitrev(g * 16u + k, lgn - 2u), branch);
        for(uint32_t p = 0; p < 8; ++p)
            f16_root(pl->gh0 + p, pl->gh0 + 8 + p,
                     f16_bitrev(p >> 2, lgn - 2u), branch);
        pl->pq_n = branch;
    }
    /* stage tables are per-length and shared across sizes; build lazily */
    f16_build_tw22(pl, lgn);
    f16_shape sh = f16_shape_of(branch);
    for(uint32_t s = 0; s < sh.cnt; ++s)
        f16_build_tw8(pl, (unsigned)__builtin_ctz(sh.len[s]));
    if(branch >= 256u){     /* 3x2 branch-halves use full-length stage plans */
        f16_shape shh = f16_shape_of_full(branch / 2u);
        for(uint32_t s = 0; s < shh.cnt; ++s)
            f16_build_tw8(pl, (unsigned)__builtin_ctz(shh.len[s]));
    }
}

/* ------------------------------------------------------------------ */
/* forward: fused decode + r22 unroll-2 at len N                       */
/* ------------------------------------------------------------------ */
static inline __m512d f16_dec1(__m512i raw, __m512i idx){
    __m512i d = _mm512_permutex2var_epi8(raw, idx, _mm512_setzero_si512());
    return _mm512_cvtepu32_pd(_mm512_castsi512_si256(d));
}
/* decode one zmm of limbs (16 complex) into 2 tiles */
static inline void f16_dec2(fcv* t0, fcv* t1, const uint64_t* src, const __m512i* IDX){
    __m512i raw = _mm512_loadu_si512((const void*)src);
    t0->re = f16_dec1(raw, IDX[0]); t0->im = f16_dec1(raw, IDX[1]);
    t1->re = f16_dec1(raw, IDX[2]); t1->im = f16_dec1(raw, IDX[3]);
}

static inline void f16_input_stage(double* data, const uint64_t* src,
                                   uint32_t n, const f16_plan* pl){
    const uint32_t l = n / 4u;                       /* complex per quarter */
    double* p0 = data;
    double* p1 = data + 2u * (size_t)l;
    double* p2 = data + 4u * (size_t)l;
    double* p3 = data + 6u * (size_t)l;
    const uint64_t* s0 = src;
    const uint64_t* s1 = src + l / 2u;               /* limbs: l cx = l/2 */
    const uint64_t* s2 = src + l;
    const uint64_t* s3 = src + 3u * (l / 2u);
    const double* twp = pl->tw22[__builtin_ctz(n)];
    const __m512i* IDX = pl->dec_idx;
    for(uint32_t i = 0; i < l / 16u; ++i){
        fcv a0, a0b, a1, a1b, a2, a2b, a3, a3b;
        f16_dec2(&a0, &a0b, s0, IDX);
        f16_dec2(&a1, &a1b, s1, IDX);
        f16_dec2(&a2, &a2b, s2, IDX);
        f16_dec2(&a3, &a3b, s3, IDX);
        s0 += 8; s1 += 8; s2 += 8; s3 += 8;

        fcv w1 = f_load(twp), w2 = f_load(twp + 16);
        fcv b0 = f_add(a0, a2);
        fcv b2 = f_mul(f_sub(a0, a2), w1);
        fcv b1 = f_add(a1, a3);
        fcv b3 = f_mul(f_sub(a1, a3), f_j(w1));
        f_store(p0, f_add(b0, b1));
        f_store(p1, f_mul(f_sub(b0, b1), w2));
        f_store(p2, f_add(b2, b3));
        f_store(p3, f_mul(f_sub(b2, b3), w2));

        w1 = f_load(twp + 32); w2 = f_load(twp + 48);
        twp += 64;
        b0 = f_add(a0b, a2b);
        b2 = f_mul(f_sub(a0b, a2b), w1);
        b1 = f_add(a1b, a3b);
        b3 = f_mul(f_sub(a1b, a3b), f_j(w1));
        f_store(p0 + 16, f_add(b0, b1));
        f_store(p1 + 16, f_mul(f_sub(b0, b1), w2));
        f_store(p2 + 16, f_add(b2, b3));
        f_store(p3 + 16, f_mul(f_sub(b2, b3), w2));
        p0 += 32; p1 += 32; p2 += 32; p3 += 32;
    }
}

/* ------------------------------------------------------------------ */
/* in-memory r22 unroll-2 stage at len n (branch first stage / final   */
/* inverse for PFA). Same butterfly and tw22 layout as the fused forms.*/
/* ------------------------------------------------------------------ */
static inline void f16_mem_r22(double* data, uint32_t n, const double* tw){
    const uint32_t l = n / 4u;
    double* p0 = data;
    double* p1 = data + 2u * (size_t)l;
    double* p2 = data + 4u * (size_t)l;
    double* p3 = data + 6u * (size_t)l;
    for(uint32_t i = 0; i < l / 16u; ++i){
        for(int u = 0; u < 2; ++u){
            fcv w1 = f_load(tw + 32 * u), w2 = f_load(tw + 32 * u + 16);
            fcv a0 = f_load(p0 + 16 * u), a1 = f_load(p1 + 16 * u);
            fcv a2 = f_load(p2 + 16 * u), a3 = f_load(p3 + 16 * u);
            fcv b0 = f_add(a0, a2), b2 = f_mul(f_sub(a0, a2), w1);
            fcv b1 = f_add(a1, a3), b3 = f_mul(f_sub(a1, a3), f_j(w1));
            f_store(p0 + 16 * u, f_add(b0, b1));
            f_store(p1 + 16 * u, f_mul(f_sub(b0, b1), w2));
            f_store(p2 + 16 * u, f_add(b2, b3));
            f_store(p3 + 16 * u, f_mul(f_sub(b2, b3), w2));
        }
        tw += 64;
        p0 += 32; p1 += 32; p2 += 32; p3 += 32;
    }
}
static inline void f16_mem_ir22(double* data, uint32_t n, const double* tw){
    const uint32_t l = n / 4u;
    double* p0 = data;
    double* p1 = data + 2u * (size_t)l;
    double* p2 = data + 4u * (size_t)l;
    double* p3 = data + 6u * (size_t)l;
    for(uint32_t i = 0; i < l / 16u; ++i){
        for(int u = 0; u < 2; ++u){
            fcv w1 = f_load(tw + 32 * u), w2 = f_load(tw + 32 * u + 16);
            fcv y0 = f_load(p0 + 16 * u), y1 = f_load(p1 + 16 * u);
            fcv y2 = f_load(p2 + 16 * u), y3 = f_load(p3 + 16 * u);
            fcv s  = f_mulc(y1, w2);
            fcv b0 = f_add(y0, s), b1 = f_sub(y0, s);
            s      = f_mulc(y3, w2);
            fcv b2 = f_add(y2, s), b3 = f_sub(y2, s);
            s = f_mulc(b2, w1);
            f_store(p0 + 16 * u, f_add(b0, s));
            f_store(p2 + 16 * u, f_sub(b0, s));
            s = f_mulc(b3, f_j(w1));
            f_store(p1 + 16 * u, f_add(b1, s));
            f_store(p3 + 16 * u, f_sub(b1, s));
        }
        tw += 64;
        p0 += 32; p1 += 32; p2 += 32; p3 += 32;
    }
}

/* ------------------------------------------------------------------ */
/* PFA (Good-Thomas) radix-M machinery, M in {3, 5, 7}.                */
/* omega_M^k = exp(-2*pi*i*k/M) tables (reference int_fft values).     */
/* ------------------------------------------------------------------ */
static const double F16_W3_RE[3] = { 1.0, -0.5, -0.5 };
static const double F16_W3_IM[3] = { 0.0, -0.86602540378443865, 0.86602540378443865 };
static const double F16_W5_RE[5] = { 1.0, 0.30901699437494742, -0.80901699437494742,
                                     -0.80901699437494742, 0.30901699437494742 };
static const double F16_W5_IM[5] = { 0.0, -0.95105651629515357, -0.58778525229247313,
                                     0.58778525229247313, 0.95105651629515357 };
static const double F16_W7_RE[7] = { 1.0, 0.62348980185873353, -0.2225209339563144,
                                     -0.90096886790241913, -0.90096886790241913,
                                     -0.2225209339563144, 0.62348980185873353 };
static const double F16_W7_IM[7] = { 0.0, -0.78183148246802981, -0.97492791218182361,
                                     -0.43388373911755812, 0.43388373911755812,
                                     0.97492791218182361, 0.78183148246802981 };

/* radix-3 Winograd butterfly (reference pfa3, e^- DFT convention) */
static inline void f16_pfa3_fwd(const fcv* x, fcv* y){
    const __m512d ch = _mm512_set1_pd(-0.5);
    const __m512d sh = _mm512_set1_pd(0.86602540378443865);
    fcv s = f_add(x[1], x[2]), d = f_sub(x[1], x[2]);
    y[0] = f_add(x[0], s);
    fcv u; u.re = _mm512_fmadd_pd(ch, s.re, x[0].re);
           u.im = _mm512_fmadd_pd(ch, s.im, x[0].im);
    fcv v; v.re = _mm512_mul_pd(sh, d.re);
           v.im = _mm512_mul_pd(sh, d.im);
    y[1].re = _mm512_add_pd(u.re, v.im); y[1].im = _mm512_sub_pd(u.im, v.re);
    y[2].re = _mm512_sub_pd(u.re, v.im); y[2].im = _mm512_add_pd(u.im, v.re);
}
static inline void f16_pfa3_inv(const fcv* x, fcv* y){
    const __m512d ch = _mm512_set1_pd(-0.5);
    const __m512d sh = _mm512_set1_pd(0.86602540378443865);
    fcv s = f_add(x[1], x[2]), d = f_sub(x[1], x[2]);
    y[0] = f_add(x[0], s);
    fcv u; u.re = _mm512_fmadd_pd(ch, s.re, x[0].re);
           u.im = _mm512_fmadd_pd(ch, s.im, x[0].im);
    fcv v; v.re = _mm512_mul_pd(sh, d.re);
           v.im = _mm512_mul_pd(sh, d.im);
    y[1].re = _mm512_sub_pd(u.re, v.im); y[1].im = _mm512_add_pd(u.im, v.re);
    y[2].re = _mm512_add_pd(u.re, v.im); y[2].im = _mm512_sub_pd(u.im, v.re);
}

/* radix-5 Winograd (reference pfa5: 5 real mults, 17 adds) */
static inline void f16_pfa5_core(const fcv* x, fcv* y, int inv){
    const __m512d CP  = _mm512_set1_pd(-0.25);
    const __m512d CM  = _mm512_set1_pd(0.55901699437494742);
    const __m512d S1  = _mm512_set1_pd(0.95105651629515357);
    const __m512d S1p = _mm512_set1_pd(1.5388417685876267);
    const __m512d S2m = _mm512_set1_pd(-0.36327126400268044);
    fcv u14 = f_add(x[1], x[4]), v14 = f_sub(x[1], x[4]);
    fcv u25 = f_add(x[2], x[3]), v25 = f_sub(x[2], x[3]);
    fcv us = f_add(u14, u25), um = f_sub(u14, u25), vs = f_sub(v14, v25);
    fcv m1; m1.re = _mm512_mul_pd(us.re, CP);  m1.im = _mm512_mul_pd(us.im, CP);
    fcv m2; m2.re = _mm512_mul_pd(um.re, CM);  m2.im = _mm512_mul_pd(um.im, CM);
    fcv m3; m3.re = _mm512_mul_pd(vs.re, S1);  m3.im = _mm512_mul_pd(vs.im, S1);
    fcv m4; m4.re = _mm512_mul_pd(v25.re, S1p); m4.im = _mm512_mul_pd(v25.im, S1p);
    fcv m5; m5.re = _mm512_mul_pd(v14.re, S2m); m5.im = _mm512_mul_pd(v14.im, S2m);
    fcv I14 = f_add(m3, m4), I23 = f_add(m3, m5);
    fcv mid = f_add(x[0], m1);
    fcv R14 = f_add(mid, m2), R23 = f_sub(mid, m2);
    y[0] = f_add(x[0], us);
    if(!inv){
        y[1].re = _mm512_add_pd(R14.re, I14.im); y[1].im = _mm512_sub_pd(R14.im, I14.re);
        y[4].re = _mm512_sub_pd(R14.re, I14.im); y[4].im = _mm512_add_pd(R14.im, I14.re);
        y[2].re = _mm512_add_pd(R23.re, I23.im); y[2].im = _mm512_sub_pd(R23.im, I23.re);
        y[3].re = _mm512_sub_pd(R23.re, I23.im); y[3].im = _mm512_add_pd(R23.im, I23.re);
    }else{
        y[1].re = _mm512_sub_pd(R14.re, I14.im); y[1].im = _mm512_add_pd(R14.im, I14.re);
        y[4].re = _mm512_add_pd(R14.re, I14.im); y[4].im = _mm512_sub_pd(R14.im, I14.re);
        y[2].re = _mm512_sub_pd(R23.re, I23.im); y[2].im = _mm512_add_pd(R23.im, I23.re);
        y[3].re = _mm512_add_pd(R23.re, I23.im); y[3].im = _mm512_sub_pd(R23.im, I23.re);
    }
}

/* radix-7 direct form (reference pfa7: chained FMAs) */
static inline void f16_pfa7_core(const fcv* x, fcv* y, int inv){
    const __m512d C1 = _mm512_set1_pd(0.62348980185873353);
    const __m512d C2 = _mm512_set1_pd(-0.2225209339563144);
    const __m512d C3 = _mm512_set1_pd(-0.90096886790241913);
    const __m512d S1 = _mm512_set1_pd(0.78183148246802981);
    const __m512d S2 = _mm512_set1_pd(0.97492791218182361);
    const __m512d S3 = _mm512_set1_pd(0.43388373911755812);
    fcv u16 = f_add(x[1], x[6]), v16 = f_sub(x[1], x[6]);
    fcv u25 = f_add(x[2], x[5]), v25 = f_sub(x[2], x[5]);
    fcv u34 = f_add(x[3], x[4]), v34 = f_sub(x[3], x[4]);
    fcv R1, R2, R3, I1, I2, I3;
#define F16_R(dst, cA, cB, cC) \
    dst.re = _mm512_fmadd_pd(cC, u34.re, _mm512_fmadd_pd(cB, u25.re, _mm512_fmadd_pd(cA, u16.re, x[0].re))); \
    dst.im = _mm512_fmadd_pd(cC, u34.im, _mm512_fmadd_pd(cB, u25.im, _mm512_fmadd_pd(cA, u16.im, x[0].im)))
    F16_R(R1, C1, C2, C3);
    F16_R(R2, C2, C3, C1);
    F16_R(R3, C3, C1, C2);
#undef F16_R
    I1.re = _mm512_fmadd_pd(S3, v34.re, _mm512_fmadd_pd(S2, v25.re, _mm512_mul_pd(S1, v16.re)));
    I1.im = _mm512_fmadd_pd(S3, v34.im, _mm512_fmadd_pd(S2, v25.im, _mm512_mul_pd(S1, v16.im)));
    I2.re = _mm512_fmsub_pd(S2, v16.re, _mm512_fmadd_pd(S1, v34.re, _mm512_mul_pd(S3, v25.re)));
    I2.im = _mm512_fmsub_pd(S2, v16.im, _mm512_fmadd_pd(S1, v34.im, _mm512_mul_pd(S3, v25.im)));
    I3.re = _mm512_fmadd_pd(S3, v16.re, _mm512_fmsub_pd(S2, v34.re, _mm512_mul_pd(S1, v25.re)));
    I3.im = _mm512_fmadd_pd(S3, v16.im, _mm512_fmsub_pd(S2, v34.im, _mm512_mul_pd(S1, v25.im)));
    y[0].re = _mm512_add_pd(_mm512_add_pd(x[0].re, u16.re), _mm512_add_pd(u25.re, u34.re));
    y[0].im = _mm512_add_pd(_mm512_add_pd(x[0].im, u16.im), _mm512_add_pd(u25.im, u34.im));
#define F16_P(ya, yb, R, I) \
    if(!inv){ ya.re = _mm512_add_pd(R.re, I.im); ya.im = _mm512_sub_pd(R.im, I.re); \
              yb.re = _mm512_sub_pd(R.re, I.im); yb.im = _mm512_add_pd(R.im, I.re); } \
    else    { ya.re = _mm512_sub_pd(R.re, I.im); ya.im = _mm512_add_pd(R.im, I.re); \
              yb.re = _mm512_add_pd(R.re, I.im); yb.im = _mm512_sub_pd(R.im, I.re); }
    F16_P(y[1], y[6], R1, I1)
    F16_P(y[2], y[5], R2, I2)
    F16_P(y[3], y[4], R3, I3)
#undef F16_P
}

static inline void f16_pfa_bfly(const fcv* x, fcv* y, uint32_t M, int inv){
    if(M == 3){ if(inv) f16_pfa3_inv(x, y); else f16_pfa3_fwd(x, y); }
    else if(M == 5) f16_pfa5_core(x, y, inv);
    else            f16_pfa7_core(x, y, inv);
}

/* lane-residue merge masks: km[d] selects lanes l with l % M == d */
static inline void f16_kmasks(uint8_t km[8], uint32_t M){
    for(uint32_t d = 0; d < M; ++d){
        uint8_t m = 0;
        for(uint32_t l = 0; l < 8; ++l)
            if(l % M == d) m |= (uint8_t)(1u << l);
        km[d] = m;
    }
}

/* PFA input stage: decode M natural-order stripes (stripe s = complex
 * [s*n, (s+1)*n), whose k mod n runs 0..n-1 sequentially) and apply the
 * radix-M DFT across the a = k mod M dimension per position. Lane l of the
 * stripe with lane-0 residue r has residue (r+l) mod M, so the per-residue
 * inputs x[a] assemble from the stripes by constant lane-residue masks. */
__attribute__((always_inline))
static inline void f16_pfa_input_impl(double* data, const uint64_t* src,
                                      uint32_t n, uint32_t M, const f16_plan* pl){
    uint8_t km[8]; f16_kmasks(km, M);
    const uint64_t* s[7]; double* p[7]; uint32_t r[7];
    for(uint32_t b = 0; b < M; ++b){
        s[b] = src + (size_t)b * (n / 2u);
        p[b] = data + 2u * (size_t)n * b;
        r[b] = (uint32_t)(((uint64_t)b * n) % M);
    }
    const __m512i* IDX = pl->dec_idx;
    for(uint32_t i = 0; i < n / 16u; ++i){
        fcv t0[7], t1[7];
        for(uint32_t b = 0; b < M; ++b){
            f16_dec2(&t0[b], &t1[b], s[b], IDX);
            s[b] += 8;
        }
        for(int u = 0; u < 2; ++u){
            const fcv* t = u ? t1 : t0;
            fcv x[7], y[7];
            for(uint32_t a = 0; a < M; ++a){
                x[a] = t[0];
                for(uint32_t b = 0; b < M; ++b){
                    uint32_t ru = (r[b] + (uint32_t)(8 * u)) % M;
                    uint32_t d  = (a + M - ru) % M;
                    x[a].re = _mm512_mask_mov_pd(x[a].re, km[d], t[b].re);
                    x[a].im = _mm512_mask_mov_pd(x[a].im, km[d], t[b].im);
                }
            }
            f16_pfa_bfly(x, y, M, 0);
            for(uint32_t b = 0; b < M; ++b)
                f_store(p[b] + 16 * u, y[b]);
        }
        for(uint32_t b = 0; b < M; ++b){
            p[b] += 32;
            r[b] = (r[b] + 16u) % M;
        }
    }
}
/* twiddle-fixup form for M >= 5 (measured: bench/pfa_io_bench). The M-DFT
 * runs directly on residue-relabeled stripe vectors,
 *   y[b](l) = sum_s w^{b r_s} w^{b l} t_s(l),
 * so the M^2 lane merges collapse to M constant per-lane cmuls. At M = 3
 * the merge form above is cheaper than the extra cmuls. */
__attribute__((always_inline))
static inline void f16_pfa_input_tw(double* data, const uint64_t* src,
                                    uint32_t n, uint32_t M, const double* wt,
                                    const f16_plan* pl){
    /* stripe POINTERS are held in residue order (sp[rho] = stripe whose
     * lane-0 residue is currently rho) and rotated by the constant
     * 16 mod M each iteration, so every vector-slot index is compile-time
     * constant and t[]/x[]/y[] fully enregister; the runtime relabel is
     * GPR pointer renaming only. Sub-tile 1 relabels by the constant
     * shift 8 mod M. */
    const uint64_t* sp[7]; double* p[7];
    for(uint32_t b = 0; b < M; ++b){
        uint32_t rho = (uint32_t)(((uint64_t)b * n) % M);
        sp[rho] = src + (size_t)b * (n / 2u);
        p[b] = data + 2u * (size_t)n * b;
    }
    const __m512i* IDX = pl->dec_idx;
    const uint32_t sh8 = 8u % M, sh16 = 16u % M;
    for(uint32_t i = 0; i < n / 16u; ++i){
        fcv t0[7], t1[7], x[7], y[7];
        for(uint32_t rho = 0; rho < M; ++rho){
            f16_dec2(&t0[rho], &t1[rho], sp[rho], IDX);
            sp[rho] += 8;
        }
        f16_pfa_bfly(t0, y, M, 0);
        for(uint32_t b = 0; b < M; ++b){
            fcv w; w.re = _mm512_load_pd(wt + 16u * b);
                   w.im = _mm512_load_pd(wt + 16u * b + 8u);
            f_store(p[b], f_mul(y[b], w));
        }
        for(uint32_t rho = 0; rho < M; ++rho)
            x[(rho + sh8) % M] = t1[rho];
        f16_pfa_bfly(x, y, M, 0);
        for(uint32_t b = 0; b < M; ++b){
            fcv w; w.re = _mm512_load_pd(wt + 16u * b);
                   w.im = _mm512_load_pd(wt + 16u * b + 8u);
            f_store(p[b] + 16, f_mul(y[b], w));
            p[b] += 32;
        }
        const uint64_t* tmp[7];
        for(uint32_t rho = 0; rho < M; ++rho) tmp[(rho + sh16) % M] = sp[rho];
        for(uint32_t rho = 0; rho < M; ++rho) sp[rho] = tmp[rho];
    }
}
/* A/B'd end-to-end vs the plain radix-3 + mem_r22 pipeline: 3x2 LOSES
 * 6-9% (192 limbs: 8.4 vs 9.1 ns/limb; 3072: 9.8 vs 10.7). The deleted
 * mem_r22/mem_ir22 sweeps (~3.5%) are outweighed by the heavier 6-stream
 * I/O kernels and the leaf-flavor rotation (branch-256 halves land on
 * leaf128). Kept switchable; revisit if leaf128 or the rad3 get cheaper. */
#ifndef F16_PFA3X2
#define F16_PFA3X2 0
#endif

/* radix-3x2 input for M = 3: fuses the branch transform's top r2 level
 * (twiddle w_n^{k1} from the tw22 w1 entries) into the radix-3 input pass.
 * 6 read streams (3 stripes x 2 half-offsets), 6 write streams (e/o half
 * per branch). Each branch-half is then an independent n/2 transform that
 * starts directly at the r8 cascade -- no separate mem_r22 sweep. */
static inline void f16_pfa3x2_input(double* data, const uint64_t* src,
                                    uint32_t n, const f16_plan* pl){
    uint8_t km[8]; f16_kmasks(km, 3u);
    const uint64_t* sj[3]; const uint64_t* sk[3];
    double* pe[3]; double* po[3];
    uint32_t rj[3], rk[3];
    for(uint32_t b = 0; b < 3; ++b){
        sj[b] = src + (size_t)b * (n / 2u);
        sk[b] = sj[b] + n / 4u;
        pe[b] = data + 2u * (size_t)n * b;
        po[b] = pe[b] + n;
        rj[b] = (uint32_t)(((uint64_t)b * n) % 3u);
        rk[b] = (uint32_t)(((uint64_t)b * n + n / 2u) % 3u);
    }
    const __m512i* IDX = pl->dec_idx;
    /* the tw22 w1 entries span j in [0, n/4); the second quarter's twiddle
     * is w_n^{j} = i * w_n^{j - n/4} (e^{+} convention), so phase 2 rewinds
     * the table and applies f_j. */
    const double* tw = pl->tw22[__builtin_ctz(n)];
    for(uint32_t i = 0; i < n / 32u; ++i){
        if(i == n / 64u) tw = pl->tw22[__builtin_ctz(n)];
        const int ph2 = i >= n / 64u;
        for(int sub = 0; sub < 2; ++sub){
            fcv tj[3], tk[3];
            for(uint32_t b = 0; b < 3; ++b){
                __m512i raw = _mm512_castsi256_si512(
                    _mm256_loadu_si256((const __m256i*)sj[b]));
                tj[b].re = f16_dec1(raw, IDX[0]);
                tj[b].im = f16_dec1(raw, IDX[1]);
                sj[b] += 4;
                raw = _mm512_castsi256_si512(
                    _mm256_loadu_si256((const __m256i*)sk[b]));
                tk[b].re = f16_dec1(raw, IDX[0]);
                tk[b].im = f16_dec1(raw, IDX[1]);
                sk[b] += 4;
            }
            fcv xj[3], xk[3], yj[3], yk[3];
            for(uint32_t a = 0; a < 3; ++a){
                xj[a] = tj[0]; xk[a] = tk[0];
                for(uint32_t b = 0; b < 3; ++b){
                    uint32_t dj = (a + 3u - rj[b]) % 3u;
                    uint32_t dk = (a + 3u - rk[b]) % 3u;
                    xj[a].re = _mm512_mask_mov_pd(xj[a].re, km[dj], tj[b].re);
                    xj[a].im = _mm512_mask_mov_pd(xj[a].im, km[dj], tj[b].im);
                    xk[a].re = _mm512_mask_mov_pd(xk[a].re, km[dk], tk[b].re);
                    xk[a].im = _mm512_mask_mov_pd(xk[a].im, km[dk], tk[b].im);
                }
            }
            f16_pfa_bfly(xj, yj, 3u, 0);
            f16_pfa_bfly(xk, yk, 3u, 0);
            fcv w; w.re = _mm512_load_pd(tw + 32 * sub);
                   w.im = _mm512_load_pd(tw + 32 * sub + 8);
            if(ph2) w = f_j(w);
            for(uint32_t b = 0; b < 3; ++b){
                f_store(pe[b], f_add(yj[b], yk[b]));
                f_store(po[b], f_mul(f_sub(yj[b], yk[b]), w));
                pe[b] += 16; po[b] += 16;
                rj[b] = (rj[b] + 8u) % 3u;
                rk[b] = (rk[b] + 8u) % 3u;
            }
        }
        tw += 64;
    }
}

/* literal-M dispatcher: lets the compiler specialize the impl per M so the
 * x[]/y[] arrays enregister and the residue arithmetic strength-reduces */
static inline void f16_pfa_input(double* data, const uint64_t* src,
                                 uint32_t n, uint32_t M, const f16_plan* pl){
    switch(M){
    /* end-to-end bisect (bench/pfa_io_bench + full-mul): the merge input
     * wins in the full-mul context for M=3 (and gains the fused r2);
     * rotated-tw wins for M=5/7. */
    case 3:  if(F16_PFA3X2) f16_pfa3x2_input(data, src, n, pl);
             else            f16_pfa_input_impl(data, src, n, 3u, pl);
             break;
    case 5:  f16_pfa_input_tw(data, src, n, 5u, pl->wt5, pl); break;
    default: f16_pfa_input_tw(data, src, n, 7u, pl->wt7, pl); break;
    }
}

/* ------------------------------------------------------------------ */
/* radix-2^3 ancestor pass at length len (table {W1, W1*w8, W2, W4})   */
/* ------------------------------------------------------------------ */
static inline void f16_r8_stage(double* data, uint32_t n, uint32_t len,
                                const double* tw){
    const size_t st = 2u * (size_t)(len / 8u);       /* doubles per eighth */
    for(uint32_t base = 0; base < n; base += len){
        double* p0 = data + 2u * (size_t)base;
        double* p1 = p0 + st;
        double* p2 = p1 + st;
        double* p3 = p2 + st;
        double* p4 = p3 + st;
        double* p5 = p4 + st;
        double* p6 = p5 + st;
        double* p7 = p6 + st;
        const double* twp = tw;
        for(uint32_t t = 0; t < len / 64u; ++t){
            fcv a0 = f_load(p0), a1 = f_load(p1), a2 = f_load(p2), a3 = f_load(p3);
            fcv a4 = f_load(p4), a5 = f_load(p5), a6 = f_load(p6), a7 = f_load(p7);
            fcv W1  = f_load(twp), W1o = f_load(twp + 16);
            fcv W2  = f_load(twp + 32), W4 = f_load(twp + 48);
            twp += 64;

            fcv e0 = f_add(a0, a4), e1 = f_add(a1, a5);
            fcv e2 = f_add(a2, a6), e3 = f_add(a3, a7);
            fcv o0 = f_mul(f_sub(a0, a4), W1);
            fcv o1 = f_mul(f_sub(a1, a5), W1o);
            fcv o2 = f_j(f_mul(f_sub(a2, a6), W1));
            fcv o3 = f_j(f_mul(f_sub(a3, a7), W1o));

            fcv b0 = f_add(e0, e2), b2 = f_mul(f_sub(e0, e2), W2);
            fcv b1 = f_add(e1, e3), b3 = f_mul(f_sub(e1, e3), f_j(W2));
            f_store(p0, f_add(b0, b1));
            f_store(p1, f_mul(f_sub(b0, b1), W4));
            f_store(p2, f_add(b2, b3));
            f_store(p3, f_mul(f_sub(b2, b3), W4));

            b0 = f_add(o0, o2); b2 = f_mul(f_sub(o0, o2), W2);
            b1 = f_add(o1, o3); b3 = f_mul(f_sub(o1, o3), f_j(W2));
            f_store(p4, f_add(b0, b1));
            f_store(p5, f_mul(f_sub(b0, b1), W4));
            f_store(p6, f_add(b2, b3));
            f_store(p7, f_mul(f_sub(b2, b3), W4));
            p0 += 16; p1 += 16; p2 += 16; p3 += 16;
            p4 += 16; p5 += 16; p6 += 16; p7 += 16;
        }
    }
}
static inline void f16_ir8_stage(double* data, uint32_t n, uint32_t len,
                                 const double* tw){
    const size_t st = 2u * (size_t)(len / 8u);
    for(uint32_t base = 0; base < n; base += len){
        double* p0 = data + 2u * (size_t)base;
        double* p1 = p0 + st;
        double* p2 = p1 + st;
        double* p3 = p2 + st;
        double* p4 = p3 + st;
        double* p5 = p4 + st;
        double* p6 = p5 + st;
        double* p7 = p6 + st;
        const double* twp = tw;
        for(uint32_t t = 0; t < len / 64u; ++t){
            fcv W1  = f_load(twp), W1o = f_load(twp + 16);
            fcv W2  = f_load(twp + 32), W4 = f_load(twp + 48);
            twp += 64;
            fcv y0 = f_load(p0), y1 = f_load(p1), y2 = f_load(p2), y3 = f_load(p3);
            /* invert e-half r22 */
            fcv s  = f_mulc(y1, W4);
            fcv b0 = f_add(y0, s), b1 = f_sub(y0, s);
            s      = f_mulc(y3, W4);
            fcv b2 = f_add(y2, s), b3 = f_sub(y2, s);
            s = f_mulc(b2, W2);
            fcv e0 = f_add(b0, s), e2 = f_sub(b0, s);
            s = f_mulc(b3, f_j(W2));
            fcv e1 = f_add(b1, s), e3 = f_sub(b1, s);
            /* invert o-half r22 */
            y0 = f_load(p4); y1 = f_load(p5); y2 = f_load(p6); y3 = f_load(p7);
            s  = f_mulc(y1, W4);
            b0 = f_add(y0, s); b1 = f_sub(y0, s);
            s  = f_mulc(y3, W4);
            b2 = f_add(y2, s); b3 = f_sub(y2, s);
            s = f_mulc(b2, W2);
            fcv o0 = f_add(b0, s), o2 = f_sub(b0, s);
            s = f_mulc(b3, f_j(W2));
            fcv o1 = f_add(b1, s), o3 = f_sub(b1, s);
            /* undo level-1 rotations and recombine */
            o0 = f_mulc(o0, W1);
            o1 = f_mulc(o1, W1o);
            o2 = f_mulc(f_mj(o2), W1);
            o3 = f_mulc(f_mj(o3), W1o);
            f_store(p0, f_add(e0, o0)); f_store(p4, f_sub(e0, o0));
            f_store(p1, f_add(e1, o1)); f_store(p5, f_sub(e1, o1));
            f_store(p2, f_add(e2, o2)); f_store(p6, f_sub(e2, o2));
            f_store(p3, f_add(e3, o3)); f_store(p7, f_sub(e3, o3));
            p0 += 16; p1 += 16; p2 += 16; p3 += 16;
            p4 += 16; p5 += 16; p6 += 16; p7 += 16;
        }
    }
}

/* ------------------------------------------------------------------ */
/* leaves. Output: stored tile slot q lane t = canonical-BR t*8+q (pi) */
/* ------------------------------------------------------------------ */

/* 64-point body: 8 tiles in v[], across-r8 + transpose + DFT8 */
static inline void f16_body64(fcv v[8], const fcv lw[4]){
    fcv W1 = lw[0], W1o = lw[1], W2 = lw[2], W4 = lw[3];
    fcv e0 = f_add(v[0], v[4]), e1 = f_add(v[1], v[5]);
    fcv e2 = f_add(v[2], v[6]), e3 = f_add(v[3], v[7]);
    fcv o0 = f_mul(f_sub(v[0], v[4]), W1);
    fcv o1 = f_mul(f_sub(v[1], v[5]), W1o);
    fcv o2 = f_j(f_mul(f_sub(v[2], v[6]), W1));
    fcv o3 = f_j(f_mul(f_sub(v[3], v[7]), W1o));

    fcv b0 = f_add(e0, e2), b2 = f_mul(f_sub(e0, e2), W2);
    fcv b1 = f_add(e1, e3), b3 = f_mul(f_sub(e1, e3), f_j(W2));
    v[0] = f_add(b0, b1); v[1] = f_mul(f_sub(b0, b1), W4);
    v[2] = f_add(b2, b3); v[3] = f_mul(f_sub(b2, b3), W4);
    b0 = f_add(o0, o2); b2 = f_mul(f_sub(o0, o2), W2);
    b1 = f_add(o1, o3); b3 = f_mul(f_sub(o1, o3), f_j(W2));
    v[4] = f_add(b0, b1); v[5] = f_mul(f_sub(b0, b1), W4);
    v[6] = f_add(b2, b3); v[7] = f_mul(f_sub(b2, b3), W4);

    f_tr8cv(v);
    f_dft8(v);
}
static inline void f16_ibody64(fcv v[8], const fcv lw[4]){
    f_idft8(v);
    f_tr8cv(v);
    fcv W1 = lw[0], W1o = lw[1], W2 = lw[2], W4 = lw[3];
    fcv s  = f_mulc(v[1], W4);
    fcv b0 = f_add(v[0], s), b1 = f_sub(v[0], s);
    s      = f_mulc(v[3], W4);
    fcv b2 = f_add(v[2], s), b3 = f_sub(v[2], s);
    s = f_mulc(b2, W2);
    fcv e0 = f_add(b0, s), e2 = f_sub(b0, s);
    s = f_mulc(b3, f_j(W2));
    fcv e1 = f_add(b1, s), e3 = f_sub(b1, s);
    s  = f_mulc(v[5], W4);
    b0 = f_add(v[4], s); b1 = f_sub(v[4], s);
    s  = f_mulc(v[7], W4);
    b2 = f_add(v[6], s); b3 = f_sub(v[6], s);
    s = f_mulc(b2, W2);
    fcv o0 = f_add(b0, s), o2 = f_sub(b0, s);
    s = f_mulc(b3, f_j(W2));
    fcv o1 = f_add(b1, s), o3 = f_sub(b1, s);
    o0 = f_mulc(o0, W1);
    o1 = f_mulc(o1, W1o);
    o2 = f_mulc(f_mj(o2), W1);
    o3 = f_mulc(f_mj(o3), W1o);
    v[0] = f_add(e0, o0); v[4] = f_sub(e0, o0);
    v[1] = f_add(e1, o1); v[5] = f_sub(e1, o1);
    v[2] = f_add(e2, o2); v[6] = f_sub(e2, o2);
    v[3] = f_add(e3, o3); v[7] = f_sub(e3, o3);
}

static inline void f16_l64const(fcv lw[4], const f16_plan* pl){
    lw[0] = f_load(pl->l64w);      lw[1] = f_load(pl->l64w + 16);
    lw[2] = f_load(pl->l64w + 32); lw[3] = f_load(pl->l64w + 48);
}

/* span runs: constants hoisted out of the per-group loop */
static inline void f16_leaf64_run(double* d, uint32_t span, const f16_plan* pl){
    fcv lw[4]; f16_l64const(lw, pl);
    for(uint32_t g = 0; g < span / 64u; ++g){
        double* p = d + 128u * (size_t)g;
        fcv v[8];
        for(int i = 0; i < 8; ++i) v[i] = f_load(p + 16 * i);
        f16_body64(v, lw);
        for(int i = 0; i < 8; ++i) f_store(p + 16 * i, v[i]);
    }
}
static inline void f16_ileaf64_run(double* d, uint32_t span, const f16_plan* pl){
    fcv lw[4]; f16_l64const(lw, pl);
    for(uint32_t g = 0; g < span / 64u; ++g){
        double* p = d + 128u * (size_t)g;
        fcv v[8];
        for(int i = 0; i < 8; ++i) v[i] = f_load(p + 16 * i);
        f16_ibody64(v, lw);
        for(int i = 0; i < 8; ++i) f_store(p + 16 * i, v[i]);
    }
}

/* two 32-point blocks batched (8 tiles): per-block across-r22, combined
 * transpose + DFT8. Lanes 0..3 = block A sub-blocks, 4..7 = block B. */
static inline void f16_leaf32x2_one(double* p, fcv W1, fcv W2){
    fcv v[8];
    for(int i = 0; i < 8; ++i) v[i] = f_load(p + 16 * i);
    for(int h = 0; h < 8; h += 4){
        fcv b0 = f_add(v[h + 0], v[h + 2]);
        fcv b2 = f_mul(f_sub(v[h + 0], v[h + 2]), W1);
        fcv b1 = f_add(v[h + 1], v[h + 3]);
        fcv b3 = f_mul(f_sub(v[h + 1], v[h + 3]), f_j(W1));
        v[h + 0] = f_add(b0, b1);
        v[h + 1] = f_mul(f_sub(b0, b1), W2);
        v[h + 2] = f_add(b2, b3);
        v[h + 3] = f_mul(f_sub(b2, b3), W2);
    }
    f_tr8cv(v);
    f_dft8(v);
    for(int i = 0; i < 8; ++i) f_store(p + 16 * i, v[i]);
}
static inline void f16_leaf32x2_run(double* d, uint32_t span, const f16_plan* pl){
    fcv W1 = f_load(pl->l32w), W2 = f_load(pl->l32w + 16);
    for(uint32_t g = 0; g < span / 64u; ++g)
        f16_leaf32x2_one(d + 128u * (size_t)g, W1, W2);
}
static inline void f16_ileaf32x2_one(double* p, fcv W1, fcv W2){
    fcv v[8];
    for(int i = 0; i < 8; ++i) v[i] = f_load(p + 16 * i);
    f_idft8(v);
    f_tr8cv(v);
    for(int h = 0; h < 8; h += 4){
        fcv s  = f_mulc(v[h + 1], W2);
        fcv b0 = f_add(v[h + 0], s), b1 = f_sub(v[h + 0], s);
        s      = f_mulc(v[h + 3], W2);
        fcv b2 = f_add(v[h + 2], s), b3 = f_sub(v[h + 2], s);
        s = f_mulc(b2, W1);
        v[h + 0] = f_add(b0, s); v[h + 2] = f_sub(b0, s);
        s = f_mulc(b3, f_j(W1));
        v[h + 1] = f_add(b1, s); v[h + 3] = f_sub(b1, s);
    }
    for(int i = 0; i < 8; ++i) f_store(p + 16 * i, v[i]);
}
static inline void f16_ileaf32x2_run(double* d, uint32_t span, const f16_plan* pl){
    fcv W1 = f_load(pl->l32w), W2 = f_load(pl->l32w + 16);
    for(uint32_t g = 0; g < span / 64u; ++g)
        f16_ileaf32x2_one(d + 128u * (size_t)g, W1, W2);
}

/* 128-point leaf: across-r22 at len 128 (2 levels, streaming over 4-tile
 * groups) + two 32x2 batches (5 levels each). Two register-clean L1 passes;
 * the old r2 + two-64-body form needed three and spilled. The r22 is the
 * same radix-2 decimation/slot convention, so global BR order is unchanged. */
static inline void f16_leaf128_run(double* d, uint32_t span, const f16_plan* pl){
    fcv W1_32 = f_load(pl->l32w), W2_32 = f_load(pl->l32w + 16);
    for(uint32_t g = 0; g < span / 128u; ++g){
        double* p = d + 256u * (size_t)g;
        for(int t = 0; t < 4; ++t){
            fcv a0 = f_load(p + 16 * t),       a1 = f_load(p + 16 * (t + 4));
            fcv a2 = f_load(p + 16 * (t + 8)), a3 = f_load(p + 16 * (t + 12));
            fcv W1; W1.re = _mm512_load_pd(pl->l128w + 32 * t);
                    W1.im = _mm512_load_pd(pl->l128w + 32 * t + 8);
            fcv W2; W2.re = _mm512_load_pd(pl->l128w + 32 * t + 16);
                    W2.im = _mm512_load_pd(pl->l128w + 32 * t + 24);
            fcv b0 = f_add(a0, a2), b2 = f_mul(f_sub(a0, a2), W1);
            fcv b1 = f_add(a1, a3), b3 = f_mul(f_sub(a1, a3), f_j(W1));
            f_store(p + 16 * t,        f_add(b0, b1));
            f_store(p + 16 * (t + 4),  f_mul(f_sub(b0, b1), W2));
            f_store(p + 16 * (t + 8),  f_add(b2, b3));
            f_store(p + 16 * (t + 12), f_mul(f_sub(b2, b3), W2));
        }
        f16_leaf32x2_one(p,       W1_32, W2_32);
        f16_leaf32x2_one(p + 128, W1_32, W2_32);
    }
}
static inline void f16_ileaf128_run(double* d, uint32_t span, const f16_plan* pl){
    fcv W1_32 = f_load(pl->l32w), W2_32 = f_load(pl->l32w + 16);
    for(uint32_t g = 0; g < span / 128u; ++g){
        double* p = d + 256u * (size_t)g;
        f16_ileaf32x2_one(p,       W1_32, W2_32);
        f16_ileaf32x2_one(p + 128, W1_32, W2_32);
        for(int t = 0; t < 4; ++t){
            fcv y0 = f_load(p + 16 * t),       y1 = f_load(p + 16 * (t + 4));
            fcv y2 = f_load(p + 16 * (t + 8)), y3 = f_load(p + 16 * (t + 12));
            fcv W1; W1.re = _mm512_load_pd(pl->l128w + 32 * t);
                    W1.im = _mm512_load_pd(pl->l128w + 32 * t + 8);
            fcv W2; W2.re = _mm512_load_pd(pl->l128w + 32 * t + 16);
                    W2.im = _mm512_load_pd(pl->l128w + 32 * t + 24);
            fcv s  = f_mulc(y1, W2);
            fcv b0 = f_add(y0, s), b1 = f_sub(y0, s);
            s      = f_mulc(y3, W2);
            fcv b2 = f_add(y2, s), b3 = f_sub(y2, s);
            s = f_mulc(b2, W1);
            f_store(p + 16 * t,        f_add(b0, s));
            f_store(p + 16 * (t + 8),  f_sub(b0, s));
            s = f_mulc(b3, f_j(W1));
            f_store(p + 16 * (t + 4),  f_add(b1, s));
            f_store(p + 16 * (t + 12), f_sub(b1, s));
        }
    }
}

static inline void f16_leaf_run(double* d, uint32_t span, uint32_t leaf,
                                const f16_plan* pl){
    if(leaf == 64u)      f16_leaf64_run(d, span, pl);
    else if(leaf == 32u) f16_leaf32x2_run(d, span, pl);
    else                 f16_leaf128_run(d, span, pl);
}
static inline void f16_ileaf_run(double* d, uint32_t span, uint32_t leaf,
                                 const f16_plan* pl){
    if(leaf == 64u)      f16_ileaf64_run(d, span, pl);
    else if(leaf == 32u) f16_ileaf32x2_run(d, span, pl);
    else                 f16_ileaf128_run(d, span, pl);
}

/* ------------------------------------------------------------------ */
/* drivers. Cache blocking: r8 stages longer than F16_BLOCK run as     */
/* full-array sweeps; everything below runs depth-first per chunk so a */
/* chunk's remaining stages + leaves happen while it is cache-hot.     */
/* ------------------------------------------------------------------ */
#ifndef F16_BLOCK
#define F16_BLOCK 2048u   /* complex; 32 KB of tile data */
#endif

/* r8 stages + leaves of one pow2 transform/branch (everything after its
 * first r22 stage), cache-chunked */
static inline void f16_fwd_core(double* data, uint32_t n, const f16_plan* pl){
    f16_shape sh = f16_shape_of(n);
    uint32_t s = 0;
    while(s < sh.cnt && sh.len[s] > F16_BLOCK){
        f16_r8_stage(data, n, sh.len[s], pl->tw8[__builtin_ctz(sh.len[s])]);
        s++;
    }
    uint32_t chunk = s < sh.cnt ? sh.len[s] : (sh.leaf == 128u ? 128u : 64u);
    for(uint32_t base = 0; base < n; base += chunk){
        double* d = data + 2u * (size_t)base;
        for(uint32_t ss = s; ss < sh.cnt; ++ss)
            f16_r8_stage(d, chunk, sh.len[ss], pl->tw8[__builtin_ctz(sh.len[ss])]);
        f16_leaf_run(d, chunk, sh.leaf, pl);
    }
}
static inline void f16_fwd(double* data, const uint64_t* src, uint32_t n,
                           const f16_plan* pl){
    f16_input_stage(data, src, n, pl);
    f16_fwd_core(data, n, pl);
}
/* full-length cores: first stage is an r8 pass at len m itself */
static inline void f16_fwd_core_full(double* data, uint32_t m, const f16_plan* pl){
    f16_shape sh = f16_shape_of_full(m);
    uint32_t s = 0;
    while(s < sh.cnt && sh.len[s] > F16_BLOCK){
        f16_r8_stage(data, m, sh.len[s], pl->tw8[__builtin_ctz(sh.len[s])]);
        s++;
    }
    uint32_t chunk = s < sh.cnt ? sh.len[s] : (sh.leaf == 128u ? 128u : 64u);
    for(uint32_t base = 0; base < m; base += chunk){
        double* d = data + 2u * (size_t)base;
        for(uint32_t ss = s; ss < sh.cnt; ++ss)
            f16_r8_stage(d, chunk, sh.len[ss], pl->tw8[__builtin_ctz(sh.len[ss])]);
        f16_leaf_run(d, chunk, sh.leaf, pl);
    }
}
static inline void f16_inv_r8_only_full(double* data, uint32_t m, const f16_plan* pl){
    f16_shape sh = f16_shape_of_full(m);
    uint32_t s = 0;
    while(s < sh.cnt && sh.len[s] > F16_BLOCK) s++;
    uint32_t chunk = s < sh.cnt ? sh.len[s] : (sh.leaf == 128u ? 128u : 64u);
    if(s < sh.cnt)
        for(uint32_t base = 0; base < m; base += chunk){
            double* d = data + 2u * (size_t)base;
            for(uint32_t ss = sh.cnt; ss-- > s; )
                f16_ir8_stage(d, chunk, sh.len[ss], pl->tw8[__builtin_ctz(sh.len[ss])]);
        }
    for(uint32_t ss = s; ss-- > 0; )
        f16_ir8_stage(data, m, sh.len[ss], pl->tw8[__builtin_ctz(sh.len[ss])]);
}
/* PFA forward. M = 3: the 3x2 input already did the branch's top r2 level,
 * so each branch-half is an independent n/2 transform starting at the r8
 * cascade. M = 5, 7: radix-M input, then mem_r22 + core per branch. */
static inline void f16_pfa_fwd(double* data, const uint64_t* src, uint32_t n,
                               uint32_t M, const f16_plan* pl){
    f16_pfa_input(data, src, n, M, pl);
    if(F16_PFA3X2 && M == 3u){
        for(uint32_t h = 0; h < 6; ++h)
            f16_fwd_core_full(data + (size_t)n * h, n / 2u, pl);
        return;
    }
    const double* tw = pl->tw22[__builtin_ctz(n)];
    for(uint32_t b = 0; b < M; ++b){
        double* d = data + 2u * (size_t)n * b;
        f16_mem_r22(d, n, tw);
        f16_fwd_core(d, n, pl);
    }
}
/* inverse r8 stages only (leaves already applied by the fused pointwise) */
static inline void f16_inv_r8_only(double* data, uint32_t n, const f16_plan* pl){
    f16_shape sh = f16_shape_of(n);
    uint32_t s = 0;
    while(s < sh.cnt && sh.len[s] > F16_BLOCK) s++;
    uint32_t chunk = s < sh.cnt ? sh.len[s] : (sh.leaf == 128u ? 128u : 64u);
    if(s < sh.cnt)
        for(uint32_t base = 0; base < n; base += chunk){
            double* d = data + 2u * (size_t)base;
            for(uint32_t ss = sh.cnt; ss-- > s; )
                f16_ir8_stage(d, chunk, sh.len[ss], pl->tw8[__builtin_ctz(sh.len[ss])]);
        }
    for(uint32_t ss = s; ss-- > 0; )
        f16_ir8_stage(data, n, sh.len[ss], pl->tw8[__builtin_ctz(sh.len[ss])]);
}
/* full inverse (leaves + r8): used by the round-trip test path */
static inline void f16_inv_leaves_r8(double* data, uint32_t n, const f16_plan* pl){
    f16_shape sh = f16_shape_of(n);
    uint32_t s = 0;
    while(s < sh.cnt && sh.len[s] > F16_BLOCK) s++;
    uint32_t chunk = s < sh.cnt ? sh.len[s] : (sh.leaf == 128u ? 128u : 64u);
    for(uint32_t base = 0; base < n; base += chunk){
        double* d = data + 2u * (size_t)base;
        f16_ileaf_run(d, chunk, sh.leaf, pl);
        for(uint32_t ss = sh.cnt; ss-- > s; )
            f16_ir8_stage(d, chunk, sh.len[ss], pl->tw8[__builtin_ctz(sh.len[ss])]);
    }
    for(uint32_t ss = s; ss-- > 0; )
        f16_ir8_stage(data, n, sh.len[ss], pl->tw8[__builtin_ctz(sh.len[ss])]);
}

/* ------------------------------------------------------------------ */
/* PQ pointwise multiply (in pi layout, A *= B elementwise-PQ)         */
/* ------------------------------------------------------------------ */
/* scalar access to canonical-BR position p in pi-stored data */
static inline double* f16_at(double* d, uint32_t p, int im){
    uint32_t g = p >> 6, t = (p >> 3) & 7u, q = p & 7u;
    return d + 16u * ((size_t)g * 8u + q) + (im ? 8u : 0u) + t;
}
static inline void f16_pq_eval_s(double* zr, double* zi,
                                 double xr, double xi, double xnr, double xni,
                                 double yr, double yi, double ynr, double yni,
                                 double wr, double wi, double sc, double qsc){
    double pqr = xr * yr - xi * yi;
    double pqi = xr * yi + xi * yr;
    double dpr = xr - xnr, dpi = xi + xni;
    double dqr = yr - ynr, dqi = yi + yni;
    double tr  = dpr * dqr - dpi * dqi;
    double ti  = dpr * dqi + dpi * dqr;
    double cr  = tr * wr - ti * wi;
    double ci  = tr * wi + ti * wr;
    *zr = pqr * sc - qsc * cr;
    *zi = pqi * sc - qsc * ci;
}
/* paired 8-lane PQ eval (reference pq_eval4_pair at 8 lanes) */
static inline void f16_pq_pair(fcv* outl, fcv* outr, fcv x, fcv xn, fcv y, fcv yn,
                               fcv w, __m512d sc, __m512d qsc){
    fcv pql = f_mul(x, y);
    fcv pqr = f_mul(xn, yn);
    fcv dp; dp.re = _mm512_sub_pd(x.re, xn.re); dp.im = _mm512_add_pd(x.im, xn.im);
    fcv dq; dq.re = _mm512_sub_pd(y.re, yn.re); dq.im = _mm512_add_pd(y.im, yn.im);
    fcv t = f_mul(dp, dq);
    fcv c = f_mul(t, w);
    __m512d qcr = _mm512_mul_pd(qsc, c.re);
    __m512d qci = _mm512_mul_pd(qsc, c.im);
    outl->re = _mm512_fmsub_pd(pql.re, sc, qcr);
    outl->im = _mm512_fmsub_pd(pql.im, sc, qci);
    outr->re = _mm512_fmsub_pd(pqr.re, sc, qcr);
    outr->im = _mm512_fmadd_pd(pqr.im, sc, qci);
}

/* single-sided 8-lane PQ eval: each lane uses its own twiddle */
static inline fcv f16_pq_eval_v(fcv x, fcv xn, fcv y, fcv yn, fcv w,
                                __m512d sc, __m512d qsc){
    fcv pq = f_mul(x, y);
    fcv dp; dp.re = _mm512_sub_pd(x.re, xn.re); dp.im = _mm512_add_pd(x.im, xn.im);
    fcv dq; dq.re = _mm512_sub_pd(y.re, yn.re); dq.im = _mm512_add_pd(y.im, yn.im);
    fcv c = f_mul(f_mul(dp, dq), w);
    fcv z;
    z.re = _mm512_fmsub_pd(pq.re, sc, _mm512_mul_pd(qsc, c.re));
    z.im = _mm512_fmsub_pd(pq.im, sc, _mm512_mul_pd(qsc, c.im));
    return z;
}

/* per-group PQ twiddle reconstruction from the compact g-table: lane t of
 * stored vector (g64, q) needs g[g64*16 + 2t + (q>>2)] (one vpermt2pd per
 * component, hoisted per group as gsel[b]) and the 1+-g pattern by q&3. */
typedef struct { __m512d gr[2], gi[2]; } f16_gsel;
static inline f16_gsel f16_pq_gsel(const double* gblk){
    const __m512i IE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i IO = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    __m512d r0 = _mm512_load_pd(gblk),      r1 = _mm512_load_pd(gblk + 8);
    __m512d i0 = _mm512_load_pd(gblk + 16), i1 = _mm512_load_pd(gblk + 24);
    f16_gsel s;
    s.gr[0] = _mm512_permutex2var_pd(r0, IE, r1);
    s.gr[1] = _mm512_permutex2var_pd(r0, IO, r1);
    s.gi[0] = _mm512_permutex2var_pd(i0, IE, i1);
    s.gi[1] = _mm512_permutex2var_pd(i0, IO, i1);
    return s;
}
static inline fcv f16_pq_w(const f16_gsel* s, uint32_t q){
    __m512d gr = s->gr[q >> 2], gi = s->gi[q >> 2];
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d nz  = _mm512_set1_pd(-0.0);
    fcv w;
    switch(q & 3u){
    case 0:  w.re = _mm512_add_pd(one, gr); w.im = gi; break;
    case 1:  w.re = _mm512_sub_pd(one, gr); w.im = _mm512_xor_pd(gi, nz); break;
    case 2:  w.re = _mm512_sub_pd(one, gi); w.im = gr; break;
    default: w.re = _mm512_add_pd(one, gi); w.im = _mm512_xor_pd(gr, nz); break;
    }
    return w;
}
/* gsel with the PFA branch rotation g' = g * w3 applied (w3 = omega_M^b) */
static inline f16_gsel f16_pq_gsel_rot(const double* gblk, double w3r, double w3i){
    f16_gsel s = f16_pq_gsel(gblk);
    const __m512d vr = _mm512_set1_pd(w3r), vi = _mm512_set1_pd(w3i);
    for(int b = 0; b < 2; ++b){
        __m512d nr = _mm512_fmsub_pd(s.gr[b], vr, _mm512_mul_pd(s.gi[b], vi));
        __m512d ni = _mm512_fmadd_pd(s.gr[b], vi, _mm512_mul_pd(s.gi[b], vr));
        s.gr[b] = nr; s.gi[b] = ni;
    }
    return s;
}
/* head twiddle for canonical positions 0..7 (lane p), from the gh0 g-values
 * rotated by w3; per-lane pattern case c = p&3 via constant blend/sign masks */
static inline fcv f16_pq_head_w(const f16_plan* pl, double w3r, double w3i){
    __m512d gr = _mm512_load_pd(pl->gh0), gi = _mm512_load_pd(pl->gh0 + 8);
    const __m512d vr = _mm512_set1_pd(w3r), vi = _mm512_set1_pd(w3i);
    __m512d gr2 = _mm512_fmsub_pd(gr, vr, _mm512_mul_pd(gi, vi));
    __m512d gi2 = _mm512_fmadd_pd(gr, vi, _mm512_mul_pd(gi, vr));
    const __m512d one = _mm512_set1_pd(1.0);
    const __m512d sre = _mm512_setr_pd(0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0);
    const __m512d sim = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    fcv w;
    w.re = _mm512_add_pd(one, _mm512_xor_pd(_mm512_mask_blend_pd((__mmask8)0xCC, gr2, gi2), sre));
    w.im = _mm512_xor_pd(_mm512_mask_blend_pd((__mmask8)0xCC, gi2, gr2), sim);
    return w;
}

/* PQ pointwise over stored group 0 (canonical positions [0, 64)).
 * Lane 0 holds the first 8 positions: their pairing is part8 across slots,
 * evaluated single-sided per lane (gather column, permute, scatter).
 * Lanes 1..7 are the conjugate-mirror pairing of bases 8/16/32, which in
 * stored coords is slot-reverse (q <-> 7-q) with the SAME part8 pattern as
 * a lane map (1<->1, 2<->3, 4<->7, 5<->6) -- four masked pq_pair calls. */
static inline void f16_pw_head(double* A, double* B, const f16_plan* pl,
                               __m512d sc, __m512d qsc){
    const __m512i P8 = _mm512_setr_epi64(0, 1, 3, 2, 7, 6, 5, 4);
    const __m256i COL = _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);

    fcv x, y;
    x.re = _mm512_i32gather_pd(COL, A, 8);
    x.im = _mm512_i32gather_pd(COL, A + 8, 8);
    y.re = _mm512_i32gather_pd(COL, B, 8);
    y.im = _mm512_i32gather_pd(COL, B + 8, 8);
    fcv w = f16_pq_head_w(pl, 1.0, 0.0);
    fcv xn; xn.re = _mm512_permutexvar_pd(P8, x.re); xn.im = _mm512_permutexvar_pd(P8, x.im);
    fcv yn; yn.re = _mm512_permutexvar_pd(P8, y.re); yn.im = _mm512_permutexvar_pd(P8, y.im);
    fcv z = f16_pq_eval_v(x, xn, y, yn, w, sc, qsc);
    _mm512_i32scatter_pd(A, COL, z.re, 8);
    _mm512_i32scatter_pd(A + 8, COL, z.im, 8);

    f16_gsel gs = f16_pq_gsel(pl->pq);
    for(uint32_t q = 0; q < 4; ++q){
        size_t vl = q, vr = 7u - q;
        fcv xq  = f_load(A + 16u * vl);
        fcv xrw = f_load(A + 16u * vr);
        fcv xnq; xnq.re = _mm512_permutexvar_pd(P8, xrw.re);
                 xnq.im = _mm512_permutexvar_pd(P8, xrw.im);
        fcv yq  = f_load(B + 16u * vl);
        fcv yrw = f_load(B + 16u * vr);
        fcv ynq; ynq.re = _mm512_permutexvar_pd(P8, yrw.re);
                 ynq.im = _mm512_permutexvar_pd(P8, yrw.im);
        fcv wq  = f16_pq_w(&gs, q);
        fcv outl, outr;
        f16_pq_pair(&outl, &outr, xq, xnq, yq, ynq, wq, sc, qsc);
        _mm512_mask_store_pd(A + 16u * vl,     (__mmask8)0xFE, outl.re);
        _mm512_mask_store_pd(A + 16u * vl + 8, (__mmask8)0xFE, outl.im);
        _mm512_mask_store_pd(A + 16u * vr,     (__mmask8)0xFE,
                             _mm512_permutexvar_pd(P8, outr.re));
        _mm512_mask_store_pd(A + 16u * vr + 8, (__mmask8)0xFE,
                             _mm512_permutexvar_pd(P8, outr.im));
    }
}

/* pointwise one group pair: stored vec (gl, q) <-> rev(vec (gr, 7-q)).
 * gl == gr handles the in-group mirror (base 64): q = 0..3 only. */
static inline void f16_pw_groups(double* A, double* B, uint32_t gl, uint32_t gr,
                                 const f16_plan* pl, __m512d sc, __m512d qsc){
    uint32_t qe = gl == gr ? 4u : 8u;
    f16_gsel gs = f16_pq_gsel(pl->pq + 32u * (size_t)gl);
    for(uint32_t q = 0; q < qe; ++q){
        size_t vl = (size_t)gl * 8u + q, vr = (size_t)gr * 8u + 7u - q;
        fcv x  = f_load(A + 16u * vl);
        fcv xn = f_rev(f_load(A + 16u * vr));
        fcv y  = f_load(B + 16u * vl);
        fcv yn = f_rev(f_load(B + 16u * vr));
        fcv w  = f16_pq_w(&gs, q);
        fcv outl, outr;
        f16_pq_pair(&outl, &outr, x, xn, y, yn, w, sc, qsc);
        f_store(A + 16u * vl, outl);
        f_store(A + 16u * vr, f_rev(outr));
    }
}

/* inverse leaf over one 64-cx stored group (or one 128 block for leaf128) */
static inline void f16_ileaf_group(double* A, uint32_t g, uint32_t leaf,
                                   const f16_plan* pl){
    if(leaf == 128u) f16_ileaf128_run(A + 256u * (size_t)(g >> 1), 128u, pl);
    else             f16_ileaf_run(A + 128u * (size_t)g, 64u, leaf, pl);
}

/* fused pointwise + inverse leaves: each mirror pair of groups is pointwise-
 * multiplied and immediately inverse-leafed while cache-hot. For leaf128 the
 * pair structure is block-aligned (gl even, partner block descending), so
 * leaves run once per touched block. */
static inline void f16_pointwise_mul_ileaves(double* A, double* B, uint32_t n,
                                             double sc_d, uint32_t leaf,
                                             const f16_plan* pl){
    const __m512d sc = _mm512_set1_pd(sc_d), qsc = _mm512_set1_pd(0.25 * sc_d);

    f16_pw_head(A, B, pl, sc, qsc);          /* group 0 */
    f16_pw_groups(A, B, 1, 1, pl, sc, qsc);  /* base 64: in-group mirror */
    if(leaf == 128u) f16_ileaf128_run(A, 128u, pl);
    else { f16_ileaf_group(A, 0, leaf, pl); f16_ileaf_group(A, 1, leaf, pl); }

    for(uint32_t base = 128; base < n; base <<= 1){
        uint32_t g0 = base / 64u;
        for(uint32_t gl = g0; gl < g0 + base / 128u; ++gl){
            uint32_t gr = 3u * g0 - 1u - gl;
            f16_pw_groups(A, B, gl, gr, pl, sc, qsc);
            if(leaf != 128u){
                f16_ileaf_group(A, gl, leaf, pl);
                f16_ileaf_group(A, gr, leaf, pl);
            }else if(base == 128u){
                /* gl, gr are the two halves of one 128-block (block-scale
                 * analog of the base-64 in-group mirror) */
                f16_ileaf128_run(A + 256u * (size_t)(gl >> 1), 128u, pl);
            }else if(gl & 1u){   /* both blocks of the pair now complete */
                f16_ileaf128_run(A + 256u * (size_t)((gl - 1u) >> 1), 128u, pl);
                f16_ileaf128_run(A + 256u * (size_t)(gr >> 1), 128u, pl);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* PFA cross-pair pointwise: primary positions in branch L pair with   */
/* the mirrored positions of branch R (the (b, M-b) antipodal pair),   */
/* twiddle g rotated by the constant w3 = omega_M^b. Unlike self-pairs */
/* every (L pos, R pos) pairing is distinct, so the full group/slot    */
/* range is iterated and both heads are evaluated single-sidedly.      */
/* ------------------------------------------------------------------ */
static inline void f16_pw_head_cross(double* L, double* R, double* oL, double* oR,
                                     const f16_plan* pl, double w3r, double w3i,
                                     __m512d sc, __m512d qsc){
    const __m512i P8 = _mm512_setr_epi64(0, 1, 3, 2, 7, 6, 5, 4);
    const __m256i COL = _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112);

    /* lane 0: positions 0..7 of both branches, partner = part8 of the other */
    fcv xl, yl, xr, yr;
    xl.re = _mm512_i32gather_pd(COL, L, 8);  xl.im = _mm512_i32gather_pd(COL, L + 8, 8);
    yl.re = _mm512_i32gather_pd(COL, oL, 8); yl.im = _mm512_i32gather_pd(COL, oL + 8, 8);
    xr.re = _mm512_i32gather_pd(COL, R, 8);  xr.im = _mm512_i32gather_pd(COL, R + 8, 8);
    yr.re = _mm512_i32gather_pd(COL, oR, 8); yr.im = _mm512_i32gather_pd(COL, oR + 8, 8);
    fcv xrp; xrp.re = _mm512_permutexvar_pd(P8, xr.re); xrp.im = _mm512_permutexvar_pd(P8, xr.im);
    fcv yrp; yrp.re = _mm512_permutexvar_pd(P8, yr.re); yrp.im = _mm512_permutexvar_pd(P8, yr.im);
    fcv xlp; xlp.re = _mm512_permutexvar_pd(P8, xl.re); xlp.im = _mm512_permutexvar_pd(P8, xl.im);
    fcv ylp; ylp.re = _mm512_permutexvar_pd(P8, yl.re); ylp.im = _mm512_permutexvar_pd(P8, yl.im);
    fcv wl = f16_pq_head_w(pl, w3r, w3i);
    fcv wr = f16_pq_head_w(pl, w3r, -w3i);
    fcv zl = f16_pq_eval_v(xl, xrp, yl, yrp, wl, sc, qsc);
    fcv zr = f16_pq_eval_v(xr, xlp, yr, ylp, wr, sc, qsc);
    _mm512_i32scatter_pd(L, COL, zl.re, 8);
    _mm512_i32scatter_pd(L + 8, COL, zl.im, 8);
    _mm512_i32scatter_pd(R, COL, zr.re, 8);
    _mm512_i32scatter_pd(R + 8, COL, zr.im, 8);

    /* lanes 1..7, full slot range: L (q, t) <-> R (7-q, part8-lane) */
    f16_gsel gs = f16_pq_gsel_rot(pl->pq, w3r, w3i);
    for(uint32_t q = 0; q < 8; ++q){
        size_t vl = q, vr = 7u - q;
        fcv x   = f_load(L + 16u * vl);
        fcv xrw = f_load(R + 16u * vr);
        fcv xn; xn.re = _mm512_permutexvar_pd(P8, xrw.re);
                xn.im = _mm512_permutexvar_pd(P8, xrw.im);
        fcv y   = f_load(oL + 16u * vl);
        fcv yrw = f_load(oR + 16u * vr);
        fcv yn; yn.re = _mm512_permutexvar_pd(P8, yrw.re);
                yn.im = _mm512_permutexvar_pd(P8, yrw.im);
        fcv w = f16_pq_w(&gs, q);
        fcv outl, outr;
        f16_pq_pair(&outl, &outr, x, xn, y, yn, w, sc, qsc);
        _mm512_mask_store_pd(L + 16u * vl,     (__mmask8)0xFE, outl.re);
        _mm512_mask_store_pd(L + 16u * vl + 8, (__mmask8)0xFE, outl.im);
        _mm512_mask_store_pd(R + 16u * vr,     (__mmask8)0xFE,
                             _mm512_permutexvar_pd(P8, outr.re));
        _mm512_mask_store_pd(R + 16u * vr + 8, (__mmask8)0xFE,
                             _mm512_permutexvar_pd(P8, outr.im));
    }
}

static inline void f16_pw_groups_cross(double* L, double* R, double* oL, double* oR,
                                       uint32_t gl, uint32_t gr, const f16_plan* pl,
                                       double w3r, double w3i,
                                       __m512d sc, __m512d qsc){
    f16_gsel gs = f16_pq_gsel_rot(pl->pq + 32u * (size_t)gl, w3r, w3i);
    for(uint32_t q = 0; q < 8; ++q){
        size_t vl = (size_t)gl * 8u + q, vr = (size_t)gr * 8u + 7u - q;
        fcv x  = f_load(L + 16u * vl);
        fcv xn = f_rev(f_load(R + 16u * vr));
        fcv y  = f_load(oL + 16u * vl);
        fcv yn = f_rev(f_load(oR + 16u * vr));
        fcv w  = f16_pq_w(&gs, q);
        fcv outl, outr;
        f16_pq_pair(&outl, &outr, x, xn, y, yn, w, sc, qsc);
        f_store(L + 16u * vl, outl);
        f_store(R + 16u * vr, f_rev(outr));
    }
}

/* fused cross pointwise + inverse leaves on both branches */
static inline void f16_pointwise_cross_ileaves(double* L, double* R,
                                               double* oL, double* oR,
                                               uint32_t n, double sc_d,
                                               double w3r, double w3i,
                                               uint32_t leaf,
                                               const f16_plan* pl){
    const __m512d sc = _mm512_set1_pd(sc_d), qsc = _mm512_set1_pd(0.25 * sc_d);

    f16_pw_head_cross(L, R, oL, oR, pl, w3r, w3i, sc, qsc);     /* group 0  */
    f16_pw_groups_cross(L, R, oL, oR, 1, 1, pl, w3r, w3i, sc, qsc); /* base 64 */
    if(leaf == 128u){
        f16_ileaf128_run(L, 128u, pl);
        f16_ileaf128_run(R, 128u, pl);
    }else{
        f16_ileaf_group(L, 0, leaf, pl); f16_ileaf_group(L, 1, leaf, pl);
        f16_ileaf_group(R, 0, leaf, pl); f16_ileaf_group(R, 1, leaf, pl);
    }
    for(uint32_t base = 128; base < n; base <<= 1){
        uint32_t g0 = base / 64u;
        for(uint32_t gl = g0; gl < 2u * g0; ++gl){       /* full range */
            uint32_t gr = 3u * g0 - 1u - gl;
            f16_pw_groups_cross(L, R, oL, oR, gl, gr, pl, w3r, w3i, sc, qsc);
            if(leaf != 128u){
                f16_ileaf_group(L, gl, leaf, pl);
                f16_ileaf_group(R, gr, leaf, pl);
            }else if(gl & 1u){
                f16_ileaf128_run(L + 256u * (size_t)((gl - 1u) >> 1), 128u, pl);
                f16_ileaf128_run(R + 256u * (size_t)(gr >> 1), 128u, pl);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* final inverse r22 at len N fused with U16 emit + SWAR carry         */
/* ------------------------------------------------------------------ */
static inline __m512i f16_mround(__m512d x){
    const __m512i bias = _mm512_set1_epi64(0x4330000000000000LL);
    x = _mm512_max_pd(x, _mm512_setzero_pd());
    return _mm512_sub_epi64(
        _mm512_castpd_si512(_mm512_add_pd(x, _mm512_castsi512_pd(bias))), bias);
}
typedef struct { __m512i lo, hi; } f16_lohi;
static inline f16_lohi f16_pack2(fcv t0, fcv t1){
    __m512i re0 = f16_mround(t0.re), im0 = f16_mround(t0.im);
    __m512i re1 = f16_mround(t1.re), im1 = f16_mround(t1.im);
    __m512i ul0 = _mm512_add_epi64(re0, _mm512_slli_epi64(im0, 16));
    __m512i uh0 = _mm512_add_epi64(_mm512_srli_epi64(im0, 48),
        _mm512_maskz_set1_epi64(_mm512_cmplt_epu64_mask(ul0, re0), 1));
    __m512i ul1 = _mm512_add_epi64(re1, _mm512_slli_epi64(im1, 16));
    __m512i uh1 = _mm512_add_epi64(_mm512_srli_epi64(im1, 48),
        _mm512_maskz_set1_epi64(_mm512_cmplt_epu64_mask(ul1, re1), 1));
    const __m512i IE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i IO = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    __m512i elo = _mm512_permutex2var_epi64(ul0, IE, ul1);
    __m512i olo = _mm512_permutex2var_epi64(ul0, IO, ul1);
    __m512i ehi = _mm512_permutex2var_epi64(uh0, IE, uh1);
    __m512i ohi = _mm512_permutex2var_epi64(uh0, IO, uh1);
    f16_lohi r;
    r.lo = _mm512_add_epi64(elo, _mm512_slli_epi64(olo, 32));
    r.hi = _mm512_add_epi64(_mm512_add_epi64(ehi, _mm512_srli_epi64(olo, 32)),
        _mm512_add_epi64(_mm512_slli_epi64(ohi, 32),
        _mm512_maskz_set1_epi64(_mm512_cmplt_epu64_mask(r.lo, elo), 1)));
    return r;
}
typedef struct { __m512i prev_hi; unsigned cin; } f16_chain;
static inline __m512i f16_chain_step(f16_chain* c, f16_lohi p){
    __m512i his = _mm512_alignr_epi64(p.hi, c->prev_hi, 7);
    __m512i sum = _mm512_add_epi64(p.lo, his);
    unsigned g  = _mm512_cmplt_epu64_mask(sum, his);
    unsigned pr = _mm512_cmpeq_epu64_mask(sum, _mm512_set1_epi64(-1));
    unsigned cn = pr + ((g << 1) | c->cin);
    unsigned cy = cn ^ pr;
    sum = _mm512_mask_add_epi64(sum, (__mmask8)cy, sum, _mm512_set1_epi64(1));
    c->cin = (cn >> 8) & 1u;
    c->prev_hi = p.hi;
    return sum;
}

/* rp gets n/2 limbs; returns 1 if the final carry-out is zero */
static inline int f16_inv_final_emit(uint64_t* rp, double* data, uint32_t n,
                                     const f16_plan* pl){
    const size_t limbs_q = n / 8u;                   /* limbs per quarter */
    f16_chain ch[4];
    for(int f = 0; f < 4; ++f){ ch[f].prev_hi = _mm512_setzero_si512(); ch[f].cin = 0; }
    double* p0 = data;
    double* p1 = data + 2u * (size_t)(n / 4u);
    double* p2 = data + 4u * (size_t)(n / 4u);
    double* p3 = data + 6u * (size_t)(n / 4u);
    uint64_t* r0 = rp;
    uint64_t* r1 = rp + limbs_q;
    uint64_t* r2 = rp + 2u * limbs_q;
    uint64_t* r3 = rp + 3u * limbs_q;
    const double* twp = pl->tw22[__builtin_ctz(n)];
    for(uint32_t t = 0; t < n / 64u; ++t){
        fcv o0[2], o1[2], o2[2], o3[2];
        for(int u = 0; u < 2; ++u){
            fcv w1 = f_load(twp + 32 * u), w2 = f_load(twp + 32 * u + 16);
            fcv y0 = f_load(p0 + 16 * u), y1 = f_load(p1 + 16 * u);
            fcv y2 = f_load(p2 + 16 * u), y3 = f_load(p3 + 16 * u);
            fcv s  = f_mulc(y1, w2);
            fcv b0 = f_add(y0, s), b1 = f_sub(y0, s);
            s      = f_mulc(y3, w2);
            fcv b2 = f_add(y2, s), b3 = f_sub(y2, s);
            s = f_mulc(b2, w1);
            o0[u] = f_add(b0, s); o2[u] = f_sub(b0, s);
            s = f_mulc(b3, f_j(w1));
            o1[u] = f_add(b1, s); o3[u] = f_sub(b1, s);
        }
        twp += 64;
        p0 += 32; p1 += 32; p2 += 32; p3 += 32;
        f16_lohi q;
        q = f16_pack2(o0[0], o0[1]); _mm512_storeu_si512((void*)r0, f16_chain_step(&ch[0], q));
        q = f16_pack2(o1[0], o1[1]); _mm512_storeu_si512((void*)r1, f16_chain_step(&ch[1], q));
        q = f16_pack2(o2[0], o2[1]); _mm512_storeu_si512((void*)r2, f16_chain_step(&ch[2], q));
        q = f16_pack2(o3[0], o3[1]); _mm512_storeu_si512((void*)r3, f16_chain_step(&ch[3], q));
        r0 += 8; r1 += 8; r2 += 8; r3 += 8;
    }
    /* junction fixups: front f's trailing (hi lane 7 + cout) -> front f+1 */
    for(int f = 0; f < 3; ++f){
        uint64_t hi7;
        memcpy(&hi7, (const char*)&ch[f].prev_hi + 56, 8);
        unsigned __int128 c = (unsigned __int128)hi7 + ch[f].cin;
        uint64_t* q = rp + (size_t)(f + 1) * limbs_q;
        uint64_t* qe = rp + n / 2u;
        while(c && q < qe){ c += *q; *q++ = (uint64_t)c; c >>= 64; }
        if(c) return 0;
    }
    {
        uint64_t hi7;
        memcpy(&hi7, (const char*)&ch[3].prev_hi + 56, 8);
        if(hi7 + ch[3].cin != 0) return 0;
    }
    return 1;
}

/* PFA inverse output stage fused with the emit: reads M branch streams
 * (each branch already fully inverse-transformed), applies the inverse
 * radix-M butterfly per tile, lane-shuffles into M natural-order region
 * streams (region b = natural complex [b*n, (b+1)*n)), and packs each
 * region with its own SWAR carry chain. Region b's tile at step t takes
 * butterfly output y[(b*n + 8t) mod M] lane-shuffled by residue masks. */
__attribute__((always_inline))
static inline int f16_pfa_emit_impl(uint64_t* rp, double* data, uint32_t n,
                                    uint32_t M, const f16_plan* pl){
    (void)pl;
    uint8_t km[8]; f16_kmasks(km, M);
    const double* br[7]; uint64_t* r[7]; uint32_t phi[7];
    f16_chain ch[7];
    const size_t limbs_b = (size_t)n / 2u;
    for(uint32_t b = 0; b < M; ++b){
        br[b]  = data + 2u * (size_t)n * b;
        r[b]   = rp + (size_t)b * limbs_b;
        phi[b] = (uint32_t)(((uint64_t)b * n) % M);
        ch[b].prev_hi = _mm512_setzero_si512();
        ch[b].cin = 0;
    }
    /* Both sub-tiles of all fronts cannot stay in registers; stage the
     * shuffled butterfly outputs in an L1 buffer and pack from it. */
    _Alignas(64) double stage[7 * 2 * 16];
    for(uint32_t t = 0; t < n / 8u; t += 2){
        for(int u = 0; u < 2; ++u){
            fcv x[7], y[7];
            for(uint32_t b = 0; b < M; ++b){
                x[b] = f_load(br[b]);
                br[b] += 16;
            }
            f16_pfa_bfly(x, y, M, 1);
            /* region b output: lane l takes y[(phi_b + l) mod M] */
            for(uint32_t b = 0; b < M; ++b){
                uint32_t ph = (phi[b] + (uint32_t)(8 * u)) % M;
                fcv o = y[0];
                for(uint32_t j = 0; j < M; ++j){
                    uint32_t d = (j + M - ph) % M;
                    o.re = _mm512_mask_mov_pd(o.re, km[d], y[j].re);
                    o.im = _mm512_mask_mov_pd(o.im, km[d], y[j].im);
                }
                f_store(stage + 32u * b + 16u * (uint32_t)u, o);
            }
        }
        for(uint32_t b = 0; b < M; ++b){
            f16_lohi q = f16_pack2(f_load(stage + 32u * b),
                                   f_load(stage + 32u * b + 16u));
            _mm512_storeu_si512((void*)r[b], f16_chain_step(&ch[b], q));
            r[b] += 8;
            phi[b] = (phi[b] + 16u) % M;
        }
    }
    /* junction fixups between region fronts; top carry must vanish */
    for(uint32_t b = 0; b + 1u < M; ++b){
        uint64_t hi7;
        memcpy(&hi7, (const char*)&ch[b].prev_hi + 56, 8);
        unsigned __int128 c = (unsigned __int128)hi7 + ch[b].cin;
        uint64_t* q = rp + (size_t)(b + 1u) * limbs_b;
        uint64_t* qe = rp + (size_t)M * limbs_b;
        while(c && q < qe){ c += *q; *q++ = (uint64_t)c; c >>= 64; }
        if(c) return 0;
    }
    {
        uint64_t hi7;
        memcpy(&hi7, (const char*)&ch[M - 1u].prev_hi + 56, 8);
        if(hi7 + ch[M - 1u].cin != 0) return 0;
    }
    return 1;
}
/* twiddle-fixup emit for M >= 5: pre-rotate by conj(w^{b l}), inverse-DFT
 * over relabeled branches; the region output is then a pure index relabel. */
__attribute__((always_inline))
static inline int f16_pfa_emit_tw(uint64_t* rp, double* data, uint32_t n,
                                  uint32_t M, const double* wt,
                                  const f16_plan* pl){
    (void)pl;
    const double* br[7]; uint64_t* r[7]; uint32_t phi[7];
    f16_chain ch[7];
    const size_t limbs_b = (size_t)n / 2u;
    for(uint32_t b = 0; b < M; ++b){
        br[b]  = data + 2u * (size_t)n * b;
        r[b]   = rp + (size_t)b * limbs_b;
        phi[b] = (uint32_t)(((uint64_t)b * n) % M);
        ch[b].prev_hi = _mm512_setzero_si512();
        ch[b].cin = 0;
    }
    /* region-slot map in phi order: rg[phi] = region whose front residue is
     * currently phi; y[] stays constant-indexed and the runtime relabel
     * moves into the scalar staging address. rg rotates by 16 mod M. */
    uint32_t rg[7];
    for(uint32_t b = 0; b < M; ++b)
        rg[(uint32_t)(((uint64_t)b * n) % M)] = b;
    const uint32_t sh8 = 8u % M, sh16 = 16u % M;
    _Alignas(64) double stage[7 * 2 * 16];
    for(uint32_t t = 0; t < n / 8u; t += 2){
        for(int u = 0; u < 2; ++u){
            fcv x[7], y[7];
            for(uint32_t b = 0; b < M; ++b){
                fcv w; w.re = _mm512_load_pd(wt + 16u * b);
                       w.im = _mm512_load_pd(wt + 16u * b + 8u);
                x[b] = f_mulc(f_load(br[b]), w);
                br[b] += 16;
            }
            f16_pfa_bfly(x, y, M, 1);
            /* y[j] belongs to the region whose phi (this sub-tile) == j */
            for(uint32_t j = 0; j < M; ++j){
                uint32_t slot = u ? rg[(j + M - sh8) % M] : rg[j];
                f_store(stage + 32u * slot + 16u * (uint32_t)u, y[j]);
            }
        }
        for(uint32_t b = 0; b < M; ++b){
            f16_lohi q = f16_pack2(f_load(stage + 32u * b),
                                   f_load(stage + 32u * b + 16u));
            _mm512_storeu_si512((void*)r[b], f16_chain_step(&ch[b], q));
            r[b] += 8;
        }
        uint32_t tmp[7];
        for(uint32_t j = 0; j < M; ++j) tmp[(j + sh16) % M] = rg[j];
        for(uint32_t j = 0; j < M; ++j) rg[j] = tmp[j];
    }
    for(uint32_t b = 0; b + 1u < M; ++b){
        uint64_t hi7;
        memcpy(&hi7, (const char*)&ch[b].prev_hi + 56, 8);
        unsigned __int128 c = (unsigned __int128)hi7 + ch[b].cin;
        uint64_t* q = rp + (size_t)(b + 1u) * limbs_b;
        uint64_t* qe = rp + (size_t)M * limbs_b;
        while(c && q < qe){ c += *q; *q++ = (uint64_t)c; c >>= 64; }
        if(c) return 0;
    }
    {
        uint64_t hi7;
        memcpy(&hi7, (const char*)&ch[M - 1u].prev_hi + 56, 8);
        if(hi7 + ch[M - 1u].cin != 0) return 0;
    }
    return 1;
}
/* stage-free emit: both butterflies' outputs are held in registers and
 * packed with constant indices (j, (j+8%M)%M); ALL per-region state --
 * store pointer and carry chain -- lives in phi-rotating slots, so the
 * per-step relabel is pure GPR/SSA renaming. Region b's chain finishes in
 * slot ((b+1)*n) mod M (phi advances by n over the whole pass). */
__attribute__((always_inline))
static inline int f16_pfa_emit_rot(uint64_t* rp, double* data, uint32_t n,
                                   uint32_t M, const double* wt,
                                   const f16_plan* pl){
    (void)pl;
    const double* br[7]; uint64_t* rph[7];
    f16_chain ch[7];
    const size_t limbs_b = (size_t)n / 2u;
    for(uint32_t b = 0; b < M; ++b){
        br[b] = data + 2u * (size_t)n * b;
        uint32_t ph = (uint32_t)(((uint64_t)b * n) % M);
        rph[ph] = rp + (size_t)b * limbs_b;
        ch[ph].prev_hi = _mm512_setzero_si512();
        ch[ph].cin = 0;
    }
    const uint32_t sh8 = 8u % M, sh16 = 16u % M;
    for(uint32_t t = 0; t < n / 8u; t += 2){
        fcv y0[7], y1[7];
        {
            fcv x[7];
            for(uint32_t b = 0; b < M; ++b){
                fcv w; w.re = _mm512_load_pd(wt + 16u * b);
                       w.im = _mm512_load_pd(wt + 16u * b + 8u);
                x[b] = f_mulc(f_load(br[b]), w);
            }
            f16_pfa_bfly(x, y0, M, 1);
        }
        {
            fcv x[7];
            for(uint32_t b = 0; b < M; ++b){
                fcv w; w.re = _mm512_load_pd(wt + 16u * b);
                       w.im = _mm512_load_pd(wt + 16u * b + 8u);
                x[b] = f_mulc(f_load(br[b] + 16), w);
                br[b] += 32;
            }
            f16_pfa_bfly(x, y1, M, 1);
        }
        for(uint32_t j = 0; j < M; ++j){
            f16_lohi q = f16_pack2(y0[j], y1[(j + sh8) % M]);
            _mm512_storeu_si512((void*)rph[j], f16_chain_step(&ch[j], q));
            rph[j] += 8;
        }
        uint64_t* tp[7]; f16_chain tc[7];
        for(uint32_t j = 0; j < M; ++j){ tp[(j + sh16) % M] = rph[j]; tc[(j + sh16) % M] = ch[j]; }
        for(uint32_t j = 0; j < M; ++j){ rph[j] = tp[j]; ch[j] = tc[j]; }
    }
    for(uint32_t b = 0; b + 1u < M; ++b){
        uint32_t sl = (uint32_t)(((uint64_t)(b + 1u) * n) % M);
        uint64_t hi7;
        memcpy(&hi7, (const char*)&ch[sl].prev_hi + 56, 8);
        unsigned __int128 c = (unsigned __int128)hi7 + ch[sl].cin;
        uint64_t* q = rp + (size_t)(b + 1u) * limbs_b;
        uint64_t* qe = rp + (size_t)M * limbs_b;
        while(c && q < qe){ c += *q; *q++ = (uint64_t)c; c >>= 64; }
        if(c) return 0;
    }
    {
        uint32_t sl = (uint32_t)(((uint64_t)M * n) % M);   /* == 0 */
        uint64_t hi7;
        memcpy(&hi7, (const char*)&ch[sl].prev_hi + 56, 8);
        if(hi7 + ch[sl].cin != 0) return 0;
    }
    return 1;
}

/* A/B'd end-to-end: the staged emit ties M=5 and wins M=7 -- holding both
 * butterflies' y[] (28 zmm at M=7) spills more than the L1 staging costs,
 * which hides under the butterfly FMAs. Keep rot for future re-evaluation
 * (it wins if the butterfly's register footprint ever shrinks). */
#ifndef F16_PFA_EMIT_ROT
#define F16_PFA_EMIT_ROT 0
#endif
/* radix-3x2 emit: reads 6 streams (the e/o halves of each branch, both
 * already inverse-transformed at n/2), applies the inverse top r2 (conj
 * w1 twiddles) + inverse radix-3, and writes 6 natural fronts (low/high
 * half of each region), one carry chain each. Mirrors f16_pfa3x2_input;
 * the mem_ir22 sweep is gone. */
static inline int f16_pfa3x2_emit(uint64_t* rp, double* data, uint32_t n,
                                  const f16_plan* pl){
    uint8_t km[8]; f16_kmasks(km, 3u);
    const double* pe[3]; const double* po[3];
    uint64_t* rf[6];
    uint32_t fres[6];
    f16_chain ch[6];
    const size_t limbs_q = (size_t)n / 4u;       /* limbs per front */
    for(uint32_t b = 0; b < 3; ++b){
        pe[b] = data + 2u * (size_t)n * b;
        po[b] = pe[b] + n;
        rf[2 * b]     = rp + (size_t)b * (n / 2u);
        rf[2 * b + 1] = rf[2 * b] + limbs_q;
        fres[2 * b]     = (uint32_t)(((uint64_t)b * n) % 3u);
        fres[2 * b + 1] = (uint32_t)(((uint64_t)b * n + n / 2u) % 3u);
    }
    for(int f = 0; f < 6; ++f){ ch[f].prev_hi = _mm512_setzero_si512(); ch[f].cin = 0; }
    const double* tw = pl->tw22[__builtin_ctz(n)];
    _Alignas(64) double stage[6 * 2 * 16];
    for(uint32_t i = 0; i < n / 32u; ++i){
        if(i == n / 64u) tw = pl->tw22[__builtin_ctz(n)];
        const int ph2 = i >= n / 64u;
        for(int sub = 0; sub < 2; ++sub){
            fcv xlo[3], xhi[3], ylo[3], yhi[3];
            fcv w; w.re = _mm512_load_pd(tw + 32 * sub);
                   w.im = _mm512_load_pd(tw + 32 * sub + 8);
            if(ph2) w = f_j(w);
            for(uint32_t b = 0; b < 3; ++b){
                fcv ee = f_load(pe[b]); pe[b] += 16;
                fcv oo = f_load(po[b]); po[b] += 16;
                fcv v = f_mulc(oo, w);
                xlo[b] = f_add(ee, v);
                xhi[b] = f_sub(ee, v);
            }
            f16_pfa_bfly(xlo, ylo, 3u, 1);
            f16_pfa_bfly(xhi, yhi, 3u, 1);
            for(uint32_t b = 0; b < 3; ++b){
                uint32_t ph = (fres[2 * b] + (uint32_t)(8 * sub)) % 3u;
                fcv o = ylo[0];
                for(uint32_t j = 0; j < 3u; ++j){
                    uint32_t d = (j + 3u - ph) % 3u;
                    o.re = _mm512_mask_mov_pd(o.re, km[d], ylo[j].re);
                    o.im = _mm512_mask_mov_pd(o.im, km[d], ylo[j].im);
                }
                f_store(stage + 32u * (2u * b) + 16u * (uint32_t)sub, o);
                ph = (fres[2 * b + 1] + (uint32_t)(8 * sub)) % 3u;
                o = yhi[0];
                for(uint32_t j = 0; j < 3u; ++j){
                    uint32_t d = (j + 3u - ph) % 3u;
                    o.re = _mm512_mask_mov_pd(o.re, km[d], yhi[j].re);
                    o.im = _mm512_mask_mov_pd(o.im, km[d], yhi[j].im);
                }
                f_store(stage + 32u * (2u * b + 1u) + 16u * (uint32_t)sub, o);
            }
        }
        for(int f = 0; f < 6; ++f){
            f16_lohi q = f16_pack2(f_load(stage + 32u * (uint32_t)f),
                                   f_load(stage + 32u * (uint32_t)f + 16u));
            _mm512_storeu_si512((void*)rf[f], f16_chain_step(&ch[f], q));
            rf[f] += 8;
            fres[f] = (fres[f] + 16u) % 3u;
        }
        tw += 64;
    }
    /* fronts are contiguous in rp order; junction f -> f+1, top must vanish */
    for(int f = 0; f + 1 < 6; ++f){
        uint64_t hi7;
        memcpy(&hi7, (const char*)&ch[f].prev_hi + 56, 8);
        unsigned __int128 c = (unsigned __int128)hi7 + ch[f].cin;
        uint64_t* q = rp + (size_t)(f + 1) * limbs_q;
        uint64_t* qe = rp + 6u * limbs_q;
        while(c && q < qe){ c += *q; *q++ = (uint64_t)c; c >>= 64; }
        if(c) return 0;
    }
    {
        uint64_t hi7;
        memcpy(&hi7, (const char*)&ch[5].prev_hi + 56, 8);
        if(hi7 + ch[5].cin != 0) return 0;
    }
    return 1;
}

static inline int f16_pfa_emit(uint64_t* rp, double* data, uint32_t n,
                               uint32_t M, const f16_plan* pl){
    switch(M){
    case 3:  return F16_PFA3X2 ? f16_pfa3x2_emit(rp, data, n, pl)
                               : f16_pfa_emit_impl(rp, data, n, 3u, pl);
#if F16_PFA_EMIT_ROT
    case 5:  return f16_pfa_emit_rot(rp, data, n, 5u, pl->wt5, pl);
    default: return f16_pfa_emit_rot(rp, data, n, 7u, pl->wt7, pl);
#else
    case 5:  return f16_pfa_emit_tw(rp, data, n, 5u, pl->wt5, pl);
    default: return f16_pfa_emit_tw(rp, data, n, 7u, pl->wt7, pl);
#endif
    }
}

/* ------------------------------------------------------------------ */
/* public entry: rp[0..an+bn) = ap[0..an) * bp[0..bn). Returns 1 on    */
/* success, 0 if the size is outside the supported band.               */
/* ------------------------------------------------------------------ */
static _Thread_local f16_plan f16_tls;

/* smallest supported transform >= need: pow2 N or M*2^L (M = 3: branch >=
 * 256; M = 5, 7: branch >= 128). Ties prefer the smaller M / pow2.
 * PFA-7 is enabled: with the pointer-rotation I/O (constant vector-slot
 * indices, the residue relabel lives in GPR pointer renaming) it beats its
 * pow2 fallback at every branch size. Raise the gate if that ever flips. */
#ifndef F16_PFA7_MIN_BRANCH
#define F16_PFA7_MIN_BRANCH 128u
#endif
typedef struct { uint32_t nfull, branch, M; } f16_size;
static inline f16_size f16_choose(uint32_t need){
    uint32_t p2 = 512;
    while(p2 < need) p2 <<= 1;
    f16_size best = { p2, p2, 1 };
    static const uint32_t Ms[3] = { 3, 5, 7 };
    for(int i = 0; i < 3; ++i){
        uint32_t M = Ms[i];
        uint32_t br = M == 3 ? 256u : 128u;
        while(M * br < need) br <<= 1;
        if(M == 7u && br < F16_PFA7_MIN_BRANCH) continue;
        if(M * br < best.nfull && M * br <= (1u << 16)){
            best.nfull = M * br; best.branch = br; best.M = M;
        }
    }
    return best;
}

static inline int fft16_mul(uint64_t* rp,
                            const uint64_t* ap, ptrdiff_t an,
                            const uint64_t* bp, ptrdiff_t bn){
    if(an <= 0 || bn <= 0) return 0;
    uint64_t need = 2u * ((uint64_t)an + (uint64_t)bn);   /* complex points */
    if(need > (1u << 16)) return 0;                        /* U16 precision cap */
    f16_size fs = f16_choose((uint32_t)need);
    const uint32_t n = fs.branch, M = fs.M, nfull = fs.nfull;

    f16_plan* pl = &f16_tls;
    f16_plan_ensure(pl, n, nfull);

    memcpy(pl->pada, ap, (size_t)an * 8);
    memset(pl->pada + an, 0, ((size_t)(nfull / 2u) - (size_t)an) * 8);
    memcpy(pl->padb, bp, (size_t)bn * 8);
    memset(pl->padb + bn, 0, ((size_t)(nfull / 2u) - (size_t)bn) * 8);

    /* B first so A's spectrum is the cache-hot one when pointwise consumes
     * and overwrites it; leaves are fused into the pointwise pass. */
    if(M == 1){
        f16_fwd(pl->db, pl->padb, n, pl);
        f16_fwd(pl->da, pl->pada, n, pl);
        f16_pointwise_mul_ileaves(pl->da, pl->db, n, 1.0 / (double)n,
                                  f16_shape_of(n).leaf, pl);
        f16_inv_r8_only(pl->da, n, pl);
        if(!f16_inv_final_emit(pl->padr, pl->da, n, pl)) return 0;
    }else{
        f16_pfa_fwd(pl->db, pl->padb, n, M, pl);
        f16_pfa_fwd(pl->da, pl->pada, n, M, pl);
        const double sc = 1.0 / (double)nfull;
        const double* wrt = M == 3 ? F16_W3_RE : M == 5 ? F16_W5_RE : F16_W7_RE;
        const double* wit = M == 3 ? F16_W3_IM : M == 5 ? F16_W5_IM : F16_W7_IM;
        /* branch 0 self-pairs; branches (b, M-b) cross-pair with omega_M^b */
        uint32_t pwleaf = (F16_PFA3X2 && M == 3u)
                            ? f16_shape_of_full(n / 2u).leaf
                            : f16_shape_of(n).leaf;
        f16_pointwise_mul_ileaves(pl->da, pl->db, n, sc, pwleaf, pl);
        for(uint32_t b = 1; b <= M / 2u; ++b){
            uint32_t bp2 = M - b;
            f16_pointwise_cross_ileaves(
                pl->da + 2u * (size_t)n * b,   pl->da + 2u * (size_t)n * bp2,
                pl->db + 2u * (size_t)n * b,   pl->db + 2u * (size_t)n * bp2,
                n, sc, wrt[b], wit[b], pwleaf, pl);
        }
        if(F16_PFA3X2 && M == 3u){
            for(uint32_t h = 0; h < 6; ++h)
                f16_inv_r8_only_full(pl->da + (size_t)n * h, n / 2u, pl);
        }else{
            const double* tw = pl->tw22[__builtin_ctz(n)];
            for(uint32_t b = 0; b < M; ++b){
                double* d = pl->da + 2u * (size_t)n * b;
                f16_inv_r8_only(d, n, pl);
                f16_mem_ir22(d, n, tw);
            }
        }
        if(!f16_pfa_emit(pl->padr, pl->da, n, M, pl)) return 0;
    }
    memcpy(rp, pl->padr, (size_t)(an + bn) * 8);
    return 1;
}

#endif /* FFT16_H */
