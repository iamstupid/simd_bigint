#pragma once
// pq16.h - AVX-512 PQ FFT bigint multiply (clean rewrite of fft16.h).
// u64-limb I/O, u16-digit internals. Public entry:
//   pq16_mul_r(rp, ap, an, bp, bn, scratch*)  /  pq16_mul(...) [thread scratch]
// returns 1 on success, 0 if an+bn is outside the supported band (caller
// falls back to the toom route). Persistent twiddle tables live in a
// thread-local plan; ALL per-call workspace comes from scratch.h.
//
// ---- algorithm + optimization inventory (every choice was pinned by
//      microbench + full-mul A/B on the fft16.h ancestor; see bench/) ----
//
// codec    operands are consumed as u16 digits (4 per limb). N complex =
//          2*(an+bn) via the PQ right-angle trick: digits (2k, 2k+1) form
//          one complex point, the pointwise step evaluates the product at
//          partner pairs (k, N/2-k) sharing cross-terms (pq_eval_pair).
//          Scale 1/N is folded into the pointwise constants.
// layout   AoSoV tiles of 8 complex per zmm pair [re0..7 | im0..7], 128 B.
//          Forward is DIF, root e^{+2*pi*i/N}, natural in / bit-reversed out.
// stages   fwd: fused u16-decode + radix-2^2 unroll-2 at len N (operands are
//          read straight from the caller's buffers with fault-suppressed
//          masked tails -- no zero-padded copy) -> radix-2^3 passes -> leaf.
//          leaf 32x2/64/128 by lg(n) parity: across-stage + 8x8 transpose +
//          lane-parallel DFT8. Leaves skip the transpose-back ("pi layout"):
//          stored tile q lane t = canonical-BR position t*8+q; the pointwise
//          table is built in pi order and the inverse leaves consume pi
//          directly, so both transposes cancel out of the transform.
//          pointwise is fused with the inverse leaves (mirror group pairs
//          multiplied then immediately inverse-leafed cache-hot); the final
//          inverse radix-2^2 at len N is fused with the u16 emit and a
//          base-2^64 SWAR carry chain per quarter-front + junction fixups,
//          so no canonical coefficient array is ever materialized.
// tables   tw22[lg]/tw8[lg] keyed by stage length, shared across transform
//          sizes; PQ g-table is prefix-stable in canonical-BR index, so the
//          largest build serves every smaller n as a prefix. Stage lengths
//          >= 2^PQ16_TWC_MIN_LG store COMPACT tables (2x smaller; squares
//          derived in-kernel at <= 1 ulp) -- a few cmuls per group against
//          megabytes of table stream at the DRAM-bound sizes. (A 4x scheme
//          deriving W4 = (W1^2)^2 at ~3 ulp fails the adversarial probe at
//          the centered cap by one rounding flip -- do not revisit blindly.)
// blocking r8 stages run through a recursive 3-tier ladder (~L3/L2/L1):
//          stages longer than the tier run as full-span sweeps, everything
//          below recurses per chunk, so each chunk finishes its remaining
//          stages + leaves (and on the inverse side starts them) while
//          cache-resident.
// PFA      Good-Thomas M in {3,5,7} x pow2 branch fills the octave between
//          powers of two (worst-case grid padding 14% instead of 100%).
//          M=3 input decodes 3 stripes with constant lane-residue masked
//          merges; M=5/7 use the twiddle-fixup form (y_b(l) = sum_s w^{b r_s}
//          w^{b l} t_s(l)) with stripe pointers held in residue order so the
//          per-iter relabel is pure GPR renaming and all vector slots are
//          compile-time constants. Pointwise pairs branch b with branch M-b
//          (antipodal mirror) rotating the shared g-table by omega_M^b.
//          M=5/7 emit pre-rotates by conj(w^{b l}) and stages both sub-tiles
//          in an L1 buffer (the register-resident variant spills at M=7).
//          Losing variants removed: radix-3x2 fused input/emit (-6..9%),
//          register-rotating emit (spills), see fft16.h history.
// centered transforms above 2^17 complex run on centered digits d - 2^15
//          (4x precision headroom, probed to 2^19 complex = 131072x131072
//          limbs with all-max and sparse adversarial operands). Centering is
//          a 1-op XOR with 0x8000 folded into a staging copy; the decode
//          sign-extends via a byte-shuffle + arithmetic shift. The signed
//          coefficient c^ is corrected as c = c^ + (S<<15) - (C<<30) with S
//          a running digit-window sum (segmented: rising/middle/falling
//          windows need half the loads and a closed-form C) and C the window
//          overlap length. The centered band emits FLAT: one natural-order
//          pass (mem_ir22 / pfa_natural) then a single sequential emit
//          stream + one carry chain -- the fused 4-front emit's ~25 streams
//          go latency-bound right where the band begins. Operand + result
//          staging double as layout normalizers: reading caller buffers
//          in-place in this BW-bound band cost up to 2x depending on
//          allocation layout. PFA-7's direct-form butterfly fails first
//          UNcentered (7*2^14 all-max); centered headroom re-admits it.
// gates    pow2/PFA-3/5 hold to 2^17 uncentered; PFA-7 capped at 2^16
//          uncentered; everything to 2^19 centered. All empirical.
#include "types.h"
#include "scratch.h"
#include <math.h>
#include <stdlib.h>

// double-vector companions to the types.h macro set (aligned forms; every
// tile is 64-byte aligned by construction)
#define load_dvec(p)     vec_fn(load_pd)((const void *)(p))
#define store_dvec(p, v) vec_fn(store_pd)((void *)(p), (v))
#define fmadd(a, b, c)   vec_fn(fmadd_pd)((a), (b), (c))
#define fmsub(a, b, c)   vec_fn(fmsub_pd)((a), (b), (c))
#define permd(idx, a)    vec_fn(permutexvar_pd)((idx), (a))

// ------------------------------------------------------------------
// complex tile: 8 complex per zmm pair
// ------------------------------------------------------------------
typedef struct { _dvec re, im; } qcv;

static inline qcv q_ld(const double *p){
    qcv r = { load_dvec(p), load_dvec(p + 8) };
    return r;
}
static inline void q_st(double *p, qcv x){
    store_dvec(p, x.re); store_dvec(p + 8, x.im);
}
static inline qcv q_add(qcv a, qcv b){
    qcv r = { add(a.re, b.re), add(a.im, b.im) };
    return r;
}
static inline qcv q_sub(qcv a, qcv b){
    qcv r = { sub(a.re, b.re), sub(a.im, b.im) };
    return r;
}
static inline qcv q_mul(qcv a, qcv w){            // a * w
    qcv r = { fmsub(a.re, w.re, mul(a.im, w.im)),
              fmadd(a.re, w.im, mul(a.im, w.re)) };
    return r;
}
static inline qcv q_mulc(qcv a, qcv w){           // a * conj(w)
    qcv r = { fmadd(a.im, w.im, mul(a.re, w.re)),
              fmsub(a.im, w.re, mul(a.re, w.im)) };
    return r;
}
static inline qcv q_j(qcv a){                     // +i * a
    qcv r = { sub(dzero(), a.im), a.re };
    return r;
}
static inline qcv q_mj(qcv a){                    // -i * a
    qcv r = { a.im, sub(dzero(), a.re) };
    return r;
}
static inline qcv q_rev(qcv a){                   // lane reverse
    const _vec ix = setr_64(7, 6, 5, 4, 3, 2, 1, 0);
    qcv r = { permd(ix, a.re), permd(ix, a.im) };
    return r;
}

#define PQ16_C8 0.70710678118654752440  /* cos(pi/4) */

// rot8[t](a) = a * e^{+2*pi*i*t/8}, t = 1, 3 (t = 0 id, t = 2 = q_j)
static inline qcv q_rot81(qcv a){
    const _dvec c = set1_d(PQ16_C8);
    qcv r = { mul(c, sub(a.re, a.im)), mul(c, add(a.re, a.im)) };
    return r;
}
static inline qcv q_rot83(qcv a){
    const _dvec c = set1_d(PQ16_C8);
    qcv r = { sub(dzero(), mul(c, add(a.re, a.im))), mul(c, sub(a.re, a.im)) };
    return r;
}
// conjugate rotations for the inverse
static inline qcv q_irot81(qcv a){
    const _dvec c = set1_d(PQ16_C8);
    qcv r = { mul(c, add(a.re, a.im)), mul(c, sub(a.im, a.re)) };
    return r;
}
static inline qcv q_irot83(qcv a){
    const _dvec c = set1_d(PQ16_C8);
    qcv r = { mul(c, sub(a.im, a.re)), sub(dzero(), mul(c, add(a.re, a.im))) };
    return r;
}

// ------------------------------------------------------------------
// radix-2^2 quad butterflies -- THE shared primitive: the fused input,
// in-memory r22 passes, r8 stages, all leaves and the fused emit are
// instances of these two on different twiddle sets.
// DIF forward: v = { x0+x2+(x1+x3),  ((x0+x2)-(x1+x3))w2,
//                    (x0-x2)w1+(x1-x3)jw1,  ((x0-x2)w1-(x1-x3)jw1)w2 }
// ------------------------------------------------------------------
static inline void q_bf4(qcv v[4], qcv w1, qcv w2){
    qcv b0 = q_add(v[0], v[2]), b2 = q_mul(q_sub(v[0], v[2]), w1);
    qcv b1 = q_add(v[1], v[3]), b3 = q_mul(q_sub(v[1], v[3]), q_j(w1));
    v[0] = q_add(b0, b1); v[1] = q_mul(q_sub(b0, b1), w2);
    v[2] = q_add(b2, b3); v[3] = q_mul(q_sub(b2, b3), w2);
}
static inline void q_ibf4(qcv v[4], qcv w1, qcv w2){
    qcv s  = q_mulc(v[1], w2);
    qcv b0 = q_add(v[0], s), b1 = q_sub(v[0], s);
    s      = q_mulc(v[3], w2);
    qcv b2 = q_add(v[2], s), b3 = q_sub(v[2], s);
    s = q_mulc(b2, w1);
    v[0] = q_add(b0, s); v[2] = q_sub(b0, s);
    s = q_mulc(b3, q_j(w1));
    v[1] = q_add(b1, s); v[3] = q_sub(b1, s);
}

// radix-2^3 on 8 tiles: level-1 split (W1/W1o rotations) + two quads.
// Outputs: v[0..3] = even half, v[4..7] = odd half (DIF slot order).
static inline void q_bf8(qcv v[8], qcv W1, qcv W1o, qcv W2, qcv W4){
    qcv e[4], o[4];
    e[0] = q_add(v[0], v[4]); e[1] = q_add(v[1], v[5]);
    e[2] = q_add(v[2], v[6]); e[3] = q_add(v[3], v[7]);
    o[0] = q_mul(q_sub(v[0], v[4]), W1);
    o[1] = q_mul(q_sub(v[1], v[5]), W1o);
    o[2] = q_j(q_mul(q_sub(v[2], v[6]), W1));
    o[3] = q_j(q_mul(q_sub(v[3], v[7]), W1o));
    q_bf4(e, W2, W4);
    q_bf4(o, W2, W4);
    for(int i = 0; i < 4; ++i){ v[i] = e[i]; v[4 + i] = o[i]; }
}
static inline void q_ibf8(qcv v[8], qcv W1, qcv W1o, qcv W2, qcv W4){
    q_ibf4(v, W2, W4);
    q_ibf4(v + 4, W2, W4);
    qcv o0 = q_mulc(v[4], W1);
    qcv o1 = q_mulc(v[5], W1o);
    qcv o2 = q_mulc(q_mj(v[6]), W1);
    qcv o3 = q_mulc(q_mj(v[7]), W1o);
    qcv e0 = v[0], e1 = v[1], e2 = v[2], e3 = v[3];
    v[0] = q_add(e0, o0); v[4] = q_sub(e0, o0);
    v[1] = q_add(e1, o1); v[5] = q_sub(e1, o1);
    v[2] = q_add(e2, o2); v[6] = q_sub(e2, o2);
    v[3] = q_add(e3, o3); v[7] = q_sub(e3, o3);
}

// ------------------------------------------------------------------
// 8x8 double transpose (24 shuffles)
// ------------------------------------------------------------------
static inline void q_tr8(_dvec r[8]){
    _dvec t0 = vec_fn(unpacklo_pd)(r[0], r[1]);
    _dvec t1 = vec_fn(unpackhi_pd)(r[0], r[1]);
    _dvec t2 = vec_fn(unpacklo_pd)(r[2], r[3]);
    _dvec t3 = vec_fn(unpackhi_pd)(r[2], r[3]);
    _dvec t4 = vec_fn(unpacklo_pd)(r[4], r[5]);
    _dvec t5 = vec_fn(unpackhi_pd)(r[4], r[5]);
    _dvec t6 = vec_fn(unpacklo_pd)(r[6], r[7]);
    _dvec t7 = vec_fn(unpackhi_pd)(r[6], r[7]);

    _dvec v0 = vec_fn(shuffle_f64x2)(t0, t2, 0x44);  // cols 0,2 lo
    _dvec v1 = vec_fn(shuffle_f64x2)(t4, t6, 0x44);
    _dvec v2 = vec_fn(shuffle_f64x2)(t0, t2, 0xEE);  // cols 4,6 hi
    _dvec v3 = vec_fn(shuffle_f64x2)(t4, t6, 0xEE);
    _dvec v4 = vec_fn(shuffle_f64x2)(t1, t3, 0x44);  // cols 1,3
    _dvec v5 = vec_fn(shuffle_f64x2)(t5, t7, 0x44);
    _dvec v6 = vec_fn(shuffle_f64x2)(t1, t3, 0xEE);  // cols 5,7
    _dvec v7 = vec_fn(shuffle_f64x2)(t5, t7, 0xEE);

    r[0] = vec_fn(shuffle_f64x2)(v0, v1, 0x88);
    r[2] = vec_fn(shuffle_f64x2)(v0, v1, 0xDD);
    r[4] = vec_fn(shuffle_f64x2)(v2, v3, 0x88);
    r[6] = vec_fn(shuffle_f64x2)(v2, v3, 0xDD);
    r[1] = vec_fn(shuffle_f64x2)(v4, v5, 0x88);
    r[3] = vec_fn(shuffle_f64x2)(v4, v5, 0xDD);
    r[5] = vec_fn(shuffle_f64x2)(v6, v7, 0x88);
    r[7] = vec_fn(shuffle_f64x2)(v6, v7, 0xDD);
}
static inline void q_tr8cv(qcv v[8]){
    _dvec re[8], im[8];
    for(int i = 0; i < 8; ++i){ re[i] = v[i].re; im[i] = v[i].im; }
    q_tr8(re); q_tr8(im);
    for(int i = 0; i < 8; ++i){ v[i].re = re[i]; v[i].im = im[i]; }
}

// ------------------------------------------------------------------
// lane-parallel 8-point DFT across 8 vectors (DIF, e^{+} root, outputs
// in sequential = bit-reversed slot order). Unscaled inverse.
// ------------------------------------------------------------------
static inline void q_dft8(qcv v[8]){
    qcv e0 = q_add(v[0], v[4]), o0 = q_sub(v[0], v[4]);
    qcv e1 = q_add(v[1], v[5]), o1 = q_rot81(q_sub(v[1], v[5]));
    qcv e2 = q_add(v[2], v[6]), o2 = q_j(q_sub(v[2], v[6]));
    qcv e3 = q_add(v[3], v[7]), o3 = q_rot83(q_sub(v[3], v[7]));

    qcv ee0 = q_add(e0, e2), eo0 = q_sub(e0, e2);
    qcv ee1 = q_add(e1, e3), eo1 = q_j(q_sub(e1, e3));
    qcv oe0 = q_add(o0, o2), oo0 = q_sub(o0, o2);
    qcv oe1 = q_add(o1, o3), oo1 = q_j(q_sub(o1, o3));

    v[0] = q_add(ee0, ee1); v[1] = q_sub(ee0, ee1);
    v[2] = q_add(eo0, eo1); v[3] = q_sub(eo0, eo1);
    v[4] = q_add(oe0, oe1); v[5] = q_sub(oe0, oe1);
    v[6] = q_add(oo0, oo1); v[7] = q_sub(oo0, oo1);
}
static inline void q_idft8(qcv v[8]){
    qcv ee0 = q_add(v[0], v[1]), ee1 = q_sub(v[0], v[1]);
    qcv eo0 = q_add(v[2], v[3]), eo1 = q_sub(v[2], v[3]);
    qcv oe0 = q_add(v[4], v[5]), oe1 = q_sub(v[4], v[5]);
    qcv oo0 = q_add(v[6], v[7]), oo1 = q_sub(v[6], v[7]);

    qcv e0 = q_add(ee0, eo0), e2 = q_sub(ee0, eo0);
    qcv t  = q_mj(eo1);
    qcv e1 = q_add(ee1, t),   e3 = q_sub(ee1, t);
    qcv o0 = q_add(oe0, oo0), o2 = q_sub(oe0, oo0);
    t      = q_mj(oo1);
    qcv o1 = q_add(oe1, t),   o3 = q_sub(oe1, t);

    o1 = q_irot81(o1); o2 = q_mj(o2); o3 = q_irot83(o3);
    v[0] = q_add(e0, o0); v[4] = q_sub(e0, o0);
    v[1] = q_add(e1, o1); v[5] = q_sub(e1, o1);
    v[2] = q_add(e2, o2); v[6] = q_sub(e2, o2);
    v[3] = q_add(e3, o3); v[7] = q_sub(e3, o3);
}

// ------------------------------------------------------------------
// plan: persistent twiddle tables + decode shuffles (thread-local,
// grown once, shared across transform sizes). NO per-call workspace
// lives here -- that comes from scratch.
// ------------------------------------------------------------------
#define PQ16_MAX_R8 4
#define PQ16_MAX_LG 21
#ifndef PQ16_FORCE_CENTERED
#define PQ16_FORCE_CENTERED 0   // test hook: centered codec at every size
#endif

// per-transform stage shape, derived from lg(n) alone
typedef struct {
    uint32_t leaf;               // 32, 64 or 128
    uint32_t cnt;                // number of r8 ancestor stages
    uint32_t len[PQ16_MAX_R8];   // their lengths, descending
} pq16_shape;

static inline pq16_shape pq16_shape_of(uint32_t n){
    unsigned lgn = (unsigned)__builtin_ctz(n);
    pq16_shape s;
    uint32_t leaf_lg = 5u + ((lgn - 7u) % 3u);    // lgn >= 9 always
    s.leaf = 1u << leaf_lg;
    s.cnt  = (lgn - 2u - leaf_lg) / 3u;
    uint32_t len = n / 4u;
    for(uint32_t i = 0; i < s.cnt; ++i, len /= 8u) s.len[i] = len;
    return s;
}

// Twiddle tables are shared across transform sizes and grown once:
//  - tw22[lg]: len-2^lg r22 unroll-2 table (input + final-emit stage of a
//    size-2^lg transform). Depends on len only.
//  - tw8[lg]:  len-2^lg r8 stage table. Depends on len only.
//  - pq: pi-ordered pointwise g-table. Prefix-stable in the canonical-BR
//    index (bitrev(k,b)/2^b is width-independent), so the largest build
//    serves every smaller n as a prefix.
typedef struct {
    uint32_t pq_n;               // pow2 branch size the PQ table covers
    double  *tw22[PQ16_MAX_LG];
    double  *tw8[PQ16_MAX_LG];
    double  *pq;                 // compact PQ g-table: per 8-tile group,
                                 // 32 doubles [re x16 | im x16]
    // leaf constants (transform-size independent)
    _Alignas(64) double l32w[2 * 16];   // w1, w2 at len 32
    _Alignas(64) double l64w[4 * 16];   // W1, W1*w8, W2, W4 at len 64
    _Alignas(64) double l128w[8 * 16];  // {W1, W2} at len 128, t = 0..3
    _Alignas(64) double gh0[16];        // head g-values: g(p>>2), p = 0..7
    _Alignas(64) double wt5[5 * 16];    // PFA lane fixup w_5^{b*(l%5)}
    _Alignas(64) double wt7[7 * 16];    // PFA lane fixup w_7^{b*(l%7)}
    _vec dec_idx[4];             // u16 digit decode shuffles
    _vec dec_idxc[4];            // centered: digit bytes at dword 2-3
    int  leaf_init;
} pq16_plan;

static inline void *pq16_alloc(size_t bytes){
    void *p = NULL;
    if(posix_memalign(&p, 128, bytes)) abort();
    return p;
}

static inline uint32_t pq16_bitrev(uint32_t x, unsigned bits){
    x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
    x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
    x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);
    x = ((x & 0x00FF00FFu) << 8) | ((x >> 8) & 0x00FF00FFu);
    x = (x << 16) | (x >> 16);
    return bits ? (x >> (32u - bits)) : 0u;
}

#define PQ16_PI 3.14159265358979323846
// root e^{+2*pi*i*k/n}
static inline void pq16_root(double *wr, double *wi, uint64_t k, uint64_t n){
    double a = (2.0 * PQ16_PI * (double)(k % n)) / (double)n;
    *wr = cos(a); *wi = sin(a);
}
// one twiddle cv: lanes l = root(scale*(j+l), len) at dst[0..7 | 8..15]
static inline void pq16_twcv(double *dst, uint64_t j, uint64_t scale, uint64_t len){
    for(int l = 0; l < 8; ++l)
        pq16_root(dst + l, dst + 8 + l, (j + (uint64_t)l) * scale, len);
}
// same multiplied by e^{+2*pi*i/8}
static inline void pq16_twcv8(double *dst, uint64_t j, uint64_t scale, uint64_t len){
    for(int l = 0; l < 8; ++l){
        double wr, wi;
        pq16_root(&wr, &wi, (j + (uint64_t)l) * scale, len);
        dst[l]     = PQ16_C8 * (wr - wi);
        dst[8 + l] = PQ16_C8 * (wr + wi);
    }
}

static inline void pq16_leaf_init(pq16_plan *pl){
    if(pl->leaf_init) return;
    pq16_twcv (pl->l32w + 0,  0, 1, 32);
    pq16_twcv (pl->l32w + 16, 0, 2, 32);
    pq16_twcv (pl->l64w + 0,  0, 1, 64);
    pq16_twcv8(pl->l64w + 16, 0, 1, 64);
    pq16_twcv (pl->l64w + 32, 0, 2, 64);
    pq16_twcv (pl->l64w + 48, 0, 4, 64);
    for(int t = 0; t < 4; ++t){
        pq16_twcv(pl->l128w + 32 * t,      (uint64_t)(8 * t), 1, 128);
        pq16_twcv(pl->l128w + 32 * t + 16, (uint64_t)(8 * t), 2, 128);
    }
    // PFA lane-twiddle fixups: wtM[b] lane l = e^{-2*pi*i*b*(l%M)/M}
    for(int b = 0; b < 5; ++b)
        for(int l = 0; l < 8; ++l){
            double th = -2.0 * PQ16_PI * (double)(b * (l % 5)) / 5.0;
            pl->wt5[16 * b + l] = cos(th); pl->wt5[16 * b + 8 + l] = sin(th);
        }
    for(int b = 0; b < 7; ++b)
        for(int l = 0; l < 8; ++l){
            double th = -2.0 * PQ16_PI * (double)(b * (l % 7)) / 7.0;
            pl->wt7[16 * b + l] = cos(th); pl->wt7[16 * b + 8 + l] = sin(th);
        }
    // digit decode: parity x tile -> 8 dword slots of (2-byte digit, 2 zero)
    for(int v = 0; v < 4; ++v){
        int parity = v & 1, tile = v >> 1;
        char idx[64];
        for(int l = 0; l < 8; ++l){
            int digit = tile * 16 + 2 * l + parity;
            idx[l * 4 + 0] = (char)(digit * 2);
            idx[l * 4 + 1] = (char)(digit * 2 + 1);
            idx[l * 4 + 2] = (char)0x40;          // zero source
            idx[l * 4 + 3] = (char)0x40;
        }
        for(int i = 32; i < 64; ++i) idx[i] = (char)0x40;
        memcpy(&pl->dec_idx[v], idx, 64);
        // centered: digit bytes at dword bytes 2-3 (sign-extend via srai)
        char idc[64];
        for(int i = 0; i < 64; ++i) idc[i] = (char)0x40;
        for(int l = 0; l < 8; ++l){
            int digit = (v >> 1) * 16 + 2 * l + (v & 1);
            idc[l * 4 + 2] = (char)(digit * 2);
            idc[l * 4 + 3] = (char)(digit * 2 + 1);
        }
        memcpy(&pl->dec_idxc[v], idc, 64);
    }
    pl->leaf_init = 1;
}

// Tables for stage lengths >= 2^PQ16_TWC_MIN_LG are stored COMPACT (2x
// smaller); the missing entries are derived in-kernel at <= 1 ulp (one
// square / one exact-constant rot8). The affected lengths sit inside the
// centered band's 4x precision headroom (adversarially re-probed).
#ifndef PQ16_TWC_MIN_LG
#define PQ16_TWC_MIN_LG 16
#endif
// len-2^lg r22 unroll-2 table: per iter {w1a, w2a, w1b, w2b}, or compact
// {w1a, w1b} (squares derived)
static inline void pq16_build_tw22(pq16_plan *pl, unsigned lg){
    if(pl->tw22[lg]) return;
    uint32_t len = 1u << lg;
    if(lg >= PQ16_TWC_MIN_LG){
        double *tab = (double *)pq16_alloc((size_t)len * 4);
        for(uint32_t i = 0; i < len / 64u; ++i){
            pq16_twcv(tab + 32u * i,       16u * i,      1, len);
            pq16_twcv(tab + 32u * i + 16u, 16u * i + 8u, 1, len);
        }
        pl->tw22[lg] = tab;
        return;
    }
    double *tab = (double *)pq16_alloc((size_t)len * 8);
    for(uint32_t i = 0; i < len / 64u; ++i){
        double  *d = tab + 64u * i;
        uint64_t j = 16u * i;
        pq16_twcv(d + 0,  j,     1, len);
        pq16_twcv(d + 16, j,     2, len);
        pq16_twcv(d + 32, j + 8, 1, len);
        pq16_twcv(d + 48, j + 8, 2, len);
    }
    pl->tw22[lg] = tab;
}
// len-2^lg r8 stage table: per iter {W1, W1*w8, W2, W4}, or compact
// {W1, W2} (W1o = rot8(W1), W4 = W2^2 derived)
static inline void pq16_build_tw8(pq16_plan *pl, unsigned lg){
    if(pl->tw8[lg]) return;
    uint32_t len = 1u << lg;
    if(lg >= PQ16_TWC_MIN_LG){
        double *tab = (double *)pq16_alloc((size_t)len * 4);
        for(uint32_t i = 0; i < len / 64u; ++i){
            pq16_twcv(tab + 32u * i,       8u * i, 1, len);
            pq16_twcv(tab + 32u * i + 16u, 8u * i, 2, len);
        }
        pl->tw8[lg] = tab;
        return;
    }
    double *tab = (double *)pq16_alloc((size_t)len * 8);
    for(uint32_t i = 0; i < len / 64u; ++i){
        double  *d = tab + 64u * i;
        uint64_t j = 8u * i;
        pq16_twcv (d + 0,  j, 1, len);
        pq16_twcv8(d + 16, j, 1, len);
        pq16_twcv (d + 32, j, 2, len);
        pq16_twcv (d + 48, j, 4, len);
    }
    pl->tw8[lg] = tab;
}
// stage-length predicate + per-iter table stride
static inline int    pq16_twc(uint32_t len){ return (unsigned)__builtin_ctz(len) >= PQ16_TWC_MIN_LG; }
static inline size_t pq16_tw_step(int twc){ return twc ? 32 : 64; }

// the 4 r22 twiddles for one unroll-2 iter
static inline void pq16_tw22_get(const double *twp, int twc,
                                 qcv *w1a, qcv *w2a, qcv *w1b, qcv *w2b){
    if(!twc){
        *w1a = q_ld(twp);      *w2a = q_ld(twp + 16);
        *w1b = q_ld(twp + 32); *w2b = q_ld(twp + 48);
        return;
    }
    *w1a = q_ld(twp);
    *w1b = q_ld(twp + 16);
    *w2a = q_mul(*w1a, *w1a);
    *w2b = q_mul(*w1b, *w1b);
}
// the 4 r8-stage twiddles
static inline void pq16_tw8_get(const double *twp, int twc,
                                qcv *W1, qcv *W1o, qcv *W2, qcv *W4){
    if(!twc){
        *W1 = q_ld(twp);      *W1o = q_ld(twp + 16);
        *W2 = q_ld(twp + 32); *W4  = q_ld(twp + 48);
        return;
    }
    *W1 = q_ld(twp);
    *W2 = q_ld(twp + 16);
    *W1o = q_rot81(*W1);
    *W4  = q_mul(*W2, *W2);
}

// grow every table the pow2 branch size needs (PFA shares them: the
// branch transform is plain pow2; the PQ table is keyed by branch too)
static inline void pq16_plan_ensure(pq16_plan *pl, uint32_t branch){
    pq16_leaf_init(pl);
    unsigned lgn = (unsigned)__builtin_ctz(branch);
    if(branch > pl->pq_n){
        // compact PQ table: group g covers canonical positions [g*64,
        // g*64+64) whose quarter-indices are g*16 + [0,16); store those 16
        // g = root(bitrev(k)/branch) values per group. Width-independent
        // fractions: the largest build serves every smaller n as a prefix.
        free(pl->pq);
        pl->pq = (double *)pq16_alloc((size_t)branch * 4);
        for(uint32_t g = 0; g < branch / 64u; ++g)
            for(uint32_t k = 0; k < 16; ++k)
                pq16_root(pl->pq + 32u * g + k, pl->pq + 32u * g + 16u + k,
                          pq16_bitrev(g * 16u + k, lgn - 2u), branch);
        for(uint32_t p = 0; p < 8; ++p)
            pq16_root(pl->gh0 + p, pl->gh0 + 8 + p,
                      pq16_bitrev(p >> 2, lgn - 2u), branch);
        pl->pq_n = branch;
    }
    pq16_build_tw22(pl, lgn);
    pq16_shape sh = pq16_shape_of(branch);
    for(uint32_t s = 0; s < sh.cnt; ++s)
        pq16_build_tw8(pl, (unsigned)__builtin_ctz(sh.len[s]));
}

// ------------------------------------------------------------------
// operand loaders + u16 digit decode. Operands are read straight from
// their buffers with fault-suppressed masked tails (no padded copy);
// rem = valid limbs remaining at p.
// ------------------------------------------------------------------
static inline _vec q_raw8(const uint64_t *p, int64_t rem){
    if(rem >= 8) return load_vec(p);
    return load_vec(p, rem <= 0 ? 0 : (0xFFu >> (8 - rem)));
}
static inline __mmask8 q_st_mask(int64_t rem){
    return k8(rem >= 8 ? 0xFF : (rem <= 0 ? 0 : (0xFFu >> (8 - rem))));
}

// 8 digits of one zmm -> 8 doubles. Uncentered: digit at dword bytes 0-1,
// zero-extended. Centered: PRE-XORED digit at dword bytes 2-3; the
// arithmetic shift sign-extends, giving d - 2^15 in [-2^15, 2^15).
static inline _dvec q_dec1(_vec raw, _vec idx){
    _vec d = vec_fn(permutex2var_epi8)(raw, idx, zero());
    return vec_fn(cvtepu32_pd)(vec_fn(castsi512_si256)(d));
}
static inline _dvec q_dec1c(_vec raw, _vec idx){
    _vec d = vec_fn(permutex2var_epi8)(raw, idx, zero());
    return vec_fn(cvtepi32_pd)(vec_fn(castsi512_si256)(vec_fn(srai_epi32)(d, 16)));
}
// decode one zmm of limbs (16 complex) into 2 tiles
static inline void q_dec2(qcv *t0, qcv *t1, _vec raw, const _vec *IDX, int cen){
    if(cen){
        t0->re = q_dec1c(raw, IDX[0]); t0->im = q_dec1c(raw, IDX[1]);
        t1->re = q_dec1c(raw, IDX[2]); t1->im = q_dec1c(raw, IDX[3]);
    }else{
        t0->re = q_dec1(raw, IDX[0]); t0->im = q_dec1(raw, IDX[1]);
        t1->re = q_dec1(raw, IDX[2]); t1->im = q_dec1(raw, IDX[3]);
    }
}

// ------------------------------------------------------------------
// forward input stage: fused u16-decode + r22 unroll-2 at len n
// ------------------------------------------------------------------
static inline void pq16_input_stage(double *data, const uint64_t *src,
                                    int64_t cnt, uint32_t n,
                                    const pq16_plan *pl, int cen){
    const uint32_t l = n / 4u;                       // complex per quarter
    double *p[4];
    const uint64_t *s[4];
    int64_t r[4];
    for(uint32_t q = 0; q < 4; ++q){
        p[q] = data + 2u * (size_t)l * q;
        s[q] = src + (size_t)q * (l / 2u);           // limbs: l cx = l/2
        r[q] = cnt - (int64_t)q * (int64_t)(l / 2u);
    }
    const double *twp = pl->tw22[__builtin_ctz(n)];
    const int twc = pq16_twc(n);
    const _vec *IDX = cen ? pl->dec_idxc : pl->dec_idx;
    for(uint32_t i = 0; i < l / 16u; ++i){
        qcv a[4], b[4];                              // sub-tile 0 / 1
        for(int q = 0; q < 4; ++q){
            q_dec2(&a[q], &b[q], q_raw8(s[q], r[q]), IDX, cen);
            s[q] += 8; r[q] -= 8;
        }
        qcv w1a, w2a, w1b, w2b;
        pq16_tw22_get(twp, twc, &w1a, &w2a, &w1b, &w2b);
        twp += pq16_tw_step(twc);
        q_bf4(a, w1a, w2a);
        q_bf4(b, w1b, w2b);
        for(int q = 0; q < 4; ++q){
            q_st(p[q], a[q]);
            q_st(p[q] + 16, b[q]);
            p[q] += 32;
        }
    }
}

// ------------------------------------------------------------------
// in-memory r22 unroll-2 stage at len n (PFA branch first stage / the
// centered band's separated final inverse)
// ------------------------------------------------------------------
static inline void pq16_mem_r22(double *data, uint32_t n, const double *tw){
    const uint32_t l = n / 4u;
    double *p[4];
    for(int s = 0; s < 4; ++s) p[s] = data + 2u * (size_t)l * (uint32_t)s;
    const int twc = pq16_twc(n);
    for(uint32_t i = 0; i < l / 16u; ++i){
        qcv w1a, w2a, w1b, w2b;
        pq16_tw22_get(tw, twc, &w1a, &w2a, &w1b, &w2b);
        tw += pq16_tw_step(twc);
        for(int u = 0; u < 2; ++u){
            qcv v[4];
            for(int s = 0; s < 4; ++s) v[s] = q_ld(p[s] + 16 * u);
            q_bf4(v, u ? w1b : w1a, u ? w2b : w2a);
            for(int s = 0; s < 4; ++s) q_st(p[s] + 16 * u, v[s]);
        }
        for(int s = 0; s < 4; ++s) p[s] += 32;
    }
}
static inline void pq16_mem_ir22(double *data, uint32_t n, const double *tw){
    const uint32_t l = n / 4u;
    double *p[4];
    for(int s = 0; s < 4; ++s) p[s] = data + 2u * (size_t)l * (uint32_t)s;
    const int twc = pq16_twc(n);
    for(uint32_t i = 0; i < l / 16u; ++i){
        qcv w1a, w2a, w1b, w2b;
        pq16_tw22_get(tw, twc, &w1a, &w2a, &w1b, &w2b);
        tw += pq16_tw_step(twc);
        for(int u = 0; u < 2; ++u){
            qcv v[4];
            for(int s = 0; s < 4; ++s) v[s] = q_ld(p[s] + 16 * u);
            q_ibf4(v, u ? w1b : w1a, u ? w2b : w2a);
            for(int s = 0; s < 4; ++s) q_st(p[s] + 16 * u, v[s]);
        }
        for(int s = 0; s < 4; ++s) p[s] += 32;
    }
}

// ------------------------------------------------------------------
// PFA (Good-Thomas) radix-M butterflies, M in {3, 5, 7}.
// omega_M^k = exp(-2*pi*i*k/M) tables (reference int_fft values).
// ------------------------------------------------------------------
static const double PQ16_W3_RE[3] = { 1.0, -0.5, -0.5 };
static const double PQ16_W3_IM[3] = { 0.0, -0.86602540378443865, 0.86602540378443865 };
static const double PQ16_W5_RE[5] = { 1.0, 0.30901699437494742, -0.80901699437494742,
                                      -0.80901699437494742, 0.30901699437494742 };
static const double PQ16_W5_IM[5] = { 0.0, -0.95105651629515357, -0.58778525229247313,
                                      0.58778525229247313, 0.95105651629515357 };
static const double PQ16_W7_RE[7] = { 1.0, 0.62348980185873353, -0.2225209339563144,
                                      -0.90096886790241913, -0.90096886790241913,
                                      -0.2225209339563144, 0.62348980185873353 };
static const double PQ16_W7_IM[7] = { 0.0, -0.78183148246802981, -0.97492791218182361,
                                      -0.43388373911755812, 0.43388373911755812,
                                      0.97492791218182361, 0.78183148246802981 };

// radix-3 Winograd (e^- DFT convention)
static inline void q_pfa3(const qcv *x, qcv *y, int inv){
    const _dvec ch = set1_d(-0.5);
    const _dvec sh = set1_d(0.86602540378443865);
    qcv s = q_add(x[1], x[2]), d = q_sub(x[1], x[2]);
    y[0] = q_add(x[0], s);
    qcv u = { fmadd(ch, s.re, x[0].re), fmadd(ch, s.im, x[0].im) };
    qcv v = { mul(sh, d.re), mul(sh, d.im) };
    if(!inv){
        y[1].re = add(u.re, v.im); y[1].im = sub(u.im, v.re);
        y[2].re = sub(u.re, v.im); y[2].im = add(u.im, v.re);
    }else{
        y[1].re = sub(u.re, v.im); y[1].im = add(u.im, v.re);
        y[2].re = add(u.re, v.im); y[2].im = sub(u.im, v.re);
    }
}

// radix-5 Winograd (5 real mults, 17 adds)
static inline void q_pfa5(const qcv *x, qcv *y, int inv){
    const _dvec CP  = set1_d(-0.25);
    const _dvec CM  = set1_d(0.55901699437494742);
    const _dvec S1  = set1_d(0.95105651629515357);
    const _dvec S1p = set1_d(1.5388417685876267);
    const _dvec S2m = set1_d(-0.36327126400268044);
    qcv u14 = q_add(x[1], x[4]), v14 = q_sub(x[1], x[4]);
    qcv u25 = q_add(x[2], x[3]), v25 = q_sub(x[2], x[3]);
    qcv us = q_add(u14, u25), um = q_sub(u14, u25), vs = q_sub(v14, v25);
    qcv m1 = { mul(us.re, CP),   mul(us.im, CP) };
    qcv m2 = { mul(um.re, CM),   mul(um.im, CM) };
    qcv m3 = { mul(vs.re, S1),   mul(vs.im, S1) };
    qcv m4 = { mul(v25.re, S1p), mul(v25.im, S1p) };
    qcv m5 = { mul(v14.re, S2m), mul(v14.im, S2m) };
    qcv I14 = q_add(m3, m4), I23 = q_add(m3, m5);
    qcv mid = q_add(x[0], m1);
    qcv R14 = q_add(mid, m2), R23 = q_sub(mid, m2);
    y[0] = q_add(x[0], us);
    if(!inv){
        y[1].re = add(R14.re, I14.im); y[1].im = sub(R14.im, I14.re);
        y[4].re = sub(R14.re, I14.im); y[4].im = add(R14.im, I14.re);
        y[2].re = add(R23.re, I23.im); y[2].im = sub(R23.im, I23.re);
        y[3].re = sub(R23.re, I23.im); y[3].im = add(R23.im, I23.re);
    }else{
        y[1].re = sub(R14.re, I14.im); y[1].im = add(R14.im, I14.re);
        y[4].re = add(R14.re, I14.im); y[4].im = sub(R14.im, I14.re);
        y[2].re = sub(R23.re, I23.im); y[2].im = add(R23.im, I23.re);
        y[3].re = add(R23.re, I23.im); y[3].im = sub(R23.im, I23.re);
    }
}

// radix-7 direct form (chained FMAs; this is the precision-critical one:
// it fails first uncentered, see the PQ16_PFA7_MAX_N gate)
static inline void q_pfa7(const qcv *x, qcv *y, int inv){
    const _dvec C1 = set1_d(0.62348980185873353);
    const _dvec C2 = set1_d(-0.2225209339563144);
    const _dvec C3 = set1_d(-0.90096886790241913);
    const _dvec S1 = set1_d(0.78183148246802981);
    const _dvec S2 = set1_d(0.97492791218182361);
    const _dvec S3 = set1_d(0.43388373911755812);
    qcv u16 = q_add(x[1], x[6]), v16 = q_sub(x[1], x[6]);
    qcv u25 = q_add(x[2], x[5]), v25 = q_sub(x[2], x[5]);
    qcv u34 = q_add(x[3], x[4]), v34 = q_sub(x[3], x[4]);
    qcv R1, R2, R3, I1, I2, I3;
#define PQ16_R(dst, cA, cB, cC) \
    dst.re = fmadd(cC, u34.re, fmadd(cB, u25.re, fmadd(cA, u16.re, x[0].re))); \
    dst.im = fmadd(cC, u34.im, fmadd(cB, u25.im, fmadd(cA, u16.im, x[0].im)))
    PQ16_R(R1, C1, C2, C3);
    PQ16_R(R2, C2, C3, C1);
    PQ16_R(R3, C3, C1, C2);
#undef PQ16_R
    I1.re = fmadd(S3, v34.re, fmadd(S2, v25.re, mul(S1, v16.re)));
    I1.im = fmadd(S3, v34.im, fmadd(S2, v25.im, mul(S1, v16.im)));
    I2.re = fmsub(S2, v16.re, fmadd(S1, v34.re, mul(S3, v25.re)));
    I2.im = fmsub(S2, v16.im, fmadd(S1, v34.im, mul(S3, v25.im)));
    I3.re = fmadd(S3, v16.re, fmsub(S2, v34.re, mul(S1, v25.re)));
    I3.im = fmadd(S3, v16.im, fmsub(S2, v34.im, mul(S1, v25.im)));
    y[0].re = add(add(x[0].re, u16.re), add(u25.re, u34.re));
    y[0].im = add(add(x[0].im, u16.im), add(u25.im, u34.im));
#define PQ16_P(ya, yb, R, I) \
    if(!inv){ ya.re = add(R.re, I.im); ya.im = sub(R.im, I.re); \
              yb.re = sub(R.re, I.im); yb.im = add(R.im, I.re); } \
    else    { ya.re = sub(R.re, I.im); ya.im = add(R.im, I.re); \
              yb.re = add(R.re, I.im); yb.im = sub(R.im, I.re); }
    PQ16_P(y[1], y[6], R1, I1)
    PQ16_P(y[2], y[5], R2, I2)
    PQ16_P(y[3], y[4], R3, I3)
#undef PQ16_P
}

static inline void q_pfa_bfly(const qcv *x, qcv *y, uint32_t M, int inv){
    if(M == 3)      q_pfa3(x, y, inv);
    else if(M == 5) q_pfa5(x, y, inv);
    else            q_pfa7(x, y, inv);
}

// lane-residue merge masks: km[d] selects lanes l with l % M == d
static inline void q_kmasks(uint8_t km[8], uint32_t M){
    for(uint32_t d = 0; d < M; ++d){
        uint8_t m = 0;
        for(uint32_t l = 0; l < 8; ++l)
            if(l % M == d) m |= (uint8_t)(1u << l);
        km[d] = m;
    }
}
// assemble the residue-(shifted) view: lane l takes y[(ph + l) mod M]
static inline qcv q_lane_merge(const qcv *y, uint32_t M, uint32_t ph,
                               const uint8_t *km){
    qcv o = y[0];
    for(uint32_t j = 0; j < M; ++j){
        uint32_t d = (j + M - ph) % M;
        o.re = vec_fn(mask_mov_pd)(o.re, k8(km[d]), y[j].re);
        o.im = vec_fn(mask_mov_pd)(o.im, k8(km[d]), y[j].im);
    }
    return o;
}

// PFA input, merge form (wins at M = 3): decode M natural-order stripes
// (stripe s = complex [s*n, (s+1)*n)) and apply the radix-M DFT across the
// a = k mod M dimension. Lane l of the stripe with lane-0 residue r has
// residue (r+l) mod M, so the per-residue inputs x[a] assemble from the
// stripes by constant lane-residue masks.
__attribute__((always_inline))
static inline void q_pfa_input_merge(double *data, const uint64_t *src,
                                     int64_t cnt, uint32_t n, uint32_t M,
                                     const pq16_plan *pl, int cen){
    uint8_t km[8]; q_kmasks(km, M);
    const uint64_t *s[7]; double *p[7]; uint32_t r[7]; int64_t rm[7];
    for(uint32_t b = 0; b < M; ++b){
        s[b]  = src + (size_t)b * (n / 2u);
        rm[b] = cnt - (int64_t)b * (int64_t)(n / 2u);
        p[b]  = data + 2u * (size_t)n * b;
        r[b]  = (uint32_t)(((uint64_t)b * n) % M);
    }
    const _vec *IDX = cen ? pl->dec_idxc : pl->dec_idx;
    for(uint32_t i = 0; i < n / 16u; ++i){
        qcv t0[7], t1[7];
        for(uint32_t b = 0; b < M; ++b){
            q_dec2(&t0[b], &t1[b], q_raw8(s[b], rm[b]), IDX, cen);
            s[b] += 8;
            rm[b] -= 8;
        }
        for(int u = 0; u < 2; ++u){
            const qcv *t = u ? t1 : t0;
            qcv x[7], y[7];
            for(uint32_t a = 0; a < M; ++a){
                x[a] = t[0];
                for(uint32_t b = 0; b < M; ++b){
                    uint32_t ru = (r[b] + (uint32_t)(8 * u)) % M;
                    uint32_t d  = (a + M - ru) % M;
                    x[a].re = vec_fn(mask_mov_pd)(x[a].re, k8(km[d]), t[b].re);
                    x[a].im = vec_fn(mask_mov_pd)(x[a].im, k8(km[d]), t[b].im);
                }
            }
            q_pfa_bfly(x, y, M, 0);
            for(uint32_t b = 0; b < M; ++b)
                q_st(p[b] + 16 * u, y[b]);
        }
        for(uint32_t b = 0; b < M; ++b){
            p[b] += 32;
            r[b] = (r[b] + 16u) % M;
        }
    }
}
// twiddle-fixup form (wins at M = 5, 7): the M-DFT runs directly on
// residue-relabeled stripe vectors, y[b](l) = sum_s w^{b r_s} w^{b l}
// t_s(l), so the M^2 lane merges collapse to M constant per-lane cmuls.
// Stripe POINTERS are held in residue order (sp[rho] = stripe whose lane-0
// residue is currently rho) and rotated by the constant 16 mod M each
// iteration: every vector-slot index is compile-time constant, the runtime
// relabel is pure GPR pointer renaming. Sub-tile 1 relabels by 8 mod M.
__attribute__((always_inline))
static inline void q_pfa_input_tw(double *data, const uint64_t *src,
                                  int64_t cnt, uint32_t n, uint32_t M,
                                  const double *wt, const pq16_plan *pl,
                                  int cen){
    const uint64_t *sp[7]; double *p[7]; int64_t rm[7];
    for(uint32_t b = 0; b < M; ++b){
        uint32_t rho = (uint32_t)(((uint64_t)b * n) % M);
        sp[rho] = src + (size_t)b * (n / 2u);
        rm[rho] = cnt - (int64_t)b * (int64_t)(n / 2u);
        p[b] = data + 2u * (size_t)n * b;
    }
    const _vec *IDX = cen ? pl->dec_idxc : pl->dec_idx;
    const uint32_t sh8 = 8u % M, sh16 = 16u % M;
    for(uint32_t i = 0; i < n / 16u; ++i){
        qcv t0[7], t1[7], x[7], y[7];
        for(uint32_t rho = 0; rho < M; ++rho){
            q_dec2(&t0[rho], &t1[rho], q_raw8(sp[rho], rm[rho]), IDX, cen);
            sp[rho] += 8;
            rm[rho] -= 8;
        }
        q_pfa_bfly(t0, y, M, 0);
        for(uint32_t b = 0; b < M; ++b){
            qcv w = q_ld(wt + 16u * b);
            q_st(p[b], q_mul(y[b], w));
        }
        for(uint32_t rho = 0; rho < M; ++rho)
            x[(rho + sh8) % M] = t1[rho];
        q_pfa_bfly(x, y, M, 0);
        for(uint32_t b = 0; b < M; ++b){
            qcv w = q_ld(wt + 16u * b);
            q_st(p[b] + 16, q_mul(y[b], w));
            p[b] += 32;
        }
        const uint64_t *tp[7]; int64_t tr[7];
        for(uint32_t rho = 0; rho < M; ++rho){
            tp[(rho + sh16) % M] = sp[rho];
            tr[(rho + sh16) % M] = rm[rho];
        }
        for(uint32_t rho = 0; rho < M; ++rho){ sp[rho] = tp[rho]; rm[rho] = tr[rho]; }
    }
}
// literal-M dispatch so the impls specialize (x[]/y[] enregister, the
// residue arithmetic strength-reduces). Merge vs tw split is end-to-end
// measured (bench/pfa_io_bench + full-mul A/B).
static inline void q_pfa_input(double *data, const uint64_t *src, int64_t cnt,
                               uint32_t n, uint32_t M, const pq16_plan *pl,
                               int cen){
    switch(M){
    case 3:  q_pfa_input_merge(data, src, cnt, n, 3u, pl, cen); break;
    case 5:  q_pfa_input_tw(data, src, cnt, n, 5u, pl->wt5, pl, cen); break;
    default: q_pfa_input_tw(data, src, cnt, n, 7u, pl->wt7, pl, cen); break;
    }
}

// ------------------------------------------------------------------
// radix-2^3 ancestor passes at length len
// ------------------------------------------------------------------
__attribute__((always_inline))
static inline void pq16_r8_stage(double *data, uint32_t n, uint32_t len,
                                 const double *tw){
    const size_t st = 2u * (size_t)(len / 8u);       // doubles per eighth
    const int twc = pq16_twc(len);
    for(uint32_t base = 0; base < n; base += len){
        double *p[8];
        p[0] = data + 2u * (size_t)base;
        for(int i = 1; i < 8; ++i) p[i] = p[i - 1] + st;
        const double *twp = tw;
        for(uint32_t t = 0; t < len / 64u; ++t){
            qcv v[8];
            for(int i = 0; i < 8; ++i) v[i] = q_ld(p[i]);
            qcv W1, W1o, W2, W4;
            pq16_tw8_get(twp, twc, &W1, &W1o, &W2, &W4);
            twp += pq16_tw_step(twc);
            q_bf8(v, W1, W1o, W2, W4);
            for(int i = 0; i < 8; ++i){ q_st(p[i], v[i]); p[i] += 16; }
        }
    }
}
__attribute__((always_inline))
static inline void pq16_ir8_stage(double *data, uint32_t n, uint32_t len,
                                  const double *tw){
    const size_t st = 2u * (size_t)(len / 8u);
    const int twc = pq16_twc(len);
    for(uint32_t base = 0; base < n; base += len){
        double *p[8];
        p[0] = data + 2u * (size_t)base;
        for(int i = 1; i < 8; ++i) p[i] = p[i - 1] + st;
        const double *twp = tw;
        for(uint32_t t = 0; t < len / 64u; ++t){
            qcv v[8];
            for(int i = 0; i < 8; ++i) v[i] = q_ld(p[i]);
            qcv W1, W1o, W2, W4;
            pq16_tw8_get(twp, twc, &W1, &W1o, &W2, &W4);
            twp += pq16_tw_step(twc);
            q_ibf8(v, W1, W1o, W2, W4);
            for(int i = 0; i < 8; ++i){ q_st(p[i], v[i]); p[i] += 16; }
        }
    }
}

// ------------------------------------------------------------------
// leaves. Output: stored tile slot q lane t = canonical-BR t*8+q (pi
// layout, no transpose-back -- the pointwise + inverse leaves consume it)
// ------------------------------------------------------------------
static inline void q_l64const(qcv lw[4], const pq16_plan *pl){
    lw[0] = q_ld(pl->l64w);      lw[1] = q_ld(pl->l64w + 16);
    lw[2] = q_ld(pl->l64w + 32); lw[3] = q_ld(pl->l64w + 48);
}

// 64-point body: 8 tiles, across-r8 + transpose + DFT8
static inline void q_body64(qcv v[8], const qcv lw[4]){
    q_bf8(v, lw[0], lw[1], lw[2], lw[3]);
    q_tr8cv(v);
    q_dft8(v);
}
static inline void q_ibody64(qcv v[8], const qcv lw[4]){
    q_idft8(v);
    q_tr8cv(v);
    q_ibf8(v, lw[0], lw[1], lw[2], lw[3]);
}

static inline void pq16_leaf64_run(double *d, uint32_t span, const pq16_plan *pl){
    qcv lw[4]; q_l64const(lw, pl);
    for(uint32_t g = 0; g < span / 64u; ++g){
        double *p = d + 128u * (size_t)g;
        qcv v[8];
        for(int i = 0; i < 8; ++i) v[i] = q_ld(p + 16 * i);
        q_body64(v, lw);
        for(int i = 0; i < 8; ++i) q_st(p + 16 * i, v[i]);
    }
}
static inline void pq16_ileaf64_run(double *d, uint32_t span, const pq16_plan *pl){
    qcv lw[4]; q_l64const(lw, pl);
    for(uint32_t g = 0; g < span / 64u; ++g){
        double *p = d + 128u * (size_t)g;
        qcv v[8];
        for(int i = 0; i < 8; ++i) v[i] = q_ld(p + 16 * i);
        q_ibody64(v, lw);
        for(int i = 0; i < 8; ++i) q_st(p + 16 * i, v[i]);
    }
}

// two 32-point blocks batched (8 tiles): per-block across-r22, combined
// transpose + DFT8. Lanes 0..3 = block A sub-blocks, 4..7 = block B.
static inline void pq16_leaf32x2_one(double *p, qcv W1, qcv W2){
    qcv v[8];
    for(int i = 0; i < 8; ++i) v[i] = q_ld(p + 16 * i);
    q_bf4(v, W1, W2);
    q_bf4(v + 4, W1, W2);
    q_tr8cv(v);
    q_dft8(v);
    for(int i = 0; i < 8; ++i) q_st(p + 16 * i, v[i]);
}
static inline void pq16_ileaf32x2_one(double *p, qcv W1, qcv W2){
    qcv v[8];
    for(int i = 0; i < 8; ++i) v[i] = q_ld(p + 16 * i);
    q_idft8(v);
    q_tr8cv(v);
    q_ibf4(v, W1, W2);
    q_ibf4(v + 4, W1, W2);
    for(int i = 0; i < 8; ++i) q_st(p + 16 * i, v[i]);
}
static inline void pq16_leaf32x2_run(double *d, uint32_t span, const pq16_plan *pl){
    qcv W1 = q_ld(pl->l32w), W2 = q_ld(pl->l32w + 16);
    for(uint32_t g = 0; g < span / 64u; ++g)
        pq16_leaf32x2_one(d + 128u * (size_t)g, W1, W2);
}
static inline void pq16_ileaf32x2_run(double *d, uint32_t span, const pq16_plan *pl){
    qcv W1 = q_ld(pl->l32w), W2 = q_ld(pl->l32w + 16);
    for(uint32_t g = 0; g < span / 64u; ++g)
        pq16_ileaf32x2_one(d + 128u * (size_t)g, W1, W2);
}

// 128-point leaf: across-r22 at len 128 (2 levels, streaming over 4-tile
// groups) + two 32x2 batches (5 levels each). Two register-clean L1
// passes; the r2 + two-64-body form needed three and spilled.
static inline void pq16_leaf128_run(double *d, uint32_t span, const pq16_plan *pl){
    qcv W1_32 = q_ld(pl->l32w), W2_32 = q_ld(pl->l32w + 16);
    for(uint32_t g = 0; g < span / 128u; ++g){
        double *p = d + 256u * (size_t)g;
        for(int t = 0; t < 4; ++t){
            qcv v[4] = { q_ld(p + 16 * t),       q_ld(p + 16 * (t + 4)),
                         q_ld(p + 16 * (t + 8)), q_ld(p + 16 * (t + 12)) };
            qcv W1 = q_ld(pl->l128w + 32 * t);
            qcv W2 = q_ld(pl->l128w + 32 * t + 16);
            q_bf4(v, W1, W2);
            q_st(p + 16 * t,        v[0]);
            q_st(p + 16 * (t + 4),  v[1]);
            q_st(p + 16 * (t + 8),  v[2]);
            q_st(p + 16 * (t + 12), v[3]);
        }
        pq16_leaf32x2_one(p,       W1_32, W2_32);
        pq16_leaf32x2_one(p + 128, W1_32, W2_32);
    }
}
static inline void pq16_ileaf128_run(double *d, uint32_t span, const pq16_plan *pl){
    qcv W1_32 = q_ld(pl->l32w), W2_32 = q_ld(pl->l32w + 16);
    for(uint32_t g = 0; g < span / 128u; ++g){
        double *p = d + 256u * (size_t)g;
        pq16_ileaf32x2_one(p,       W1_32, W2_32);
        pq16_ileaf32x2_one(p + 128, W1_32, W2_32);
        for(int t = 0; t < 4; ++t){
            qcv v[4] = { q_ld(p + 16 * t),       q_ld(p + 16 * (t + 4)),
                         q_ld(p + 16 * (t + 8)), q_ld(p + 16 * (t + 12)) };
            qcv W1 = q_ld(pl->l128w + 32 * t);
            qcv W2 = q_ld(pl->l128w + 32 * t + 16);
            q_ibf4(v, W1, W2);
            q_st(p + 16 * t,        v[0]);
            q_st(p + 16 * (t + 4),  v[1]);
            q_st(p + 16 * (t + 8),  v[2]);
            q_st(p + 16 * (t + 12), v[3]);
        }
    }
}

static inline void pq16_leaf_run(double *d, uint32_t span, uint32_t leaf,
                                 const pq16_plan *pl){
    if(leaf == 64u)      pq16_leaf64_run(d, span, pl);
    else if(leaf == 32u) pq16_leaf32x2_run(d, span, pl);
    else                 pq16_leaf128_run(d, span, pl);
}
static inline void pq16_ileaf_run(double *d, uint32_t span, uint32_t leaf,
                                  const pq16_plan *pl){
    if(leaf == 64u)      pq16_ileaf64_run(d, span, pl);
    else if(leaf == 32u) pq16_ileaf32x2_run(d, span, pl);
    else                 pq16_ileaf128_run(d, span, pl);
}

// ------------------------------------------------------------------
// drivers. Recursive blocking ladder (~L3-slice / L2 / L1 of tile data):
// stages too long for the current tier run as full-span sweeps, everything
// below recurses into tier-sized chunks so a chunk's remaining stages +
// leaves happen while cache-resident.
// ------------------------------------------------------------------
#ifndef PQ16_BLOCK
#define PQ16_BLOCK 2048u    /* L1 tier: 32 KB of tile data */
#endif
#ifndef PQ16_BLOCK2
#define PQ16_BLOCK2 32768u  /* L2 tier: 512 KB */
#endif
#ifndef PQ16_BLOCK3
#define PQ16_BLOCK3 131072u /* L3 tier: 2 MB */
#endif
static const uint32_t PQ16_TIERS[3] = { PQ16_BLOCK3, PQ16_BLOCK2, PQ16_BLOCK };

static void pq16_fwd_rec(double *d, uint32_t span, const pq16_shape *sh,
                         uint32_t s, unsigned tier, const pq16_plan *pl){
    while(tier < 3 && PQ16_TIERS[tier] >= span) tier++;
    if(tier == 3){
        for(uint32_t ss = s; ss < sh->cnt; ++ss)
            pq16_r8_stage(d, span, sh->len[ss], pl->tw8[__builtin_ctz(sh->len[ss])]);
        pq16_leaf_run(d, span, sh->leaf, pl);
        return;
    }
    uint32_t chunk = PQ16_TIERS[tier];
    while(s < sh->cnt && sh->len[s] > chunk){
        pq16_r8_stage(d, span, sh->len[s], pl->tw8[__builtin_ctz(sh->len[s])]);
        s++;
    }
    for(uint32_t base = 0; base < span; base += chunk)
        pq16_fwd_rec(d + 2u * (size_t)base, chunk, sh, s, tier + 1u, pl);
}
static void pq16_inv_rec(double *d, uint32_t span, const pq16_shape *sh,
                         uint32_t s, unsigned tier, const pq16_plan *pl){
    while(tier < 3 && PQ16_TIERS[tier] >= span) tier++;
    if(tier == 3){
        for(uint32_t ss = sh->cnt; ss-- > s; )
            pq16_ir8_stage(d, span, sh->len[ss], pl->tw8[__builtin_ctz(sh->len[ss])]);
        return;
    }
    uint32_t chunk = PQ16_TIERS[tier];
    uint32_t s1 = s;
    while(s1 < sh->cnt && sh->len[s1] > chunk) s1++;
    for(uint32_t base = 0; base < span; base += chunk)
        pq16_inv_rec(d + 2u * (size_t)base, chunk, sh, s1, tier + 1u, pl);
    for(uint32_t ss = s1; ss-- > s; )
        pq16_ir8_stage(d, span, sh->len[ss], pl->tw8[__builtin_ctz(sh->len[ss])]);
}

static inline void pq16_fwd_core(double *data, uint32_t n, const pq16_plan *pl){
    pq16_shape sh = pq16_shape_of(n);
    pq16_fwd_rec(data, n, &sh, 0, 0, pl);
}
// inverse r8 stages only (leaves already applied by the fused pointwise)
static inline void pq16_inv_r8_only(double *data, uint32_t n, const pq16_plan *pl){
    pq16_shape sh = pq16_shape_of(n);
    pq16_inv_rec(data, n, &sh, 0, 0, pl);
}
static inline void pq16_fwd(double *data, const uint64_t *src, int64_t cnt,
                            uint32_t n, const pq16_plan *pl, int cen){
    pq16_input_stage(data, src, cnt, n, pl, cen);
    pq16_fwd_core(data, n, pl);
}
// PFA forward: radix-M input, then mem_r22 + core per branch
static inline void pq16_pfa_fwd(double *data, const uint64_t *src, int64_t cnt,
                                uint32_t n, uint32_t M, const pq16_plan *pl,
                                int cen){
    q_pfa_input(data, src, cnt, n, M, pl, cen);
    const double *tw = pl->tw22[__builtin_ctz(n)];
    for(uint32_t b = 0; b < M; ++b){
        double *d = data + 2u * (size_t)n * b;
        pq16_mem_r22(d, n, tw);
        pq16_fwd_core(d, n, pl);
    }
}

// ------------------------------------------------------------------
// PQ pointwise multiply (in pi layout, A *= B elementwise-PQ).
// Per-group twiddle reconstruction from the compact g-table: lane t of
// stored vector (g, q) needs g[g*16 + 2t + (q>>2)] (one vpermt2pd per
// component, hoisted per group) and the 1 +- g pattern by q & 3.
// ------------------------------------------------------------------
// paired 8-lane PQ eval (partner positions k, N/2-k share cross-terms)
static inline void q_pq_pair(qcv *outl, qcv *outr, qcv x, qcv xn, qcv y, qcv yn,
                             qcv w, _dvec sc, _dvec qsc){
    qcv pql = q_mul(x, y);
    qcv pqr = q_mul(xn, yn);
    qcv dp = { sub(x.re, xn.re), add(x.im, xn.im) };
    qcv dq = { sub(y.re, yn.re), add(y.im, yn.im) };
    qcv t = q_mul(dp, dq);
    qcv c = q_mul(t, w);
    _dvec qcr = mul(qsc, c.re);
    _dvec qci = mul(qsc, c.im);
    outl->re = fmsub(pql.re, sc, qcr);
    outl->im = fmsub(pql.im, sc, qci);
    outr->re = fmsub(pqr.re, sc, qcr);
    outr->im = fmadd(pqr.im, sc, qci);
}
// single-sided 8-lane PQ eval: each lane uses its own twiddle
static inline qcv q_pq_eval(qcv x, qcv xn, qcv y, qcv yn, qcv w,
                            _dvec sc, _dvec qsc){
    qcv pq = q_mul(x, y);
    qcv dp = { sub(x.re, xn.re), add(x.im, xn.im) };
    qcv dq = { sub(y.re, yn.re), add(y.im, yn.im) };
    qcv c = q_mul(q_mul(dp, dq), w);
    qcv z = { fmsub(pq.re, sc, mul(qsc, c.re)),
              fmsub(pq.im, sc, mul(qsc, c.im)) };
    return z;
}

typedef struct { _dvec gr[2], gi[2]; } q_gsel;
static inline q_gsel q_pq_gsel(const double *gblk){
    const _vec IE = setr_64(0, 2, 4, 6, 8, 10, 12, 14);
    const _vec IO = setr_64(1, 3, 5, 7, 9, 11, 13, 15);
    _dvec r0 = load_dvec(gblk),      r1 = load_dvec(gblk + 8);
    _dvec i0 = load_dvec(gblk + 16), i1 = load_dvec(gblk + 24);
    q_gsel s;
    s.gr[0] = vec_fn(permutex2var_pd)(r0, IE, r1);
    s.gr[1] = vec_fn(permutex2var_pd)(r0, IO, r1);
    s.gi[0] = vec_fn(permutex2var_pd)(i0, IE, i1);
    s.gi[1] = vec_fn(permutex2var_pd)(i0, IO, i1);
    return s;
}
static inline qcv q_pq_w(const q_gsel *s, uint32_t q){
    _dvec gr = s->gr[q >> 2], gi = s->gi[q >> 2];
    const _dvec one = set1_d(1.0);
    const _dvec nz  = set1_d(-0.0);
    qcv w;
    switch(q & 3u){
    case 0:  w.re = add(one, gr); w.im = gi; break;
    case 1:  w.re = sub(one, gr); w.im = vec_fn(xor_pd)(gi, nz); break;
    case 2:  w.re = sub(one, gi); w.im = gr; break;
    default: w.re = add(one, gi); w.im = vec_fn(xor_pd)(gr, nz); break;
    }
    return w;
}
// gsel with the PFA branch rotation g' = g * w3 (w3 = omega_M^b)
static inline q_gsel q_pq_gsel_rot(const double *gblk, double w3r, double w3i){
    q_gsel s = q_pq_gsel(gblk);
    const _dvec vr = set1_d(w3r), vi = set1_d(w3i);
    for(int b = 0; b < 2; ++b){
        _dvec nr = fmsub(s.gr[b], vr, mul(s.gi[b], vi));
        _dvec ni = fmadd(s.gr[b], vi, mul(s.gi[b], vr));
        s.gr[b] = nr; s.gi[b] = ni;
    }
    return s;
}
// head twiddle for canonical positions 0..7 (lane p), from the gh0
// g-values rotated by w3; the per-lane 1 +- g pattern (case = p & 3) is
// applied with constant blend/sign masks
static inline qcv q_pq_head_w(const pq16_plan *pl, double w3r, double w3i){
    _dvec gr = load_dvec(pl->gh0), gi = load_dvec(pl->gh0 + 8);
    const _dvec vr = set1_d(w3r), vi = set1_d(w3i);
    _dvec gr2 = fmsub(gr, vr, mul(gi, vi));
    _dvec gi2 = fmadd(gr, vi, mul(gi, vr));
    const _dvec one = set1_d(1.0);
    const _dvec sre = vec_fn(setr_pd)(0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0);
    const _dvec sim = vec_fn(setr_pd)(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    qcv w;
    w.re = add(one, vec_fn(xor_pd)(vec_fn(mask_blend_pd)(k8(0xCC), gr2, gi2), sre));
    w.im = vec_fn(xor_pd)(vec_fn(mask_blend_pd)(k8(0xCC), gi2, gr2), sim);
    return w;
}

// the part8 lane map (pairing pattern of canonical positions 0..7) and the
// column-gather index for stored group 0 lane 0
#define PQ16_P8()  setr_64(0, 1, 3, 2, 7, 6, 5, 4)
#define PQ16_COL() _mm256_setr_epi32(0, 16, 32, 48, 64, 80, 96, 112)
static inline qcv q_perm8(qcv a, _vec P8){
    qcv r = { permd(P8, a.re), permd(P8, a.im) };
    return r;
}

// pointwise over stored group 0 (canonical positions [0, 64)). Lane 0
// holds the first 8 positions: their pairing is part8 across slots,
// evaluated single-sided per lane (gather column, permute, scatter).
// Lanes 1..7 are the conjugate-mirror pairing of bases 8/16/32, which in
// stored coords is slot-reverse (q <-> 7-q) with the SAME part8 pattern as
// a lane map -- four masked pq_pair calls.
static inline void q_pw_head(double *A, double *B, const pq16_plan *pl,
                             _dvec sc, _dvec qsc){
    const _vec P8 = PQ16_P8();
    const __m256i COL = PQ16_COL();

    qcv x, y;
    x.re = vec_fn(i32gather_pd)(COL, A, 8);
    x.im = vec_fn(i32gather_pd)(COL, A + 8, 8);
    y.re = vec_fn(i32gather_pd)(COL, B, 8);
    y.im = vec_fn(i32gather_pd)(COL, B + 8, 8);
    qcv w  = q_pq_head_w(pl, 1.0, 0.0);
    qcv xn = q_perm8(x, P8);
    qcv yn = q_perm8(y, P8);
    qcv z = q_pq_eval(x, xn, y, yn, w, sc, qsc);
    vec_fn(i32scatter_pd)(A, COL, z.re, 8);
    vec_fn(i32scatter_pd)(A + 8, COL, z.im, 8);

    q_gsel gs = q_pq_gsel(pl->pq);
    for(uint32_t q = 0; q < 4; ++q){
        size_t vl = q, vr = 7u - q;
        qcv xq  = q_ld(A + 16u * vl);
        qcv xnq = q_perm8(q_ld(A + 16u * vr), P8);
        qcv yq  = q_ld(B + 16u * vl);
        qcv ynq = q_perm8(q_ld(B + 16u * vr), P8);
        qcv wq  = q_pq_w(&gs, q);
        qcv outl, outr;
        q_pq_pair(&outl, &outr, xq, xnq, yq, ynq, wq, sc, qsc);
        vec_fn(mask_store_pd)(A + 16u * vl,     k8(0xFE), outl.re);
        vec_fn(mask_store_pd)(A + 16u * vl + 8, k8(0xFE), outl.im);
        outr = q_perm8(outr, P8);
        vec_fn(mask_store_pd)(A + 16u * vr,     k8(0xFE), outr.re);
        vec_fn(mask_store_pd)(A + 16u * vr + 8, k8(0xFE), outr.im);
    }
}

// pointwise one group pair: stored vec (gl, q) <-> rev(vec (gr, 7-q)).
// gl == gr handles the in-group mirror (base 64): q = 0..3 only.
static inline void q_pw_groups(double *A, double *B, uint32_t gl, uint32_t gr,
                               const pq16_plan *pl, _dvec sc, _dvec qsc){
    uint32_t qe = gl == gr ? 4u : 8u;
    q_gsel gs = q_pq_gsel(pl->pq + 32u * (size_t)gl);
    for(uint32_t q = 0; q < qe; ++q){
        size_t vl = (size_t)gl * 8u + q, vr = (size_t)gr * 8u + 7u - q;
        qcv x  = q_ld(A + 16u * vl);
        qcv xn = q_rev(q_ld(A + 16u * vr));
        qcv y  = q_ld(B + 16u * vl);
        qcv yn = q_rev(q_ld(B + 16u * vr));
        qcv w  = q_pq_w(&gs, q);
        qcv outl, outr;
        q_pq_pair(&outl, &outr, x, xn, y, yn, w, sc, qsc);
        q_st(A + 16u * vl, outl);
        q_st(A + 16u * vr, q_rev(outr));
    }
}

// inverse leaf over one 64-cx stored group (or its 128 block for leaf128)
static inline void q_ileaf_group(double *A, uint32_t g, uint32_t leaf,
                                 const pq16_plan *pl){
    if(leaf == 128u) pq16_ileaf128_run(A + 256u * (size_t)(g >> 1), 128u, pl);
    else             pq16_ileaf_run(A + 128u * (size_t)g, 64u, leaf, pl);
}

// fused pointwise + inverse leaves: each mirror pair of groups is
// multiplied and immediately inverse-leafed while cache-hot. For leaf128
// the pair structure is block-aligned (gl even, partner descending), so
// leaves run once per touched block.
static inline void pq16_pointwise_ileaves(double *A, double *B, uint32_t n,
                                          double sc_d, uint32_t leaf,
                                          const pq16_plan *pl){
    const _dvec sc = set1_d(sc_d), qsc = set1_d(0.25 * sc_d);

    q_pw_head(A, B, pl, sc, qsc);            // group 0
    q_pw_groups(A, B, 1, 1, pl, sc, qsc);    // base 64: in-group mirror
    if(leaf == 128u) pq16_ileaf128_run(A, 128u, pl);
    else { q_ileaf_group(A, 0, leaf, pl); q_ileaf_group(A, 1, leaf, pl); }

    for(uint32_t base = 128; base < n; base <<= 1){
        uint32_t g0 = base / 64u;
        for(uint32_t gl = g0; gl < g0 + base / 128u; ++gl){
            uint32_t gr = 3u * g0 - 1u - gl;
            q_pw_groups(A, B, gl, gr, pl, sc, qsc);
            if(leaf != 128u){
                q_ileaf_group(A, gl, leaf, pl);
                q_ileaf_group(A, gr, leaf, pl);
            }else if(base == 128u){
                // gl, gr are the two halves of one 128-block (block-scale
                // analog of the base-64 in-group mirror)
                pq16_ileaf128_run(A + 256u * (size_t)(gl >> 1), 128u, pl);
            }else if(gl & 1u){   // both blocks of the pair now complete
                pq16_ileaf128_run(A + 256u * (size_t)((gl - 1u) >> 1), 128u, pl);
                pq16_ileaf128_run(A + 256u * (size_t)(gr >> 1), 128u, pl);
            }
        }
    }
}

// ------------------------------------------------------------------
// PFA cross-pair pointwise: primary positions in branch L pair with the
// mirrored positions of branch R (the (b, M-b) antipodal pair), twiddle g
// rotated by the constant w3 = omega_M^b. Unlike self-pairs every (L, R)
// pairing is distinct: the full group/slot range is iterated and both
// heads are evaluated single-sidedly.
// ------------------------------------------------------------------
static inline void q_pw_head_cross(double *L, double *R, double *oL, double *oR,
                                   const pq16_plan *pl, double w3r, double w3i,
                                   _dvec sc, _dvec qsc){
    const _vec P8 = PQ16_P8();
    const __m256i COL = PQ16_COL();

    // lane 0: positions 0..7 of both branches, partner = part8 of the other
    qcv xl, yl, xr, yr;
    xl.re = vec_fn(i32gather_pd)(COL, L, 8);  xl.im = vec_fn(i32gather_pd)(COL, L + 8, 8);
    yl.re = vec_fn(i32gather_pd)(COL, oL, 8); yl.im = vec_fn(i32gather_pd)(COL, oL + 8, 8);
    xr.re = vec_fn(i32gather_pd)(COL, R, 8);  xr.im = vec_fn(i32gather_pd)(COL, R + 8, 8);
    yr.re = vec_fn(i32gather_pd)(COL, oR, 8); yr.im = vec_fn(i32gather_pd)(COL, oR + 8, 8);
    qcv xrp = q_perm8(xr, P8), yrp = q_perm8(yr, P8);
    qcv xlp = q_perm8(xl, P8), ylp = q_perm8(yl, P8);
    qcv wl = q_pq_head_w(pl, w3r, w3i);
    qcv wr = q_pq_head_w(pl, w3r, -w3i);
    qcv zl = q_pq_eval(xl, xrp, yl, yrp, wl, sc, qsc);
    qcv zr = q_pq_eval(xr, xlp, yr, ylp, wr, sc, qsc);
    vec_fn(i32scatter_pd)(L, COL, zl.re, 8);
    vec_fn(i32scatter_pd)(L + 8, COL, zl.im, 8);
    vec_fn(i32scatter_pd)(R, COL, zr.re, 8);
    vec_fn(i32scatter_pd)(R + 8, COL, zr.im, 8);

    // lanes 1..7, full slot range: L (q, t) <-> R (7-q, part8-lane)
    q_gsel gs = q_pq_gsel_rot(pl->pq, w3r, w3i);
    for(uint32_t q = 0; q < 8; ++q){
        size_t vl = q, vr = 7u - q;
        qcv x  = q_ld(L + 16u * vl);
        qcv xn = q_perm8(q_ld(R + 16u * vr), P8);
        qcv y  = q_ld(oL + 16u * vl);
        qcv yn = q_perm8(q_ld(oR + 16u * vr), P8);
        qcv w = q_pq_w(&gs, q);
        qcv outl, outr;
        q_pq_pair(&outl, &outr, x, xn, y, yn, w, sc, qsc);
        vec_fn(mask_store_pd)(L + 16u * vl,     k8(0xFE), outl.re);
        vec_fn(mask_store_pd)(L + 16u * vl + 8, k8(0xFE), outl.im);
        outr = q_perm8(outr, P8);
        vec_fn(mask_store_pd)(R + 16u * vr,     k8(0xFE), outr.re);
        vec_fn(mask_store_pd)(R + 16u * vr + 8, k8(0xFE), outr.im);
    }
}

static inline void q_pw_groups_cross(double *L, double *R, double *oL, double *oR,
                                     uint32_t gl, uint32_t gr, const pq16_plan *pl,
                                     double w3r, double w3i, _dvec sc, _dvec qsc){
    q_gsel gs = q_pq_gsel_rot(pl->pq + 32u * (size_t)gl, w3r, w3i);
    for(uint32_t q = 0; q < 8; ++q){
        size_t vl = (size_t)gl * 8u + q, vr = (size_t)gr * 8u + 7u - q;
        qcv x  = q_ld(L + 16u * vl);
        qcv xn = q_rev(q_ld(R + 16u * vr));
        qcv y  = q_ld(oL + 16u * vl);
        qcv yn = q_rev(q_ld(oR + 16u * vr));
        qcv w  = q_pq_w(&gs, q);
        qcv outl, outr;
        q_pq_pair(&outl, &outr, x, xn, y, yn, w, sc, qsc);
        q_st(L + 16u * vl, outl);
        q_st(R + 16u * vr, q_rev(outr));
    }
}

// fused cross pointwise + inverse leaves on both branches
static inline void pq16_pointwise_cross_ileaves(double *L, double *R,
                                                double *oL, double *oR,
                                                uint32_t n, double sc_d,
                                                double w3r, double w3i,
                                                uint32_t leaf,
                                                const pq16_plan *pl){
    const _dvec sc = set1_d(sc_d), qsc = set1_d(0.25 * sc_d);

    q_pw_head_cross(L, R, oL, oR, pl, w3r, w3i, sc, qsc);          // group 0
    q_pw_groups_cross(L, R, oL, oR, 1, 1, pl, w3r, w3i, sc, qsc);  // base 64
    if(leaf == 128u){
        pq16_ileaf128_run(L, 128u, pl);
        pq16_ileaf128_run(R, 128u, pl);
    }else{
        q_ileaf_group(L, 0, leaf, pl); q_ileaf_group(L, 1, leaf, pl);
        q_ileaf_group(R, 0, leaf, pl); q_ileaf_group(R, 1, leaf, pl);
    }
    for(uint32_t base = 128; base < n; base <<= 1){
        uint32_t g0 = base / 64u;
        for(uint32_t gl = g0; gl < 2u * g0; ++gl){       // full range
            uint32_t gr = 3u * g0 - 1u - gl;
            q_pw_groups_cross(L, R, oL, oR, gl, gr, pl, w3r, w3i, sc, qsc);
            if(leaf != 128u){
                q_ileaf_group(L, gl, leaf, pl);
                q_ileaf_group(R, gr, leaf, pl);
            }else if(gl & 1u){
                pq16_ileaf128_run(L + 256u * (size_t)((gl - 1u) >> 1), 128u, pl);
                pq16_ileaf128_run(R + 256u * (size_t)(gr >> 1), 128u, pl);
            }
        }
    }
}

// ------------------------------------------------------------------
// emit layer: round, pack 2 tiles -> 8 limbs (lo/hi), SWAR carry chain.
// ------------------------------------------------------------------
// unsigned magic-bias round (clips negatives to 0): x in [0, 2^51)
static inline _vec q_mround(_dvec x){
    const _vec bias = set1_64(0x4330000000000000LL);
    x = vec_fn(max_pd)(x, dzero());
    return sub(as_ivec(add(x, as_dvec(bias))), bias);
}
// signed magic-bias round (no clip): |x| < 2^51
static inline _vec q_mround_s(_dvec x){
    const _vec bias = set1_64(0x4338000000000000LL);
    return sub(as_ivec(add(x, as_dvec(bias))), bias);
}
typedef struct { _vec lo, hi; } q_lohi;
// 2 tiles of (re, im) integer coefficient pairs -> 8 packed u64 limbs.
// Wide split: each digit may carry up to 64 bits (lo + hi parts), so the
// codec band is precision-, not container-, limited.
static inline q_lohi q_pack2i(_vec re0, _vec im0, _vec re1, _vec im1){
    _vec ul0 = add(re0, slli(im0, 16));
    _vec uh0 = add(srli(im0, 48), maskz(ltu(ul0, re0), set1_64(1)));
    _vec ul1 = add(re1, slli(im1, 16));
    _vec uh1 = add(srli(im1, 48), maskz(ltu(ul1, re1), set1_64(1)));
    const _vec IE = setr_64(0, 2, 4, 6, 8, 10, 12, 14);
    const _vec IO = setr_64(1, 3, 5, 7, 9, 11, 13, 15);
    _vec elo = vec_fn(permutex2var_epi64)(ul0, IE, ul1);
    _vec olo = vec_fn(permutex2var_epi64)(ul0, IO, ul1);
    _vec ehi = vec_fn(permutex2var_epi64)(uh0, IE, uh1);
    _vec ohi = vec_fn(permutex2var_epi64)(uh0, IO, uh1);
    q_lohi r;
    r.lo = add(elo, slli(olo, 32));
    r.hi = add(add(ehi, srli(olo, 32)),
               add(slli(ohi, 32), maskz(ltu(r.lo, elo), set1_64(1))));
    return r;
}
static inline q_lohi q_pack2(qcv t0, qcv t1){
    return q_pack2i(q_mround(t0.re), q_mround(t0.im),
                    q_mround(t1.re), q_mround(t1.im));
}

// SWAR carry chain: each step adds the previous step's hi (lane-shifted)
// into this step's lo and resolves the 8-lane carry ripple with a
// generate/propagate mask trick (one 8-bit add does the ripple).
typedef struct { _vec prev_hi; unsigned cin; } q_chain;
static inline _vec q_chain_step(q_chain *c, q_lohi p){
    _vec his = alignr64(p.hi, c->prev_hi, 7);
    _vec sum = add(p.lo, his);
    unsigned g  = (unsigned)ltu(sum, his);
    unsigned pr = (unsigned)eq(sum, ones());
    unsigned cn = pr + ((g << 1) | c->cin);
    unsigned cy = cn ^ pr;
    sum = add(sum, set1_64(1), k8(cy), sum);
    c->cin = (cn >> 8) & 1u;
    c->prev_hi = p.hi;
    return sum;
}
// close nf parallel front chains over rp[0, rl): front f's trailing carry
// (hi lane 7 + carry-in) ripples into front f+1; the carry past rl must
// vanish (returns 0 otherwise -- defensive, indicates a precision fault)
static inline int q_chains_close(uint64_t *rp, int64_t rl, const q_chain *ch,
                                 uint32_t nf, size_t limbs_front){
    for(uint32_t f = 0; f < nf; ++f){
        uint64_t hi7;
        memcpy(&hi7, (const char *)&ch[f].prev_hi + 56, 8);
        unsigned __int128 c = (unsigned __int128)hi7 + ch[f].cin;
        if((size_t)(f + 1) * limbs_front >= (size_t)rl){
            if(c) return 0;
            continue;
        }
        uint64_t *q = rp + (size_t)(f + 1) * limbs_front;
        uint64_t *qe = rp + rl;
        while(c && q < qe){ c += *q; *q++ = (uint64_t)c; c >>= 64; }
        if(c) return 0;
    }
    return 1;
}

// final inverse r22 at len n FUSED with the u16 emit: 4 quarter-fronts,
// each with its own carry chain (uncentered band)
static inline int pq16_inv_final_emit(uint64_t *rp, int64_t rl, double *data,
                                      uint32_t n, const pq16_plan *pl){
    const size_t limbs_q = n / 8u;                   // limbs per quarter
    q_chain ch[4];
    int64_t rem[4];
    for(int f = 0; f < 4; ++f){
        ch[f].prev_hi = zero(); ch[f].cin = 0;
        rem[f] = rl - (int64_t)((size_t)f * limbs_q);
    }
    // named per-front state: a front-indexed o[4][2] defeats SROA here
    // (clang keeps it in memory, ~20 spills in the hot loop)
    double *p0 = data;
    double *p1 = data + 2u * (size_t)(n / 4u);
    double *p2 = data + 4u * (size_t)(n / 4u);
    double *p3 = data + 6u * (size_t)(n / 4u);
    uint64_t *r0 = rp;
    uint64_t *r1 = rp + limbs_q;
    uint64_t *r2 = rp + 2u * limbs_q;
    uint64_t *r3 = rp + 3u * limbs_q;
    const double *twp = pl->tw22[__builtin_ctz(n)];
    const int twc = pq16_twc(n);
    for(uint32_t t = 0; t < n / 64u; ++t){
        qcv o0[2], o1[2], o2[2], o3[2];
        qcv w1a, w2a, w1b, w2b;
        pq16_tw22_get(twp, twc, &w1a, &w2a, &w1b, &w2b);
        twp += pq16_tw_step(twc);
        for(int u = 0; u < 2; ++u){
            qcv v[4] = { q_ld(p0 + 16 * u), q_ld(p1 + 16 * u),
                         q_ld(p2 + 16 * u), q_ld(p3 + 16 * u) };
            q_ibf4(v, u ? w1b : w1a, u ? w2b : w2a);
            o0[u] = v[0]; o1[u] = v[1]; o2[u] = v[2]; o3[u] = v[3];
        }
        p0 += 32; p1 += 32; p2 += 32; p3 += 32;
        q_lohi q;
        q = q_pack2(o0[0], o0[1]);
        store_vec(r0, q_chain_step(&ch[0], q), q_st_mask(rem[0]));
        q = q_pack2(o1[0], o1[1]);
        store_vec(r1, q_chain_step(&ch[1], q), q_st_mask(rem[1]));
        q = q_pack2(o2[0], o2[1]);
        store_vec(r2, q_chain_step(&ch[2], q), q_st_mask(rem[2]));
        q = q_pack2(o3[0], o3[1]);
        store_vec(r3, q_chain_step(&ch[3], q), q_st_mask(rem[3]));
        r0 += 8; r1 += 8; r2 += 8; r3 += 8;
        rem[0] -= 8; rem[1] -= 8; rem[2] -= 8; rem[3] -= 8;
    }
    return q_chains_close(rp, rl, ch, 4, limbs_q);
}

// ------------------------------------------------------------------
// centered codec corrections. Digits were pre-centered in the staging
// copy as (i16)(d ^ 0x8000) = d - 2^15; the product is recovered from the
// signed coefficients via  c_k = c^_k + (S_k << 15) - (C_k << 30)  with S
// the running window digit sum and C the closed-form window overlap.
// ------------------------------------------------------------------
typedef struct {
    const uint16_t *a; const uint16_t *b;
    int64_t na, nb;              // digit counts (4 * limbs)
    int xored;                   // streams are xor-centered: un-xor on load
} q_cctx;

// 16 digits at offset k (may be negative / past the end): fault-suppressed
// masked load, invalid lanes zero
static inline _vec q_dig16(const uint16_t *d, int64_t ndig, int64_t k, int xored){
    uint32_t lo = k < 0 ? (uint32_t)(-k > 16 ? 16 : -k) : 0;
    int64_t rem = ndig - k;
    uint32_t hi = rem >= 16 ? 0u : (rem <= 0 ? 16u : (uint32_t)(16 - rem));
    __mmask16 m = k16((0xFFFFu << lo) & (0xFFFFu >> hi));
    __m256i v = _mm256_maskz_loadu_epi16(m, d + k);
    if(xored)
        v = _mm256_maskz_mov_epi16(m,
            _mm256_xor_si256(v, _mm256_set1_epi16((short)0x8000)));
    return vec_fn(cvtepu16_epi32)(v);
}
static inline _vec q_prefix16_i32(_vec x){
    const _vec z = zero();
    x = vec_fn(add_epi32)(x, vec_fn(alignr_epi32)(x, z, 15));
    x = vec_fn(add_epi32)(x, vec_fn(alignr_epi32)(x, z, 14));
    x = vec_fn(add_epi32)(x, vec_fn(alignr_epi32)(x, z, 12));
    x = vec_fn(add_epi32)(x, vec_fn(alignr_epi32)(x, z, 8));
    return x;
}
// shared tail: dS prefix-sum -> even/odd S lanes -> corrections with the
// given even/odd C vectors. Updates *S past the 16-digit window.
static inline void q_corr_tail(_vec *ce, _vec *co, _vec dS, int64_t *S,
                               _vec Ce, _vec Co){
    _vec pS = q_prefix16_i32(dS);
    const _vec IE = setr_32(0, 2, 4, 6, 8, 10, 12, 14, 0, 0, 0, 0, 0, 0, 0, 0);
    const _vec IO = setr_32(1, 3, 5, 7, 9, 11, 13, 15, 0, 0, 0, 0, 0, 0, 0, 0);
    _vec base = set1_64(*S);
    _vec Se = add(base, vec_fn(cvtepi32_epi64)(
        vec_fn(castsi512_si256)(perm32(IE, pS))));
    _vec So = add(base, vec_fn(cvtepi32_epi64)(
        vec_fn(castsi512_si256)(perm32(IO, pS))));
    __m128i top = vec_fn(extracti32x4_epi32)(pS, 3);
    *S += (int32_t)_mm_extract_epi32(top, 3);
    *ce = sub(slli(Se, 15), slli(Ce, 30));
    *co = sub(slli(So, 15), slli(Co, 30));
}
// generic window (handles segment-straddling and out-of-range windows):
// dS = a_fwd + b_fwd - a_lag - b_lag, C(j) = min(j+1, na) - clamp(j+1-nb)
static inline void q_corr_generic(_vec *ce, _vec *co, const q_cctx *cx,
                                  int64_t k, int64_t *S){
    _vec af = q_dig16(cx->a, cx->na, k, cx->xored);
    _vec bf = q_dig16(cx->b, cx->nb, k, cx->xored);
    _vec al = q_dig16(cx->a, cx->na, k - cx->nb, cx->xored);
    _vec bl = q_dig16(cx->b, cx->nb, k - cx->na, cx->xored);
    _vec dS = vec_fn(sub_epi32)(vec_fn(add_epi32)(af, bf),
                                vec_fn(add_epi32)(al, bl));
    const _vec lane2 = setr_64(1, 3, 5, 7, 9, 11, 13, 15);
    _vec j1e = add(set1_64(k), lane2);               // j+1, even digits
    _vec j1o = add(j1e, set1_64(1));
    _vec nav = set1_64(cx->na), nbv = set1_64(cx->nb);
    _vec z = zero();
    _vec Ce = sub(vec_fn(min_epi64)(j1e, nav),
        vec_fn(min_epi64)(vec_fn(max_epi64)(sub(j1e, nbv), z), nav));
    _vec Co = sub(vec_fn(min_epi64)(j1o, nav),
        vec_fn(min_epi64)(vec_fn(max_epi64)(sub(j1o, nbv), z), nav));
    q_corr_tail(ce, co, dS, S, Ce, Co);
}
// segmented windows: a 16-digit window entirely inside one validity
// segment needs half the digit loads and a linear C.
//   S1 [0, lo):        dS = a_f + b_f          C = k+1        (rising)
//   S2 [lo, hi):       dS = f_fwd - f_lag      C = lo         (constant;
//                      f = the LONGER operand -- note an < bn is legal)
//   S3 [hi, na+nb):    dS = -(a_lag + b_lag)   C = na+nb-1-k  (falling)
// with lo = min(na, nb), hi = max(na, nb).
static inline void q_corr_seg(_vec *ce, _vec *co, const q_cctx *cx,
                              int64_t k, int64_t *S, int seg){
    const __m256i X = _mm256_set1_epi16((short)0x8000);
    _vec dS;
    if(seg == 1){
        __m256i a = _mm256_loadu_si256((const __m256i *)(cx->a + k));
        __m256i b = _mm256_loadu_si256((const __m256i *)(cx->b + k));
        if(cx->xored){ a = _mm256_xor_si256(a, X); b = _mm256_xor_si256(b, X); }
        dS = vec_fn(add_epi32)(vec_fn(cvtepu16_epi32)(a), vec_fn(cvtepu16_epi32)(b));
    }else if(seg == 2){
        const uint16_t *f; const uint16_t *l2;
        if(cx->na >= cx->nb){ f = cx->a + k; l2 = cx->a + k - cx->nb; }
        else                { f = cx->b + k; l2 = cx->b + k - cx->na; }
        __m256i a = _mm256_loadu_si256((const __m256i *)f);
        __m256i l = _mm256_loadu_si256((const __m256i *)l2);
        if(cx->xored){ a = _mm256_xor_si256(a, X); l = _mm256_xor_si256(l, X); }
        dS = vec_fn(sub_epi32)(vec_fn(cvtepu16_epi32)(a), vec_fn(cvtepu16_epi32)(l));
    }else{
        __m256i l = _mm256_loadu_si256((const __m256i *)(cx->a + k - cx->nb));
        __m256i m = _mm256_loadu_si256((const __m256i *)(cx->b + k - cx->na));
        if(cx->xored){ l = _mm256_xor_si256(l, X); m = _mm256_xor_si256(m, X); }
        dS = vec_fn(sub_epi32)(zero(),
            vec_fn(add_epi32)(vec_fn(cvtepu16_epi32)(l), vec_fn(cvtepu16_epi32)(m)));
    }
    _vec Ce, Co;
    const _vec lane2 = setr_64(0, 2, 4, 6, 8, 10, 12, 14);
    if(seg == 1){
        _vec je = add(set1_64(k + 1), lane2);
        Ce = je; Co = add(je, set1_64(1));
    }else if(seg == 2){
        Ce = Co = set1_64(cx->na <= cx->nb ? cx->na : cx->nb);
    }else{
        _vec je = sub(set1_64(cx->na + cx->nb - 1 - k), lane2);
        Ce = je; Co = sub(je, set1_64(1));
    }
    q_corr_tail(ce, co, dS, S, Ce, Co);
}
static inline void q_corr_auto(_vec *ce, _vec *co, const q_cctx *cx,
                               int64_t k, int64_t *S){
    int64_t k2 = k + 16;
    int64_t lo = cx->nb <= cx->na ? cx->nb : cx->na;
    int64_t hi = cx->nb <= cx->na ? cx->na : cx->nb;
    if(k2 <= lo)                              q_corr_seg(ce, co, cx, k, S, 1);
    else if(k >= lo && k2 <= hi)              q_corr_seg(ce, co, cx, k, S, 2);
    else if(k >= hi && k2 <= cx->na + cx->nb) q_corr_seg(ce, co, cx, k, S, 3);
    else                                      q_corr_generic(ce, co, cx, k, S);
}
// sum of digits j with 0 <= j < min(t, ndig) (front checkpoint helper)
static inline int64_t q_digsum(const uint16_t *d, int64_t ndig, int64_t t,
                               int xored){
    if(t > ndig) t = ndig;
    if(t <= 0) return 0;
    const __m256i X = _mm256_set1_epi16((short)0x8000);
    int64_t s = 0, i = 0;
    _vec acc = zero();
    for(; i + 32 <= t; i += 32){
        __m256i a = _mm256_loadu_si256((const __m256i *)(d + i));
        __m256i b = _mm256_loadu_si256((const __m256i *)(d + i + 16));
        if(xored){ a = _mm256_xor_si256(a, X); b = _mm256_xor_si256(b, X); }
        acc = vec_fn(add_epi32)(acc, vec_fn(add_epi32)(
            vec_fn(cvtepu16_epi32)(a), vec_fn(cvtepu16_epi32)(b)));
        if((i & 8191) == 8160){ s += vec_fn(reduce_add_epi32)(acc); acc = zero(); }
    }
    s += vec_fn(reduce_add_epi32)(acc);
    for(; i < t; ++i) s += (uint16_t)(xored ? d[i] ^ 0x8000 : d[i]);
    return s;
}
static inline int64_t q_sbase(const q_cctx *cx, int64_t k0){
    int x = cx->xored;
    return q_digsum(cx->a, cx->na, k0, x) - q_digsum(cx->a, cx->na, k0 - cx->nb, x)
         + q_digsum(cx->b, cx->nb, k0, x) - q_digsum(cx->b, cx->nb, k0 - cx->na, x);
}

// flat centered emit: data is already fully inverse-transformed into
// natural order (the final r22 / PFA butterfly ran as a separate mem
// pass). One sequential spectrum stream + the correction digit streams +
// one write stream + a single carry chain: prefetch-friendly where the
// fused emit's ~25 concurrent streams go latency-bound (this whole band
// is DRAM-bound).
static inline int pq16_emit_c_flat(uint64_t *rp, int64_t rl, const double *data,
                                   uint32_t n, const q_cctx *cx){
    q_chain ch; ch.prev_hi = zero(); ch.cin = 0;
    int64_t S = 0, kd = 0, rem = rl;
    // coefficients beyond na+nb digits are exactly zero: stop there
    const uint32_t tend = (uint32_t)(((uint64_t)(cx->na + cx->nb) + 31) / 32);
    const uint32_t tlim = tend < n / 16u ? tend : n / 16u;
    for(uint32_t t = 0; t < tlim; ++t){
        qcv o0 = q_ld(data + 32u * (size_t)t);
        qcv o1 = q_ld(data + 32u * (size_t)t + 16u);
        _vec e0, c0, e1, c1;
        q_corr_auto(&e0, &c0, cx, kd, &S);
        q_corr_auto(&e1, &c1, cx, kd + 16, &S);
        kd += 32;
        q_lohi q = q_pack2i(add(q_mround_s(o0.re), e0),
                            add(q_mround_s(o0.im), c0),
                            add(q_mround_s(o1.re), e1),
                            add(q_mround_s(o1.im), c1));
        store_vec(rp, q_chain_step(&ch, q), q_st_mask(rem));
        rp += 8; rem -= 8;
    }
    uint64_t hi7;
    memcpy(&hi7, (const char *)&ch.prev_hi + 56, 8);
    return hi7 + ch.cin == 0;
}

// ------------------------------------------------------------------
// PFA inverse output. Uncentered: fused with the emit -- M branch
// streams, inverse radix-M per tile, lane-shuffle into M natural-order
// region fronts (region b = natural complex [b*n, (b+1)*n)), one carry
// chain each. Region b's tile at step t takes butterfly output
// y[(b*n + 8t) mod M] lane-shifted by the residue masks.
// ------------------------------------------------------------------
// merge form (M = 3). Both sub-tiles of all fronts cannot stay in
// registers; the shuffled outputs stage through an L1 buffer.
__attribute__((always_inline))
static inline int q_pfa_emit_merge(uint64_t *rp, int64_t rl, double *data,
                                   uint32_t n, uint32_t M){
    uint8_t km[8]; q_kmasks(km, M);
    const double *br[7]; uint64_t *r[7]; uint32_t phi[7];
    int64_t rme[7];
    q_chain ch[7];
    const size_t limbs_b = (size_t)n / 2u;
    for(uint32_t b = 0; b < M; ++b){
        br[b]  = data + 2u * (size_t)n * b;
        r[b]   = rp + (size_t)b * limbs_b;
        rme[b] = rl - (int64_t)((size_t)b * limbs_b);
        phi[b] = (uint32_t)(((uint64_t)b * n) % M);
        ch[b].prev_hi = zero(); ch[b].cin = 0;
    }
    _Alignas(64) double stage[7 * 2 * 16];
    for(uint32_t t = 0; t < n / 8u; t += 2){
        for(int u = 0; u < 2; ++u){
            qcv x[7], y[7];
            for(uint32_t b = 0; b < M; ++b){
                x[b] = q_ld(br[b]);
                br[b] += 16;
            }
            q_pfa_bfly(x, y, M, 1);
            for(uint32_t b = 0; b < M; ++b){
                uint32_t ph = (phi[b] + (uint32_t)(8 * u)) % M;
                q_st(stage + 32u * b + 16u * (uint32_t)u, q_lane_merge(y, M, ph, km));
            }
        }
        for(uint32_t b = 0; b < M; ++b){
            q_lohi q = q_pack2(q_ld(stage + 32u * b), q_ld(stage + 32u * b + 16u));
            store_vec(r[b], q_chain_step(&ch[b], q), q_st_mask(rme[b]));
            r[b] += 8; rme[b] -= 8;
            phi[b] = (phi[b] + 16u) % M;
        }
    }
    return q_chains_close(rp, rl, ch, M, limbs_b);
}
// twiddle-fixup form (M = 5, 7): pre-rotate by conj(w^{b l}), inverse-DFT
// over relabeled branches; the region output is then a pure index relabel
// (rg[phi] rotates by 16 mod M, all vector slots stay constant).
__attribute__((always_inline))
static inline int q_pfa_emit_tw(uint64_t *rp, int64_t rl, double *data,
                                uint32_t n, uint32_t M, const double *wt){
    const double *br[7]; uint64_t *r[7];
    int64_t rme[7];
    q_chain ch[7];
    const size_t limbs_b = (size_t)n / 2u;
    for(uint32_t b = 0; b < M; ++b){
        br[b]  = data + 2u * (size_t)n * b;
        r[b]   = rp + (size_t)b * limbs_b;
        rme[b] = rl - (int64_t)((size_t)b * limbs_b);
        ch[b].prev_hi = zero(); ch[b].cin = 0;
    }
    uint32_t rg[7];
    for(uint32_t b = 0; b < M; ++b)
        rg[(uint32_t)(((uint64_t)b * n) % M)] = b;
    const uint32_t sh8 = 8u % M, sh16 = 16u % M;
    _Alignas(64) double stage[7 * 2 * 16];
    for(uint32_t t = 0; t < n / 8u; t += 2){
        for(int u = 0; u < 2; ++u){
            qcv x[7], y[7];
            for(uint32_t b = 0; b < M; ++b){
                qcv w = q_ld(wt + 16u * b);
                x[b] = q_mulc(q_ld(br[b]), w);
                br[b] += 16;
            }
            q_pfa_bfly(x, y, M, 1);
            for(uint32_t j = 0; j < M; ++j){
                uint32_t slot = u ? rg[(j + M - sh8) % M] : rg[j];
                q_st(stage + 32u * slot + 16u * (uint32_t)u, y[j]);
            }
        }
        for(uint32_t b = 0; b < M; ++b){
            q_lohi q = q_pack2(q_ld(stage + 32u * b), q_ld(stage + 32u * b + 16u));
            store_vec(r[b], q_chain_step(&ch[b], q), q_st_mask(rme[b]));
            r[b] += 8; rme[b] -= 8;
        }
        uint32_t tmp[7];
        for(uint32_t j = 0; j < M; ++j) tmp[(j + sh16) % M] = rg[j];
        for(uint32_t j = 0; j < M; ++j) rg[j] = tmp[j];
    }
    return q_chains_close(rp, rl, ch, M, limbs_b);
}
static inline int pq16_pfa_emit(uint64_t *rp, int64_t rl, double *data,
                                uint32_t n, uint32_t M, const pq16_plan *pl){
    switch(M){
    case 3:  return q_pfa_emit_merge(rp, rl, data, n, 3u);
    case 5:  return q_pfa_emit_tw(rp, rl, data, n, 5u, pl->wt5);
    default: return q_pfa_emit_tw(rp, rl, data, n, 7u, pl->wt7);
    }
}

// PFA inverse butterfly to natural order, no pack (centered band: the
// corrections then run in the prefetch-friendly flat emit). dst must not
// alias data -- the spare spectrum buffer serves.
__attribute__((always_inline))
static inline void q_pfa_natural_impl(double *dst, const double *data,
                                      uint32_t n, uint32_t M){
    uint8_t km[8]; q_kmasks(km, M);
    const double *br[7]; double *w[7]; uint32_t phi[7];
    for(uint32_t b = 0; b < M; ++b){
        br[b]  = data + 2u * (size_t)n * b;
        w[b]   = dst + 2u * (size_t)n * b;
        phi[b] = (uint32_t)(((uint64_t)b * n) % M);
    }
    for(uint32_t t = 0; t < n / 8u; ++t){
        qcv x[7], y[7];
        for(uint32_t b = 0; b < M; ++b){
            x[b] = q_ld(br[b]);
            br[b] += 16;
        }
        q_pfa_bfly(x, y, M, 1);
        for(uint32_t b = 0; b < M; ++b){
            q_st(w[b], q_lane_merge(y, M, phi[b], km));
            w[b] += 16;
            phi[b] = (phi[b] + 8u) % M;
        }
    }
}
static inline void pq16_pfa_natural(double *dst, const double *data,
                                    uint32_t n, uint32_t M){
    switch(M){
    case 3:  q_pfa_natural_impl(dst, data, n, 3u); break;
    case 5:  q_pfa_natural_impl(dst, data, n, 5u); break;
    default: q_pfa_natural_impl(dst, data, n, 7u); break;
    }
}

// ------------------------------------------------------------------
// size chooser + public entry
// ------------------------------------------------------------------
static SCRATCH_TLS pq16_plan pq16_tls_plan;

// smallest supported transform >= need: pow2 N or M*2^L (M = 3: branch >=
// 256; M = 5, 7: branch >= 128). Ties prefer the smaller M / pow2.
#ifndef PQ16_PFA7_MIN_BRANCH
#define PQ16_PFA7_MIN_BRANCH 128u
#endif
// largest supported transform (complex points); probed empirically with
// adversarial operands: pow2 and PFA-3/5 hold to 2^17 uncentered, the
// direct-form radix-7 butterfly's longer FMA chains break first at
// N = 7*2^14, so PFA-7 is precision-capped at 2^16 uncentered. The
// centered band's headroom admits everything to 2^19.
#ifndef PQ16_MAX_N
#define PQ16_MAX_N (1u << 17)
#endif
#ifndef PQ16_PFA7_MAX_N
#define PQ16_PFA7_MAX_N (1u << 16)
#endif
#ifndef PQ16_MAX_N_C
#define PQ16_MAX_N_C (1u << 19)
#endif
typedef struct { uint32_t nfull, branch, M; } pq16_size;

static inline pq16_size pq16_choose(uint32_t need, uint32_t maxn, int cen){
    uint32_t p2 = 512;
    while(p2 < need) p2 <<= 1;
    pq16_size best = { p2, p2, 1 };
    static const uint32_t Ms[3] = { 3, 5, 7 };
    for(int i = 0; i < 3; ++i){
        uint32_t M = Ms[i];
        uint32_t br = M == 3 ? 256u : 128u;
        while(M * br < need) br <<= 1;
        if(M == 7u && br < PQ16_PFA7_MIN_BRANCH) continue;
        if(M == 7u && !cen && M * br > PQ16_PFA7_MAX_N) continue;
        if(M * br < best.nfull && M * br <= maxn){
            best.nfull = M * br; best.branch = br; best.M = M;
        }
    }
    return best;
}

// rp[0 .. an+bn) = {ap, an} * {bp, bn}. Returns 1 on success, 0 if the
// size is outside the supported band. rp must not overlap the inputs.
// Workspace comes from sc; the twiddle tables persist thread-locally.
static inline int pq16_mul_r(uint64_t *rp,
                             const uint64_t *ap, ptrdiff_t an,
                             const uint64_t *bp, ptrdiff_t bn,
                             scratch *sc){
    if(an <= 0 || bn <= 0) return 0;
    uint64_t need = 2u * ((uint64_t)an + (uint64_t)bn);   // complex points
    const int cen = PQ16_FORCE_CENTERED || need > PQ16_MAX_N;
    const uint32_t maxn = cen ? PQ16_MAX_N_C : PQ16_MAX_N;
    if(need > maxn) return 0;                             // precision cap
    pq16_size fs = pq16_choose((uint32_t)need, maxn, cen);
    const uint32_t n = fs.branch, M = fs.M, nfull = fs.nfull;

    pq16_plan *pl = &pq16_tls_plan;
    pq16_plan_ensure(pl, n);

    SCRATCH(sc);
    // both spectra in one allocation, manually 128B-aligned (one full
    // tile; the stack tier guarantees only 64); the +32 doubles keep db
    // 128B-aligned while staggering it off an exact power-of-two stride
    // from da
    double *daw = SALLOC(sc, double, 4 * (size_t)nfull + 48);
    double *da = (double *)(((uintptr_t)daw + 127) & ~(uintptr_t)127);
    double *db = da + 2 * (size_t)nfull + 32;

    const int64_t rl = (int64_t)(an + bn);
    q_cctx cx = { (const uint16_t *)ap, (const uint16_t *)bp,
                  4 * (int64_t)an, 4 * (int64_t)bn, 0 };
    const uint64_t *sa = ap, *sb = bp;
    int64_t ca_cnt = an, cb_cnt = bn;
    uint64_t *cr = NULL;
    if(cen){
        // staging: centering XOR rides the copy; the copies also normalize
        // the memory layout (reading caller buffers in-place in this
        // BW-bound band costs up to 2x depending on allocation layout)
        const uint64_t XC = 0x8000800080008000ull;
        uint64_t *ca = SALLOC(sc, uint64_t, (size_t)nfull / 2u + 8);
        uint64_t *cb = SALLOC(sc, uint64_t, (size_t)nfull / 2u + 8);
        cr = SALLOC(sc, uint64_t, (size_t)rl + 8);
        for(int64_t i = 0; i < an; ++i) ca[i] = ap[i] ^ XC;
        memset(ca + an, 0, ((size_t)(nfull / 2u) - (size_t)an) * 8);
        for(int64_t i = 0; i < bn; ++i) cb[i] = bp[i] ^ XC;
        memset(cb + bn, 0, ((size_t)(nfull / 2u) - (size_t)bn) * 8);
        cx.a = (const uint16_t *)ca;
        cx.b = (const uint16_t *)cb;
        cx.xored = 1;
        sa = ca; sb = cb;
        ca_cnt = cb_cnt = (int64_t)(nfull / 2u);
    }

    // B first so A's spectrum is the cache-hot one when the pointwise
    // consumes and overwrites it; inverse leaves are fused into the
    // pointwise pass.
    if(M == 1){
        pq16_fwd(db, sb, cb_cnt, n, pl, cen);
        pq16_fwd(da, sa, ca_cnt, n, pl, cen);
        pq16_pointwise_ileaves(da, db, n, 1.0 / (double)n,
                               pq16_shape_of(n).leaf, pl);
        pq16_inv_r8_only(da, n, pl);
        if(cen){
            pq16_mem_ir22(da, n, pl->tw22[__builtin_ctz(n)]);
            if(!pq16_emit_c_flat(cr, rl, da, n, &cx)) return 0;
            memcpy(rp, cr, (size_t)rl * 8);
        }else{
            if(!pq16_inv_final_emit(rp, rl, da, n, pl)) return 0;
        }
    }else{
        pq16_pfa_fwd(db, sb, cb_cnt, n, M, pl, cen);
        pq16_pfa_fwd(da, sa, ca_cnt, n, M, pl, cen);
        const double sc_d = 1.0 / (double)nfull;
        const double *wrt = M == 3 ? PQ16_W3_RE : M == 5 ? PQ16_W5_RE : PQ16_W7_RE;
        const double *wit = M == 3 ? PQ16_W3_IM : M == 5 ? PQ16_W5_IM : PQ16_W7_IM;
        const uint32_t pwleaf = pq16_shape_of(n).leaf;
        // branch 0 self-pairs; branches (b, M-b) cross-pair with omega_M^b
        pq16_pointwise_ileaves(da, db, n, sc_d, pwleaf, pl);
        for(uint32_t b = 1; b <= M / 2u; ++b){
            uint32_t b2 = M - b;
            pq16_pointwise_cross_ileaves(
                da + 2u * (size_t)n * b,  da + 2u * (size_t)n * b2,
                db + 2u * (size_t)n * b,  db + 2u * (size_t)n * b2,
                n, sc_d, wrt[b], wit[b], pwleaf, pl);
        }
        const double *tw = pl->tw22[__builtin_ctz(n)];
        for(uint32_t b = 0; b < M; ++b){
            double *d = da + 2u * (size_t)n * b;
            pq16_inv_r8_only(d, n, pl);
            pq16_mem_ir22(d, n, tw);
        }
        if(cen){
            pq16_pfa_natural(db, da, n, M);
            if(!pq16_emit_c_flat(cr, rl, db, (uint32_t)(M * n), &cx)) return 0;
            memcpy(rp, cr, (size_t)rl * 8);
        }else{
            if(!pq16_pfa_emit(rp, rl, da, n, M, pl)) return 0;
        }
    }
    return 1;
}

static inline int pq16_mul(uint64_t *rp,
                           const uint64_t *ap, ptrdiff_t an,
                           const uint64_t *bp, ptrdiff_t bn){
    return pq16_mul_r(rp, ap, an, bp, bn, scratch_thread());
}
