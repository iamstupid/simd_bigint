#pragma once
#include "types.h"

#define carry_prop_t(add, sub, prp, r, a, b, carry, ...) { \
    _vec res = add(a, b); \
    __mmask16 _prop = eq(res, prp); \
    __mmask16 _carr = gtu(res, MASK52); \
    carry |= _carr << 1; \
    carry += _prop; \
    _prop ^= carry; \
    carry >>=8; \
    res = and_v(sub(res, MASK52, _prop, res), MASK52); \
    store_vec(r++, res, ##__VA_ARGS__); \
}

#define carry_prop(r, a, b, carry, ...) carry_prop_t(add, sub, MASK52, r, a, b, carry, ##__VA_ARGS__)
#define borrow_prop(r, a, b, carry, ...) carry_prop_t(sub, add, zero(), r, a, b, carry, ##__VA_ARGS__)

#define canonize(vec, carry, sigcarry) do {                         \
    const _vec M = MASK52;\
    _vec hi  = srli(vec, 52);                /* full per-lane overflow */ \
    _vec his = alignr64(hi, carry, 7);       /* hi[i-1]; lane0 <- carry[7] */ \
    _vec _canon_t = add(and_v(vec, M), his); /* low + carry-in, < 2^53 */ \
    carry    = hi;                                                  \
    unsigned g = (unsigned)gtu(_canon_t, M); /* generate : t >  MASK */ \
    unsigned p = (unsigned)eq (_canon_t, M); /* propagate: t == MASK */ \
    unsigned chain = p + ((g << 1) | sigcarry);  /* SWAR carry ripple   */  \
    sigcarry = chain >> 8;                   /* carry out of lane 7  */     \
    __mmask8 cin = (__mmask8)(p ^ chain);    /* lanes that received a carry */ \
    vec = and_v(sub(_canon_t, M, cin, _canon_t), M);         \
} while(0)

#define canon_pos(suf, pre, ret, ...) \
CAT(mpn_canon_pos, suf)(pvec r, cpvec a, uint64_t n, ##__VA_ARGS__){ \
    int sig = 0; pre \
    for(n = (n+7)>>3; n; --n){ \
        _vec t = load_vec(a++); \
        canonize(t, _c, sig); \
        store_vec(r++, t); \
    } \
    ret \
}

static inline void canon_pos(EMPTY,_vec _c = zero(); , EMPTY)
static inline _vec canon_pos(_c,EMPTY,return add(_c, set1_64(sig));,_vec _c)
static inline _vec mpn_u52_add_canon(pvec r, cpvec a, cpvec b, uint64_t n, _vec _c){
    int sig = 0;
    for(n=(n+7)>>3;n;--n){
        _vec t = add(load_vec(a++), load_vec(b++));
        canonize(t, _c, sig);
        store_vec(r++, t);
    }
    return add(_c, set1_64(sig));
}

#define canonneg_init(carry) \
    const _vec M = MASK52; \
    const _vec Z = zero(); \
    _vec bw_v = slli(srli((carry), 63), 12); \
    _vec hi_v = and_v(set1_64(0xFFF), (carry)); \
    int bin = 0, cin = 0;

#define canonneg_done() \
    add(sub(hi_v, bw_v), sub(set1_64(cin), set1_64(bin)))

#define canonneg(vec) do { \
    _vec cn_u = (vec); \
    _vec cn_lo = and_v(cn_u, M); \
    _vec cn_hi = srli(cn_u, 52); \
    _vec cn_bw = slli(srli(cn_u, 63), 12); \
    _vec cn_w = sub(cn_lo, alignr64(cn_bw, bw_v, 7)); \
    bw_v = cn_bw; \
    unsigned cn_gb = (unsigned)gtu(cn_w, M); \
    unsigned cn_pb = (unsigned)eq(cn_w, Z); \
    unsigned cn_bc = cn_pb + ((cn_gb << 1) | (unsigned)bin); \
    bin = (int)(cn_bc >> 8); \
    _vec cn_rr = and_v(add(cn_w, M, (__mmask8)(cn_pb ^ cn_bc), cn_w), M); \
    _vec cn_t = add(cn_rr, alignr64(cn_hi, hi_v, 7)); \
    hi_v = cn_hi; \
    unsigned cn_gc = (unsigned)gtu(cn_t, M); \
    unsigned cn_pc = (unsigned)eq(cn_t, M); \
    unsigned cn_cc = cn_pc + ((cn_gc << 1) | (unsigned)cin); \
    cin = (int)(cn_cc >> 8); \
    (vec) = and_v(sub(cn_t, M, (__mmask8)(cn_pc ^ cn_cc), cn_t), M); \
} while(0)

#define canon_neg(suf, pre, ret, ...) \
CAT(mpn_canon_neg, suf)(pvec r, cpvec a, uint64_t n, ##__VA_ARGS__){ \
    pre \
    for(n = (n+7)>>3; n; --n){ \
        _vec t = load_vec(a++); \
        canonneg(t); \
        store_vec(r++, t); \
    } \
    ret \
}

static inline void canon_neg(EMPTY,canonneg_init(zero()), EMPTY)
static inline _vec canon_neg(_c,canonneg_init(_c),return canonneg_done();,_vec _c)
// last-vector load mask for an n-limb region (0xff when n is a multiple of 8)
#define u52_rsh1_tail_mask(n) ((__mmask8)(((n) & 7) ? (1u << ((n) & 7)) - 1 : 0xffu))

// Tail-masked signed canonicalization: exactly n limbs are read and written
// (the final vector's loads/stores are masked). For a value that fits n limbs
// the carry out of the region is genuinely zero, so nothing is dropped --
// unlike the vec-rounded mpn_canon_neg, which must not be used when live
// foreign data sits past the region (it would absorb those lanes into the
// chain and drop the resulting boundary carry).
static inline void mpn_canon_neg_tail(pvec r, cpvec a, uint64_t n){
    canonneg_init(zero())
    uint64_t cnt = (n + 7) >> 3;
    const __mmask8 lm = u52_rsh1_tail_mask(n);
    for(; cnt > 1; --cnt){
        _vec t = load_vec(a++);
        canonneg(t);
        store_vec(r++, t);
    }
    _vec t = load_vec(a, lm);
    canonneg(t);
    store_vec(r, t, lm);
}

static inline _vec mpn_u52_sub_canon(pvec r, cpvec a, cpvec b, uint64_t n, _vec _c){
    canonneg_init(_c)
    for(n=(n+7)>>3;n;--n){
        _vec t = sub(load_vec(a++), load_vec(b++));
        canonneg(t);
        store_vec(r++, t);
    }
    return canonneg_done();
}

// canonneg folds: resolve a region's redundant signed lanes IN the pass that
// applies its final operand, instead of a separate canonicalization sweep.
// r[0,n) = canonneg(r (op) x[0,xlen)), with x zero past xlen and both tails
// masked. The folded value must be non-negative (interpolation invariants),
// so no borrow leaves the region. With every mixed region finished by a fold
// and untouched regions holding child lanes, a composite output's lane bound
// equals its children's -- depth-independent with no standalone canon pass.
#define u52_fold_x(i) \
    ((int64_t)(xlen - (i) * 8) >= 8 ? load_vec(x + (i)) : \
     ((int64_t)(xlen - (i) * 8) > 0 ? load_vec(x + (i), (__mmask8)((1u << (xlen - (i) * 8)) - 1)) : zero()))
#define u52_canonneg_fold_impl(OP) { \
    canonneg_init(zero()) \
    const uint64_t cnt = (n + 7) >> 3; \
    const __mmask8 lm = u52_rsh1_tail_mask(n); \
    uint64_t i = 0; \
    for(; i + 1 < cnt; ++i){ \
        _vec t = OP(load_vec(r + i), u52_fold_x(i)); \
        canonneg(t); \
        store_vec(r + i, t); \
    } \
    _vec t = OP(load_vec(r + i, lm), u52_fold_x(i)); \
    canonneg(t); \
    store_vec(r + i, t, lm); \
}
static inline void mpn_u52_fold_add_canon(pvec r, cpvec x, uint64_t xlen, uint64_t n){
    u52_canonneg_fold_impl(add)
}
static inline void mpn_u52_fold_sub_canon(pvec r, cpvec x, uint64_t xlen, uint64_t n){
    u52_canonneg_fold_impl(sub)
}
#undef u52_canonneg_fold_impl

// r[0,n) = canonneg(r + x - y); x masked past xlen, y past ylen. NOTE: like
// every canonneg fold, the span must reach the top of the number (the chain's
// final borrow/carry must be genuinely zero, which only the whole-value
// non-negativity argument guarantees) -- a fold that stops mid-number drops a
// live boundary carry.
#define u52_fold_y(i) \
    ((int64_t)(ylen - (i) * 8) >= 8 ? load_vec(y + (i)) : \
     ((int64_t)(ylen - (i) * 8) > 0 ? load_vec(y + (i), (__mmask8)((1u << (ylen - (i) * 8)) - 1)) : zero()))
static inline void mpn_u52_fold_addsub_canon(pvec r, cpvec x, uint64_t xlen,
                                             cpvec y, uint64_t ylen, uint64_t n){
    canonneg_init(zero())
    const uint64_t cnt = (n + 7) >> 3;
    const __mmask8 lm = u52_rsh1_tail_mask(n);
    uint64_t i = 0;
    for(; i + 1 < cnt; ++i){
        _vec t = sub(add(load_vec(r + i), u52_fold_x(i)), u52_fold_y(i));
        canonneg(t);
        store_vec(r + i, t);
    }
    _vec t = sub(add(load_vec(r + i, lm), u52_fold_x(i)), u52_fold_y(i));
    canonneg(t);
    store_vec(r + i, t, lm);
}

// Exact halving of a canonical vector: lane i gets bit 0 of the limb above it
// (lane 0 of `nx` supplies the limb above lane 7 of `lo`). Two codegen
// variants: funnel shift (VBMI2 vpshrdq) vs shift/shift/ternlog.
#ifdef U52_RSH1_USE_FUNNEL
#define u52_rsh1_pair(lo, nx) \
    and_v(fshr64i(slli((lo), 12), alignr64((nx), (lo), 1), 13), MASK52)
#else
// (lo>>1 | xs<<51) & MASK52 as one ternlog: f(a,b,c) = (a|b)&c = 0xA8
#define u52_rsh1_pair(lo, nx) \
    ternary(srli((lo), 1), slli(alignr64((nx), (lo), 1), 51), MASK52, 0xA8)
#endif


// r = (a + b) >> 1, canonical. a/b may be redundant (any 64-bit lanes whose
// lanewise sum does not wrap). The value a+b must be even, and n must cover
// the full result so the carry out of the region (the return value) is zero.
// Source loads are tail-masked (lanes past n may be live foreign data); the
// destination is written in full vectors and must be vector-padded.
static inline _vec mpn_u52_add_rsh1_canon(pvec r, cpvec a, cpvec b, uint64_t n, _vec _c){
    int sig = 0;
    uint64_t cnt = (n + 7) >> 3;
    const __mmask8 lm = u52_rsh1_tail_mask(n);
    _vec prev, t;
    if(cnt == 1){
        t = add(load_vec(a, lm), load_vec(b, lm));
        canonize(t, _c, sig);
        store_vec(r, u52_rsh1_pair(t, zero()));
        return add(_c, set1_64(sig));
    }
    t = add(load_vec(a++), load_vec(b++));
    canonize(t, _c, sig);
    for(prev = t, --cnt; --cnt; prev = t){
        t = add(load_vec(a++), load_vec(b++));
        canonize(t, _c, sig);
        store_vec(r++, u52_rsh1_pair(prev, t));
    }
    t = add(load_vec(a, lm), load_vec(b, lm));
    canonize(t, _c, sig);
    store_vec(r++, u52_rsh1_pair(prev, t));
    store_vec(r, u52_rsh1_pair(t, zero()));
    return add(_c, set1_64(sig));
}

// r = (a - b) >> 1, canonical. a/b may be redundant; the value a-b must be
// non-negative and even (lanewise differences may wrap; canonneg resolves).
// Same tail-masked-load / full-vector-store contract as the add variant.
static inline _vec mpn_u52_sub_rsh1_canon(pvec r, cpvec a, cpvec b, uint64_t n, _vec _c){
    canonneg_init(_c)
    uint64_t cnt = (n + 7) >> 3;
    const __mmask8 lm = u52_rsh1_tail_mask(n);
    _vec prev, t;
    if(cnt == 1){
        t = sub(load_vec(a, lm), load_vec(b, lm));
        canonneg(t);
        store_vec(r, u52_rsh1_pair(t, zero()));
        return canonneg_done();
    }
    t = sub(load_vec(a++), load_vec(b++));
    canonneg(t);
    for(prev = t, --cnt; --cnt; prev = t){
        t = sub(load_vec(a++), load_vec(b++));
        canonneg(t);
        store_vec(r++, u52_rsh1_pair(prev, t));
    }
    t = sub(load_vec(a, lm), load_vec(b, lm));
    canonneg(t);
    store_vec(r++, u52_rsh1_pair(prev, t));
    store_vec(r, u52_rsh1_pair(t, zero()));
    return canonneg_done();
}

#define u52_d3_ax(v) /* lane residue mod 3, <= 78 */                           \
    add(vec_fn(popcnt_epi64)(and_v((v), set1_64(0x5555555555555ull))),         \
        add(vec_fn(popcnt_epi64)(and_v((v), set1_64(0xAAAAAAAAAAAAAull))),     \
            vec_fn(popcnt_epi64)(and_v((v), set1_64(0xAAAAAAAAAAAAAull)))))
#define u52_d3_prefix(P) { /* inclusive lane prefix sum, in place */           \
    P = add(P, lane_lsh(P, 1));                                                \
    P = add(P, lane_lsh(P, 2));                                                \
    P = add(P, lane_lsh(P, 4));                                                \
}
#define u52_d3_mod3(x) /* exact on [0, 8191]: q = (x*2731)>>13 */              \
    sub((x), add(add(srli(vec_fn(mullo_epi64)((x), set1_64(2731)), 13),        \
                     srli(vec_fn(mullo_epi64)((x), set1_64(2731)), 13)),       \
                 srli(vec_fn(mullo_epi64)((x), set1_64(2731)), 13)))
#define u52_d3_quot(v, P, ax, vd) /* lo52((v - borrow) * 3^-1) */              \
    madd52lo(zero(), and_v(sub((v), u52_d3_mod3(add(sub((P), (ax)),            \
                                                    sub((P), (ax))))), MASK52), (vd))

// ---- fused canonicalize + encode --------------------------------------

// pack52: canonical digit vector -> 52 packed bytes (reference pack52_vec;
// byte index 7 of digit 0 is zero for canonical digits and acts as filler)
alignas(64) static const uint8_t u52_enc_rperm[64] = {
     0,  1,  2,  3,  4,  5,  6,  7,  9, 10, 11, 12, 13,
    14, 15,  7, 19, 20, 21, 22, 23,  7,  7,  7, 28, 29,
    30, 31,  7,  7,  7,  7, 38,  7,  7,  7,  7,  7,  7,
    48, 49, 50, 51, 52, 53, 54, 55,  7, 58, 59, 60, 61,
    62, 63,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
};
alignas(64) static const uint8_t u52_enc_lperm[64] = {
     7,  7,  7,  7,  7,  7,  8,  9,  7,  7,  7,  7,  7,
    16, 17, 18,  7,  7,  7, 24, 25, 26, 27, 28,  7,  7,
    32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46,
     7,  7,  7,  7,  7,  7, 56, 57, 58,  7,  7,  7,  7,
     7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,
};

#define u52_pack52(d) \
    or_v(srlv(permb(load_vec((cpvec)u52_enc_rperm), (d)), setr_64(0, 4, 0, 4, 0, 0, 4, 0)), \
         sllv(permb(load_vec((cpvec)u52_enc_lperm), (d)), setr_64(4, 0, 4, 0, 4, 4, 0, 0)))

// signed-encode variant for a class-1 (signed redundant) top-level product:
// the dual-chain canonneg rides the packing pass, absorbing what would
// otherwise be the top level's fold -- the largest single canon span.
static inline void u64_from_u52_canonneg(uint64_t *rp, cpvec a, uint64_t out_limbs){
    canonneg_init(zero())
    uint8_t *p = (uint8_t *)rp;
    int64_t rem = (int64_t)(out_limbs * 8);
    for(; rem >= 52; rem -= 52, p += 52, ++a){
        _vec t = load_vec(a);
        canonneg(t);
        vec_fn(mask_storeu_epi8)(p, (1ull << 52) - 1, u52_pack52(t));
    }
    if(rem > 0){
        _vec t = load_vec(a);
        canonneg(t);
        vec_fn(mask_storeu_epi8)(p, ~0ull >> (64 - rem), u52_pack52(t));
    }
}

// canonicalize the NON-NEGATIVE redundant u52 product {a, vector-padded} and
// pack it into out_limbs u64 limbs at rp in one pass. Composite muls return
// canonical lanes already (canonize is then a cheap no-op pass); the basecase
// returns non-negative redundant lanes, which the positive canonize chain
// resolves. Only the peeled final block uses a masked store.
static inline void u64_from_u52_canon(uint64_t *rp, cpvec a, uint64_t out_limbs){
    int sig = 0;
    _vec _c = zero();
    uint8_t *p = (uint8_t *)rp;
    int64_t rem = (int64_t)(out_limbs * 8);
    for(; rem >= 52; rem -= 52, p += 52, ++a){
        _vec t = load_vec(a);
        canonize(t, _c, sig);
        vec_fn(mask_storeu_epi8)(p, (1ull << 52) - 1, u52_pack52(t));
    }
    if(rem > 0){
        _vec t = load_vec(a);
        canonize(t, _c, sig);
        vec_fn(mask_storeu_epi8)(p, ~0ull >> (64 - rem), u52_pack52(t));
    }
}

// r = p0 + 2*p1 + 4*p2 (+ 8*p3), composed lanewise (12-bit headroom: at most
// 15*(2^52-1) < 2^56) and canonicalized in the same pass; writes exactly n+1
// limbs (top <= 6 resp. 14). p0..p2 have n limbs, the top part s <= n;
// p3 == NULL selects the 3-part form (call sites pass a literal, so the
// branches fold after inlining). Masked loads/stores touch only the boundary
// vectors: full-speed segments while the top part is live, then without it.
static inline void mpn_u52_eval2_canon(pvec r, cpvec p0, cpvec p1, cpvec p2, cpvec p3,
                                       uint64_t n, uint64_t s){
    int sig = 0;
    _vec _c = zero();
    cpvec top = p3 ? p3 : p2;
    const uint64_t vtot = (n + 8) >> 3;          // vec_count(n + 1)
    uint64_t i = 0;
    for(; (i + 1) * 8 <= s; ++i){                // every operand full
        _vec t = add(load_vec(p0 + i), slli(load_vec(p1 + i), 1));
        if(p3) t = add(t, slli(load_vec(p2 + i), 2));
        t = add(t, slli(load_vec(top + i), p3 ? 3 : 2));
        canonize(t, _c, sig);
        store_vec(r + i, t);
    }
    if((s & 7) && (i + 1) * 8 <= n){             // top part's boundary vector
        const __mmask8 ms = (__mmask8)((1u << (s & 7)) - 1);
        _vec t = add(load_vec(p0 + i), slli(load_vec(p1 + i), 1));
        if(p3) t = add(t, slli(load_vec(p2 + i), 2));
        t = add(t, slli(load_vec(top + i, ms), p3 ? 3 : 2));
        canonize(t, _c, sig);
        store_vec(r + i, t);
        ++i;
    }
    for(; (i + 1) * 8 <= n; ++i){                // top exhausted
        _vec t = add(load_vec(p0 + i), slli(load_vec(p1 + i), 1));
        if(p3) t = add(t, slli(load_vec(p2 + i), 2));
        canonize(t, _c, sig);
        store_vec(r + i, t);
    }
    for(; i < vtot; ++i){                        // n boundary + the carry limb
        const int64_t rn = (int64_t)n - (int64_t)(i * 8);
        const int64_t rs = (int64_t)s - (int64_t)(i * 8);
        const int64_t rr = (int64_t)(n + 1) - (int64_t)(i * 8);
        const __mmask8 mn = rn > 0 ? (__mmask8)((1u << rn) - 1) : 0;
        const __mmask8 ms = rs > 0 ? (__mmask8)((1u << rs) - 1) : 0;
        const __mmask8 mr = rr >= 8 ? (__mmask8)0xff : (__mmask8)((1u << rr) - 1);
        _vec t = add(load_vec(p0 + i, mn), slli(load_vec(p1 + i, mn), 1));
        if(p3) t = add(t, slli(load_vec(p2 + i, mn), 2));
        t = add(t, slli(load_vec(top + i, ms), p3 ? 3 : 2));
        canonize(t, _c, sig);
        store_vec(r + i, t, mr);
    }
}

#undef canonize
#undef canon_pos
#undef canonneg_init
#undef canonneg_done
#undef canonneg
#undef canon_neg