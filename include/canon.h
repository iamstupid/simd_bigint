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
    static const _vec M = MASK52;\
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
static inline _vec mpn_u52_sub_canon(pvec r, cpvec a, cpvec b, uint64_t n, _vec _c){
    canonneg_init(_c)
    for(n=(n+7)>>3;n;--n){
        _vec t = sub(load_vec(a++), load_vec(b++));
        canonneg(t);
        store_vec(r++, t);
    }
    return canonneg_done();
}

#undef canonize
#undef canon_pos
#undef canonneg_init
#undef canonneg_done
#undef canonneg
#undef canon_neg