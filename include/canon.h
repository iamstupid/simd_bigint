#pragma once
#include "types.h"
#define canonize(vec, carry, sigcarry) do {                         \
    static const _vec MASK52 = set1_64((1ull<<52)-1) ;\
    _vec hi  = srli(vec, 52);                /* full per-lane overflow */ \
    _vec his = alignr64(hi, carry, 7);       /* hi[i-1]; lane0 <- carry[7] */ \
    _vec _canon_t = add(and_v(vec, MASK52), his); /* low + carry-in, < 2^53 */ \
    carry    = hi;                                                  \
    unsigned g = (unsigned)gtu(_canon_t, MASK52); /* generate : t >  MASK */ \
    unsigned p = (unsigned)eq (_canon_t, MASK52); /* propagate: t == MASK */ \
    unsigned chain = p + ((g << 1) | sigcarry);  /* SWAR carry ripple   */  \
    sigcarry = chain >> 8;                   /* carry out of lane 7  */     \
    __mmask8 cin = (__mmask8)(p ^ chain);    /* lanes that received a carry */ \
    vec = and_v(sub(_canon_t, MASK52, cin, _canon_t), MASK52);         \
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
    const _vec M = set1_64((1ll<<52)-1); \
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

/**
inline static _vec mpn_u52_sub_canon_simd(pvec r, cpvec a, cpvec b, uint64_t n, _vec carry){
    const _vec M   = set1_64((1ll<<52)-1);
    const _vec Z   = zero();
    _vec bw_v = slli(srli(carry, 63), 12), hi_v = and_v(set1_64(0xFFF), carry);          // cross-vector: bias-borrow and carry (lane 7 -> next lane 0)
    int  bin  = 0, cin  = 0;          // cross-vector: unit-borrow and unit-carry prefix nets
    for(n = (n+7)>>3; n; --n){
        _vec u  = sub(load_vec(a++), load_vec(b++));   // signed, in band
        _vec lo = and_v(u, M);                         // [0, 2^52)
        _vec hi = srli(u, 52);                         // [0, 2^12)  non-negative carry  (u>>>52)
        _vec bw = slli(srli(u, 63), 12);               // {0, 2^12}  borrow owed to next limb

        // ---- borrow-only prefix: w = lo - bw[i-1], ripple unit borrows ----
        _vec w   = sub(lo, alignr64(bw, bw_v, 7));  bw_v = bw;   // [-2^12, 2^52)
        unsigned gb = gtu(w, M);                       // w < 0   (generate: lo < bias borrow)
        unsigned pb = eq (w, Z);                       // w == 0  (propagate unit borrow)
        unsigned bc = pb + ((gb<<1) | (unsigned)bin);
        bin = (int)(bc >> 8);
        _vec rr = and_v(add(w, M, (__mmask8)(pb^bc), w), M);  // [0, 2^52)

        // ---- carry-only prefix: t = rr + hi[i-1], ripple unit carries ----
        _vec t   = add(rr, alignr64(hi, hi_v, 7));  hi_v = hi;   // [0, 2^52+2^12)
        unsigned gc = gtu(t, M);                       // t >= 2^52  (generate)
        unsigned pc = eq (t, M);                       // t == MASK  (propagate)
        unsigned cc = pc + ((gc<<1) | (unsigned)cin);
        cin = (int)(cc >> 8);
        _vec res = and_v(sub(t, M, (__mmask8)(pc^cc), t), M);   // +1 via -MASK

        store_vec(r++, res);
    }
    return add(sub(hi_v, bw_v), sub(set1_64(cin), set1_64(bin)));
}
*/
