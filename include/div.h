#pragma once
#include "types.h"
#include "mul_vec.h"
#include "recip_3by2_seed.h"

typedef struct _u832{ _vec lo, hi; } _u832;

#define INLINE static inline __attribute__((always_inline))

INLINE _u832 mul_zmm(const _limb* a, _vec b){
    _vec t[9] = {zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()};
#define mulacc(ind) {\
    t[ind+1] = madd52hi(t[ind+1], b, splat_load(a, ind)); \
    t[ind] = madd52lo(t[ind], b, splat_load(a, ind)); \
}
    mulacc(7); mulacc(6); mulacc(5); mulacc(4);
    mulacc(3); mulacc(2); mulacc(1); mulacc(0);
    _vec x, y;
    t[0] = add(t[0], alignr64(t[1], zero() , 7)); t[8] = add(t[8], alignr64(zero() , t[1], 7));
    x    = add(alignr64(t[2], zero() , 6), alignr64(t[3], zero() , 5));
    y    = add(alignr64(zero() , t[2], 6), alignr64(zero() , t[3], 5));
    t[0] = add(t[0], alignr64(t[4], zero() , 4)); t[8] = add(t[8], alignr64(zero() , t[4], 4));
    x    = add(x,    alignr64(t[5], zero() , 3)); y    = add(y,    alignr64(zero() , t[5], 3));
    t[0] = add(t[0], alignr64(t[6], zero() , 2)); t[8] = add(t[8], alignr64(zero() , t[6], 2));
    x    = add(x,    alignr64(t[7], zero() , 1)); y    = add(y,    alignr64(zero() , t[7], 1));
    return (_u832){.lo = add(x,t[0]), .hi = add(y,t[8])};
#undef mulacc
}

INLINE _u832 mul_zmm_2(const _limb* a, _vec b){
    // 2 zmm * zmm
    _vec t[17] = {zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero(), zero()};
#define mulacc(ind) {\
    t[ind+1] = madd52hi(t[ind+1], b, splat_load(a, ind)); \
    t[ind] = madd52lo(t[ind], b, splat_load(a, ind)); \
}
    mulacc(7+8); mulacc(6+8); mulacc(5+8); mulacc(4+8);
    mulacc(3+8); mulacc(2+8); mulacc(1+8); mulacc(0+8);
    mulacc(7); mulacc(6); mulacc(5); mulacc(4);
    mulacc(3); mulacc(2); mulacc(1); mulacc(0);
    _vec x, y;
    t[0] = add(t[0], alignr64(t[1], zero() , 7)); t[8] = add(t[8], alignr64(t[9] , t[1], 7));
    x    = add(alignr64(t[2], zero() , 6), alignr64(t[3], zero() , 5));
    y    = add(alignr64(t[10], t[2], 6), alignr64(t[11], t[3], 5));
    t[0] = add(t[0], alignr64(t[4], zero() , 4)); t[8] = add(t[8], alignr64(t[12] , t[4], 4));
    x    = add(x,    alignr64(t[5], zero() , 3)); y    = add(y,    alignr64(t[13] , t[5], 3));
    t[0] = add(t[0], alignr64(t[6], zero() , 2)); t[8] = add(t[8], alignr64(t[14] , t[6], 2));
    x    = add(x,    alignr64(t[7], zero() , 1)); y    = add(y,    alignr64(t[15] , t[7], 1));
    return (_u832){.lo = add(x,t[0]), .hi = add(y,t[8])};
#undef mulacc
}

INLINE _u832 load_832(cpvec t){
    return (_u832){.lo = load_vec(t), .hi = load_vec(t+1)};
}

INLINE _u832 add_zmm(_u832 a, _u832 b){
    return (_u832){.lo = add(a.lo, b.lo), .hi = add(a.hi, b.hi)};
}

// ===================== OurDiv3by2 (Edamatsu Alg. 3) =====================
// block = one zmm = 8 u52 lanes = base beta = 2^416.  All comparison-feeding
// values are kept canonical (lanes < 2^52).

// single redundant block -> canonical, threading vector carry + scalar SWAR carry
#define DIV_canon(vec, carry, sig) do {                                      \
    const _vec M = MASK52; const _vec ONE = set1_64(1);                      \
    _vec hi  = srli(vec, 52);                                                \
    _vec his = alignr64(hi, carry, 7);                                       \
    _vec _ct = add(and_v(vec, M), his);                                      \
    carry    = hi;                                                           \
    unsigned _p = (unsigned)eq (_ct, M);                                     \
    unsigned _g = (unsigned)gtu(_ct, M);                                     \
    unsigned _cf = (_g << 1) | (sig);                                        \
    if(!_p){                /* no propagate lane -> no cascade (common) */   \
        sig = _cf >> 8;                                                      \
        vec = and_v(add(_ct, ONE, (__mmask8)_cf, _ct), M);                   \
    } else {                /* rare cascade -> full SWAR ripple */           \
        unsigned _ch = _p + _cf;                                             \
        sig = _ch >> 8;                                                      \
        __mmask8 _ci = (__mmask8)(_p ^ _ch);                                 \
        vec = and_v(add(_ct, ONE, _ci, _ct), M);                            \
    }                                                                        \
} while(0)

// canonicalize redundant 2-block (lo,hi) -> canonical; *c2 = carry beyond beta^2
INLINE _u832 canon2(_u832 r, unsigned* c2){
    _vec carry = zero(); unsigned sig = 0;
    DIV_canon(r.lo, carry, sig);
    DIV_canon(r.hi, carry, sig);
    *c2 = sig | ((unsigned)(neq(carry, zero())) >> 7);
    return r;
}

// canonical block a + b + (*c) -> canonical block; *c = carry-out bit
INLINE _vec block_addc(_vec a, _vec b, unsigned* c){
    const _vec M = MASK52; const _vec ONE = set1_64(1);
    _vec res = add(a, b);                            // lanes < 2^53
    unsigned p = (unsigned)eq(res, M);
    unsigned g = (unsigned)gtu(res, M);
    unsigned cf = (g << 1) | (*c);
    if(!p){                                          // no propagate lane (common)
        *c = cf >> 8;
        return and_v(add(res, ONE, (__mmask8)cf, res), M);
    }
    unsigned ch = p + cf; *c = ch >> 8;
    __mmask8 ci = (__mmask8)(p ^ ch);
    return and_v(add(res, ONE, ci, res), M);        // +1 (mod 2^52) on carry lanes
}
// canonical block a - b - (*bw) mod beta -> canonical block; *bw = borrow-out bit
INLINE _vec block_subb(_vec a, _vec b, unsigned* bw){
    const _vec M = MASK52;
    _vec res = sub(a, b);                            // a-b lanewise (mod 2^64)
    unsigned p = (unsigned)eq(res, zero());          // ==0 propagates borrow
    unsigned g = (unsigned)gtu(res, M);              // underflow -> borrow generate
    unsigned bf = (g << 1) | (*bw);
    if(!p){                                          // no zero lane -> no cascade (common)
        *bw = bf >> 8;
        return and_v(add(res, M, (__mmask8)bf, res), M);
    }
    unsigned ch = p + bf; *bw = ch >> 8;
    __mmask8 bi = (__mmask8)(p ^ ch);
    return and_v(add(res, M, bi, res), M);           // -1 (mod 2^52) on borrow lanes
}
// canonical compares (a>=b)
INLINE int block_ge(_vec a, _vec b){
    uint8_t ne = (uint8_t)neq(a, b); if(!ne) return 1;
    uint8_t lt = (uint8_t)ltu(a, b); return (uint8_t)(ne ^ lt) > lt;
}
INLINE int block_ge2(_vec a1, _vec a0, _vec b1, _vec b0){
    uint8_t ne = (uint8_t)neq(a1, b1);
    if(ne){ uint8_t lt = (uint8_t)ltu(a1, b1); return (uint8_t)(ne ^ lt) > lt; }
    return block_ge(a0, b0);
}

typedef struct { _vec q1; _u832 r; } _div32;

// <u2,u1,u0> / <d1,d0>, v = 3/2 reciprocal of <d1,d0>.  Requires beta/2 <= d1 <
// beta and <u2,u1> < <d1,d0>.  Returns quotient block q1 and remainder <r1,r0>.
INLINE _div32 div3by2(const _limb* u, const _limb* d, _vec v){
    const _vec u0 = load_vec((cpvec)(u+0));
    const _vec u1 = load_vec((cpvec)(u+8));
    const _vec u2 = load_vec((cpvec)(u+16));
    const _vec d0 = load_vec((cpvec)(d+0));
    const _vec d1 = load_vec((cpvec)(d+8));

    // 1-2: <q1,q0> = v*u2 + <u2,u1>
    _u832 q = mul_zmm(u+16, v);          // q.lo=q0, q.hi=q1 (redundant)
    q.lo = add(q.lo, u1);
    q.hi = add(q.hi, u2);
    unsigned c2; q = canon2(q, &c2);
    _vec q0 = q.lo, q1 = q.hi;

    // 3: q1 = (q1+1) mod beta ; overflow if it wrapped or block-2 carry
    unsigned cinc = 1; q1 = block_addc(q1, zero(), &cinc);
    int overflow = (int)(c2 | cinc);

    _vec r0, r1;
    if(overflow){
        // 5-6: r1 = (u1 - d0) mod beta ; r0 = u0
        unsigned bb = 0; r1 = block_subb(u1, d0, &bb); r0 = u0;
    } else {
        // 8: <r1,r0> = (<u1,u0> - <d1,d0>*q1) mod beta^2
        _u832 dq = mul_zmm_2(d, q1); unsigned cc; dq = canon2(dq, &cc);
        unsigned bb = 0;
        r0 = block_subb(u0, dq.lo, &bb);
        r1 = block_subb(u1, dq.hi, &bb);
    }
    // 10: if r1 >= q0 : q1--, <r1,r0> += <d1,d0>
    if(block_ge(r1, q0)){
        unsigned bd = 1; q1 = block_subb(q1, zero(), &bd);
        unsigned cc = 0; r0 = block_addc(r0, d0, &cc); r1 = block_addc(r1, d1, &cc);
    }
    // 14: if <r1,r0> >= <d1,d0> : q1++, <r1,r0> -= <d1,d0>
    if(block_ge2(r1, r0, d1, d0)){
        unsigned ci = 1; q1 = block_addc(q1, zero(), &ci);
        unsigned bb = 0; r0 = block_subb(r0, d0, &bb); r1 = block_subb(r1, d1, &bb);
    }
    return (_div32){.q1 = q1, .r = (_u832){.lo = r0, .hi = r1}};
}
#undef DIV_canon

#undef INLINE