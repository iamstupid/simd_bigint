#pragma once
#include "types.h"

#define canonize(vec, carry, sigcarry) {                         \
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
}

static inline void mul_u52_vec(pvec r, const _limb* a, _vec b, uint64_t n){
    _vec t[17];
    #define mulacc(ind) {\
        t[ind+8] = madd52lo(t[ind+8], b, splat_load(a, ind)); \
        t[ind+9] = zero(); \
        t[ind+9] = madd52hi(t[ind+9], b, splat_load(a, ind)); \
    }
    #define mulacc2(ind) {\
        t[ind+9] = madd52hi(t[ind+9], b, splat_load(a, ind)); \
        t[ind+8] = madd52lo(t[ind+8], b, splat_load(a, ind)); \
    }
    #define _clear(ind) t[ind] = zero()
    _clear(8); _clear(1); _clear(2); _clear(3);
    _clear(4); _clear(5); _clear(6); _clear(7);
    // we use t[0] as carry
    t[0] = zero();
    unsigned sigcarry = 0;
    for(;n>7;n-=8, a+=8){
        mulacc(0); mulacc(1); mulacc(2); mulacc(3);
        mulacc(4); mulacc(5); mulacc(6); mulacc(7);
        _vec k1, k2, k3, k4, k12, k34;
        k1 = add(t[8]                    , alignr64(t[9] , t[1], 7)); t[1]=t[9]; t[8] = t[16];
        k2 = add(alignr64(t[10], t[2], 6), alignr64(t[11], t[3], 5)); t[2]=t[10]; t[3] = t[11];
        k3 = add(alignr64(t[12], t[4], 4), alignr64(t[13], t[5], 3)); t[4]=t[12]; t[5] = t[13];
        k4 = add(alignr64(t[14], t[6], 2), alignr64(t[15], t[7], 1)); t[6]=t[14]; t[7] = t[15];
        k12 = add(k1, k2);
        k34 = add(k3, k4);
        k12 = add(k12, k34);
        canonize(k12, t[0], sigcarry);
        store_vec(r++, k12);
    }
    _vec k1=zero(), k2=zero(), k3=zero(), k4=zero(), k12=zero(), k34=zero();
    _vec t0=zero(), t1=zero();
    _clear(16); _clear(15); _clear(14); _clear(13);
    _clear(12); _clear(11); _clear(10); _clear(9);
    switch(n){
        // tail remaining
        case 7: mulacc2(6); k1 = add(alignr64(zero(), t[15], 1), t[16]);
        case 6: mulacc2(5);
        case 5: mulacc2(4); k2 = add(alignr64(zero(), t[13], 3), alignr64(zero(), t[14], 2)); k12 = add(k1, k2);
        case 4: mulacc2(3);
        case 3: mulacc2(2); k3 = add(alignr64(zero(), t[11], 5), alignr64(zero(), t[12], 4)); k12 = add(k12, k3);
        case 2: mulacc2(1);
        case 1: mulacc2(0); k4 = add(alignr64(zero(), t[ 9], 7), alignr64(zero(), t[10], 6)); t1 = add(k12, k4);
        case 0:
        k1 = add(t[8]                    , alignr64(t[9] , t[1], 7));
        k2 = add(alignr64(t[10], t[2], 6), alignr64(t[11], t[3], 5));
        k3 = add(alignr64(t[12], t[4], 4), alignr64(t[13], t[5], 3));
        k4 = add(alignr64(t[14], t[6], 2), alignr64(t[15], t[7], 1));
        k12 = add(k1, k2);
        k34 = add(k3, k4);
        t0 = add(k12, k34);
        canonize(t0, t[0], sigcarry);
        store_vec(r++, t0);
        if(n){
            canonize(t1, t[0], sigcarry);
            store_vec(r, t1);
        }
    }
    #undef mulacc
    #undef mulacc2
    #undef _clear
}
#undef canonize

// ---- fast-path canonicalize: common case (no lane >= MASK52 after the
// one-level his-absorb) needs no SWAR ripple and no &M -- just inject the
// incoming carry at lane0.  Cold path (a generate/propagate lane, ~2^-45 per
// block) falls back to the full SWAR.  Stays in the mask domain (no kmov<->GPR).
#define canonize_fast(vec, carry, sigcarry) {                         \
    const _vec M = MASK52;                                            \
    const _vec ONE = set1_64(1);                                      \
    _vec hi  = srli(vec, 52);                                         \
    _vec his = alignr64(hi, carry, 7);                                \
    _vec _ct = add(and_v(vec, M), his);                               \
    carry    = hi;                                                    \
    __mmask8 ov = geu(_ct, M);             /* any lane >= MASK (=p|g) */ \
    if(!ov){                                                          \
        vec = add(_ct, ONE, (__mmask8)(sigcarry), _ct); /* +carry@lane0 */ \
        sigcarry = 0;                                                 \
    } else {                                                          \
        unsigned g = (unsigned)gtu(_ct, M);                           \
        unsigned p = (unsigned)eq (_ct, M);                           \
        unsigned chain = p + ((g << 1) | sigcarry);                   \
        sigcarry = chain >> 8;                                        \
        __mmask8 cin = (__mmask8)(p ^ chain);                         \
        vec = and_v(add(_ct, ONE, cin, _ct), M);                      \
    }                                                                 \
}

static inline void mul_u52_vec_fast(pvec r, const _limb* a, _vec b, uint64_t n){
    _vec t[17];
    #define mulacc(ind) {\
        t[ind+8] = madd52lo(t[ind+8], b, splat_load(a, ind)); \
        t[ind+9] = zero(); \
        t[ind+9] = madd52hi(t[ind+9], b, splat_load(a, ind)); \
    }
    #define mulacc2(ind) {\
        t[ind+9] = madd52hi(t[ind+9], b, splat_load(a, ind)); \
        t[ind+8] = madd52lo(t[ind+8], b, splat_load(a, ind)); \
    }
    #define _clear(ind) t[ind] = zero()
    _clear(8); _clear(1); _clear(2); _clear(3);
    _clear(4); _clear(5); _clear(6); _clear(7);
    t[0] = zero();
    unsigned sigcarry = 0;
    for(;n>7;n-=8, a+=8){
        mulacc(0); mulacc(1); mulacc(2); mulacc(3);
        mulacc(4); mulacc(5); mulacc(6); mulacc(7);
        _vec k1, k2, k3, k4, k12, k34;
        k1 = add(t[8]                    , alignr64(t[9] , t[1], 7)); t[1]=t[9]; t[8] = t[16];
        k2 = add(alignr64(t[10], t[2], 6), alignr64(t[11], t[3], 5)); t[2]=t[10]; t[3] = t[11];
        k3 = add(alignr64(t[12], t[4], 4), alignr64(t[13], t[5], 3)); t[4]=t[12]; t[5] = t[13];
        k4 = add(alignr64(t[14], t[6], 2), alignr64(t[15], t[7], 1)); t[6]=t[14]; t[7] = t[15];
        k12 = add(k1, k2);
        k34 = add(k3, k4);
        k12 = add(k12, k34);
        canonize_fast(k12, t[0], sigcarry);
        store_vec(r++, k12);
    }
    _vec k1=zero(), k2=zero(), k3=zero(), k4=zero(), k12=zero(), k34=zero();
    _vec t0=zero(), t1=zero();
    _clear(16); _clear(15); _clear(14); _clear(13);
    _clear(12); _clear(11); _clear(10); _clear(9);
    switch(n){
        case 7: mulacc2(6); k1 = add(alignr64(zero(), t[15], 1), t[16]);
        case 6: mulacc2(5);
        case 5: mulacc2(4); k2 = add(alignr64(zero(), t[13], 3), alignr64(zero(), t[14], 2)); k12 = add(k1, k2);
        case 4: mulacc2(3);
        case 3: mulacc2(2); k3 = add(alignr64(zero(), t[11], 5), alignr64(zero(), t[12], 4)); k12 = add(k12, k3);
        case 2: mulacc2(1);
        case 1: mulacc2(0); k4 = add(alignr64(zero(), t[ 9], 7), alignr64(zero(), t[10], 6)); t1 = add(k12, k4);
        case 0:
        k1 = add(t[8]                    , alignr64(t[9] , t[1], 7));
        k2 = add(alignr64(t[10], t[2], 6), alignr64(t[11], t[3], 5));
        k3 = add(alignr64(t[12], t[4], 4), alignr64(t[13], t[5], 3));
        k4 = add(alignr64(t[14], t[6], 2), alignr64(t[15], t[7], 1));
        k12 = add(k1, k2);
        k34 = add(k3, k4);
        t0 = add(k12, k34);
        canonize_fast(t0, t[0], sigcarry);
        store_vec(r++, t0);
        if(n){
            canonize_fast(t1, t[0], sigcarry);
            store_vec(r, t1);
        }
    }
    #undef mulacc
    #undef mulacc2
    #undef _clear
}
#undef canonize_fast