#pragma once
#include "types.h"
#define _MPN_CANON canon_u52_pos
#define canonize(vec, carry, sigcarry) do {                         \
    static const _vec MASK52 = set1_64((1ull<<52)-1) ;\
    _vec hi  = srli(vec, 52);                /* full per-lane overflow */ \
    _vec his = alignr64(hi, carry, 7);       /* hi[i-1]; lane0 <- carry[7] */ \
    _vec t   = add(and_v(vec, MASK52), his); /* low + carry-in, < 2^53     */ \
    carry    = hi;                                                  \
    unsigned g = (unsigned)gtu(t, MASK52);   /* generate : t >  MASK */     \
    unsigned p = (unsigned)eq (t, MASK52);   /* propagate: t == MASK */     \
    unsigned chain = p + ((g << 1) | sigcarry);  /* SWAR carry ripple   */  \
    sigcarry = chain >> 8;                   /* carry out of lane 7  */     \
    __mmask8 cin = (__mmask8)(p ^ chain);    /* lanes that received a carry */ \
    vec = and_v(sub(t, MASK52, cin, t), MASK52);                       \
} while(0)
#define _MPN_ADDSUB add
#define flex_add(r, a, b) r = add((a), (b))
#include "canon_impl"
#undef merge_carry_tail
#undef _MPN_CANON
#undef canonize
#undef _MPN_ADDSUB
#undef flex_add

#define _MPN_CANON canon_u52
#define canonize(vec, carry, sigcarry) do { \
    int64_t buffer[8];
    store_vec()
}while(0);
#define _MPN_ADDSUB sub
#define flex_add(r, a, b) r = sub((a), (b))
#include "canon_impl"
#undef merge_carry_tail
#undef _MPN_CANON
#undef canonize
#undef _MPN_ADDSUB
#undef flex_add