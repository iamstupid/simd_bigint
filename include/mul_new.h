#pragma once
#include "types.h"
#include "assert.h"

#define mul_accum(a, b, ind) {\
    bx = splat_load(b, ind); \
    n[ind] = madd52lo(n[ind], a, bx); \
    n[ind+1] = madd52hi(n[ind+1], a, bx); \
}

#define apply_shift_n(ind) { \
    bx = alignr64(n[ind], m[ind], 8-(ind));\
    n[0] = add(n[0], bx); \
    m[ind] = n[ind]; \
}

// n0 : output accumulator (persistent across the chunk)
// n1 : single rotating diagonal reg — incoming offset ind+1, then reborn as offset ind
#define mul1_substep(a, ind) {                              \
    n1 = madd52hi(n1, (a), bx[ind]);   /* offset ind+1 := L(ind+1)+H(ind) */ \
    n0 = add(n0, alignr64(n1, m[ind], 7-(ind)));            \
    m[ind] = n1;                                            \
    n1 = madd52lo(zero(), (a), bx[ind]); /* offset ind := L(ind), feeds next */ \
}
#define mul1_substep0(a) {                                  \
    n1  = madd52hi(n1, (a), bx[0]);                         \
    n0  = add(n0, alignr64(n1, m[0], 7));                   \
    m[0] = n1;                                              \
    lo0 = madd52lo(zero(), (a), bx[0]);  /* off the add chain */ \
}

#define mul1_harvest(a, ind)

inline static void mul_u52_1(pvec r, cpvec a, cpvec b, uint64_t an, const uint64_t bn){
    _vec n0, n1, lo0, ax, m[8], bx[8];
    m[0]=m[1]=m[2]=m[3]=m[4]=m[5]=m[6]=m[7]=zero();
    if(bn > 4){
        bx[7] = splat_load(b, 7);
        bx[6] = splat_load(b, 6);
        bx[5] = splat_load(b, 5);
        bx[4] = splat_load(b, 4);
    }
    bx[3] = splat_load(b, 3);
    bx[2] = splat_load(b, 2);
    bx[1] = splat_load(b, 1);
    bx[0] = splat_load(b, 0);
    for(an = (an + 7) >> 3; an; --an){
        ax = load_vec(a++);
        n0 = zero();
        n1 = zero();                 // top diagonal (offset bn) seed
        switch(bn){
            case 8: mul1_substep (ax, 7);
            case 7: mul1_substep (ax, 6);
            case 6: mul1_substep (ax, 5);
            case 5: mul1_substep (ax, 4);
            case 4: mul1_substep (ax, 3);
            case 3: mul1_substep (ax, 2);
            case 2: mul1_substep (ax, 1);
            case 1: mul1_substep0(ax);
        }
        n0 = add(n0, lo0);
        store_vec(r++, n0);
    }
    n0 = zero(); n1 = zero();
    switch(bn){               // even ind -> n0, odd ind -> n1
        case 8: n0 = add(n0, alignr64(zero(), m[7], 0));
        case 7: n1 = add(n1, alignr64(zero(), m[6], 1));
        case 6: n0 = add(n0, alignr64(zero(), m[5], 2));
        case 5: n1 = add(n1, alignr64(zero(), m[4], 3));
        case 4: n0 = add(n0, alignr64(zero(), m[3], 4));
        case 3: n1 = add(n1, alignr64(zero(), m[2], 5));
        case 2: n0 = add(n0, alignr64(zero(), m[1], 6));
        case 1: n1 = add(n1, alignr64(zero(), m[0], 7));
    }
    n0 = add(n0, n1);
    store_vec(r++, n0);
}

inline static void mul_u52_bc_new(pvec r, cpvec a, cpvec b, uint64_t an, uint64_t bn){
    assert(an >= bn);
    /** Left to be implemented */
}