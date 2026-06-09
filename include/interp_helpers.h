#pragma once
#include "types.h"
#include "assert.h"
#include "scratch.h"
#include "canon.h"
#include <cstdint>

static inline __mmask8 u52_limb_mask(uint64_t n) {
    const uint64_t r = n & 7;
    return r ? (__mmask8)((1u << r) - 1u) : (__mmask8)0xff;
}

static inline int64_t u52_cmp_and_sub(pvec r, cpvec a, cpvec b, uint64_t na, uint64_t nb) {
    assert(na > 0 && nb > 0);
    uint8_t sig = 0;
    if(na < nb){
        SWAP(cpvec, a, b);
        SWAP(uint64_t, na, nb);
        sig = 1;
    }
    if(!na) return 0;
#include "cmp_head"
    return 0;
compared:
    am = (1<<(at+1)) - 1, bm = (1<<(bt+1)) - 1;
    uint64_t i;
    uint16_t lbr = 0;
    for(i = 0; i < nb; ++i){
        ax = load_vec(a++), bx = load_vec(b++);
        borrow_prop(r, ax, bx, lbr);
    }
    if(na == nb){
        ax = load_vec(a, am), bx = load_vec(b, bm);
        borrow_prop(r, ax, bx, lbr, am);
    }else{
        ax = load_vec(a++), bx = load_vec(b, bm);
        borrow_prop(r, ax, bx, lbr);
        for(++i;i<na;++i){
            ax = load_vec(a++);
            borrow_prop(r, ax, zero(), lbr);
        }
        ax = load_vec(a, am);
        borrow_prop(r, ax, zero(), lbr, am);
    }
    return sig ? -(int64_t)(na*8+at+1) : (int64_t)(na*8+at+1);
}


#define flex_add(r, a, b) r = add((a), (b))
#define _MPN_ADDSUB_N add_nc_52
#include "addsub_nc_impl"
#undef flex_add
#undef _MPN_ADDSUB_N

#define flex_add(r, a, b) r = sub((a), (b))
#define _MPN_ADDSUB_N sub_nc_52
#include "addsub_nc_impl"
#undef flex_add
#undef _MPN_ADDSUB_N

