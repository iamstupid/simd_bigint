#pragma once
#include "types.h"
#include "mul_basecase_le6.h"
#include <immintrin.h>

#define mul_accum(a, b, ind) {\
    bx = splat_load(b, ind); \
    n[ind] = madd52lo(n[ind], a, bx); \
    n[ind+1] = madd52hi(n[ind+1], a, bx); \
}

#define clean_up(n) { \
    n[0] = n[8]; \
    n[1] = zero(), n[2] = zero(), n[3] = zero(), n[4] = zero(); \
    n[5] = zero(), n[6] = zero(), n[7] = zero(), n[8] = zero(); \
}

#define apply_shift_n(ind) { \
    n[ind] = lane_rotl(n[ind], ind);\
    const __mmask8 imask = (1 << ind) - 1;\
    n[0] = add(n[0], n[ind], ~imask, n[0]); \
    n[8] = add(n[8], n[ind], imask, n[8]); \
    n[ind] = zero(); \
}

#define harvest {\
    apply_shift_n(7); \
    apply_shift_n(6); \
    apply_shift_n(5); \
    apply_shift_n(4); \
    apply_shift_n(3); \
    apply_shift_n(2); \
    apply_shift_n(1); \
}

inline static void mul_u52_basecase(pvec r, cpvec a, cpvec b, uint64_t an, uint64_t bn){
    if(an < bn){
        cpvec t = a; a = b; b = t;
        uint64_t tn = an; an = bn; bn = tn;
    }
    _vec n[9], bx, ax;
    cpvec bp, ap;
    n[0] = zero();
    n[1] = zero(), n[2] = zero(), n[3] = zero(), n[4] = zero();
    n[5] = zero(), n[6] = zero(), n[7] = zero(), n[8] = zero();
    --an, --bn;
    const uint64_t au = an >> 3, bu = bn >> 3, su = au + bu;
    an &= 7, bn &= 7;
    uint64_t i, j;
    for(i = 0; i < su; ++i){
        if(i >= bu){
            // saturating b
            bp = b + bu; ap = a + (i - bu);
            j = i >= au ? su - i : bu + 1;
            ax = load_vec(ap);
            switch(bn){
                non_sat:
                for(; j; ++ap, --bp, --j){
                    ax = load_vec(ap);
                    case 7: mul_accum(ax, bp, 7);
                    case 6: mul_accum(ax, bp, 6);
                    case 5: mul_accum(ax, bp, 5);
                    case 4: mul_accum(ax, bp, 4);
                    case 3: mul_accum(ax, bp, 3);
                    case 2: mul_accum(ax, bp, 2);
                    case 1: mul_accum(ax, bp, 1);
                    case 0: mul_accum(ax, bp, 0);
                }
            }
            if(i >= au){
                ax = load_vec(bp);
                switch(an){
                    case 7: mul_accum(ax, ap, 7);
                    case 6: mul_accum(ax, ap, 6);
                    case 5: mul_accum(ax, ap, 5);
                    case 4: mul_accum(ax, ap, 4);
                    case 3: mul_accum(ax, ap, 3);
                    case 2: mul_accum(ax, ap, 2);
                    case 1: mul_accum(ax, ap, 1);
                    case 0: mul_accum(ax, ap, 0);
                }
            }
        }else{
            // non-saturating
            j = i+1;
            bp = b + i; ap = a;
            ax = load_vec(a);
            goto non_sat;
        }
        harvest;
        store_vec(r, n[0]); ++r;
        n[0] = n[8]; n[8] = zero();
    }
    if(an < bn){
        if(bn >= 4){ // a tunable tail threshold;
            bn = an;
            bp = a + au;
            ax = load_vec(b + bu);
            goto ctail;
        }
    }else{
        if(an >= 4){
            bp = b + bu;
            ax = load_vec(a + au);
            ctail:
            switch(bn){
                case 7: mul_accum(ax, bp, 7);
                case 6: mul_accum(ax, bp, 6); apply_shift_n(7);
                case 5: mul_accum(ax, bp, 5); apply_shift_n(6);
                case 4: mul_accum(ax, bp, 4); apply_shift_n(5);
                case 3: mul_accum(ax, bp, 3); apply_shift_n(4);
                case 2: mul_accum(ax, bp, 2); apply_shift_n(3);
                case 1: mul_accum(ax, bp, 1); apply_shift_n(2);
                case 0: mul_accum(ax, bp, 0); apply_shift_n(1);
            }
            store_vec(r, n[0]);
            store_vec(r+1, n[8]);
        }
    }
    store_vec(r, n[0]);
    return;
}

int64_t u52_cmp_and_sub(pvec r, cpvec a, cpvec b, uint64_t na, uint64_t nb){
    // do a - b; r = b - a if b > a, and there we return - |b - a|.
    // a and b needs to be canonical; otherwise it is difficult to compare.
    uint8_t flag = 0;
    if(na < nb){
        flag = 1;
        cpvec t = a; a = b; b = t;
        uint64_t nt = na; na = nb; nb = nt;
        goto subs;
    }

    subs:
    
}