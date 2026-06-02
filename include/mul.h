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

int64_t u52_cmp_and_sub(pvec r, cpvec a, cpvec b, uint64_t na, uint64_t nb) {
    const _vec mask52 = set1_64((1ull << 52) - 1);  // also "-1 mod 2^52" and the 52-bit mask
    __mmask16  lbr = 0;                              // borrow carried between vectors
    uint64_t   i = 0, lnz = 0;
    uint8_t    lnz_mask = 0;

    cpvec    ap, bp;     // ap = larger operand, bp = smaller
    uint64_t al, bl;     // last-vector index of each
    uint8_t  flag;       // 1 => true result a-b is negative (operands swapped)

    // remember the most-significant vector that came out nonzero
    #define TRACK(v) do {                                       \
        __mmask16 _tk_nz = neq((v), zero());                    \
        if (_tk_nz) { lnz = i; lnz_mask = _tk_nz; }             \
    } while (0)

    // resolve one vector of limbwise difference _vr with own-borrows _br:
    // ripple cross-lane / cross-vector borrow through lbr, normalize, store, track.
    #define RESOLVE(vr_in, br_in) do {                          \
        _vec      _rs_vr = (vr_in);                             \
        __mmask16 _rs_br = (br_in);                             \
        __mmask16 _rs_ms = eq(_rs_vr, zero()); /* zeros forward a borrow */ \
        lbr |= _rs_br << 1;                    /* borrow lands one lane up */ \
        lbr += _rs_ms;                         /* ripple across zero runs   */ \
        _rs_ms ^= lbr;                         /* lanes that take a -1       */ \
        lbr >>= 8;                             /* borrow-out -> next vector  */ \
        _rs_vr = and_v(add(_rs_vr, mask52, _rs_ms, _rs_vr), mask52); \
        store_vec(r++, _rs_vr);                                 \
        TRACK(_rs_vr);                                          \
    } while (0)

    // ---- orient: choose larger/smaller and the result sign ----
    --na, --nb;
    if (na != nb) {
        int abig = na > nb;
        ap = abig ? a : b;  bp = abig ? b : a;
        al = (abig ? na : nb) >> 3;
        bl = (abig ? nb : na) >> 3;
        flag = !abig;
    } else {
        int cmp = 0;
        cpvec pa = a + (na >> 3), pb = b + (nb >> 3);
        for (; pa >= a; --pa, --pb) {
            _vec va = load_vec(pa), vb = load_vec(pb);
            if (neq(va, vb)) { cmp = ltu(va, vb) ? -1 : 1; al = bl = (uint64_t)(pa - a); break; }
        }
        if (!cmp) return 0;                 // a == b
        int aGE = cmp > 0;
        ap = aGE ? a : b;  bp = aGE ? b : a;
        flag = !aGE;
    }

    // ---- subtract: overlap, then borrow tail, then plain copy ----
    for (;        i <= bl; ++i) { _vec vr = sub(load_vec(ap + i), load_vec(bp + i));
                                  RESOLVE(vr, ltu(mask52, vr)); }
    for (; lbr && i <= al; ++i)   RESOLVE(load_vec(ap + i), 0);
    for (;        i <= al; ++i) { _vec v = load_vec(ap + i); store_vec(r++, v); TRACK(v); }

    #undef RESOLVE
    #undef TRACK

    const int64_t len = (int64_t)lnz * 8 + (32 - __builtin_clz((uint32_t)lnz_mask));
    return flag ? -len : len;
}

