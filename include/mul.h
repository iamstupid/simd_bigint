#pragma once
#include "types.h"
#include "mul_basecase_le6.h"
#include "assert.h"
#include <immintrin.h>
#include "scratch.h"

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
    assert(an >= bn); // contract: caller guarantees a is larger
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
        bn = an;
        bp = a + au;
        ax = load_vec(b + bu);
    }else{
        bp = b + bu;
        ax = load_vec(a + au);
    }
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
    if(an + bn >= 8) store_vec(r+1, n[8]);
    return;
}

static inline int64_t u52_cmp_and_sub(pvec r, cpvec a, cpvec b, uint64_t na, uint64_t nb) {
    assert(na >= nb); // contract: caller guarantees a is larger
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
    if (na > nb) {
        ap = a;  bp = b;
        al = na >> 3;
        bl = nb >> 3;
        flag = 0;
    } else {
        int cmp = 0;
        cpvec pa = a + (na >> 3), pb = b + (na >> 3);
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

static void mul_u52_dispatch(pvec r, cpvec a, cpvec b, uint64_t an, uint64_t bn, scratch *s);
// contract: caller guarantees a is larger

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

static inline void mul_u52_karatsuba(pvec r, cpvec a, cpvec b, uint64_t an, uint64_t bn, scratch *s) {
    assert(an >= bn); // contract: caller guarantees a is larger
    uint64_t split_len = an + 15 >> 4;
    SCRATCH(s);
    pvec point_n1  = SALLOC(s, _vec, split_len * 2);
    pvec point_0   = r;
    pvec point_inf = r + split_len * 2;

    an -= split_len * 8;
    bn -= split_len * 8;

    pvec a_n1 = point_0, b_n1 = point_0 + split_len;
    // reuse point_0 space for n=-1 operands; this is before eval of point_0, so no overwrite risk
    uint8_t flag = 0;
    int64_t sa_len = u52_cmp_and_sub(a_n1, a, a + split_len, split_len << 3, an);
    if(sa_len < 0){
        sa_len = -sa_len;
        flag = 1;
    }
    int64_t sb_len = u52_cmp_and_sub(b_n1, b, b + split_len, split_len << 3, bn);
    if(sb_len < 0){
        sb_len = -sb_len;
        flag ^= 1;
    }
    mul_u52_dispatch(point_n1, a_n1, b_n1, (uint64_t)sa_len, (uint64_t)sb_len, s);
    mul_u52_dispatch(point_0, a, b, split_len << 3, split_len << 3, s);
    mul_u52_dispatch(point_inf, a + split_len, b + split_len, an, bn, s);
    // point_n1 = point_0 + point_inf - mid
    // which means that mid = point_0 + point_inf - point_n1
    // note that mid is guaranteed >= 0, so the subtraction doesn't need compare
    add_nc_52(point_inf, point_0 + split_len, point_inf, split_len*8);
    add_nc_52(point_0 + split_len, point_inf, point_0, split_len*8);
    add_nc_52(point_inf, point_inf, point_inf+split_len, an + bn - split_len * 8);
    if(flag) add_nc_52(point_0 + split_len, point_0 + split_len, point_n1,sa_len + sb_len);
    else sub_nc_52(point_0 + split_len, point_0 + split_len, point_n1, sa_len + sb_len);
    // canonize
}