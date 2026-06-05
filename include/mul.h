#pragma once
#include "types.h"
#include "mul_basecase_le6.h"
#include "assert.h"
#include "scratch.h"
#include "canon.h"

static inline __mmask8 u52_limb_mask(uint64_t n) {
    const uint64_t r = n & 7;
    return r ? (__mmask8)((1u << r) - 1u) : (__mmask8)0xff;
}

static inline uint64_t u52_vec_count(uint64_t n) {
    return (n + 7) >> 3;
}

static inline pvec u52_pvec_add(pvec p, uint64_t limbs) {
    return (pvec)((uint64_t *)(void *)p + limbs);
}

static inline cpvec u52_cpvec_add(cpvec p, uint64_t limbs) {
    return (cpvec)((const uint64_t *)(const void *)p + limbs);
}

#ifndef MUL_U52_KARATSUBA_THRESHOLD
#define MUL_U52_KARATSUBA_THRESHOLD 88
#endif

// n0 : output accumulator (persistent across the chunk)
// n1 : single rotating diagonal reg; incoming offset ind+1, then reborn as offset ind
#define mul1_substep(a, ind) {                              \
    n1 = madd52hi(n1, (a), bx[ind]);                        \
    n0 = add(n0, alignr64(n1, m[ind], 7-(ind)));            \
    m[ind] = n1;                                            \
    n1 = madd52lo(zero(), (a), bx[ind]);                    \
}
#define mul1_substep0(a) {                                  \
    n1  = madd52hi(n1, (a), bx[0]);                         \
    n0  = add(n0, alignr64(n1, m[0], 7));                   \
    m[0] = n1;                                              \
    lo0 = madd52lo(zero(), (a), bx[0]);                     \
}

inline static void mul_u52_bn1(pvec r, cpvec a, uint64_t an, _vec b0){
    if(!an) return;
    _vec carry = zero();
    uint64_t c = an >> 3;
    const uint64_t tail = an & 7;
#pragma GCC unroll 4
    for(; c; --c){
        const _vec ax = load_vec(a++);
        _vec hi = madd52hi(zero(), ax, b0);
        store_vec(r++, add(madd52lo(zero(), ax, b0), alignr64(hi, carry, 7)));
        carry = hi;
    }
    if(tail){
        const _vec ax = load_vec(a, u52_limb_mask(tail));
        const _vec hi = madd52hi(zero(), ax, b0);
        const _vec lo = madd52lo(zero(), ax, b0);
        store_vec(r, add(lo, alignr64(hi, carry, 7)), u52_limb_mask(tail + 1));
    }else{
        store_vec(r, alignr64(zero(), carry, 7), u52_limb_mask(1));
    }
}

inline static void mul_u52_bn2(pvec r, cpvec a, uint64_t an, _vec b0, _vec b1){
    if(!an) return;
    _vec pMid = zero(), pH1 = zero();
    uint64_t c = an >> 3;
    const uint64_t tail = an & 7;

    #define MUL2_CHUNK(ax, dst) do {                       \
        _vec L0 = madd52lo(zero(), (ax), b0);              \
        _vec H0 = madd52hi(zero(), (ax), b0);              \
        _vec L1 = madd52lo(zero(), (ax), b1);              \
        _vec H1 = madd52hi(zero(), (ax), b1);              \
        _vec mid = add(H0, L1);                            \
        (dst) = add(L0, add(alignr64(mid, pMid, 7),        \
                            alignr64(H1,  pH1,  6)));      \
        pMid = mid; pH1 = H1;                              \
    } while(0)

#pragma GCC unroll 2
    for(; c; --c){
        const _vec ax = load_vec(a++);
        _vec o;
        MUL2_CHUNK(ax, o);
        store_vec(r++, o);
    }
    if(tail){
        const _vec ax = load_vec(a, u52_limb_mask(tail));
        _vec o;
        const uint64_t tail_limbs = tail + 2;
        MUL2_CHUNK(ax, o);
        if(tail_limbs <= 8){
            store_vec(r, o, u52_limb_mask(tail_limbs));
        }else{
            store_vec(r++, o);
            store_vec(r, add(alignr64(zero(), pMid, 7),
                             alignr64(zero(), pH1,  6)), u52_limb_mask(tail_limbs - 8));
        }
    }else{
        store_vec(r, add(alignr64(zero(), pMid, 7),
                         alignr64(zero(), pH1,  6)), u52_limb_mask(2));
    }
    #undef MUL2_CHUNK
}

inline static void mul_u52_1(pvec r, cpvec a, cpvec b, uint64_t an, const uint64_t bn){
    if(!an) return;
    if(bn == 1) return mul_u52_bn1(r, a, an, splat_load(b, 0));
    else if(bn == 2) return mul_u52_bn2(r, a, an, splat_load(b, 0), splat_load(b, 1));
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
    const uint64_t tail = an & 7;
    for(an = an >> 3; an; --an){
        ax = load_vec(a++);
        n0 = zero();
        n1 = zero();
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
    if(tail){
        ax = load_vec(a, u52_limb_mask(tail));
        n0 = zero();
        n1 = zero();
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
        const uint64_t tail_limbs = tail + bn;
        if(tail_limbs <= 8) store_vec(r, n0, u52_limb_mask(tail_limbs));
        else store_vec(r++, n0);
    }
    n0 = zero(); n1 = zero();
    switch(bn){
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
    if(tail){
        const uint64_t tail_limbs = tail + bn;
        if(tail_limbs > 8) store_vec(r, n0, u52_limb_mask(tail_limbs - 8));
    }else{
        store_vec(r, n0, u52_limb_mask(bn));
    }
}

#define mul_accum(a, b, ind) {\
    bx = splat_load(b, ind); \
    n[ind] = madd52lo(n[ind], a, bx); \
    n[ind+1] = madd52hi(n[ind+1], a, bx); \
}

#define apply_shift_n(ind) { \
    n[0]   = add(n[0], alignr64(n[ind], m[ind], 8 - (ind))); \
    m[ind] = n[ind]; \
    n[ind] = zero(); \
}

#define apply_shift_z(ind) { \
    n[0]   = add(n[0], alignr64(n[ind], zero(), 8 - (ind))); \
    m[ind] = n[ind]; \
    n[ind] = zero(); \
}

#define harvest { \
    n[0] = add(n[0], m[8]); m[8] = n[8]; n[8] = zero(); \
    apply_shift_n(7); apply_shift_n(6); apply_shift_n(5); apply_shift_n(4); \
    apply_shift_n(3); apply_shift_n(2); apply_shift_n(1); \
}
#define harvest_z { \
    m[8]=n[8]; n[8] = zero(); \
    apply_shift_z(7); apply_shift_z(6); apply_shift_z(5); apply_shift_z(4); \
    apply_shift_z(3); apply_shift_z(2); apply_shift_z(1); \
}

#define add_flush_tail { \
    n[0] = add(n[0], m[8]); \
    n[0] = add(n[0], alignr64(zero(), m[7], 1)); \
    n[0] = add(n[0], alignr64(zero(), m[6], 2)); \
    n[0] = add(n[0], alignr64(zero(), m[5], 3)); \
    n[0] = add(n[0], alignr64(zero(), m[4], 4)); \
    n[0] = add(n[0], alignr64(zero(), m[3], 5)); \
    n[0] = add(n[0], alignr64(zero(), m[2], 6)); \
    n[0] = add(n[0], alignr64(zero(), m[1], 7)); \
}

#define apply_shift_n_p(ind) { \
    n[ind] = lane_rotl(n[ind], ind);\
    const __mmask8 imask = (1 << ind) - 1;\
    n[0] = add(n[0], n[ind], ~imask, n[0]); \
    n[8] = add(n[8], n[ind], imask, n[8]); \
}

inline static void mul_u52_basecase(pvec r, cpvec a, cpvec b, uint64_t an, uint64_t bn){
    assert(an >= bn);
    if(bn <= 8) return mul_u52_1(r, a, b, an, bn);
    _vec n[9], m[9], bx, ax;
    cpvec bp, ap;
    n[0]=zero(); n[1]=zero(); n[2]=zero(); n[3]=zero(); n[4]=zero();
    n[5]=zero(); n[6]=zero(); n[7]=zero(); n[8]=zero();
    --an, --bn;
    const uint64_t au = an >> 3, bu = bn >> 3, su = au + bu;
    an &= 7, bn &= 7;
    uint64_t i, j;
    for(i = 0; i < su; ++i){
        if(i >= bu){
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
            j = i + 1; bp = b + i; ap = a;
            ax = load_vec(a);
            goto non_sat;
        }
        if(i) harvest else harvest_z;
        store_vec(r, n[0]); ++r;
        n[0] = zero();
    }
    const uint64_t an_tail = an, bn_tail = bn;
    const uint64_t tail_limbs = an_tail + bn_tail + 2;
    if(an_tail < bn_tail){
        bn = an_tail;
        bp = a + au;
        ax = load_vec(b + bu, u52_limb_mask(bn_tail + 1));
    }else{
        bp = b + bu;
        ax = load_vec(a + au, u52_limb_mask(an_tail + 1));
    }
    switch(bn){
        case 7: mul_accum(ax, bp, 7);
        case 6: mul_accum(ax, bp, 6); apply_shift_n_p(7);
        case 5: mul_accum(ax, bp, 5); apply_shift_n_p(6);
        case 4: mul_accum(ax, bp, 4); apply_shift_n_p(5);
        case 3: mul_accum(ax, bp, 3); apply_shift_n_p(4);
        case 2: mul_accum(ax, bp, 2); apply_shift_n_p(3);
        case 1: mul_accum(ax, bp, 1); apply_shift_n_p(2);
        case 0: mul_accum(ax, bp, 0); apply_shift_n_p(1);
    }
    add_flush_tail;
    if(tail_limbs <= 8){
        store_vec(r, n[0], u52_limb_mask(tail_limbs));
    }else{
        store_vec(r, n[0]);
        store_vec(r+1, n[8], u52_limb_mask(tail_limbs - 8));
    }
}

static inline int64_t u52_cmp_and_sub(pvec r, cpvec a, cpvec b, uint64_t na, uint64_t nb) {
    assert(na > 0);
    assert(nb > 0);
    const _vec mask52 = set1_64((1ull << 52) - 1);

    #define U52_RESOLVE_FULL(vr_in, br_in) do {                \
        _vec _vr = (vr_in);                                    \
        __mmask16 _z = eq(_vr, zero());                        \
        __mmask16 _c = lbr | ((__mmask16)(br_in) << 1);        \
        _c += _z;                                              \
        _z ^= _c;                                              \
        lbr = _c >> 8;                                         \
        _vr = and_v(add(_vr, mask52, _z, _vr), mask52);        \
        store_vec(r++, _vr);                                   \
    } while(0)
    #define U52_RESOLVE_MSK(vr_in, br_in, mask) do {           \
        _vec _vr = (vr_in);                                    \
        __mmask16 _z = eq(_vr, zero());                        \
        __mmask16 _c = lbr | ((__mmask16)(br_in) << 1);        \
        _c += _z;                                              \
        _z ^= _c;                                              \
        lbr = _c >> 8;                                         \
        _vr = and_v(add(_vr, mask52, _z, _vr), mask52);        \
        store_vec(r++, _vr, mask);                             \
    } while(0)

    uint8_t sig = 0;

    if(na < nb){
        cpvec t = a; a = b; b = t;
        uint64_t nt = na; na = nb; nb = nt;
        sig = 1;
    }
    if(!na){
        return 0;
    }
    uint8_t at = --na & 7, bt = --nb & 7;
    __mmask8 am = (1<<(at+1)) - 1, bm = (1<<(bt+1)) - 1;
    _vec ax, bx;
    na >>= 3, nb >>= 3;
    cpvec ap = a + na, bp = b + nb;

    for(ax = load_vec(ap, am); na > nb; --na, ax = load_vec(--ap)){
        if(neq(ax, zero())) goto compared;
        at = 7;
    }

    for(bx = load_vec(bp, bm); ~na; --na, --nb){
        const uint8_t neq_r = (uint8_t)neq(ax, bx);
        if(neq_r){
            const uint8_t lt_r = (uint8_t)ltu(ax, bx);
            const uint8_t gt_r = neq_r ^ lt_r;
            if(lt_r > gt_r){
                cpvec t = a; a = b; b = t;
                uint64_t nt = na; na = nb; nb = nt;
                uint8_t mt = at; at = bt; bt = mt;
                sig ^= 1;
            }
            goto compared;
        }
        if(!na) break;
        ax = load_vec(--ap);
        bx = load_vec(--bp);
        at = bt = 7;
    }
    return 0;
compared:
    {
        am = (1<<(at+1)) - 1, bm = (1<<(bt+1)) - 1;
        uint64_t i;
        uint8_t lbr = 0;
        for(i = 0; i < nb; ++i){
            ax = load_vec(a++), bx = load_vec(b++);
            _vec rx = sub(ax, bx);
            unsigned br = gtu(rx, mask52);
            U52_RESOLVE_FULL(rx, br);
        }
        if(na == nb){
            ax = load_vec(a, am), bx = load_vec(b, bm);
            _vec rx = sub(ax, bx);
            unsigned br = gtu(rx, mask52);
            U52_RESOLVE_MSK(rx, br, am);
        }else{
            ax = load_vec(a++), bx = load_vec(b, bm);
            _vec rx = sub(ax, bx);
            unsigned br = gtu(rx, mask52);
            U52_RESOLVE_FULL(rx, br);
            for(++i;i<na;++i){
                ax = load_vec(a++);
                U52_RESOLVE_FULL(ax, 0);
            }
            ax = load_vec(a, am);
            U52_RESOLVE_MSK(ax, 0, am);
        }
        #undef U52_RESOLVE_MSK
        #undef U52_RESOLVE_FULL
        return sig ? -(int64_t)(na*8+at+1) : (int64_t)(na*8+at+1);
    }
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

static inline void add_nc_52_tail_safe(pvec r, cpvec a, cpvec b, uint64_t n) {
    _vec r0, a0, b0;
    uint64_t cnt = n >> 3;
    const uint64_t tail = n & 7;
#pragma GCC unroll 4
    for(; cnt; --cnt){
        a0 = load_vec(a++);
        b0 = load_vec(b++);
        r0 = add(a0, b0);
        store_vec(r++, r0);
    }
    if(tail){
        const __mmask8 m = u52_limb_mask(tail);
        a0 = load_vec(a, m);
        b0 = load_vec(b, m);
        r0 = add(a0, b0);
        store_vec(r, r0, m);
    }
}

static inline void mul_u52_karatsuba(pvec r, cpvec a, cpvec b, uint64_t an, uint64_t bn, scratch *s) {
    assert(an >= bn); // contract: caller guarantees a is larger
    const uint64_t split_limbs = an >> 1;
    if(!split_limbs || bn <= split_limbs){
        return mul_u52_basecase(r, a, b, an, bn);
    }
    const uint64_t a1_len = an - split_limbs;
    const uint64_t b1_len = bn - split_limbs;
    const uint64_t eval_limbs = a1_len > split_limbs ? a1_len : split_limbs;
    cpvec a1 = u52_cpvec_add(a, split_limbs);
    cpvec b1 = u52_cpvec_add(b, split_limbs);
    SCRATCH(s);
    pvec point_n1  = SALLOC(s, _vec, u52_vec_count(eval_limbs * 2));
    pvec point_0   = r;
    pvec point_mid = u52_pvec_add(r, split_limbs);
    pvec point_inf = u52_pvec_add(r, split_limbs * 2);

    pvec a_n1 = point_0, b_n1 = u52_pvec_add(point_0, eval_limbs);
    // reuse point_0 space for n=-1 operands; this is before eval of point_0, so no overwrite risk
    uint8_t flag = 0;
    int64_t sa_len = u52_cmp_and_sub(a_n1, a, a1, split_limbs, a1_len);
    if(sa_len < 0){
        sa_len = -sa_len;
        flag = 1;
    }
    int64_t sb_len = u52_cmp_and_sub(b_n1, b, b1, split_limbs, b1_len);
    if(sb_len < 0){
        sb_len = -sb_len;
        flag ^= 1;
    }
    const uint64_t n1_len = (sa_len && sb_len) ? (uint64_t)sa_len + (uint64_t)sb_len : 0;
    if(n1_len){
        mul_u52_dispatch(point_n1, a_n1, b_n1, (uint64_t)sa_len, (uint64_t)sb_len, s);
    }
    mul_u52_dispatch(point_0, a, b, split_limbs, split_limbs, s);
    mul_u52_dispatch(point_inf, a1, b1, a1_len, b1_len, s);
    // point_n1 = point_0 + point_inf - mid
    // which means that mid = point_0 + point_inf - point_n1
    // note that mid is guaranteed >= 0, so the subtraction doesn't need compare
    add_nc_52(point_inf, point_mid, point_inf, split_limbs);
    add_nc_52(point_mid, point_inf, point_0, split_limbs);
    const uint64_t vinf_len = a1_len + b1_len;
    if(vinf_len > split_limbs){
        add_nc_52_tail_safe(point_inf, point_inf, u52_pvec_add(point_inf, split_limbs), vinf_len - split_limbs);
    }
    if(n1_len){
        if(flag){
            add_nc_52(point_mid, point_mid, point_n1, n1_len);
        }else{
            sub_nc_52(point_mid, point_mid, point_n1, n1_len);
        }
    }
}

static void mul_u52_dispatch(pvec r, cpvec a, cpvec b, uint64_t an, uint64_t bn, scratch *s) {
    if(!an || !bn) return;
    if(an < bn){
        uint64_t nt = an; an = bn; bn = nt;
        cpvec t = a; a = b; b = t;
    }
    if(bn >= MUL_U52_KARATSUBA_THRESHOLD){
        return mul_u52_karatsuba(r, a, b, an, bn, s);
    }
    return mul_u52_basecase(r, a, b, an, bn);
}
