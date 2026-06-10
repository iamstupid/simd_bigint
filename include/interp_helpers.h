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
    uint8_t at = --na & 7, bt = --nb & 7;
    __mmask8 am = (__mmask8)((1u<<(at+1)) - 1), bm = (__mmask8)((1u<<(bt+1)) - 1);
    na >>= 3; nb >>= 3;
    cpvec ap = a + na, bp = b + nb;
    _vec ax, bx;

    /* skip a's vectors that sit above b's range (compared against zero) */
    for(ax = load_vec(ap, am); na > nb; --na, ax = load_vec(--ap)){
        if(neq(ax, zero())) goto compared;   /* a strictly longer -> a > b */
        at = 7;
    }
    /* same vector index now: walk down comparing a vs b */
    for(bx = load_vec(bp, bm); ; --na, --nb){
        const uint8_t ne = (uint8_t)neq(ax, bx);
        if(ne){
            const uint8_t lt = (uint8_t)ltu(ax, bx);
            /* lt/gt lane-masks are disjoint, so lt>gt iff the top differing lane has a<b */
            if(lt > (uint8_t)(ne ^ lt)){
                SWAP(cpvec,a,b); SWAP(uint64_t,na,nb); SWAP(uint8_t,at,bt); sig ^= 1;
            }
            goto compared;
        }
        if(!na) return 0;                     /* every lane equal -> a == b */
        ax = load_vec(--ap); bx = load_vec(--bp); at = bt = 7;
    }
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

// r = a + b, canonical (each limb < 2^52). Add is commutative so operands are
// swapped to put the longer first. The carry out of limb (max-1) lands in limb
// `max`: when extra_limb is set it is written (result occupies max+1 limbs, the
// top one possibly 0) and max+1 is returned; otherwise it is dropped and max is
// returned. extra_limb is a runtime bool, but inlined at a constant call site it
// folds away, so callers in the "always carried" regime pay no branch.
static inline uint64_t u52_add_carry(pvec r, cpvec a, cpvec b, uint64_t na, uint64_t nb, bool extra_limb){
    assert(na > 0 && nb > 0);
    if(na < nb){ cpvec t = a; a = b; b = t; uint64_t n = na; na = nb; nb = n; }

    const uint8_t  at = (uint8_t)((na - 1) & 7), bt = (uint8_t)((nb - 1) & 7);
    const __mmask8 am = (__mmask8)((1u << (at + 1)) - 1);
    const __mmask8 bm = (__mmask8)((1u << (bt + 1)) - 1);
    const uint64_t nav = (na - 1) >> 3, nbv = (nb - 1) >> 3;
    uint16_t carry = 0;
    _vec ax, bx;
    uint64_t i;

    // full a+b vectors below b's top vector
    for(i = 0; i < nbv; ++i){
        ax = load_vec(a++), bx = load_vec(b++);
        carry_prop(r, ax, bx, carry);
    }
    // reach a's top vector, with bx = the matching b lanes (zero beyond b)
    if(nav == nbv){
        ax = load_vec(a, am), bx = load_vec(b, bm);
    }else{
        ax = load_vec(a++), bx = load_vec(b, bm);   // vector nbv: a full, b partial
        carry_prop(r, ax, bx, carry);
        for(++i; i < nav; ++i){                      // a-only full vectors
            ax = load_vec(a++);
            carry_prop(r, ax, zero(), carry);
        }
        ax = load_vec(a, am), bx = zero();           // a's top vector
    }

    // top vector: store a's lanes, plus the carry limb if requested
    if(extra_limb && at != 7){
        // carry-out terminates in lane at+1 (a zero lane); widen the store to keep it
        const __mmask8 sm = (__mmask8)((1u << (at + 2)) - 1);
        carry_prop(r, ax, bx, carry, sm);
        return na + 1;
    }
    carry_prop(r, ax, bx, carry, am);
    if(extra_limb){                                  // at == 7: carry spills into a new limb
        store_vec(r, set1_64((uint64_t)(carry & 1)), (__mmask8)0x01);
        return na + 1;
    }
    return na;
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

// r = a - 2*b, lanewise non-canonical (toom interpolation step v2 -= 2*vinf)
#define flex_add(r, a, b) r = sub((a), slli((b), 1))
#define _MPN_ADDSUB_N sublsh1_nc_52
#include "addsub_nc_impl"
#undef flex_add
#undef _MPN_ADDSUB_N

// w = (x -+ y) >> 1, canonical out; x/y may be redundant. Toom interpolation's
// signed-vm1 halving step: y is a magnitude, so pass sub_y = (y's sign is negative).
static inline void u52_addsub_rsh1_canon(pvec r, cpvec x, cpvec y, uint64_t n, bool sub_y){
    if(sub_y) mpn_u52_sub_rsh1_canon(r, x, y, n, zero());
    else      mpn_u52_add_rsh1_canon(r, x, y, n, zero());
}

// Exact halving of a REDUNDANT signed value, staying redundant: per lane,
// r_i = sra(d_i, 1) + ((d_{i+1} & 1) << 51), where d = x -+ y lanewise. This
// is a pure bit identity (2*(d>>1) = d - (d&1), and an even value forces lane
// 0's parity bit to 0), so unlike the *_rsh1_canon kernels there is NO carry
// resolution at all: no canonize SWAR, no mask<->GPR traffic. Use whenever the
// consumer tolerates redundant lanes (only divexact3 demands canonical input).
// Source loads are tail-masked; destination written in full vectors (padded).
#define u52_rsh1_nc_pair(lo, nx) \
    add(srai((lo), 1), slli(and_v(alignr64((nx), (lo), 1), set1_64(1)), 51))
#define u52_addsub_rsh1_nc_impl(OP) { \
    uint64_t cnt = (n + 7) >> 3; \
    const __mmask8 lm = u52_limb_mask(n); \
    _vec prev, t; \
    if(cnt == 1){ \
        t = OP(load_vec(x, lm), load_vec(y, lm)); \
        store_vec(r, u52_rsh1_nc_pair(t, zero())); \
        return; \
    } \
    t = OP(load_vec(x++), load_vec(y++)); \
    for(prev = t, --cnt; --cnt; prev = t){ \
        t = OP(load_vec(x++), load_vec(y++)); \
        store_vec(r++, u52_rsh1_nc_pair(prev, t)); \
    } \
    t = OP(load_vec(x, lm), load_vec(y, lm)); \
    store_vec(r++, u52_rsh1_nc_pair(prev, t)); \
    store_vec(r, u52_rsh1_nc_pair(t, zero())); \
}
static inline void mpn_u52_add_rsh1_nc(pvec r, cpvec x, cpvec y, uint64_t n){
    u52_addsub_rsh1_nc_impl(add)
}
static inline void mpn_u52_sub_rsh1_nc(pvec r, cpvec x, cpvec y, uint64_t n){
    u52_addsub_rsh1_nc_impl(sub)
}
#undef u52_addsub_rsh1_nc_impl
static inline void u52_addsub_rsh1_nc(pvec r, cpvec x, cpvec y, uint64_t n, bool sub_y){
    if(sub_y) mpn_u52_sub_rsh1_nc(r, x, y, n);
    else      mpn_u52_add_rsh1_nc(r, x, y, n);
}

// t2 step fused: r = (x - y + z) >> 1, redundant signed; z has nz <= n limbs
// (lanes past nz contribute zero). Same halving identity as above.
static inline void mpn_u52_subadd_rsh1_nc(pvec r, cpvec x, cpvec y, cpvec z,
                                          uint64_t n, uint64_t nz){
    uint64_t cnt = (n + 7) >> 3;
    const __mmask8 lm = u52_limb_mask(n);
    _vec prev, t;
    uint64_t i = 0;
#define u52_t2_term(MX, MZ) \
    add(sub(load_vec(x + i, MX), load_vec(y + i, MX)), \
        ((int64_t)(nz - i * 8) >= 8 ? load_vec(z + i) : \
         ((int64_t)(nz - i * 8) > 0 ? load_vec(z + i, (__mmask8)((1u << (nz - i * 8)) - 1)) : zero())) MZ)
    if(cnt == 1){
        t = u52_t2_term(lm, );
        store_vec(r, u52_rsh1_nc_pair(t, zero()));
        return;
    }
    t = u52_t2_term(0xff, );
    for(prev = t, ++i, --cnt; --cnt; prev = t, ++i){
        t = u52_t2_term(0xff, );
        store_vec(r++, u52_rsh1_nc_pair(prev, t));
    }
    t = u52_t2_term(lm, );
    store_vec(r++, u52_rsh1_nc_pair(prev, t));
    store_vec(r, u52_rsh1_nc_pair(t, zero()));
#undef u52_t2_term
}

// r = x - y - z lanewise non-canonical (interpolation v1 <- v1 - v0 - tm1)
static inline void u52_sub3_nc(pvec r, cpvec x, cpvec y, cpvec z, uint64_t n){
    uint64_t cnt = n >> 3;
    const uint64_t tail = n & 7;
    for(; cnt; --cnt)
        store_vec(r++, sub(sub(load_vec(x++), load_vec(y++)), load_vec(z++)));
    if(tail){
        const __mmask8 m = (__mmask8)((1u << tail) - 1);
        store_vec(r, sub(sub(load_vec(x, m), load_vec(y, m)), load_vec(z, m)), m);
    }
}

// r += x - y lanewise non-canonical (fused c1 placement: pp+n += tm1 - c3)
static inline void u52_addsub3_nc(pvec r, cpvec x, cpvec y, uint64_t n){
    uint64_t cnt = n >> 3;
    const uint64_t tail = n & 7;
    for(; cnt; --cnt, ++r)
        store_vec(r, add(load_vec((cpvec)r), sub(load_vec(x++), load_vec(y++))));
    if(tail){
        const __mmask8 m = (__mmask8)((1u << tail) - 1);
        store_vec(r, add(load_vec((cpvec)r, m), sub(load_vec(x, m), load_vec(y, m))), m);
    }
}

// In-place capable exact division by 3 of a canonical value (hand-pipelined
// port of divexact3_SIMD_ref.cpp). Carry-free within a vector: per-lane
// residues via popcount (2^i = (-1)^i mod 3), borrow chain as a lane prefix
// sum, quotient by IFMA multiply with 3^-1 mod 2^52. The inter-vector residue
// carry never leaves the vector domain: blocks chain through a lane-7 splat,
// reduced by a vectorized mod3 once per block (the reference extracts to a
// GPR and does %3 there). Main loop is 4x unrolled; the tail runs vector-at-
// a-time with a masked final load/store, so short and odd lengths stay on the
// fast path (the reference fell back to a scalar loop).
static inline void mpn_u52_divexact3(pvec r, cpvec a, uint64_t n){
    const _vec vd = set1_64(3002399751580331ull);    // 3^-1 mod 2^52
    const _vec seven = set1_64(7);
    _vec cv = zero();                                // splatted carry residue, {0,1,2}
    uint64_t cnt = n >> 3;
    const uint64_t tail = n & 7;
    for(; cnt >= 4; cnt -= 4, a += 4, r += 4){
        const _vec v0 = load_vec(a), v1 = load_vec(a + 1);
        const _vec v2 = load_vec(a + 2), v3 = load_vec(a + 3);
        const _vec x0 = u52_d3_ax(v0), x1 = u52_d3_ax(v1);
        const _vec x2 = u52_d3_ax(v2), x3 = u52_d3_ax(v3);
        _vec p0 = x0, p1 = x1, p2 = x2, p3 = x3;
        u52_d3_prefix(p0); u52_d3_prefix(p1); u52_d3_prefix(p2); u52_d3_prefix(p3);
        p0 = add(p0, cv);                            // <= 624 + 2
        p1 = add(p1, perm64(seven, p0));             // chain via lane-7 splat
        p2 = add(p2, perm64(seven, p1));
        p3 = add(p3, perm64(seven, p2));             // <= 4*624 + 2 < 8192
        cv = u52_d3_mod3(perm64(seven, p3));
        store_vec(r,     u52_d3_quot(v0, p0, x0, vd));
        store_vec(r + 1, u52_d3_quot(v1, p1, x1, vd));
        store_vec(r + 2, u52_d3_quot(v2, p2, x2, vd));
        store_vec(r + 3, u52_d3_quot(v3, p3, x3, vd));
    }
    for(; cnt; --cnt, ++a, ++r){
        const _vec v = load_vec(a);
        const _vec ax = u52_d3_ax(v);
        _vec P = ax;
        u52_d3_prefix(P);
        P = add(P, cv);
        cv = u52_d3_mod3(perm64(seven, P));
        store_vec(r, u52_d3_quot(v, P, ax, vd));
    }
    if(tail){
        const __mmask8 m = (__mmask8)((1u << tail) - 1);
        const _vec v = load_vec(a, m);
        const _vec ax = u52_d3_ax(v);
        _vec P = ax;
        u52_d3_prefix(P);
        P = add(P, cv);
        store_vec(r, u52_d3_quot(v, P, ax, vd), m);
    }
}

