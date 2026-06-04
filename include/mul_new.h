#pragma once
#include "types.h"
#include "assert.h"
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

inline static void mul_u52_bn1(pvec r, cpvec a, uint64_t an, _vec b0){
    _vec carry = zero();
    uint64_t c = (an + 7) >> 3;
    if(c&1){
        _vec ax = load_vec(a++);
        _vec hi = madd52hi(zero(), ax, b0);
        store_vec(r++, add(madd52lo(zero(), ax, b0), alignr64(hi, carry, 7)));
        carry = hi;
    }
    if(c&2){
        _vec a0=load_vec(a+0), a1=load_vec(a+1);
        a += 2;
        _vec h0=madd52hi(zero(),a0,b0), h1=madd52hi(zero(),a1,b0);  // all independent -> fill p0
        _vec l0=madd52lo(zero(),a0,b0), l1=madd52lo(zero(),a1,b0);
        store_vec(r+0, add(l0, alignr64(h0, carry, 7)));   // [carry7, h0_0..6]
        store_vec(r+1, add(l1, alignr64(h1, h0,    7)));
        r += 2; carry = h1;
    }
    for(c>>=2; c; --c){
        _vec a0=load_vec(a+0), a1=load_vec(a+1), a2=load_vec(a+2), a3=load_vec(a+3);
        a += 4;
        _vec h0=madd52hi(zero(),a0,b0), h1=madd52hi(zero(),a1,b0),
             h2=madd52hi(zero(),a2,b0), h3=madd52hi(zero(),a3,b0);  // all independent -> fill p0
        _vec l0=madd52lo(zero(),a0,b0), l1=madd52lo(zero(),a1,b0),
             l2=madd52lo(zero(),a2,b0), l3=madd52lo(zero(),a3,b0);
        store_vec(r+0, add(l0, alignr64(h0, carry, 7)));   // [carry7, h0_0..6]
        store_vec(r+1, add(l1, alignr64(h1, h0,    7)));
        store_vec(r+2, add(l2, alignr64(h2, h1,    7)));
        store_vec(r+3, add(l3, alignr64(h3, h2,    7)));
        r += 4; carry = h3;
    }
    store_vec(r++, alignr64(zero(), carry, 7));             // harvest: [carry7, 0..]
}

inline static void mul_u52_bn2(pvec r, cpvec a, uint64_t an, _vec b0, _vec b1){
    _vec pMid = zero(), pH1 = zero();
    uint64_t c = (an + 7) >> 3;

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

    for(; c >= 2; c -= 2){
        _vec a0 = load_vec(a+0), a1 = load_vec(a+1);
        a += 2;
        _vec o0, o1;
        MUL2_CHUNK(a0, o0);
        MUL2_CHUNK(a1, o1);
        store_vec(r+0, o0); store_vec(r+1, o1);
        r += 2;
    }
    if(c){                                  // 0 or 1 chunk left
        _vec ax = load_vec(a++), o;
        MUL2_CHUNK(ax, o);
        store_vec(r++, o);
    }

    store_vec(r++, add(alignr64(zero(), pMid, 7),
                       alignr64(zero(), pH1,  6)));
    #undef MUL2_CHUNK
}

inline static void mul_u52_1(pvec r, cpvec a, cpvec b, uint64_t an, const uint64_t bn){
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

#define mul_accum(a, b, ind) {\
    bx = splat_load(b, ind); \
    n[ind] = madd52lo(n[ind], a, bx); \
    n[ind+1] = madd52hi(n[ind+1], a, bx); \
}

// pull diagonal ind: current-cycle part (high lanes) + previous cycle's
// wrap (low lanes, kept in m[ind]); then hand this cycle's diagonal forward.
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

// diagonal 0 is already the live base in n[0]. ind==8 has zero in-cycle
// shift (alignr imm 0 == m[8]), so it folds to a plain add.
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

// the residual wraps left in m[] form one more output vector —
// this is exactly what the old code stored as n[8].
#define flush_tail { \
    n[0] = m[8]; \
    n[0] = add(n[0], alignr64(zero(), m[7], 1)); \
    n[0] = add(n[0], alignr64(zero(), m[6], 2)); \
    n[0] = add(n[0], alignr64(zero(), m[5], 3)); \
    n[0] = add(n[0], alignr64(zero(), m[4], 4)); \
    n[0] = add(n[0], alignr64(zero(), m[3], 5)); \
    n[0] = add(n[0], alignr64(zero(), m[2], 6)); \
    n[0] = add(n[0], alignr64(zero(), m[1], 7)); \
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
        n[0] = zero();                 // <-- was: n[0] = n[8]; n[8] = zero();
    }
    const uint64_t tail = an + bn >= 7;
    if(an < bn){ bn = an; bp = a + au; ax = load_vec(b + bu); }
    else       {           bp = b + bu; ax = load_vec(a + au); }
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
    store_vec(r, n[0]);
    if(tail) store_vec(r+1, n[8]);
}
