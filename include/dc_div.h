#pragma once
// Divide-and-conquer block division (u52, beta = 2^416 = 8 digits/block),
// modelled on GMP's mpn_dcpi1_div_qr / _qr_n.  The bulk work is full
// multiplication (mul_u52_dispatch_canon) instead of the rank-1 submul, and the
// recursion is sub-quadratic.  Non-power-of-2 sizes are handled the GMP way:
// the recursion splits n into lo=n>>1 / hi=n-lo (odd ok), and the dispatcher
// peels the quotient in dn-block chunks with a properly-handled irregular top
// chunk (never a naive split).
#include "divrem.h"
#include "mul.h"

#define INLINE static inline __attribute__((always_inline))

#ifndef DC_THRESHOLD
#define DC_THRESHOLD 8          /* blocks: below this, schoolbook base case */
#endif

// ---- block multi-precision helpers (carry/borrow threaded across blocks) ----

// np[0..len) -= dp[0..len) ; returns borrow.
INLINE unsigned block_sub_n(_limb* np, const _limb* dp, uint64_t len){
    unsigned b = 0;
    for(uint64_t i = 0; i < len; i++){
        _vec x = load_vec((cpvec)(np + 8*i));
        x = block_subb(x, load_vec((cpvec)(dp + 8*i)), &b);
        store_vec((pvec)(np + 8*i), x);
    }
    return b;
}
// qp[0..len) -= 1 ; returns borrow out.
INLINE unsigned block_sub_1(_limb* qp, uint64_t len){
    unsigned b = 1;
    for(uint64_t i = 0; i < len && b; i++){
        _vec x = load_vec((cpvec)(qp + 8*i));
        x = block_subb(x, zero(), &b);
        store_vec((pvec)(qp + 8*i), x);
    }
    return b;
}
// compare canonical len-block arrays: 1 if a>b, -1 if a<b, 0 if equal.
INLINE int block_cmp(const _limb* a, const _limb* b, uint64_t len){
    for(uint64_t i = len; i-- > 0; ){
        _vec av = load_vec((cpvec)(a + 8*i)), bv = load_vec((cpvec)(b + 8*i));
        if(!block_eq(av, bv)) return block_ge(av, bv) ? 1 : -1;
    }
    return 0;
}
// tp[0..an+bn) = a[0..an) * b[0..bn)  (blocks), canonical.  Uses the arena.
INLINE void blk_mul(_limb* tp, const _limb* a, uint64_t an, const _limb* b, uint64_t bn){
    scratch* sc = scratch_thread();
    SCRATCH(sc);
    if(an >= bn) mul_u52_dispatch_canon((pvec)tp, (cpvec)a, (cpvec)b, an*8, bn*8, sc);
    else         mul_u52_dispatch_canon((pvec)tp, (cpvec)b, (cpvec)a, bn*8, an*8, sc);
    // dispatch_canon only folds class-1; a class-0 result may still be
    // non-negative-redundant (lanes >= 2^52).  The block sub/add chains need
    // canonical digits, so force a positive canonicalize.
    mpn_canon_pos((pvec)tp, (cpvec)tp, (an+bn)*8);
}

// ---- schoolbook base case (mpn_sbpi1_div_qr analogue), returns qh ----
// np: nn blocks (modified -> remainder in low dn), dp: dn>=2 normalized, v=recip.
// quotient to qp[0..nn-dn); top dn blocks may be >= d (qh captures that).
INLINE int blk_sbpi1(_limb* qp, _limb* np, uint64_t nn, const _limb* dp, uint64_t dn, _vec v){
    const uint64_t qn = nn - dn;
    int qh = block_cmp(np + 8*qn, dp, dn) >= 0;     // top dn blocks vs d
    if(qh) block_sub_n(np + 8*qn, dp, dn);
    const _vec d1 = load_vec((cpvec)(dp + 8*(dn-1)));
    const _vec d0 = load_vec((cpvec)(dp + 8*(dn-2)));
    for(uint64_t j = qn; j-- > 0; ){
        _vec n2 = load_vec((cpvec)(np + 8*(j+dn)));
        _vec qhat;
        if(__builtin_expect(block_eq(n2, d1), 0)){
            qhat = MASK52;
            _vec cy = block_submul_vec(np + 8*j, dp, dn, qhat);
            unsigned b = 0; _vec top = block_subb(n2, cy, &b);
            while(b){ qhat = block_dec(qhat); unsigned c = block_add_n(np + 8*j, dp, dn);
                      top = block_addc(top, zero(), &c); b -= c; }
            (void)top;
        } else {
            _div32 R = div3by2(np + 8*(j+dn-2), dp + 8*(dn-2), v);
            qhat = R.q1; _vec r0 = R.r.lo, r1 = R.r.hi;
            _vec cy = block_submul_vec(np + 8*j, dp, dn-2, qhat);
            unsigned b = 0;
            r0 = block_subb(r0, cy, &b);
            r1 = block_subb(r1, zero(), &b);
            while(b){ qhat = block_dec(qhat); unsigned c = block_add_n(np + 8*j, dp, dn-2);
                      r0 = block_addc(r0, d0, &c); r1 = block_addc(r1, d1, &c); b -= c; }
            store_vec((pvec)(np + 8*(j+dn-2)), r0);
            store_vec((pvec)(np + 8*(j+dn-1)), r1);
        }
        store_vec((pvec)(np + 8*(j+dn)), zero());   // zero consumed top block (clean remainder)
        store_vec((pvec)(qp + 8*j), qhat);
    }
    return qh;
}

// ---- 2n/n recursive division (mpn_dcpi1_div_qr_n analogue), returns qh ----
// np: 2n blocks -> quotient qp[0..n), remainder in np[0..n). dp: n blocks. tp: n blocks scratch.
static int blk_dcpi1_div_qr_n(_limb* qp, _limb* np, const _limb* dp, uint64_t n, _vec v, _limb* tp){
    const uint64_t lo = n >> 1, hi = n - lo;
    int qh, ql; unsigned cy;
    // high half quotient: divide top 2hi blocks by top hi blocks of d
    if(hi < DC_THRESHOLD) qh = blk_sbpi1(qp + 8*lo, np + 8*(2*lo), 2*hi, dp + 8*lo, hi, v);
    else                  qh = blk_dcpi1_div_qr_n(qp + 8*lo, np + 8*(2*lo), dp + 8*lo, hi, v, tp);
    blk_mul(tp, qp + 8*lo, hi, dp, lo);             // Qhi * Dlo  (n blocks)
    cy = block_sub_n(np + 8*lo, tp, n);
    if(qh) cy += block_sub_n(np + 8*n, dp, lo);
    while(cy){ qh -= block_sub_1(qp + 8*lo, hi); cy -= block_add_n(np + 8*lo, dp, n); }
    // low half quotient: divide np[hi..hi+2lo) by top lo blocks of d
    if(lo < DC_THRESHOLD) ql = blk_sbpi1(qp, np + 8*hi, 2*lo, dp + 8*hi, lo, v);
    else                  ql = blk_dcpi1_div_qr_n(qp, np + 8*hi, dp + 8*hi, lo, v, tp);
    blk_mul(tp, dp, hi, qp, lo);                    // Dhi(low hi) * Qlo  (n blocks)
    cy = block_sub_n(np, tp, n);
    if(ql) cy += block_sub_n(np + 8*lo, dp, hi);
    while(cy){ block_sub_1(qp, lo); cy -= block_add_n(np, dp, n); }
    return qh;
}

// ---- one quotient chunk of c blocks from window np[0..c+dn) / d ; returns qh ----
static int blk_chunk_divide(_limb* qp, _limb* np, uint64_t c, const _limb* dp, uint64_t dn, _vec v, _limb* tp){
    if(c < DC_THRESHOLD)
        return blk_sbpi1(qp, np, c + dn, dp, dn, v);     // full-divisor schoolbook
    // c >= THRESHOLD (>=2): 2c/c on the top, then cross-correct with low (dn-c) of d
    const uint64_t k = dn - c;
    int qh = blk_dcpi1_div_qr_n(qp, np + 8*k, dp + 8*k, c, v, tp);   // top 2c / top c of d
    if(c != dn){
        blk_mul(tp, qp, c, dp, k);                       // Q * Dlow  (dn blocks)
        unsigned cy = block_sub_n(np, tp, dn);
        if(qh) cy += block_sub_n(np + 8*c, dp, k);
        while(cy){ qh -= block_sub_1(qp, c); cy -= block_add_n(np, dp, dn); }
    }
    return qh;
}

// ---- general dispatcher (mpn_dcpi1_div_qr analogue), returns qh ----
// np: nn blocks -> remainder in low dn ; quotient qp[0..nn-dn). dp: dn>=2 normalized.
static int blk_dcpi1_div_qr(_limb* qp, _limb* np, uint64_t nn, const _limb* dp, uint64_t dn, _vec v, _limb* tp){
    const uint64_t qn = nn - dn;
    uint64_t c = qn % dn; if(c == 0) c = dn;            // top chunk size in [1,dn]
    uint64_t p = qn - c;
    int qh = blk_chunk_divide(qp + 8*p, np + 8*p, c, dp, dn, v, tp);
    while(p > 0){
        p -= dn;
        blk_dcpi1_div_qr_n(qp + 8*p, np + 8*p, dp, dn, v, tp);
    }
    return qh;
}

// ===================== u64 wrapper (D&C core) =====================
// Same packed-u64 contract as divrem.h's divrem_u64, but the normalized division
// uses the sub-quadratic blk_dcpi1_div_qr above (falling back to the schoolbook
// divrem_core when dn < DC_THRESHOLD).  Reuses all of divrem.h's conversion
// machinery (u52_from_u64_lsh / u52_rshift / recip seed / u64_from_u52_canon).
static void divrem_u64_dc(uint64_t* qp, uint64_t* rp,
                          const uint64_t* np, uint64_t nn64,
                          const uint64_t* dp, uint64_t dn64,
                          uint64_t* qn64_out, uint64_t* rn64_out){
    const uint64_t Dbits = u64_bit_length(dp, dn64);
    const uint64_t Nbits = u64_bit_length(np, nn64);
    if(Nbits < Dbits){
        for(uint64_t i = 0; i < dn64; i++) rp[i] = (i < nn64) ? np[i] : 0;
        qp[0] = 0; *qn64_out = 1; *rn64_out = dn64;
        return;
    }
    const uint64_t dn = (Dbits + 415) / 416;
    const uint64_t s  = 416 * dn - Dbits;
    const uint64_t Nb = Nbits + s;
    const uint64_t nn = (Nb + 415) / 416 + 1;
    const uint64_t qn = nn - dn;

    scratch* sc = scratch_thread();
    SCRATCH(sc);

    pvec D52 = SALLOC0(sc, _vec, dn + 1);
    pvec N52 = SALLOC0(sc, _vec, nn);
    pvec Q52 = SALLOC0(sc, _vec, qn + 2);
    u52_from_u64_lsh(D52, dp, dn64, s);
    u52_from_u64_lsh(N52, np, nn64, s);

    uint64_t I13[13] = {0};
    {
        const unsigned cl = (unsigned)__builtin_clzll(dp[dn64-1]);
        _vec top8 = (dn64 >= 8)
            ? load_vec((cpvec)(dp + dn64 - 8))
            : alignr64(load_vec((cpvec)dp, (__mmask8)0x7f), zero(), 7);
        _vec norm = fshl64v(top8, alignr64(top8, zero(), 7), set1_64(cl));
        store_vec((pvec)(I13 + 5), norm);
    }
    _vec v = recip_3by2_seed(I13);

    if(dn < DC_THRESHOLD){
        divrem_core((_limb*)Q52, (_limb*)N52, nn, (const _limb*)D52, dn, v);
    } else {
        pvec tp = SALLOC0(sc, _vec, dn + 1);     // cross-product scratch (<= dn blocks)
        blk_dcpi1_div_qr((_limb*)Q52, (_limb*)N52, nn, (const _limb*)D52, dn, v, (_limb*)tp);
    }

    const uint64_t qm = nn64 - dn64 + 1;
    u64_from_u52_canon(qp, (cpvec)Q52, qm);
    uint64_t qn64 = qm; while(qn64 && qp[qn64-1] == 0) --qn64; if(!qn64) qn64 = 1;
    *qn64_out = qn64;

    memset((_limb*)N52 + 8*dn, 0, 8 * 8);
    u52_rshift((_limb*)N52, (_limb*)N52, dn, s);
    u64_from_u52_canon(rp, (cpvec)N52, dn64);
    uint64_t rn64 = dn64; while(rn64 && rp[rn64-1] == 0) --rn64; if(!rn64) rn64 = 1;
    *rn64_out = rn64;
}

#undef INLINE
