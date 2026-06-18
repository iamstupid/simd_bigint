#pragma once
// Normalized u52 long division (schoolbook / Knuth Algorithm D) at block
// granularity: one "limb" is a zmm block = 8 u52 digits = base beta = 2^416.
//
//   divrem_core : the normalized core. Dividend/divisor/quotient are u52 digit
//                 arrays (8 digits per block); the divisor's top 2 blocks must
//                 be normalized (top digit's bit 51 set) and v = recip_3by2 of
//                 those top 2 blocks.  Per step: div3by2 estimates one quotient
//                 block from the top 3 dividend blocks / top 2 divisor blocks,
//                 then submul_1 subtracts qhat * (the lower dn-2 divisor blocks)
//                 over the array, with a single add-back correction.
//
//   divrem_u64  : packed-u64 wrapper.  Bit-normalizes N and D so D fills dn
//                 blocks, converts u64->u52, builds the reciprocal, runs the
//                 core, then converts quotient back and right-shifts the
//                 remainder out of the normalized domain.
#define SIMD_MPN_FFT 0
#include "div.h"
#include "u64cvt.h"
#include <stdlib.h>

#define INLINE static inline __attribute__((always_inline))

// canon.h defines `canonize`, but mul_vec.h (pulled in via div.h) #undef's it;
// redefine the positive streaming canonicalize here (one SWAR carry pass).
#ifndef canonize
#define canonize(vec, carry, sigcarry) do {                                    \
    const _vec M = MASK52;                                                      \
    _vec hi  = srli(vec, 52);                                                   \
    _vec his = alignr64(hi, carry, 7);                                          \
    _vec _ct = add(and_v(vec, M), his);                                        \
    carry    = hi;                                                              \
    unsigned g = (unsigned)gtu(_ct, M);                                        \
    unsigned p = (unsigned)eq (_ct, M);                                        \
    unsigned chain = p + ((g << 1) | sigcarry);                                \
    sigcarry = chain >> 8;                                                      \
    __mmask8 cin = (__mmask8)(p ^ chain);                                      \
    vec = and_v(sub(_ct, M, cin, _ct), M);                                    \
} while(0)
#endif

// ===================== block helpers (base beta = 2^416) =====================

// qhat - 1  (block decrement, canonical in/out)
INLINE _vec block_dec(_vec a){ unsigned b = 1; return block_subb(a, zero(), &b); }

// all 8 digits equal?
INLINE int block_eq(_vec a, _vec b){ return (uint8_t)neq(a, b) == 0; }

// np[0..len) += dp[0..len)  (len blocks); returns carry-out bit
INLINE unsigned block_add_n(_limb* np, const _limb* dp, uint64_t len){
    unsigned c = 0;
    for(uint64_t i = 0; i < len; i++){
        _vec x = load_vec((cpvec)(np + 8*i));
        x = block_addc(x, load_vec((cpvec)(dp + 8*i)), &c);
        store_vec((pvec)(np + 8*i), x);
    }
    return c;
}

// np[0..len) -= qhat * dp[0..len)  (len blocks, len >= 1); returns the high
// carry block cy such that (np_old - qhat*dp) = np_new - cy*beta^len.
//
// Fused streaming submul, modelled on mul_u52_vec: it accumulates the product
// with deferred carry (the t[] diagonal), and per output block does exactly one
// canonize (resolve the product block) + one borrow_prop (np[i] -= block),
// threading a single borrow.  That is 2 canon-class passes per block instead of
// the 5 of a per-block mul_zmm + canon2 + 3 carry ops.
INLINE _vec block_submul_vec(_limb* np, const _limb* dp, uint64_t len, _vec b){
    _vec t[17];
#define mulacc_s(ind) {                                              \
        t[(ind)+8] = madd52lo(t[(ind)+8], b, splat_load(dp, ind));   \
        t[(ind)+9] = zero();                                         \
        t[(ind)+9] = madd52hi(t[(ind)+9], b, splat_load(dp, ind));   \
    }
    t[8]=zero(); t[1]=zero(); t[2]=zero(); t[3]=zero();
    t[4]=zero(); t[5]=zero(); t[6]=zero(); t[7]=zero();
    t[0]=zero();                          // t[0] = canonize carry (product overflow)
    int sig = 0;                          // SWAR carry for the product canonize
    unsigned bw = 0;                      // borrow chain for the subtract
    pvec rp = (pvec)np;
    for(uint64_t n = len; n; --n, dp += 8){
        mulacc_s(0); mulacc_s(1); mulacc_s(2); mulacc_s(3);
        mulacc_s(4); mulacc_s(5); mulacc_s(6); mulacc_s(7);
        _vec k1,k2,k3,k4,k12,k34;
        k1 = add(t[8],                     alignr64(t[9],  t[1], 7)); t[1]=t[9];  t[8]=t[16];
        k2 = add(alignr64(t[10], t[2], 6), alignr64(t[11], t[3], 5)); t[2]=t[10]; t[3]=t[11];
        k3 = add(alignr64(t[12], t[4], 4), alignr64(t[13], t[5], 3)); t[4]=t[12]; t[5]=t[13];
        k4 = add(alignr64(t[14], t[6], 2), alignr64(t[15], t[7], 1)); t[6]=t[14]; t[7]=t[15];
        k12 = add(k1, k2); k34 = add(k3, k4); k12 = add(k12, k34);
        canonize(k12, t[0], sig);                  // canonical product block
        _vec npv = load_vec((cpvec)rp);
        borrow_prop(rp, npv, k12, bw);             // *rp = npv - k12 ; rp++ ; thread bw
    }
    // the multiply's high spill-over block (= product block `len`)
    t[9]=zero(); t[10]=zero(); t[11]=zero(); t[12]=zero();
    t[13]=zero(); t[14]=zero(); t[15]=zero();
    _vec k1 = add(t[8],                     alignr64(t[9],  t[1], 7));
    _vec k2 = add(alignr64(t[10], t[2], 6), alignr64(t[11], t[3], 5));
    _vec k3 = add(alignr64(t[12], t[4], 4), alignr64(t[13], t[5], 3));
    _vec k4 = add(alignr64(t[14], t[6], 2), alignr64(t[15], t[7], 1));
    _vec top = add(add(k1,k2), add(k3,k4));
    canonize(top, t[0], sig);
    unsigned c = bw;                               // cy = top + final borrow (top <= beta-2)
    return block_addc(top, zero(), &c);
#undef mulacc_s
}

// ===================== normalized core =====================

// np: nn blocks dividend digits, MODIFIED -> remainder lands in low dn blocks.
// dp: dn blocks divisor digits, normalized (digit 8*dn-1 bit 51 set), dn >= 2.
// v : recip_3by2_seed of the top 2 divisor blocks <dp[dn-1], dp[dn-2]>.
// qp: qn = nn - dn blocks quotient output.  Requires the top dividend block to
//     be < the top divisor block (the u64 wrapper guarantees this via a zero
//     padding block).
static inline void divrem_core(_limb* qp, _limb* np, uint64_t nn,
                               const _limb* dp, uint64_t dn, _vec v){
    const uint64_t qn = nn - dn;
    const _vec d1 = load_vec((cpvec)(dp + 8*(dn-1)));
    const _vec d0 = load_vec((cpvec)(dp + 8*(dn-2)));
    for(uint64_t j = qn; j-- > 0; ){
        _vec n2 = load_vec((cpvec)(np + 8*(j+dn)));
        _vec qhat;
        if(__builtin_expect(block_eq(n2, d1), 0)){
            // rare: <n2,n1> >= <d1,d0> -> qhat would overflow; use beta-1 and a
            // full submul over all dn blocks (Knuth D4-D6).
            qhat = MASK52;
            _vec cy = block_submul_vec(np + 8*j, dp, dn, qhat);
            unsigned b = 0; _vec top = block_subb(n2, cy, &b);
            while(b){
                qhat = block_dec(qhat);
                unsigned c = block_add_n(np + 8*j, dp, dn);
                top = block_addc(top, zero(), &c);
                b -= c;
            }
            (void)top;   // np[j+dn] is above the final remainder; dead.
        } else {
            // <n2,n1> < <d1,d0> guaranteed -> div3by2 precondition holds.
            _div32 R = div3by2(np + 8*(j+dn-2), dp + 8*(dn-2), v);
            qhat = R.q1; _vec r0 = R.r.lo, r1 = R.r.hi;
            // subtract qhat * (lower dn-2 divisor blocks) from np[j..j+dn-3];
            // its high carry cy borrows into <r1,r0>.
            _vec cy = block_submul_vec(np + 8*j, dp, dn-2, qhat);
            unsigned b = 0;
            r0 = block_subb(r0, cy, &b);
            r1 = block_subb(r1, zero(), &b);
            while(b){                       // qhat too large -> add D back
                qhat = block_dec(qhat);
                unsigned c = block_add_n(np + 8*j, dp, dn-2);
                r0 = block_addc(r0, d0, &c);
                r1 = block_addc(r1, d1, &c);
                b -= c;
            }
            store_vec((pvec)(np + 8*(j+dn-2)), r0);
            store_vec((pvec)(np + 8*(j+dn-1)), r1);
        }
        store_vec((pvec)(qp + 8*j), qhat);
    }
}

// ===================== u64 wrapper =====================

// Fused decode + left-shift: r = (value of ap[0..an)) << s, as u52 digits.
// One pass over ap -- no intermediate shifted-u64 buffer.  The whole-digit part
// of the shift (kd = s/52) is the output digit offset; the sub-digit part
// (kb = s%52) is a u52 funnel across digits during the decode.  r must be
// pre-zeroed for out_blocks blocks (caller does this; the low kd digits and any
// padding stay zero).
static inline void u52_from_u64_lsh(pvec r, const uint64_t* ap, uint64_t an, uint64_t s){
    const uint64_t kd = s / 52; const unsigned kb = (unsigned)(s % 52);
    const uint64_t bits = u64_bit_length(ap, an);
    const uint64_t n52 = (bits + 51) / 52;
    const uint8_t* p = (const uint8_t*)ap;
    int64_t rem = (int64_t)(an * 8);
    const _vec perm = load_vec((cpvec)u52_dec_perm);
    const _vec sh = setr_64(0, 4, 0, 4, 0, 4, 0, 4);
    const _vec kbv = set1_64(kb), k52 = set1_64(52 - kb), M = MASK52;
    _limb* out = (_limb*)r + kd;
    uint64_t blocks = (n52 + 7) >> 3, bi = 0;
    _vec prev = zero();
    for(; blocks; --blocks, p += 52, rem -= 52, ++bi){
        const _vec w = vec_fn(maskz_loadu_epi8)(
            rem >= 64 ? ~0ull : (rem > 0 ? (~0ull >> (64 - rem)) : 0), p);
        _vec dec = and_v(srlv(permb(perm, w), sh), M);          // 8 u52 digits
        _vec o = kb ? and_v(or_v(sllv(dec, kbv), srlv(alignr64(dec, prev, 7), k52)), M)
                    : dec;                                      // funnel by kb
        store_vec((pvec)(out + 8*bi), o);
        prev = dec;
    }
    if(kb)  // top digit's spill-out -> one more digit
        store_vec((pvec)(out + 8*bi), and_v(srlv(alignr64(zero(), prev, 7), k52), M));
}

// r[0..n) = a[0..n] >> s   (reads a[w .. w+n]; caller zero-pads beyond; r may alias a).
static inline void mpn_rshift_simd(uint64_t* r, const uint64_t* a, uint64_t n, uint64_t s){
    const uint64_t w = s >> 6; const unsigned b = (unsigned)(s & 63);
    const uint64_t* ap = a + w;
    if(b == 0){
        uint64_t i = 0;
        for(; i + 8 <= n; i += 8) store_vec((pvec)(r+i), load_vec((cpvec)(ap+i)));
        if(i < n){ __mmask8 m = (__mmask8)((1u << (n-i)) - 1);
                   store_vec((pvec)(r+i), load_vec((cpvec)(ap+i), m), m); }
        return;
    }
    const _vec cb = set1_64(b);
    uint64_t i = 0;
    for(; i + 8 <= n; i += 8){
        _vec cur = load_vec((cpvec)(ap+i)), above = load_vec((cpvec)(ap+i+1));
        store_vec((pvec)(r+i), fshr64v(cur, above, cb));     // (cur>>b)|(above<<(64-b))
    }
    if(i < n){ __mmask8 m = (__mmask8)((1u << (n-i)) - 1);
        _vec cur = load_vec((cpvec)(ap+i), m), above = load_vec((cpvec)(ap+i+1), m);
        store_vec((pvec)(r+i), fshr64v(cur, above, cb), m); }
}

// Quotient Q = floor(N/D), remainder R = N mod D, on packed u64 limbs.
//   np[0..nn64), dp[0..dn64)  (dp[dn64-1] != 0).
// Writes Q to qp (caller: >= nn64-dn64+1 limbs) and *qn64_out its length;
// writes R to rp (caller: >= dn64 limbs) and *rn64_out its length.
// Requires the divisor to span at least 2 blocks (bitlen(D) > 416).
// Uses the per-thread scratch arena (no per-call heap allocation).
static inline void divrem_u64(uint64_t* qp, uint64_t* rp,
                              const uint64_t* np, uint64_t nn64,
                              const uint64_t* dp, uint64_t dn64,
                              uint64_t* qn64_out, uint64_t* rn64_out){
    const uint64_t Dbits = u64_bit_length(dp, dn64);
    const uint64_t Nbits = u64_bit_length(np, nn64);
    if(Nbits < Dbits){                          // N < D: Q = 0, R = N
        for(uint64_t i = 0; i < dn64; i++) rp[i] = (i < nn64) ? np[i] : 0;
        qp[0] = 0; *qn64_out = 1; *rn64_out = dn64;
        return;
    }
    const uint64_t dn = (Dbits + 415) / 416;    // divisor blocks
    const uint64_t s  = 416 * dn - Dbits;       // normalization shift, 0..415
    const uint64_t Nb = Nbits + s;              // normalized dividend bit length
    const uint64_t nn = (Nb + 415) / 416 + 1;   // dividend blocks (+1 padding)
    const uint64_t qn = nn - dn;

    scratch* sc = scratch_thread();
    SCRATCH(sc);

    // u52 digit buffers, normalized in one fused decode+shift pass (no u64
    // intermediate).  D52 gets an extra block for the funnel spill; Q gets a
    // spare zero block because a ceil(416*k/64)-limb conversion reads k+1
    // blocks; N's block above the remainder is zeroed for the same reason.
    pvec D52 = SALLOC0(sc, _vec, dn + 1);
    pvec N52 = SALLOC0(sc, _vec, nn);
    pvec Q52 = SALLOC0(sc, _vec, qn + 1);
    u52_from_u64_lsh(D52, dp, dn64, s);
    u52_from_u64_lsh(N52, np, nn64, s);

    // reciprocal seed: depends only on the top 7 limbs (top 448 bits) of the
    // normalized divisor.  Take the top zmm of raw D and funnel-shift left by
    // clz so its MSB lands at bit 63 of lane 7; lanes 1..7 are those 7 limbs.
    uint64_t I13[13] = {0};
    {
        const unsigned cl = (unsigned)__builtin_clzll(dp[dn64-1]);
        // dn>=2 => Dbits>416 => dn64>=7; the only <8 case is dn64==7.
        _vec top8 = (dn64 >= 8)
            ? load_vec((cpvec)(dp + dn64 - 8))
            : alignr64(load_vec((cpvec)dp, (__mmask8)0x7f), zero(), 7);  // dp[0..6] -> lanes 1..7
        _vec norm = fshl64v(top8, alignr64(top8, zero(), 7), set1_64(cl));
        store_vec((pvec)(I13 + 5), norm);       // lanes 1..7 -> I13[6..12]
    }
    _vec v = recip_3by2_seed(I13);

    divrem_core((_limb*)Q52, (_limb*)N52, nn, (const _limb*)D52, dn, v);

    // quotient u52 -> u64
    const uint64_t Ql = (416 * qn + 63) / 64;
    uint64_t* Qb = SALLOC0(sc, uint64_t, Ql + 1);
    u64_from_u52_canon(Qb, (cpvec)Q52, Ql);
    uint64_t qn64 = Ql; while(qn64 && Qb[qn64-1] == 0) --qn64; if(!qn64) qn64 = 1;
    for(uint64_t i = 0; i < qn64; i++) qp[i] = Qb[i];
    *qn64_out = qn64;

    // remainder: low dn blocks of N52 hold R<<s ; convert then right-shift by s.
    // zero the block above the remainder so the conversion's over-read is clean.
    memset((_limb*)N52 + 8*dn, 0, 8 * 8);
    const uint64_t Rl = (416 * dn + 63) / 64;
    uint64_t* Rb = SALLOC0(sc, uint64_t, Rl + (s >> 6) + 2);  // zero-padded for rshift
    u64_from_u52_canon(Rb, (cpvec)N52, Rl);
    mpn_rshift_simd(Rb, Rb, dn64, s);           // R = (R<<s) >> s, low dn64 limbs
    uint64_t rn64 = dn64; while(rn64 && Rb[rn64-1] == 0) --rn64; if(!rn64) rn64 = 1;
    for(uint64_t i = 0; i < dn64; i++) rp[i] = Rb[i];
    *rn64_out = rn64;
}

#undef INLINE
