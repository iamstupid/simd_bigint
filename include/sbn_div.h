#pragma once
// Scalar basecase division for small operands: a faithful port of GMP's
// mpn_sbpi1_div_qr (Moeller-Granlund 3/2) + invert_pi1 + udiv_qr_3by2, using
// the namespaced sbn_ asm kernels (submul_1 / add_n / sub_n).  Same algorithm
// and inner loop as GMP's basecase, so ~same speed.  Requires dn >= 3.
#include "sbn_mpn.h"
#include <stdint.h>

// ---- double-limb primitives (compile to mulx / adc) ----
#define sbn_umul_ppmm(hi,lo,a,b) do{ \
    __uint128_t _p=(__uint128_t)(uint64_t)(a)*(uint64_t)(b); \
    (lo)=(uint64_t)_p; (hi)=(uint64_t)(_p>>64);}while(0)
#define sbn_add_ssaaaa(sh,sl,ah,al,bh,bl) do{ \
    uint64_t _al=(al),_s=_al+(bl); (sl)=_s; (sh)=(ah)+(bh)+(_s<_al);}while(0)
#define sbn_sub_ddmmss(sh,sl,ah,al,bh,bl) do{ \
    uint64_t _al=(al),_bl=(bl); (sl)=_al-_bl; (sh)=(ah)-(bh)-(_al<_bl);}while(0)

static inline uint64_t sbn_invert_limb(uint64_t d){      // floor((2^128-1)/d) - 2^64
    __uint128_t num = ((__uint128_t)(~d) << 64) | ~(uint64_t)0;
    return (uint64_t)(num / d);
}
// 2-limb reciprocal of <d1,d0> (d1 normalized: top bit set).  GMP invert_pi1.
static inline uint64_t sbn_invert_pi1(uint64_t d1, uint64_t d0){
    uint64_t v, p, t1, t0, mask;
    v = sbn_invert_limb(d1);
    p = d1 * v; p += d0;
    if(p < d0){ v--; mask = -(uint64_t)(p >= d1); p -= d1; v += mask; p -= mask & d1; }
    sbn_umul_ppmm(t1, t0, d0, v); p += t1;
    if(p < t1){ v--; if(p >= d1){ if(p > d1 || t0 >= d0) v--; } }
    return v;
}

// GMP udiv_qr_3by2: <n2,n1,n0> / <d1,d0>, dinv = sbn_invert_pi1(d1,d0).
#define sbn_udiv_qr_3by2(q, r1, r0, n2, n1, n0, d1, d0, dinv) do { \
    uint64_t _q0,_t1,_t0,_mask;                                    \
    sbn_umul_ppmm((q), _q0, (n2), (dinv));                         \
    sbn_add_ssaaaa((q), _q0, (q), _q0, (n2), (n1));                \
    (r1) = (n1) - (d1) * (q);                                      \
    sbn_sub_ddmmss((r1),(r0),(r1),(n0),(d1),(d0));                 \
    sbn_umul_ppmm(_t1,_t0,(d0),(q));                               \
    sbn_sub_ddmmss((r1),(r0),(r1),(r0),_t1,_t0);                   \
    (q)++;                                                         \
    _mask = -(uint64_t)((r1) >= _q0);                             \
    (q) += _mask;                                                  \
    sbn_add_ssaaaa((r1),(r0),(r1),(r0),_mask&(d1),_mask&(d0));     \
    if((r1) >= (d1)){ if((r1) > (d1) || (r0) >= (d0)){            \
        (q)++; sbn_sub_ddmmss((r1),(r0),(r1),(r0),(d1),(d0)); } } \
} while(0)

static inline int sbn_cmp(const uint64_t* a, const uint64_t* b, long n){
    for(long i = n-1; i >= 0; i--) if(a[i] != b[i]) return a[i] > b[i] ? 1 : -1;
    return 0;
}
static inline uint64_t sbn_lshift(uint64_t* rp, const uint64_t* up, long n, unsigned c){
    uint64_t carry = 0;
    for(long i = 0; i < n; i++){ uint64_t x = up[i]; rp[i] = (x << c) | carry; carry = x >> (64 - c); }
    return carry;
}
static inline void sbn_rshift(uint64_t* rp, const uint64_t* up, long n, unsigned c){
    uint64_t carry = 0;
    for(long i = n; i-- > 0; ){ uint64_t x = up[i]; rp[i] = (x >> c) | carry; carry = x << (64 - c); }
}

// Verbatim port of mpn_sbpi1_div_qr.  np[0..nn) normalized, dp[dn-1] top bit
// set, dn>2.  Quotient to qp[0..nn-dn); remainder left in np[0..dn).
static inline uint64_t sbn_sbpi1_div_qr(uint64_t* qp, uint64_t* np, long nn,
                                        const uint64_t* dp, long dn, uint64_t dinv){
    uint64_t qh, n1, n0, d1, d0, cy, cy1, q;
    np += nn;
    qh = sbn_cmp(np - dn, dp, dn) >= 0;
    if(qh) sbn_sub_n(np - dn, np - dn, dp, dn);
    qp += nn - dn;
    dn -= 2;
    d1 = dp[dn + 1]; d0 = dp[dn + 0];
    np -= 2;
    n1 = np[1];
    for(long i = nn - (dn + 2); i > 0; i--){
        np--;
        if(n1 == d1 && np[1] == d0){
            q = ~(uint64_t)0;
            sbn_submul_1(np - dn, dp, dn + 2, q);
            n1 = np[1];
        } else {
            sbn_udiv_qr_3by2(q, n1, n0, n1, np[1], np[0], d1, d0, dinv);
            cy = sbn_submul_1(np - dn, dp, dn, q);
            cy1 = n0 < cy; n0 = n0 - cy;
            cy = n1 < cy1; n1 = n1 - cy1;
            np[0] = n0;
            if(cy){ n1 += d1 + sbn_add_n(np - dn, np - dn, dp, dn + 1); q--; }
        }
        *--qp = q;
    }
    np[1] = n1;
    return qh;
}

// Q = floor(N/D), R = N mod D.  np[0..nn), dp[0..dn) (dn>=3, dp[dn-1]!=0).
// qp >= nn-dn+1 limbs, rp >= dn limbs.  *qn_out/*rn_out get trimmed lengths.
static inline void sbn_divrem(uint64_t* qp, uint64_t* rp,
                              const uint64_t* np, long nn,
                              const uint64_t* dp, long dn,
                              long* qn_out, long* rn_out){
    const unsigned cnt = (unsigned)__builtin_clzll(dp[dn-1]);
    uint64_t n2[nn + 1];
    uint64_t d2[dn];
    const uint64_t* d2p;
    if(cnt){
        sbn_lshift(d2, dp, dn, cnt); d2p = d2;
        n2[nn] = sbn_lshift(n2, np, nn, cnt);
    } else {
        d2p = dp;
        for(long i = 0; i < nn; i++) n2[i] = np[i];
        n2[nn] = 0;
    }
    uint64_t dinv = sbn_invert_pi1(d2p[dn-1], d2p[dn-2]);
    sbn_sbpi1_div_qr(qp, n2, nn + 1, d2p, dn, dinv);   // qh == 0 (quotient fits)
    long qn = nn - dn + 1; while(qn > 1 && qp[qn-1] == 0) qn--;
    *qn_out = qn;
    if(cnt) sbn_rshift(rp, n2, dn, cnt);
    else    for(long i = 0; i < dn; i++) rp[i] = n2[i];
    long rn = dn; while(rn > 1 && rp[rn-1] == 0) rn--;
    *rn_out = rn;
}
