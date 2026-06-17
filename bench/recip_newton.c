// recip_newton: the Edamatsu-Takahashi (IEEE TETC 2023) reciprocal scheme --
// a *pure-vector* Newton-Raphson for the SAME 416-bit reciprocal
//   v = floor((2^1248-1)/D) - 2^416   (D = 832-bit normalized divisor)
// that recip_3by2 computes, but built by vector Newton instead of a scalar
// mulx seed.  Fraction form r ~ 2^832/D, iterate r <- r*(2 - f*r); both
// rescales (>>832, >>416) are whole-vector aligned so extraction is a vec
// offset.  Each iter = two madd52 multiplies.  Final exact correction reuses
// the shared verify tail (multiply-back vs T=[~i0,~i1], +-1), identical to ours
// -- so a head-to-head isolates exactly the seed method (scalar vs vector).
//
//  build: gcc -O3 -march=native -I.. recip_newton.c ../reciprocal_u416.s -o /tmp/n -lgmp -DNEWT_TEST
#include "../include/types.h"
#include "../include/canon.h"
#include "../include/mul.h"
#include "../include/u64cvt.h"
#include <math.h>

#define canonize(vec, carry, sigcarry) do {                                    \
    const _vec M = MASK52;                                                     \
    _vec hi  = srli(vec, 52);                                                  \
    _vec his = alignr64(hi, carry, 7);                                         \
    _vec _ct = add(and_v(vec, M), his);                                        \
    carry    = hi;                                                             \
    unsigned g = (unsigned)gtu(_ct, M);                                        \
    unsigned p = (unsigned)eq (_ct, M);                                        \
    unsigned chain = p + ((g << 1) | sigcarry);                                \
    sigcarry = chain >> 8;                                                     \
    __mmask8 cin = (__mmask8)(p ^ chain);                                      \
    vec = and_v(sub(_ct, M, cin, _ct), M);                                     \
} while(0)

static inline _vec inc52(_vec v){ __mmask8 p=eq(v,MASK52); unsigned c=(unsigned)p+1u; __mmask8 f=(__mmask8)((unsigned)p^c);
    v=add(v,set1_64(1),kandn(p,f),v); v=and_v(v,zero(),kand(p,f),v); return v; }
static inline _vec dec52(_vec v){ __mmask8 z=eq(v,zero()); unsigned c=(unsigned)z+1u; __mmask8 f=(__mmask8)((unsigned)z^c);
    v=sub(v,set1_64(1),kandn(z,f),v); v=or_v(v,MASK52,kand(z,f),v); return v; }
static inline int gt1(_vec a,_vec b){ uint8_t ne=(uint8_t)neq(a,b); if(!ne) return 0;
    uint8_t lt=(uint8_t)ltu(a,b); return ((uint8_t)(ne^lt)>lt); }
// full 3-block compare: returns 1 iff [a2:a1:a0] > [b2:b1:b0]  (a2/b2 high)
static inline int gt3(_vec a0,_vec a1,_vec a2,_vec b0,_vec b1,_vec b2){
    if((uint8_t)neq(a2,b2)) return gt1(a2,b2);
    if((uint8_t)neq(a1,b1)) return gt1(a1,b1);
    if((uint8_t)neq(a0,b0)) return gt1(a0,b0);
    return 0; }

// ---- the vector Newton reciprocal -----------------------------------------
// Carry the reciprocal as R = 2^416 + v with v an 8-limb block (Q in
// [2^416,2^417) => limb-8 is always exactly 1).  Folding the implicit 2^416
// turns the rescales into "add i1" / "add T", and keeps every broadcast
// operand <= 8 limbs (mul_u52_1 supports bn<=8 only).  Returns v (8 limbs).
//   G   = (D*R)>>832 = i1 + (D*v)>>832
//   T   = 2^417 - G
//   R'  = (R*T)>>416 = T + (v*T)>>416 ;  v' = R' mod 2^416
static const int NEWT_ITERS = 3;
static inline _vec newton_recip_v(const _vec D52[2], const uint64_t I[13]){
    const _vec i1 = D52[1];
    // seed v from double: R0 = round(2^480/D_hi) placed at limb7 (= 52*7), so
    // v0 = R0 mod 2^416 has only limb7 set (top limb), ~52-bit accurate.
    double dhi = (double)I[12] + (double)I[11]*0x1p-64 + (double)I[10]*0x1p-128;
    double r0  = 0x1p480 / dhi;                 // ~ Q in (2^416,2^417]
    int e; double m = frexp(r0, &e);            // r0 = m*2^e, e in {417,418}
    _vec v;
    int degenerate = (e >= 418);                // r0 ~ 2^417 (D ~ 2^831): v ~ 2^416-1
    if(degenerate){
        v = MASK52;                             // all-ones seed; skip Newton, tail corrects
    } else {
        uint64_t mant = (uint64_t)ldexp(m, 53); // 53-bit mantissa
        int sh = e - 53;                        // bit position of R0's lsb (e==417 -> 364)
        uint64_t lane[8] = {0};                 // v0 = R0 mod 2^416
        unsigned __int128 full = (unsigned __int128)mant << (sh % 52);
        int li = sh / 52;
        while(full && li < 8){ lane[li] = (uint64_t)(full & ((1ull<<52)-1)); full >>= 52; ++li; }
        v = load_vec((cpvec)lane);
    }

    const _vec two417[2] = { zero(), setr_64(2,0,0,0,0,0,0,0) };  // 2^417 (limb8=2)
    _vec P[3];                                   // products (<=24 limbs)
    for(int it=0; it<NEWT_ITERS && !degenerate; ++it){
        // P = D * v  (16 x 8) -> 24 limbs
        mul_u52_1(P, (cpvec)D52, (cpvec)&v, 16, 8);
        mpn_canon_pos((pvec)P, (cpvec)P, 24);
        // G = i1 + (D*v)>>832 = i1 + P[16..23]   (9-limb)
        _vec G[2]; const _vec ga[2] = { i1, zero() }, gb[2] = { P[2], zero() };
        mpn_u52_add_canon((pvec)G, (cpvec)ga, (cpvec)gb, 9, zero());
        // T = 2^417 - G   (9-limb)
        _vec T[2];
        mpn_u52_sub_canon((pvec)T, (cpvec)two417, (cpvec)G, 9, zero());
        // P = v * T  (a=T streamed an=9, b=v broadcast bn=8) -> 17 limbs
        mul_u52_1(P, (cpvec)T, (cpvec)&v, 9, 8);
        mpn_canon_pos((pvec)P, (cpvec)P, 17);
        // R' = T + (v*T)>>416 = T + P[8..16] ; v' = R' mod 2^416
        _vec R2[2]; const _vec ra[2] = { T[0], T[1] }, rb[2] = { P[1], P[2] };
        mpn_u52_add_canon((pvec)R2, (cpvec)ra, (cpvec)rb, 9, zero());
        v = R2[0];                               // low 8 limbs
    }
    return v;
}

_vec recip_newton(const uint64_t I[13]){
    _vec D52[2]; u52_from_u64(D52, I, 13);
    const _vec i0 = D52[0], i1 = D52[1];
    _vec seed = newton_recip_v(D52, I);          // candidate v (8 limbs)

    // ---- exact-correct tail.  The Newton seed has a one-sided floor bias
    // (candidate - V in [1,4]); correct with one multiply-back P=seed*I then
    // cheap incremental P -/+ I per +-1 step (no re-multiply).  Fixed 6 passes
    // cover the observed range with margin.  Full 3-block compare vs T.
    _vec P[3]; mul_u52_1(P, (cpvec)D52, (cpvec)&seed, 16, 8); mpn_canon_pos((pvec)P, (cpvec)P, 24);
    const _vec T0 = MASK52, c1 = xor_v(i0, MASK52), c2 = xor_v(i1, MASK52);
    for(int pass=0; pass<6; ++pass){
        if(gt3(P[0],P[1],P[2], T0,c1,c2)){                 // P>T: too big -> DEC, P-=I
            seed = dec52(seed);
            _vec R[3]; __mmask16 bw=0; pvec r=R;
            borrow_prop(r,P[0],i0,bw); borrow_prop(r,P[1],i1,bw); borrow_prop(r,P[2],zero(),bw);
            P[0]=R[0]; P[1]=R[1]; P[2]=R[2];
        } else {
            _vec Q[3]; __mmask16 cy=0; pvec q=Q;            // Q = P + I
            carry_prop(q,P[0],i0,cy); carry_prop(q,P[1],i1,cy); carry_prop(q,P[2],zero(),cy);
            if(gt3(Q[0],Q[1],Q[2], T0,c1,c2)) break;        // P<=T<P+I: exact
            seed = inc52(seed); P[0]=Q[0]; P[1]=Q[1]; P[2]=Q[2];  // too small -> INC, P+=I
        }
    }
    return seed;
}

// ---------------------------------------------------------------------------
#ifdef NEWT_TEST
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
static uint64_t rng(uint64_t*s){uint64_t x=*s;x^=x<<7;x^=x>>9;x*=0x9e3779b97f4a7c15ull;*s=x;return x;}
static long g_lo=1<<30,g_hi=-(1<<30),nbad=0; static mpz_t P416,P1248;
static void check(const uint64_t I[13]){
    _vec out=recip_newton(I); uint64_t la[8]; store_vec((pvec)la,out);
    mpz_t mI,V,num,vk,t;mpz_inits(mI,V,num,vk,t,NULL);mpz_import(mI,13,-1,8,0,0,I);
    mpz_sub_ui(num,P1248,1);mpz_fdiv_q(V,num,mI);mpz_sub(V,V,P416);
    mpz_set_ui(vk,0);for(int i=7;i>=0;i--){mpz_mul_2exp(vk,vk,52);mpz_add_ui(vk,vk,(unsigned long)la[i]);}
    mpz_sub(t,vk,V);long e=mpz_get_si(t);if(e<g_lo)g_lo=e;if(e>g_hi)g_hi=e;
    if(e){if(nbad<5)gmp_printf("  MISMATCH err=%ld I=%#Zx\n",e,mI);nbad++;}mpz_clears(mI,V,num,vk,t,NULL);}
int main(int c,char**v){mpz_inits(P416,P1248,NULL);mpz_ui_pow_ui(P416,2,416);mpz_ui_pow_ui(P1248,2,1248);
 uint64_t s=c>1?strtoull(v[1],0,0):0xABCDEF;
 for(long t=0;t<1000000;t++){uint64_t I[13];for(int i=0;i<13;i++)I[i]=rng(&s);I[12]|=1ull<<63;check(I);}
 for(long t=0;t<300000;t++){uint64_t I[13];for(int i=0;i<6;i++)I[i]=~0ull;for(int i=6;i<13;i++)I[i]=rng(&s);I[12]|=1ull<<63;check(I);}
 for(long k=0;k<300000;k++){uint64_t I[13]={0};I[12]=1ull<<63;I[0]=(uint64_t)k;check(I);}
 printf("recip_newton vs exact V: err[%ld,%ld] mismatch=%ld\n",g_lo,g_hi,nbad);return nbad?1:0;}
#endif
