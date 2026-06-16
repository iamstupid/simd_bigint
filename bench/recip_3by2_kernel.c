// recip_3by2: C form of the monolithic AVX-512 3/2-reciprocal kernel.
//
//   in : I = u64[13], normalized 832-bit divisor (top bit of I[12] set)
//   out: one zmm (8 u52 lanes) = V = floor((B^3-1)/I) - B,  B = 2^416
//        the exact 3/2 block reciprocal (drop the implicit leading B).
//
// Pipeline (each step maps 1:1 to an asm stage):
//   1. recip_mono on the top 7 u64 (= floor(I/2^384))      -> 448-bit seed
//   2. from_u64_for_reciprocal: bits[31..446] -> 8x u52    -> seed block `recip`
//   3. decode I -> two canonical u52 blocks  i0 (low), i1 (high)
//   4. reverse product  P = recip * I  (mul1, 3 blocks)  then canonize
//   5. target  T = B^3-1 - B*I = [all1, ~i0, ~i1]         (just NOT the divisor)
//   6. correct (|err|<=1):  DEC if P>T (free) ; else INC if P+I<=T (+I)
//
// This is the correctness blueprint; the multiply/canonize/compare already use
// the project's SIMD primitives, so the asm port is mostly scheduling.
//
//   build/test: gcc -O3 -march=native -I.. recip_3by2_kernel.c ../reciprocal_u416.s -o /tmp/k -lgmp
#define SIMD_MPN_FFT 0
#include "../include/canon.h"
#include "../include/mul.h"
#include "../include/u64cvt.h"

#ifdef __cplusplus
extern "C"
#endif
void recip_mono(const uint64_t *D, uint64_t *xout);

// --- step 2: 448-bit reciprocal value -> 8 u52 lanes ----------------------
// Extract bits[31..446] of the 7-limb value R (lane i = bits[31+52i .. +51]).
// The leading 1 (bit 447) sits just above lane 7 -> naturally dropped; starting
// at bit 31 (not 32) is the "extra low bit" that offsets recip_mono's 2^895.
static inline _vec from_u64_for_reciprocal(const uint64_t R[7]){
    uint64_t lane[8];
    for(int i=0;i<8;i++){
        uint64_t b = 31u + 52u*(uint64_t)i, w = b>>6, off = b&63;
        uint64_t lo = R[w] >> off;
        uint64_t hi = (off && w+1<7) ? (R[w+1] << (64-off)) : 0;   // bit 447 never reached
        lane[i] = (lo | hi) & ((1ull<<52)-1);
    }
    return setr_64(lane[0],lane[1],lane[2],lane[3],lane[4],lane[5],lane[6],lane[7]);
}

// --- step 6 helpers: single-block +-1 with SWAR carry/borrow ripple -------
static inline _vec inc52(_vec v){
    __mmask8 p = eq(v, MASK52);                 // all-ones lanes propagate carry
    unsigned chain = (unsigned)p + 1u;          // inject +1 at lane 0
    __mmask8 flip = (__mmask8)((unsigned)p ^ chain);
    v = add(v, set1_64(1), kandn(p, flip), v);  // absorbing lane (flip & ~p) += 1
    v = and_v(v, zero(), kand(p, flip), v);     // propagated all-ones lanes -> 0
    return v;                                   // carry-out = chain>>8 (seed<B-1: never)
}
static inline _vec dec52(_vec v){
    __mmask8 z = eq(v, zero());                 // zero lanes propagate borrow
    unsigned chain = (unsigned)z + 1u;
    __mmask8 flip = (__mmask8)((unsigned)z ^ chain);
    v = sub(v, set1_64(1), kandn(z, flip), v);  // absorbing lane (flip & ~z) -= 1
    v = or_v(v, MASK52, kand(z, flip), v);      // borrowed zero lanes -> all-ones
    return v;
}

// unsigned compare of two canonical blocks (lane 7 most significant): +1/-1/0
static inline int cmp_block(_vec a, _vec b){
    uint8_t ne = (uint8_t)neq(a,b);
    if(!ne) return 0;
    uint8_t lt = (uint8_t)ltu(a,b);
    uint8_t gt = (uint8_t)(ne ^ lt);            // lt,gt disjoint; lt>gt iff top diff lane is a<b
    return (lt > gt) ? -1 : 1;
}
static inline int cmp3(const _vec P[3], const _vec T[3]){
    int c = cmp_block(P[2],T[2]); if(c) return c;
    c     = cmp_block(P[1],T[1]); if(c) return c;
    return  cmp_block(P[0],T[0]);
}

_vec recip_3by2(const uint64_t I[13]){
    // 1. scalar 448-bit reciprocal of the high part D = floor(I/2^384)
    uint64_t recip_buf[7];
    recip_mono(I + 6, recip_buf);                 // ~ floor(2^895 / D)

    // 2. seed block
    _vec recip = from_u64_for_reciprocal(recip_buf);

    // 3. divisor -> two canonical u52 blocks
    _vec I52[2];
    u52_from_u64(I52, I, 13);                     // I52[0]=i0 (low 416b), I52[1]=i1 (high)
    const _vec i0 = I52[0], i1 = I52[1];

    // 4. reverse product P = recip * I  (a=I streamed 16 limbs, b=recip 8 broadcast)
    _vec P[3];
    mul_u52_1(P, (cpvec)I52, (cpvec)&recip, 16, 8);
    mpn_canon_pos((pvec)P, (cpvec)P, 24);          // redundant -> canonical 3 blocks

    // 5. target T = B^3-1 - B*I = [all1, ~i0, ~i1]
    const _vec T[3] = { MASK52, xor_v(i0, MASK52), xor_v(i1, MASK52) };

    // 6. correct: at most one of DEC / INC fires (|seed - V| <= 1)
    if(cmp3(P, T) > 0){
        recip = dec52(recip);                     // seed too big  (P > T, free)
    } else {
        _vec Q[3]; __mmask16 cy = 0; pvec q = Q;  // Q = P + I
        carry_prop(q, P[0], i0,     cy);
        carry_prop(q, P[1], i1,     cy);
        carry_prop(q, P[2], zero(), cy);
        if(cmp3((const _vec*)Q, T) <= 0)
            recip = inc52(recip);                 // seed too small (P + I <= T)
    }
    return recip;                                  // == V
}

// ---------------------------------------------------------------------------
#ifdef KERNEL_TEST
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint64_t rng(uint64_t*s){uint64_t x=*s;x^=x<<7;x^=x>>9;x*=0x9e3779b97f4a7c15ull;*s=x;return x;}

static long g_lo=1<<30, g_hi=-(1<<30), nbad=0;
static mpz_t P416,P1248;
static void check(const uint64_t I[13]){
    _vec out = recip_3by2(I);
    uint64_t lane[8]; store_vec((pvec)lane, out);
    mpz_t mI,V,num,vk,t; mpz_inits(mI,V,num,vk,t,NULL);
    mpz_import(mI,13,-1,8,0,0,I);
    // V = floor((2^1248-1)/I) - 2^416
    mpz_sub_ui(num,P1248,1); mpz_fdiv_q(V,num,mI); mpz_sub(V,V,P416);
    // vk = sum lane[i] << 52i
    mpz_set_ui(vk,0);
    for(int i=7;i>=0;i--){ mpz_mul_2exp(vk,vk,52); mpz_add_ui(vk,vk,(unsigned long)lane[i]); }
    mpz_sub(t,vk,V); long e=mpz_get_si(t);
    if(e<g_lo)g_lo=e; if(e>g_hi)g_hi=e;
    if(e){ if(nbad<3) gmp_printf("  MISMATCH err=%ld  I=%#Zx\n",e,mI); nbad++; }
    mpz_clears(mI,V,num,vk,t,NULL);
}
// directly validate the hand-derived SWAR inc52/dec52 against scalar +-1
static int unit_incdec(uint64_t *s){
    int bad=0;
    for(long t=0;t<2000000;t++){
        uint64_t in[8], expi[8], expd[8];
        for(int i=0;i<8;i++) in[i]=rng(s)&((1ull<<52)-1);
        if((t&7)==0) for(int i=0;i< (int)(rng(s)&7);i++) in[i]=(1ull<<52)-1; // force ones-runs
        // scalar reference (block as base-2^52 little-endian, ignore final carry/borrow)
        unsigned __int128 c=1; for(int i=0;i<8;i++){c+=in[i];expi[i]=(uint64_t)(c&((1ull<<52)-1));c>>=52;}
        long long b=-1; for(int i=0;i<8;i++){long long v=(long long)in[i]+b; if(v<0){v+=(1ll<<52);b=-1;}else b=0; expd[i]=(uint64_t)v;}
        _vec v = setr_64(in[0],in[1],in[2],in[3],in[4],in[5],in[6],in[7]);
        uint64_t oi[8],od[8]; store_vec((pvec)oi, inc52(v)); store_vec((pvec)od, dec52(v));
        for(int i=0;i<8;i++){ if(oi[i]!=expi[i]||od[i]!=expd[i]){bad++; break;} }
    }
    return bad;
}
int main(int argc,char**argv){
    mpz_inits(P416,P1248,NULL); mpz_ui_pow_ui(P416,2,416); mpz_ui_pow_ui(P1248,2,1248);
    uint64_t s = argc>1?strtoull(argv[1],0,0):0xABCDEF;
    { uint64_t su=s^0x55; int b=unit_incdec(&su); printf("inc52/dec52 unit: %s\n", b?"FAIL":"ok"); if(b) return 1; }
    long N=2000000;
    for(long t=0;t<N;t++){ uint64_t I[13]; for(int i=0;i<13;i++)I[i]=rng(&s); I[12]|=0x8000000000000000ull; check(I); }
    // adversarial: low all ones / minimal & maximal high / 2^831+k / I0=B-1
    for(long t=0;t<500000;t++){ uint64_t I[13]; for(int i=0;i<6;i++)I[i]=~0ull; for(int i=6;i<13;i++)I[i]=rng(&s); I[12]|=0x8000000000000000ull; check(I); }
    for(long k=0;k<500000;k++){ uint64_t I[13]={0}; I[12]=0x8000000000000000ull; I[0]=(uint64_t)k; check(I); }
    for(long t=0;t<500000;t++){ uint64_t I[13]; for(int i=0;i<13;i++)I[i]=~0ull; I[12]^=(rng(&s)&0x7fffffffffffffffull); I[12]|=0x8000000000000000ull; check(I); }
    { uint64_t I[13]; for(int i=0;i<7;i++)I[i]=(i<6)?~0ull:0xffffffffull; for(int i=7;i<12;i++)I[i]=0; I[6]=0xffffffffull; I[12]=0x8000000000000000ull; check(I);} // I0=B-1, I1 minimal
    printf("recip_3by2 vs exact V: err in [%ld,%ld]  mismatches=%ld\n", g_lo,g_hi,nbad);
    return nbad?1:0;
}
#endif
