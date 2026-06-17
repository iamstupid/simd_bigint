// recip_3by2_ref: literal pre-image of the monolithic asm kernel.
//
// Mirrors the user's pseudocode 1:1 so the asm port is a transliteration:
//   * vpermb + vpsrlvq seed load  (bits[31..446] of recip_mono -> 8x u52)
//   * 16x2 madd52lo/hi schoolbook  seed(8 lanes) * D(16 broadcast digits)
//   * alignr64 diagonal reduction  -> 3 redundant blocks -> canonize
//   * T = [all1, ~i0, ~i1];  DEC if P>T (free) else +I, INC if P+I<=T
//
//   build/test: gcc -O3 -march=native -I.. recip_3by2_ref.c \
//               ../reciprocal_u416.s -o /tmp/kref -lgmp -DREF_TEST && /tmp/kref
#include "../include/types.h"
#include "../include/canon.h"   // carry_prop (canonize is #undef'd at EOF -> re-add below)
#include "../include/u64cvt.h"  // u52_from_u64

#ifdef __cplusplus
extern "C"
#endif
void recip_mono(const uint64_t *D, uint64_t *xout);

// canonize is private to canon.h (undef'd at end); the kernel needs it, so
// reproduce it verbatim here -- this is exactly what the asm block implements.
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

// --- seed load: bits[31..446] of the 7-limb recip_mono output -> 8x u52 -----
static const uint8_t seed_byte_perm[64] = {
     3, 4, 5, 6, 7, 8, 9,10,   10,11,12,13,14,15,16,17,
    16,17,18,19,20,21,22,23,   23,24,25,26,27,28,29,30,
    29,30,31,32,33,34,35,36,   36,37,38,39,40,41,42,43,
    42,43,44,45,46,47,48,49,   49,50,51,52,53,54,55,56,
};
static inline _vec seed_load(const uint64_t R[7]){
    const _vec perm  = load_vec((cpvec)seed_byte_perm);
    const _vec vshft = setr_64(7,3,7,3,7,3,7,3);
    _vec w = load_vec(R, 0x7f);                 // 7 qwords, lane7 = 0
    return and_v(srlv(permb(perm, w), vshft), MASK52);
}

// --- inc/dec single block (verified in recip_3by2_kernel.c) -----------------
static inline _vec inc52(_vec v){
    __mmask8 p = eq(v, MASK52);
    unsigned chain = (unsigned)p + 1u;
    __mmask8 flip = (__mmask8)((unsigned)p ^ chain);
    v = add(v, set1_64(1), kandn(p, flip), v);
    v = and_v(v, zero(), kand(p, flip), v);
    return v;
}
static inline _vec dec52(_vec v){
    __mmask8 z = eq(v, zero());
    unsigned chain = (unsigned)z + 1u;
    __mmask8 flip = (__mmask8)((unsigned)z ^ chain);
    v = sub(v, set1_64(1), kandn(z, flip), v);
    v = or_v(v, MASK52, kand(z, flip), v);
    return v;
}

// unsigned 2-block compare: returns 1 iff [a1:a0] > [b1:b0] (a1/b1 high block)
static inline int gt2(_vec a0, _vec a1, _vec b0, _vec b1){
    uint8_t ne = (uint8_t)neq(a1,b1);
    if(ne){ uint8_t lt=(uint8_t)ltu(a1,b1); return ((uint8_t)(ne^lt) > lt); }
    ne = (uint8_t)neq(a0,b0);
    if(!ne) return 0;
    uint8_t lt=(uint8_t)ltu(a0,b0); return ((uint8_t)(ne^lt) > lt);
}

_vec recip_3by2_ref(const uint64_t I[13]){
    // 1. scalar 448-bit reciprocal of the high part D = floor(I/2^384)
    uint64_t recip_buf[7];
    recip_mono(I + 6, recip_buf);

    // 2. seed block
    _vec seed = seed_load(recip_buf);

    // 3. divisor -> two canonical u52 blocks i0(low),i1(high); also flat dd[16]
    _vec I52[2];
    u52_from_u64(I52, I, 13);
    const _vec i0 = I52[0], i1 = I52[1];
    uint64_t dd[16];
    store_vec((pvec)dd, i0);
    store_vec((pvec)(dd+8), i1);

    // 4. reverse product P = seed * D  (seed = 8-lane multiplicand, dd broadcast)
    _vec temp[17];
    temp[0] = zero();
    for(int i=0;i<16;i++){
        _vec b = set1_64(dd[i]);
        temp[i]   = madd52lo(temp[i], seed, b);
        temp[i+1] = madd52hi(zero(), seed, b);
    }
    const _vec Z = zero();
    _vec carry = zero(); unsigned sig = 0;
    _vec P0,P1,P2,k1,k2,k3,k4;
    // block 0: sum_{i=0..7} lane_lsh(temp[i], i)   (i=0 is identity; alignr imm
    // wraps mod 8 so an imm-8 "shift" would yield 0 -- use the register directly)
    k1 = add(temp[0],                alignr64(temp[1],Z,7));
    k2 = add(alignr64(temp[2],Z,6), alignr64(temp[3],Z,5));
    k3 = add(alignr64(temp[4],Z,4), alignr64(temp[5],Z,3));
    k4 = add(alignr64(temp[6],Z,2), alignr64(temp[7],Z,1));
    P0 = add(add(k1,k2),add(k3,k4)); canonize(P0,carry,sig);
    // block 1: alignr64(temp[i+8], temp[i], 8-i)   (i=0 -> temp[8] identity)
    k1 = add(temp[8],                       alignr64(temp[9], temp[1],7));
    k2 = add(alignr64(temp[10],temp[2],6), alignr64(temp[11],temp[3],5));
    k3 = add(alignr64(temp[12],temp[4],4), alignr64(temp[13],temp[5],3));
    k4 = add(alignr64(temp[14],temp[6],2), alignr64(temp[15],temp[7],1));
    P1 = add(add(k1,k2),add(k3,k4)); canonize(P1,carry,sig);
    // block 2: temp[16] + high spills of rows 9..15   (temp[16] identity)
    k1 = add(temp[16],                     alignr64(Z,temp[9], 7));
    k2 = add(alignr64(Z,temp[10],6),       alignr64(Z,temp[11],5));
    k3 = add(alignr64(Z,temp[12],4),       alignr64(Z,temp[13],3));
    k4 = add(alignr64(Z,temp[14],2),       alignr64(Z,temp[15],1));
    P2 = add(add(k1,k2),add(k3,k4)); canonize(P2,carry,sig);

    // 5. target T = [all1, ~i0, ~i1]  (compare uses high two blocks)
    const _vec c1 = xor_v(i0, MASK52);   // ~i0
    const _vec c2 = xor_v(i1, MASK52);   // ~i1

    // 6. correction
    if(gt2(P1,P2, c1,c2)){
        seed = dec52(seed);
    } else {
        _vec Q[3]; __mmask16 cy = 0; pvec q = Q;   // Q = P + I
        carry_prop(q, P0, i0,  cy);
        carry_prop(q, P1, i1,  cy);
        carry_prop(q, P2, Z,   cy);
        if(!gt2(Q[1],Q[2], c1,c2))
            seed = inc52(seed);
    }
    return seed;
}

// ---------------------------------------------------------------------------
#ifdef REF_TEST
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>
static uint64_t rng(uint64_t*s){uint64_t x=*s;x^=x<<7;x^=x>>9;x*=0x9e3779b97f4a7c15ull;*s=x;return x;}
static long g_lo=1<<30,g_hi=-(1<<30),nbad=0; static mpz_t P416,P1248;
static void check(const uint64_t I[13]){
    _vec out=recip_3by2_ref(I);
    uint64_t lane[8]; store_vec((pvec)lane,out);
    mpz_t mI,V,num,vk,t; mpz_inits(mI,V,num,vk,t,NULL);
    mpz_import(mI,13,-1,8,0,0,I);
    mpz_sub_ui(num,P1248,1); mpz_fdiv_q(V,num,mI); mpz_sub(V,V,P416);
    mpz_set_ui(vk,0); for(int i=7;i>=0;i--){mpz_mul_2exp(vk,vk,52);mpz_add_ui(vk,vk,(unsigned long)lane[i]);}
    mpz_sub(t,vk,V); long e=mpz_get_si(t);
    if(e<g_lo)g_lo=e; if(e>g_hi)g_hi=e;
    if(e){ if(nbad<4) gmp_printf("  MISMATCH err=%ld  I=%#Zx\n",e,mI); nbad++; }
    mpz_clears(mI,V,num,vk,t,NULL);
}
int main(int argc,char**argv){
    mpz_inits(P416,P1248,NULL); mpz_ui_pow_ui(P416,2,416); mpz_ui_pow_ui(P1248,2,1248);
    uint64_t s=argc>1?strtoull(argv[1],0,0):0xABCDEF;
    long N=3000000;
    for(long t=0;t<N;t++){ uint64_t I[13]; for(int i=0;i<13;i++)I[i]=rng(&s); I[12]|=0x8000000000000000ull; check(I); }
    for(long t=0;t<1000000;t++){ uint64_t I[13]; for(int i=0;i<6;i++)I[i]=~0ull; for(int i=6;i<13;i++)I[i]=rng(&s); I[12]|=0x8000000000000000ull; check(I); }
    for(long k=0;k<1000000;k++){ uint64_t I[13]={0}; I[12]=0x8000000000000000ull; I[0]=(uint64_t)k; check(I); }
    for(long t=0;t<1000000;t++){ uint64_t I[13]; for(int i=0;i<13;i++)I[i]=~0ull; I[12]^=(rng(&s)&0x7fffffffffffffffull); I[12]|=0x8000000000000000ull; check(I); }
    { uint64_t I[13]={0}; for(int i=0;i<6;i++)I[i]=~0ull; I[6]=0xffffffffull; I[12]=0x8000000000000000ull; check(I); } // I0=B-1
    printf("recip_3by2_ref vs exact V: err in [%ld,%ld]  mismatches=%ld\n",g_lo,g_hi,nbad);
    return nbad?1:0;
}
#endif
