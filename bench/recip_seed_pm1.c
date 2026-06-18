// Validate the stripped seed kernel (src/avx512/recip_3by2_seed.s): the
// truncated recip_mono load is within +-1 of the exact 3/2 reciprocal V.
// Proof in 3by2_recip.md Sec.6; here we confirm seed - V in {-1,0} empirically.
//   build: gcc -O3 -march=native recip_seed_pm1.c ../src/avx512/recip_3by2_seed.s -lgmp
#include <gmp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
__m512i recip_3by2_seed(const uint64_t*);
static uint64_t rng(uint64_t*s){uint64_t x=*s;x^=x<<7;x^=x>>9;x*=0x9e3779b97f4a7c15ull;*s=x;return x;}
static long g_lo=1<<30,g_hi=-(1<<30),nbad=0; static mpz_t P416,P1248;
static void check(const uint64_t I[13]){
    __m512i o=recip_3by2_seed(I); uint64_t la[8]; _mm512_storeu_si512(la,o);
    mpz_t mI,V,num,vk,t; mpz_inits(mI,V,num,vk,t,NULL); mpz_import(mI,13,-1,8,0,0,I);
    mpz_sub_ui(num,P1248,1); mpz_fdiv_q(V,num,mI); mpz_sub(V,V,P416);
    mpz_set_ui(vk,0); for(int i=7;i>=0;i--){mpz_mul_2exp(vk,vk,52);mpz_add_ui(vk,vk,(unsigned long)la[i]);}
    mpz_sub(t,vk,V); long e=mpz_get_si(t); if(e<g_lo)g_lo=e; if(e>g_hi)g_hi=e;
    if(e< -1 || e>1){ if(nbad<4) gmp_printf("  OUT OF RANGE e=%ld I=%#Zx\n",e,mI); nbad++; }
    mpz_clears(mI,V,num,vk,t,NULL);
}
int main(int c,char**v){ mpz_inits(P416,P1248,NULL); mpz_ui_pow_ui(P416,2,416); mpz_ui_pow_ui(P1248,2,1248);
 uint64_t s=c>1?strtoull(v[1],0,0):0xABCDEF;
 for(long t=0;t<3000000;t++){uint64_t I[13];for(int i=0;i<13;i++)I[i]=rng(&s);I[12]|=1ull<<63;check(I);}
 for(long t=0;t<1000000;t++){uint64_t I[13];for(int i=0;i<6;i++)I[i]=~0ull;for(int i=6;i<13;i++)I[i]=rng(&s);I[12]|=1ull<<63;check(I);}
 for(long t=0;t<1000000;t++){uint64_t I[13];for(int i=0;i<13;i++)I[i]=~0ull;I[12]^=(rng(&s)&0x7fffffffffffffffull);I[12]|=1ull<<63;check(I);}
 for(long k=0;k<500000;k++){uint64_t I[13]={0};I[12]=1ull<<63;I[0]=(uint64_t)k;check(I);}
 {uint64_t I[13]={0};for(int i=0;i<6;i++)I[i]=~0ull;I[6]=0xffffffffull;I[12]=1ull<<63;check(I);} // I0=B-1
 printf("seed - V in [%ld,%ld]   out-of-[-1,1]=%ld\n",g_lo,g_hi,nbad);
 return (g_lo< -1||g_hi>1)?1:0; }
