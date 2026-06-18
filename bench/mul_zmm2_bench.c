// mul_zmm_2 (16x8 -> low 16 limbs = (a*b) mod 2^832): correctness vs
// mul_u52_1(16,8) low 16 limbs, then robust throughput + reg-resident latency.
#define _GNU_SOURCE
#include "../include/div.h"
#include "../include/mul.h"
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static uint64_t now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec*1000000000ull+t.tv_nsec;}
static double ghz(){uint64_t a=1;const uint64_t N=2000000000ull;for(uint64_t i=0;i<50000000ull;i++)a+=a>>1^i;uint64_t t0=now();__asm__ volatile(""::"r"(a));for(uint64_t i=0;i<N;i++)__asm__ volatile("add $1,%0":"+r"(a));uint64_t t1=now();return (double)N/(t1-t0);}
static uint64_t rng(uint64_t*s){uint64_t x=*s;x^=x<<7;x^=x>>9;x*=0x9e3779b97f4a7c15ull;*s=x;return x;}
#define M52 ((1ull<<52)-1)
int main(int c,char**v){
  cpu_set_t st;CPU_ZERO(&st);CPU_SET(0,&st);sched_setaffinity(0,sizeof st,&st);
  uint64_t s=c>1?strtoull(v[1],0,0):0xABCDEF; double g=ghz();
  // ---- correctness: low 16 limbs of a(16)*b(8) ----
  int bad=0;
  for(int t=0;t<2000000;t++){
    alignas(64) uint64_t A[16],B[8];
    for(int i=0;i<16;i++)A[i]=rng(&s)&M52; for(int i=0;i<8;i++)B[i]=rng(&s)&M52;
    _u832 r = mul_zmm_2(A, load_vec((cpvec)B));
    alignas(64) _vec z[2]={r.lo,r.hi}; mpn_canon_pos((pvec)z,(cpvec)z,16);
    alignas(64) _vec rr[3]; mul_u52_1(rr,(cpvec)A,(cpvec)B,16,8); mpn_canon_pos((pvec)rr,(cpvec)rr,24);
    if(memcmp(z,rr,2*sizeof(_vec))) bad++;   // low 16 limbs
  }
  printf("mul_zmm_2 correctness (low 16 vs mul_u52_1): %s (%d)\n", bad?"FAIL":"OK", bad);
  const int M=4096, MM=M-1;
  static alignas(64) uint64_t Atab[4096][16], Btab[4096][8];
  for(int i=0;i<M;i++){for(int j=0;j<16;j++)Atab[i][j]=rng(&s)&M52; for(int j=0;j<8;j++)Btab[i][j]=rng(&s)&M52;}
  static _vec Bv[4096]; for(int i=0;i<M;i++)Bv[i]=load_vec((cpvec)Btab[i]);
  volatile uint64_t sink=0; const int LOOPS=400;
  { _vec acc=zero(); double bt=1e300;
    for(int rep=0;rep<9;rep++){uint64_t t0=now();for(int l=0;l<LOOPS;l++)for(int i=0;i<M;i++){int idx=(i+l)&MM;_u832 r=mul_zmm_2(Atab[idx],Bv[idx]);acc=_mm512_xor_si512(acc,_mm512_xor_si512(r.lo,r.hi));}uint64_t t1=now();double ns=(double)(t1-t0)/((double)LOOPS*M);if(ns<bt)bt=ns;}
    sink^=_mm512_reduce_add_epi64(acc); printf("mul_zmm_2 throughput: %.2f cyc\n",bt*g); }
  // context: mul_u52_1(16,8) full 24-limb product
  { _vec acc=zero(); static _vec rr[3]; double br=1e300;
    for(int rep=0;rep<9;rep++){uint64_t t0=now();for(int l=0;l<LOOPS;l++)for(int i=0;i<M;i++){int idx=(i+l)&MM;mul_u52_1(rr,(cpvec)Atab[idx],(cpvec)Btab[idx],16,8);acc=_mm512_xor_si512(acc,_mm512_xor_si512(_mm512_xor_si512(rr[0],rr[1]),rr[2]));}uint64_t t1=now();double ns=(double)(t1-t0)/((double)LOOPS*M);if(ns<br)br=ns;}
    sink^=_mm512_reduce_add_epi64(acc); printf("mul_u52_1(16,8) thr : %.2f cyc  (full 24-limb, for context)\n",br*g); }
  { alignas(64) uint64_t A[16]; for(int i=0;i<16;i++)A[i]=Atab[0][i]; _vec b=Bv[0],MASK=_mm512_set1_epi64(M52); double bl=1e300;
    for(int rep=0;rep<9;rep++){uint64_t t0=now();for(int i=0;i<8000000;i++){_u832 r=mul_zmm_2(A,b);b=_mm512_and_si512(r.lo,MASK);}uint64_t t1=now();double ns=(double)(t1-t0)/8000000.0;if(ns<bl)bl=ns;}
    sink^=_mm512_cvtsi512_si32(b); printf("mul_zmm_2 latency(lo->b reg): %.2f cyc\n",bl*g); }
  printf("ghz %.3f sink=%lx\n",g,(unsigned long)sink); return bad?1:0;}
