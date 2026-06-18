// Benchmark mul_zmm (8x8 -> 16-limb block multiply, div.h), with a correctness
// check vs mul_u52_1 (both canonicalized).
#define _GNU_SOURCE
#include "../include/div.h"
#include "../include/mul.h"
#include <sched.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
static uint64_t now(){timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec*1000000000ull+t.tv_nsec;}
static double ghz(){uint64_t a=1;const uint64_t N=2000000000ull;for(uint64_t i=0;i<50000000ull;i++)a+=a>>1^i;uint64_t t0=now();__asm__ volatile(""::"r"(a));for(uint64_t i=0;i<N;i++)__asm__ volatile("add $1,%0":"+r"(a));uint64_t t1=now();return (double)N/(t1-t0);}
static uint64_t rng(uint64_t*s){uint64_t x=*s;x^=x<<7;x^=x>>9;x*=0x9e3779b97f4a7c15ull;*s=x;return x;}
#define M52 ((1ull<<52)-1)
int main(int c,char**v){
  cpu_set_t st;CPU_ZERO(&st);CPU_SET(0,&st);sched_setaffinity(0,sizeof st,&st);
  uint64_t s=c>1?strtoull(v[1],0,0):0xABCDEF;
  // ---- correctness vs mul_u52_1 ----
  int bad=0;
  for(int t=0;t<2000000;t++){
    alignas(64) uint64_t A[8],B[8];
    for(int i=0;i<8;i++){A[i]=rng(&s)&M52;B[i]=rng(&s)&M52;}
    _u832 r = mul_zmm(A, load_vec((cpvec)B));
    alignas(64) _vec z[2]={r.lo,r.hi}; mpn_canon_pos((pvec)z,(cpvec)z,16);
    alignas(64) _vec rr[2]; mul_u52_1(rr,(cpvec)A,(cpvec)B,8,8); mpn_canon_pos((pvec)rr,(cpvec)rr,16);
    if(memcmp(z,rr,sizeof z)) bad++;
  }
  printf("mul_zmm correctness vs mul_u52_1: %s (%d)\n", bad?"FAIL":"OK", bad);
  double g=ghz();
  // ---- throughput: table of independent inputs ----
  const int M=4096;
  static alignas(64) uint64_t Atab[4096][8], Btab[4096][8];
  for(int i=0;i<M;i++)for(int j=0;j<8;j++){Atab[i][j]=rng(&s)&M52;Btab[i][j]=rng(&s)&M52;}
  static _vec Bv[4096]; for(int i=0;i<M;i++)Bv[i]=load_vec((cpvec)Btab[i]);
  volatile uint64_t sink=0; const int LOOPS=400;
  for(int i=0;i<M;i++){_u832 r=mul_zmm(Atab[i],Bv[i]);sink^=(uint64_t)_mm512_cvtsi512_si32(r.lo);}
  double bt=1e300;
  for(int rep=0;rep<9;rep++){uint64_t t0=now();for(int l=0;l<LOOPS;l++)for(int i=0;i<M;i++){_u832 r=mul_zmm(Atab[i],Bv[i]);sink^=(uint64_t)_mm512_cvtsi512_si32(r.lo);}uint64_t t1=now();double ns=(double)(t1-t0)/((double)LOOPS*M);if(ns<bt)bt=ns;}
  printf("mul_zmm throughput: %.2f ns  %.1f cyc\n",bt,bt*g);
  // ---- latency: feed lo back into a ----
  { alignas(64) uint64_t A[8]; for(int i=0;i<8;i++)A[i]=Atab[0][i]; _vec B=Bv[0];
    double bl=1e300;
    for(int rep=0;rep<9;rep++){uint64_t t0=now();for(int i=0;i<8000000;i++){_u832 r=mul_zmm(A,B);_mm512_store_si512((void*)A,_mm512_and_si512(r.lo,_mm512_set1_epi64(M52)));}uint64_t t1=now();double ns=(double)(t1-t0)/8000000.0;if(ns<bl)bl=ns;}
    sink^=A[0];
    printf("mul_zmm latency   : %.2f ns  %.1f cyc  (lo->a chain)\n",bl,bl*g);
  }
  printf("ghz %.3f sink=%lx\n",g,(unsigned long)sink);
  return bad?1:0;
}
