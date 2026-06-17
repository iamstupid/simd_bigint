// Throughput + latency of the fused recip_3by2 kernel, vs recip_mono alone
// (the scalar seed it inlines) and vs GMP mpn reciprocal at the same size.
//   build: gcc -O3 -march=native -I.. recip_3by2_bench.c ../recip_3by2.s \
//          ../reciprocal_u416.s -o /tmp/b3 -lgmp && /tmp/b3
#define _GNU_SOURCE
#include <gmp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
__m512i recip_3by2(const uint64_t*);
void recip_mono(const uint64_t*, uint64_t*);
static uint64_t now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec*1000000000ull+t.tv_nsec;}
static uint64_t rng(uint64_t*s){uint64_t x=*s;x^=x<<7;x^=x>>9;x*=0x9e3779b97f4a7c15ull;*s=x;return x;}
static double ghz(){uint64_t a=1;const uint64_t N=2000000000ull;for(uint64_t i=0;i<50000000ull;i++)a+=a>>1^i;uint64_t t0=now();__asm__ volatile(""::"r"(a));for(uint64_t i=0;i<N;i++)__asm__ volatile("add $1,%0":"+r"(a));uint64_t t1=now();return (double)N/(t1-t0);}
int main(int c,char**v){
  cpu_set_t st;CPU_ZERO(&st);CPU_SET(0,&st);sched_setaffinity(0,sizeof st,&st);
  uint64_t s=c>1?strtoull(v[1],0,0):0x1234;double g=ghz();
  const int M=4096; static uint64_t T[4096][13];
  for(int i=0;i<M;i++){for(int j=0;j<13;j++)T[i][j]=rng(&s);T[i][12]|=1ull<<63;}
  volatile uint64_t sink=0;
  // ---- recip_3by2 throughput ----
  for(int i=0;i<M;i++) sink^=_mm512_cvtsi512_si32(recip_3by2(T[i]));
  double best=1e300;
  for(int r=0;r<9;r++){uint64_t t0=now();for(int l=0;l<100;l++)for(int i=0;i<M;i++)sink^=_mm512_cvtsi512_si32(recip_3by2(T[i]));uint64_t t1=now();double ns=(double)(t1-t0)/(100.0*M);if(ns<best)best=ns;}
  printf("recip_3by2  throughput: %.2f ns  %.1f cyc\n",best,best*g);
  // ---- recip_3by2 latency (chain output lane0 into next input top) ----
  { uint64_t I[13]; for(int j=0;j<13;j++)I[j]=T[0][j];
    double bl=1e300;
    for(int r=0;r<9;r++){uint64_t t0=now();for(int i=0;i<2000000;i++){__m512i o=recip_3by2(I);uint64_t l0=_mm512_cvtsi512_si32(o)|((uint64_t)_mm512_cvtsi512_si32(o)<<32);I[0]^=l0;I[12]|=1ull<<63;}uint64_t t1=now();double ns=(double)(t1-t0)/2000000.0;if(ns<bl)bl=ns;}
    printf("recip_3by2  latency   : %.2f ns  %.1f cyc\n",bl,bl*g);
  }
  // ---- recip_mono alone (the inlined scalar seed) ----
  { uint64_t o[7]; for(int i=0;i<M;i++){recip_mono(T[i]+6,o);sink^=o[0];}
    double bm=1e300;
    for(int r=0;r<9;r++){uint64_t t0=now();for(int l=0;l<100;l++)for(int i=0;i<M;i++){recip_mono(T[i]+6,o);sink^=o[3];}uint64_t t1=now();double ns=(double)(t1-t0)/(100.0*M);if(ns<bm)bm=ns;}
    printf("recip_mono  throughput: %.2f ns  %.1f cyc   (scalar seed only)\n",bm,bm*g);
  }
  printf("ghz %.3f  sink=%lx\n",g,(unsigned long)sink);
  return 0;}
