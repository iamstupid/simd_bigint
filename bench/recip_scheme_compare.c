// Head-to-head reciprocal of the 832-bit divisor (the 416-bit v = floor((2^1248-1)/I)-2^416):
//   recip_newton    : Edamatsu-Takahashi pure-vector Newton (intrinsics C)
//   recip_3by2_ref  : our scalar-seed + verify scheme (intrinsics C)
//   recip_3by2      : our scheme, hand asm
#define _GNU_SOURCE
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#define NEWT_NOMAIN
#include "recip_newton.c"
__m512i recip_3by2(const uint64_t*);       // asm
_vec    recip_3by2_ref(const uint64_t*);    // our C
static uint64_t now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec*1000000000ull+t.tv_nsec;}
static uint64_t rng(uint64_t*s){uint64_t x=*s;x^=x<<7;x^=x>>9;x*=0x9e3779b97f4a7c15ull;*s=x;return x;}
static double ghz(){uint64_t a=1;const uint64_t N=2000000000ull;for(uint64_t i=0;i<50000000ull;i++)a+=a>>1^i;uint64_t t0=now();__asm__ volatile(""::"r"(a));for(uint64_t i=0;i<N;i++)__asm__ volatile("add $1,%0":"+r"(a));uint64_t t1=now();return (double)N/(t1-t0);}
static uint64_t T[4096][13];
#define THR(NAME, CALL) do{ for(int i=0;i<M;i++) sink^=(uint64_t)_mm512_cvtsi512_si32(CALL); \
  double best=1e300; for(int r=0;r<9;r++){uint64_t t0=now();for(int l=0;l<100;l++)for(int i=0;i<M;i++)sink^=(uint64_t)_mm512_cvtsi512_si32(CALL);uint64_t t1=now();double ns=(double)(t1-t0)/(100.0*M);if(ns<best)best=ns;} \
  printf("%-26s %.2f ns  %.1f cyc\n",NAME,best,best*g);}while(0)
int main(int c,char**v){
  cpu_set_t st;CPU_ZERO(&st);CPU_SET(0,&st);sched_setaffinity(0,sizeof st,&st);
  uint64_t s=c>1?strtoull(v[1],0,0):0x1234;double g=ghz();const int M=4096;volatile uint64_t sink=0;
  for(int i=0;i<M;i++){for(int j=0;j<13;j++)T[i][j]=rng(&s);T[i][12]|=1ull<<63;}
  THR("their newton (C, 3 it):", recip_newton(T[i]));
  THR("our recip_3by2 (C):",     recip_3by2_ref(T[i]));
  THR("our recip_3by2 (asm):",   recip_3by2(T[i]));
  printf("ghz %.3f sink=%lx\n",g,(unsigned long)sink);return 0;}
