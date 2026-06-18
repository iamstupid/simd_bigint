// mul_u52_vec (per-block canon) vs mul_u52_vec_fast (fast-path canon) vs
// mul_u52_1 (+canon).  Correctness vs mul_u52_1+canon (no GMP), then bench.
#define _GNU_SOURCE
#include "../include/mul.h"
#include "../include/mul_vec.h"
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
static int nblk(int n){return (n+8+7)/8;}
// correctness: fast vs reference (mul_u52_1+canon).  'force' sprinkles all-ones
// limbs to exercise the cold path.
static int test_n(int n,uint64_t*s,int force){
  alignas(64) uint64_t A[8*64]={0},Bm[8];
  for(int i=0;i<n;i++)A[i]=(force&&((rng(s)&3)==0))?M52:(rng(s)&M52);
  for(int i=0;i<8;i++)Bm[i]=(force&&((rng(s)&3)==0))?M52:(rng(s)&M52);
  alignas(64) _vec rf[64],rr[64];memset(rf,0,sizeof rf);memset(rr,0,sizeof rr);
  mul_u52_vec_fast(rf,A,load_vec((cpvec)Bm),n);
  if(n>=8) mul_u52_1(rr,(cpvec)A,(cpvec)Bm,n,8); else mul_u52_1(rr,(cpvec)Bm,(cpvec)A,8,n);
  mpn_canon_pos((pvec)rr,(cpvec)rr,n+8);
  return memcmp(rf,rr,(size_t)nblk(n)*sizeof(_vec));
}
int main(int c,char**v){
  cpu_set_t st;CPU_ZERO(&st);CPU_SET(0,&st);sched_setaffinity(0,sizeof st,&st);
  uint64_t s=c>1?strtoull(v[1],0,0):0xABCDEF;
  int bad=0,badf=0;
  for(int n=1;n<=200;n++)for(int r=0;r<1000;r++){if(test_n(n,&s,0))bad++;}
  for(int n=1;n<=200;n++)for(int r=0;r<2000;r++){if(test_n(n,&s,1))badf++;}   // force cold path
  printf("fast-path correctness: random %s (%d) ; forced-allones %s (%d)\n",
         bad?"FAIL":"OK",bad, badf?"FAIL":"OK",badf);
  double g=ghz();
  int sizes[]={16,32,64,128,256};
  printf("\n n    vec(per-blk)  vec_fast    mul_u52_1+canon   mul_u52_1\n");
  for(int si=0;si<5;si++){
    int n=sizes[si];
    alignas(64) uint64_t A[8*64]={0},Bm[8];
    for(int i=0;i<n;i++)A[i]=rng(&s)&M52; for(int i=0;i<8;i++)Bm[i]=rng(&s)&M52;
    _vec Bv=load_vec((cpvec)Bm);
    alignas(64) _vec rr[64];memset(rr,0,sizeof rr);
    volatile uint64_t sink=0; const int IT=200000;
    double b0=1e300,bf=1e300,bc=1e300,b1=1e300;
    for(int r=0;r<9;r++){uint64_t t0=now();for(int i=0;i<IT;i++){mul_u52_vec(rr,A,Bv,n);sink^=((uint64_t*)rr)[0];}uint64_t t1=now();double ns=(double)(t1-t0)/IT;if(ns<b0)b0=ns;}
    for(int r=0;r<9;r++){uint64_t t0=now();for(int i=0;i<IT;i++){mul_u52_vec_fast(rr,A,Bv,n);sink^=((uint64_t*)rr)[0];}uint64_t t1=now();double ns=(double)(t1-t0)/IT;if(ns<bf)bf=ns;}
    for(int r=0;r<9;r++){uint64_t t0=now();for(int i=0;i<IT;i++){mul_u52_1(rr,(cpvec)A,(cpvec)Bm,n,8);mpn_canon_pos((pvec)rr,(cpvec)rr,n+8);sink^=((uint64_t*)rr)[0];}uint64_t t1=now();double ns=(double)(t1-t0)/IT;if(ns<bc)bc=ns;}
    for(int r=0;r<9;r++){uint64_t t0=now();for(int i=0;i<IT;i++){mul_u52_1(rr,(cpvec)A,(cpvec)Bm,n,8);sink^=((uint64_t*)rr)[0];}uint64_t t1=now();double ns=(double)(t1-t0)/IT;if(ns<b1)b1=ns;}
    printf("%4d   %6.1f cyc   %6.1f cyc   %6.1f cyc      %6.1f cyc\n",n,b0*g,bf*g,bc*g,b1*g);
  }
  printf("ghz %.3f\n",g);
  return (bad||badf)?1:0;
}
