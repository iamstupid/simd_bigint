#define _GNU_SOURCE
#include "../include/div.h"
#include <gmp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static uint64_t now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec*1000000000ull+t.tv_nsec;}
static double ghz(){uint64_t a=1;const uint64_t N=2000000000ull;for(uint64_t i=0;i<50000000ull;i++)a+=a>>1^i;uint64_t t0=now();__asm__ volatile(""::"r"(a));for(uint64_t i=0;i<N;i++)__asm__ volatile("add $1,%0":"+r"(a));uint64_t t1=now();return (double)N/(t1-t0);}
#define M52 ((1ull<<52)-1)
static void mpz_to_u52(uint64_t*L,int n,const mpz_t X){mpz_t t;mpz_init_set(t,X);for(int i=0;i<n;i++){L[i]=mpz_get_ui(t)&M52;mpz_fdiv_q_2exp(t,t,52);}mpz_clear(t);}
int main(int c,char**v){
  cpu_set_t st;CPU_ZERO(&st);CPU_SET(0,&st);sched_setaffinity(0,sizeof st,&st);double g=ghz();
  gmp_randstate_t rs;gmp_randinit_default(rs);gmp_randseed_ui(rs,c>1?strtoul(v[1],0,0):1);
  mpz_t D,U,UH,U0,beta;mpz_inits(D,U,UH,U0,beta,NULL);mpz_ui_pow_ui(beta,2,416);
  const int M=2048,MM=M-1;
  static alignas(64) uint64_t Ut[2048][24], Dt[2048][16]; static _vec Vt[2048];
  for(int i=0;i<M;i++){mpz_urandomb(D,rs,832);mpz_setbit(D,831);mpz_urandomm(UH,rs,D);mpz_urandomb(U0,rs,416);mpz_mul(U,UH,beta);mpz_add(U,U,U0);
    mpz_to_u52(Ut[i],24,U);mpz_to_u52(Dt[i],16,D);uint64_t D64[13]={0};size_t cn;mpz_export(D64,&cn,-1,8,0,0,D);Vt[i]=recip_3by2_seed(D64);}
  volatile uint64_t sink=0;const int LOOPS=2000;
  // throughput
  { _vec acc=zero();double bt=1e300;
    for(int rep=0;rep<9;rep++){uint64_t t0=now();for(int l=0;l<LOOPS;l++)for(int i=0;i<M;i++){int idx=(i+l)&MM;_div32 R=div3by2(Ut[idx],Dt[idx],Vt[idx]);acc=_mm512_xor_si512(acc,_mm512_xor_si512(R.q1,_mm512_xor_si512(R.r.lo,R.r.hi)));}uint64_t t1=now();double ns=(double)(t1-t0)/((double)LOOPS*M);if(ns<bt)bt=ns;}
    sink^=_mm512_reduce_add_epi64(acc);printf("div3by2 throughput: %.1f cyc\n",bt*g); }
  // latency: feed q1 into next dividend's u2 (store) - chained
  { alignas(64) uint64_t U[24],D16[16];for(int i=0;i<24;i++)U[i]=Ut[0][i];for(int i=0;i<16;i++)D16[i]=Dt[0][i];_vec vv=Vt[0],MASK=_mm512_set1_epi64(M52);double bl=1e300;
    for(int rep=0;rep<9;rep++){uint64_t t0=now();for(int i=0;i<3000000;i++){_div32 R=div3by2(U,D16,vv);_mm512_store_si512((void*)(U+16),_mm512_and_si512(R.q1,MASK));}uint64_t t1=now();double ns=(double)(t1-t0)/3000000.0;if(ns<bl)bl=ns;}
    sink^=U[16];printf("div3by2 latency   : %.1f cyc  (q1->u2 chain, ~50%% mispred)\n",bl*g); }
  // latency, VALID precondition: feed remainder <r1,r0> back as <u2,u1> (< d always)
  { alignas(64) uint64_t U[24],D16[16];for(int i=0;i<24;i++)U[i]=Ut[0][i];for(int i=0;i<16;i++)D16[i]=Dt[0][i];_vec vv=Vt[0],MASK=_mm512_set1_epi64(M52);double bl=1e300;
    for(int rep=0;rep<9;rep++){uint64_t t0=now();for(int i=0;i<3000000;i++){_div32 R=div3by2(U,D16,vv);_mm512_store_si512((void*)(U+16),_mm512_and_si512(R.r.hi,MASK));_mm512_store_si512((void*)(U+8),_mm512_and_si512(R.r.lo,MASK));}uint64_t t1=now();double ns=(double)(t1-t0)/3000000.0;if(ns<bl)bl=ns;}
    sink^=U[16]^U[8];printf("div3by2 latency   : %.1f cyc  (r->u chain, valid precond)\n",bl*g); }
  printf("ghz %.3f sink=%lx\n",g,(unsigned long)sink);return 0;}
