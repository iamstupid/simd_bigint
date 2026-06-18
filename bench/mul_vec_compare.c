// mul_u52_vec (fused canon) vs mul_u52_1 + mpn_canon_pos.  Same product
// A[n] * B[8 lanes]; correctness by direct compare (no GMP), then head-to-head
// throughput.  build: gcc -O3 -march=native -I.. mul_vec_compare.c -o /tmp/mv
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

// canonical-output block count for A[n]*B[8]
static int nblk(int n){ return (n+8+7)/8; }

static int test_n(int n, uint64_t *s){
    int av = (n+7)/8;                       // A vecs (padded)
    alignas(64) uint64_t A[ 8*64 ] = {0};   // padded scalar limbs
    alignas(64) uint64_t Bm[8];
    for(int i=0;i<n;i++)   A[i]  = rng(s)&M52;
    for(int i=0;i<8;i++)   Bm[i] = rng(s)&M52;
    alignas(64) _vec rv[64], rr[64];
    memset(rv,0,sizeof rv); memset(rr,0,sizeof rr);
    // fused kernel
    mul_u52_vec(rv, A, load_vec((cpvec)Bm), n);
    // mul_u52_1 (keep an>=bn) + canon
    if(n>=8) mul_u52_1(rr, (cpvec)A, (cpvec)Bm, n, 8);
    else     mul_u52_1(rr, (cpvec)Bm, (cpvec)A, 8, n);
    mpn_canon_pos((pvec)rr, (cpvec)rr, n+8);
    int b = nblk(n);
    return memcmp(rv, rr, (size_t)b*sizeof(_vec));
}

int main(int c,char**v){
    cpu_set_t st;CPU_ZERO(&st);CPU_SET(0,&st);sched_setaffinity(0,sizeof st,&st);
    uint64_t s=c>1?strtoull(v[1],0,0):0xABCDEF;
    // ---- correctness ----
    int bad=0,first=0;
    for(int n=1;n<=200;n++){ for(int rep=0;rep<2000;rep++){ if(test_n(n,&s)){ if(!bad)first=n; bad++; } } }
    printf("correctness: %s  (mismatches=%d%s)\n", bad?"FAIL":"OK", bad, bad?"":" over n=1..200 x2000");
    if(bad){ printf("  first failing n=%d\n",first); }
    // ---- benchmark ----
    double g=ghz();
    int sizes[]={16,32,64,128,256};
    printf("\n n   mul_u52_vec   mul_u52_1     mul_u52_1+canon\n");
    for(int si=0;si<5;si++){
        int n=sizes[si], av=(n+7)/8;
        alignas(64) uint64_t A[8*64]={0}, Bm[8];
        for(int i=0;i<n;i++)A[i]=rng(&s)&M52; for(int i=0;i<8;i++)Bm[i]=rng(&s)&M52;
        _vec Bv=load_vec((cpvec)Bm);
        alignas(64) _vec rv[64],rr[64]; memset(rv,0,sizeof rv);memset(rr,0,sizeof rr);
        volatile uint64_t sink=0;
        const int IT=200000;
        double bv=1e300,b1=1e300,bc=1e300;
        for(int r=0;r<9;r++){uint64_t t0=now();for(int i=0;i<IT;i++){mul_u52_vec(rv,A,Bv,n);sink^=((uint64_t*)rv)[0];}uint64_t t1=now();double ns=(double)(t1-t0)/IT;if(ns<bv)bv=ns;}
        for(int r=0;r<9;r++){uint64_t t0=now();for(int i=0;i<IT;i++){mul_u52_1(rr,(cpvec)A,(cpvec)Bm,n,8);sink^=((uint64_t*)rr)[0];}uint64_t t1=now();double ns=(double)(t1-t0)/IT;if(ns<b1)b1=ns;}
        for(int r=0;r<9;r++){uint64_t t0=now();for(int i=0;i<IT;i++){mul_u52_1(rr,(cpvec)A,(cpvec)Bm,n,8);mpn_canon_pos((pvec)rr,(cpvec)rr,n+8);sink^=((uint64_t*)rr)[0];}uint64_t t1=now();double ns=(double)(t1-t0)/IT;if(ns<bc)bc=ns;}
        printf("%4d  %5.1f cyc    %5.1f cyc    %5.1f cyc   (sink=%lx)\n",n,bv*g,b1*g,bc*g,(unsigned long)sink);
    }
    printf("ghz %.3f\n",g);
    return bad?1:0;
}
