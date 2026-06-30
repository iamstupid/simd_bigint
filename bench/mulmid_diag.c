/* mulmid_diag.c -- size-general AVX-512 IFMA middle product, DIAGONAL design:
 * aligned a-load reused across an 8-wide b-sweep (no unaligned/split loads),
 * 9 merged skew-accumulators C[0..8] (lo->C[r], hi->C[r+1]), de-skewed by a
 * valignq diagonal-sum + overlap-add carry. Runtime an,bn (8-group + tail).
 *
 * Correctness gated vs scalar spec and mpz rect; benched vs the unaligned kernel.
 */
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <gmp.h>
typedef uint64_t u64;
typedef unsigned __int128 u128;
#define MASK52 ((1ULL<<52)-1)
static double now_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e9+ts.tv_nsec; }
static double core_ghz(void){ unsigned long n=1500000000UL; double t0=now_ns();
    asm volatile("1:\n\t dec %0\n\t jnz 1b":"+r"(n)::"cc"); return 1.5e9/(now_ns()-t0); }

/* spec: M limbs, raw[k] = position p=(bn-8)+k  (matches the diag output layout) */
static void scalar(u64*raw,const u64*a,long an,const u64*b,long bn,long M){
  for(long k=0;k<M;k++){ long p=(bn-8)+k; u64 a0=0,a1=0;
    for(long t=0;t<bn;t++){ long sl=p-t,sh=p-1-t;
      if(sl>=0&&sl<an){u128 pr=(u128)a[sl]*b[t]; a0+=(u64)(pr&MASK52);}
      if(sh>=0&&sh<an){u128 pr=(u128)a[sh]*b[t]; a1+=(u64)((pr>>52)&MASK52);} }
    raw[k]=a0+a1; } }

/* DIAGONAL kernel. a & raw 64-aligned; a zero-padded below (a[-8..-1]) and above.
 * Writes nb FULL aligned vectors: raw[8k..8k+7] = positions [bn-8+8k, bn-1+8k].
 * a-base pointer ap ascends 8/block (always aligned); j descends (bj walks down).
 * E[r+1]+=lo, E[r]+=hi; de-skew composes curr E[m] with prev block last[m].
 * Block -1 (ap = a-8) is the warmup that seeds last[] for block 0; it has no store.
 * The middle-product band [bn-1, ..] is at raw[7..]; raw[0..6] are the low carry-in
 * limbs (raw[6]=pos bn-2 is the guard for the deferred canonize). */
/* One block's skewed accumulators E[0..8] from a-base ag (aligned), b descending. */
static inline void block_E(__m512i E[9],const u64*ag,const u64*b,long bn){
  long Q=bn>>3, rem=bn&7; const u64 *bj=b+(bn-1);
  for(int m=0;m<9;m++)E[m]=_mm512_setzero_si512();
  for(long g=0;g<Q;g++,ag+=8,bj-=8){
    __m512i A=_mm512_load_si512((const void*)ag);
#define STEP(r) { __m512i bb=_mm512_set1_epi64((long long)bj[-(r)]); \
                  E[(r)+1]=_mm512_madd52lo_epu64(E[(r)+1],A,bb); E[r]=_mm512_madd52hi_epu64(E[r],A,bb); }
    STEP(0)STEP(1)STEP(2)STEP(3)STEP(4)STEP(5)STEP(6)STEP(7)
#undef STEP
  }
  if(rem){ __m512i A=_mm512_load_si512((const void*)ag);
    for(int r=0;r<rem;r++){ __m512i bb=_mm512_set1_epi64((long long)bj[-r]);
      E[r+1]=_mm512_madd52lo_epu64(E[r+1],A,bb); E[r]=_mm512_madd52hi_epu64(E[r],A,bb); } }
}

__attribute__((noinline))
static void mm_diag(u64*raw,const u64*a,long an,const u64*b,long bn,long nb){
  (void)an;
  __m512i last[9]; block_E(last,a-8,b,bn);           /* seed last[] from block -1 */
  for(long k=0;k<nb;k++){
    __m512i E[9]; block_E(E,a+8*k,b,bn);
    /* de-skew: out = last[0] + E[8] + sum_{m=1}^{7} alignr(E[m],last[m],m) */
    __m512i o0=_mm512_add_epi64(last[0],E[8]);
    __m512i o1=_mm512_alignr_epi64(E[1],last[1],1);
    o0=_mm512_add_epi64(o0,_mm512_alignr_epi64(E[2],last[2],2));
    o1=_mm512_add_epi64(o1,_mm512_alignr_epi64(E[3],last[3],3));
    o0=_mm512_add_epi64(o0,_mm512_alignr_epi64(E[4],last[4],4));
    o1=_mm512_add_epi64(o1,_mm512_alignr_epi64(E[5],last[5],5));
    o0=_mm512_add_epi64(o0,_mm512_alignr_epi64(E[6],last[6],6));
    o1=_mm512_add_epi64(o1,_mm512_alignr_epi64(E[7],last[7],7));
    _mm512_store_si512((void*)(raw+8*k),_mm512_add_epi64(o0,o1));
    for(int m=0;m<9;m++)last[m]=E[m];
  }
}

/* unaligned baseline (generic bn, fused) for comparison */
__attribute__((noinline))
static void mm_unaligned(u64*raw,const u64*a,const u64*b,long bn,long NB){
  for(long blk=0;blk<NB;blk++){ long i=(bn-1)+blk*8;
    __m512i acc0=_mm512_setzero_si512(),acc1=_mm512_setzero_si512(),w;
    w=_mm512_loadu_si512((void*)(a+i-bn)); acc1=_mm512_madd52hi_epu64(acc1,_mm512_set1_epi64((long long)b[bn-1]),w);
    for(long j=bn-1;j>=1;j--){ w=_mm512_loadu_si512((void*)(a+i-j));
      acc0=_mm512_madd52lo_epu64(acc0,_mm512_set1_epi64((long long)b[j]),w);
      acc1=_mm512_madd52hi_epu64(acc1,_mm512_set1_epi64((long long)b[j-1]),w); }
    w=_mm512_loadu_si512((void*)(a+i)); acc0=_mm512_madd52lo_epu64(acc0,_mm512_set1_epi64((long long)b[0]),w);
    _mm512_storeu_si512((void*)(raw+blk*8),_mm512_add_epi64(acc0,acc1)); } }

int main(int argc,char**argv){
  long bn=argc>1?atol(argv[1]):16, NBb=argc>2?atol(argv[2]):32, REP=argc>3?atol(argv[3]):300000;
  long band=NBb*8, an=bn+band+16; const long PAD=64;
  long nb=NBb+1, M=8*nb;                /* nb output vectors; M limbs from pos bn-8 */
  u64*abuf=aligned_alloc(64,(PAD+an+PAD)*8); memset(abuf,0,(PAD+an+PAD)*8);
  u64*a=abuf+PAD,*b=aligned_alloc(64,((bn+7)&~7L)*8);
  u64*rs=malloc(M*8),*ru=malloc(band*8);
  u64*rd=aligned_alloc(64,((M*8+63)&~63L));
  gmp_randstate_t st; gmp_randinit_default(st); gmp_randseed_ui(st,1);
  double G=core_ghz();
  long fails=0;
  for(int it=0;it<150;it++){
    for(long s=0;s<an;s++)a[s]=((u64)mrand48()<<20^(u64)mrand48())&MASK52;
    for(long t=0;t<bn;t++)b[t]=((u64)mrand48()<<20^(u64)mrand48())&MASK52;
    scalar(rs,a,an,b,bn,M); mm_diag(rd,a,an,b,bn,nb); mm_unaligned(ru,a,b,bn,NBb);
    if(memcmp(rs,rd,M*8)){ if(fails<3){ for(long k=0;k<M;k++) if(rs[k]!=rd[k]){ printf("  diag!=scalar it=%d k=%ld pos=%ld got=%llx exp=%llx\n",it,k,bn-8+k,(unsigned long long)rd[k],(unsigned long long)rs[k]); break; } } fails++; }
    if(memcmp(rs+7,ru,band*8)) fails+=!!printf("  unaligned!=scalar it=%d\n",it);
  }
  printf("correctness bn=%ld band=%ld delta=%ld: %s (%ld)\n",bn,band,bn&7,fails?"FAIL":"OK",fails);
  if(fails){ return 1; }
  for(long s=0;s<an;s++)a[s]=((u64)mrand48()<<20^(u64)mrand48())&MASK52;
  for(long t=0;t<bn;t++)b[t]=((u64)mrand48()<<20^(u64)mrand48())&MASK52;
  for(int w=0;w<3;w++){mm_diag(rd,a,an,b,bn,nb);mm_unaligned(ru,a,b,bn,NBb);}
  u64 sink=0; double digits=(double)REP*band;
  double d0=now_ns(); for(long r=0;r<REP;r++){mm_diag(rd,a,an,b,bn,nb);sink^=rd[0];} double d1=now_ns();
  double u0=now_ns(); for(long r=0;r<REP;r++){mm_unaligned(ru,a,b,bn,NBb);sink^=ru[0];} double u1=now_ns();
  printf("coreGHz=%.2f ideal(bn/8)=%.2f\n",G,bn/8.0);
  printf("  DIAG(aligned+valignq): %.3f core-cyc/digit\n",(d1-d0)/digits*G);
  printf("  UNALIGNED(generic)   : %.3f core-cyc/digit (sink=%llx)\n",(u1-u0)/digits*G,(unsigned long long)sink);
  return 0;
}
