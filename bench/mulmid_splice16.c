/* mulmid_splice16.c -- bn=16 middle product: UNALIGNED sliding loads vs
 * ALIGNED loads + valignq splice. Settles whether avoiding split loads wins
 * on Zen5 (split=1/cyc, aligned=2/cyc measured). Correctness vs scalar spec. */
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
typedef uint64_t u64;
typedef unsigned __int128 u128;
#define MASK52 ((1ULL<<52)-1)
#define BN 16
static double now_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e9+ts.tv_nsec; }
static double core_ghz(void){ unsigned long n=1500000000UL; double t0=now_ns();
    asm volatile("1:\n\t dec %0\n\t jnz 1b":"+r"(n)::"cc"); return 1.5e9/(now_ns()-t0); }

static void scalar(u64*raw,const u64*a,long an,const u64*b,long band){
  for(long k=0;k<band;k++){ long p=(BN-1)+k; u64 a0=0,a1=0;
    for(long t=0;t<BN;t++){ long sl=p-t,sh=p-1-t;
      if(sl>=0&&sl<an){u128 pr=(u128)a[sl]*b[t]; a0+=(u64)(pr&MASK52);}
      if(sh>=0&&sh<an){u128 pr=(u128)a[sh]*b[t]; a1+=(u64)((pr>>52)&MASK52);} }
    raw[k]=a0+a1; } }

/* UNALIGNED: sliding loadu (mostly split) */
__attribute__((noinline))
static void mm_unaligned(u64*raw,const u64*a,const u64*b,long NB){
  __m512i B[BN]; for(int t=0;t<BN;t++)B[t]=_mm512_set1_epi64((long long)b[t]);
  for(long blk=0;blk<NB;blk++){ long i=(BN-1)+blk*8;
    __m512i acc0=_mm512_setzero_si512(),acc1=_mm512_setzero_si512(),w;
    w=_mm512_loadu_si512((void*)(a+i-BN)); acc1=_mm512_madd52hi_epu64(acc1,B[BN-1],w);
    for(long j=BN-1;j>=1;j--){ w=_mm512_loadu_si512((void*)(a+i-j));
      acc0=_mm512_madd52lo_epu64(acc0,B[j],w); acc1=_mm512_madd52hi_epu64(acc1,B[j-1],w); }
    w=_mm512_loadu_si512((void*)(a+i)); acc0=_mm512_madd52lo_epu64(acc0,B[0],w);
    _mm512_storeu_si512((void*)(raw+blk*8),_mm512_add_epi64(acc0,acc1)); } }

/* SPLICE: 4 aligned loads A0..A3 cover a[base..base+31], window(j)=valignq.
 * base=blk*8-8 (64-aligned); off=23-j -> (A[off/8+1],A[off/8],off%8). */
__attribute__((noinline))
static void mm_splice(u64*raw,const u64*a,const u64*b,long NB){
  __m512i B[BN]; for(int t=0;t<BN;t++)B[t]=_mm512_set1_epi64((long long)b[t]);
  for(long blk=0;blk<NB;blk++){ long base=blk*8-8;
    __m512i A0=_mm512_load_si512((void*)(a+base)),     A1=_mm512_load_si512((void*)(a+base+8));
    __m512i A2=_mm512_load_si512((void*)(a+base+16)),  A3=_mm512_load_si512((void*)(a+base+24));
    __m512i acc0=_mm512_setzero_si512(),acc1=_mm512_setzero_si512();
    /* j=16 off7 (A1,A0,7): hi b15 */
    __m512i w16=_mm512_alignr_epi64(A1,A0,7); acc1=_mm512_madd52hi_epu64(acc1,B[15],w16);
    /* j=15 off8 = A1 */                       acc0=_mm512_madd52lo_epu64(acc0,B[15],A1); acc1=_mm512_madd52hi_epu64(acc1,B[14],A1);
    __m512i w;
    w=_mm512_alignr_epi64(A2,A1,1); acc0=_mm512_madd52lo_epu64(acc0,B[14],w); acc1=_mm512_madd52hi_epu64(acc1,B[13],w); /*j14*/
    w=_mm512_alignr_epi64(A2,A1,2); acc0=_mm512_madd52lo_epu64(acc0,B[13],w); acc1=_mm512_madd52hi_epu64(acc1,B[12],w); /*j13*/
    w=_mm512_alignr_epi64(A2,A1,3); acc0=_mm512_madd52lo_epu64(acc0,B[12],w); acc1=_mm512_madd52hi_epu64(acc1,B[11],w); /*j12*/
    w=_mm512_alignr_epi64(A2,A1,4); acc0=_mm512_madd52lo_epu64(acc0,B[11],w); acc1=_mm512_madd52hi_epu64(acc1,B[10],w); /*j11*/
    w=_mm512_alignr_epi64(A2,A1,5); acc0=_mm512_madd52lo_epu64(acc0,B[10],w); acc1=_mm512_madd52hi_epu64(acc1,B[9], w); /*j10*/
    w=_mm512_alignr_epi64(A2,A1,6); acc0=_mm512_madd52lo_epu64(acc0,B[9], w); acc1=_mm512_madd52hi_epu64(acc1,B[8], w); /*j9 */
    w=_mm512_alignr_epi64(A2,A1,7); acc0=_mm512_madd52lo_epu64(acc0,B[8], w); acc1=_mm512_madd52hi_epu64(acc1,B[7], w); /*j8 */
    /* j=7 off16 = A2 */            acc0=_mm512_madd52lo_epu64(acc0,B[7], A2);acc1=_mm512_madd52hi_epu64(acc1,B[6], A2);/*j7 */
    w=_mm512_alignr_epi64(A3,A2,1); acc0=_mm512_madd52lo_epu64(acc0,B[6], w); acc1=_mm512_madd52hi_epu64(acc1,B[5], w); /*j6 */
    w=_mm512_alignr_epi64(A3,A2,2); acc0=_mm512_madd52lo_epu64(acc0,B[5], w); acc1=_mm512_madd52hi_epu64(acc1,B[4], w); /*j5 */
    w=_mm512_alignr_epi64(A3,A2,3); acc0=_mm512_madd52lo_epu64(acc0,B[4], w); acc1=_mm512_madd52hi_epu64(acc1,B[3], w); /*j4 */
    w=_mm512_alignr_epi64(A3,A2,4); acc0=_mm512_madd52lo_epu64(acc0,B[3], w); acc1=_mm512_madd52hi_epu64(acc1,B[2], w); /*j3 */
    w=_mm512_alignr_epi64(A3,A2,5); acc0=_mm512_madd52lo_epu64(acc0,B[2], w); acc1=_mm512_madd52hi_epu64(acc1,B[1], w); /*j2 */
    w=_mm512_alignr_epi64(A3,A2,6); acc0=_mm512_madd52lo_epu64(acc0,B[1], w); acc1=_mm512_madd52hi_epu64(acc1,B[0], w); /*j1 */
    w=_mm512_alignr_epi64(A3,A2,7); acc0=_mm512_madd52lo_epu64(acc0,B[0], w); /*j0 epilogue lo b0*/
    _mm512_storeu_si512((void*)(raw+blk*8),_mm512_add_epi64(acc0,acc1)); } }

int main(int argc,char**argv){
  long NB=argc>1?atol(argv[1]):32, REP=argc>2?atol(argv[2]):400000;
  long band=NB*8, an=BN+band; const long PAD=8;
  u64*abuf=aligned_alloc(64,(PAD+an+8)*8); memset(abuf,0,(PAD+an+8)*8);
  u64*a=abuf+PAD,*b=aligned_alloc(64,BN*8),*rs=malloc(band*8),*ru=malloc(band*8),*rp=malloc(band*8);
  double G=core_ghz();
  long fails=0;
  for(int it=0;it<100;it++){
    for(long s=0;s<an;s++)a[s]=((u64)mrand48()<<20^(u64)mrand48())&MASK52;
    for(long t=0;t<BN;t++)b[t]=((u64)mrand48()<<20^(u64)mrand48())&MASK52;
    scalar(rs,a,an,b,band); mm_unaligned(ru,a,b,NB); mm_splice(rp,a,b,NB);
    if(memcmp(rs,ru,band*8))fails+=!!printf("  unaligned!=scalar it=%d\n",it);
    if(memcmp(rs,rp,band*8))fails+=!!printf("  splice!=scalar it=%d\n",it);
  }
  printf("correctness bn=%d band=%ld: %s\n",BN,band,fails?"FAIL":"OK");
  for(long s=0;s<an;s++)a[s]=((u64)mrand48()<<20^(u64)mrand48())&MASK52;
  for(long t=0;t<BN;t++)b[t]=((u64)mrand48()<<20^(u64)mrand48())&MASK52;
  for(int w=0;w<3;w++){mm_unaligned(ru,a,b,NB);mm_splice(rp,a,b,NB);}
  u64 sink=0; double digits=(double)REP*band;
  double u0=now_ns(); for(long r=0;r<REP;r++){mm_unaligned(ru,a,b,NB);sink^=ru[0];} double u1=now_ns();
  double p0=now_ns(); for(long r=0;r<REP;r++){mm_splice(rp,a,b,NB);sink^=rp[0];} double p1=now_ns();
  double up=(u1-u0)/digits, pp=(p1-p0)/digits;
  printf("coreGHz=%.2f ideal(bn/8)=%.2f core-cyc/digit\n",G,BN/8.0);
  printf("  UNALIGNED(split): %.3f ns  %.3f core-cyc/digit\n",up,up*G);
  printf("  SPLICE(valignq) : %.3f ns  %.3f core-cyc/digit  (sink=%llx)\n",pp,pp*G,(unsigned long long)sink);
  free(abuf);free(b);free(rs);free(ru);free(rp); return 0;
}
