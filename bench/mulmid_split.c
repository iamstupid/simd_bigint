/* mulmid_split.c -- user's lo/hi-SEPARATED middle product (no warmup; endpoint
 * masks). A short (an+1), B long (bn+1). Band R[k]=sum_{i+j=k} A[i]B[j], k in [an,bn].
 * 8 accs even=lo/odd=hi; B0/B1 (offset 1) pack two co-diagonal products per acc.
 * Harvest: sum_even (de-skew lo, imm 8,6,4,2), sum_odd (hi, imm 7,5,3,1) vs last;
 * O[m]=R[an+m]=sum_even[l]+sum_odd[l]; O[0]=Rl[an] (mask sum_odd lane0, last=0). */
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
typedef uint64_t u64; typedef unsigned __int128 u128;
#define MASK52 ((1ULL<<52)-1)
static double now_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e9+ts.tv_nsec; }
static double core_ghz(void){ unsigned long n=1500000000UL; double t0=now_ns();
    asm volatile("1:\n\t dec %0\n\t jnz 1b":"+r"(n)::"cc"); return 1.5e9/(now_ns()-t0); }

/* scalar spec: O[m] for m=0..bn-an, O[m]=Rl[an+m]+Rh[an+m-1]; O[0] drops Rh[an-1]. */
static void scalar(u64*O,const u64*A,long an,const u64*B,long bn){
  long M=bn-an+1;
  for(long m=0;m<M;m++){ long k=an+m; u64 lo=0,hi=0;
    for(long i=0;i<=an;i++){ long j=k-i;  if(j>=0&&j<=bn){u128 p=(u128)A[i]*B[j]; lo+=(u64)(p&MASK52);} }
    if(m>0) for(long i=0;i<=an;i++){ long j=(k-1)-i; if(j>=0&&j<=bn){u128 p=(u128)A[i]*B[j]; hi+=(u64)((p>>52)&MASK52);} }
    O[m]=lo+hi; }
}

/* Pure accumulation of the strip whose output-base is i -> acc[0..7] (even=lo,
 * odd=hi). NO de-skew/harvest. Used both for the seed strip and each output vec. */
static inline void accum(__m512i acc[8],const u64*A,long an,const u64*B,long i){
  __m512i Z=_mm512_setzero_si512(); for(int t=0;t<8;t++)acc[t]=Z;
  long k=i, j=an;                                /* anti-diagonal walk k+j=an+i */
  for(; j> -8; k+=8, j-=8){
    __m512i B0=_mm512_loadu_si512((const void*)(B+k));
    __m512i B1=_mm512_loadu_si512((const void*)(B+k+1));
#define BC(m) _mm512_set1_epi64((long long)A[j+(m)])     /* A0[m]=A[j+m] */
    __m512i a_m1=BC(-1),a0=BC(0),a1=BC(1),a2=BC(2),a3=BC(3),a4=BC(4),a5=BC(5),a6=BC(6);
#undef BC
    acc[0]=_mm512_madd52lo_epu64(acc[0],B0,a0); acc[0]=_mm512_madd52lo_epu64(acc[0],B1,a_m1);
    acc[1]=_mm512_madd52hi_epu64(acc[1],B0,a0); acc[1]=_mm512_madd52hi_epu64(acc[1],B1,a_m1);
    acc[2]=_mm512_madd52lo_epu64(acc[2],B0,a2); acc[2]=_mm512_madd52lo_epu64(acc[2],B1,a1);
    acc[3]=_mm512_madd52hi_epu64(acc[3],B0,a2); acc[3]=_mm512_madd52hi_epu64(acc[3],B1,a1);
    acc[4]=_mm512_madd52lo_epu64(acc[4],B0,a4); acc[4]=_mm512_madd52lo_epu64(acc[4],B1,a3);
    acc[5]=_mm512_madd52hi_epu64(acc[5],B0,a4); acc[5]=_mm512_madd52hi_epu64(acc[5],B1,a3);
    acc[6]=_mm512_madd52lo_epu64(acc[6],B0,a6); acc[6]=_mm512_madd52lo_epu64(acc[6],B1,a5);
    acc[7]=_mm512_madd52hi_epu64(acc[7],B0,a6); acc[7]=_mm512_madd52hi_epu64(acc[7],B1,a5);
  }
}

/* A padded [-8,an+8]=0, B padded both sides. O[0..bn-an]. */
__attribute__((noinline))
static void mm_split(u64*O,const u64*A,long an,const u64*B,long bn){
  long NV=(bn-an+1+7)/8;
  __m512i last[8]; accum(last,A,an,B,-8);          /* seed strip: accumulation only */
  for(long iv=0;iv<NV;iv++){
    long i=8*iv;
    __m512i acc[8]; accum(acc,A,an,B,i);
    /* harvest: de-skew lo (even, imm 8/6/4/2) and hi (odd, imm 7/5/3/1) vs last */
    __m512i se=acc[0];
    se=_mm512_add_epi64(se,_mm512_alignr_epi64(acc[2],last[2],6));
    se=_mm512_add_epi64(se,_mm512_alignr_epi64(acc[4],last[4],4));
    se=_mm512_add_epi64(se,_mm512_alignr_epi64(acc[6],last[6],2));
    __m512i so=_mm512_alignr_epi64(acc[1],last[1],7);
    so=_mm512_add_epi64(so,_mm512_alignr_epi64(acc[3],last[3],5));
    so=_mm512_add_epi64(so,_mm512_alignr_epi64(acc[5],last[5],3));
    so=_mm512_add_epi64(so,_mm512_alignr_epi64(acc[7],last[7],1));
    if(iv==0) so=_mm512_maskz_mov_epi64(0xFE,so);     /* drop Rh[an-1] at O[0] */
    _mm512_storeu_si512((void*)(O+i),_mm512_add_epi64(se,so));
    for(int t=0;t<8;t++)last[t]=acc[t];
  }
}

/* ---- 1-LOAD merged-accumulator port of the SAME function onto diag's loop ----
 * Long operand B is aligned-loaded ONCE per group (1 load / 16 madds); the SHORT
 * operand A drives the loop -> exactly ceil((an+1)/8) groups, NO wasted step.
 * Skew via 9 merged accs (diag scheme): lo(B,A[r])->E[r+1], hi(B,A[r])->E[r].
 * raw[p]=Rl[p]+Rh[p-1] at p=(aL-8)+k; so O[m]=raw[m+7]; O[0] drops Rh[an-1]. */
static inline void diag_block_E(__m512i E[9],const u64*Bg,const u64*A,long aL){
  long Q=aL>>3, rem=aL&7; const u64*Aj=A+(aL-1);
  for(int m=0;m<9;m++)E[m]=_mm512_setzero_si512();
  for(long g=0;g<Q;g++,Bg+=8,Aj-=8){
    __m512i Bv=_mm512_load_si512((const void*)Bg);
#define STEP(r){ __m512i aa=_mm512_set1_epi64((long long)Aj[-(r)]); \
   E[(r)+1]=_mm512_madd52lo_epu64(E[(r)+1],Bv,aa); E[r]=_mm512_madd52hi_epu64(E[r],Bv,aa); }
    STEP(0)STEP(1)STEP(2)STEP(3)STEP(4)STEP(5)STEP(6)STEP(7)
#undef STEP
  }
  if(rem){ __m512i Bv=_mm512_load_si512((const void*)Bg);
    for(int r=0;r<rem;r++){ __m512i aa=_mm512_set1_epi64((long long)Aj[-r]);
      E[r+1]=_mm512_madd52lo_epu64(E[r+1],Bv,aa); E[r]=_mm512_madd52hi_epu64(E[r],Bv,aa);} }
}
__attribute__((noinline))
static void diag_core(u64*raw,const u64*B,const u64*A,long aL,long nb){
  __m512i last[9]; diag_block_E(last,B-8,A,aL);          /* seed block -1 */
  for(long k=0;k<nb;k++){
    __m512i E[9]; diag_block_E(E,B+8*k,A,aL);
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
/* O[m]=R[an+m]; B long / A short. raw 64-aligned, >= nb*8 limbs (nb=ceil((M+7)/8)). */
static void mm_split1(u64*O,u64*raw,const u64*A,long an,const u64*B,long bn){
  long M=bn-an+1, aL=an+1, nb=(M+14)/8;
  diag_core(raw,B,A,aL,nb);
  for(long m=0;m<M;m++)O[m]=raw[m+7];
  u64 hl=0; for(long i=0;i<an;i++){ long j=an-1-i; if(j>=0&&j<=bn){u128 p=(u128)A[i]*B[j]; hl+=(u64)((p>>52)&MASK52);} }
  O[0]-=hl;                                              /* drop the below-band hi-leak Rh[an-1] */
}

/* ---- mm_split2: separated lo/hi + DOWNWARD window + delayed (1-loop) harvest ----
 * Window A[j-7..j] (extends DOWN) tiles A[0..an] in exactly S=ceil((an+1)/8) steps
 * (no wasted step). acc[2r]=lo, acc[2r+1]=hi of diagonal offset -2r (r=0..3); each
 * acc lane l = partial sum over A-residues {an-2r, an-2r-1}. B0/B1 pack the 2
 * co-diagonals. Keeps separated lo/hi -> both boundary masks available in-lane. */
static inline void accum2(__m512i acc[8],const u64*A,long an,const u64*B,long iv){
  __m512i Z=_mm512_setzero_si512(); for(int t=0;t<8;t++)acc[t]=Z;
  long S=(an+8)/8, j=an, k=8*iv;
  for(long s=0;s<S;s++, j-=8, k+=8){
    __m512i B0=_mm512_loadu_si512((const void*)(B+k));
    __m512i B1=_mm512_loadu_si512((const void*)(B+k+1));
#define ST(r){ __m512i am=_mm512_set1_epi64((long long)A[j-2*(r)]), am1=_mm512_set1_epi64((long long)A[j-2*(r)-1]); \
   acc[2*(r)]  =_mm512_madd52lo_epu64(acc[2*(r)],  B0,am); acc[2*(r)]  =_mm512_madd52lo_epu64(acc[2*(r)],  B1,am1); \
   acc[2*(r)+1]=_mm512_madd52hi_epu64(acc[2*(r)+1],B0,am); acc[2*(r)+1]=_mm512_madd52hi_epu64(acc[2*(r)+1],B1,am1); }
    ST(0)ST(1)ST(2)ST(3)
#undef ST
  }
}
/* de-skew: full_lo[P+p]=Σ_r lo_acc[r] lane(p+2r)  (reaches UP into next block);
 *          Rh[P+p-1]  =Σ_r hi_acc[r] lane(p-1+2r) (acc[0] reaches DOWN into prev). */
__attribute__((noinline))
static void mm_split2(u64*O,const u64*A,long an,const u64*B,long bn){
  long M=bn-an+1, NVout=(M+7)/8;
  __m512i cur[8],nxt[8],prev_hi0=_mm512_setzero_si512();
  accum2(cur,A,an,B,0);
  for(long iv=0;iv<NVout;iv++){
    accum2(nxt,A,an,B,iv+1);
    __m512i se=cur[0];
    se=_mm512_add_epi64(se,_mm512_alignr_epi64(nxt[2],cur[2],2));
    se=_mm512_add_epi64(se,_mm512_alignr_epi64(nxt[4],cur[4],4));
    se=_mm512_add_epi64(se,_mm512_alignr_epi64(nxt[6],cur[6],6));
    __m512i so=_mm512_alignr_epi64(cur[1],prev_hi0,7);
    so=_mm512_add_epi64(so,_mm512_alignr_epi64(nxt[3],cur[3],1));
    so=_mm512_add_epi64(so,_mm512_alignr_epi64(nxt[5],cur[5],3));
    so=_mm512_add_epi64(so,_mm512_alignr_epi64(nxt[7],cur[7],5));
    if(iv==0) so=_mm512_maskz_mov_epi64(0xFE,so);        /* drop Rh[an-1] at O[0] */
    _mm512_storeu_si512((void*)(O+8*iv),_mm512_add_epi64(se,so));
    prev_hi0=cur[1];
    for(int t=0;t<8;t++)cur[t]=nxt[t];
  }
}

int main(int argc,char**argv){
  long an=argc>1?atol(argv[1]):15, bn=argc>2?atol(argv[2]):79;
  const long PADA=16, PADBlo=16, PADBhi=64;
  u64*Abuf=aligned_alloc(64,(PADA+an+1+PADA)*8); memset(Abuf,0,(PADA+an+1+PADA)*8); u64*A=Abuf+PADA;
  u64*Bbuf=aligned_alloc(64,(PADBlo+bn+1+PADBhi)*8); memset(Bbuf,0,(PADBlo+bn+1+PADBhi)*8); u64*B=Bbuf+PADBlo;
  long M=bn-an+1, nb=(M+14)/8;
  u64*Os=malloc((M+8)*8),*Ok=malloc((M+8)*8),*Ok1=malloc((M+8)*8),*Ok2=malloc((M+8)*8);
  u64*rd=aligned_alloc(64,(size_t)nb*64);
  srand48(1);
  for(long i=0;i<=an;i++)A[i]=((u64)mrand48()<<20^(u64)mrand48())&MASK52;
  for(long j=0;j<=bn;j++)B[j]=((u64)mrand48()<<20^(u64)mrand48())&MASK52;
  scalar(Os,A,an,B,bn);
  memset(Ok,0,(M+8)*8); mm_split(Ok,A,an,B,bn);
  memset(Ok1,0,(M+8)*8); mm_split1(Ok1,rd,A,an,B,bn);
  memset(Ok2,0,(M+8)*8); mm_split2(Ok2,A,an,B,bn);
  long fails=0,fails1=0,fails2=0;
  for(long m=0;m<M;m++) if(Os[m]!=Ok[m]){ if(fails<8) printf("  split  mismatch m=%ld pos=%ld got=%llx exp=%llx\n",m,an+m,(unsigned long long)Ok[m],(unsigned long long)Os[m]); fails++; }
  for(long m=0;m<M;m++) if(Os[m]!=Ok1[m]){ if(fails1<8) printf("  split1 mismatch m=%ld pos=%ld got=%llx exp=%llx\n",m,an+m,(unsigned long long)Ok1[m],(unsigned long long)Os[m]); fails1++; }
  for(long m=0;m<M;m++) if(Os[m]!=Ok2[m]){ if(fails2<8) printf("  split2 mismatch m=%ld pos=%ld got=%llx exp=%llx\n",m,an+m,(unsigned long long)Ok2[m],(unsigned long long)Os[m]); fails2++; }
  printf("an=%ld bn=%ld band=%ld: split %s(%ld) | split1 %s(%ld) | split2 %s(%ld)\n",an,bn,M,fails?"FAIL":"OK",fails,fails1?"FAIL":"OK",fails1,fails2?"FAIL":"OK",fails2);
  if(fails||fails1||fails2) return 1;
  /* --- benchmark --- */
  long REP=argc>3?atol(argv[3]):200000;
  double G=core_ghz(), aL=an+1;
  long aLi=an+1;
  for(int w=0;w<3;w++){ mm_split(Ok,A,an,B,bn); diag_core(rd,B,A,aLi,nb); mm_split2(Ok2,A,an,B,bn); }
  u64 sink=0;
  double digits=(double)REP*M;
  double t0=now_ns();
  for(long r=0;r<REP;r++){ mm_split(Ok,A,an,B,bn); sink^=Ok[0]^Ok[M-1]; }
  double t1=now_ns();
  double d0=now_ns();
  for(long r=0;r<REP;r++){ diag_core(rd,B,A,aLi,nb); sink^=rd[7]; }
  double d1=now_ns();
  double e0=now_ns();
  for(long r=0;r<REP;r++){ mm_split2(Ok2,A,an,B,bn); sink^=Ok2[0]^Ok2[M-1]; }
  double e1=now_ns();
  printf("  short=an+1=%.0f  ideal~(an+1)/8=%.2f  coreGHz=%.2f\n", aL, aL/8.0, G);
  printf("  split  (2-load up,   harvest-now)  : %.3f core-cyc/digit\n",(t1-t0)/digits*G);
  printf("  split1 (1-load, diag merged accs)  : %.3f core-cyc/digit\n",(d1-d0)/digits*G);
  printf("  split2 (2-load down, harvest-later): %.3f core-cyc/digit  (sink=%llx)\n",(e1-e0)/digits*G,(unsigned long long)sink);
  return 0;
}
