/* zen5_ports.c -- raw throughput of the IFMA-mulmid ops (addresses varied to
 * defeat hoisting; 12 madd chains so madd is throughput- not latency-bound). */
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
typedef uint64_t u64;
static double now_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e9+ts.tv_nsec; }
static double core_ghz(void){ unsigned long n=1500000000UL; double t0=now_ns();
    asm volatile("1:\n\t dec %0\n\t jnz 1b":"+r"(n)::"cc"); return 1.5e9/(now_ns()-t0); }
static u64 SINK=0;
#define BUFB (1<<16)
#define MASK 255   /* base = (k&255)*64 -> 16KB span, stays in L1 */

int main(void){
    double G=core_ghz();
    char *buf=aligned_alloc(64, BUFB+1024); memset(buf,1,BUFB+1024);
    __m512i c1=_mm512_set1_epi64(0x9e3779b97f4a7c15ULL), c2=_mm512_set1_epi64(3);
    const long K=30000000;

    /* madd throughput, 12 chains */
    { __m512i a[12]; for(int u=0;u<12;u++)a[u]=c1;
      double t=now_ns();
      for(long k=0;k<K;k++) for(int u=0;u<12;u++) a[u]=_mm512_madd52lo_epu64(a[u],c1,c2);
      double ns=now_ns()-t; for(int u=0;u<12;u++)SINK^=_mm512_reduce_add_epi64(a[u]);
      printf("madd x12chain : %.2f op/cyc\n",(K*12.0)/(ns*G)); }

    /* aligned load throughput, 8/iter, varying base */
    { __m512i s[8]; for(int u=0;u<8;u++)s[u]=_mm512_setzero_si512();
      double t=now_ns();
      for(long k=0;k<K;k++){ long o=(k&MASK)*64;
        for(int u=0;u<8;u++) s[u]=_mm512_xor_si512(s[u],_mm512_load_si512((void*)(buf+o+u*64))); }
      double ns=now_ns()-t; for(int u=0;u<8;u++)SINK^=_mm512_reduce_add_epi64(s[u]);
      printf("load aligned  : %.2f op/cyc\n",(K*8.0)/(ns*G)); }

    /* SPLIT load (+8 -> crosses 64B line), 8/iter, varying base */
    { __m512i s[8]; for(int u=0;u<8;u++)s[u]=_mm512_setzero_si512();
      double t=now_ns();
      for(long k=0;k<K;k++){ long o=(k&MASK)*64+8;
        for(int u=0;u<8;u++) s[u]=_mm512_xor_si512(s[u],_mm512_loadu_si512((void*)(buf+o+u*64))); }
      double ns=now_ns()-t; for(int u=0;u<8;u++)SINK^=_mm512_reduce_add_epi64(s[u]);
      printf("load SPLIT    : %.2f op/cyc\n",(K*8.0)/(ns*G)); }

    /* valignq throughput, 8 chains */
    { __m512i r[8]; for(int u=0;u<8;u++)r[u]=c1;
      double t=now_ns();
      for(long k=0;k<K;k++) for(int u=0;u<8;u++) r[u]=_mm512_alignr_epi64(r[u],c2,1);
      double ns=now_ns()-t; for(int u=0;u<8;u++)SINK^=_mm512_reduce_add_epi64(r[u]);
      printf("valignq x8    : %.2f op/cyc\n",(K*8.0)/(ns*G)); }

    /* madd(8 chains) + valignq(4/iter): does valign steal madd? */
    { __m512i a[8],r[4]; for(int u=0;u<8;u++)a[u]=c1; for(int u=0;u<4;u++)r[u]=c2;
      double t=now_ns();
      for(long k=0;k<K;k++){ for(int u=0;u<8;u++)a[u]=_mm512_madd52lo_epu64(a[u],c1,c2);
        for(int u=0;u<4;u++)r[u]=_mm512_alignr_epi64(r[u],c1,1); }
      double ns=now_ns()-t; for(int u=0;u<8;u++)SINK^=_mm512_reduce_add_epi64(a[u]); for(int u=0;u<4;u++)SINK^=_mm512_reduce_add_epi64(r[u]);
      printf("madd8+valignq4: %.2f madd/cyc  %.2f align/cyc\n",(K*8.0)/(ns*G),(K*4.0)/(ns*G)); }

    /* madd(8 chains) + SPLIT load(4/iter): does split-load steal madd? */
    { __m512i a[8],s[4]; for(int u=0;u<8;u++)a[u]=c1; for(int u=0;u<4;u++)s[u]=_mm512_setzero_si512();
      double t=now_ns();
      for(long k=0;k<K;k++){ long o=(k&MASK)*64+8;
        for(int u=0;u<8;u++)a[u]=_mm512_madd52lo_epu64(a[u],c1,c2);
        for(int u=0;u<4;u++)s[u]=_mm512_xor_si512(s[u],_mm512_loadu_si512((void*)(buf+o+u*64))); }
      double ns=now_ns()-t; for(int u=0;u<8;u++)SINK^=_mm512_reduce_add_epi64(a[u]); for(int u=0;u<4;u++)SINK^=_mm512_reduce_add_epi64(s[u]);
      printf("madd8+SPLITld4: %.2f madd/cyc  %.2f splitld/cyc\n",(K*8.0)/(ns*G),(K*4.0)/(ns*G)); }

    /* madd(8) + aligned load(4): baseline for comparison */
    { __m512i a[8],s[4]; for(int u=0;u<8;u++)a[u]=c1; for(int u=0;u<4;u++)s[u]=_mm512_setzero_si512();
      double t=now_ns();
      for(long k=0;k<K;k++){ long o=(k&MASK)*64;
        for(int u=0;u<8;u++)a[u]=_mm512_madd52lo_epu64(a[u],c1,c2);
        for(int u=0;u<4;u++)s[u]=_mm512_xor_si512(s[u],_mm512_load_si512((void*)(buf+o+u*64))); }
      double ns=now_ns()-t; for(int u=0;u<8;u++)SINK^=_mm512_reduce_add_epi64(a[u]); for(int u=0;u<4;u++)SINK^=_mm512_reduce_add_epi64(s[u]);
      printf("madd8+alignld4: %.2f madd/cyc  %.2f alignld/cyc\n",(K*8.0)/(ns*G),(K*4.0)/(ns*G)); }

    printf("coreGHz=%.2f sink=%llx\n", G,(unsigned long long)SINK);
    return 0;
}
