// Head-to-head: recip_mono (reciprocal_u416.s) vs GMP's reciprocal primitives
// at n=7 limbs (448-bit normalized divisor).
//
// At n=7 GMP is BELOW INV_NEWTON_THRESHOLD (=200), so:
//   __gmpn_invert     -> mpn_sbpi1_div_q      (exact schoolbook division)
//   __gmpn_invertappr -> mpn_sbpi1_divappr_q  (approx schoolbook division)
// i.e. GMP inverts small operands by SCHOOLBOOK DIVISION, not Newton.
//
// recip_mono computes X = floor(2^895/D); GMP computes ip with
//   B^n + ip = floor((B^{2n}-1)/D)  (n=7, B^n=2^448) -- a different scaling
//   and one extra bit, but the same primitive (n-limb approx reciprocal).
// Outputs are NOT bit-identical; this compares the cost of the primitive.
//
// Build:
//  gcc -O3 -march=native recip_vs_gmp.c ../reciprocal_u416.s -o /tmp/rvg -lgmp
#define _GNU_SOURCE
#include <gmp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N 7

extern void recip_mono(const uint64_t *D, uint64_t *xout);
// internal GMP symbols (exported by libgmp)
extern void      __gmpn_invert    (mp_limb_t *, const mp_limb_t *, mp_size_t, mp_limb_t *);
extern mp_limb_t __gmpn_invertappr(mp_limb_t *, const mp_limb_t *, mp_size_t, mp_limb_t *);

static uint64_t now_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW,&ts);
  return (uint64_t)ts.tv_sec*1000000000ull+(uint64_t)ts.tv_nsec; }
static uint64_t rng64(uint64_t *s){ uint64_t x=*s; x^=x<<7; x^=x>>9; x*=0x9e3779b97f4a7c15ull; *s=x; return x; }
static void gen_norm(uint64_t *s, uint64_t D[N]){ for(int i=0;i<N;i++) D[i]=rng64(s);
  D[N-1]|=0x8000000000000000ull; }

static double measure_ghz(void){
  uint64_t a=1; const uint64_t M=2000000000ull;
  for(uint64_t i=0;i<50000000ull;i++) a+=a>>1^i;
  uint64_t t0=now_ns(); __asm__ volatile("":"+r"(a));
  for(uint64_t i=0;i<M;i++) __asm__ volatile("add $1,%0":"+r"(a));
  uint64_t t1=now_ns(); volatile uint64_t sink=a;(void)sink;
  return (double)M/(double)(t1-t0);
}

// ---- wrappers with a uniform (D, out, scratch) signature ----
typedef void (*recip_fn)(const uint64_t *D, uint64_t *out, uint64_t *scratch);
static void w_mono   (const uint64_t *D, uint64_t *out, uint64_t *s){ (void)s; recip_mono(D,out); }
static void w_invert (const uint64_t *D, uint64_t *out, uint64_t *s){ __gmpn_invert(out,D,N,s); }
static void w_invappr(const uint64_t *D, uint64_t *out, uint64_t *s){ __gmpn_invertappr(out,D,N,s); }

static double ghz;

static void bench_one(const char *name, recip_fn fn, uint64_t seed){
  uint64_t scratch[4*N];

  // throughput: independent normalized divisors
  const int M=4096;
  static uint64_t Dtab[4096][N];
  uint64_t s=seed; for(int i=0;i<M;i++) gen_norm(&s,Dtab[i]);
  uint64_t out[N], acc=0;
  for(int i=0;i<M;i++){ fn(Dtab[i],out,scratch); acc^=out[3]; }
  double tp=1e300;
  for(int rep=0;rep<7;rep++){
    const int L=200; uint64_t t0=now_ns();
    for(int l=0;l<L;l++) for(int i=0;i<M;i++){ fn(Dtab[i],out,scratch); acc^=out[3]; }
    uint64_t t1=now_ns(); double ns=(double)(t1-t0)/((long)L*M);
    if(ns<tp) tp=ns;
  }

  // latency: dependent chain, out -> next D (renormalized)
  uint64_t D[N]; uint64_t s2=seed^0xdeadbeef; gen_norm(&s2,D); uint64_t X[N];
  for(int i=0;i<100000;i++){ fn(D,X,scratch); memcpy(D,X,sizeof D); D[N-1]|=0x8000000000000000ull; }
  double lat=1e300;
  for(int rep=0;rep<7;rep++){
    const int K=2000000; uint64_t t0=now_ns();
    for(int i=0;i<K;i++){ fn(D,X,scratch); memcpy(D,X,sizeof D); D[N-1]|=0x8000000000000000ull; }
    uint64_t t1=now_ns(); double ns=(double)(t1-t0)/K;
    if(ns<lat) lat=ns;
  }

  printf("%-20s  thr %6.2f ns (%5.1f cyc)   lat %6.2f ns (%5.1f cyc)   [acc=%016lx sink=%016lx]\n",
         name, tp, tp*ghz, lat, lat*ghz, (unsigned long)acc,(unsigned long)(D[0]^X[N-1]));
}

int main(int argc,char**argv){
  cpu_set_t set; CPU_ZERO(&set); CPU_SET(0,&set); sched_setaffinity(0,sizeof(set),&set);
  uint64_t seed = argc>1 ? strtoull(argv[1],NULL,0) : 0x1234567890abcdefull;
  ghz = measure_ghz();
  printf("n=%d limbs (448-bit), core %.3f GHz, lat includes ~1ns memcpy\n", N, ghz);
  printf("%-20s  %-26s %-26s\n","routine","throughput","latency");
  bench_one("recip_mono (asm)", w_mono,    seed);
  bench_one("gmpn_invertappr",  w_invappr, seed);
  bench_one("gmpn_invert",      w_invert,  seed);
  return 0;
}
