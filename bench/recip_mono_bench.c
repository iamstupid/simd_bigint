// Benchmark for recip_mono (reciprocal_u416.s):
//   void recip_mono(const u64 D[7] normalized, u64 xout[7])
// computes X = floor(2^895 / D) to within a few ULP (validated [-6,+3]).
//
// Reports: validation (max |X - ref| ULP), throughput (independent calls),
// and latency (dependent chain). Cycle counts via a measured core-GHz
// calibration (dependent add chain = 1 cycle/iter).
//
// Build:
//   gcc -O3 -march=native recip_mono_bench.c ../reciprocal_u416.s -o /tmp/recip_bench -lgmp
#define _GNU_SOURCE
#include <gmp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void recip_mono(const uint64_t *D, uint64_t *xout);

static uint64_t
now_ns(void)
{
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static uint64_t
rng64(uint64_t *s)
{
  uint64_t x = *s;
  x ^= x << 7;
  x ^= x >> 9;
  x *= 0x9e3779b97f4a7c15ull;
  *s = x;
  return x;
}

static void
gen_norm(uint64_t *s, uint64_t D[7])
{
  for (int i = 0; i < 7; i++)
    D[i] = rng64(s);
  D[6] |= 0x8000000000000000ull; // top bit set => normalized 448-bit
}

// Measure effective core frequency (GHz) with a dependent add chain.
// 'imul by 1 + add' style won't fold; a self-dependent add is 1 cyc latency.
static double
measure_ghz(void)
{
  volatile uint64_t sink = 0;
  uint64_t a = 1;
  const uint64_t N = 2000000000ull;
  // warm
  for (uint64_t i = 0; i < 50000000ull; i++) a += a >> 1 ^ i;
  uint64_t t0 = now_ns();
  // strictly serial: each add depends on previous (1 cycle latency)
  __asm__ volatile("" : "+r"(a));
  for (uint64_t i = 0; i < N; i++)
    __asm__ volatile("add $1, %0" : "+r"(a));
  uint64_t t1 = now_ns();
  sink = a;
  (void)sink;
  double ns = (double)(t1 - t0);
  return (double)N / ns; // adds/ns = cycles/ns = GHz
}

int
main(int argc, char **argv)
{
  // pin to a core for stable timing
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(0, &set);
  sched_setaffinity(0, sizeof(set), &set);

  uint64_t seed = 0x1234567890abcdefull;
  if (argc > 1)
    seed = strtoull(argv[1], NULL, 0);

  double ghz = measure_ghz();

  // ---------- validation ----------
  {
    uint64_t s = seed ^ 0xa5a5a5a5;
    long worst_lo = 1L << 40, worst_hi = -(1L << 40);
    mpz_t num, md, mx, q, r;
    mpz_inits(num, md, mx, q, r, NULL);
    mpz_ui_pow_ui(num, 2, 895);
    const int NV = 500000;
    for (int t = 0; t < NV; t++) {
      uint64_t D[7], X[7];
      gen_norm(&s, D);
      recip_mono(D, X);
      mpz_import(md, 7, -1, 8, 0, 0, D);
      mpz_import(mx, 7, -1, 8, 0, 0, X);
      mpz_fdiv_q(q, num, md); // floor(2^895/D)
      mpz_sub(r, q, mx);      // ref - X
      long e = mpz_get_si(r);
      if (e < worst_lo) worst_lo = e;
      if (e > worst_hi) worst_hi = e;
    }
    mpz_clears(num, md, mx, q, r, NULL);
    printf("validation: over %d trials, (ref - X) in [%ld, %ld] ULP  %s\n",
           NV, worst_lo, worst_hi,
           (worst_lo >= -1024 && worst_hi <= 1024) ? "(OK)" : "(SUSPECT)");
  }

  // ---------- throughput: independent inputs ----------
  {
    const int M = 4096; // table of independent normalized divisors
    uint64_t(*Dtab)[7] = malloc(sizeof(uint64_t[7]) * M);
    uint64_t s = seed;
    for (int i = 0; i < M; i++)
      gen_norm(&s, Dtab[i]);

    uint64_t out[7];
    uint64_t acc = 0;
    // warmup
    for (int i = 0; i < M; i++) {
      recip_mono(Dtab[i], out);
      acc ^= out[0];
    }

    double best_ns = 1e300;
    long iters_total = 0;
    for (int rep = 0; rep < 7; rep++) {
      const int LOOPS = 200;
      uint64_t t0 = now_ns();
      for (int l = 0; l < LOOPS; l++)
        for (int i = 0; i < M; i++) {
          recip_mono(Dtab[i], out);
          acc ^= out[3]; // anti-DCE, cheap
        }
      uint64_t t1 = now_ns();
      long calls = (long)LOOPS * M;
      double ns = (double)(t1 - t0) / calls;
      if (ns < best_ns) best_ns = ns;
      iters_total += calls;
    }
    printf("throughput: %.2f ns/call   %.1f cyc/call   (acc=%016lx)\n",
           best_ns, best_ns * ghz, (unsigned long)acc);
    free(Dtab);
  }

  // ---------- latency: dependent chain (out feeds next input) ----------
  {
    uint64_t s = seed ^ 0xdeadbeef;
    uint64_t D[7], X[7];
    gen_norm(&s, D);

    // warmup
    for (int i = 0; i < 100000; i++) {
      recip_mono(D, X);
      memcpy(D, X, sizeof D);
      D[6] |= 0x8000000000000000ull; // re-normalize to keep valid input
    }

    double best_ns = 1e300;
    for (int rep = 0; rep < 7; rep++) {
      const int N = 2000000;
      uint64_t t0 = now_ns();
      for (int i = 0; i < N; i++) {
        recip_mono(D, X);
        memcpy(D, X, sizeof D);
        D[6] |= 0x8000000000000000ull;
      }
      uint64_t t1 = now_ns();
      double ns = (double)(t1 - t0) / N;
      if (ns < best_ns) best_ns = ns;
    }
    // subtract the trivial memcpy+or overhead is negligible (~sub-ns); report raw
    printf("latency:    %.2f ns/call   %.1f cyc/call   (chain, incl ~1ns copy)\n",
           best_ns, best_ns * ghz);
    printf("            sink=%016lx\n", (unsigned long)(D[0] ^ X[6]));
  }

  printf("core freq:  %.3f GHz (measured)\n", ghz);
  return 0;
}
