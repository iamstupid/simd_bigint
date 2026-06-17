// Head-to-head: recip_mono (reciprocal_u416.s) vs recip_mono_opt
// (reciprocal_u416_opt.s, reordered multiply to shed adcx/adox).
//   void recip_mono     (const u64 D[7] normalized, u64 xout[7])
//   void recip_mono_opt (const u64 D[7] normalized, u64 xout[7])
//
// Reports per-variant: validation (max |X-ref| ULP), throughput (independent
// calls), latency (dependent chain), plus a cross-check that both variants
// produce bit-identical output. Cycles via measured core-GHz calibration.
//
// Build:
//   gcc -O3 -march=native recip_opt_compare.c \
//       ../reciprocal_u416.s ../reciprocal_u416_opt.s -o /tmp/recip_opt -lgmp
#define _GNU_SOURCE
#include <gmp.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void recip_mono(const uint64_t *D, uint64_t *xout);
extern void recip_mono_opt(const uint64_t *D, uint64_t *xout);

typedef void (*recip_fn)(const uint64_t *, uint64_t *);

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

static double
measure_ghz(void)
{
  volatile uint64_t sink = 0;
  uint64_t a = 1;
  const uint64_t N = 2000000000ull;
  for (uint64_t i = 0; i < 50000000ull; i++) a += a >> 1 ^ i;
  uint64_t t0 = now_ns();
  __asm__ volatile("" : "+r"(a));
  for (uint64_t i = 0; i < N; i++)
    __asm__ volatile("add $1, %0" : "+r"(a));
  uint64_t t1 = now_ns();
  sink = a;
  (void)sink;
  double ns = (double)(t1 - t0);
  return (double)N / ns;
}

// ref - X over NV random normalized divisors; fills [lo,hi]
static void
validate(recip_fn fn, uint64_t seed, int NV, long *lo, long *hi)
{
  uint64_t s = seed ^ 0xa5a5a5a5;
  long worst_lo = 1L << 40, worst_hi = -(1L << 40);
  mpz_t num, md, mx, q, r;
  mpz_inits(num, md, mx, q, r, NULL);
  mpz_ui_pow_ui(num, 2, 895);
  for (int t = 0; t < NV; t++) {
    uint64_t D[7], X[7];
    gen_norm(&s, D);
    fn(D, X);
    mpz_import(md, 7, -1, 8, 0, 0, D);
    mpz_import(mx, 7, -1, 8, 0, 0, X);
    mpz_fdiv_q(q, num, md);
    mpz_sub(r, q, mx);
    long e = mpz_get_si(r);
    if (e < worst_lo) worst_lo = e;
    if (e > worst_hi) worst_hi = e;
  }
  mpz_clears(num, md, mx, q, r, NULL);
  *lo = worst_lo;
  *hi = worst_hi;
}

static double
bench_throughput(recip_fn fn, uint64_t seed, double ghz, uint64_t *accout)
{
  const int M = 4096;
  uint64_t(*Dtab)[7] = malloc(sizeof(uint64_t[7]) * M);
  uint64_t s = seed;
  for (int i = 0; i < M; i++)
    gen_norm(&s, Dtab[i]);
  uint64_t out[7];
  uint64_t acc = 0;
  for (int i = 0; i < M; i++) { fn(Dtab[i], out); acc ^= out[0]; }
  double best_ns = 1e300;
  for (int rep = 0; rep < 9; rep++) {
    const int LOOPS = 200;
    uint64_t t0 = now_ns();
    for (int l = 0; l < LOOPS; l++)
      for (int i = 0; i < M; i++) { fn(Dtab[i], out); acc ^= out[3]; }
    uint64_t t1 = now_ns();
    double ns = (double)(t1 - t0) / ((long)LOOPS * M);
    if (ns < best_ns) best_ns = ns;
  }
  free(Dtab);
  *accout = acc;
  (void)ghz;
  return best_ns;
}

static double
bench_latency(recip_fn fn, uint64_t seed, uint64_t *sinkout)
{
  uint64_t s = seed ^ 0xdeadbeef;
  uint64_t D[7], X[7];
  gen_norm(&s, D);
  for (int i = 0; i < 100000; i++) {
    fn(D, X);
    memcpy(D, X, sizeof D);
    D[6] |= 0x8000000000000000ull;
  }
  double best_ns = 1e300;
  for (int rep = 0; rep < 9; rep++) {
    const int N = 2000000;
    uint64_t t0 = now_ns();
    for (int i = 0; i < N; i++) {
      fn(D, X);
      memcpy(D, X, sizeof D);
      D[6] |= 0x8000000000000000ull;
    }
    uint64_t t1 = now_ns();
    double ns = (double)(t1 - t0) / N;
    if (ns < best_ns) best_ns = ns;
  }
  *sinkout = D[0] ^ X[6];
  return best_ns;
}

int
main(int argc, char **argv)
{
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(0, &set);
  sched_setaffinity(0, sizeof(set), &set);

  uint64_t seed = 0x1234567890abcdefull;
  if (argc > 1)
    seed = strtoull(argv[1], NULL, 0);

  double ghz = measure_ghz();

  // ---------- cross-check: bit-identical output? ----------
  {
    uint64_t s = seed ^ 0xc0ffee;
    const int NC = 1000000;
    long mism = 0, maxdiff = 0;
    mpz_t a, b, d;
    mpz_inits(a, b, d, NULL);
    for (int t = 0; t < NC; t++) {
      uint64_t D[7], Xa[7], Xb[7];
      gen_norm(&s, D);
      recip_mono(D, Xa);
      recip_mono_opt(D, Xb);
      if (memcmp(Xa, Xb, sizeof Xa) != 0) {
        mism++;
        mpz_import(a, 7, -1, 8, 0, 0, Xa);
        mpz_import(b, 7, -1, 8, 0, 0, Xb);
        mpz_sub(d, a, b);
        mpz_abs(d, d);
        long dd = mpz_get_si(d);
        if (dd > maxdiff) maxdiff = dd;
      }
    }
    mpz_clears(a, b, d, NULL);
    printf("cross-check: %d trials, %ld differ (max |orig-opt| = %ld ULP)  %s\n",
           NC, mism, maxdiff,
           mism == 0 ? "(bit-identical)" : "(differ)");
  }

  // ---------- validation ----------
  {
    long lo, hi;
    validate(recip_mono, seed, 300000, &lo, &hi);
    printf("validate orig: (ref-X) in [%ld, %ld] ULP\n", lo, hi);
    validate(recip_mono_opt, seed, 300000, &lo, &hi);
    printf("validate opt : (ref-X) in [%ld, %ld] ULP\n", lo, hi);
  }

  // ---------- throughput ----------
  {
    uint64_t acc_o, acc_p;
    double o = bench_throughput(recip_mono, seed, ghz, &acc_o);
    double p = bench_throughput(recip_mono_opt, seed, ghz, &acc_p);
    printf("\nthroughput  orig: %.2f ns  %.1f cyc\n", o, o * ghz);
    printf("throughput  opt : %.2f ns  %.1f cyc   (%.2f%% %s)\n",
           p, p * ghz, (o - p) / o * 100.0, p < o ? "faster" : "slower");
    printf("            (acc_o=%016lx acc_p=%016lx)\n",
           (unsigned long)acc_o, (unsigned long)acc_p);
  }

  // ---------- latency ----------
  {
    uint64_t sink_o, sink_p;
    double o = bench_latency(recip_mono, seed, &sink_o);
    double p = bench_latency(recip_mono_opt, seed, &sink_p);
    printf("\nlatency     orig: %.2f ns  %.1f cyc\n", o, o * ghz);
    printf("latency     opt : %.2f ns  %.1f cyc   (%.2f%% %s)\n",
           p, p * ghz, (o - p) / o * 100.0, p < o ? "faster" : "slower");
    printf("            (sink_o=%016lx sink_p=%016lx)\n",
           (unsigned long)sink_o, (unsigned long)sink_p);
  }

  printf("\ncore freq:  %.3f GHz (measured)\n", ghz);
  return 0;
}
