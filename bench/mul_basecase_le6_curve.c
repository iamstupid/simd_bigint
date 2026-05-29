#include "../include/mul_basecase_le6.h"

#include <gmp.h>
#include <inttypes.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern void __gmpn_mul_basecase(mp_limb_t *rp, const mp_limb_t *up,
                                mp_size_t un, const mp_limb_t *vp,
                                mp_size_t vn);

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
gmp_mul_basecase_norm(uint64_t *rp, const uint64_t *ap, size_t an,
                      const uint64_t *bp, size_t bn)
{
  if (an >= bn)
    __gmpn_mul_basecase((mp_limb_t *)rp, (const mp_limb_t *)ap, (mp_size_t)an,
                        (const mp_limb_t *)bp, (mp_size_t)bn);
  else
    __gmpn_mul_basecase((mp_limb_t *)rp, (const mp_limb_t *)bp, (mp_size_t)bn,
                        (const mp_limb_t *)ap, (mp_size_t)an);
}

static void
ref_mul(uint64_t *rp, const uint64_t *ap, size_t an, const uint64_t *bp,
        size_t bn)
{
  memset(rp, 0, (an + bn) * sizeof(rp[0]));
  for (size_t j = 0; j < bn; ++j) {
    unsigned __int128 carry = 0;
    for (size_t i = 0; i < an; ++i) {
      unsigned __int128 t = (unsigned __int128)ap[i] * bp[j] + rp[i + j] + carry;
      rp[i + j] = (uint64_t)t;
      carry = t >> 64;
    }
    rp[j + an] = (uint64_t)carry;
  }
}

static void
verify_pair(size_t an, size_t bn)
{
  uint64_t a[6], b[6], simd[12], gmp[12], ref[12];
  uint64_t seed = 0x3141592653589793ull + an * 257 + bn;

  for (size_t trial = 0; trial < 4096; ++trial) {
    for (size_t i = 0; i < 6; ++i) {
      a[i] = rng64(&seed);
      b[i] = rng64(&seed);
    }
    if ((trial & 7) == 0) a[0] = UINT64_MAX;
    if ((trial & 15) == 0) b[0] = UINT64_MAX;
    if ((trial & 31) == 0) a[an - 1] = UINT64_MAX;
    if ((trial & 63) == 0) b[bn - 1] = UINT64_MAX;

    memset(simd, 0xa5, sizeof(simd));
    memset(gmp, 0x5a, sizeof(gmp));
    ref_mul(ref, a, an, b, bn);
    simd_mpn_mul_basecase_le6(simd, a, an, b, bn);
    gmp_mul_basecase_norm(gmp, a, an, b, bn);

    if (memcmp(simd, ref, (an + bn) * sizeof(uint64_t)) != 0 ||
        memcmp(gmp, ref, (an + bn) * sizeof(uint64_t)) != 0) {
      fprintf(stderr, "verify failed an=%zu bn=%zu trial=%zu\n", an, bn, trial);
      exit(1);
    }
  }
}

static uint64_t
bench_simd(size_t an, size_t bn, size_t reps, const uint64_t *a,
           const uint64_t *b)
{
  alignas(64) uint64_t r[16];
  uint64_t h = 0;
  const size_t last = an + bn - 1;
  for (size_t k = 0; k < reps; ++k) {
    simd_mpn_mul_basecase_le6(r, a, an, b, bn);
    h ^= r[0] + r[last] + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  }
  return h;
}

static uint64_t
bench_gmp(size_t an, size_t bn, size_t reps, const uint64_t *a,
          const uint64_t *b)
{
  alignas(64) uint64_t r[16];
  uint64_t h = 0;
  const size_t last = an + bn - 1;
  for (size_t k = 0; k < reps; ++k) {
    gmp_mul_basecase_norm(r, a, an, b, bn);
    h ^= r[0] + r[last] + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  }
  return h;
}

static double
best_time_ns(uint64_t (*fn)(size_t, size_t, size_t, const uint64_t *,
                            const uint64_t *),
             size_t an, size_t bn, size_t reps, int rounds, const uint64_t *a,
             const uint64_t *b, uint64_t *hash)
{
  double best = 1e300;
  for (int r = 0; r < rounds; ++r) {
    uint64_t t0 = now_ns();
    uint64_t h = fn(an, bn, reps, a, b);
    *hash ^= h + 0x9e3779b97f4a7c15ull + (*hash << 6) + (*hash >> 2);
    uint64_t t1 = now_ns();
    double ns = (double)(t1 - t0) / (double)reps;
    if (ns < best) best = ns;
  }
  return best;
}

int
main(int argc, char **argv)
{
  size_t reps = argc > 1 ? strtoull(argv[1], 0, 0) : 20000000ull;
  int rounds = argc > 2 ? atoi(argv[2]) : 11;
  const char *csv_path = argc > 3 ? argv[3] : "/tmp/simd_bigint_mul_le6_vs_gmp.csv";

  FILE *csv = fopen(csv_path, "w");
  if (!csv) {
    perror(csv_path);
    return 1;
  }

  fprintf(csv, "an,bn,simd_ns,gmp_ns,speedup,simd_ns_per_product_limb,"
               "gmp_ns_per_product_limb\n");

  uint64_t hash = 0;
  printf("reps=%zu rounds=%d\n", reps, rounds);

  for (size_t an = 1; an <= 6; ++an) {
    for (size_t bn = 1; bn <= 6; ++bn) {
      alignas(64) uint64_t a[6], b[6];
      uint64_t seed = 0x2718281828459045ull + an * 4099 + bn * 17;
      for (size_t i = 0; i < 6; ++i) {
        a[i] = rng64(&seed);
        b[i] = rng64(&seed);
      }
      a[an - 1] |= 1;
      b[bn - 1] |= 1;

      verify_pair(an, bn);

      double simd_ns = best_time_ns(bench_simd, an, bn, reps, rounds, a, b, &hash);
      double gmp_ns = best_time_ns(bench_gmp, an, bn, reps, rounds, a, b, &hash);
      double plimbs = (double)(an + bn);

      fprintf(csv, "%zu,%zu,%.9f,%.9f,%.9f,%.9f,%.9f\n", an, bn, simd_ns,
              gmp_ns, gmp_ns / simd_ns, simd_ns / plimbs, gmp_ns / plimbs);

      printf("%zux%zu: simd %.4f ns  gmp %.4f ns  speedup %.3fx\n", an, bn,
             simd_ns, gmp_ns, gmp_ns / simd_ns);
      fflush(stdout);
    }
  }

  fclose(csv);
  printf("csv=%s hash=%016" PRIx64 "\n", csv_path, hash);
  return 0;
}
