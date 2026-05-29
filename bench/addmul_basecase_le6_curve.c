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
extern mp_limb_t __gmpn_add_n(mp_limb_t *rp, const mp_limb_t *ap,
                              const mp_limb_t *bp, mp_size_t n);
extern mp_limb_t __gmpn_addmul_1(mp_limb_t *rp, const mp_limb_t *up,
                                 mp_size_t n, mp_limb_t v0);

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

static uint64_t
gmp_addmul1_rows(uint64_t *rp, const uint64_t *ap, size_t an,
                 const uint64_t *bp, size_t bn)
{
  uint64_t carry_out = 0;

  if (an < bn) {
    const uint64_t *tp = ap;
    ap = bp;
    bp = tp;
    size_t tn = an;
    an = bn;
    bn = tn;
  }

  for (size_t j = 0; j < bn; ++j) {
    uint64_t cy = __gmpn_addmul_1((mp_limb_t *)(rp + j),
                                  (const mp_limb_t *)ap, (mp_size_t)an, bp[j]);
    unsigned __int128 hi = (unsigned __int128)rp[j + an] + cy + carry_out;
    rp[j + an] = (uint64_t)hi;
    carry_out = (uint64_t)(hi >> 64);
  }

  return carry_out;
}

static uint64_t
gmp_tmp_mul_add(uint64_t *rp, const uint64_t *ap, size_t an, const uint64_t *bp,
                size_t bn)
{
  alignas(64) uint64_t tmp[16];
  const size_t rn = an + bn;

  if (an >= bn)
    __gmpn_mul_basecase((mp_limb_t *)tmp, (const mp_limb_t *)ap, (mp_size_t)an,
                        (const mp_limb_t *)bp, (mp_size_t)bn);
  else
    __gmpn_mul_basecase((mp_limb_t *)tmp, (const mp_limb_t *)bp, (mp_size_t)bn,
                        (const mp_limb_t *)ap, (mp_size_t)an);

  return __gmpn_add_n((mp_limb_t *)rp, (const mp_limb_t *)rp,
                      (const mp_limb_t *)tmp, (mp_size_t)rn);
}

static uint64_t
ref_addmul(uint64_t *rp, const uint64_t *ap, size_t an, const uint64_t *bp,
           size_t bn)
{
  uint64_t carry_out = 0;

  for (size_t j = 0; j < bn; ++j) {
    unsigned __int128 carry = 0;
    for (size_t i = 0; i < an; ++i) {
      unsigned __int128 t = (unsigned __int128)ap[i] * bp[j] + rp[i + j] + carry;
      rp[i + j] = (uint64_t)t;
      carry = t >> 64;
    }

    unsigned __int128 hi = (unsigned __int128)rp[j + an] + carry + carry_out;
    rp[j + an] = (uint64_t)hi;
    carry_out = (uint64_t)(hi >> 64);
  }

  return carry_out;
}

static void
verify_pair(size_t an, size_t bn)
{
  uint64_t a[6], b[6], init[16], simd[16], rows[16], tmp[16], ref[16];
  uint64_t seed = 0xabcdef9876543210ull + an * 257 + bn;

  for (size_t trial = 0; trial < 4096; ++trial) {
    for (size_t i = 0; i < 16; ++i) {
      uint64_t z = rng64(&seed);
      if ((trial & 127) == 0) z = UINT64_MAX;
      init[i] = z;
    }
    for (size_t i = 0; i < 6; ++i) {
      a[i] = rng64(&seed);
      b[i] = rng64(&seed);
    }
    if ((trial & 7) == 0) a[0] = UINT64_MAX;
    if ((trial & 15) == 0) b[0] = UINT64_MAX;
    if ((trial & 31) == 0) a[an - 1] = UINT64_MAX;
    if ((trial & 63) == 0) b[bn - 1] = UINT64_MAX;

    memcpy(simd, init, sizeof(simd));
    memcpy(rows, init, sizeof(rows));
    memcpy(tmp, init, sizeof(tmp));
    memcpy(ref, init, sizeof(ref));

    uint64_t c_ref = ref_addmul(ref, a, an, b, bn);
    uint64_t c_simd = simd_mpn_addmul_basecase_le6(simd, a, an, b, bn);
    uint64_t c_rows = gmp_addmul1_rows(rows, a, an, b, bn);
    uint64_t c_tmp = gmp_tmp_mul_add(tmp, a, an, b, bn);

    if (c_simd != c_ref || c_rows != c_ref || c_tmp != c_ref ||
        memcmp(simd, ref, (an + bn) * sizeof(uint64_t)) != 0 ||
        memcmp(rows, ref, (an + bn) * sizeof(uint64_t)) != 0 ||
        memcmp(tmp, ref, (an + bn) * sizeof(uint64_t)) != 0) {
      fprintf(stderr, "verify failed an=%zu bn=%zu trial=%zu\n", an, bn, trial);
      exit(1);
    }
  }
}

typedef uint64_t (*bench_fn)(uint64_t *, const uint64_t *, size_t,
                             const uint64_t *, size_t);

static uint64_t
simd_call(uint64_t *rp, const uint64_t *ap, size_t an, const uint64_t *bp,
          size_t bn)
{
  return simd_mpn_addmul_basecase_le6(rp, ap, an, bp, bn);
}

static uint64_t
bench_one(bench_fn fn, size_t an, size_t bn, size_t reps, const uint64_t *a,
          const uint64_t *b, const uint64_t *init)
{
  alignas(64) uint64_t r[16];
  memcpy(r, init, sizeof(r));

  uint64_t h = 0;
  const size_t last = an + bn - 1;
  for (size_t k = 0; k < reps; ++k) {
    uint64_t cy = fn(r, a, an, b, bn);
    h ^= r[0] + r[last] + cy + 0x9e3779b97f4a7c15ull + (h << 6) + (h >> 2);
  }
  return h;
}

static double
best_time_ns(bench_fn fn, size_t an, size_t bn, size_t reps, int rounds,
             const uint64_t *a, const uint64_t *b, const uint64_t *init,
             uint64_t *hash)
{
  double best = 1e300;
  for (int r = 0; r < rounds; ++r) {
    uint64_t t0 = now_ns();
    uint64_t h = bench_one(fn, an, bn, reps, a, b, init);
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
  size_t reps = argc > 1 ? strtoull(argv[1], 0, 0) : 2000000ull;
  int rounds = argc > 2 ? atoi(argv[2]) : 11;
  const char *csv_path =
      argc > 3 ? argv[3] : "/tmp/simd_bigint_addmul_le6_vs_gmp.csv";

  FILE *csv = fopen(csv_path, "w");
  if (!csv) {
    perror(csv_path);
    return 1;
  }

  fprintf(csv, "an,bn,simd_ns,gmp_rows_ns,gmp_tmp_ns,rows_speedup,"
               "tmp_speedup\n");

  uint64_t hash = 0;
  printf("reps=%zu rounds=%d\n", reps, rounds);

  for (size_t an = 1; an <= 6; ++an) {
    for (size_t bn = 1; bn <= 6; ++bn) {
      alignas(64) uint64_t a[6], b[6], init[16];
      uint64_t seed = 0x2718281828459045ull + an * 4099 + bn * 17;

      for (size_t i = 0; i < 6; ++i) {
        a[i] = rng64(&seed);
        b[i] = rng64(&seed);
      }
      for (size_t i = 0; i < 16; ++i)
        init[i] = rng64(&seed);
      a[an - 1] |= 1;
      b[bn - 1] |= 1;

      verify_pair(an, bn);

      double simd_ns =
          best_time_ns(simd_call, an, bn, reps, rounds, a, b, init, &hash);
      double rows_ns = best_time_ns(gmp_addmul1_rows, an, bn, reps, rounds, a,
                                    b, init, &hash);
      double tmp_ns = best_time_ns(gmp_tmp_mul_add, an, bn, reps, rounds, a, b,
                                   init, &hash);

      fprintf(csv, "%zu,%zu,%.9f,%.9f,%.9f,%.9f,%.9f\n", an, bn, simd_ns,
              rows_ns, tmp_ns, rows_ns / simd_ns, tmp_ns / simd_ns);

      printf("%zux%zu: simd %.4f ns  gmp-rows %.4f ns %.3fx  "
             "gmp-tmp %.4f ns %.3fx\n",
             an, bn, simd_ns, rows_ns, rows_ns / simd_ns, tmp_ns,
             tmp_ns / simd_ns);
      fflush(stdout);
    }
  }

  fclose(csv);
  printf("csv=%s hash=%016" PRIx64 "\n", csv_path, hash);
  return 0;
}
