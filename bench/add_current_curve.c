#define _POSIX_C_SOURCE 200112L

#include <gmp.h>
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#undef mpn_add_n
#define mpn_add_n simd_current_mpn_add_n
#include "../include/add.h"
#undef mpn_add_n

extern mp_limb_t __gmpn_add_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

#define MAX_LIMBS 2048

static uint64_t splitmix64(uint64_t *x) {
  uint64_t z = (*x += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

static void fill_operands(_limb *ap, _limb *bp) {
  uint64_t seed = UINT64_C(0x2718281828459045);

  for (uint64_t i = 0; i < MAX_LIMBS; i++) {
    ap[i] = splitmix64(&seed);
    bp[i] = splitmix64(&seed);

    switch (i & 63) {
    case 0:
      ap[i] = UINT64_MAX - 2;
      bp[i] = 9;
      break;
    case 1:
    case 2:
      ap[i] = UINT64_MAX;
      bp[i] = 0;
      break;
    case 17:
      ap[i] = UINT64_MAX;
      bp[i] = UINT64_MAX;
      break;
    case 18:
      ap[i] = UINT64_MAX;
      bp[i] = 0;
      break;
    default:
      break;
    }
  }
}

NOINLINE static uint64_t scalar_add_n(_limb *rp, _limb *ap, _limb *bp,
                                      uint64_t n) {
  unsigned __int128 carry = 0;

  for (uint64_t i = 0; i < n; i++) {
    const unsigned __int128 s = (unsigned __int128)ap[i] + bp[i] + carry;
    rp[i] = (_limb)s;
    carry = s >> 64;
  }

  return (uint64_t)carry;
}

NOINLINE static uint64_t current_add_n(_limb *rp, _limb *ap, _limb *bp,
                                       uint64_t n) {
  return simd_current_mpn_add_n(rp, ap, bp, n, 0);
}

NOINLINE static uint64_t gmp_add_n(_limb *rp, _limb *ap, _limb *bp,
                                   uint64_t n) {
  return __gmpn_add_n((mp_ptr)rp, (mp_srcptr)ap, (mp_srcptr)bp, (mp_size_t)n);
}

static uint64_t hash_limbs(const _limb *p, uint64_t n) {
  uint64_t h = UINT64_C(0x9e3779b97f4a7c15) ^ n;
  for (uint64_t i = 0; i < n; i++)
    h = (h ^ p[i]) * UINT64_C(0xbf58476d1ce4e5b9);
  return h;
}

static uint64_t ns_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
  return (uint64_t)ts.tv_sec * UINT64_C(1000000000) + (uint64_t)ts.tv_nsec;
}

typedef uint64_t (*add_fn)(_limb *, _limb *, _limb *, uint64_t);

static _limb *alloc_aligned_limbs(const char *name) {
  void *p = 0;
  const int err = posix_memalign(&p, 64, MAX_LIMBS * sizeof(_limb));
  if (err) {
    errno = err;
    perror(name);
    exit(1);
  }
  if (((uintptr_t)p & 63) != 0) {
    fprintf(stderr, "%s is not 64-byte aligned: %p\n", name, p);
    exit(1);
  }
  memset(p, 0, MAX_LIMBS * sizeof(_limb));
  return (_limb *)p;
}

static double time_one(add_fn fn, _limb *rp, _limb *ap, _limb *bp, uint64_t n,
                       uint64_t reps, unsigned rounds, uint64_t *hash) {
  double best = 1.0e300;

  for (unsigned r = 0; r < rounds; r++) {
    memset(rp, 0, n * sizeof(*rp));
    const uint64_t t0 = ns_now();
    uint64_t acc = 0;

    for (uint64_t rep = 0; rep < reps + r; rep++)
      acc += fn(rp, ap, bp, n);

    const uint64_t t1 = ns_now();
    const double ns = (double)(t1 - t0) / (double)(reps + r);
    if (ns < best)
      best = ns;
    *hash = (*hash * UINT64_C(0x9e3779b97f4a7c15)) ^ hash_limbs(rp, n) ^ acc;
  }

  return best;
}

static void verify_one(_limb *tmp, _limb *ref, _limb *ap, _limb *bp,
                       uint64_t n) {
  memset(tmp, 0, n * sizeof(*tmp));
  memset(ref, 0, n * sizeof(*ref));

  const uint64_t cref = scalar_add_n(ref, ap, bp, n);
  const uint64_t ccur = current_add_n(tmp, ap, bp, n);
  if (cref != ccur || memcmp(tmp, ref, n * sizeof(*tmp)) != 0) {
    fprintf(stderr, "current add verification failed at n=%" PRIu64 "\n", n);
    exit(1);
  }

  memset(tmp, 0, n * sizeof(*tmp));
  const uint64_t cgmp = gmp_add_n(tmp, ap, bp, n);
  if (cref != cgmp || memcmp(tmp, ref, n * sizeof(*tmp)) != 0) {
    fprintf(stderr, "GMP add verification failed at n=%" PRIu64 "\n", n);
    exit(1);
  }
}

int main(int argc, char **argv) {
  const uint64_t max_n = argc > 1 ? strtoull(argv[1], 0, 0) : 512;
  const uint64_t target_limb_ops =
      argc > 2 ? strtoull(argv[2], 0, 0) : 800000;
  const uint64_t min_reps = argc > 3 ? strtoull(argv[3], 0, 0) : 2000;
  const unsigned rounds = argc > 4 ? (unsigned)strtoul(argv[4], 0, 0) : 7;
  const char *csv_path =
      argc > 5 ? argv[5] : "/tmp/simd_bigint_add_current_curve.csv";

  if (max_n == 0 || max_n > MAX_LIMBS) {
    fprintf(stderr, "max_n must be in 1..%u\n", MAX_LIMBS);
    return 1;
  }

  _limb *const a = alloc_aligned_limbs("a");
  _limb *const b = alloc_aligned_limbs("b");
  _limb *const r = alloc_aligned_limbs("r");
  _limb *const ref = alloc_aligned_limbs("ref");

  fill_operands(a, b);

  FILE *csv = fopen(csv_path, "w");
  if (!csv) {
    perror(csv_path);
    return 1;
  }

  fprintf(csv, "n,current_ns,gmp_ns,scalar_ns,current_ns_per_limb,"
               "gmp_ns_per_limb,scalar_ns_per_limb,reps\n");

  uint64_t hash = 0;
  printf("max_n=%" PRIu64 " target_limb_ops=%" PRIu64
         " min_reps=%" PRIu64 " rounds=%u\n",
         max_n, target_limb_ops, min_reps, rounds);

  for (uint64_t n = 1; n <= max_n; n++) {
    verify_one(r, ref, a, b, n);

    uint64_t reps = target_limb_ops / n;
    if (reps < min_reps)
      reps = min_reps;

    const double current_ns = time_one(current_add_n, r, a, b, n, reps, rounds,
                                       &hash);
    const double gmp_ns = time_one(gmp_add_n, r, a, b, n, reps, rounds, &hash);
    const double scalar_ns =
        time_one(scalar_add_n, r, a, b, n, reps, rounds, &hash);

    fprintf(csv,
            "%" PRIu64 ",%.9f,%.9f,%.9f,%.12f,%.12f,%.12f,%" PRIu64 "\n",
            n, current_ns, gmp_ns, scalar_ns, current_ns / (double)n,
            gmp_ns / (double)n, scalar_ns / (double)n, reps);

    if (n <= 16 || (n & 31) == 0)
      printf("%4" PRIu64 " limbs: current %.4f ns/limb, gmp %.4f ns/limb, "
             "scalar %.4f ns/limb\n",
             n, current_ns / (double)n, gmp_ns / (double)n,
             scalar_ns / (double)n);
  }

  fclose(csv);
  printf("csv=%s hash=%016" PRIx64 "\n", csv_path, hash);

  free(a);
  free(b);
  free(r);
  free(ref);

  return 0;
}
