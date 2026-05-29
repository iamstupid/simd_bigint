#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <gmp.h>

#include "../include/add.h"

static_assert(sizeof(mp_limb_t) == sizeof(_limb));

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

static constexpr size_t MAX_LIMBS = 2048;

static uint64_t splitmix64(uint64_t *x) {
  uint64_t z = (*x += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

static void fill_operands(_limb *ap, _limb *bp) {
  uint64_t seed = 0x3141592653589793ull;
  for (size_t i = 0; i < MAX_LIMBS; i++) {
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

NOINLINE uint64_t scalar_add_n(_limb *rp, const _limb *ap, const _limb *bp,
                               size_t n) {
  unsigned __int128 carry = 0;
  for (size_t i = 0; i < n; i++) {
    const unsigned __int128 s = (unsigned __int128)ap[i] + bp[i] + carry;
    rp[i] = (_limb)s;
    carry = s >> 64;
  }
  return (uint64_t)carry;
}

NOINLINE uint64_t simd_add_n_wrap(_limb *rp, const _limb *ap, const _limb *bp,
                                  size_t n) {
  return simd_mpn_add_n_nonzero(rp, ap, bp, n);
}

NOINLINE uint64_t gmp_add_n_wrap(_limb *rp, const _limb *ap, const _limb *bp,
                                 size_t n) {
  return mpn_add_n((mp_ptr)rp, (mp_srcptr)ap, (mp_srcptr)bp, (mp_size_t)n);
}

static uint64_t hash_limbs(const _limb *p, size_t n) {
  uint64_t h = 0x9e3779b97f4a7c15ull ^ n;
  for (size_t i = 0; i < n; i++)
    h = (h ^ p[i]) * 0xbf58476d1ce4e5b9ull;
  return h;
}

using add_fn = uint64_t (*)(_limb *, const _limb *, const _limb *, size_t);

static double time_one(add_fn fn, _limb *rp, const _limb *ap, const _limb *bp,
                       size_t n, uint64_t reps, unsigned rounds,
                       uint64_t *hash) {
  double best_ns = 1.0e300;

  for (unsigned r = 0; r < rounds; r++) {
    memset(rp, 0, n * sizeof(*rp));
    const auto t0 = std::chrono::steady_clock::now();
    uint64_t acc = 0;
    for (uint64_t rep = 0; rep < reps + r; rep++)
      acc += fn(rp, ap, bp, n);
    const auto t1 = std::chrono::steady_clock::now();

    const double ns =
        (double)std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0)
            .count() /
        (double)(reps + r);
    best_ns = std::min(best_ns, ns);
    *hash = (*hash * 0x9e3779b97f4a7c15ull) ^ hash_limbs(rp, n) ^ acc;
  }

  return best_ns;
}

static void verify_one(_limb *tmp, _limb *ref, const _limb *ap,
                       const _limb *bp, size_t n) {
  memset(tmp, 0, n * sizeof(*tmp));
  memset(ref, 0, n * sizeof(*ref));

  const uint64_t cref = scalar_add_n(ref, ap, bp, n);
  const uint64_t cgot = simd_add_n_wrap(tmp, ap, bp, n);

  if (cref != cgot || memcmp(tmp, ref, n * sizeof(*tmp)) != 0) {
    fprintf(stderr, "simd_mpn_add_n verification failed at n=%zu\n", n);
    exit(1);
  }

  memset(tmp, 0, n * sizeof(*tmp));
  const uint64_t cgmp = gmp_add_n_wrap(tmp, ap, bp, n);

  if (cref != cgmp || memcmp(tmp, ref, n * sizeof(*tmp)) != 0) {
    fprintf(stderr, "GMP mpn_add_n verification failed at n=%zu\n", n);
    exit(1);
  }
}

int main(int argc, char **argv) {
  const size_t max_n = argc > 1 ? strtoull(argv[1], nullptr, 0) : 512;
  const uint64_t target_limb_ops =
      argc > 2 ? strtoull(argv[2], nullptr, 0) : 800000;
  const uint64_t min_reps = argc > 3 ? strtoull(argv[3], nullptr, 0) : 2000;
  const unsigned rounds =
      argc > 4 ? (unsigned)strtoul(argv[4], nullptr, 0) : 7;
  const char *csv_path =
      argc > 5 ? argv[5] : "/tmp/simd_bigint_add_n_curve.csv";

  if (max_n == 0 || max_n > MAX_LIMBS) {
    fprintf(stderr, "max_n must be in 1..%zu\n", MAX_LIMBS);
    return 1;
  }

  alignas(64) _limb a[MAX_LIMBS];
  alignas(64) _limb b[MAX_LIMBS];
  alignas(64) _limb r[MAX_LIMBS];
  alignas(64) _limb ref[MAX_LIMBS];

  fill_operands(a, b);

  FILE *csv = fopen(csv_path, "w");
  if (csv == nullptr) {
    perror(csv_path);
    return 1;
  }
  fprintf(csv,
          "n,simd_ns,gmp_ns,scalar_ns,simd_ns_per_limb,gmp_ns_per_limb,"
          "scalar_ns_per_limb,reps\n");

  uint64_t hash = 0;
  printf("max_n=%zu target_limb_ops=%" PRIu64 " min_reps=%" PRIu64
         " rounds=%u\n",
         max_n, target_limb_ops, min_reps, rounds);

  for (size_t n = 1; n <= max_n; n++) {
    verify_one(r, ref, a, b, n);
    const uint64_t reps = std::max<uint64_t>(min_reps, target_limb_ops / n);
    const double simd_ns =
        time_one(simd_add_n_wrap, r, a, b, n, reps, rounds, &hash);
    const double gmp_ns =
        time_one(gmp_add_n_wrap, r, a, b, n, reps, rounds, &hash);
    const double scalar_ns =
        time_one(scalar_add_n, r, a, b, n, reps, rounds, &hash);

    fprintf(csv, "%zu,%.9f,%.9f,%.9f,%.12f,%.12f,%.12f,%" PRIu64 "\n", n,
            simd_ns, gmp_ns, scalar_ns, simd_ns / (double)n,
            gmp_ns / (double)n, scalar_ns / (double)n, reps);

    if (n <= 16 || (n & 31) == 0)
      printf("%4zu limbs: simd %.4f ns/limb, gmp %.4f ns/limb, "
             "scalar %.4f ns/limb\n",
             n, simd_ns / (double)n, gmp_ns / (double)n,
             scalar_ns / (double)n);
  }

  fclose(csv);
  printf("csv=%s hash=%016" PRIx64 "\n", csv_path, hash);

  return 0;
}
