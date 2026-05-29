#define _POSIX_C_SOURCE 200112L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#undef mpn_add_n
#define mpn_add_n simd_current_mpn_add_n
#include "../include/add.h"
#undef mpn_add_n

#define MAX_LIMBS 4096
#define GUARD_LIMBS 16
#define GUARD_WORD UINT64_C(0x6a09e667f3bcc909)

static uint64_t splitmix64(uint64_t *x) {
  uint64_t z = (*x += UINT64_C(0x9e3779b97f4a7c15));
  z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
  return z ^ (z >> 31);
}

static _limb *alloc_limbs(const char *name) {
  void *p = 0;
  const int err =
      posix_memalign(&p, 64, (MAX_LIMBS + GUARD_LIMBS) * sizeof(_limb));
  if (err) {
    errno = err;
    perror(name);
    exit(1);
  }
  return (_limb *)p;
}

static uint8_t scalar_add_n(_limb *rp, const _limb *ap, const _limb *bp,
                            uint64_t n, uint8_t carry) {
  unsigned __int128 c = carry;

  for (uint64_t i = 0; i < n; i++) {
    const unsigned __int128 s = (unsigned __int128)ap[i] + bp[i] + c;
    rp[i] = (_limb)s;
    c = s >> 64;
  }

  return (uint8_t)c;
}

static void fill_case(_limb *ap, _limb *bp, uint64_t n, uint64_t trial,
                      uint64_t *seed) {
  for (uint64_t i = 0; i < n + GUARD_LIMBS; i++) {
    ap[i] = GUARD_WORD;
    bp[i] = GUARD_WORD;
  }

  switch (trial & 7) {
  case 0:
    for (uint64_t i = 0; i < n; i++) {
      ap[i] = 0;
      bp[i] = 0;
    }
    break;
  case 1:
    for (uint64_t i = 0; i < n; i++) {
      ap[i] = UINT64_MAX;
      bp[i] = 0;
    }
    break;
  case 2:
    for (uint64_t i = 0; i < n; i++) {
      ap[i] = UINT64_MAX;
      bp[i] = UINT64_MAX;
    }
    break;
  case 3:
    for (uint64_t i = 0; i < n; i++) {
      ap[i] = (i & 1) ? UINT64_MAX : 0;
      bp[i] = (i & 1) ? 0 : UINT64_MAX;
    }
    break;
  default:
    for (uint64_t i = 0; i < n; i++) {
      ap[i] = splitmix64(seed);
      bp[i] = splitmix64(seed);
    }
    if (n) {
      ap[0] = UINT64_MAX - (trial & 3);
      bp[0] = 8 + (trial & 7);
    }
    if (n > 17) {
      ap[17] = UINT64_MAX;
      bp[17] = UINT64_MAX;
    }
    if (n > 18) {
      ap[18] = UINT64_MAX;
      bp[18] = 0;
    }
    break;
  }
}

static int check_guard(const _limb *rp, uint64_t n) {
  for (uint64_t i = 0; i < GUARD_LIMBS; i++)
    if (rp[n + i] != GUARD_WORD)
      return 0;
  return 1;
}

int main(int argc, char **argv) {
  const uint64_t max_n = argc > 1 ? strtoull(argv[1], 0, 0) : 2048;
  const uint64_t trials = argc > 2 ? strtoull(argv[2], 0, 0) : 32;

  if (max_n == 0 || max_n > MAX_LIMBS) {
    fprintf(stderr, "max_n must be in 1..%u\n", MAX_LIMBS);
    return 1;
  }

  _limb *a = alloc_limbs("a");
  _limb *b = alloc_limbs("b");
  _limb *r = alloc_limbs("r");
  _limb *ref = alloc_limbs("ref");
  uint64_t seed = UINT64_C(0x243f6a8885a308d3);
  uint64_t hash = UINT64_C(0xbb67ae8584caa73b);

  for (uint64_t n = 1; n <= max_n; n++) {
    for (uint64_t trial = 0; trial < trials; trial++) {
      fill_case(a, b, n, trial, &seed);

      for (uint8_t cin = 0; cin <= 1; cin++) {
        for (uint64_t i = 0; i < n + GUARD_LIMBS; i++) {
          r[i] = GUARD_WORD;
          ref[i] = GUARD_WORD;
        }

        const uint8_t cref = scalar_add_n(ref, a, b, n, cin);
        const uint8_t cgot = simd_current_mpn_add_n(r, a, b, n, cin);

        if (cref != cgot || memcmp(r, ref, n * sizeof(*r)) != 0 ||
            !check_guard(r, n)) {
          fprintf(stderr,
                  "add correctness failed: n=%" PRIu64 " trial=%" PRIu64
                  " cin=%u cref=%u cgot=%u guard=%s\n",
                  n, trial, cin, cref, cgot, check_guard(r, n) ? "ok" : "bad");
          return 1;
        }

        for (uint64_t i = 0; i < n; i++)
          hash = (hash ^ r[i]) * UINT64_C(0x9e3779b97f4a7c15);
        hash ^= cgot + n + (trial << 8) + ((uint64_t)cin << 40);
      }
    }
  }

  printf("ok max_n=%" PRIu64 " trials=%" PRIu64 " hash=%016" PRIx64 "\n",
         max_n, trials, hash);

  free(a);
  free(b);
  free(r);
  free(ref);
  return 0;
}
