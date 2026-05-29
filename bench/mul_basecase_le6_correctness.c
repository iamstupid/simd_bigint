#include "../include/mul_basecase_le6.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

int
main(void)
{
  uint64_t a[6], b[6], got[16], ref[16];
  uint64_t seed = 0x123456789abcdef0ull;

  for (size_t an = 1; an <= 6; ++an) {
    for (size_t bn = 1; bn <= 6; ++bn) {
      for (size_t trial = 0; trial < 20000; ++trial) {
        for (size_t i = 0; i < 6; ++i) {
          uint64_t x = rng64(&seed);
          uint64_t y = rng64(&seed);
          if ((trial & 7) == 0) x = UINT64_MAX;
          if ((trial & 15) == 0) y = UINT64_MAX;
          if ((trial & 31) == 0) x = (i & 1) ? 0 : UINT64_MAX;
          a[i] = x;
          b[i] = y;
        }

        for (size_t i = 0; i < 16; ++i)
          got[i] = ref[i] = 0xfeedfacecafebeefull;

        simd_mpn_mul_basecase_le6(got, a, an, b, bn);
        ref_mul(ref, a, an, b, bn);

        if (memcmp(got, ref, (an + bn) * sizeof(got[0])) != 0) {
          fprintf(stderr, "mismatch an=%zu bn=%zu trial=%zu\n", an, bn, trial);
          for (size_t i = 0; i < an + bn; ++i)
            fprintf(stderr, "  [%zu] got=%016llx ref=%016llx\n", i,
                    (unsigned long long)got[i], (unsigned long long)ref[i]);
          return 1;
        }
        if (got[an + bn] != 0xfeedfacecafebeefull) {
          fprintf(stderr, "overstore an=%zu bn=%zu trial=%zu\n", an, bn, trial);
          return 1;
        }
      }
    }
  }

  puts("ok mul_basecase_le6");
  return 0;
}
