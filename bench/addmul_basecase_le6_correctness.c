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

int
main(void)
{
  uint64_t a[6], b[6], got[16], ref[16];
  uint64_t seed = 0x0ddc0ffeebadc0deull;

  for (size_t an = 1; an <= 6; ++an) {
    for (size_t bn = 1; bn <= 6; ++bn) {
      for (size_t trial = 0; trial < 50000; ++trial) {
        for (size_t i = 0; i < 16; ++i) {
          uint64_t z = rng64(&seed);
          if ((trial & 127) == 0)
            z = UINT64_MAX;
          else if ((trial & 255) == 1)
            z = i & 1 ? UINT64_MAX : 0;
          got[i] = ref[i] = z;
        }

        for (size_t i = 0; i < 6; ++i) {
          uint64_t x = rng64(&seed);
          uint64_t y = rng64(&seed);
          if ((trial & 7) == 0) x = UINT64_MAX;
          if ((trial & 15) == 0) y = UINT64_MAX;
          if ((trial & 31) == 0) x = (i & 1) ? 0 : UINT64_MAX;
          if ((trial & 63) == 0) y = (i & 1) ? UINT64_MAX : 0;
          a[i] = x;
          b[i] = y;
        }

        uint64_t canary = got[an + bn];
        uint64_t cref = ref_addmul(ref, a, an, b, bn);
        uint64_t cgot = simd_mpn_addmul_basecase_le6(got, a, an, b, bn);

        if (cgot != cref ||
            memcmp(got, ref, (an + bn) * sizeof(got[0])) != 0) {
          fprintf(stderr,
                  "mismatch an=%zu bn=%zu trial=%zu cgot=%llu cref=%llu\n",
                  an, bn, trial, (unsigned long long)cgot,
                  (unsigned long long)cref);
          for (size_t i = 0; i < an + bn; ++i)
            fprintf(stderr, "  [%zu] got=%016llx ref=%016llx\n", i,
                    (unsigned long long)got[i], (unsigned long long)ref[i]);
          return 1;
        }
        if (got[an + bn] != canary) {
          fprintf(stderr, "overstore an=%zu bn=%zu trial=%zu\n", an, bn, trial);
          return 1;
        }
      }
    }
  }

  puts("ok addmul_basecase_le6");
  return 0;
}
