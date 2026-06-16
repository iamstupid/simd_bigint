// p50 GMP differential battery (mirrors tests/p30_test.c gates):
//   1. exhaustive small shapes 1..40 x 1..40
//   2. trunc-boundary totals (powers of 2 +-3 limbs)
//   3. adversarial all-0xFF (incl. large), random large, unbalanced
// Build: clang -O3 -march=native -std=c11 -I../include p50_test.c \
//          -I/tmp/gmp-build /tmp/gmp-build/.libs/libgmp.a -lm -o p50_test
#define SCRATCH_IMPLEMENTATION
#include "p50.h"
#include <gmp.h>
#include <stdio.h>

static uint64_t rngs = 42;
static uint64_t xr(void){ rngs ^= rngs << 13; rngs ^= rngs >> 7; rngs ^= rngs << 17; return rngs; }

#define MAXL (1u << 21)
static uint64_t A[MAXL], B[MAXL], R[2*MAXL], G[2*MAXL];

static int one(size_t an, size_t bn, int pat){
    for(size_t i = 0; i < an; ++i)
        A[i] = pat == 2 ? ~0ull : pat == 1 ? (i * 0x9E3779B97F4A7C15ull) : xr();
    for(size_t i = 0; i < bn; ++i)
        B[i] = pat == 2 ? ~0ull : pat == 1 ? (~i * 0xC2B2AE3D27D4EB4Full) : xr();
    int ok = p50_mul(R, A, (ptrdiff_t)an, B, (ptrdiff_t)bn);
    if(!ok){ printf("FAIL an=%zu bn=%zu pat=%d: returned 0\n", an, bn, pat); return 1; }
    if(an >= bn)
        mpn_mul((mp_limb_t *)G, (const mp_limb_t *)A, (mp_size_t)an,
                (const mp_limb_t *)B, (mp_size_t)bn);
    else
        mpn_mul((mp_limb_t *)G, (const mp_limb_t *)B, (mp_size_t)bn,
                (const mp_limb_t *)A, (mp_size_t)an);
    for(size_t i = 0; i < an + bn; ++i)
        if(R[i] != G[i]){
            printf("FAIL an=%zu bn=%zu pat=%d limb %zu got %016llx want %016llx\n",
                   an, bn, pat, i, (unsigned long long)R[i],
                   (unsigned long long)G[i]);
            return 1;
        }
    return 0;
}

static int onesq(size_t an, int pat){
    for(size_t i = 0; i < an; ++i)
        A[i] = pat == 2 ? ~0ull : pat == 1 ? (i * 0x9E3779B97F4A7C15ull) : xr();
    int ok = p50_sqr(R, A, (ptrdiff_t)an);
    if(!ok){ printf("SQR FAIL an=%zu pat=%d: returned 0\n", an, pat); return 1; }
    mpn_sqr((mp_limb_t *)G, (const mp_limb_t *)A, (mp_size_t)an);
    for(size_t i = 0; i < 2 * an; ++i)
        if(R[i] != G[i]){
            printf("SQR FAIL an=%zu pat=%d limb %zu got %016llx want %016llx\n",
                   an, pat, i, (unsigned long long)R[i],
                   (unsigned long long)G[i]);
            return 1;
        }
    return 0;
}

int main(void){
    int fails = 0;
    long cases = 0;
    for(size_t an = 1; an <= 40; ++an)
        for(size_t bn = 1; bn <= 40; ++bn){
            fails += one(an, bn, 0);
            ++cases;
        }
    printf("gate 1 (exhaustive 1..40^2): %ld cases, %d fails\n", cases, fails);
    int f2 = 0; cases = 0;
    for(int lgt = 1; lgt <= 14; ++lgt)
        for(int d = -3; d <= 3; ++d){
            long tot = (1l << lgt) + d;
            if(tot < 2) continue;
            size_t an = (size_t)(tot + 1) / 2, bn = (size_t)tot - an;
            if(bn < 1) continue;
            f2 += one(an, bn, 0);
            f2 += one((size_t)(tot * 5) / 6, (size_t)tot - (size_t)(tot * 5) / 6, 0);
            cases += 2;
        }
    printf("gate 2 (boundary totals lg 1..14): %ld cases, %d fails\n", cases, f2);
    fails += f2;
    int f3 = 0; cases = 0;
    /* adversarial all-max incl. big */
    f3 += one(1000, 1000, 2);      ++cases;
    f3 += one(65536, 65536, 2);    ++cases;
    f3 += one(1u << 20, 1u << 20, 2); ++cases;
    f3 += one((1u << 20) + 7, (1u << 19) - 3, 2); ++cases;
    /* random shapes */
    for(int t = 0; t < 24; ++t){
        size_t an = 1 + xr() % 50000, bn = 1 + xr() % 50000;
        f3 += one(an, bn, (int)(xr() % 3));
        ++cases;
    }
    printf("gate 3 (adversarial/random): %ld cases, %d fails\n", cases, f3);
    fails += f3;
    int f4 = 0; cases = 0;
    for(size_t an = 1; an <= 80; ++an){ f4 += onesq(an, 0); ++cases; }
    for(int lgt = 1; lgt <= 14; ++lgt)
        for(int d = -3; d <= 3; ++d){
            long an = (1l << lgt) + d;
            if(an < 1) continue;
            f4 += onesq((size_t)an, 0); ++cases;
        }
    f4 += onesq(1000, 2);          ++cases;
    f4 += onesq(65536, 2);         ++cases;
    f4 += onesq(1u << 20, 2);      ++cases;
    f4 += onesq((1u << 20) + 7, 2); ++cases;
    for(int t = 0; t < 24; ++t){
        f4 += onesq(1 + xr() % 50000, (int)(xr() % 3)); ++cases;
    }
    printf("gate 4 (squaring): %ld cases, %d fails\n", cases, f4);
    fails += f4;
    if(fails){ printf("FAILED: %d\n", fails); return 1; }
    printf("OK: all cases\n");
    return 0;
}
