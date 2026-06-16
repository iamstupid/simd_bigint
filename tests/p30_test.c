// p30.h validation: GMP differential across the truncation grid,
// boundary totals, adversarial patterns at the caps, random shapes.
// Build: clang -O3 -march=native -std=c11 -I../include p30_test.c \
//          -I/tmp/gmp-build /tmp/gmp-build/.libs/libgmp.a -lm -o p30_test
#define SCRATCH_IMPLEMENTATION
#include "p30.h"
#include <gmp.h>
#include <stdio.h>

#define MAXL (1u << 21)
static uint64_t A[MAXL], B[MAXL];
static uint64_t R[2 * MAXL + 16], REF[2 * MAXL + 16];

static uint64_t rng = 42;
static uint64_t xr(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int fails, runs;

static void fill(uint64_t *p, size_t n, int pat){
    for(size_t i = 0; i < n; ++i)
        p[i] = pat == 1 ? ~(uint64_t)0
             : pat == 2 ? ((i & 31) == 0 ? ~(uint64_t)0 : 0) : xr();
}
static void check(size_t an, size_t bn, int pat){
    fill(A, an, pat);
    fill(B, bn, pat);
    if(an >= bn) mpn_mul((mp_ptr)REF, (mp_srcptr)A, an, (mp_srcptr)B, bn);
    else         mpn_mul((mp_ptr)REF, (mp_srcptr)B, bn, (mp_srcptr)A, an);
    ++runs;
    if(!p30_mul(R, A, (ptrdiff_t)an, B, (ptrdiff_t)bn)){
        printf("FAIL an=%zu bn=%zu pat=%d: returned 0\n", an, bn, pat);
        ++fails;
        return;
    }
    if(memcmp(R, REF, (an + bn) * 8)){
        size_t i = 0;
        while(R[i] == REF[i]) ++i;
        printf("FAIL an=%zu bn=%zu pat=%d limb %zu got %016llx want %016llx\n",
               an, bn, pat, i, (unsigned long long)R[i],
               (unsigned long long)REF[i]);
        ++fails;
    }
}

static void checksq(size_t an, int pat){
    fill(A, an, pat);
    mpn_sqr((mp_ptr)REF, (mp_srcptr)A, an);
    ++runs;
    if(!p30_sqr(R, A, (ptrdiff_t)an)){
        printf("SQR FAIL an=%zu pat=%d: returned 0\n", an, pat);
        ++fails;
        return;
    }
    if(memcmp(R, REF, 2 * an * 8)){
        size_t i = 0;
        while(R[i] == REF[i]) ++i;
        printf("SQR FAIL an=%zu pat=%d limb %zu got %016llx want %016llx\n",
               an, pat, i, (unsigned long long)R[i],
               (unsigned long long)REF[i]);
        ++fails;
    }
}

int main(void){
    /* gate 1: exhaustive small (covers M = 1..4, every tv mod 16) */
    for(size_t an = 1; an <= 40; ++an)
        for(size_t bn = 1; bn <= 40; ++bn)
            check(an, bn, 0);
    printf("gate 1 (exhaustive 1..40^2): %d cases, %d fails\n", runs, fails);

    /* gate 2: totals straddling every pow2-vector boundary */
    int g1 = runs;
    for(int lg = 1; lg <= 14; ++lg){
        size_t edge = ((size_t)1 << lg) * P30_W;
        for(int d = -3; d <= 3; ++d){
            size_t tot = edge + (size_t)d;
            size_t an = tot / 2, bn = tot - an;
            check(an, bn, 0);
            if(tot / 3 >= 1 && bn > tot / 3)
                check(an + tot / 3, bn - tot / 3, 0);
        }
    }
    printf("gate 2 (boundary totals lg 1..14): %d cases, %d fails\n",
           runs - g1, fails);

    /* gate 3: adversarial + random + the caps */
    int g2 = runs;
    check(4096, 4096, 1);
    check(4096, 4096, 2);
    check(65536, 65536, 1);
    check(100000, 17, 1);
    for(int t = 0; t < 40; ++t){
        size_t an = 1 + xr() % 50000, bn = 1 + xr() % 50000;
        check(an, bn, (int)(xr() % 3));
    }
    check(1u << 20, 1u << 20, 0);                /* 2^21 total */
    check((1u << 21) - 3, (1u << 21) - 1, 0);    /* near cap, odd */
    check(1u << 21, 1u << 21, 1);                /* CAP, all-max: CRT
                                                    headroom worst case */
    printf("gate 3 (adversarial/random/caps): %d cases, %d fails\n",
           runs - g2, fails);

    /* gate 4: squaring (binary, ladder and f3 paths share the gates
     * the build overrides) */
    int g3 = runs;
    for(size_t an = 1; an <= 80; ++an) checksq(an, 0);
    for(int lg = 1; lg <= 14; ++lg){
        size_t edge = ((size_t)1 << lg) * P30_W;
        for(int d = -3; d <= 3; ++d){
            checksq((edge + (size_t)d) / 2, 0);
            checksq((edge + (size_t)d) / 3, 0);
        }
    }
    checksq(4096, 1);
    checksq(4096, 2);
    checksq(65536, 1);
    for(int t = 0; t < 40; ++t)
        checksq(1 + xr() % 50000, (int)(xr() % 3));
    checksq(1u << 20, 0);
    checksq(1u << 21, 1);                        /* cap, all-max */
    printf("gate 4 (squaring): %d cases, %d fails\n", runs - g3, fails);

    /* band rejections */
    if(p30_mul(R, A, (ptrdiff_t)(P30_MIN_OPERAND_CAP + 1),
               B, (ptrdiff_t)(P30_MIN_OPERAND_CAP + 1)))
        { printf("FAIL: accepted beyond CRT cap\n"); ++fails; }

    if(fails) printf("FAILED: %d of %d\n", fails, runs);
    else      printf("OK: all %d cases\n", runs);
    return fails != 0;
}
