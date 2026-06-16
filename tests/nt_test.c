// ntmul.h dispatch validation: GMP differential at every dispatch
// seam -- the p50/p30 total-length threshold, the p30 CRT cap (min
// operand) and transform cap (total), squares and rectangles.
// Build: clang -O2 -march=native -std=c11 -I../include nt_test.c \
//          -I/tmp/gmp-build /tmp/gmp-build/.libs/libgmp.a -lm -o nt_test
#define SCRATCH_IMPLEMENTATION
#include "ntmul.h"
#include <gmp.h>
#include <stdio.h>
#include <string.h>

#define MAXL (1u << 22)
static uint64_t A[MAXL], B[MAXL];
static uint64_t R[2 * MAXL + 16], REF[2 * MAXL + 16];

static uint64_t rng = 42;
static uint64_t xr(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int fails, runs;

static void fill(uint64_t *p, size_t n, int pat){
    for(size_t i = 0; i < n; ++i)
        p[i] = pat == 1 ? ~(uint64_t)0 : xr();
}
static void check(size_t an, size_t bn, int pat){
    fill(A, an, pat);
    fill(B, bn, pat);
    if(an >= bn) mpn_mul((mp_ptr)REF, (mp_srcptr)A, an, (mp_srcptr)B, bn);
    else         mpn_mul((mp_ptr)REF, (mp_srcptr)B, bn, (mp_srcptr)A, an);
    ++runs;
    if(!nt_mul(R, A, (ptrdiff_t)an, B, (ptrdiff_t)bn)){
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
    if(!nt_sqr(R, A, (ptrdiff_t)an)){
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
    /* small + random (p50 small band) */
    for(size_t an = 1; an <= 24; ++an)
        for(size_t bn = 1; bn <= 24; ++bn)
            check(an, bn, 0);
    for(int t = 0; t < 16; ++t){
        check(1 + xr() % 50000, 1 + xr() % 50000, (int)(xr() % 2));
        checksq(1 + xr() % 50000, (int)(xr() % 2));
    }
    /* dispatch seams: band edges + an M-step inside the fill band */
    static const size_t seams[3] = { NT_Z_LO, 360448u, NT_Z_HI };
    for(int e = 0; e < 3; ++e)
        for(int d = -2; d <= 2; ++d){
            size_t tot = seams[e] + (size_t)d;
            check(tot / 2, tot - tot / 2, 0);
            check(tot / 4, tot - tot / 4, 0);
            checksq(tot / 2, 0);
        }
    /* inside the p30 band, adversarial */
    check(1u << 19, 1u << 19, 1);
    checksq(1u << 19, 1);
    /* beyond the p30 CRT cap (min operand): both engines' worst case */
    check(P30_MIN_OPERAND_CAP + 1, P30_MIN_OPERAND_CAP + 1, 1);
    checksq(P30_MIN_OPERAND_CAP + 1, 1);
    /* beyond the p30 transform cap, min operand inside the CRT cap */
    check((1u << 22) - 5, 1u << 21, 0);
    printf("%d cases, %d fails\n", runs, fails);
    if(fails){ printf("FAILED\n"); return 1; }
    printf("OK: all cases\n");
    return 0;
}
