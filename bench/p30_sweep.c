// 1/16-octave sweep of balanced mul (an = bn = n) through the p30
// truncated-NTT engine, n in [2048, 2^21] u64 limbs (total at the band
// cap 2^22). Every point verified against GMP before timing.
// Emits CSV: n,ns_limb
// Build: clang -O3 -march=native -std=c11 -I../include p30_sweep.c \
//          -I/tmp/gmp-build /tmp/gmp-build/.libs/libgmp.a -lm -o p30_sweep
#define _POSIX_C_SOURCE 199309L
#define SCRATCH_IMPLEMENTATION
#include "p30.h"
#include <gmp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rng = 42;
static uint64_t xr(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int cmpd(const void *x, const void *y){ double a = *(const double *)x, b = *(const double *)y; return (a > b) - (a < b); }
#define PASSES 7
#define MAXN ((size_t)1 << 21)

int main(void){
    uint64_t *A = malloc(MAXN * 8), *B = malloc(MAXN * 8);
    uint64_t *R = malloc(2 * MAXN * 8 + 128), *REF = malloc(2 * MAXN * 8 + 128);
    for(size_t i = 0; i < MAXN; ++i){ A[i] = xr(); B[i] = xr(); }

    printf("n,ns_limb\n");
    size_t last = 0;
    for(int k = 0; ; ++k){
        size_t n = (size_t)llround(2048.0 * pow(2.0, k / 16.0));
        if(n > MAXN) break;
        if(n == last) continue;
        last = n;

        mpn_mul((mp_ptr)REF, (mp_srcptr)A, n, (mp_srcptr)B, n);
        if(!p30_mul(R, A, (ptrdiff_t)n, B, (ptrdiff_t)n)){
            fprintf(stderr, "OOB n=%zu\n", n);
            return 1;
        }
        if(memcmp(R, REF, 2 * n * 8)){
            fprintf(stderr, "MISMATCH n=%zu\n", n);
            return 1;
        }
        long reps = (long)(400000 / n) + 1;
        double tp[PASSES];
        for(int p = 0; p < PASSES; ++p){
            double t0 = now();
            for(long q = 0; q < reps; ++q)
                p30_mul(R, A, (ptrdiff_t)n, B, (ptrdiff_t)n);
            tp[p] = (now() - t0) / reps;
        }
        qsort(tp, PASSES, 8, cmpd);
        printf("%zu,%.4f\n", n, tp[PASSES / 2] / n * 1e9);
        fflush(stdout);
    }
    return 0;
}
