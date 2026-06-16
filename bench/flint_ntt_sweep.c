// 1/16-octave sweep of balanced mul (an = bn = n) through FLINT's
// fft_small (small-primes truncated NTT, mpn_mul_default_mpn_ctx),
// n in [2^16, 2^22] u64 limbs. Every point is verified against GMP
// mpn_mul before timing. Emits CSV: n,ns_limb
// Build (flint static lib in /tmp/flint-build):
//   clang -O2 -march=native -std=c11 \
//     -I/tmp/flint-build/src -I../reference/flint/src -I/tmp/gmp-build \
//     flint_ntt_sweep.c /tmp/flint-build/libflint.a \
//     /tmp/gmp-build/.libs/libgmp.a -lmpfr -lpthread -lm -o flint_ntt_sweep
#define _POSIX_C_SOURCE 199309L
#include "flint.h"
#include "fft_small.h"
#include <gmp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rng = 42;
static uint64_t xr(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int cmpd(const void* x, const void* y){ double a = *(const double*)x, b = *(const double*)y; return (a > b) - (a < b); }
#define PASSES 7
#define MAXN ((size_t)1 << 22)

int main(void){
    uint64_t *A = malloc(MAXN * 8), *B = malloc(MAXN * 8);
    uint64_t *R = malloc(2 * MAXN * 8), *REF = malloc(2 * MAXN * 8);
    for(size_t i = 0; i < MAXN; ++i){ A[i] = xr(); B[i] = xr(); }

    printf("n,ns_limb\n");
    for(int k = 0; ; ++k){
        size_t n = (size_t)llround(65536.0 * pow(2.0, k / 16.0));
        if(n > MAXN) break;
        static size_t last = 0;
        if(n == last) continue;
        last = n;

        mpn_mul((mp_ptr)REF, (mp_srcptr)A, n, (mp_srcptr)B, n);
        mpn_mul_default_mpn_ctx((nn_ptr)R, (nn_srcptr)A, (slong)n, (nn_srcptr)B, (slong)n);
        if(memcmp(R, REF, 2 * n * 8)){
            fprintf(stderr, "MISMATCH n=%zu\n", n);
            return 1;
        }
        long reps = (long)(2000000 / n) + 1;
        double tp[PASSES];
        for(int p = 0; p < PASSES; ++p){
            double t0 = now();
            for(long q = 0; q < reps; ++q)
                mpn_mul_default_mpn_ctx((nn_ptr)R, (nn_srcptr)A, (slong)n, (nn_srcptr)B, (slong)n);
            tp[p] = (now() - t0) / reps;
        }
        qsort(tp, PASSES, 8, cmpd);
        printf("%zu,%.4f\n", n, tp[PASSES / 2] / n * 1e9);
        fflush(stdout);
    }
    return 0;
}
