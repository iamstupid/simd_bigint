// Same protocol as flint_ntt_sweep.c but through an np=4-restricted copy of
// fft_small/mpn_mul.c (4-prime NTTs only; the chooser then lands on bits=88
// for this whole range). Generate the restricted driver with:
//   sed 's/void mpn_ctx_mpn_mul(/void mpn_ctx_mpn_mul_np4(/;
//        s/if (np % P->nthreads != 0)/if (np != 4 || np % P->nthreads != 0)/' \
//     ../reference/flint/src/fft_small/mpn_mul.c > mpn_mul_np4.c
// and compile both TUs with gcc (matching the libflint.a build), renaming the
// clashing exported symbols:
//   gcc -O3 -march=native -std=c11 -Dcrt_data_init=t_cdi -Dcrt_data_clear=t_cdc
//     -Dmpn_ctx_init=t_mci -Dmpn_ctx_clear=t_mcc -Dmpn_ctx_fit_buffer=t_mcfb
//     -Dsd_fft_ctx_point_mul=t_pm -Dsd_fft_ctx_point_sqr=t_ps
//     -I<flint-build>/src -I../reference/flint/src -I<gmp-build>
//     mpn_mul_np4.c flint_ntt_np4_sweep.c <flint-build>/libflint.a
//     <gmp-build>/.libs/libgmp.a -lmpfr -lpthread -lm
// driver (mpn_ctx_mpn_mul_np4): 4-prime NTTs only, bits in {84, 88, 92}
#define _POSIX_C_SOURCE 199309L
#include "flint.h"
#include "fft_small.h"
#include <gmp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

void mpn_ctx_mpn_mul_np4(mpn_ctx_t R, ulong* z, const ulong* a, ulong an, const ulong* b, ulong bn);

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
    mpn_ctx_struct* ctx = get_default_mpn_ctx();

    printf("n,ns_limb\n");
    for(int k = 0; ; ++k){
        size_t n = (size_t)llround(65536.0 * pow(2.0, k / 16.0));
        if(n > MAXN) break;
        static size_t last = 0;
        if(n == last) continue;
        last = n;

        mpn_mul((mp_ptr)REF, (mp_srcptr)A, n, (mp_srcptr)B, n);
        mpn_ctx_mpn_mul_np4(ctx, R, A, n, B, n);
        if(memcmp(R, REF, 2 * n * 8)){
            fprintf(stderr, "MISMATCH n=%zu\n", n);
            return 1;
        }
        long reps = (long)(2000000 / n) + 1;
        double tp[PASSES];
        for(int p = 0; p < PASSES; ++p){
            double t0 = now();
            for(long q = 0; q < reps; ++q)
                mpn_ctx_mpn_mul_np4(ctx, R, A, n, B, n);
            tp[p] = (now() - t0) / reps;
        }
        qsort(tp, PASSES, 8, cmpd);
        printf("%zu,%.4f\n", n, tp[PASSES / 2] / n * 1e9);
        fflush(stdout);
    }
    return 0;
}
