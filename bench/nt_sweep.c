// Combined-dispatch 1/16-octave sweep (balanced an = bn = n):
// nt_mul (p50/p30 dispatch), bare p30_mul, bare p50_mul, nt_sqr --
// every nt point GMP-verified, solo process.
// Build: clang -O3 -march=native -std=c11 -I../include nt_sweep.c \
//          -I/tmp/gmp-build /tmp/gmp-build/.libs/libgmp.a -lm -o nt_sweep
#define _POSIX_C_SOURCE 199309L
#define SCRATCH_IMPLEMENTATION
#include "ntmul.h"
#include <gmp.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rngs = 42;
static uint64_t xr(void){ rngs ^= rngs << 13; rngs ^= rngs >> 7; rngs ^= rngs << 17; return rngs; }

#define MAXN (1u << 21)
static uint64_t A[MAXN], B[MAXN], R[2*MAXN], G[2*MAXN];

static double bench(int which, size_t n){
    int reps = n > (1u<<19) ? 3 : n > (1u<<15) ? 7 : 31;
    double best = 1e30;
    for(int pass = 0; pass < 5; ++pass){
        double t0 = now();
        for(int r = 0; r < reps; ++r)
            switch(which){
            case 0: nt_mul(R, A, (ptrdiff_t)n, B, (ptrdiff_t)n); break;
            case 1: p30_mul(R, A, (ptrdiff_t)n, B, (ptrdiff_t)n); break;
            case 2: p50_mul(R, A, (ptrdiff_t)n, B, (ptrdiff_t)n); break;
            case 3: nt_sqr(R, A, (ptrdiff_t)n); break;
            }
        double t = (now() - t0) / reps / (double)n * 1e9;
        if(t < best) best = t;
    }
    return best;
}

int main(void){
    printf("n,nt,p30,p50,ntsqr\n");
    for(int oct = 11; oct < 21; ++oct)
        for(int s = 0; s < 16; ++s){
            double f = (double)(1u << oct) * pow(2.0, s / 16.0);
            size_t n = (size_t)f;
            if(n > MAXN) break;
            for(size_t i = 0; i < n; ++i){ A[i] = xr(); B[i] = xr(); }
            if(!nt_mul(R, A, (ptrdiff_t)n, B, (ptrdiff_t)n)){
                fprintf(stderr, "band reject n=%zu\n", n); return 1;
            }
            mpn_mul((mp_limb_t *)G, (const mp_limb_t *)A, (mp_size_t)n,
                    (const mp_limb_t *)B, (mp_size_t)n);
            for(size_t i = 0; i < 2*n; ++i)
                if(R[i] != G[i]){ fprintf(stderr, "MUL VERIFY FAIL n=%zu\n", n); return 1; }
            if(!nt_sqr(R, A, (ptrdiff_t)n)){
                fprintf(stderr, "sqr band reject n=%zu\n", n); return 1;
            }
            mpn_sqr((mp_limb_t *)G, (const mp_limb_t *)A, (mp_size_t)n);
            for(size_t i = 0; i < 2*n; ++i)
                if(R[i] != G[i]){ fprintf(stderr, "SQR VERIFY FAIL n=%zu\n", n); return 1; }
            printf("%zu,%.4f,%.4f,%.4f,%.4f\n", n,
                   bench(0, n), bench(1, n), bench(2, n), bench(3, n));
            fflush(stdout);
        }
    return 0;
}
