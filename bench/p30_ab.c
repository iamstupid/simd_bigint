// in-process A/B: ladder (mixed 3*2^k) vs binary-only, same layout
#define _POSIX_C_SOURCE 199309L
#define SCRATCH_IMPLEMENTATION
#include "p30.h"
#include <stdio.h>
#include <time.h>
static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9*t.tv_nsec; }
static uint64_t rngs = 42;
static uint64_t xr(void){ rngs ^= rngs<<13; rngs ^= rngs>>7; rngs ^= rngs<<17; return rngs; }
int main(void){
    enum { MAXN = 1<<21 };
    static uint64_t A[MAXN], B[MAXN], R[2*MAXN];
    for(size_t i = 0; i < MAXN; ++i){ A[i] = xr(); B[i] = xr(); }
    printf("n,binary,ladder,ratio\n");
    /* mixed-eligible points: tv in (2^{lg-1}, 3*2^{lg-2}]: pick n per
     * 1/8 octave inside the mixed window of each octave */
    for(int lgn = 12; lgn <= 21; ++lgn){
        for(int f = 0; f < 4; ++f){
            /* n spanning the mixed window: tv = 2n/16 in (2^{lg-1}, 3*2^{lg-2}] */
            double lo = 1.02, hi = 1.46;
            double m = lo + (hi - lo) * f / 3.0;
            size_t n = (size_t)((double)((size_t)1 << (lgn - 1)) * m);
            if(n > MAXN) continue;
            double tb = 1e30, tl = 1e30;
            int reps = n > (1<<18) ? 3 : (n > (1<<15) ? 9 : 33);
            for(int pass = 0; pass < 5; ++pass){
                p30_ladder = 0;
                double t0 = now();
                for(int r = 0; r < reps; ++r) p30_mul(R, A, (ptrdiff_t)n, B, (ptrdiff_t)n);
                double t = (now() - t0) / reps / (double)n * 1e9;
                if(t < tb) tb = t;
                p30_ladder = 1;
                t0 = now();
                for(int r = 0; r < reps; ++r) p30_mul(R, A, (ptrdiff_t)n, B, (ptrdiff_t)n);
                t = (now() - t0) / reps / (double)n * 1e9;
                if(t < tl) tl = t;
            }
            printf("%zu,%.3f,%.3f,%.3f\n", n, tb, tl, tb/tl);
            fflush(stdout);
        }
    }
    return 0;
}
