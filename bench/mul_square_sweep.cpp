// Square-shape sweep an == bn in [8,500] step 1: IFMA basecase vs toom-22
// (mul_u52_karatsuba) vs toom-33, all called directly at top level (recursion
// inside goes through mul_u52_dispatch as usual).
// Emits CSV: n,basecase_ns,toom22_ns,toom33_ns  (median ns per call)
// Build: clang++ -O3 -march=native -I../include mul_square_sweep.cpp -o mul_square_sweep
// Run:   ./mul_square_sweep > mul_square_sweep_8_500.csv
#include "mul.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#define M52 ((1ull << 52) - 1)
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rng = 42;
static uint64_t xr(){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int cmpd(const void* x, const void* y){ double a = *(const double*)x, b = *(const double*)y; return (a > b) - (a < b); }
#define PASSES 11

int main(){
    scratch sc = scratch_create_ex(4096, 1u << 26, 1u << 26);
    printf("n,basecase_ns,toom22_ns,toom33_ns\n");
    for(size_t n = 8; n <= 500; ++n){
        std::vector<uint64_t> A((n + 7) / 8 * 8, 0), B((n + 7) / 8 * 8, 0);
        std::vector<uint64_t> R(((2 * n + 7) / 8 * 8) + 8, 0);
        for(size_t i = 0; i < n; i++){ A[i] = xr() & M52; B[i] = xr() & M52; }
        long reps = (long)(10000000 / (n * n)) * 4 + 4;
        double tb[PASSES], tk[PASSES], t3[PASSES];
        pvec r = (pvec)R.data();
        cpvec a = (cpvec)A.data(), b = (cpvec)B.data();
        for(int p = 0; p < PASSES; p++){
            double t0;
            t0 = now();
            for(long q = 0; q < reps; q++) mul_u52_basecase(r, a, b, n, n);
            tb[p] = (now() - t0) / reps;
            t0 = now();
            for(long q = 0; q < reps; q++) mul_u52_karatsuba(r, a, b, n, n, &sc);
            tk[p] = (now() - t0) / reps;
            t0 = now();
            for(long q = 0; q < reps; q++) mul_u52_toom33(r, a, b, n, n, &sc);
            t3[p] = (now() - t0) / reps;
        }
        qsort(tb, PASSES, 8, cmpd); qsort(tk, PASSES, 8, cmpd); qsort(t3, PASSES, 8, cmpd);
        printf("%zu,%.2f,%.2f,%.2f\n", n,
               tb[PASSES / 2] * 1e9, tk[PASSES / 2] * 1e9, t3[PASSES / 2] * 1e9);
        fflush(stdout);
    }
    scratch_destroy(&sc);
    return 0;
}
