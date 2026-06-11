// 1/16-octave sweep of balanced mul (an = bn = n), n in [128, 16384] u64
// limbs: fft16 (AVX-512) vs the reference int_fft (AVX2). One process per
// timed implementation (argv[1] = "f16" | "ref") -- interleaving different
// implementations in one process cross-contaminates small-size timings.
// Every point is verified against GMP mpn_mul before timing.
// Emits CSV: n,ns_limb
// Build: clang++ -O3 -march=native -I../include fft16_sweep.cpp \
//          -I<gmp-build> <gmp-build>/.libs/libgmp.a -o fft16_sweep
extern "C" {
#include "fft16.h"
}
#define INT_FFT_IMPLEMENTATION
#include "../reference/int_fft.hpp"
#include <gmp.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rng = 42;
static uint64_t xr(){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int cmpd(const void* x, const void* y){ double a = *(const double*)x, b = *(const double*)y; return (a > b) - (a < b); }
#define PASSES 7

int main(int argc, char** argv){
    const bool time_f16 = !(argc > 1 && !strcmp(argv[1], "ref"));
    std::vector<size_t> sizes;
    for(int k = 0; ; ++k){
        size_t n = (size_t)llround(128.0 * std::pow(2.0, k / 16.0));
        if(n > 16384) break;
        if(sizes.empty() || n != sizes.back()) sizes.push_back(n);
    }
    std::vector<uint64_t> A(16384), B(16384), R(32768 + 16), REF(32768 + 16);
    for(auto& x : A) x = xr();
    for(auto& x : B) x = xr();

    printf("n,ns_limb\n");
    for(size_t n : sizes){
        mpn_mul((mp_ptr)REF.data(), (mp_srcptr)A.data(), n, (mp_srcptr)B.data(), n);
        if(time_f16){
            if(!fft16_mul(R.data(), A.data(), (ptrdiff_t)n, B.data(), (ptrdiff_t)n)) continue;
        }else{
            if(!fft::mul(R.data(), A.data(), (ptrdiff_t)n, B.data(), (ptrdiff_t)n)) continue;
        }
        if(memcmp(R.data(), REF.data(), 2 * n * 8)){
            fprintf(stderr, "MISMATCH n=%zu\n", n);
            return 1;
        }
        long reps = (long)(300000 / n) + 1;
        double tp[PASSES];
        for(int p = 0; p < PASSES; ++p){
            double t0 = now();
            if(time_f16)
                for(long q = 0; q < reps; ++q)
                    fft16_mul(R.data(), A.data(), (ptrdiff_t)n, B.data(), (ptrdiff_t)n);
            else
                for(long q = 0; q < reps; ++q)
                    fft::mul(R.data(), A.data(), (ptrdiff_t)n, B.data(), (ptrdiff_t)n);
            tp[p] = (now() - t0) / reps;
        }
        qsort(tp, PASSES, 8, cmpd);
        printf("%zu,%.4f\n", n, tp[PASSES / 2] / n * 1e9);
        fflush(stdout);
    }
    return 0;
}
