// First performance read of the AVX-512 fft16 port vs the AVX2 reference
// int_fft (U16 band) and GMP, balanced shapes. Median ns per input limb.
// Build: clang++ -O3 -march=native -I../include fft16_vs_ref_bench.cpp \
//          -I<gmp-build> <gmp-build>/.libs/libgmp.a -o fft16_vs_ref
extern "C" {
#include "fft16.h"
}
#define INT_FFT_IMPLEMENTATION
#include "../reference/int_fft.hpp"
#include <gmp.h>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rng = 42;
static uint64_t xr(){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int cmpd(const void* x, const void* y){ double a = *(const double*)x, b = *(const double*)y; return (a > b) - (a < b); }
#define PASSES 9

int main(){
    printf("n,ref_avx2_ns_limb,fft16_ns_limb,gmp_ns_limb,avx2/fft16\n");
    std::vector<uint64_t> A(16384), B(16384), R1(32768), R2(32768), R3(32768);
    for(auto& x : A) x = xr();
    for(auto& x : B) x = xr();
    for(size_t n : {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072,
                    4096, 6144, 8192, 12288, 16384}){
        if(!fft16_mul(R2.data(), A.data(), n, B.data(), n)) continue;
        fft::mul(R1.data(), A.data(), n, B.data(), n);
        mpn_mul((mp_ptr)R3.data(), (mp_srcptr)A.data(), n, (mp_srcptr)B.data(), n);
        if(memcmp(R1.data(), R2.data(), 2 * n * 8) || memcmp(R1.data(), R3.data(), 2 * n * 8)){
            fprintf(stderr, "MISMATCH n=%zu\n", n);
            return 1;
        }
        long reps = (long)(200000 / n) + 1;
        double tr[PASSES], tf[PASSES], tg[PASSES];
        for(int p = 0; p < PASSES; ++p){
            double t0 = now();
            for(long q = 0; q < reps; ++q) fft::mul(R1.data(), A.data(), n, B.data(), n);
            tr[p] = (now() - t0) / reps;
            t0 = now();
            for(long q = 0; q < reps; ++q) fft16_mul(R2.data(), A.data(), n, B.data(), n);
            tf[p] = (now() - t0) / reps;
            t0 = now();
            for(long q = 0; q < reps; ++q) mpn_mul((mp_ptr)R3.data(), (mp_srcptr)A.data(), n, (mp_srcptr)B.data(), n);
            tg[p] = (now() - t0) / reps;
        }
        qsort(tr, PASSES, 8, cmpd); qsort(tf, PASSES, 8, cmpd); qsort(tg, PASSES, 8, cmpd);
        double a = tr[PASSES/2] / n * 1e9, b = tf[PASSES/2] / n * 1e9, g = tg[PASSES/2] / n * 1e9;
        printf("%zu,%.3f,%.3f,%.3f,%.3f\n", n, a, b, g, a / b);
        fflush(stdout);
    }
    return 0;
}
