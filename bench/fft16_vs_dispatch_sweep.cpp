// Crossover sweep: fft16 (AVX-512 PQ FFT) vs the u52 toom dispatch
// (simd_mpn_mul), balanced an = bn = n, n in [128, 512] step 2.
// One process per timed implementation (argv[1] = "f16" | "u52"); every
// point verified against GMP mpn_mul. Emits CSV: n,ns_limb
// Build: clang++ -O3 -march=native -I../include fft16_vs_dispatch_sweep.cpp \
//          ../src/x86_64/mul_basecase_le6.S -I<gmp-build> \
//          <gmp-build>/.libs/libgmp.a -o fft16_vs_dispatch
#define SCRATCH_IMPLEMENTATION
#include "u64cvt.h"
extern "C" {
#include "fft16.h"
}
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

int main(int argc, char** argv){
    const bool time_f16 = !(argc > 1 && !strcmp(argv[1], "u52"));
    std::vector<uint64_t> A(512), B(512), R(1024 + 16), REF(1024 + 16);
    for(auto& x : A) x = xr();
    for(auto& x : B) x = xr();

    printf("n,ns_limb\n");
    for(size_t n = 128; n <= 512; n += 2){
        mpn_mul((mp_ptr)REF.data(), (mp_srcptr)A.data(), n, (mp_srcptr)B.data(), n);
        if(time_f16){
            if(!fft16_mul(R.data(), A.data(), (ptrdiff_t)n, B.data(), (ptrdiff_t)n)) continue;
        }else{
            simd_mpn_mul(R.data(), A.data(), n, B.data(), n);
        }
        if(memcmp(R.data(), REF.data(), 2 * n * 8)){
            fprintf(stderr, "MISMATCH n=%zu\n", n);
            return 1;
        }
        long reps = (long)(400000 / n) + 1;
        double tp[PASSES];
        for(int p = 0; p < PASSES; ++p){
            double t0 = now();
            if(time_f16)
                for(long q = 0; q < reps; ++q)
                    fft16_mul(R.data(), A.data(), (ptrdiff_t)n, B.data(), (ptrdiff_t)n);
            else
                for(long q = 0; q < reps; ++q)
                    simd_mpn_mul(R.data(), A.data(), n, B.data(), n);
            tp[p] = (now() - t0) / reps;
        }
        qsort(tp, PASSES, 8, cmpd);
        printf("%zu,%.4f\n", n, tp[PASSES / 2] / n * 1e9);
        fflush(stdout);
    }
    return 0;
}
