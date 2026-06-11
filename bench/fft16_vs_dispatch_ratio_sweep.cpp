// Unbalanced crossover sweep: fft16 vs the u52 toom dispatch over the nine
// usual relative size classes bn = r*an, r in {1, .8, .65, .5, .4, 1/3,
// .25, .2, .125}, an in [128, 4096] step 4. One process per timed
// implementation (argv[1] = "f16" | "u52"); every point verified against
// GMP. Emits CSV: ratio,an,bn,ns_call
// Build: clang++ -O3 -march=native -I../include fft16_vs_dispatch_ratio_sweep.cpp \
//          ../src/x86_64/mul_basecase_le6.S -I<gmp-build> \
//          <gmp-build>/.libs/libgmp.a -o fft16_vs_dispatch_ratio
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
#define PASSES 7

int main(int argc, char** argv){
    const bool time_f16 = !(argc > 1 && !strcmp(argv[1], "u52"));
    const double ratios[] = {1.0, 0.8, 0.65, 0.5, 0.4, 1.0/3, 0.25, 0.2, 0.125};
    std::vector<uint64_t> A(4096), B(4096), R(8192 + 16), REF(8192 + 16);
    for(auto& x : A) x = xr();
    for(auto& x : B) x = xr();

    printf("ratio,an,bn,ns_call\n");
    for(int ri = 0; ri < 9; ++ri){
        for(size_t an = 128; an <= 4096; an += 4){
            size_t bn = (size_t)(an * ratios[ri] + 0.5);
            if(bn < 1) bn = 1;
            if(2 * (an + bn) > (1u << 16)) break;
            mpn_mul((mp_ptr)REF.data(), (mp_srcptr)A.data(), an, (mp_srcptr)B.data(), bn);
            if(time_f16){
                if(!fft16_mul(R.data(), A.data(), (ptrdiff_t)an, B.data(), (ptrdiff_t)bn)) continue;
            }else{
                simd_mpn_mul(R.data(), A.data(), an, B.data(), bn);
            }
            if(memcmp(R.data(), REF.data(), (an + bn) * 8)){
                fprintf(stderr, "MISMATCH r=%.3f an=%zu bn=%zu\n", ratios[ri], an, bn);
                return 1;
            }
            long reps = (long)(400000 / (an + bn)) + 1;
            double tp[PASSES];
            for(int p = 0; p < PASSES; ++p){
                double t0 = now();
                if(time_f16)
                    for(long q = 0; q < reps; ++q)
                        fft16_mul(R.data(), A.data(), (ptrdiff_t)an, B.data(), (ptrdiff_t)bn);
                else
                    for(long q = 0; q < reps; ++q)
                        simd_mpn_mul(R.data(), A.data(), an, B.data(), bn);
                tp[p] = (now() - t0) / reps;
            }
            qsort(tp, PASSES, 8, cmpd);
            printf("%.4f,%zu,%zu,%.1f\n", ratios[ri], an, bn, tp[PASSES / 2] * 1e9);
        }
        fflush(stdout);
    }
    return 0;
}
