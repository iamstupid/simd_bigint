// Sweep simd_mpn_mul against this repo's GMP (built with clang, including its
// FFT delegate) over 9 relative size classes. All sizes are u64 limbs:
//   bn = {1/8, 1/5, 1/4, 1/3, 0.4, 0.5, 0.65, 0.8, 1} * an
//   an in [8,2048]: step 1 (an <= 256), 2 (256 < an <= 512), 4 (an > 512)
// Emits CSV: an,bn,ratio,gmp_ns,simd_ns,speedup   (median ns per call)
// Build: clang++ -O3 -march=native -I../include gmp_vs_simd_sweep.cpp \
//          ../src/x86_64/mul_basecase_le6.S -I<gmp-build> <gmp-build>/.libs/libgmp.a \
//          -o gmp_vs_simd_sweep
#define SCRATCH_IMPLEMENTATION
#include "u64cvt.h"
#include <gmp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rng = 42;
static uint64_t xr(){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int cmpd(const void* x, const void* y){ double a = *(const double*)x, b = *(const double*)y; return (a > b) - (a < b); }
#define PASSES 7

int main(){
    const double ratios[] = {1.0/8, 1.0/5, 1.0/4, 1.0/3, 0.4, 0.5, 0.65, 0.8, 1.0};
    std::vector<uint64_t> A(2048 + 8), B(2048 + 8), R1(4096 + 16), R2(4096 + 16);
    for(auto &x : A) x = xr();
    for(auto &x : B) x = xr();
    printf("an,bn,ratio,gmp_ns,simd_ns,speedup\n");
    for(size_t an = 8; an <= 2048; an += (an < 256 ? 1 : (an < 512 ? 2 : 4))){
        for(int ri = 0; ri < 9; ri++){
            size_t bn = (size_t)(an * ratios[ri] + 0.5);
            if(bn < 1) bn = 1;
            if(bn > an) bn = an;
            long reps = (long)(400000 / (an + bn)) + 1;
            double tg[PASSES], ts[PASSES];
            for(int p = 0; p < PASSES; p++){
                double t0 = now();
                for(long q = 0; q < reps; q++)
                    mpn_mul((mp_ptr)R1.data(), (mp_srcptr)A.data(), an, (mp_srcptr)B.data(), bn);
                tg[p] = (now() - t0) / reps;
                t0 = now();
                for(long q = 0; q < reps; q++)
                    simd_mpn_mul(R2.data(), A.data(), an, B.data(), bn);
                ts[p] = (now() - t0) / reps;
            }
            if(memcmp(R1.data(), R2.data(), (an + bn) * 8) != 0){
                fprintf(stderr, "MISMATCH an=%zu bn=%zu\n", an, bn);
                return 1;
            }
            qsort(tg, PASSES, 8, cmpd); qsort(ts, PASSES, 8, cmpd);
            double g = tg[PASSES / 2] * 1e9, s = ts[PASSES / 2] * 1e9;
            printf("%zu,%zu,%.4f,%.1f,%.1f,%.4f\n", an, bn, ratios[ri], g, s, g / s);
        }
        if((an & 63) == 0) fflush(stdout);
    }
    return 0;
}
