// Sweep mul_u52_toom32 vs mul_u52_dispatch (karatsuba/basecase) over the
// toom32 shape window: an in [60,400] step 4, bn in [an/2, 3an/4] step 2.
// Emits CSV: an,bn,ratio,dispatch_ns,toom32_ns,speedup
// Build: clang++ -O3 -march=native -I../include toom32_vs_dispatch_sweep.cpp -o toom32_sweep
// Run:   ./toom32_sweep > toom32_vs_dispatch_sweep_60_400.csv
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
#define PASSES 9

int main(){
    scratch sc = scratch_create_ex(4096, 1u << 26, 1u << 26);
    printf("an,bn,ratio,dispatch_ns,toom32_ns,speedup\n");
    for(size_t an = 60; an <= 400; an += 4){
        for(size_t bn = (an + 1) / 2; bn * 4 <= an * 3; bn += 2){
            uint64_t n = (2 * an >= 3 * bn) ? (an + 2) / 3 : ((bn + 1) >> 1);
            uint64_t s = an - 2 * n, t = bn - n;
            if(!(0 < s && s <= n && 0 < t && t <= n && s + t > n)) continue;
            std::vector<uint64_t> A((an + 7) / 8 * 8, 0), B((bn + 7) / 8 * 8, 0);
            std::vector<uint64_t> R(((an + bn + 7) / 8 * 8) + 8, 0);
            for(size_t i = 0; i < an; i++) A[i] = xr() & M52;
            for(size_t i = 0; i < bn; i++) B[i] = xr() & M52;
            long reps = (long)(2000000 / (an + bn)); if(reps < 64) reps = 64;
            double td[PASSES], tt[PASSES];
            for(int p = 0; p < PASSES; p++){
                double t0;
                t0 = now();
                for(long r = 0; r < reps; r++)
                    mul_u52_dispatch((pvec)R.data(), (cpvec)A.data(), (cpvec)B.data(), an, bn, &sc);
                td[p] = (now() - t0) / reps;
                t0 = now();
                for(long r = 0; r < reps; r++)
                    mul_u52_toom32((pvec)R.data(), (cpvec)A.data(), (cpvec)B.data(), an, bn, &sc);
                tt[p] = (now() - t0) / reps;
            }
            qsort(td, PASSES, 8, cmpd); qsort(tt, PASSES, 8, cmpd);
            double md = td[PASSES / 2] * 1e9, mt = tt[PASSES / 2] * 1e9;
            printf("%zu,%zu,%.4f,%.1f,%.1f,%.4f\n", an, bn, (double)bn / an, md, mt, md / mt);
            fflush(stdout);
        }
    }
    scratch_destroy(&sc);
    return 0;
}
