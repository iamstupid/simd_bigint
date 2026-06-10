// Benchmark-based threshold tuner for the u52 mul dispatch (GMP tuneup
// analog). Build with -DU52_TUNE_BUILD so the dispatch thresholds become the
// mutable globals defined here; each threshold is then found bottom-up by
// walking sizes and switching when the contender wins sustainedly.
// Measurement hygiene: A/B interleaved within each size (immune to slow
// machine drift), medians of PASSES, sustained-win + confirmation window.
//
// Build: clang++ -O3 -march=native -DU52_TUNE_BUILD -I../include tune_u52.cpp -o tune_u52
// Run:   ./tune_u52 > ../include/u52-mparam.h
#include "mul.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

uint64_t u52_thr_t22 = 85, u52_thr_t33 = 320, u52_thr_t32ok = 96, u52_thr_t42ok = 96;

#define M52 ((1ull << 52) - 1)
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rng = 42;
static uint64_t xr(){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int cmpd(const void* x, const void* y){ double a = *(const double*)x, b = *(const double*)y; return (a > b) - (a < b); }
#define PASSES 9

static scratch g_sc;
static std::vector<uint64_t> A, B, R;
static void fill(size_t an, size_t bn){
    A.assign((an + 7) / 8 * 8, 0); B.assign((bn + 7) / 8 * 8, 0);
    R.assign(((an + bn + 7) / 8 * 8) + 16, 0);
    for(size_t i = 0; i < an; i++) A[i] = xr() & M52;
    for(size_t i = 0; i < bn; i++) B[i] = xr() & M52;
}

typedef int (*mulfn)(pvec, cpvec, cpvec, uint64_t, uint64_t, scratch*);
static int call_basecase(pvec r, cpvec a, cpvec b, uint64_t an, uint64_t bn, scratch*){
    mul_u52_basecase(r, a, b, an, bn);
    return 0;
}

// interleaved A/B medians; returns ratio tB/tA (<1 means B faster)
static double ab_ratio(mulfn fa, mulfn fb, size_t an, size_t bn){
    fill(an, bn);
    long reps = (long)(800000 / (an + bn)) + 2;
    double ta[PASSES], tb[PASSES];
    pvec r = (pvec)R.data(); cpvec a = (cpvec)A.data(), b = (cpvec)B.data();
    for(int p = 0; p < PASSES; p++){
        double t0 = now();
        for(long q = 0; q < reps; q++) fa(r, a, b, an, bn, &g_sc);
        ta[p] = now() - t0;
        t0 = now();
        for(long q = 0; q < reps; q++) fb(r, a, b, an, bn, &g_sc);
        tb[p] = now() - t0;
    }
    qsort(ta, PASSES, 8, cmpd); qsort(tb, PASSES, 8, cmpd);
    return tb[PASSES / 2] / ta[PASSES / 2];
}

// walk sizes ascending; threshold = first size opening a window where B wins
// >= 3 of 4, confirmed by >= 4 of the following 6
static uint64_t find_threshold(const char* name, mulfn fa, mulfn fb,
                               uint64_t lo, uint64_t hi, uint64_t step,
                               double an_ratio){
    std::vector<int> wins;
    std::vector<uint64_t> sizes;
    for(uint64_t bn = lo; bn <= hi; bn += step){
        size_t an = (size_t)(bn * an_ratio + 0.5);
        double rat = ab_ratio(fa, fb, an, bn);
        sizes.push_back(bn);
        wins.push_back(rat < 0.995);
        fprintf(stderr, "  %s bn=%llu an=%zu B/A=%.4f%s\n", name,
                (unsigned long long)bn, an, rat, rat < 0.995 ? " WIN" : "");
        size_t k = wins.size();
        if(k >= 10){
            for(size_t i = 0; i + 10 <= k; i++){
                int w4 = wins[i] + wins[i+1] + wins[i+2] + wins[i+3];
                int c6 = wins[i+4] + wins[i+5] + wins[i+6] + wins[i+7] + wins[i+8] + wins[i+9];
                if(w4 >= 3 && c6 >= 4) return sizes[i];
            }
        }
    }
    // fall back: first window of 4 with >= 3 wins
    for(size_t i = 0; i + 4 <= wins.size(); i++)
        if(wins[i] + wins[i+1] + wins[i+2] + wins[i+3] >= 3) return sizes[i];
    return hi + 1;
}

// dispatch wrappers that force a band by temporarily setting thresholds
static int disp(pvec r, cpvec a, cpvec b, uint64_t an, uint64_t bn, scratch* s){
    return mul_u52_dispatch(r, a, b, an, bn, s);
}
static uint64_t save33;
static int disp_no33(pvec r, cpvec a, cpvec b, uint64_t an, uint64_t bn, scratch* s){
    uint64_t k = u52_thr_t33; u52_thr_t33 = ~0ull;
    int c = mul_u52_dispatch(r, a, b, an, bn, s);
    u52_thr_t33 = k;
    return c;
}
static int disp_33now(pvec r, cpvec a, cpvec b, uint64_t an, uint64_t bn, scratch* s){
    uint64_t k = u52_thr_t33; u52_thr_t33 = bn;
    int c = mul_u52_dispatch(r, a, b, an, bn, s);
    u52_thr_t33 = k;
    return c;
}
static int kara(pvec r, cpvec a, cpvec b, uint64_t an, uint64_t bn, scratch* s){
    return mul_u52_karatsuba(r, a, b, an, bn, s);
}

int main(){
    g_sc = scratch_create_ex(4096, 1u << 26, 1u << 26);

    fprintf(stderr, "== T22: basecase vs toom22 (squares) ==\n");
    uint64_t t22 = find_threshold("t22", call_basecase, kara, 48, 160, 2, 1.0);
    // lane-budget clamp: fold internals reach 9*L0 = 18*(T22-1)*2^52 and the
    // canonneg machinery needs |lane| < 2^63, so T22-1 < 2^63/(18*2^52) ~ 113
    if(t22 > 112) t22 = 112;
    u52_thr_t22 = t22;

    fprintf(stderr, "== T32_OK: ladder-sans-toom32 vs toom32 at 3:2 ==\n");
    uint64_t t32 = find_threshold("t32", kara, mul_u52_toom32, t22, 240, 2, 1.5);
    u52_thr_t32ok = t32;

    fprintf(stderr, "== T42_OK: ladder-sans-toom42 vs toom42 at 2:1 ==\n");
    uint64_t t42 = find_threshold("t42", kara, mul_u52_toom42, t22, 240, 2, 2.0);
    u52_thr_t42ok = t42;

    fprintf(stderr, "== T33: X2 ladder vs toom33 (squares), pass 1 ==\n");
    uint64_t t33 = find_threshold("t33", disp_no33, disp_33now, 200, 520, 8, 1.0);
    u52_thr_t33 = t33;
    fprintf(stderr, "== T33 fixpoint pass 2 (self-recursion live) ==\n");
    uint64_t t33b = find_threshold("t33b", disp_no33, disp_33now,
                                   t33 > 264 ? t33 - 64 : 200, t33 + 96, 8, 1.0);
    if(t33b < ~0ull - 1) t33 = t33b;
    u52_thr_t33 = t33;

    printf("/* u52-mparam.h - GENERATED by bench/tune_u52; do not hand-edit\n"
           " * defaults. Thresholds in limbs of the smaller operand. */\n");
    printf("#ifndef MUL_U52_T22_THRESHOLD\n#define MUL_U52_T22_THRESHOLD %llu\n#endif\n",
           (unsigned long long)t22);
    printf("#ifndef MUL_U52_T33_THRESHOLD\n#define MUL_U52_T33_THRESHOLD %llu\n#endif\n",
           (unsigned long long)t33);
    printf("#ifndef MUL_U52_T32_OK_THRESHOLD\n#define MUL_U52_T32_OK_THRESHOLD %llu\n#endif\n",
           (unsigned long long)t32);
    printf("#ifndef MUL_U52_T42_OK_THRESHOLD\n#define MUL_U52_T42_OK_THRESHOLD %llu\n#endif\n",
           (unsigned long long)t42);
    fprintf(stderr, "RESULT: T22=%llu T33=%llu T32_OK=%llu T42_OK=%llu\n",
            (unsigned long long)t22, (unsigned long long)t33,
            (unsigned long long)t32, (unsigned long long)t42);
    scratch_destroy(&g_sc);
    return 0;
}
