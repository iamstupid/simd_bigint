// Compare, at matched inner-loop work (~n^2 scalar 52x52 products):
//   midmul_basecase_simple(a[n], b[2n])   -- middle band, bn = 2*an
//   mul_u52_basecase      (a[n], b[n])     -- full square product, an = bn = n
// Both convolve n limbs of `a` against n output diagonals, so throughput
// (ns per scalar product) is directly comparable; midmul pays extra valignq
// (alignr64) to build shifted b-vectors, mul_basecase loads aligned.
// Verifies both against GMP, then emits CSV: n,midmul_ns,mul_ns
// Build: clang++ -O3 -march=native -I../include midmul_vs_mulbase_sweep.cpp -o midmul_vs_mulbase_sweep -lgmp
// Run:   ./midmul_vs_mulbase_sweep [n_max=512] [n_min=8] [step=1] > midmul_vs_mulbase_sweep.csv
#include "mulmid.h"
#include "mul.h"
#include <gmp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

#define M52 ((1ull << 52) - 1)
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rng = 0x243f6a8885a308d3ull;
static uint64_t xr(){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int cmpd(const void* x, const void* y){ double a = *(const double*)x, b = *(const double*)y; return (a > b) - (a < b); }
#define PASSES 11

// out = sum_p r[p] * 2^(52p) over nlimb redundant base-2^52 limbs
static void value_of(mpz_t out, const uint64_t *r, int64_t nlimb){
    mpz_set_ui(out, 0);
    mpz_t t; mpz_init(t);
    for(int64_t p = nlimb - 1; p >= 0; --p){
        mpz_mul_2exp(out, out, 52);
        mpz_set_ui(t, (unsigned long)(r[p] >> 32)); mpz_mul_2exp(t, t, 32);
        mpz_add(out, out, t);
        mpz_add_ui(out, out, (unsigned long)(r[p] & 0xffffffffull));
    }
    mpz_clear(t);
}
static void limb_to_mpz(mpz_t o, uint64_t v){ mpz_set_ui(o, (unsigned long)(v >> 32)); mpz_mul_2exp(o, o, 32); mpz_add_ui(o, o, (unsigned long)(v & 0xffffffffull)); }
// midmul band ref: sum_{an-1 <= i+j <= bn-1} a[i]b[j] 2^(52(i+j-(an-1)))
static void ref_band(mpz_t out, const uint64_t *a, const uint64_t *b, int64_t an, int64_t bn){
    mpz_set_ui(out, 0); mpz_t term, bb; mpz_inits(term, bb, NULL);
    for(int64_t i = 0; i < an; ++i) for(int64_t j = 0; j < bn; ++j){
        int64_t k = i + j; if(k < an - 1 || k > bn - 1) continue;
        limb_to_mpz(term, a[i]); limb_to_mpz(bb, b[j]); mpz_mul(term, term, bb);
        mpz_mul_2exp(term, term, 52 * (k - (an - 1))); mpz_add(out, out, term);
    }
    mpz_clears(term, bb, NULL);
}
// full product ref: (sum a[i]2^52i)*(sum b[j]2^52j)
static void ref_full(mpz_t out, const uint64_t *a, const uint64_t *b, int64_t an, int64_t bn){
    mpz_t A, B, t; mpz_inits(A, B, t, NULL); mpz_set_ui(A, 0); mpz_set_ui(B, 0);
    for(int64_t i = an - 1; i >= 0; --i){ mpz_mul_2exp(A, A, 52); limb_to_mpz(t, a[i]); mpz_add(A, A, t); }
    for(int64_t j = bn - 1; j >= 0; --j){ mpz_mul_2exp(B, B, 52); limb_to_mpz(t, b[j]); mpz_add(B, B, t); }
    mpz_mul(out, A, B); mpz_clears(A, B, t, NULL);
}

int main(int argc, char** argv){
    int64_t N_MAX = argc > 1 ? atol(argv[1]) : 512;
    int64_t N_MIN = argc > 2 ? atol(argv[2]) : 8;
    int64_t STEP  = argc > 3 ? atol(argv[3]) : 1;
    mpz_t vout, vref; mpz_inits(vout, vref, NULL);
    long fmid = 0, fmul = 0;
    printf("n,midmul_ns,mul_ns\n");
    for(int64_t n = N_MIN; n <= N_MAX; n += STEP){
        int64_t mid_bn = 2 * n, mid_rn = mid_bn - n + 1, mid_nout = mid_rn + 1; // midmul: a[n] x b[2n]
        std::vector<uint64_t> A(((n + 7) / 8) * 8 + 8, 0);
        std::vector<uint64_t> B(((mid_bn + 7) / 8) * 8 + 32, 0);   // 2n, shared: mul uses first n
        std::vector<uint64_t> Rm(((mid_nout + 7) / 8) * 8 + 16, 0);
        std::vector<uint64_t> Rs(((2 * n + 7) / 8) * 8 + 16, 0);
        for(int64_t i = 0; i < n; ++i)      A[i] = xr() & M52;
        for(int64_t j = 0; j < mid_bn; ++j) B[j] = xr() & M52;
        plimb rm = (plimb)Rm.data(); pvec rs = (pvec)Rs.data();
        const _limb *a = (const _limb*)A.data(), *b = (const _limb*)B.data();
        cpvec av = (cpvec)A.data(), bv = (cpvec)B.data();

        // correctness
        midmul_basecase_simple(rm, a, b, n, mid_bn);
        value_of(vout, Rm.data(), mid_nout); ref_band(vref, A.data(), B.data(), n, mid_bn);
        if(mpz_cmp(vout, vref) != 0){ if(++fmid <= 3) fprintf(stderr, "MIDMUL mismatch n=%ld\n", (long)n); }
        mul_u52_basecase(rs, av, bv, n, n);
        value_of(vout, Rs.data(), 2 * n); ref_full(vref, A.data(), B.data(), n, n);
        if(mpz_cmp(vout, vref) != 0){ if(++fmul <= 3) fprintf(stderr, "MUL mismatch n=%ld\n", (long)n); }

        // timing
        long reps = (long)(20000000.0 / ((double)n * n)) + 4;
        double tm[PASSES], ts[PASSES];
        for(int p = 0; p < PASSES; ++p){
            double t0 = now();
            for(long q = 0; q < reps; ++q) midmul_basecase_simple(rm, a, b, n, mid_bn);
            tm[p] = (now() - t0) / reps;
            t0 = now();
            for(long q = 0; q < reps; ++q) mul_u52_basecase(rs, av, bv, n, n);
            ts[p] = (now() - t0) / reps;
        }
        qsort(tm, PASSES, sizeof(double), cmpd); qsort(ts, PASSES, sizeof(double), cmpd);
        printf("%ld,%.3f,%.3f\n", (long)n, tm[PASSES/2]*1e9, ts[PASSES/2]*1e9);
        fflush(stdout);
    }
    fprintf(stderr, "correctness: midmul %ld mismatches, mul %ld mismatches\n", fmid, fmul);
    mpz_clears(vout, vref, NULL);
    return (fmid || fmul) ? 1 : 0;
}
