// Sweep midmul_basecase_simple on the nominal shape bn = 2*an, an in [8, AN_MAX].
// The middle product output r (radix 2^52, redundant) satisfies
//   value(r) = sum_p r[p]*2^(52p) = sum_{an-1 <= i+j <= bn-1} a[i]*b[j]*2^(52*(i+j-(an-1)))
// i.e. the middle band of the schoolbook product a*b, carry-free from below.
// Verifies that identity via GMP for every size, then times median ns/call.
// Emits CSV: an,bn,rn,ns
// Build: clang++ -O3 -march=native -I../include midmul_simple_sweep.cpp -o midmul_simple_sweep -lgmp
// Run:   ./midmul_simple_sweep > midmul_simple_sweep.csv
#include "mulmid.h"
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

// value(r) over rn+1 redundant base-2^52 limbs -> mpz
static void limbs_to_mpz(mpz_t out, const uint64_t *r, int64_t nlimb){
    mpz_set_ui(out, 0);
    mpz_t t; mpz_init(t);
    for(int64_t p = nlimb - 1; p >= 0; --p){
        mpz_mul_2exp(out, out, 52);          // out <<= 52
        mpz_add_ui(out, out, (unsigned long)r[p]);  // r[p] fits 64b; add via set+add
        if(r[p] >> 32){ mpz_set_ui(t, r[p] >> 32); mpz_mul_2exp(t, t, 32); mpz_add(out, out, t); mpz_sub_ui(out, out, (unsigned long)(r[p] >> 32) << 32); }
    }
    mpz_clear(t);
}
// robust 64-bit limb accumulate: out = sum r[p] * 2^(52p)
static void value_of(mpz_t out, const uint64_t *r, int64_t nlimb){
    mpz_set_ui(out, 0);
    mpz_t t; mpz_init(t);
    for(int64_t p = nlimb - 1; p >= 0; --p){
        mpz_mul_2exp(out, out, 52);
        mpz_set_ui(t, (unsigned long)(r[p] >> 32));   // hi 32
        mpz_mul_2exp(t, t, 32);
        mpz_add(out, out, t);
        mpz_add_ui(out, out, (unsigned long)(r[p] & 0xffffffffull)); // lo 32
    }
    mpz_clear(t);
}

// exact middle band reference: sum_{an-1 <= i+j <= bn-1} a[i]b[j] 2^(52(i+j-(an-1)))
static void ref_band(mpz_t out, const uint64_t *a, const uint64_t *b, int64_t an, int64_t bn){
    mpz_set_ui(out, 0);
    mpz_t term; mpz_init(term);
    for(int64_t i = 0; i < an; ++i){
        for(int64_t j = 0; j < bn; ++j){
            int64_t k = i + j;
            if(k < an - 1 || k > bn - 1) continue;   // keep the middle band [an-1, bn-1]
            mpz_set_ui(term, (unsigned long)(a[i] >> 32)); mpz_mul_2exp(term, term, 32); mpz_add_ui(term, term, (unsigned long)(a[i] & 0xffffffffull));
            mpz_mul_ui(term, term, (unsigned long)(b[j] & 0xffffffffull));
            {   mpz_t hi; mpz_init(hi); mpz_set_ui(hi, (unsigned long)(a[i] >> 32)); mpz_mul_2exp(hi, hi, 32); mpz_add_ui(hi, hi, (unsigned long)(a[i] & 0xffffffffull));
                mpz_mul_ui(hi, hi, (unsigned long)(b[j] >> 32)); mpz_mul_2exp(hi, hi, 32); mpz_add(term, term, hi); mpz_clear(hi); }
            mpz_mul_2exp(term, term, 52 * (k - (an - 1)));
            mpz_add(out, out, term);
        }
    }
    mpz_clear(term);
}

int main(int argc, char** argv){
    int64_t AN_MAX = argc > 1 ? atol(argv[1]) : 512;
    int64_t AN_MIN = argc > 2 ? atol(argv[2]) : 8;
    int64_t STEP   = argc > 3 ? atol(argv[3]) : 1;
    mpz_t vout, vref; mpz_inits(vout, vref, NULL);
    (void)limbs_to_mpz;
    long fails = 0;
    printf("an,bn,rn,ns\n");
    for(int64_t an = AN_MIN; an <= AN_MAX; an += STEP){
        int64_t bn = 2 * an;
        int64_t rn = bn - an + 1;               // == an+1 output diagonals
        int64_t nout = rn + 1;                  // output limbs
        std::vector<uint64_t> A(((an + 7) / 8) * 8 + 8, 0);
        std::vector<uint64_t> B(((bn + 7) / 8) * 8 + 32, 0);
        std::vector<uint64_t> R(((nout + 7) / 8) * 8 + 16, 0);
        for(int64_t i = 0; i < an; ++i) A[i] = xr() & M52;
        for(int64_t j = 0; j < bn; ++j) B[j] = xr() & M52;
        plimb r = (plimb)R.data();
        const _limb *a = (const _limb*)A.data(), *b = (const _limb*)B.data();

        // correctness
        midmul_basecase_simple(r, a, b, an, bn);
        value_of(vout, R.data(), nout);
        ref_band(vref, A.data(), B.data(), an, bn);
        if(mpz_cmp(vout, vref) != 0){
            fails++;
            if(fails <= 5){
                mpz_t d; mpz_init(d); mpz_sub(d, vout, vref);
                gmp_fprintf(stderr, "MISMATCH an=%ld bn=%ld  vout-vref sgn=%d bits=%lu\n",
                            (long)an, (long)bn, mpz_sgn(d), mpz_sgn(d)? mpz_sizeinbase(d,2):0);
                mpz_clear(d);
            }
        }

        // timing
        long reps = (long)(20000000.0 / ((double)an * an)) + 4;
        double t[PASSES];
        for(int p = 0; p < PASSES; ++p){
            double t0 = now();
            for(long q = 0; q < reps; ++q) midmul_basecase_simple(r, a, b, an, bn);
            t[p] = (now() - t0) / reps;
        }
        qsort(t, PASSES, sizeof(double), cmpd);
        printf("%ld,%ld,%ld,%.3f\n", (long)an, (long)bn, (long)rn, t[PASSES/2] * 1e9);
        fflush(stdout);
    }
    fprintf(stderr, "correctness: %ld mismatches\n", fails);
    mpz_clears(vout, vref, NULL);
    return fails ? 1 : 0;
}
