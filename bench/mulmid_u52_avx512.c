/* mulmid_u52_avx512.c -- AVX-512 IFMA basecase middle product (u52, beta=2^52),
 * Zen5-tuned: stationary 8-lane accumulators, UNALIGNED sliding a-loads, each
 * load reused for lo+hi, rolling b-broadcast, and the latency-breaking
 *   t = vpmadd52(0,x,y) ; acc += t   (split acc0=lo-terms / acc1=hi-terms)
 * so the carried dependency rides vpaddq (FP23, idle) not vpmadd52 (FP01).
 *
 * Per output position p (digit of the band [bn-1, .. ]):
 *   acc0[p] = sum_{t=0..bn-1} lo52(b[t]*a[p-t])
 *   acc1[p] = sum_{t=0..bn-1} hi52(b[t]*a[p-1-t])      (a[idx]=0 for idx<0/>=an)
 *   raw[p]  = acc0[p] + acc1[p]   (< 2*bn*2^52, fits u64; canonize is deferred)
 *
 * Build: clang -std=gnu17 -O3 -march=native mulmid_u52_avx512.c \
 *              /home/dev/bigint/GIMP/.libs/libgmp.a -o /tmp/.../mm
 */
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>
#include <gmp.h>

typedef uint64_t u64;
typedef unsigned __int128 u128;
#define MASK52 ((1ULL << 52) - 1)

/* ---------- scalar spec (defines the exact value) ---------- */
static void mulmid_scalar(u64 *raw, const u64 *a, long an, const u64 *b, long bn, long band) {
    for (long k = 0; k < band; k++) {
        long p = (bn - 1) + k;
        u64 acc0 = 0, acc1 = 0;
        for (long t = 0; t < bn; t++) {
            long slo = p - t, shi = (p - 1) - t;
            if (slo >= 0 && slo < an) { u128 pr = (u128)a[slo] * b[t]; acc0 += (u64)(pr & MASK52); }
            if (shi >= 0 && shi < an) { u128 pr = (u128)a[shi] * b[t]; acc1 += (u64)((pr >> 52) & MASK52); }
        }
        raw[k] = acc0 + acc1;
    }
}

/* ---------- AVX-512 IFMA kernel ---------- */
/* a must be addressable on [-bn, ..]; band = NB*8 output positions. */
static void mulmid_avx512(u64 *raw, const u64 *a, long an, const u64 *b, long bn, long NB) {
    const __m512i zero = _mm512_setzero_si512();
    (void)an;
    for (long blk = 0; blk < NB; blk++) {
        long i = (bn - 1) + blk * 8;                 /* block base output position */
        __m512i acc0 = zero, acc1 = zero;

        /* prologue: hi(b[bn-1] * a[i-bn ..]) */
        __m512i ah = _mm512_set1_epi64((long long)b[bn - 1]);
        __m512i curr = _mm512_loadu_si512((const void *)(a + (i - bn)));
        acc1 = _mm512_add_epi64(acc1, _mm512_madd52hi_epu64(zero, ah, curr));

        /* main: each curr feeds lo(b[j]) and hi(b[j-1]); b broadcast rolls */
        for (long j = bn - 1; j >= 1; j--) {
            __m512i al = ah;                                       /* = b[j]   */
            ah = _mm512_set1_epi64((long long)b[j - 1]);           /* = b[j-1] */
            curr = _mm512_loadu_si512((const void *)(a + (i - j)));
            acc0 = _mm512_add_epi64(acc0, _mm512_madd52lo_epu64(zero, al, curr));
            acc1 = _mm512_add_epi64(acc1, _mm512_madd52hi_epu64(zero, ah, curr));
        }

        /* epilogue: lo(b[0] * a[i ..])  (ah == b[0] here) */
        curr = _mm512_loadu_si512((const void *)(a + i));
        acc0 = _mm512_add_epi64(acc0, _mm512_madd52lo_epu64(zero, ah, curr));

        _mm512_storeu_si512((void *)(raw + blk * 8), _mm512_add_epi64(acc0, acc1));
    }
}

/* ---------- independent mpz reference: kernel value as an integer ---------- */
static void rect_mpz(mpz_t rect, const u64 *a, long an, const u64 *b, long bn, long band) {
    long band_lo = bn - 1, band_hi = bn - 1 + band - 1;
    mpz_set_ui(rect, 0);
    mpz_t term; mpz_init(term);
    for (long s = 0; s < an; s++)
        for (long t = 0; t < bn; t++) {
            u128 pr = (u128)a[s] * b[t];
            u64 lo = (u64)(pr & MASK52), hi = (u64)((pr >> 52) & MASK52);
            long plo = s + t, phi = s + t + 1;
            if (plo >= band_lo && plo <= band_hi) { mpz_set_ui(term, lo); mpz_mul_2exp(term, term, 52 * plo); mpz_add(rect, rect, term); }
            if (phi >= band_lo && phi <= band_hi) { mpz_set_ui(term, hi); mpz_mul_2exp(term, term, 52 * phi); mpz_add(rect, rect, term); }
        }
    mpz_clear(term);
}

static double now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e9 + ts.tv_nsec; }

int main(int argc, char **argv) {
    long bn  = argc > 1 ? atol(argv[1]) : 24;
    long NB  = argc > 2 ? atol(argv[2]) : 32;     /* blocks of 8 output digits  */
    long REP = argc > 3 ? atol(argv[3]) : 200000;
    long band = NB * 8;
    long an  = bn + band;                          /* long operand length        */
    const long PAD = 8;

    u64 *abuf = aligned_alloc(64, (PAD + an + 8) * 8);
    memset(abuf, 0, (PAD + an + 8) * 8);
    u64 *a = abuf + PAD;
    u64 *b = aligned_alloc(64, ((bn + 7) & ~7L) * 8);
    u64 *raw_s = malloc(band * 8), *raw_v = malloc(band * 8);

    gmp_randstate_t rs; gmp_randinit_default(rs); gmp_randseed_ui(rs, 1);
    mpz_t r; mpz_init(r);

    /* ---- correctness: 200 random instances, simd==scalar (bit) and ==rect (mpz) ---- */
    long fails = 0;
    for (int it = 0; it < 200; it++) {
        for (long s = 0; s < an; s++) a[s] = ((u64)mrand48() << 20 ^ (u64)mrand48()) & MASK52;
        for (long t = 0; t < bn; t++) b[t] = ((u64)mrand48() << 20 ^ (u64)mrand48()) & MASK52;
        mulmid_scalar(raw_s, a, an, b, bn, band);
        mulmid_avx512(raw_v, a, an, b, bn, NB);
        if (memcmp(raw_s, raw_v, band * 8) != 0) { fails++; if (fails <= 3) printf("  simd!=scalar at it=%d\n", it); continue; }
        /* raw as integer == rect */
        mpz_t got, term; mpz_inits(got, term, NULL); mpz_set_ui(got, 0);
        for (long k = 0; k < band; k++) { mpz_set_ui(term, raw_v[k]); mpz_mul_2exp(term, term, 52 * (bn - 1 + k)); mpz_add(got, got, term); }
        rect_mpz(r, a, an, b, bn, band);
        if (mpz_cmp(got, r) != 0) { fails++; if (fails <= 3) gmp_printf("  raw!=rect at it=%d\n", it); }
        mpz_clears(got, term, NULL);
    }
    printf("correctness bn=%ld band=%ld: %s (%ld fails)\n", bn, band, fails ? "FAIL" : "OK", fails);

    /* ---- benchmark ---- */
    for (long s = 0; s < an; s++) a[s] = ((u64)mrand48() << 20 ^ (u64)mrand48()) & MASK52;
    for (long t = 0; t < bn; t++) b[t] = ((u64)mrand48() << 20 ^ (u64)mrand48()) & MASK52;
    mulmid_avx512(raw_v, a, an, b, bn, NB);               /* warm */
    u64 sink = 0;
    unsigned aux;
    double t0 = now_ns(); unsigned long long c0 = __rdtscp(&aux);
    for (long r2 = 0; r2 < REP; r2++) {
        mulmid_avx512(raw_v, a, an, b, bn, NB);
        sink ^= raw_v[0] ^ raw_v[band - 1];
    }
    unsigned long long c1 = __rdtscp(&aux); double t1 = now_ns();
    double digits = (double)REP * band;
    double ns = t1 - t0, tsc = (double)(c1 - c0);
    double ns_per_digit = ns / digits;
    printf("bench  bn=%ld band=%ld rep=%ld: %.3f ns/digit | TSC %.3f tick/digit | "
           "core-cyc/digit ~%.3f (x5.0) | ideal bn/8=%.3f | sink=%llx\n",
           bn, band, REP, ns_per_digit, tsc / digits, ns_per_digit * 5.0, bn / 8.0, (unsigned long long)sink);

    mpz_clear(r); free(abuf); free(b); free(raw_s); free(raw_v);
    return 0;
}
