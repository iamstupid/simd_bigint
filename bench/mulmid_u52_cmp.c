/* mulmid_u52_cmp.c -- compare FUSED (acc=madd52(acc,..)) vs SPLIT (t=madd52(0,..); acc+=t)
 * for the AVX-512 IFMA middle product on Zen5, to learn where vpaddq runs:
 *   fused == split  -> vpaddq is on idle pipes (FP23), split is free
 *   fused <  split  -> vpaddq competes with vpmadd52 (FP01), prefer fused
 * Calibrates core GHz so core-cyc/digit is real (not a guessed x5).
 */
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <x86intrin.h>

typedef uint64_t u64;
typedef unsigned __int128 u128;
#define MASK52 ((1ULL << 52) - 1)

/* SPLIT: t = madd52(0,x,y); acc += t   (acc0=lo, acc1=hi) */
__attribute__((noinline))
static void mm_split(u64 *raw, const u64 *a, const u64 *b, long bn, long NB) {
    const __m512i Z = _mm512_setzero_si512();
    for (long blk = 0; blk < NB; blk++) {
        long i = (bn - 1) + blk * 8;
        __m512i acc0 = Z, acc1 = Z, ah = _mm512_set1_epi64((long long)b[bn - 1]);
        __m512i curr = _mm512_loadu_si512((const void *)(a + (i - bn)));
        acc1 = _mm512_add_epi64(acc1, _mm512_madd52hi_epu64(Z, ah, curr));
        for (long j = bn - 1; j >= 1; j--) {
            __m512i al = ah; ah = _mm512_set1_epi64((long long)b[j - 1]);
            curr = _mm512_loadu_si512((const void *)(a + (i - j)));
            acc0 = _mm512_add_epi64(acc0, _mm512_madd52lo_epu64(Z, al, curr));
            acc1 = _mm512_add_epi64(acc1, _mm512_madd52hi_epu64(Z, ah, curr));
        }
        curr = _mm512_loadu_si512((const void *)(a + i));
        acc0 = _mm512_add_epi64(acc0, _mm512_madd52lo_epu64(Z, ah, curr));
        _mm512_storeu_si512((void *)(raw + blk * 8), _mm512_add_epi64(acc0, acc1));
    }
}

/* FUSED: acc = madd52(acc,x,y)  (acc0=lo, acc1=hi); latency hidden across blocks */
__attribute__((noinline))
static void mm_fused(u64 *raw, const u64 *a, const u64 *b, long bn, long NB) {
    const __m512i Z = _mm512_setzero_si512();
    for (long blk = 0; blk < NB; blk++) {
        long i = (bn - 1) + blk * 8;
        __m512i acc0 = Z, acc1 = Z, ah = _mm512_set1_epi64((long long)b[bn - 1]);
        __m512i curr = _mm512_loadu_si512((const void *)(a + (i - bn)));
        acc1 = _mm512_madd52hi_epu64(acc1, ah, curr);
        for (long j = bn - 1; j >= 1; j--) {
            __m512i al = ah; ah = _mm512_set1_epi64((long long)b[j - 1]);
            curr = _mm512_loadu_si512((const void *)(a + (i - j)));
            acc0 = _mm512_madd52lo_epu64(acc0, al, curr);
            acc1 = _mm512_madd52hi_epu64(acc1, ah, curr);
        }
        curr = _mm512_loadu_si512((const void *)(a + i));
        acc0 = _mm512_madd52lo_epu64(acc0, ah, curr);
        _mm512_storeu_si512((void *)(raw + blk * 8), _mm512_add_epi64(acc0, acc1));
    }
}

/* PREBCAST: hoist all b broadcasts into registers once, reuse across blocks
 * (tests whether per-block vpbroadcastq is eating FP-pipe throughput) */
__attribute__((noinline))
static void mm_prebcast(u64 *raw, const u64 *a, const u64 *b, long bn, long NB) {
    __m512i B[64];
    for (long t = 0; t < bn; t++) B[t] = _mm512_set1_epi64((long long)b[t]);
    for (long blk = 0; blk < NB; blk++) {
        long i = (bn - 1) + blk * 8;
        __m512i acc0 = _mm512_setzero_si512(), acc1 = _mm512_setzero_si512();
        __m512i curr = _mm512_loadu_si512((const void *)(a + (i - bn)));
        acc1 = _mm512_madd52hi_epu64(acc1, B[bn - 1], curr);
        for (long j = bn - 1; j >= 1; j--) {
            curr = _mm512_loadu_si512((const void *)(a + (i - j)));
            acc0 = _mm512_madd52lo_epu64(acc0, B[j], curr);
            acc1 = _mm512_madd52hi_epu64(acc1, B[j - 1], curr);
        }
        curr = _mm512_loadu_si512((const void *)(a + i));
        acc0 = _mm512_madd52lo_epu64(acc0, B[0], curr);
        _mm512_storeu_si512((void *)(raw + blk * 8), _mm512_add_epi64(acc0, acc1));
    }
}

static double now_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec * 1e9 + ts.tv_nsec; }

/* core GHz via a ~1 cyc/iter dependent loop (dec->jnz critical path) */
static double core_ghz(void) {
    unsigned long n = 1500000000UL;
    double t0 = now_ns();
    asm volatile("1:\n\t dec %0\n\t jnz 1b" : "+r"(n) :: "cc");
    double t1 = now_ns();
    return 1500000000.0 / (t1 - t0);   /* iters / ns = GHz */
}

int main(int argc, char **argv) {
    long bn  = argc > 1 ? atol(argv[1]) : 24;
    long NB  = argc > 2 ? atol(argv[2]) : 32;
    long REP = argc > 3 ? atol(argv[3]) : 300000;
    long band = NB * 8, an = bn + band;
    const long PAD = 8;
    u64 *abuf = aligned_alloc(64, (PAD + an + 8) * 8); memset(abuf, 0, (PAD + an + 8) * 8);
    u64 *a = abuf + PAD, *b = aligned_alloc(64, ((bn + 7) & ~7L) * 8), *raw = malloc(band * 8);
    for (long s = 0; s < an; s++) a[s] = ((u64)mrand48() << 20 ^ (u64)mrand48()) & MASK52;
    for (long t = 0; t < bn; t++) b[t] = ((u64)mrand48() << 20 ^ (u64)mrand48()) & MASK52;

    double GHz = core_ghz();
    double TSCGHz;
    { unsigned aux; double t0 = now_ns(); unsigned long long c0 = __rdtscp(&aux);
      for (volatile long k = 0; k < 50000000; k++); unsigned long long c1 = __rdtscp(&aux); double t1 = now_ns();
      TSCGHz = (double)(c1 - c0) / (t1 - t0); }

    u64 sink = 0; unsigned aux;
    double digits = (double)REP * band;

    /* warm + time each */
    for (int w = 0; w < 3; w++) { mm_split(raw, a, b, bn, NB); mm_fused(raw, a, b, bn, NB); }

    double s0 = now_ns();
    for (long r = 0; r < REP; r++) { mm_split(raw, a, b, bn, NB); sink ^= raw[0] ^ raw[band - 1]; }
    double s1 = now_ns();
    double f0 = now_ns();
    for (long r = 0; r < REP; r++) { mm_fused(raw, a, b, bn, NB); sink ^= raw[0] ^ raw[band - 1]; }
    double f1 = now_ns();
    double p0 = now_ns();
    for (long r = 0; r < REP; r++) { mm_prebcast(raw, a, b, bn, NB); sink ^= raw[0] ^ raw[band - 1]; }
    double p1 = now_ns();
    (void)aux;

    double sp = (s1 - s0) / digits, fp = (f1 - f0) / digits, pp = (p1 - p0) / digits;
    printf("bn=%ld band=%ld  coreGHz=%.2f tscGHz=%.2f  ideal(bn/8)=%.2f core-cyc/digit\n",
           bn, band, GHz, TSCGHz, bn / 8.0);
    printf("  SPLIT : %.3f ns/digit  %.3f core-cyc/digit\n", sp, sp * GHz);
    printf("  FUSED : %.3f ns/digit  %.3f core-cyc/digit\n", fp, fp * GHz);
    printf("  PREBC : %.3f ns/digit  %.3f core-cyc/digit   (sink=%llx)\n", pp, pp * GHz, (unsigned long long)sink);
    free(abuf); free(b); free(raw);
    return 0;
}
