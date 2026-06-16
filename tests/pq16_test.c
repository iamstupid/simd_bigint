// pq16.h validation: every product is checked against GMP mpn_mul AND
// against the original fft16.h implementation (disjoint namespaces, both
// headers live in this TU).
//   gate 1: balanced grid -- every pow2 / PFA-3/5/7 grid point in band
//   gate 2: adversarial operands at the band caps (all-max digits drive
//           the worst-case coefficient magnitude; the sparse pattern hits
//           the carry/junction paths)
//   gate 3: random shapes across the whole band, both an >< bn orders
// Build:  clang -O2 -march=native -std=c11 -I../include pq16_test.c \
//           -I/tmp/gmp-build /tmp/gmp-build/.libs/libgmp.a -lm -o pq16_test
// Centered codec at every size: add -DPQ16_FORCE_CENTERED=1
#define SCRATCH_IMPLEMENTATION
#include "pq16.h"
#include "fft16.h"
#include <gmp.h>
#include <stdio.h>

#define MAXL 131072u            /* per-operand limb cap (2*(an+bn) <= 2^19) */
static uint64_t A[MAXL], B[MAXL];
static uint64_t R[2 * MAXL + 16], RO[2 * MAXL + 16], REF[2 * MAXL + 16];

static uint64_t rng = 42;
static uint64_t xr(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static int fails, runs;

// pattern: 0 random, 1 all-max digits, 2 sparse (mostly zero, max spikes)
static void fill(uint64_t *p, size_t n, int pat){
    for(size_t i = 0; i < n; ++i)
        p[i] = pat == 1 ? ~0ull : pat == 2 ? ((i & 63) == 0 ? ~0ull : 0) : xr();
}

static void check(size_t an, size_t bn, int pat, const char *tag){
    fill(A, an, pat);
    fill(B, bn, pat);
    if(an >= bn) mpn_mul((mp_ptr)REF, (mp_srcptr)A, an, (mp_srcptr)B, bn);
    else         mpn_mul((mp_ptr)REF, (mp_srcptr)B, bn, (mp_srcptr)A, an);
    ++runs;
    if(!pq16_mul(R, A, (ptrdiff_t)an, B, (ptrdiff_t)bn)){
        printf("FAIL %s an=%zu bn=%zu pat=%d: pq16_mul returned 0\n", tag, an, bn, pat);
        ++fails;
        return;
    }
    if(memcmp(R, REF, (an + bn) * 8)){
        size_t i = 0;
        while(R[i] == REF[i]) ++i;
        printf("FAIL %s an=%zu bn=%zu pat=%d: limb %zu got %016llx want %016llx\n",
               tag, an, bn, pat, i,
               (unsigned long long)R[i], (unsigned long long)REF[i]);
        ++fails;
        return;
    }
    // cross-check the original implementation on the same inputs
    if(fft16_mul(RO, A, (ptrdiff_t)an, B, (ptrdiff_t)bn) &&
       memcmp(RO, REF, (an + bn) * 8)){
        printf("FAIL(old!) %s an=%zu bn=%zu pat=%d: fft16 mismatches GMP\n",
               tag, an, bn, pat);
        ++fails;
    }
}

int main(void){
    // gate 1: every transform grid point, balanced (an = bn = nfull/4)
    for(uint32_t n = 512; n <= (1u << 19); n <<= 1)
        check(n / 4, n / 4, 0, "pow2");
    for(uint32_t br = 256; 3u * br <= (1u << 19); br <<= 1)
        check(3 * br / 4, 3 * br / 4, 0, "pfa3");
    for(uint32_t br = 128; 5u * br <= (1u << 19); br <<= 1)
        check(5 * br / 4, 5 * br / 4, 0, "pfa5");
    for(uint32_t br = 128; 7u * br <= (1u << 19); br <<= 1)
        check(7 * br / 4, 7 * br / 4, 0, "pfa7");
    printf("gate 1 (grid):        %d cases\n", runs);

    // gate 2: adversarial patterns at the caps and band landmarks
    int g1 = runs;
    static const size_t caps[] = { 8192, 16384, 32768, 40960, 49152, 57344,
                                   65536, 81920, 98304, 114688, 131072 };
    for(size_t i = 0; i < sizeof caps / sizeof *caps; ++i){
        check(caps[i], caps[i], 1, "advmax");
        check(caps[i], caps[i], 2, "advsparse");
    }
    // PFA-7 uncentered precision cap: need = 7*2^14 exactly
    check(7 * 4096, 7 * 4096, 1, "pfa7cap");
    printf("gate 2 (adversarial): %d cases\n", runs - g1);

    // gate 3: random shapes, full band, both orders
    int g2 = runs;
    for(int t = 0; t < 300; ++t){
        size_t an = 1 + xr() % MAXL;
        size_t bn = 1 + xr() % MAXL;
        while(2 * (an + bn) > (1u << 19)){
            if(an > bn) an = 1 + an / 2; else bn = 1 + bn / 2;
        }
        check(an, bn, 0, "rand");
    }
    printf("gate 3 (random):      %d cases\n", runs - g2);

    printf(fails ? "FAILED: %d of %d\n" : "OK: all %d cases\n",
           fails ? fails : runs, fails ? runs : runs);
    return fails != 0;
}
