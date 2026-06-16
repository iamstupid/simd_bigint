/* Differential test for fft16.h (AVX-512 PQ FFT multiply), C11.
 * Gate 1: fwd -> scale -> inv -> emit round-trip (transform + emit, no
 *         pointwise) for every supported N.
 * Gate 2: full multiply vs GMP mpn_mul over balanced, unbalanced, random
 *         and adversarial (all-0xFF) shapes.
 * Build: clang -O2 -march=native -std=c11 -I../include fft16_test.c \
 *          -I<gmp-build> <gmp-build>/.libs/libgmp.a -lm -o fft16_test
 */
#include "fft16.h"
#include <gmp.h>
#include <stdio.h>

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t xr(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

static uint64_t RTS[262144], RTO[262144];
static int roundtrip(uint32_t n){
    f16_plan* pl = &f16_tls;
    f16_plan_ensure(pl, n, n);
    size_t limbs = n / 2u;
    for(size_t i = 0; i < limbs; ++i) RTS[i] = xr();

    f16_fwd(pl->da, RTS, (int64_t)limbs, n, pl);
    for(size_t i = 0; i < 2u * (size_t)n; ++i) pl->da[i] *= 1.0 / (double)n;
    f16_inv_leaves_r8(pl->da, n, pl);
    if(!f16_inv_final_emit(RTO, (int64_t)limbs, pl->da, n, pl)){
        printf("FAIL rt n=%u: carry out\n", n);
        return 1;
    }
    for(size_t i = 0; i < limbs; ++i){
        if(RTO[i] != RTS[i]){
            printf("FAIL rt n=%u limb=%zu got=%016llx ref=%016llx\n", n, i,
                   (unsigned long long)RTO[i], (unsigned long long)RTS[i]);
            return 1;
        }
    }
    return 0;
}

static int roundtrip_pfa(uint32_t branch, uint32_t M){
    f16_plan* pl = &f16_tls;
    uint32_t nfull = M * branch;
    f16_plan_ensure(pl, branch, nfull);
    size_t limbs = nfull / 2u;
    for(size_t i = 0; i < limbs; ++i) RTS[i] = xr();

    f16_pfa_fwd(pl->da, RTS, (int64_t)limbs, branch, M, pl);
    for(size_t i = 0; i < 2u * (size_t)nfull; ++i) pl->da[i] *= 1.0 / (double)nfull;
    if(F16_PFA3X2 && M == 3u){
        /* 3x2 pipeline: branch-halves are independent n/2 transforms; the
         * top r2 inverse lives in the fused emit */
        for(uint32_t h = 0; h < 6; ++h)
            { double* d = pl->da + (size_t)branch * h;
              f16_ileaf_run(d, branch / 2u, f16_shape_of_full(branch / 2u).leaf, pl);
              f16_inv_r8_only_full(d, branch / 2u, pl); }
    }else{
        const double* tw = pl->tw22[__builtin_ctz(branch)];
        for(uint32_t b = 0; b < M; ++b){
            double* d = pl->da + 2u * (size_t)branch * b;
            f16_inv_leaves_r8(d, branch, pl);
            f16_mem_ir22(d, branch, tw);
        }
    }
    if(!f16_pfa_emit(RTO, (int64_t)limbs, pl->da, branch, M, pl)){
        printf("FAIL rt-pfa M=%u br=%u: carry out\n", M, branch);
        return 1;
    }
    for(size_t i = 0; i < limbs; ++i){
        if(RTO[i] != RTS[i]){
            printf("FAIL rt-pfa M=%u br=%u limb=%zu got=%016llx ref=%016llx\n",
                   M, branch, i,
                   (unsigned long long)RTO[i], (unsigned long long)RTS[i]);
            return 1;
        }
    }
    return 0;
}

static uint64_t A[131072], B[131072], R[262144], REF[262144];

static int mul_case(size_t an, size_t bn, int adversarial){
    for(size_t i = 0; i < an; ++i) A[i] = adversarial ? ~0ull : xr();
    for(size_t i = 0; i < bn; ++i) B[i] = adversarial ? ~0ull : xr();
    if(!fft16_mul(R, A, (ptrdiff_t)an, B, (ptrdiff_t)bn)){
        printf("FAIL mul an=%zu bn=%zu: returned 0\n", an, bn);
        return 1;
    }
    if(an >= bn) mpn_mul((mp_ptr)REF, (mp_srcptr)A, an, (mp_srcptr)B, bn);
    else         mpn_mul((mp_ptr)REF, (mp_srcptr)B, bn, (mp_srcptr)A, an);
    for(size_t i = 0; i < an + bn; ++i){
        if(R[i] != REF[i]){
            printf("FAIL mul an=%zu bn=%zu adv=%d limb=%zu got=%016llx ref=%016llx\n",
                   an, bn, adversarial, i,
                   (unsigned long long)R[i], (unsigned long long)REF[i]);
            return 1;
        }
    }
    return 0;
}

int main(void){
    int fails = 0, cases = 0;

    /* gate 1: round-trip at every supported pow2 N (covers all 3 leaves)
     * and every PFA family at several branch sizes */
    for(uint32_t n = 512; n <= 131072; n <<= 1){
        fails += roundtrip(n); cases++;
    }
    for(uint32_t br = 256; br <= 32768; br <<= 1){ fails += roundtrip_pfa(br, 3); cases++; }
    for(uint32_t br = 128; br <= 16384; br <<= 1){ fails += roundtrip_pfa(br, 5); cases++; }
    for(uint32_t br = 128; br <= 8192;  br <<= 1){ fails += roundtrip_pfa(br, 7); cases++; }
    if(fails){ printf("ROUND-TRIP FAILED (%d)\n", fails); return 1; }
    printf("gate 1 ok: %d round-trips\n", cases);

    /* gate 2: GMP differential (incl. sizes that select each PFA family) */
    cases = 0;
    const size_t balanced[] = {128, 160, 192, 200, 224, 256, 320, 384, 448,
                               500, 512, 1000, 1024, 1536, 2048, 2240, 3072,
                               4096, 5120, 6144, 8192, 8960, 12288, 14336,
                               16000, 16384, 20480, 24576, 28672, 32768,
                               40960, 49152, 65536, 98304, 131072};
    for(size_t i = 0; i < sizeof(balanced)/sizeof(*balanced); ++i){
        fails += mul_case(balanced[i], balanced[i], 0); cases++;
    }
    /* adversarial all-0xFF at band edges and leaf-parity sizes */
    const size_t adv[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384,
                          24576, 28672, 32768, 65536, 131072};
    for(size_t i = 0; i < sizeof(adv)/sizeof(*adv); ++i){
        fails += mul_case(adv[i], adv[i], 1); cases++;
    }
    /* unbalanced + random shapes */
    for(int k = 0; k < 200; ++k){
        size_t an = 1 + xr() % 32000;
        size_t bn = 1 + xr() % 32000;
        if(2 * (an + bn) > (1u << 17)) { k--; continue; }
        fails += mul_case(an, bn, 0); cases++;
    }
    if(fails){ printf("FAILED: %d cases\n", fails); return 1; }
    printf("gate 2 ok: %d mul cases match GMP\n", cases);
    return 0;
}
