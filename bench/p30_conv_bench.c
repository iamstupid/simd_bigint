// conv-pass microbench: production schoolbook conv16 vs parity-split
// Karatsuba (docs §3c-B follow-up).
//
// Karatsuba formulation that fits the lane layout: the LINEAR split
// a0 + x^8 a1 produces 15-wide sub-products that cannot pack into an
// 8-lane half. The PARITY split works: with y = x^2,
//     c_even(y) = ae*be + y*(ao*bo)          all three products are
//     c_odd (y) = (ae+ao)(be+bo) - ae*be     TWISTED 8-point convs
//                 - ao*bo                     mod (y^8 - ww)
// Twisted-8 outputs are exactly 8 lanes -> TWO spectral positions
// batch into zmm halves at full lane use: 3 batched convs x 16 lane-
// products = 24 vpmuludq per position vs 32 for schoolbook.
// Windows via vpermt2d with 8 shared constant index vectors (selecting
// from [X | ww*X] with per-half wrap); b-lane pair-broadcasts via two
// immediate shuffles (no index registers).
//
// Build: clang -O3 -march=native -std=c11 -I../include p30_conv_bench.c \
//          -lm -o p30_conv_bench
#define _POSIX_C_SOURCE 199309L
#define SCRATCH_IMPLEMENTATION
#include "p30_ref.h"
#include <stdio.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rngs = 42;
static uint64_t xr(void){ rngs ^= rngs << 13; rngs ^= rngs >> 7; rngs ^= rngs << 17; return rngs; }

/* ---- scalar reference: c = a*b mod (x^16 - ww), canonical inputs --- */
static void conv16_ref(uint32_t *c, const uint32_t *a, const uint32_t *b,
                       uint32_t ww, uint32_t p){
    /* matches the kernels' Montgomery stray: result = (a*b mod) * R^-1 */
    uint32_t rinv = p30_invm((uint32_t)(((uint64_t)1 << 32) % p), p);
    for(int l = 0; l < 16; ++l){
        p30_u128 s1 = 0, s2 = 0;
        for(int i = 0; i < 16; ++i){
            int j = l - i;
            if(j >= 0) s1 += (p30_u128)a[i] * b[j];
            else       s2 += (p30_u128)a[i] * b[j + 16];
        }
        uint32_t v = (uint32_t)((s1 % p + (p30_u128)ww * (uint32_t)(s2 % p)) % p);
        c[l] = p30_mulm(v, rinv, p);
    }
}

/* ==== V1: parity-split Karatsuba, 2 positions per call ============== */
/* twisted-window index vectors: W_i[l] = X[l-i] if (l&7) >= i else
 * ww*X[l-i+8]  ->  vpermt2d from (X : AW) with idx_i[l] =
 * (l&7) >= i ? l-i : 24+l-i. Shared constants (built once). */
static __m512i KIDX[8];
static __m512i ITLV_E, ITLV_O, DEIL_E, DEIL_O, ROT1;
static void kara_init(void){
    uint32_t t[16];
    for(int i = 1; i < 8; ++i){
        for(int l = 0; l < 16; ++l)
            t[l] = (uint32_t)(((l & 7) >= i) ? l - i : 24 + l - i);
        KIDX[i] = _mm512_loadu_si512(t);
    }
    /* y * m: per-half rotate up by 1, wrapped lane gets ww*m */
    for(int l = 0; l < 16; ++l)
        t[l] = (uint32_t)(((l & 7) >= 1) ? l - 1 : 24 + l - 1);
    ROT1 = _mm512_loadu_si512(t);
    /* deinterleave a position pair: E = even coeffs [k | k1], O = odd */
    for(int l = 0; l < 8; ++l){ t[l] = (uint32_t)(2 * l); t[8 + l] = (uint32_t)(16 + 2 * l); }
    DEIL_E = _mm512_loadu_si512(t);
    for(int l = 0; l < 8; ++l){ t[l] = (uint32_t)(2 * l + 1); t[8 + l] = (uint32_t)(16 + 2 * l + 1); }
    DEIL_O = _mm512_loadu_si512(t);
    /* re-interleave: position k from (CE half0, CO half0) etc. */
    for(int l = 0; l < 8; ++l){ t[2 * l] = (uint32_t)l; t[2 * l + 1] = (uint32_t)(16 + l); }
    ITLV_E = _mm512_loadu_si512(t);
    for(int l = 0; l < 8; ++l){ t[2 * l] = (uint32_t)(8 + l); t[2 * l + 1] = (uint32_t)(16 + 8 + l); }
    ITLV_O = _mm512_loadu_si512(t);
}

/* b-pair broadcast for step i: lanes 0..7 = B[i], lanes 8..15 = B[8+i]
 * (two immediate shuffles, no index registers) */
#define KBCAST(B, i) \
    _mm512_shuffle_i32x4( \
        _mm512_shuffle_epi32((B), (_MM_PERM_ENUM)(((i) & 3) * 0x55)), \
        _mm512_shuffle_epi32((B), (_MM_PERM_ENUM)(((i) & 3) * 0x55)), \
        (int)((((i) >> 2) & 1) * 0x55 + 0xA0))
/* sel: lanes0,1 = src128[(i>>2)&1], lanes2,3 = src128[2 + ((i>>2)&1)]:
 * imm = b01b01 | 10+b 10+b << ...: (b) | (b)<<2 | (2+b)<<4 | (2+b)<<6
 *     = b*0x05 + 0xA0  with b = (i>>2)&1                              */

/* one twisted-8 batched conv: acc (re, ro) += conv(X via windows, B) */
#define KCONV8(X, AW, B, RE, RO) do{                                     \
    RE = _mm512_add_epi64(RE, _mm512_mul_epu32((X), kb_tab[0]));         \
    RO = _mm512_add_epi64(RO,                                            \
        _mm512_mul_epu32(_mm512_srli_epi64((X), 32), kb_tab[0]));        \
    for(int i_ = 1; i_ < 8; ++i_){                                       \
        __m512i w_ = _mm512_permutex2var_epi32((X), KIDX[i_], (AW));     \
        RE = _mm512_add_epi64(RE, _mm512_mul_epu32(w_, kb_tab[i_]));     \
        RO = _mm512_add_epi64(RO,                                        \
            _mm512_mul_epu32(_mm512_srli_epi64(w_, 32), kb_tab[i_]));    \
    }                                                                    \
}while(0)

static inline __m512i kshrp(__m512i x, __m512i vp){    /* [0,2p)->[0,p) */
    return _mm512_min_epu32(x, _mm512_sub_epi32(x, vp));
}
static inline __m512i kshrp3(__m512i x, __m512i vp, __m512i vp2){
    /* [0, ~2.65p) -> [0, p): one fold vs 2p, one vs p */
    x = _mm512_min_epu32(x, _mm512_sub_epi32(x, vp2));
    return _mm512_min_epu32(x, _mm512_sub_epi32(x, vp));
}
/* wide REDC of (re,ro) 64-bit accs (< 2^63.x) -> packed u32 in [0,2p) */
static inline __m512i kredc(__m512i re, __m512i ro, __m512i vJ, __m512i vP,
                            __m512i vP2w){
    const __m512i M32 = _mm512_set1_epi64(0xFFFFFFFFu);
    __m512i me = _mm512_mul_epu32(re, vJ), mo = _mm512_mul_epu32(ro, vJ);
    __m512i ue = _mm512_mul_epu32(me, vP), uo = _mm512_mul_epu32(mo, vP);
    __mmask8 ce = _mm512_test_epi64_mask(re, M32);
    __mmask8 co = _mm512_test_epi64_mask(ro, M32);
    __m512i se = _mm512_add_epi64(_mm512_srli_epi64(re, 32),
                                  _mm512_srli_epi64(ue, 32));
    __m512i so = _mm512_add_epi64(_mm512_srli_epi64(ro, 32),
                                  _mm512_srli_epi64(uo, 32));
    se = _mm512_mask_add_epi64(se, ce, se, _mm512_set1_epi64(1));
    so = _mm512_mask_add_epi64(so, co, so, _mm512_set1_epi64(1));
    se = _mm512_min_epu64(se, _mm512_sub_epi64(se, vP2w));
    so = _mm512_min_epu64(so, _mm512_sub_epi64(so, vP2w));
    __m512i r = _mm512_mask_mov_epi32(se, (__mmask16)0xAAAA,
                                      _mm512_slli_epi64(so, 32));
    return r;          /* < 2.65p: caller folds to its target range */
}

/* fa[k], fa[k+1] *= fb (mod x^16 - ww_k / ww_k1); canonical inputs */
static void conv16_kara2_dev(uint32_t *fa, const uint32_t *fb,
                             p30_cc wwk, p30_cc wwk1,
                             const p30_prime *pp, __m512i vp){
    const __m512i vP2w = _mm512_set1_epi64(pp->p2);
    const __m512i vJ = _mm512_set1_epi64(pp->J);
    const __m512i vP = _mm512_set1_epi64(pp->p);
    __m512i vp2 = _mm512_add_epi32(vp, vp);
    /* per-pair vector twiddle constants (halves k / k1) */
    __m512i wc = _mm512_mask_mov_epi32(_mm512_set1_epi32((int)wwk.c),
                     (__mmask16)0xFF00, _mm512_set1_epi32((int)wwk1.c));
    __m512i wr = _mm512_mask_mov_epi32(_mm512_set1_epi32((int)wwk.rec),
                     (__mmask16)0xFF00, _mm512_set1_epi32((int)wwk1.rec));

    __m512i xa0 = _mm512_load_si512(fa);
    __m512i xa1 = _mm512_load_si512(fa + 16);
    __m512i xb0 = _mm512_load_si512(fb);
    __m512i xb1 = _mm512_load_si512(fb + 16);
    /* parity deinterleave into [k | k1] halves */
    __m512i AE = _mm512_permutex2var_epi32(xa0, DEIL_E, xa1);
    __m512i AO = _mm512_permutex2var_epi32(xa0, DEIL_O, xa1);
    __m512i BE = _mm512_permutex2var_epi32(xb0, DEIL_E, xb1);
    __m512i BO = _mm512_permutex2var_epi32(xb0, DEIL_O, xb1);
    /* sums (< 2p -> 1 min to < p keeps the 8-product bound) */
    __m512i AS = kshrp(_mm512_add_epi32(AE, AO), vp);
    __m512i BS = kshrp(_mm512_add_epi32(BE, BO), vp);
    /* ww-scaled window companions, canonical */
    __m512i AWE = kshrp(p30_bar2p(AE, wc, wr, vp), vp);
    __m512i AWO = kshrp(p30_bar2p(AO, wc, wr, vp), vp);
    __m512i AWS = kshrp(p30_bar2p(AS, wc, wr, vp), vp);

    /* b broadcast tables (8 each) -- registers, built by shuffles */
    __m512i kb_tab[8];
    __m512i re0 = _mm512_setzero_si512(), ro0 = _mm512_setzero_si512();
    __m512i re2 = _mm512_setzero_si512(), ro2 = _mm512_setzero_si512();
    __m512i res = _mm512_setzero_si512(), ros = _mm512_setzero_si512();
#define KFILL(B) do{ \
        kb_tab[0] = KBCAST(B, 0); kb_tab[1] = KBCAST(B, 1);              \
        kb_tab[2] = KBCAST(B, 2); kb_tab[3] = KBCAST(B, 3);              \
        kb_tab[4] = KBCAST(B, 4); kb_tab[5] = KBCAST(B, 5);              \
        kb_tab[6] = KBCAST(B, 6); kb_tab[7] = KBCAST(B, 7);              \
    }while(0)
    KFILL(BE); KCONV8(AE, AWE, BE, re0, ro0);
    KFILL(BO); KCONV8(AO, AWO, BO, re2, ro2);
    KFILL(BS); KCONV8(AS, AWS, BS, res, ros);
#undef KFILL
    /* NOTE: msum_acc - m0_acc - m2_acc is NOT exact in the raw
     * accumulator domain (the wrapped window terms use REDUCED ww*x
     * values: the identity holds only mod p, the u64 difference can go
     * negative). Combine AFTER REDC, in u32 with a +2p offset. */
    __m512i m0 = kshrp3(kredc(re0, ro0, vJ, vP, vP2w), vp, vp2);
    __m512i m2 = kshrp3(kredc(re2, ro2, vJ, vP, vP2w), vp, vp2);
    __m512i ms = kshrp3(kredc(res, ros, vJ, vP, vP2w), vp, vp2);
    __m512i m1 = _mm512_sub_epi32(
        _mm512_add_epi32(ms, vp2),
        _mm512_add_epi32(m0, m2));                /* (0, 3p) */
    /* c_even = m0 + y*m2: rotate m2 within halves, wrap gets ww*m2 */
    __m512i awm2 = p30_bar2p(m2, wc, wr, vp);     /* [0, 2p) */
    __m512i ym2 = _mm512_permutex2var_epi32(m2, ROT1, awm2);
    __m512i CE = _mm512_add_epi32(m0, ym2);       /* < 3p */
    __m512i CO = m1;                              /* < 3p */
    /* re-interleave to per-position vectors, output lazy [0,4p) */
    _mm512_store_si512(fa,      _mm512_permutex2var_epi32(CE, ITLV_E, CO));
    _mm512_store_si512(fa + 16, _mm512_permutex2var_epi32(CE, ITLV_O, CO));
}

/* ==================================================================== */
int main(void){
    p30_plan *pl = &p30_tls_plan;
    p30_plan_ensure(pl, 16);
    kara_init();
    const p30_prime *pp = &pl->b.pr[4];
    __m512i vp = _mm512_set1_epi32((int)pp->p);

    /* ---- validate both against the scalar reference ---- */
    enum { TV = 4096 };
    static uint32_t FA[TV * 16], FB[TV * 16], SA[TV * 16], REF[16];
    for(size_t i = 0; i < TV * 16; ++i){
        FA[i] = (uint32_t)(xr() % pp->p);
        FB[i] = (uint32_t)(xr() % pp->p);
    }
    memcpy(SA, FA, sizeof FA);
    for(size_t k = 0; k < TV; ++k){
        conv16_ref(REF, SA + 16 * k, FB + 16 * k, pp->w[k % (1u << 16)].c, pp->p);
        p30_conv16(SA + 16 * k, FB + 16 * k, pp->w[k % (1u << 16)], pp, vp);
        for(int l = 0; l < 16; ++l)
            if(SA[16 * k + l] % pp->p != REF[l]){
                printf("V0 FAIL k=%zu l=%d\n", k, l);
                return 1;
            }
    }
    memcpy(SA, FA, sizeof FA);
    for(size_t k = 0; k < TV; k += 2){
        uint32_t ref2[32];
        conv16_ref(ref2,      SA + 16 * k,      FB + 16 * k,      pp->w[k % (1u<<16)].c,     pp->p);
        conv16_ref(ref2 + 16, SA + 16 * k + 16, FB + 16 * k + 16, pp->w[(k+1) % (1u<<16)].c, pp->p);
        p30_conv16_kara2(SA + 16 * k, FB + 16 * k,
                         pp->w[k % (1u<<16)], pp->w[(k+1) % (1u<<16)], pp, vp);
        for(int l = 0; l < 32; ++l)
            if(SA[16 * k + l] % pp->p != ref2[l]){
                printf("KARA FAIL k=%zu l=%d got %u want %u\n",
                       k, l, SA[16 * k + l] % pp->p, ref2[l]);
                return 1;
            }
    }
    printf("validation OK (schoolbook + kara2, %d positions)\n", TV);

    /* ---- bench: L1-hot (64 positions) and streaming (TV) ---- */
    for(int big = 0; big < 2; ++big){
        size_t tvb = big ? TV : 64;
        long reps = big ? 2000 : 100000;
        double b0 = 1e30, b1 = 1e30;
        for(int pass = 0; pass < 7; ++pass){
            double t0 = now();
            for(long r = 0; r < reps; ++r)
                for(size_t k = 0; k < tvb; ++k)
                    p30_conv16(FA + 16 * k, FB + 16 * k,
                               pp->w[k % (1u << 16)], pp, vp);
            double t = (now() - t0) / (double)(reps * tvb) * 1e9;
            if(t < b0) b0 = t;
        }
        for(int pass = 0; pass < 7; ++pass){
            double t0 = now();
            for(long r = 0; r < reps; ++r)
                for(size_t k = 0; k + 2 <= tvb; k += 2)
                    p30_conv16_kara2(FA + 16 * k, FB + 16 * k,
                                     pp->w[k % (1u << 16)],
                                     pp->w[(k + 1) % (1u << 16)], pp, vp);
            double t = (now() - t0) / (double)(reps * tvb) * 1e9;
            if(t < b1) b1 = t;
        }
        printf("%s: schoolbook %.3f ns/pos  kara2 %.3f ns/pos  (ratio %.3f)\n",
               big ? "streaming" : "L1-hot   ", b0, b1, b0 / b1);
    }
    return 0;
}
