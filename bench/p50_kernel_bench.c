// p50 transform leaf kernels: the pieces that DIFFER from p30
// (everything else ports structurally):
//   K1 radix-8 forward moth pass (Barrett52 butterflies, lazy [0,4p))
//   K2 conv8 mod (x^8 - ww), SCHOOLBOOK: sliding [aw | a] windows
//      (valignq) x b-lane broadcasts, madd52 column accumulation,
//      ONE wide REDC52 per output (uniform R^-1 stray)
//   K3 conv8 parity-split KARATSUBA: y = x^2 -> 3 twisted conv4s,
//      two spectral positions batched per zmm (4+4 lanes); vector
//      (c, rec) constants are FINE here -- madd52 is lane-wise, the
//      p30 broadcast-rec trap does not exist in 64-bit lanes.
// Products: school 64/pos, kara 48/pos (= 16 vs 12 madd pairs).
//
// Build: clang -O3 -march=native -std=c11 p50_kernel_bench.c -lm \
//          -o /tmp/p50_kernel_bench
#define _POSIX_C_SOURCE 199309L
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 u128;
typedef uint64_t u64;

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static u64 rngs = 42;
static u64 xr(void){ rngs ^= rngs << 13; rngs ^= rngs >> 7; rngs ^= rngs << 17; return rngs; }
static volatile u64 SINK;

#define M52 ((1ull << 52) - 1)
#define P50 1108307720798209ull
static u64 mulm(u64 a, u64 b){ return (u64)((u128)a * b % P50); }
static u64 powm(u64 a, u64 e){
    u64 r = 1;
    for(; e; e >>= 1, a = mulm(a, a)) if(e & 1) r = mulm(r, a);
    return r;
}
static u64 invm(u64 a){ return powm(a, P50 - 2); }
static u64 rec52(u64 c){ return (u64)(((u128)c << 52) / P50); }

/* ---- primitives ---- */
static inline __m512i b52v(__m512i a, __m512i vc, __m512i vrec, __m512i vpn){
    __m512i z = _mm512_setzero_si512();
    __m512i q = _mm512_madd52hi_epu64(z, a, vrec);
    __m512i s = _mm512_madd52lo_epu64(z, a, vc);
    s = _mm512_madd52lo_epu64(s, q, vpn);
    return _mm512_and_si512(s, _mm512_set1_epi64((long long)M52));
}
static inline __m512i sh2_52(__m512i x, __m512i vp2){
    return _mm512_min_epu64(x, _mm512_sub_epi64(x, vp2));
}
static inline __m512i shp_52(__m512i x, __m512i vp){
    return _mm512_min_epu64(x, _mm512_sub_epi64(x, vp));
}
/* wide REDC52 of (lo, hi) columns (t < ~2^104) -> [0, 2p) */
static inline __m512i redc52(__m512i lo, __m512i hi, __m512i vJ, __m512i vP,
                             __m512i vM){
    __m512i lom = _mm512_and_si512(lo, vM);
    __m512i m = _mm512_and_si512(
        _mm512_madd52lo_epu64(_mm512_setzero_si512(), lom, vJ), vM);
    __m512i r = _mm512_madd52hi_epu64(
        _mm512_add_epi64(hi, _mm512_srli_epi64(lo, 52)), m, vP);
    __mmask8 cy = _mm512_test_epi64_mask(lom, vM);
    r = _mm512_mask_add_epi64(r, cy, r, _mm512_set1_epi64(1));
    return r;            /* < ~2.05p */
}

typedef struct { u64 c, rec; } cc;
static cc cc_of(u64 v){ cc t = { v, rec52(v) }; return t; }

/* ---- K1: radix-8 forward moth (port of p30_fwd8 to 64-bit) ---- */
#define BF(lo_, hi_, T) do{                                              \
        __m512i a_  = sh2_52(x[lo_], vp2);                               \
        __m512i wb_ = b52v(x[hi_], _mm512_set1_epi64((long long)(T).c),  \
                           _mm512_set1_epi64((long long)(T).rec), vpn);  \
        x[lo_] = _mm512_add_epi64(a_, wb_);                              \
        x[hi_] = _mm512_add_epi64(a_, _mm512_sub_epi64(vp2, wb_));       \
    }while(0)
static void fwd8(u64 *B, size_t m, const cc *tw,
                 __m512i vpn, __m512i vp2){
    size_t e = m >> 3;
    for(size_t v = 0; v < e; ++v){
        __m512i x[8];
        for(int c = 0; c < 8; ++c)
            x[c] = _mm512_load_si512(B + 8 * (v + (size_t)c * e));
        BF(0,4,tw[0]); BF(1,5,tw[0]); BF(2,6,tw[0]); BF(3,7,tw[0]);
        BF(0,2,tw[1]); BF(1,3,tw[1]); BF(4,6,tw[2]); BF(5,7,tw[2]);
        BF(0,1,tw[3]); BF(2,3,tw[4]); BF(4,5,tw[5]); BF(6,7,tw[6]);
        for(int c = 0; c < 8; ++c)
            _mm512_store_si512(B + 8 * (v + (size_t)c * e), x[c]);
    }
}

/* ---- K2: conv8 schoolbook ---- */
/* fa lazy [0,4p) -> shrunk; fb canonical; out lazy < 2.05p */
static void conv8_school(u64 *fa, const u64 *fb, cc ww,
                         __m512i vp, __m512i vpn, __m512i vp2,
                         __m512i vJ, __m512i vP, __m512i vM){
    __m512i a = _mm512_load_si512(fa);
    a = shp_52(sh2_52(a, vp2), vp);                 /* canonical */
    __m512i aw = shp_52(b52v(a, _mm512_set1_epi64((long long)ww.c),
                             _mm512_set1_epi64((long long)ww.rec), vpn),
                        vp);
    __m512i lo = _mm512_setzero_si512(), hi = _mm512_setzero_si512();
#define CSTEP(i) do{                                                     \
        __m512i b_ = _mm512_set1_epi64((long long)fb[i]);                \
        __m512i w_ = (i) ? _mm512_alignr_epi64(a, aw, 8 - (i)) : a;      \
        lo = _mm512_madd52lo_epu64(lo, w_, b_);                          \
        hi = _mm512_madd52hi_epu64(hi, w_, b_);                          \
    }while(0)
    CSTEP(0); CSTEP(1); CSTEP(2); CSTEP(3);
    CSTEP(4); CSTEP(5); CSTEP(6); CSTEP(7);
#undef CSTEP
    _mm512_store_si512(fa, redc52(lo, hi, vJ, vP, vM));
}

/* ---- K3: conv8 parity-split kara, 2 positions per call ---- */
static __m512i KIDX[4], DEIL_E, DEIL_O, ITLV_E, ITLV_O, ROT1;
static __m512i KBIDX[4];
static void kara_init(void){
    u64 t[8];
    for(int i = 0; i < 4; ++i){
        for(int l = 0; l < 8; ++l) t[l] = (u64)((l < 4) ? i : 4 + i);
        KBIDX[i] = _mm512_loadu_si512(t);
    }
    for(int i = 1; i < 4; ++i){
        for(int l = 0; l < 8; ++l)
            t[l] = (u64)(((l & 3) >= i) ? l - i : 12 + l - i);
        KIDX[i] = _mm512_loadu_si512(t);
    }
    for(int l = 0; l < 8; ++l)
        t[l] = (u64)(((l & 3) >= 1) ? l - 1 : 12 + l - 1);
    ROT1 = _mm512_loadu_si512(t);
    for(int l = 0; l < 4; ++l){ t[l] = (u64)(2*l); t[4+l] = (u64)(8 + 2*l); }
    DEIL_E = _mm512_loadu_si512(t);
    for(int l = 0; l < 4; ++l){ t[l] = (u64)(2*l+1); t[4+l] = (u64)(8 + 2*l+1); }
    DEIL_O = _mm512_loadu_si512(t);
    for(int l = 0; l < 4; ++l){ t[2*l] = (u64)l; t[2*l+1] = (u64)(8 + l); }
    ITLV_E = _mm512_loadu_si512(t);
    for(int l = 0; l < 4; ++l){ t[2*l] = (u64)(4 + l); t[2*l+1] = (u64)(12 + l); }
    ITLV_O = _mm512_loadu_si512(t);
}
/* b broadcast for step i: lanes 0-3 = B[i], 4-7 = B[4+i] */
static inline __m512i kbcast(__m512i B, int i){
    return _mm512_permutexvar_epi64(KBIDX[i], B);
}
#define KCONV4(X, AW, B, LO, HI) do{                                     \
        __m512i b0_ = kbcast(B, 0);                                      \
        LO = _mm512_madd52lo_epu64(LO, (X), b0_);                        \
        HI = _mm512_madd52hi_epu64(HI, (X), b0_);                        \
        for(int i_ = 1; i_ < 4; ++i_){                                   \
            __m512i w_ = _mm512_permutex2var_epi64((X), KIDX[i_], (AW)); \
            __m512i b_ = kbcast(B, i_);                                  \
            LO = _mm512_madd52lo_epu64(LO, w_, b_);                      \
            HI = _mm512_madd52hi_epu64(HI, w_, b_);                      \
        }                                                                \
    }while(0)
static void conv8_kara(u64 *fa, const u64 *fb, cc wwk, cc wwk1,
                       __m512i vp, __m512i vpn, __m512i vp2,
                       __m512i vJ, __m512i vP, __m512i vM){
    /* per-pair vector twiddles (lanes 0-3 = k, 4-7 = k+1): lane-wise
     * madd52 -> vector (c, rec) needs no special handling */
    __m512i wc = _mm512_mask_mov_epi64(_mm512_set1_epi64((long long)wwk.c),
                     (__mmask8)0xF0, _mm512_set1_epi64((long long)wwk1.c));
    __m512i wr = _mm512_mask_mov_epi64(_mm512_set1_epi64((long long)wwk.rec),
                     (__mmask8)0xF0, _mm512_set1_epi64((long long)wwk1.rec));
    __m512i xa0 = _mm512_load_si512(fa);
    __m512i xa1 = _mm512_load_si512(fa + 8);
    xa0 = shp_52(sh2_52(xa0, vp2), vp);
    xa1 = shp_52(sh2_52(xa1, vp2), vp);
    __m512i xb0 = _mm512_load_si512(fb);
    __m512i xb1 = _mm512_load_si512(fb + 8);
    xb0 = shp_52(sh2_52(xb0, vp2), vp);
    xb1 = shp_52(sh2_52(xb1, vp2), vp);
    __m512i AE = _mm512_permutex2var_epi64(xa0, DEIL_E, xa1);
    __m512i AO = _mm512_permutex2var_epi64(xa0, DEIL_O, xa1);
    __m512i BE = _mm512_permutex2var_epi64(xb0, DEIL_E, xb1);
    __m512i BO = _mm512_permutex2var_epi64(xb0, DEIL_O, xb1);
    __m512i AS = shp_52(_mm512_add_epi64(AE, AO), vp);
    __m512i BS = shp_52(_mm512_add_epi64(BE, BO), vp);
    __m512i AWE = shp_52(b52v(AE, wc, wr, vpn), vp);
    __m512i AWO = shp_52(b52v(AO, wc, wr, vpn), vp);
    __m512i AWS = shp_52(b52v(AS, wc, wr, vpn), vp);
    __m512i l0 = _mm512_setzero_si512(), h0 = _mm512_setzero_si512();
    __m512i l2 = _mm512_setzero_si512(), h2 = _mm512_setzero_si512();
    __m512i ls = _mm512_setzero_si512(), hs = _mm512_setzero_si512();
    KCONV4(AE, AWE, BE, l0, h0);
    KCONV4(AO, AWO, BO, l2, h2);
    KCONV4(AS, AWS, BS, ls, hs);
    /* combine post-REDC in canonical-ish domain (p30 lesson) */
    __m512i m0 = shp_52(redc52(l0, h0, vJ, vP, vM), vp);
    __m512i m2 = shp_52(redc52(l2, h2, vJ, vP, vM), vp);
    __m512i ms = shp_52(redc52(ls, hs, vJ, vP, vM), vp);
    m0 = shp_52(m0, vp); m2 = shp_52(m2, vp); ms = shp_52(ms, vp);
    __m512i m1 = _mm512_sub_epi64(_mm512_add_epi64(ms, vp2),
                                  _mm512_add_epi64(m0, m2));
    __m512i awm2 = b52v(m2, wc, wr, vpn);
    __m512i ym2 = _mm512_permutex2var_epi64(m2, ROT1, awm2);
    __m512i CE = _mm512_add_epi64(m0, ym2);
    __m512i CO = m1;
    _mm512_store_si512(fa,     _mm512_permutex2var_epi64(CE, ITLV_E, CO));
    _mm512_store_si512(fa + 8, _mm512_permutex2var_epi64(CE, ITLV_O, CO));
}

/* ---- scalar conv8 reference (matches the R^-1 stray) ---- */
static void conv8_ref(u64 *c, const u64 *a, const u64 *b, u64 ww){
    u64 rinv = invm((u64)(((u128)1 << 52) % P50));
    for(int l = 0; l < 8; ++l){
        u128 s1 = 0, s2 = 0;
        for(int i = 0; i < 8; ++i){
            int j = l - i;
            if(j >= 0) s1 += (u128)(a[i] % P50) * (b[j] % P50);
            else       s2 += (u128)(a[i] % P50) * (b[j + 8] % P50);
        }
        u64 v = (u64)((s1 % P50 + (u128)ww * (u64)(s2 % P50)) % P50);
        c[l] = mulm(v, rinv);
    }
}

int main(void){
    kara_init();
    u64 p = P50;
    __m512i vp = _mm512_set1_epi64((long long)p);
    __m512i vpn = _mm512_set1_epi64((long long)((1ull << 52) - p));
    __m512i vp2 = _mm512_set1_epi64((long long)(2 * p));
    __m512i vJ, vP = _mm512_set1_epi64((long long)p);
    __m512i vM = _mm512_set1_epi64((long long)M52);
    {   u64 pi = 1;
        for(int i = 0; i < 6; ++i) pi *= 2 - p * pi;
        vJ = _mm512_set1_epi64((long long)((0 - pi) & M52));
    }

    /* ---- validate convs ---- */
    {
        _Alignas(64) u64 A[16], Bv[16], S[16], ref[16];
        int bad = 0;
        for(int t = 0; t < 20000 && !bad; ++t){
            u64 ww0 = xr() % p, ww1 = xr() % p;
            for(int i = 0; i < 16; ++i){
                A[i] = (t & 1) ? xr() % (4 * p) : (4*p - 1 - (u64)i);
                Bv[i] = (t & 1) ? xr() % (4 * p) : (4*p - 17 - (u64)i);
            }
            memcpy(S, A, sizeof A);
            conv8_ref(ref, A, Bv, ww0);
            conv8_school(S, Bv, cc_of(ww0), vp, vpn, vp2, vJ, vP, vM);
            for(int l = 0; l < 8; ++l)
                if(S[l] % p != ref[l]){
                    printf("school FAIL t=%d l=%d\n", t, l); bad = 1;
                }
            /* school expects fb canonical: canonicalize for that path */
            for(int i = 0; i < 16; ++i) Bv[i] %= p;
            memcpy(S, A, sizeof A);
            conv8_ref(ref, A, Bv, ww0);
            conv8_ref(ref + 8, A + 8, Bv + 8, ww1);
            conv8_kara(S, Bv, cc_of(ww0), cc_of(ww1),
                       vp, vpn, vp2, vJ, vP, vM);
            for(int l = 0; l < 16; ++l)
                if(S[l] % p != ref[l]){
                    printf("kara FAIL t=%d l=%d got %llu want %llu\n", t, l,
                           (unsigned long long)(S[l] % p),
                           (unsigned long long)ref[l]);
                    bad = 1;
                }
        }
        if(bad) return 1;
        printf("conv8 validation OK (school + kara, lazy/adversarial)\n");
    }

    /* fix: school path above got lazy fb -- it shrinks internally?
     * (it does not: document that fb must be canonical, as in p30) */

    /* ---- bench ---- */
    enum { NV = 512 };                       /* 512 positions, L1 */
    static _Alignas(64) u64 FA[NV * 8], FB[NV * 8];
    static cc WT[NV];
    for(size_t i = 0; i < NV * 8; ++i){
        FA[i] = xr() % (4 * p);
        FB[i] = xr() % p;
    }
    for(int i = 0; i < NV; ++i) WT[i] = cc_of(xr() % p);
    long reps = 20000;
    double t_s = 1e30, t_k = 1e30, t_m = 1e30;
    for(int pass = 0; pass < 7; ++pass){
        double t0 = now();
        for(long r = 0; r < reps; ++r)
            for(size_t k = 0; k < NV; ++k)
                conv8_school(FA + 8 * k, FB + 8 * k, WT[k],
                             vp, vpn, vp2, vJ, vP, vM);
        double t = (now() - t0) / ((double)reps * NV) * 1e9;
        SINK = FA[xr() % (NV * 8)];
        if(t < t_s) t_s = t;
    }
    for(int pass = 0; pass < 7; ++pass){
        double t0 = now();
        for(long r = 0; r < reps; ++r)
            for(size_t k = 0; k + 2 <= NV; k += 2)
                conv8_kara(FA + 8 * k, FB + 8 * k, WT[k], WT[k + 1],
                           vp, vpn, vp2, vJ, vP, vM);
        double t = (now() - t0) / ((double)reps * NV) * 1e9;
        SINK = FA[xr() % (NV * 8)];
        if(t < t_k) t_k = t;
    }
    /* moth pass: m = NV vectors, 3 levels */
    {
        cc tw[7];
        for(int i = 0; i < 7; ++i) tw[i] = cc_of(xr() % p);
        for(int pass = 0; pass < 7; ++pass){
            double t0 = now();
            for(long r = 0; r < reps; ++r)
                fwd8(FA, NV, tw, vpn, vp2);
            double t = (now() - t0) / ((double)reps * NV * 8) * 1e9;
            SINK = FA[xr() % (NV * 8)];
            if(t < t_m) t_m = t;
        }
    }
    printf("conv8: school %.3f | kara %.3f ns/pos (ratio %.3f)\n",
           t_s, t_k, t_s / t_k);
    printf("fwd8 moth: %.4f ns/coeff/3-levels (%.4f ns/coeff/level)\n",
           t_m, t_m / 3);
    /* context: per-limb projections (T=80: 0.8 coeff/limb, x4 primes,
     * 2 transforms-ish + 1 inverse): conv/limb = ns_pos/8 * 0.8 * 4 */
    double cpl = (t_s < t_k ? t_s : t_k) / 8 * 0.8 * 4;
    printf("projection: conv ~%.2f ns/limb (p30: ~6.5-8)\n", cpl);
    return 0;
}
