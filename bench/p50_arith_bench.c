// p50 arithmetic shootout: IFMA52 (vpmadd52) vs DOUBLE-FMA (FLINT
// fft_small style) for ~50-bit-prime NTT kernels, on the three units
// that decide the p50 engine:
//   T1 mulmod stream   r[i] = a[i] * w mod p      (block twiddle)
//   T2 butterfly stream (a, b) -> (a + wb, a - wb) lazy ranges
//   T3 dot8 (conv-style) out = sum_{k<8} a_k * b_k mod p
// T3 is IFMA's home turf: madd52lo/hi accumulate 104-bit columns
// in-instruction; doubles must REDUCE EVERY PRODUCT before a float
// add (the h,l product splits are exact but their sums are not).
//
// Double mulmod (Shoup precomputation w_pinv = w/p, round-to-nearest):
//   h = a*w; l = fma(a, w, -h)           (exact double-double product)
//   q = round(a * w_pinv)                (roundscale OR magic-constant)
//   r = fnmadd(q, p, h) + l              in (-p, p), exact integer
// IFMA Barrett52 (the p30 lemma at R = 2^52, any a < 2^52):
//   q = madd52hi(0, a, rec); r = (madd52lo(madd52lo(0, a, c), q,
//   2^52 - p)) & M52         in [0, 2p)
//
// Build: clang -O3 -march=native -std=c11 p50_arith_bench.c -lm \
//          -o /tmp/p50_arith_bench
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

#define P50 1108307720798209ull          /* 63*2^44 + 1, lg 49.977 */
#define M52 ((1ull << 52) - 1)

static u64 mulm(u64 a, u64 b){ return (u64)((u128)a * b % P50); }

/* ---- IFMA Barrett52 ------------------------------------------------ */
typedef struct { u64 c, rec; } cc52;
static cc52 cc52_of(u64 c){
    cc52 t = { c, (u64)(((u128)c << 52) / P50) };
    return t;
}
static inline __m512i b52_mul(__m512i a, __m512i vc, __m512i vrec,
                              __m512i vpneg){
    __m512i z = _mm512_setzero_si512();
    __m512i q = _mm512_madd52hi_epu64(z, a, vrec);
    __m512i s = _mm512_madd52lo_epu64(z, a, vc);
    s = _mm512_madd52lo_epu64(s, q, vpneg);
    return _mm512_and_si512(s, _mm512_set1_epi64((long long)M52));
}
/* lazy fold [0,4p) -> [0,2p) */
static inline __m512i i52_sh2(__m512i x, __m512i vp2){
    return _mm512_min_epu64(x, _mm512_sub_epi64(x, vp2));
}

/* ---- double-FMA mulmod (two rounding flavors) ---------------------- */
static inline __m512d d_round(__m512d x){
    return _mm512_roundscale_pd(x, _MM_FROUND_TO_NEAREST_INT |
                                   _MM_FROUND_NO_EXC);
}
static inline __m512d d_round_magic(__m512d x){
    const __m512d MG = _mm512_set1_pd(0x1.8p52);
    return _mm512_sub_pd(_mm512_add_pd(x, MG), MG);
}
static inline __m512d d_mulmod_rs(__m512d a, __m512d w, __m512d wpinv,
                                  __m512d p){
    __m512d h = _mm512_mul_pd(a, w);
    __m512d l = _mm512_fmsub_pd(a, w, h);
    __m512d q = d_round(_mm512_mul_pd(a, wpinv));
    __m512d t = _mm512_fnmadd_pd(q, p, h);
    return _mm512_add_pd(t, l);              /* (-p, p) */
}
static inline __m512d d_mulmod_mg(__m512d a, __m512d w, __m512d wpinv,
                                  __m512d p){
    __m512d h = _mm512_mul_pd(a, w);
    __m512d l = _mm512_fmsub_pd(a, w, h);
    __m512d q = d_round_magic(_mm512_mul_pd(a, wpinv));
    __m512d t = _mm512_fnmadd_pd(q, p, h);
    return _mm512_add_pd(t, l);
}
/* fold to (-p, p) (FLINT reduce_pm1n): for the unmultiplied operand */
static inline __m512d d_redpm(__m512d a, __m512d p, __m512d pinv){
    __m512d q = d_round_magic(_mm512_mul_pd(a, pinv));
    return _mm512_fnmadd_pd(q, p, a);
}

int main(void){
    const u64 p = P50;
    u64 wv = xr() % p;
    cc52 w52 = cc52_of(wv);
    __m512i vc = _mm512_set1_epi64((long long)w52.c);
    __m512i vrec = _mm512_set1_epi64((long long)w52.rec);
    __m512i vpneg = _mm512_set1_epi64((long long)((1ull << 52) - p));
    __m512i vp2i = _mm512_set1_epi64((long long)(2 * p));
    __m512d vw = _mm512_set1_pd((double)wv);
    __m512d vwp = _mm512_set1_pd((double)wv / (double)p);
    __m512d vp = _mm512_set1_pd((double)p);
    __m512d vpinv = _mm512_set1_pd(1.0 / (double)p);

    /* ---- correctness: both mulmods vs scalar, incl. edges ---- */
    {
        _Alignas(64) u64 ai[8];
        _Alignas(64) double ad[8];
        _Alignas(64) u64 ri[8];
        _Alignas(64) double rd[8];
        int bad = 0;
        for(int t = 0; t < 100000 && !bad; ++t){
            for(int l = 0; l < 8; ++l){
                u64 a = (t < 4) ? (2*p - 1 - (u64)l) :        /* edge */
                        (t & 1) ? xr() % (2*p) : xr() % p;
                ai[l] = a;
                ad[l] = (double)(int64_t)(a >= p ? a - p : a); /* canon */
            }
            __m512i r1 = b52_mul(_mm512_load_si512(ai), vc, vrec, vpneg);
            _mm512_store_si512(ri, r1);
            __m512d r2 = d_mulmod_mg(_mm512_load_pd(ad), vw, vwp, vp);
            _mm512_store_pd(rd, r2);
            for(int l = 0; l < 8; ++l){
                u64 ref = mulm(ai[l] % p, wv);
                if(ri[l] % p != ref){ printf("IFMA FAIL\n"); bad = 1; }
                long long ir = (long long)rd[l];
                u64 rr = ir < 0 ? (u64)(ir + (long long)p) : (u64)ir;
                if(rr % p != mulm((u64)((int64_t)ad[l]), wv)){
                    printf("DBL FAIL a=%f got %lld\n", ad[l], ir); bad = 1;
                }
            }
        }
        if(bad) return 1;
        printf("mulmod validation OK (IFMA + double, edges incl.)\n");
    }

    enum { N = 1 << 12 };                    /* 32KB per array: L1 */
    static _Alignas(64) u64 IA[N], IB[N];
    static _Alignas(64) double DA[N], DB[N];
    for(int i = 0; i < N; ++i){
        u64 v = xr() % p;
        IA[i] = v; DA[i] = (double)v;
        v = xr() % p;
        IB[i] = v; DB[i] = (double)v;
    }
    long reps = 40000;

    /* ---- T1: mulmod stream (PING-PONG: values evolve across reps,
     * so no rep's work is algebraically removable; ranges stable:
     * IFMA [0,2p) -> any a < 2^52 ok; double (-p,p) -> |a| <= 2p ok) */
    static _Alignas(64) u64 IC[N];
    static _Alignas(64) double DC[N];
    double t_i = 1e30, t_dr = 1e30, t_dm = 1e30;
    for(int pass = 0; pass < 7; ++pass){
        u64 *s = IA, *d = IC;
        double t0 = now();
        for(long r = 0; r < reps; ++r){
            for(int i = 0; i < N; i += 8)
                _mm512_store_si512(d + i,
                    b52_mul(_mm512_load_si512(s + i), vc, vrec, vpneg));
            u64 *t_ = s; s = d; d = t_;
        }
        double t = (now() - t0) / ((double)reps * N) * 1e9;
        SINK = s[xr() % N];
        if(t < t_i) t_i = t;
    }
    for(int pass = 0; pass < 7; ++pass){
        double *s = DA, *d = DC;
        double t0 = now();
        for(long r = 0; r < reps; ++r){
            for(int i = 0; i < N; i += 8)
                _mm512_store_pd(d + i,
                    d_mulmod_rs(_mm512_load_pd(s + i), vw, vwp, vp));
            double *t_ = s; s = d; d = t_;
        }
        double t = (now() - t0) / ((double)reps * N) * 1e9;
        SINK = (u64)s[xr() % N];
        if(t < t_dr) t_dr = t;
    }
    for(int pass = 0; pass < 7; ++pass){
        double *s = DA, *d = DC;
        double t0 = now();
        for(long r = 0; r < reps; ++r){
            for(int i = 0; i < N; i += 8)
                _mm512_store_pd(d + i,
                    d_mulmod_mg(_mm512_load_pd(s + i), vw, vwp, vp));
            double *t_ = s; s = d; d = t_;
        }
        double t = (now() - t0) / ((double)reps * N) * 1e9;
        SINK = (u64)s[xr() % N];
        if(t < t_dm) t_dm = t;
    }
    printf("T1 mulmod   : IFMA %.4f | dbl(rndscale) %.4f | dbl(magic) %.4f ns/elem\n",
           t_i / 1, t_dr / 1, t_dm / 1);

    /* ---- T2: butterfly stream, ping-pong ---- */
    double b_i = 1e30, b_d = 1e30;
    for(int pass = 0; pass < 7; ++pass){
        u64 *s = IA, *d = IC;
        double t0 = now();
        for(long r = 0; r < reps; ++r){
            for(int i = 0; i < N/2; i += 8){
                __m512i a = _mm512_load_si512(s + i);
                __m512i b = _mm512_load_si512(s + N/2 + i);
                __m512i as = i52_sh2(a, vp2i);
                __m512i wb = b52_mul(b, vc, vrec, vpneg);
                _mm512_store_si512(d + i, _mm512_add_epi64(as, wb));
                _mm512_store_si512(d + N/2 + i,
                    _mm512_add_epi64(as, _mm512_sub_epi64(vp2i, wb)));
            }
            u64 *t_ = s; s = d; d = t_;
        }
        double t = (now() - t0) / ((double)reps * N) * 1e9;
        SINK = s[xr() % N];
        if(t < b_i) b_i = t;
    }
    for(int pass = 0; pass < 7; ++pass){
        double *s = DA, *d = DC;
        double t0 = now();
        for(long r = 0; r < reps; ++r){
            for(int i = 0; i < N/2; i += 8){
                __m512d a = _mm512_load_pd(s + i);
                __m512d b = _mm512_load_pd(s + N/2 + i);
                __m512d ar = d_redpm(a, vp, vpinv);
                __m512d wb = d_mulmod_mg(b, vw, vwp, vp);
                _mm512_store_pd(d + i, _mm512_add_pd(ar, wb));
                _mm512_store_pd(d + N/2 + i, _mm512_sub_pd(ar, wb));
            }
            double *t_ = s; s = d; d = t_;
        }
        double t = (now() - t0) / ((double)reps * N) * 1e9;
        SINK = (u64)s[xr() % N];
        if(t < b_d) b_d = t;
    }
    printf("T2 butterfly: IFMA %.4f | dbl(magic) %.4f ns/pair-elem\n",
           b_i / 1, b_d / 1);

    /* ---- T3: dot8 (conv-style), output feeds back into row r&7 ---- */
    u64 pinv52 = 1;
    for(int it = 0; it < 6; ++it) pinv52 *= 2 - p * pinv52;
    __m512i vJ52 = _mm512_set1_epi64((long long)((0 - pinv52) & M52));
    __m512i vPi  = _mm512_set1_epi64((long long)p);
    __m512i vM52 = _mm512_set1_epi64((long long)M52);
    double d_i = 1e30, d_d = 1e30;
    for(int pass = 0; pass < 7; ++pass){
        double t0 = now();
        for(long r = 0; r < reps/8; ++r){
            u64 *out = IA + (r & 7) * (N/8);
            for(int i = 0; i < N/8; i += 8){
                __m512i lo = _mm512_setzero_si512();
                __m512i hi = _mm512_setzero_si512();
                for(int k = 0; k < 8; ++k){
                    __m512i a = _mm512_load_si512(IA + k * (N/8) + i);
                    __m512i b = _mm512_load_si512(IB + k * (N/8) + i);
                    lo = _mm512_madd52lo_epu64(lo, a, b);
                    hi = _mm512_madd52hi_epu64(hi, a, b);
                }
                __m512i lom = _mm512_and_si512(lo, vM52);
                __m512i m = _mm512_and_si512(
                    _mm512_madd52lo_epu64(_mm512_setzero_si512(),
                                          lom, vJ52), vM52);
                __m512i t = _mm512_madd52hi_epu64(
                    _mm512_add_epi64(hi, _mm512_srli_epi64(lo, 52)),
                    m, vPi);
                _mm512_store_si512(out + i, t);
            }
        }
        double t = (now() - t0) / ((double)(reps/8) * (N/8)) * 1e9;
        SINK = IA[xr() % N];
        if(t < d_i) d_i = t;
    }
    for(int pass = 0; pass < 7; ++pass){
        double t0 = now();
        for(long r = 0; r < reps/8; ++r){
            double *out = DA + (r & 7) * (N/8);
            for(int i = 0; i < N/8; i += 8){
                __m512d acc = _mm512_setzero_pd();
                for(int k = 0; k < 8; ++k){
                    __m512d a = _mm512_load_pd(DA + k * (N/8) + i);
                    __m512d b = _mm512_load_pd(DB + k * (N/8) + i);
                    __m512d h = _mm512_mul_pd(a, b);
                    __m512d l = _mm512_fmsub_pd(a, b, h);
                    __m512d q = d_round_magic(_mm512_mul_pd(h, vpinv));
                    __m512d tt = _mm512_fnmadd_pd(q, vp, h);
                    acc = _mm512_add_pd(acc, _mm512_add_pd(tt, l));
                }
                acc = d_redpm(acc, vp, vpinv);
                _mm512_store_pd(out + i, acc);
            }
        }
        double t = (now() - t0) / ((double)(reps/8) * (N/8)) * 1e9;
        SINK = (u64)DA[xr() % N];
        if(t < d_d) d_d = t;
    }
    printf("T3 dot8     : IFMA %.4f | dbl %.4f ns/output-elem (8 products)\n",
           d_i / 1, d_d / 1);

    return 0;
}
