// p50x4 I/O kernels: implementation + benchmark, parameterized by
// trunk size T in {80, 84, 88} (user: the machinery should support a
// trunk-size LADDER -- caps 2^40 / 2^32 / 2^24 limbs, smaller
// transforms for everything that fits).
//
// INPUT  limbs -> 4 residue arrays: per trunk, extract the lo
//        (T-52 bits) and hi (52 bits) PIECES with table-driven
//        permutex2var gathers + vpshrdvq funnels (pieces <= 52 bits
//        always span <= 2 limbs, any T); residue =
//        REDC52(hi * C) + lo with C = 2^T mod p (= hi*2^(T-52) + lo).
// OUTPUT 4 residue arrays -> carried limbs, in TRUNK RADIX:
//        (a) step-major batched Garner digits (Barrett52),
//        (b) compose V = sum v_k*W_k in radix-2^52 IFMA columns,
//        (c) normalize, (d) regroup 52 -> T into 3 trunk words
//        (lo64, hi(T-64)) b0,b1,b2, (e) delayed-stream add + SWAR
//        radix-2^T carry chains, then the only limb-touching step:
//        trunk->limb regroup via shrdv | sllv (sllv shift >= 64
//        auto-zeroes the absent second source).
//
// Build: clang -O3 -march=native -std=c11 p50_io_bench.c -lm \
//          -o /tmp/p50_io_bench
#define _POSIX_C_SOURCE 199309L
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef unsigned __int128 u128;
typedef uint64_t u64;
typedef uint32_t u32;

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static u64 rngs = 42;
static u64 xr(void){ rngs ^= rngs << 13; rngs ^= rngs >> 7; rngs ^= rngs << 17; return rngs; }
static volatile u64 SINK;

#define NP 4
#define M52 ((1ull << 52) - 1)
static const u64 PR[NP] = { 1025844348715009ull, 1072023837081601ull,
                            1086317488242689ull, 1108307720798209ull };

/* ---- scalar helpers ---- */
static u64 mulm(u64 a, u64 b, u64 p){ return (u64)((u128)a * b % p); }
static u64 powm(u64 a, u64 e, u64 p){
    u64 r = 1;
    for(; e; e >>= 1, a = mulm(a, a, p))
        if(e & 1) r = mulm(r, a, p);
    return r;
}
static u64 invm(u64 a, u64 p){ return powm(a, p - 2, p); }
static u64 rec52(u64 c, u64 p){ return (u64)(((u128)c << 52) / p); }

/* ---- IFMA primitives ---- */
static inline __m512i b52(__m512i a, __m512i vc, __m512i vrec, __m512i vpn){
    __m512i z = _mm512_setzero_si512();
    __m512i q = _mm512_madd52hi_epu64(z, a, vrec);
    __m512i s = _mm512_madd52lo_epu64(z, a, vc);
    s = _mm512_madd52lo_epu64(s, q, vpn);
    return _mm512_and_si512(s, _mm512_set1_epi64((long long)M52));
}
static inline __m512i shr1p(__m512i x, __m512i vp){     /* [0,2p)->[0,p) */
    return _mm512_min_epu64(x, _mm512_sub_epi64(x, vp));
}

/* ================= per-T plan ================= */
typedef struct {
    int T, LW;                   /* trunk bits, lo width = T-52 */
    int BT, BL, nv;              /* extraction block: BT trunks/BL limbs */
    int vbase[2];
    _Alignas(64) u64 loidx[2][8], losh[2][8];
    _Alignas(64) u64 hiidx[2][8], hish[2][8];
    int OT, OL, onv;             /* regroup superblock: OT trunks/OL limbs */
    int obase[24];
    _Alignas(64) u64 ouidx[24][8], osh[24][8], olsh[24][8];
    unsigned char oswp[24];      /* lanes with s >= 64: funnel from (hi,0)
                                  * (vpshrdvq counts are mod 64) */
    /* per-prime input constants */
    u64 C[NP], Crec_unused[NP];
    u64 J[NP];
    /* Garner constants */
    u64 ppm[NP][NP], ppm_rec[NP][NP];
    u64 inv[NP], inv_rec[NP];
    /* compose weight 52-bit words: W1(1), W2(2), W3(3) */
    u64 w1[1], w2[2], w3[3];
} tplan;

static void tplan_build(tplan *tp, int T){
    tp->T = T;
    tp->LW = T - 52;
    /* extraction block: smallest multiple of 8 trunks with whole limbs */
    int bt = 8;
    while((bt * T) % 64) bt += 8;
    tp->BT = bt; tp->BL = bt * T / 64; tp->nv = bt / 8;
    for(int v = 0; v < tp->nv; ++v){
        tp->vbase[v] = (T * 8 * v) >> 6;
        for(int l = 0; l < 8; ++l){
            long long bp = (long long)T * (8*v + l);
            tp->loidx[v][l] = (u64)((bp >> 6) - tp->vbase[v]);
            tp->losh[v][l]  = (u64)(bp & 63);
            bp += tp->LW;
            tp->hiidx[v][l] = (u64)((bp >> 6) - tp->vbase[v]);
            tp->hish[v][l]  = (u64)(bp & 63);
        }
    }
    /* output regroup superblock: limbs multiple of 8 */
    int ol = tp->BL;
    while(ol % 8) ol += tp->BL;
    tp->OL = ol; tp->OT = ol * 64 / T; tp->onv = ol / 8;
    for(int ov = 0; ov < tp->onv; ++ov){
        tp->obase[ov] = (int)(((long long)64 * 8 * ov) / T);
        tp->oswp[ov] = 0;
        for(int l = 0; l < 8; ++l){
            long long L = 8*ov + l;
            long long u = (64*L) / T;
            long long s = 64*L - (long long)T*u;
            tp->ouidx[ov][l] = (u64)(u - tp->obase[ov]);
            tp->osh[ov][l]   = (u64)s;
            tp->olsh[ov][l]  = (u64)(T - s);     /* >= 64 -> sllv zero */
            if(s >= 64) tp->oswp[ov] |= (unsigned char)(1u << l);
        }
    }
    for(int q = 0; q < NP; ++q){
        u64 p = PR[q];
        tp->C[q] = (u64)(((u128)1 << T) % p);    /* T <= 88: shift via u128 ok? */
        u64 pi = 1;
        for(int i = 0; i < 6; ++i) pi *= 2 - p * pi;
        tp->J[q] = (0 - pi) & M52;
    }
    for(int i = 1; i < NP; ++i){
        u64 p = PR[i], pp = 1;
        for(int k = 0; k < i; ++k){
            tp->ppm[i][k] = pp;
            tp->ppm_rec[i][k] = rec52(pp, p);
            pp = mulm(pp, PR[k] % p, p);
        }
        tp->inv[i] = invm(pp, p);
        tp->inv_rec[i] = rec52(tp->inv[i], p);
    }
    u128 w = PR[0];
    tp->w1[0] = (u64)(w & M52);                  /* p0 < 2^50: 1 word */
    w = (u128)PR[0] * PR[1];
    tp->w2[0] = (u64)(w & M52); tp->w2[1] = (u64)(w >> 52);
    /* W3 = p0*p1*p2 < 2^150: 3 words */
    {   u128 lo = (u128)PR[0] * PR[1];
        /* W3 = lo * p2: split */
        u64 a0 = (u64)(lo & M52), a1 = (u64)((lo >> 52) & M52),
            a2 = (u64)(lo >> 104);
        u128 c0 = (u128)a0 * PR[2];
        u128 c1 = (u128)a1 * PR[2] + (u64)(c0 >> 52);
        u128 c2 = (u128)a2 * PR[2] + (u64)(c1 >> 52);
        tp->w3[0] = (u64)(c0 & M52);
        tp->w3[1] = (u64)(c1 & M52);
        tp->w3[2] = (u64)(c2 & M52);             /* < 2^52 (150-104=46) */
    }
}

/* ================= INPUT kernel ================= */
static void p50_input(u64 *out[NP], const u64 *a, size_t ntr,
                      const tplan *tp){
    /* caller guarantees a[] readable for BL-limb blocks (padded) */
    const __m512i vM52 = _mm512_set1_epi64((long long)M52);
    const __m512i vMLO = _mm512_set1_epi64((long long)((1ull << tp->LW) - 1));
    __m512i vC[NP], vJ[NP], vP[NP];
    for(int q = 0; q < NP; ++q){
        vC[q] = _mm512_set1_epi64((long long)tp->C[q]);
        vJ[q] = _mm512_set1_epi64((long long)tp->J[q]);
        vP[q] = _mm512_set1_epi64((long long)PR[q]);
    }
    for(size_t t0 = 0; t0 < ntr; t0 += (size_t)tp->BT){
        size_t lb = t0 * (size_t)tp->T / 64;
        for(int v = 0; v < tp->nv; ++v){
            const u64 *base = a + lb + (size_t)tp->vbase[v];
            __m512i L0 = _mm512_loadu_si512(base);
            __m512i L1 = _mm512_loadu_si512(base + 8);
            __m512i iA, sA, A, B, hi, lo;
            iA = _mm512_load_si512(tp->loidx[v]);
            sA = _mm512_load_si512(tp->losh[v]);
            A = _mm512_permutex2var_epi64(L0, iA, L1);
            B = _mm512_permutex2var_epi64(L0,
                    _mm512_add_epi64(iA, _mm512_set1_epi64(1)), L1);
            lo = _mm512_and_si512(_mm512_shrdv_epi64(A, B, sA), vMLO);
            iA = _mm512_load_si512(tp->hiidx[v]);
            sA = _mm512_load_si512(tp->hish[v]);
            A = _mm512_permutex2var_epi64(L0, iA, L1);
            B = _mm512_permutex2var_epi64(L0,
                    _mm512_add_epi64(iA, _mm512_set1_epi64(1)), L1);
            hi = _mm512_and_si512(_mm512_shrdv_epi64(A, B, sA), vM52);
            size_t ot = t0 + (size_t)8 * v;
            for(int q = 0; q < NP; ++q){
                /* REDC52(hi*C) + lo, one fold to [0,2p)... two to [0,p) */
                __m512i z = _mm512_setzero_si512();
                __m512i xl = _mm512_madd52lo_epu64(z, hi, vC[q]);
                __m512i xh = _mm512_madd52hi_epu64(z, hi, vC[q]);
                __m512i xm = _mm512_and_si512(xl, vM52);
                __m512i m = _mm512_and_si512(
                    _mm512_madd52lo_epu64(z, xm, vJ[q]), vM52);
                __m512i r = _mm512_madd52hi_epu64(xh, m, vP[q]);
                __mmask8 cy = _mm512_test_epi64_mask(xm, vM52);
                r = _mm512_mask_add_epi64(r, cy, r, _mm512_set1_epi64(1));
                r = _mm512_add_epi64(r, lo);          /* < 2p + 2^LW */
                r = shr1p(r, vP[q]);                  /* < ~p + eps */
                r = shr1p(r, vP[q]);                  /* canonical */
                _mm512_storeu_si512(out[q] + ot, r);
            }
        }
    }
}

/* ================= OUTPUT kernel ================= */
/* chain state for the radix-2^T SWAR carry */
typedef struct { unsigned cin; } cst;
static inline void chainT(__m512i *lo, __m512i *hi, __m512i alo, __m512i ahi,
                          cst *cs, __m512i LIM){
    __m512i slo = _mm512_add_epi64(*lo, alo);
    __mmask8 lc = _mm512_cmplt_epu64_mask(slo, alo);
    __m512i shi = _mm512_add_epi64(_mm512_add_epi64(*hi, ahi),
                                   _mm512_maskz_set1_epi64(lc, 1));
    __m512i LM1 = _mm512_sub_epi64(LIM, _mm512_set1_epi64(1));
    __mmask8 g  = _mm512_cmpge_epu64_mask(shi, LIM);
    __mmask8 plo = _mm512_cmpeq_epi64_mask(slo, _mm512_set1_epi64(-1));
    __mmask8 pr = plo & _mm512_cmpeq_epi64_mask(shi, LM1);
    unsigned cn = (unsigned)pr + (((unsigned)g << 1) | cs->cin);
    unsigned cy = cn ^ (unsigned)pr;
    slo = _mm512_mask_add_epi64(slo, (__mmask8)cy, slo,
                                _mm512_set1_epi64(1));
    __mmask8 bump = (__mmask8)cy & plo;          /* lo wrapped */
    shi = _mm512_mask_add_epi64(shi, bump, shi, _mm512_set1_epi64(1));
    __mmask8 outs = g | ((__mmask8)cy & pr);
    shi = _mm512_mask_sub_epi64(shi, outs, shi, LIM);
    cs->cin = (cn >> 8) & 1u;
    *lo = slo; *hi = shi;
}

static void p50_output(u64 *out, u64 *const r[NP], size_t ntr,
                       const tplan *tp, u64 *scratch){
    const int T = tp->T;
    const __m512i vM52 = _mm512_set1_epi64((long long)M52);
    /* temp streams, 2 lanes of zero padding in front for the delays */
    size_t st = ntr + 64;
    u64 *B0l = scratch,     *B0h = B0l + st;
    u64 *B1l = B0h + st,    *B1h = B1l + st;
    u64 *B2  = B1h + st;
    u64 *Tl  = B2 + st,     *Th = Tl + st;
    memset(B0l, 0, 8 * 8); memset(B0h, 0, 8 * 8);
    memset(B1l, 0, 8 * 8); memset(B1h, 0, 8 * 8); memset(B2, 0, 8 * 8);

    /* pass 1: digits + compose + normalize + regroup -> b streams */
    __m512i vp[NP], vpn[NP], vp2[NP];
    for(int q = 0; q < NP; ++q){
        vp[q]  = _mm512_set1_epi64((long long)PR[q]);
        vpn[q] = _mm512_set1_epi64((long long)((1ull << 52) - PR[q]));
        vp2[q] = _mm512_set1_epi64((long long)(2 * PR[q]));
    }
    for(size_t j = 0; j < ntr; j += 8){
        __m512i vd[NP];
        vd[0] = shr1p(_mm512_loadu_si512(r[0] + j), vp[0]);
        for(int i = 1; i < NP; ++i){
            __m512i acc = vd[0];
            for(int k = 1; k < i; ++k){
                __m512i u = b52(vd[k],
                        _mm512_set1_epi64((long long)tp->ppm[i][k]),
                        _mm512_set1_epi64((long long)tp->ppm_rec[i][k]),
                        vpn[i]);
                acc = _mm512_add_epi64(acc, u);
                acc = _mm512_min_epu64(acc,
                        _mm512_sub_epi64(acc, vp2[i]));
            }
            __m512i d = _mm512_sub_epi64(
                _mm512_add_epi64(_mm512_loadu_si512(r[i] + j), vp2[i]),
                acc);
            __m512i v = b52(d,
                    _mm512_set1_epi64((long long)tp->inv[i]),
                    _mm512_set1_epi64((long long)tp->inv_rec[i]), vpn[i]);
            vd[i] = shr1p(v, vp[i]);
        }
        /* compose: radix-52 columns */
        __m512i c0 = vd[0], c1, c2, c3;
        c0 = _mm512_madd52lo_epu64(c0, vd[1],
                 _mm512_set1_epi64((long long)tp->w1[0]));
        c1 = _mm512_madd52hi_epu64(_mm512_setzero_si512(), vd[1],
                 _mm512_set1_epi64((long long)tp->w1[0]));
        c0 = _mm512_madd52lo_epu64(c0, vd[2],
                 _mm512_set1_epi64((long long)tp->w2[0]));
        c1 = _mm512_madd52hi_epu64(c1, vd[2],
                 _mm512_set1_epi64((long long)tp->w2[0]));
        c1 = _mm512_madd52lo_epu64(c1, vd[2],
                 _mm512_set1_epi64((long long)tp->w2[1]));
        c2 = _mm512_madd52hi_epu64(_mm512_setzero_si512(), vd[2],
                 _mm512_set1_epi64((long long)tp->w2[1]));
        c0 = _mm512_madd52lo_epu64(c0, vd[3],
                 _mm512_set1_epi64((long long)tp->w3[0]));
        c1 = _mm512_madd52hi_epu64(c1, vd[3],
                 _mm512_set1_epi64((long long)tp->w3[0]));
        c1 = _mm512_madd52lo_epu64(c1, vd[3],
                 _mm512_set1_epi64((long long)tp->w3[1]));
        c2 = _mm512_madd52hi_epu64(c2, vd[3],
                 _mm512_set1_epi64((long long)tp->w3[1]));
        c2 = _mm512_madd52lo_epu64(c2, vd[3],
                 _mm512_set1_epi64((long long)tp->w3[2]));
        c3 = _mm512_madd52hi_epu64(_mm512_setzero_si512(), vd[3],
                 _mm512_set1_epi64((long long)tp->w3[2]));
        /* normalize to exact radix-52 */
        c1 = _mm512_add_epi64(c1, _mm512_srli_epi64(c0, 52));
        c0 = _mm512_and_si512(c0, vM52);
        c2 = _mm512_add_epi64(c2, _mm512_srli_epi64(c1, 52));
        c1 = _mm512_and_si512(c1, vM52);
        c3 = _mm512_add_epi64(c3, _mm512_srli_epi64(c2, 52));
        c2 = _mm512_and_si512(c2, vM52);
        /* regroup 52 -> T (trunk words as lo64 + hi(T-64)) */
        const int Thw = T - 64;
        __m512i b0l = _mm512_or_si512(c0, _mm512_slli_epi64(c1, 52));
        __m512i b0h = _mm512_and_si512(_mm512_srli_epi64(c1, 12),
                          _mm512_set1_epi64((1ll << Thw) - 1));
        __m512i b1l = _mm512_or_si512(_mm512_srli_epi64(c1, T - 52),
                          _mm512_slli_epi64(c2, 104 - T));
        __m512i b1h = _mm512_and_si512(
                          _mm512_or_si512(_mm512_srli_epi64(c2, T - 40),
                                          _mm512_slli_epi64(c3, 92 - T)),
                          _mm512_set1_epi64((1ll << Thw) - 1));
        __m512i b2 = _mm512_srli_epi64(c3, 2 * T - 156);
        _mm512_storeu_si512(B0l + 8 + j, b0l);
        _mm512_storeu_si512(B0h + 8 + j, b0h);
        _mm512_storeu_si512(B1l + 8 + j, b1l);
        _mm512_storeu_si512(B1h + 8 + j, b1h);
        _mm512_storeu_si512(B2  + 8 + j, b2);
    }
    /* pass 2: delayed adds + radix-2^T carry chains */
    {
        __m512i LIM = _mm512_set1_epi64(1ll << (T - 64));
        cst cA = {0}, cB = {0};
        for(size_t j = 0; j < ntr; j += 8){
            __m512i lo = _mm512_loadu_si512(B0l + 8 + j);
            __m512i hi = _mm512_loadu_si512(B0h + 8 + j);
            chainT(&lo, &hi, _mm512_loadu_si512(B1l + 7 + j),
                             _mm512_loadu_si512(B1h + 7 + j), &cA, LIM);
            chainT(&lo, &hi, _mm512_loadu_si512(B2 + 6 + j),
                             _mm512_setzero_si512(), &cB, LIM);
            _mm512_storeu_si512(Tl + j, lo);
            _mm512_storeu_si512(Th + j, hi);
        }
        /* stream tails beyond ntr are guaranteed zero by the caller's
         * trunk count headroom (bench allocates +2 coefficients) */
    }
    /* pass 3: trunks -> limbs (generic funnel regroup) */
    {
        size_t nl = ntr * (size_t)T / 64;
        for(size_t L0i = 0; L0i < nl; L0i += (size_t)tp->OL){
            size_t tb = L0i * 64 / (size_t)T;
            for(int ov = 0; ov < tp->onv; ++ov){
                size_t base = tb + (size_t)tp->obase[ov];
                __m512i Wl0 = _mm512_loadu_si512(Tl + base);
                __m512i Wl1 = _mm512_loadu_si512(Tl + base + 8);
                __m512i Wh0 = _mm512_loadu_si512(Th + base);
                __m512i Wh1 = _mm512_loadu_si512(Th + base + 8);
                __m512i iu = _mm512_load_si512(tp->ouidx[ov]);
                __m512i sh = _mm512_load_si512(tp->osh[ov]);
                __m512i ls = _mm512_load_si512(tp->olsh[ov]);
                __m512i la = _mm512_permutex2var_epi64(Wl0, iu, Wl1);
                __m512i ha = _mm512_permutex2var_epi64(Wh0, iu, Wh1);
                __m512i lb = _mm512_permutex2var_epi64(Wl0,
                    _mm512_add_epi64(iu, _mm512_set1_epi64(1)), Wl1);
                /* s >= 64 lanes: trunk>>s = hi>>(s-64): funnel (hi, 0) */
                __mmask8 sw = (__mmask8)tp->oswp[ov];
                __m512i fa = _mm512_mask_blend_epi64(sw, la, ha);
                __m512i fb = _mm512_maskz_mov_epi64((__mmask8)~sw, ha);
                __m512i limb = _mm512_or_si512(
                    _mm512_shrdv_epi64(fa, fb, sh),
                    _mm512_sllv_epi64(lb, ls));
                _mm512_storeu_si512(out + L0i + 8 * (size_t)ov, limb);
            }
        }
    }
}

/* add v (u128) at bit position bp of the limb array (ripple carry) */
static void add_at(u64 *ref, long long bp, u128 v){
    size_t q = (size_t)(bp >> 6);
    int s = (int)(bp & 63);
    u64 lo64 = (u64)v, hi64 = (u64)(v >> 64);
    u64 w[3];
    if(s){
        w[0] = lo64 << s;
        w[1] = (lo64 >> (64 - s)) | (hi64 << s);
        w[2] = hi64 >> (64 - s);
    }else{ w[0] = lo64; w[1] = hi64; w[2] = 0; }
    u64 c = 0;
    for(int i = 0; i < 3; ++i){
        u64 o = ref[q + i];
        u64 t1 = o + w[i];
        u64 c1 = t1 < o;
        ref[q + i] = t1 + c;
        c = c1 | (ref[q + i] < t1);
    }
    for(size_t z = q + 3; c; ++z){ ref[z] += 1; c = (ref[z] == 0); }
}

/* ================= scalar references ================= */
static u64 ref_trunk(const u64 *a, size_t nlimb, int T, size_t t, int dlt,
                     int width){
    long long bp = (long long)T * (long long)t + dlt;
    size_t q = (size_t)(bp >> 6);
    int s = (int)(bp & 63);
    u128 v = a[q];
    if(q + 1 < nlimb + 16) v |= (u128)a[q + 1] << 64;
    return (u64)((v >> s) & (((u128)1 << width) - 1));
}

int main(void){
    int Ts[3] = { 80, 84, 88 };
    for(int ti = 0; ti < 3; ++ti){
        int T = Ts[ti];
        static tplan tp;
        tplan_build(&tp, T);
        printf("==== T = %d (cycle %d trunks / %d limbs) ====\n",
               T, tp.BT, tp.BL);

        enum { NTR_MAX = 1 << 21 };
        size_t ntr = 1 << 16;                    /* validation size */
        size_t nlimb = ntr * (size_t)T / 64;
        u64 *limbs = aligned_alloc(64, (nlimb + 64) * 8);
        u64 *out[NP];
        for(int q = 0; q < NP; ++q)
            out[q] = aligned_alloc(64, (ntr + 64) * 8);
        for(size_t i = 0; i < nlimb + 64; ++i) limbs[i] = xr();

        /* ---- input validation ---- */
        p50_input(out, limbs, ntr, &tp);
        int bad = 0;
        for(size_t t = 0; t < ntr && bad < 3; ++t){
            u64 lo = ref_trunk(limbs, nlimb, T, t, 0, tp.LW);
            u64 hi = ref_trunk(limbs, nlimb, T, t, tp.LW, 52);
            for(int q = 0; q < NP; ++q){
                u64 ref = ((u128)hi * (((u128)1 << tp.LW) % PR[q])
                           + lo) % PR[q];
                /* hi*2^(T-52)+lo mod p */
                ref = (u64)(((u128)hi * ((((u128)1) << tp.LW) % PR[q]) + lo)
                            % PR[q]);
                if(out[q][t] % PR[q] != ref ||
                   out[q][t] >= PR[q]){
                    printf("  input FAIL t=%zu q=%d got %llu want %llu\n",
                           t, q, (unsigned long long)out[q][t],
                           (unsigned long long)ref);
                    ++bad;
                }
            }
        }
        if(!bad) printf("  input validation OK (%zu trunks)\n", ntr);

        /* ---- output validation: random V_j, incl. all-max ---- */
        /* V_j < 2^(2T+8); reference accumulation in a limb array */
        size_t ovtr = 1 << 12;
        size_t ovl = ovtr * (size_t)T / 64;
        u64 *rr[NP];
        for(int q = 0; q < NP; ++q)
            rr[q] = aligned_alloc(64, (ovtr + 64) * 8);
        u64 *ref = calloc(ovl + 64, 8);
        u64 *got = aligned_alloc(64, (ovl + 64) * 8);
        u64 *scr = aligned_alloc(64, 7 * (ovtr + 64) * 8);
        memset(got, 0, (ovl + 64) * 8);
        for(int advmax = 0; advmax < 2; ++advmax){
            memset(ref, 0, (ovl + 64) * 8);
            for(size_t j = 0; j < ovtr; ++j){
                /* V = v2*2^(2T) + v1*2^T + v0, each < 2^T-ish; cap 2T+8 */
                u64 v0 = advmax ? ~0ull : xr();
                u64 v1 = advmax ? ~0ull : xr();
                u64 v2 = advmax ? 255 : (xr() & 255);
                u128 Vlo = ((u128)(v1 & ((1ull << (T - 64)) - 1)) << 64) | v0;
                /* V as u128 low + (high bits) -- build V < 2^(2T+8):
                 * V = Vlo + (vmid << T) + (v2 << 2T), Vlo < 2^T */
                u64 vmid = v1;                   /* < 2^64 <= 2^T */
                /* residues */
                for(int q = 0; q < NP; ++q){
                    u64 p = PR[q];
                    u128 m = (((u128)1) << T) % p;
                    u64 r0 = (u64)(Vlo % p);
                    u64 r1 = mulm((u64)(((u128)vmid) % p), (u64)(m % p), p);
                    u64 r2 = mulm(mulm((u64)v2, (u64)(m % p), p),
                                  (u64)(m % p), p);
                    rr[q][j] = (r0 + r1 + r2) % p;
                }
                /* reference accumulate V * 2^(T j) limbwise */
                add_at(ref, (long long)T * (long long)j, Vlo);
                add_at(ref, (long long)T * (long long)j + T, (u128)vmid);
                add_at(ref, (long long)T * (long long)j + 2*T, (u128)v2);
            }
            p50_output(got, (u64 *const *)rr, ovtr, &tp, scr);
            size_t cmp = ovl;                    /* whole-limb region */
            int b2 = 0;
            for(size_t i = 0; i < cmp && b2 < 3; ++i)
                if(got[i] != ref[i]){
                    printf("  output FAIL %s limb %zu got %016llx want %016llx\n",
                           advmax ? "(all-max)" : "(random)", i,
                           (unsigned long long)got[i],
                           (unsigned long long)ref[i]);
                    ++b2;
                }
            if(!b2) printf("  output validation OK (%s, %zu coeffs)\n",
                           advmax ? "all-max" : "random", ovtr);
            bad += b2;
        }
        if(bad){ printf("  T=%d FAILED\n", T); return 1; }

        /* ---- perf ---- */
        for(int big = 0; big < 2; ++big){
            size_t n = big ? (size_t)1 << 20 : (size_t)1 << 13;
            size_t nl = n * (size_t)T / 64;
            u64 *L = aligned_alloc(64, (nl + 64) * 8);
            u64 *O[NP], *S, *G;
            for(int q = 0; q < NP; ++q)
                O[q] = aligned_alloc(64, (n + 64) * 8);
            S = aligned_alloc(64, 7 * (n + 64) * 8);
            G = aligned_alloc(64, (nl + 64) * 8);
            for(size_t i = 0; i < nl + 64; ++i) L[i] = xr();
            long reps = big ? 8 : 800;
            double t_in = 1e30, t_out = 1e30;
            for(int pass = 0; pass < 5; ++pass){
                double t0 = now();
                for(long rep = 0; rep < reps; ++rep)
                    p50_input(O, L, n, &tp);
                double t = (now() - t0) / ((double)reps * nl) * 1e9;
                SINK = O[0][xr() % n];
                if(t < t_in) t_in = t;
            }
            for(size_t i = 0; i < n + 64; ++i)
                for(int q = 0; q < NP; ++q) O[q][i] = xr() % PR[q];
            for(int pass = 0; pass < 5; ++pass){
                double t0 = now();
                for(long rep = 0; rep < reps; ++rep)
                    p50_output(G, (u64 *const *)O, n, &tp, S);
                double t = (now() - t0) / ((double)reps * nl) * 1e9;
                SINK = G[xr() % nl];
                if(t < t_out) t_out = t;
            }
            printf("  %s: input %.3f ns/limb | output %.3f ns/limb\n",
                   big ? "DRAM (1M trunks)" : "L2   (8K trunks) ",
                   t_in, t_out);
            free(L); free(S); free(G);
            for(int q = 0; q < NP; ++q) free(O[q]);
        }
        free(limbs); free(ref); free(got); free(scr);
        for(int q = 0; q < NP; ++q){ free(out[q]); free(rr[q]); }
    }
    printf("ALL OK\n");
    return 0;
}
