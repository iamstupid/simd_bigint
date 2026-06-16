// p30 I/O kernels: design, implementation and microbench (docs §4).
//
// INPUT  p30_input_fused:   u64 limbs -> 5 residue arrays in [0,2p),
//        one pass, per prime one fused REDC:
//          t = hi32*R2 + lo32*R1   (R1 = 2^32 mod p, R2 = R1^2 mod p,
//                                   t < 2*2^32*p)
//          REDC(t) = (t + (lo32(t)*J mod 2^32)*p) >> 32  in [0, 3p)
//          one min-sub -> [0, 2p)            (J = -p^-1 mod 2^32)
//        (also a prime-at-a-time variant to measure the fusion win)
//
// OUTPUT p30_output_crt:    5 residue arrays ([0,2p) accepted) ->
//        carried u64 array (zlen+2 limbs), one pass:
//        1. Garner digits, ascending primes, Barrett-by-constant with
//           NO pre-reductions (any-u32 property): v0 = shrink(r0);
//           acc_i = v0 (+ barrett(v_k, prodmod)) with min-subs;
//           v_i = shrink(barrett(r_i - acc_i + 2p_i, invprod))
//        2. compose c = v0 + v1*W1 + v2*W2 + v3*W3 + v4*W4 via 32-bit
//           chunked partial products into 64-bit column accumulators
//           (10 vpmuludq per 8 coefficients), recombined into three
//           limb-aligned streams (L0, L1, L2): c_j sits at limbs j..j+2
//           (64-bit chunking).
//        3. out[j] = L0_j + L1_{j-1} + L2_{j-2}: two chained SWAR
//           carry steps (pq16 q_chain shape), no scalar u128 loop.
//
// Build: clang -O3 -march=native -std=c11 p30_io_bench.c -o p30_io_bench
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

#define NP 5
static const u64 PR[NP] = { 918552577, 935329793, 943718401,
                            985661441, 998244353 };

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static u64 rngs = 42;
static u64 xr(void){ rngs ^= rngs << 13; rngs ^= rngs >> 7; rngs ^= rngs << 17; return rngs; }

/* ---- constants ----------------------------------------------------- */
typedef struct {
    u32 p, p2, J, R1, R2;
} in_const;

typedef struct {
    /* Garner step i (1..4): digit = shrink(barrett(r_i - acc + 2p_i, INV)) */
    u32 inv[NP], inv_rec[NP];          /* (prod_{j<i} p_j)^-1 mod p_i      */
    u32 ppm[NP][NP], ppm_rec[NP][NP];  /* prod_{j<k} p_j mod p_i, k=1..i-1 */
    /* compose chunks: W_i = prod_{j<i} p_j as 32-bit chunks               */
    u32 w1c[1], w2c[2], w3c[3], w4c[4];
} out_const;

static in_const  IC[NP];
static out_const OC;

static u32 mulm32(u64 a, u64 b, u64 p){ return (u32)(a * b % p); }
static u32 powm32(u64 a, u64 e, u64 p){
    u64 r = 1;
    for(; e; e >>= 1, a = a * a % p)
        if(e & 1) r = r * a % p;
    return (u32)r;
}
static u32 rec_of(u32 c, u32 p){ return (u32)(((u128)c << 32) / p); }

static void init_consts(void){
    for(int q = 0; q < NP; ++q){
        u32 p = (u32)PR[q];
        u32 pinv = 1;
        for(int i = 0; i < 5; ++i) pinv *= 2 - p * pinv;
        IC[q].p = p;
        IC[q].p2 = 2 * p;
        IC[q].J = 0 - pinv;
        IC[q].R1 = (u32)(((u64)1 << 32) % p);
        IC[q].R2 = mulm32(IC[q].R1, IC[q].R1, p);
    }
    for(int i = 1; i < NP; ++i){
        u64 pi = PR[i], pp = 1;
        for(int k = 0; k < i; ++k){
            OC.ppm[i][k] = (u32)pp;
            OC.ppm_rec[i][k] = rec_of((u32)pp, (u32)pi);
            pp = pp * (PR[k] % pi) % pi;
        }
        OC.inv[i] = powm32(pp, pi - 2, pi);
        OC.inv_rec[i] = rec_of(OC.inv[i], (u32)pi);
    }
    u128 w = PR[0];
    OC.w1c[0] = (u32)w;
    w *= PR[1];
    OC.w2c[0] = (u32)w; OC.w2c[1] = (u32)(w >> 32);
    w *= PR[2];
    OC.w3c[0] = (u32)w; OC.w3c[1] = (u32)(w >> 32); OC.w3c[2] = (u32)(w >> 64);
    w *= PR[3];
    OC.w4c[0] = (u32)w; OC.w4c[1] = (u32)(w >> 32);
    OC.w4c[2] = (u32)(w >> 64); OC.w4c[3] = (u32)(w >> 96);
}

/* ==================== INPUT KERNEL ================================= */
/* 8 limbs (one zmm of u64) -> 8 residues in the low dwords of u64 lanes */
__attribute__((always_inline))
static inline __m512i in_redc8(__m512i x, __m512i R1, __m512i R2,
                               __m512i J, __m512i p){
    __m512i hi = _mm512_srli_epi64(x, 32);
    __m512i t  = _mm512_add_epi64(_mm512_mul_epu32(x, R1),
                                  _mm512_mul_epu32(hi, R2));
    __m512i m  = _mm512_mul_epu32(t, J);
    return _mm512_srli_epi64(_mm512_add_epi64(t, _mm512_mul_epu32(m, p)), 32);
}
/* pack two 8x(u64 low-dword) results into 16 sequential u32 lanes */
static inline __m512i pack_idx(void){
    static const u32 idx[16] = { 0, 2, 4, 6, 8, 10, 12, 14,
                                 16, 18, 20, 22, 24, 26, 28, 30 };
    return _mm512_loadu_si512(idx);
}

/* fused: one pass over the limbs, all 5 primes per iteration */
static void p30_input_fused(u32 *restrict out[NP],
                            const u64 *restrict a, size_t n){
    const __m512i PK = pack_idx();
    size_t i = 0;
    for(; i + 16 <= n; i += 16){
        __m512i x0 = _mm512_loadu_si512(a + i);
        __m512i x1 = _mm512_loadu_si512(a + i + 8);
        for(int q = 0; q < NP; ++q){
            __m512i R1 = _mm512_set1_epi64(IC[q].R1);
            __m512i R2 = _mm512_set1_epi64(IC[q].R2);
            __m512i J  = _mm512_set1_epi64(IC[q].J);
            __m512i p  = _mm512_set1_epi64(IC[q].p);
            __m512i r0 = in_redc8(x0, R1, R2, J, p);
            __m512i r1 = in_redc8(x1, R1, R2, J, p);
            __m512i r  = _mm512_permutex2var_epi32(r0, PK, r1);
            __m512i p2 = _mm512_set1_epi32((int)IC[q].p2);
            r = _mm512_min_epu32(r, _mm512_sub_epi32(r, p2));
            _mm512_storeu_si512(out[q] + i, r);
        }
    }
    for(; i < n; ++i)
        for(int q = 0; q < NP; ++q)
            out[q][i] = (u32)(a[i] % PR[q]);
}

/* prime-at-a-time: 5 passes (A/B reference for the fusion win) */
static void p30_input_split(u32 *restrict out[NP],
                            const u64 *restrict a, size_t n){
    const __m512i PK = pack_idx();
    for(int q = 0; q < NP; ++q){
        __m512i R1 = _mm512_set1_epi64(IC[q].R1);
        __m512i R2 = _mm512_set1_epi64(IC[q].R2);
        __m512i J  = _mm512_set1_epi64(IC[q].J);
        __m512i p  = _mm512_set1_epi64(IC[q].p);
        __m512i p2 = _mm512_set1_epi32((int)IC[q].p2);
        size_t i = 0;
        for(; i + 16 <= n; i += 16){
            __m512i r0 = in_redc8(_mm512_loadu_si512(a + i), R1, R2, J, p);
            __m512i r1 = in_redc8(_mm512_loadu_si512(a + i + 8), R1, R2, J, p);
            __m512i r  = _mm512_permutex2var_epi32(r0, PK, r1);
            r = _mm512_min_epu32(r, _mm512_sub_epi32(r, p2));
            _mm512_storeu_si512(out[q] + i, r);
        }
        for(; i < n; ++i) out[q][i] = (u32)(a[i] % PR[q]);
    }
}

/* ==================== OUTPUT KERNEL ================================ */
static inline __m512i bar_noredc(__m512i a, u32 c, u32 c_rec, u32 mod){
    const __m512i vrec = _mm512_set1_epi32((int)c_rec);
    __m512i pe = _mm512_mul_epu32(a, vrec);
    __m512i po = _mm512_mul_epu32(_mm512_srli_epi64(a, 32), vrec);
    __m512i q  = _mm512_mask_mov_epi32(_mm512_srli_epi64(pe, 32),
                                       (__mmask16)0xAAAA, po);
    return _mm512_sub_epi32(_mm512_mullo_epi32(a, _mm512_set1_epi32((int)c)),
                            _mm512_mullo_epi32(q, _mm512_set1_epi32((int)mod)));
}
static inline __m512i shrink32(__m512i x, u32 m){
    const __m512i vm = _mm512_set1_epi32((int)m);
    return _mm512_min_epu32(x, _mm512_sub_epi32(x, vm));
}

/* SWAR carry-chain state (pq16 q_chain) */
typedef struct { __m512i prev; unsigned cin; } chain;
static inline __m512i chain_step(chain *c, __m512i lo, __m512i addend_prev_hi){
    /* sum = lo + (addend stream shifted in from previous vector) */
    __m512i his = _mm512_alignr_epi64(addend_prev_hi, c->prev, 7);
    __m512i sum = _mm512_add_epi64(lo, his);
    unsigned g  = _mm512_cmplt_epu64_mask(sum, his);
    unsigned pr = _mm512_cmpeq_epu64_mask(sum, _mm512_set1_epi64(-1));
    unsigned cn = pr + ((g << 1) | c->cin);
    unsigned cy = cn ^ pr;
    sum = _mm512_mask_add_epi64(sum, (__mmask8)cy, sum, _mm512_set1_epi64(1));
    c->cin = (cn >> 8) & 1u;
    c->prev = addend_prev_hi;
    return sum;
}

/* Garner digits for 16 coefficients: r[q] 16-lane u32 in [0,2p_q),
 * out v[q] canonical [0,p_q). */
__attribute__((always_inline))
static inline void garner16(__m512i v[NP], const __m512i r[NP]){
    v[0] = shrink32(r[0], IC[0].p);
    for(int i = 1; i < NP; ++i){
        u32 pi = IC[i].p, p2i = IC[i].p2;
        __m512i acc = v[0];                       /* < p0 < p_i */
        for(int k = 1; k < i; ++k){
            __m512i t = bar_noredc(v[k], OC.ppm[i][k], OC.ppm_rec[i][k], pi);
            acc = shrink32(_mm512_add_epi32(acc, t), p2i);   /* < 2p_i */
        }
        __m512i d = _mm512_sub_epi32(
            _mm512_add_epi32(r[i], _mm512_set1_epi32((int)p2i)), acc);
        __m512i t = bar_noredc(d, OC.inv[i], OC.inv_rec[i], pi);
        v[i] = shrink32(shrink32(t, p2i), pi);    /* [0,2p) -> [0,p) */
    }
}

/* compose 8 coefficients (u64-lane digit vectors, values < 2^30) into
 * three limb-aligned streams */
__attribute__((always_inline))
static inline void compose8(__m512i d[NP], __m512i *L0, __m512i *L1, __m512i *L2){
    __m512i col0 = _mm512_add_epi64(d[0],
        _mm512_mul_epu32(d[1], _mm512_set1_epi64(OC.w1c[0])));
    col0 = _mm512_add_epi64(col0,
        _mm512_mul_epu32(d[2], _mm512_set1_epi64(OC.w2c[0])));
    col0 = _mm512_add_epi64(col0,
        _mm512_mul_epu32(d[3], _mm512_set1_epi64(OC.w3c[0])));
    col0 = _mm512_add_epi64(col0,
        _mm512_mul_epu32(d[4], _mm512_set1_epi64(OC.w4c[0])));
    __m512i col1 = _mm512_add_epi64(
        _mm512_mul_epu32(d[2], _mm512_set1_epi64(OC.w2c[1])),
        _mm512_mul_epu32(d[3], _mm512_set1_epi64(OC.w3c[1])));
    col1 = _mm512_add_epi64(col1,
        _mm512_mul_epu32(d[4], _mm512_set1_epi64(OC.w4c[1])));
    __m512i col2 = _mm512_add_epi64(
        _mm512_mul_epu32(d[3], _mm512_set1_epi64(OC.w3c[2])),
        _mm512_mul_epu32(d[4], _mm512_set1_epi64(OC.w4c[2])));
    __m512i col3 = _mm512_mul_epu32(d[4], _mm512_set1_epi64(OC.w4c[3]));

    /* c = col0 + col1<<32 + col2<<64 + col3<<96  ->  (L0, L1, L2) */
    __m512i l0 = _mm512_add_epi64(col0, _mm512_slli_epi64(col1, 32));
    __mmask8 k0 = _mm512_cmplt_epu64_mask(l0, col0);
    __m512i t = _mm512_add_epi64(_mm512_srli_epi64(col1, 32), col2);
    t = _mm512_mask_add_epi64(t, k0, t, _mm512_set1_epi64(1));
    __m512i l1 = _mm512_add_epi64(t, _mm512_slli_epi64(col3, 32));
    __mmask8 k1 = _mm512_cmplt_epu64_mask(l1, t);
    __m512i l2 = _mm512_srli_epi64(col3, 32);
    l2 = _mm512_mask_add_epi64(l2, k1, l2, _mm512_set1_epi64(1));
    *L0 = l0; *L1 = l1; *L2 = l2;
}

/* out[0 .. zlen+2) = sum_j c_j * 2^64j, c_j from residues r[5][j] */
static void p30_output_crt(u64 *restrict out,
                           const u32 *restrict r[NP], size_t zlen){
    const __m512i ILO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i IHI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    const __m512i M32 = _mm512_set1_epi64(0xFFFFFFFFu);

    chain c1 = { _mm512_setzero_si512(), 0 };
    chain c2 = { _mm512_setzero_si512(), 0 };
    __m512i l2prev = _mm512_setzero_si512();   /* previous L2 vector */

    size_t j = 0;
    for(; j + 16 <= zlen; j += 16){
        __m512i rv[NP], v[NP];
        for(int q = 0; q < NP; ++q)
            rv[q] = _mm512_loadu_si512(r[q] + j);
        garner16(v, rv);

        /* split into even/odd u64-lane digit streams and compose */
        __m512i de[NP], dd[NP];
        for(int q = 0; q < NP; ++q){
            de[q] = _mm512_and_si512(v[q], M32);
            dd[q] = _mm512_srli_epi64(v[q], 32);
        }
        __m512i L0e, L1e, L2e, L0o, L1o, L2o;
        compose8(de, &L0e, &L1e, &L2e);
        compose8(dd, &L0o, &L1o, &L2o);
        /* interleave back to sequential coefficient order (2 vecs/stream) */
        __m512i L0a = _mm512_permutex2var_epi64(L0e, ILO, L0o);
        __m512i L0b = _mm512_permutex2var_epi64(L0e, IHI, L0o);
        __m512i L1a = _mm512_permutex2var_epi64(L1e, ILO, L1o);
        __m512i L1b = _mm512_permutex2var_epi64(L1e, IHI, L1o);
        __m512i L2a = _mm512_permutex2var_epi64(L2e, ILO, L2o);
        __m512i L2b = _mm512_permutex2var_epi64(L2e, IHI, L2o);

        /* out_j = L0_j + L1_{j-1} + L2_{j-2}: chain c1 delays its addend
           stream by one lane; for the two-lane delay of L2 feed chain c2
           an already one-lane-delayed L2 stream (alignr vs previous) */
        __m512i L2da = _mm512_alignr_epi64(L2a, l2prev, 7);
        __m512i L2db = _mm512_alignr_epi64(L2b, L2a, 7);
        l2prev = L2b;
        __m512i s;
        s = chain_step(&c1, L0a, L1a);
        s = chain_step(&c2, s, L2da);
        _mm512_storeu_si512(out + j, s);
        s = chain_step(&c1, L0b, L1b);
        s = chain_step(&c2, s, L2db);
        _mm512_storeu_si512(out + j + 8, s);
    }
    /* scalar pendings from the chain state:
       pend_L1   = L1_{j-1}  = c1.prev lane 7
       pend_L2_2 = L2_{j-2}  = c2.prev lane 7 (the delayed stream's lane 7)
       pend_L2_1 = L2_{j-1}  = l2prev lane 7                                */
    u64 lane7[8];
    _mm512_storeu_si512(lane7, c1.prev);
    u64 pend_L1 = lane7[7];
    _mm512_storeu_si512(lane7, c2.prev);
    u64 pend_L2_2 = lane7[7];
    _mm512_storeu_si512(lane7, l2prev);
    u64 pend_L2_1 = lane7[7];
    u128 carry = (u128)c1.cin + c2.cin;

    for(; j < zlen; ++j){
        /* scalar Garner + compose for the tail */
        u64 v[NP];
        v[0] = r[0][j] % PR[0];
        for(int i = 1; i < NP; ++i){
            u64 pi = PR[i], t = v[0] % pi;
            for(int k = 1; k < i; ++k)
                t = (t + v[k] * (u64)OC.ppm[i][k]) % pi;
            v[i] = (r[i][j] % pi + pi - t) % pi * OC.inv[i] % pi;
        }
        u128 x = v[4];
        x = x * PR[3] + v[3];
        x = x * PR[2] + v[2];
        x = x * PR[1] + v[1];
        u128 m0 = (u128)(u64)x * PR[0] + v[0];
        u128 m1 = (u128)(u64)(x >> 64) * PR[0] + (u64)(m0 >> 64);
        u128 s = (u128)(u64)m0 + pend_L1 + pend_L2_2 + carry;
        out[j] = (u64)s;
        carry = s >> 64;
        pend_L1 = (u64)m1;
        pend_L2_2 = pend_L2_1;
        pend_L2_1 = (u64)(m1 >> 64);
    }
    /* final two limbs */
    u128 s = (u128)pend_L1 + pend_L2_2 + carry;
    out[zlen] = (u64)s;
    out[zlen + 1] = (u64)((s >> 64) + pend_L2_1);
}

/* ---- batched output variant: step-major Garner through an L1 buffer.
 * The fused form spills (~30 live constants); here every inner loop
 * holds only its own constants:
 *   phase 1 (per digit step i): sweep the batch computing digit i from
 *     digits 0..i-1 in the buffer  -- ~10 broadcasts live
 *   phase 2: compose + interleave + carry chains over the batch        */
#define OB 256                       /* batch coefficients (5KB digits) */
static void p30_output_crt_batched(u64 *restrict out,
                                   const u32 *restrict r[NP], size_t zlen){
    const __m512i ILO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i IHI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    const __m512i M32 = _mm512_set1_epi64(0xFFFFFFFFu);
    _Alignas(64) static u32 vb[NP][OB];

    chain c1 = { _mm512_setzero_si512(), 0 };
    chain c2 = { _mm512_setzero_si512(), 0 };
    __m512i l2prev = _mm512_setzero_si512();

    size_t j = 0;
    for(; j + OB <= zlen; j += OB){
        /* digit 0 */
        for(int t = 0; t < OB; t += 16)
            _mm512_store_si512(vb[0] + t,
                shrink32(_mm512_loadu_si512(r[0] + j + t), IC[0].p));
        /* digit steps 1..4, step-major */
        for(int i = 1; i < NP; ++i){
            u32 pi = IC[i].p, p2i = IC[i].p2;
            for(int t = 0; t < OB; t += 16){
                __m512i acc = _mm512_load_si512(vb[0] + t);
                for(int k = 1; k < i; ++k){
                    __m512i u = bar_noredc(_mm512_load_si512(vb[k] + t),
                                           OC.ppm[i][k], OC.ppm_rec[i][k], pi);
                    acc = shrink32(_mm512_add_epi32(acc, u), p2i);
                }
                __m512i d = _mm512_sub_epi32(
                    _mm512_add_epi32(_mm512_loadu_si512(r[i] + j + t),
                                     _mm512_set1_epi32((int)p2i)), acc);
                __m512i v = bar_noredc(d, OC.inv[i], OC.inv_rec[i], pi);
                _mm512_store_si512(vb[i] + t, shrink32(v, pi));
            }
        }
        /* compose + chains */
        for(int t = 0; t < OB; t += 16){
            __m512i de[NP], dd[NP];
            for(int q = 0; q < NP; ++q){
                __m512i v = _mm512_load_si512(vb[q] + t);
                de[q] = _mm512_and_si512(v, M32);
                dd[q] = _mm512_srli_epi64(v, 32);
            }
            __m512i L0e, L1e, L2e, L0o, L1o, L2o;
            compose8(de, &L0e, &L1e, &L2e);
            compose8(dd, &L0o, &L1o, &L2o);
            __m512i L0a = _mm512_permutex2var_epi64(L0e, ILO, L0o);
            __m512i L0b = _mm512_permutex2var_epi64(L0e, IHI, L0o);
            __m512i L1a = _mm512_permutex2var_epi64(L1e, ILO, L1o);
            __m512i L1b = _mm512_permutex2var_epi64(L1e, IHI, L1o);
            __m512i L2a = _mm512_permutex2var_epi64(L2e, ILO, L2o);
            __m512i L2b = _mm512_permutex2var_epi64(L2e, IHI, L2o);
            __m512i L2da = _mm512_alignr_epi64(L2a, l2prev, 7);
            __m512i L2db = _mm512_alignr_epi64(L2b, L2a, 7);
            l2prev = L2b;
            __m512i s;
            s = chain_step(&c1, L0a, L1a);
            s = chain_step(&c2, s, L2da);
            _mm512_storeu_si512(out + j + t, s);
            s = chain_step(&c1, L0b, L1b);
            s = chain_step(&c2, s, L2db);
            _mm512_storeu_si512(out + j + t + 8, s);
        }
    }
    /* tail: same scalar path as the fused variant */
    u64 lane7[8];
    _mm512_storeu_si512(lane7, c1.prev);
    u64 pend_L1 = lane7[7];
    _mm512_storeu_si512(lane7, c2.prev);
    u64 pend_L2_2 = lane7[7];
    _mm512_storeu_si512(lane7, l2prev);
    u64 pend_L2_1 = lane7[7];
    u128 carry = (u128)c1.cin + c2.cin;
    for(; j < zlen; ++j){
        u64 v[NP];
        v[0] = r[0][j] % PR[0];
        for(int i = 1; i < NP; ++i){
            u64 pi = PR[i], t = v[0] % pi;
            for(int k = 1; k < i; ++k)
                t = (t + v[k] * (u64)OC.ppm[i][k]) % pi;
            v[i] = (r[i][j] % pi + pi - t) % pi * OC.inv[i] % pi;
        }
        u128 x = v[4];
        x = x * PR[3] + v[3];
        x = x * PR[2] + v[2];
        x = x * PR[1] + v[1];
        u128 m0 = (u128)(u64)x * PR[0] + v[0];
        u128 m1 = (u128)(u64)(x >> 64) * PR[0] + (u64)(m0 >> 64);
        u128 s = (u128)(u64)m0 + pend_L1 + pend_L2_2 + carry;
        out[j] = (u64)s;
        carry = s >> 64;
        pend_L1 = (u64)m1;
        pend_L2_2 = pend_L2_1;
        pend_L2_1 = (u64)(m1 >> 64);
    }
    u128 s = (u128)pend_L1 + pend_L2_2 + carry;
    out[zlen] = (u64)s;
    out[zlen + 1] = (u64)((s >> 64) + pend_L2_1);
}

/* ---- scalar references for validation ------------------------------ */
static void ref_input(u32 *out[NP], const u64 *a, size_t n){
    for(size_t i = 0; i < n; ++i)
        for(int q = 0; q < NP; ++q)
            out[q][i] = (u32)(a[i] % PR[q]);
}
static void ref_output(u64 *out, const u32 *r[NP], size_t zlen){
    memset(out, 0, (zlen + 2) * 8);
    for(size_t j = 0; j < zlen; ++j){
        u64 v[NP];
        u64 r0 = r[0][j] % PR[0];
        v[0] = r0;
        for(int i = 1; i < NP; ++i){
            u64 pi = PR[i], t = v[0] % pi;
            for(int k = 1; k < i; ++k)
                t = (t + v[k] * (u64)OC.ppm[i][k]) % pi;
            v[i] = (r[i][j] % pi + pi - t) % pi * OC.inv[i] % pi;
        }
        u128 x = v[4];
        x = x * PR[3] + v[3];
        x = x * PR[2] + v[2];
        x = x * PR[1] + v[1];
        u128 m0 = (u128)(u64)x * PR[0] + v[0];
        u128 m1 = (u128)(u64)(x >> 64) * PR[0] + (u64)(m0 >> 64);
        u64 c[3] = { (u64)m0, (u64)m1, (u64)(m1 >> 64) };
        u128 cy = 0;
        for(int k = 0; k < 3; ++k){
            cy += (u128)out[j + k] + c[k];
            out[j + k] = (u64)cy;
            cy >>= 64;
        }
        for(size_t t2 = j + 3; cy && t2 < zlen + 2; ++t2){
            cy += out[t2];
            out[t2] = (u64)cy;
            cy >>= 64;
        }
    }
}

/* ===================================================================== */
#define MAXN (1u << 22)
static u64  LIMBS[MAXN + 16];
static u32  RES[NP][MAXN + 64], RES2[NP][MAXN + 64];
static u64  OUT[MAXN + 64], OUTREF[MAXN + 64];

int main(void){
    init_consts();

    /* ---- validate input kernels ---- */
    {
        size_t n = 5000;
        for(size_t i = 0; i < n; ++i)
            LIMBS[i] = (i < 8) ? (i & 1 ? ~(u64)0 : 0)
                               : (i == 8 ? 0x8000000000000000ull : xr());
        u32 *o1[NP], *o2[NP];
        for(int q = 0; q < NP; ++q){ o1[q] = RES[q]; o2[q] = RES2[q]; }
        ref_input(o2, LIMBS, n);
        p30_input_fused(o1, LIMBS, n);
        for(int q = 0; q < NP; ++q)
            for(size_t i = 0; i < n; ++i){
                u32 r = o1[q][i];
                if(r >= IC[q].p2 || r % IC[q].p != o2[q][i]){
                    printf("INPUT FUSED FAIL q=%d i=%zu\n", q, i);
                    return 1;
                }
            }
        p30_input_split(o1, LIMBS, n);
        for(int q = 0; q < NP; ++q)
            for(size_t i = 0; i < n; ++i)
                if(o1[q][i] % IC[q].p != o2[q][i] || o1[q][i] >= IC[q].p2){
                    printf("INPUT SPLIT FAIL q=%d i=%zu\n", q, i);
                    return 1;
                }
        printf("input kernels: validation OK (%zu limbs incl. edges)\n", n);
    }

    /* ---- validate output kernel: ground-truth 150-bit coefficients ---- */
    {
        size_t zl = 4444;
        const u32 *rr[NP];
        for(int q = 0; q < NP; ++q) rr[q] = RES[q];
        for(int pat = 0; pat < 3; ++pat){
            for(size_t j = 0; j < zl; ++j){
                /* coefficient: random < P, or P-1 (max digits), or 0 */
                u64 v[NP];
                for(int q = 0; q < NP; ++q){
                    u64 base = pat == 1 ? PR[q] - 1 : pat == 2 ? 0 : xr() % PR[q];
                    /* present residues LAZY in [0,2p) half the time */
                    RES[q][j] = (u32)(base + ((xr() & 1) ? PR[q] : 0));
                    v[q] = base;
                    (void)v;
                }
            }
            ref_output(OUTREF, rr, zl);
            p30_output_crt(OUT, rr, zl);
            if(memcmp(OUT, OUTREF, (zl + 2) * 8)){
                size_t i = 0;
                while(OUT[i] == OUTREF[i]) ++i;
                printf("OUTPUT FAIL pat=%d limb %zu got %016llx want %016llx\n",
                       pat, i, (unsigned long long)OUT[i],
                       (unsigned long long)OUTREF[i]);
                return 1;
            }
        }
        /* odd lengths exercise the scalar tail + flush */
        for(size_t zl2 = 1; zl2 < 40; ++zl2){
            for(size_t j = 0; j < zl2; ++j)
                for(int q = 0; q < NP; ++q)
                    RES[q][j] = (u32)(xr() % IC[q].p2);
            ref_output(OUTREF, rr, zl2);
            p30_output_crt(OUT, rr, zl2);
            if(memcmp(OUT, OUTREF, (zl2 + 2) * 8)){
                printf("OUTPUT TAIL FAIL zl=%zu\n", zl2);
                return 1;
            }
        }
        for(size_t zl2 = 1; zl2 < 600; zl2 += 13){
            for(size_t j = 0; j < zl2; ++j)
                for(int q = 0; q < NP; ++q)
                    RES[q][j] = (u32)(xr() % IC[q].p2);
            ref_output(OUTREF, rr, zl2);
            p30_output_crt_batched(OUT, rr, zl2);
            if(memcmp(OUT, OUTREF, (zl2 + 2) * 8)){
                printf("OUTPUT BATCHED FAIL zl=%zu\n", zl2);
                return 1;
            }
        }
        printf("output batched: validation OK\n");
        printf("output kernel: validation OK (3 patterns x 4444 + tails 1..39)\n");
    }

    /* ---- microbench ---- */
    u32 *oarr[NP];
    const u32 *rarr[NP];
    for(int q = 0; q < NP; ++q){ oarr[q] = RES[q]; rarr[q] = RES[q]; }
    for(size_t i = 0; i < MAXN; ++i) LIMBS[i] = xr();

    printf("\n%-14s %10s %12s %12s %10s\n",
           "kernel", "n limbs", "ns/limb", "GB/s", "(traffic)");
    for(int k = 0; k < 4; ++k){
        const char *name = k == 0 ? "input fused" : k == 1 ? "input split"
                         : k == 2 ? "output crt" : "output batched";
        for(int lg = 12; lg <= 22; lg += 5){
            size_t n = (size_t)1 << lg;
            long reps = (long)((1u << 24) / n) + 1;
            double best = 1e30;
            for(int pass = 0; pass < 7; ++pass){
                double t0 = now();
                for(long t = 0; t < reps; ++t){
                    if(k == 0) p30_input_fused(oarr, LIMBS, n);
                    else if(k == 1) p30_input_split(oarr, LIMBS, n);
                    else if(k == 2) p30_output_crt(OUT, rarr, n);
                    else p30_output_crt_batched(OUT, rarr, n);
                }
                double dt = (now() - t0) / reps;
                if(dt < best) best = dt;
            }
            double bytes = k < 2 ? (8.0 + 20.0) * n : (20.0 + 8.0) * n;
            printf("%-14s %10zu %12.4f %12.2f %8.0fB/limb\n",
                   name, n, best / n * 1e9, bytes / best / 1e9, bytes / n);
        }
    }
    return 0;
}
