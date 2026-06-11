/* Microbench for the PFA I/O layer (input stage + emit stage).
 * Variant A = current: per-residue vectors assembled by M^2 lane-residue
 * masked moves, then the radix-M butterfly.
 * Variant B = twiddle-fixup: butterfly runs directly on residue-relabeled
 * stripe vectors (y[b](l) = sum_s w^{b*r_s} w^{b*l} t_s(l)); the lane
 * dimension is handled by one constant per-lane cmul w^{b*l} per output.
 * M^2 merges -> M cmuls. Same identity transposed for the emit.
 *
 * Validation: fwd-input followed by emit is M * identity on the digit
 * stream (DFT then inverse-DFT across residues, exact for integer data),
 * so every variant must reproduce M * digits limb-exactly.
 *
 * Build: clang -O3 -march=native -std=c11 -I../include pfa_io_bench.c -lm
 */
#define _POSIX_C_SOURCE 199309L
#include "fft16.h"
#include <stdio.h>
#include <time.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9*t.tv_nsec; }
static uint64_t rng = 42;
static uint64_t xr(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int cmpd(const void* x, const void* y){ double a = *(const double*)x, b = *(const double*)y; return (a > b) - (a < b); }
#define PASSES 9

/* per-lane fixup tables: wm[b] lanes l = e^{-2pi i b (l%M) / M} */
static _Alignas(64) double WT[7][16];
static void build_wt(uint32_t M){
    for(uint32_t b = 0; b < M; ++b)
        for(uint32_t l = 0; l < 8; ++l){
            double th = -2.0 * 3.14159265358979323846 * (double)(b * (l % M)) / (double)M;
            WT[b][l] = cos(th); WT[b][8 + l] = sin(th);
        }
}

/* ---- variant B input: relabeled butterfly + lane-twiddle fixup ---------- */
__attribute__((always_inline))
static inline void pfa_input_B(double* data, const uint64_t* src,
                               uint32_t n, uint32_t M, const f16_plan* pl){
    const uint64_t* s[7]; double* p[7]; uint32_t r[7];
    for(uint32_t b = 0; b < M; ++b){
        s[b] = src + (size_t)b * (n / 2u);
        p[b] = data + 2u * (size_t)n * b;
        r[b] = (uint32_t)(((uint64_t)b * n) % M);
    }
    const __m512i* IDX = pl->dec_idx;
    for(uint32_t i = 0; i < n / 16u; ++i){
        fcv t0[7], t1[7];
        for(uint32_t b = 0; b < M; ++b){
            f16_dec2(&t0[b], &t1[b], s[b], IDX);
            s[b] += 8;
        }
        for(int u = 0; u < 2; ++u){
            const fcv* t = u ? t1 : t0;
            fcv x[7], y[7];
            /* relabel: x[rho] = stripe with lane-0 residue rho */
            for(uint32_t b = 0; b < M; ++b){
                uint32_t rho = (r[b] + (uint32_t)(8 * u)) % M;
                x[rho] = t[b];
            }
            f16_pfa_bfly(x, y, M, 0);
            for(uint32_t b = 0; b < M; ++b){
                fcv w; w.re = _mm512_load_pd(WT[b]); w.im = _mm512_load_pd(WT[b] + 8);
                f_store(p[b] + 16 * u, f_mul(y[b], w));
            }
        }
        for(uint32_t b = 0; b < M; ++b){
            p[b] += 32;
            r[b] = (r[b] + 16u) % M;
        }
    }
}
static void pfa_input_B3(double* d, const uint64_t* s, uint32_t n, const f16_plan* pl){ pfa_input_B(d, s, n, 3, pl); }
static void pfa_input_B5(double* d, const uint64_t* s, uint32_t n, const f16_plan* pl){ pfa_input_B(d, s, n, 5, pl); }
static void pfa_input_B7(double* d, const uint64_t* s, uint32_t n, const f16_plan* pl){ pfa_input_B(d, s, n, 7, pl); }

/* ---- variant B emit: lane-twiddle prefix + relabeled inverse butterfly -- */
__attribute__((always_inline))
static inline int pfa_emit_B(uint64_t* rp, double* data, uint32_t n,
                             uint32_t M, const f16_plan* pl){
    (void)pl;
    const double* br[7]; uint64_t* r[7]; uint32_t phi[7];
    f16_chain ch[7];
    const size_t limbs_b = (size_t)n / 2u;
    for(uint32_t b = 0; b < M; ++b){
        br[b]  = data + 2u * (size_t)n * b;
        r[b]   = rp + (size_t)b * limbs_b;
        phi[b] = (uint32_t)(((uint64_t)b * n) % M);
        ch[b].prev_hi = _mm512_setzero_si512();
        ch[b].cin = 0;
    }
    _Alignas(64) double stage[7 * 2 * 16];
    for(uint32_t t = 0; t < n / 8u; t += 2){
        for(int u = 0; u < 2; ++u){
            fcv x[7], v[7];
            for(uint32_t b = 0; b < M; ++b){
                fcv w; w.re = _mm512_load_pd(WT[b]); w.im = _mm512_load_pd(WT[b] + 8);
                x[b] = f_mulc(f_load(br[b]), w);
                br[b] += 16;
            }
            f16_pfa_bfly(x, v, M, 1);
            /* region b output = v[phi_b] (pure relabel, no merges) */
            for(uint32_t b = 0; b < M; ++b){
                uint32_t ph = (phi[b] + (uint32_t)(8 * u)) % M;
                f_store(stage + 32u * b + 16u * (uint32_t)u, v[ph]);
            }
        }
        for(uint32_t b = 0; b < M; ++b){
            f16_lohi q = f16_pack2(f_load(stage + 32u * b),
                                   f_load(stage + 32u * b + 16u));
            _mm512_storeu_si512((void*)r[b], f16_chain_step(&ch[b], q));
            r[b] += 8;
            phi[b] = (phi[b] + 16u) % M;
        }
    }
    for(uint32_t b = 0; b + 1u < M; ++b){
        uint64_t hi7;
        memcpy(&hi7, (const char*)&ch[b].prev_hi + 56, 8);
        unsigned __int128 c = (unsigned __int128)hi7 + ch[b].cin;
        uint64_t* q = rp + (size_t)(b + 1u) * limbs_b;
        uint64_t* qe = rp + (size_t)M * limbs_b;
        while(c && q < qe){ c += *q; *q++ = (uint64_t)c; c >>= 64; }
        if(c) return 0;
    }
    return 1;
}
static int pfa_emit_B3(uint64_t* r, double* d, uint32_t n, const f16_plan* pl){ return pfa_emit_B(r, d, n, 3, pl); }
static int pfa_emit_B5(uint64_t* r, double* d, uint32_t n, const f16_plan* pl){ return pfa_emit_B(r, d, n, 5, pl); }
static int pfa_emit_B7(uint64_t* r, double* d, uint32_t n, const f16_plan* pl){ return pfa_emit_B(r, d, n, 7, pl); }

typedef void (*infn)(double*, const uint64_t*, uint32_t, const f16_plan*);
typedef int  (*emfn)(uint64_t*, double*, uint32_t, const f16_plan*);

static void inA3(double* d, const uint64_t* s, uint32_t n, const f16_plan* pl){ f16_pfa_input_impl(d, s, n, 3, pl); }
static void inA5(double* d, const uint64_t* s, uint32_t n, const f16_plan* pl){ f16_pfa_input_impl(d, s, n, 5, pl); }
static void inA7(double* d, const uint64_t* s, uint32_t n, const f16_plan* pl){ f16_pfa_input_impl(d, s, n, 7, pl); }
static int emA3(uint64_t* r, double* d, uint32_t n, const f16_plan* pl){ return f16_pfa_emit_impl(r, d, n, 3, pl); }
static int emA5(uint64_t* r, double* d, uint32_t n, const f16_plan* pl){ return f16_pfa_emit_impl(r, d, n, 5, pl); }
static int emA7(uint64_t* r, double* d, uint32_t n, const f16_plan* pl){ return f16_pfa_emit_impl(r, d, n, 7, pl); }

int main(void){
    f16_plan* pl = &f16_tls;
    printf("M,branch,inA,inB,emA,emB  (ns per full-N complex)\n");
    const uint32_t Ms[3] = { 3, 5, 7 };
    const uint32_t brs[2] = { 256, 8192 };
    infn inA[3] = { inA3, inA5, inA7 }, inB[3] = { pfa_input_B3, pfa_input_B5, pfa_input_B7 };
    emfn emA[3] = { emA3, emA5, emA7 }, emB[3] = { pfa_emit_B3, pfa_emit_B5, pfa_emit_B7 };

    for(int mi = 0; mi < 3; ++mi){
        uint32_t M = Ms[mi];
        build_wt(M);
        for(int bi = 0; bi < 2; ++bi){
            uint32_t n = brs[bi], nfull = M * n;
            f16_plan_ensure(pl, n, nfull);
            size_t limbs = nfull / 2u;
            for(size_t i = 0; i < limbs; ++i) pl->pada[i] = xr();
            pl->pada[limbs - 1] = 0;     /* M*digit must not carry out the top */
            uint64_t* refr = pl->padb;   /* borrow as reference output buf */

            /* validation: input variant -> emit variant == M * digits.
             * M * digit stream as limbs: digits are u16; M*d < 2^19 so the
             * packed reference is digit-wise M*d via u32 spill-free math. */
            int bad = 0;
            for(int iv = 0; iv < 2 && !bad; ++iv)
                for(int ev = 0; ev < 2 && !bad; ++ev){
                    (iv ? inB[mi] : inA[mi])(pl->da, pl->pada, n, pl);
                    if(!(ev ? emB[mi] : emA[mi])(pl->padr, pl->da, n, pl)){
                        printf("M=%u br=%u iv=%d ev=%d carry fail\n", M, n, iv, ev);
                        bad = 1; break;
                    }
                    /* reference: per digit d -> M*d, repacked */
                    unsigned __int128 c = 0;
                    for(size_t i = 0; i < limbs; ++i){
                        uint64_t L = pl->pada[i];
                        unsigned __int128 v = c;
                        v += (unsigned __int128)(uint64_t)(M * (L & 0xFFFFu));
                        v += (unsigned __int128)(uint64_t)(M * ((L >> 16) & 0xFFFFu)) << 16;
                        v += (unsigned __int128)(uint64_t)(M * ((L >> 32) & 0xFFFFu)) << 32;
                        v += (unsigned __int128)(uint64_t)(M * ((L >> 48) & 0xFFFFu)) << 48;
                        refr[i] = (uint64_t)v;
                        c = v >> 64;
                    }
                    for(size_t i = 0; i < limbs; ++i)
                        if(pl->padr[i] != refr[i]){
                            printf("MISMATCH M=%u br=%u iv=%d ev=%d limb=%zu\n",
                                   M, n, iv, ev, i);
                            bad = 1; break;
                        }
                }
            if(bad) return 1;

            long reps = (long)(2000000 / nfull) + 1;
            double ta[4][PASSES];
            for(int p = 0; p < PASSES; ++p){
                double t0;
                t0 = now(); for(long q = 0; q < reps; ++q) inA[mi](pl->da, pl->pada, n, pl);
                ta[0][p] = (now() - t0) / reps;
                t0 = now(); for(long q = 0; q < reps; ++q) inB[mi](pl->da, pl->pada, n, pl);
                ta[1][p] = (now() - t0) / reps;
                t0 = now(); for(long q = 0; q < reps; ++q) emA[mi](pl->padr, pl->da, n, pl);
                ta[2][p] = (now() - t0) / reps;
                t0 = now(); for(long q = 0; q < reps; ++q) emB[mi](pl->padr, pl->da, n, pl);
                ta[3][p] = (now() - t0) / reps;
            }
            double med[4];
            for(int k = 0; k < 4; ++k){
                qsort(ta[k], PASSES, 8, cmpd);
                med[k] = ta[k][PASSES / 2] / nfull * 1e9;
            }
            printf("%u,%u,%.4f,%.4f,%.4f,%.4f\n", M, n, med[0], med[1], med[2], med[3]);
        }
    }
    return 0;
}

/* ---- variant C: B + single-tile grain for M=7 -------------------------- */
__attribute__((always_inline))
static inline void pfa_input_C7i(double* data, const uint64_t* src,
                                 uint32_t n, const f16_plan* pl){
    const uint32_t M = 7;
    const uint64_t* s[7]; double* p[7]; uint32_t r[7];
    for(uint32_t b = 0; b < M; ++b){
        s[b] = src + (size_t)b * (n / 2u);
        p[b] = data + 2u * (size_t)n * b;
        r[b] = (uint32_t)(((uint64_t)b * n) % M);
    }
    const __m512i* IDX = pl->dec_idx;
    for(uint32_t i = 0; i < n / 8u; ++i){
        fcv t[7], x[7], y[7];
        for(uint32_t b = 0; b < M; ++b){
            __m512i raw = _mm512_castsi256_si512(
                _mm256_loadu_si256((const __m256i*)s[b]));
            t[b].re = f16_dec1(raw, IDX[0]);
            t[b].im = f16_dec1(raw, IDX[1]);
            s[b] += 4;
        }
        for(uint32_t b = 0; b < M; ++b) x[r[b]] = t[b];
        f16_pfa_bfly(x, y, M, 0);
        for(uint32_t b = 0; b < M; ++b){
            fcv w; w.re = _mm512_load_pd(WT[b]); w.im = _mm512_load_pd(WT[b] + 8);
            f_store(p[b], f_mul(y[b], w));
            p[b] += 16;
            r[b] = (r[b] + 8u) % M;
        }
    }
}
void pfa_io_bench_extra(void);
void pfa_io_bench_extra(void){}
