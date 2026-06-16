// Input-fusion traffic probe (docs §4 follow-up): is the all-prime
// BLOCK-COLUMN fused input pass (28 B/limb, first k1 levels absorbed)
// actually faster than the alternatives, given its strided limb reads
// and strided residue writes?
//
//   U  unfused:        linear input pass (28 B) + per-prime strided
//                      column RW pass (40 B)            = 68 B/limb
//   P  per-prime fused: 5 passes, each strided limb read + strided
//                      residue write                    = 60 B/limb
//   F  all-prime fused: one pass, strided limb read shared by all 5
//                      primes (block-column stays L1-hot), strided
//                      residue writes                   = 28 B/limb
//
// All variants run the same placeholder per-vector compute (the real
// REDC decode + a fixed butterfly-cost stand-in) so the comparison
// isolates the ACCESS PATTERN. Matrix view: M = n/16 vectors as
// 2^k1 rows x 2^k2 cols; column element stride = 2^k2 vectors.
// Buffers live on a THP-backed mmap (production scratch is THP).
// Build: clang -O3 -march=native -std=c11 p30_fuse_probe.c -o p30_fuse_probe
#define _GNU_SOURCE
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

typedef unsigned __int128 u128;
typedef uint64_t u64;
typedef uint32_t u32;

#define NP 5
static const u64 PR[NP] = { 918552577, 935329793, 943718401,
                            985661441, 998244353 };
static u32 R1c[NP], R2c[NP], Jc[NP], P2c[NP];

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static u64 rngs = 42;
static u64 xr(void){ rngs ^= rngs << 13; rngs ^= rngs >> 7; rngs ^= rngs << 17; return rngs; }

static void init_consts(void){
    for(int q = 0; q < NP; ++q){
        u32 p = (u32)PR[q], pinv = 1;
        for(int i = 0; i < 5; ++i) pinv *= 2 - p * pinv;
        Jc[q] = 0 - pinv;
        R1c[q] = (u32)(((u64)1 << 32) % p);
        R2c[q] = (u32)((u64)R1c[q] * R1c[q] % p);
        P2c[q] = 2 * p;
    }
}

__attribute__((always_inline))
static inline __m512i redc8(__m512i x, __m512i R1, __m512i R2, __m512i J, __m512i p){
    __m512i hi = _mm512_srli_epi64(x, 32);
    __m512i t  = _mm512_add_epi64(_mm512_mul_epu32(x, R1),
                                  _mm512_mul_epu32(hi, R2));
    __m512i m  = _mm512_mul_epu32(t, J);
    return _mm512_srli_epi64(_mm512_add_epi64(t, _mm512_mul_epu32(m, p)), 32);
}
static inline __m512i pack_idx(void){
    static const u32 idx[16] = { 0,2,4,6,8,10,12,14, 16,18,20,22,24,26,28,30 };
    return _mm512_loadu_si512(idx);
}
/* decode 16 limbs for prime q -> one packed u32 residue vector */
__attribute__((always_inline))
static inline __m512i dec16(const u64 *src, int q, __m512i PK){
    __m512i R1 = _mm512_set1_epi64(R1c[q]);
    __m512i R2 = _mm512_set1_epi64(R2c[q]);
    __m512i J  = _mm512_set1_epi64(Jc[q]);
    __m512i p  = _mm512_set1_epi64(PR[q]);
    __m512i r0 = redc8(_mm512_loadu_si512(src), R1, R2, J, p);
    __m512i r1 = redc8(_mm512_loadu_si512(src + 8), R1, R2, J, p);
    __m512i r  = _mm512_permutex2var_epi32(r0, PK, r1);
    __m512i p2 = _mm512_set1_epi32((int)P2c[q]);
    return _mm512_min_epu32(r, _mm512_sub_epi32(r, p2));
}
/* stand-in for k1 levels of column butterflies on one vector: a few
 * dependent cheap ops (~ the add/min share of real butterflies); the
 * SAME stand-in runs in every variant so it cancels in the A/B */
__attribute__((always_inline))
static inline __m512i bfly_standin(__m512i v, __m512i p2){
    for(int t = 0; t < 4; ++t){
        v = _mm512_add_epi32(v, _mm512_alignr_epi32(v, v, 1));
        v = _mm512_min_epu32(v, _mm512_sub_epi32(v, p2));
    }
    return v;
}

/* ---- variant U: linear input pass, then per-prime strided col pass -- */
static void run_U(const u64 *a, u32 *r[NP], size_t n, int k1){
    const __m512i PK = pack_idx();
    /* linear input pass */
    for(size_t i = 0; i < n; i += 16)
        for(int q = 0; q < NP; ++q)
            _mm512_storeu_si512(r[q] + i, dec16(a + i, q, PK));
    /* per-prime blocked column pass (RW, strided) */
    size_t M = n / 16;                       /* vectors */
    size_t rows = (size_t)1 << k1, cols = M >> k1;
    for(int q = 0; q < NP; ++q){
        __m512i p2 = _mm512_set1_epi32((int)P2c[q]);
        for(size_t c = 0; c < cols; ++c)
            for(size_t row = 0; row < rows; ++row){
                u32 *v = r[q] + (row * cols + c) * 16;
                _mm512_storeu_si512(v,
                    bfly_standin(_mm512_loadu_si512(v), p2));
            }
    }
}
/* ---- variant P: per-prime fused (strided limb reads, 5 passes) ----- */
static void run_P(const u64 *a, u32 *r[NP], size_t n, int k1){
    const __m512i PK = pack_idx();
    size_t M = n / 16;
    size_t rows = (size_t)1 << k1, cols = M >> k1;
    for(int q = 0; q < NP; ++q){
        __m512i p2 = _mm512_set1_epi32((int)P2c[q]);
        for(size_t c = 0; c < cols; ++c)
            for(size_t row = 0; row < rows; ++row){
                size_t v = (row * cols + c) * 16;
                _mm512_storeu_si512(r[q] + v,
                    bfly_standin(dec16(a + v, q, PK), p2));
            }
    }
}
/* ---- variant F: all-prime block-column fused (one pass) ------------ */
static void run_F(const u64 *a, u32 *r[NP], size_t n, int k1){
    const __m512i PK = pack_idx();
    size_t M = n / 16;
    size_t rows = (size_t)1 << k1, cols = M >> k1;
    for(size_t c = 0; c < cols; ++c)
        for(int q = 0; q < NP; ++q){          /* limb block stays L1-hot */
            __m512i p2 = _mm512_set1_epi32((int)P2c[q]);
            for(size_t row = 0; row < rows; ++row){
                size_t v = (row * cols + c) * 16;
                _mm512_storeu_si512(r[q] + v,
                    bfly_standin(dec16(a + v, q, PK), p2));
            }
        }
}

static void run_F2(const u64 *a, u32 *r[NP], size_t n, int k1, int nt);
int main(int argc, char **argv){
    init_consts();
    size_t n = argc > 1 ? (size_t)atol(argv[1]) : ((size_t)1 << 22);
    int k1 = argc > 2 ? atoi(argv[2]) : 7;

    /* THP-backed buffers (production scratch is THP) */
    size_t lb = n * 8, rb = n * 4;
    size_t total = lb + NP * rb + 6 * (2u << 20);
    unsigned char *base = mmap(NULL, total, PROT_READ | PROT_WRITE,
                               MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if(base == MAP_FAILED){ perror("mmap"); return 1; }
    madvise(base, total, MADV_HUGEPAGE);
    unsigned char *al = base + ((2u << 20) - ((uintptr_t)base & ((2u << 20) - 1)));
    u64 *A = (u64 *)al;
    u32 *r[NP];
    for(int q = 0; q < NP; ++q)
        r[q] = (u32 *)(al + lb + (size_t)q * (rb + (1u << 16)));  /* stagger */
    for(size_t i = 0; i < n; ++i) A[i] = xr();
    for(int q = 0; q < NP; ++q) memset(r[q], 0, rb);

    printf("n = %zu limbs, k1 = %d (stride %zu KB), traffic models: "
           "U 68 / P 60 / F 28 B/limb\n",
           n, k1, ((n / 16) >> k1) * 64 / 1024);
    const char *names[5] = { "U unfused   ", "P prime-fused", "F blockcol  ",
                             "F2 tiled    ", "F2 tiled+NT " };
    for(int v = 0; v < 5; ++v){
        double best = 1e30;
        for(int pass = 0; pass < 5; ++pass){
            double t0 = now();
            if(v == 0) run_U(A, r, n, k1);
            else if(v == 1) run_P(A, r, n, k1);
            else if(v == 2) run_F(A, r, n, k1);
            else run_F2(A, r, n, k1, v == 4);
            double t = now() - t0;
            if(t < best) best = t;
        }
        double model = v == 0 ? 68 : v == 1 ? 60 : 28;
        printf("%s  %8.4f ns/limb   eff-BW %6.2f GB/s (at model %2.0fB)\n",
               names[v], best / n * 1e9, model * n / best / 1e9, model);
    }
    return 0;
}

/* ---- F2: block-column fused with COLUMN TILING (T consecutive
 * vectors per row visit -> prefetchable 1-2KB bursts, limbs L2-hot
 * across primes); optional NT stores kill the RFO on the writes ----- */
#define TILE 16
static void run_F2(const u64 *a, u32 *r[NP], size_t n, int k1, int nt){
    const __m512i PK = pack_idx();
    size_t M = n / 16;
    size_t rows = (size_t)1 << k1, cols = M >> k1;
    for(size_t ct = 0; ct < cols; ct += TILE)
        for(int q = 0; q < NP; ++q){
            __m512i p2 = _mm512_set1_epi32((int)P2c[q]);
            for(size_t row = 0; row < rows; ++row){
                size_t base = (row * cols + ct) * 16;
                for(int t = 0; t < TILE; ++t){
                    __m512i v = bfly_standin(dec16(a + base + 16 * (size_t)t, q, PK), p2);
                    if(nt) _mm512_stream_si512((void *)(r[q] + base + 16 * (size_t)t), v);
                    else   _mm512_storeu_si512(r[q] + base + 16 * (size_t)t, v);
                }
            }
        }
    if(nt) _mm_sfence();
}
