// p30 NTT core-arithmetic microbench: 16-lane epi32 butterflies over a
// 30-bit prime, Shoup (precomputed twiddle quotient) vs Montgomery.
// Validates each kernel against a scalar reference, then times a
// radix-2 DIF stage pass at L1 / L2 / L3+ working sets.
//   p = 998244353 (largest 5 primes p = k*2^22+1 < 2^30; this is the rep)
//   lazy ranges: values live in [0, 2p), 4p < 2^32 so a+b never wraps
// Build: clang -O3 -march=native -std=c11 p30_kernel_bench.c -o p30_kernel_bench
#define _POSIX_C_SOURCE 199309L
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define P 998244353u
#define LG2VAL 23

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rng = 42;
static uint64_t xr(void){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

// ---- scalar reference ------------------------------------------------
static uint32_t mulmod_ref(uint32_t a, uint32_t b){      // canonical inputs
    return (uint32_t)((uint64_t)a * b % P);
}
static uint32_t powmod(uint32_t a, uint64_t e){
    uint64_t r = 1, b = a;
    for(; e; e >>= 1, b = b * b % P)
        if(e & 1) r = r * b % P;
    return (uint32_t)r;
}

// ---- Shoup: wq = floor(w * 2^32 / p), r = w*x - hi32(x*wq)*p in [0,2p)
static inline __m512i mm_shoup(__m512i x, __m512i w, __m512i wq, __m512i p){
    __m512i pe = _mm512_mul_epu32(x, wq);                       // even lanes
    __m512i po = _mm512_mul_epu32(_mm512_srli_epi64(x, 32),
                                  _mm512_srli_epi64(wq, 32));   // odd lanes
    __m512i q  = _mm512_mask_mov_epi32(_mm512_srli_epi64(pe, 32),
                                       (__mmask16)0xAAAA, po);
    return _mm512_sub_epi32(_mm512_mullo_epi32(x, w),
                            _mm512_mullo_epi32(q, p));
}
// ---- Montgomery: w in Montgomery form (wR mod p), J = -p^-1 mod 2^32
// r = (x*w + (x*w*J mod 2^32)*p) >> 32 in [0, 2p)
static inline __m512i mm_mont(__m512i x, __m512i w, __m512i J, __m512i p){
    __m512i te = _mm512_mul_epu32(x, w);
    __m512i to = _mm512_mul_epu32(_mm512_srli_epi64(x, 32),
                                  _mm512_srli_epi64(w, 32));
    __m512i me = _mm512_mul_epu32(te, J);                       // lo32 use only
    __m512i mo = _mm512_mul_epu32(to, J);
    __m512i re = _mm512_srli_epi64(_mm512_add_epi64(te, _mm512_mul_epu32(me, p)), 32);
    __m512i ro = _mm512_srli_epi64(_mm512_add_epi64(to, _mm512_mul_epu32(mo, p)), 32);
    return _mm512_mask_mov_epi32(re, (__mmask16)0xAAAA, _mm512_slli_epi64(ro, 32));
}

// lazy add/sub: inputs [0,2p) -> outputs [0,2p)
static inline __m512i addmod2p(__m512i a, __m512i b, __m512i p2){
    __m512i s = _mm512_add_epi32(a, b);
    return _mm512_min_epu32(s, _mm512_sub_epi32(s, p2));
}
static inline __m512i submod2p(__m512i a, __m512i b, __m512i p2){
    __m512i d = _mm512_sub_epi32(a, b);
    return _mm512_min_epu32(d, _mm512_add_epi32(d, p2));
}

// ---- stage kernels: one radix-2 DIF level over n points, stride n/2,
// twiddle per 16-lane group (block-twiddle layout: same w for the group)
typedef struct { uint32_t* tw; uint32_t* twq; } twtab;

static void stage_shoup(uint32_t* d, uint32_t n, const twtab* T){
    const __m512i p  = _mm512_set1_epi32((int)P);
    const __m512i p2 = _mm512_set1_epi32((int)(2 * P));
    uint32_t h = n / 2;
    for(uint32_t j = 0; j < h; j += 16){
        __m512i w  = _mm512_load_si512(T->tw + j);
        __m512i wq = _mm512_load_si512(T->twq + j);
        __m512i a = _mm512_load_si512(d + j);
        __m512i b = _mm512_load_si512(d + h + j);
        __m512i s = addmod2p(a, b, p2);
        __m512i t = _mm512_add_epi32(_mm512_sub_epi32(a, b), p2);  // [0,4p)
        _mm512_store_si512(d + j, s);
        _mm512_store_si512(d + h + j, mm_shoup(t, w, wq, p));
    }
}
static void stage_mont(uint32_t* d, uint32_t n, const twtab* T){
    const __m512i p  = _mm512_set1_epi32((int)P);
    const __m512i p2 = _mm512_set1_epi32((int)(2 * P));
    uint32_t h = n / 2;
    __m512i Jv = _mm512_set1_epi32((int)0x3B7FFFFF);   /* -P^-1 mod 2^32, checked in main */
    for(uint32_t j = 0; j < h; j += 16){
        __m512i w  = _mm512_load_si512(T->tw + j);     // Montgomery-form twiddles
        __m512i a = _mm512_load_si512(d + j);
        __m512i b = _mm512_load_si512(d + h + j);
        __m512i s = addmod2p(a, b, p2);
        __m512i t = _mm512_add_epi32(_mm512_sub_epi32(a, b), p2);
        _mm512_store_si512(d + j, s);
        _mm512_store_si512(d + h + j, mm_mont(t, w, Jv, p));
    }
}

// pure ALU throughput: chain-free mulmods on registers
static uint32_t alu_loop(int shoup, long iters){
    __m512i x[8], w, wq, p = _mm512_set1_epi32((int)P);
    __m512i Jv = _mm512_set1_epi32((int)0x3B7FFFFF);
    for(int i = 0; i < 8; ++i) x[i] = _mm512_set1_epi32(12345 + i);
    w  = _mm512_set1_epi32(7777777);
    wq = _mm512_set1_epi32((int)(((unsigned __int128)7777777 << 32) / P));
    for(long it = 0; it < iters; ++it)
        for(int i = 0; i < 8; ++i)
            x[i] = shoup ? mm_shoup(x[i], w, wq, p) : mm_mont(x[i], w, Jv, p);
    uint32_t s = 0;
    for(int i = 0; i < 8; ++i) s ^= (uint32_t)_mm512_reduce_add_epi32(x[i]);
    return s;
}

static int barrett_main(void);
int main(void){
    // J = -P^-1 mod 2^32 sanity
    uint32_t pinv = 1;                          // Newton: inverse of P mod 2^32
    for(int i = 0; i < 5; ++i) pinv *= 2 - P * pinv;
    uint32_t J = (uint32_t)(0u - pinv);
    if(J != 0x3B7FFFFFu){ printf("J = %08x (update constant!)\n", J); return 1; }

    // ---- validate both mulmods over random + edge inputs ----
    (void)0;
    for(int t = 0; t < 100000; ++t){
        uint32_t x = (uint32_t)(xr() % (4 * P));            // butterfly diff range
        uint32_t w = (uint32_t)(xr() % P);
        if(t == 0){ x = 4 * P - 1; w = P - 1; }
        if(t == 1){ x = 0; w = 0; }
        uint32_t wq = (uint32_t)(((unsigned __int128)w << 32) / P);
        __m512i r = mm_shoup(_mm512_set1_epi32((int)x), _mm512_set1_epi32((int)w),
                             _mm512_set1_epi32((int)wq), _mm512_set1_epi32((int)P));
        uint32_t got;
        memcpy(&got, &r, 4);
        if(got >= 2 * P || got % P != mulmod_ref(x % P, w)){
            printf("SHOUP FAIL x=%u w=%u got=%u\n", x, w, got);
            return 1;
        }
        // montgomery: wm = w*R mod p; result should be x*w*R*R^-1 = x*w (mod p)
        uint32_t wm = (uint32_t)(((unsigned __int128)w << 32) % P);
        r = mm_mont(_mm512_set1_epi32((int)x), _mm512_set1_epi32((int)wm),
                    _mm512_set1_epi32((int)J), _mm512_set1_epi32((int)P));
        memcpy(&got, &r, 4);
        if(got >= 2 * P || got % P != mulmod_ref(x % P, w)){
            printf("MONT FAIL x=%u w=%u got=%u (wm=%u)\n", x, w, got, wm);
            return 1;
        }
    }
    printf("mulmod validation OK (shoup + montgomery, 100k cases)\n");

    // ---- ALU throughput (dependent chains x8 = throughput-ish) ----
    for(int s = 1; s >= 0; --s){
        long iters = 4000000;
        double t0 = now();
        volatile uint32_t sink = alu_loop(s, iters);
        (void)sink;
        double dt = now() - t0;
        printf("%-10s reg-chain x8: %.3f ns per 16-lane mulmod\n",
               s ? "shoup" : "montgomery", dt / (iters * 8.0) * 1e9);
    }

    // ---- stage pass at three working-set sizes ----
    uint32_t g = 3;                                  // 3 generates (Z/P)^*
    (void)g; (void)powmod;
    for(int variant = 0; variant < 2; ++variant){
        for(uint32_t lgn = 13; lgn <= 23; lgn += 5){ // 32KB, 1MB, 32MB
            uint32_t n = 1u << lgn;
            uint32_t* d  = aligned_alloc(64, n * 4);
            uint32_t* tw = aligned_alloc(64, n / 2 * 4);
            uint32_t* twq = aligned_alloc(64, n / 2 * 4);
            for(uint32_t i = 0; i < n; ++i) d[i] = (uint32_t)(xr() % (2 * P));
            for(uint32_t i = 0; i < n / 2; ++i){
                uint32_t w = (uint32_t)(xr() % P);
                tw[i] = w;
                twq[i] = (uint32_t)(((unsigned __int128)w << 32) / P);
            }
            twtab T = { tw, twq };
            long reps = (long)(1u << 25) / n + 1;
            double best = 1e30;
            for(int pass = 0; pass < 7; ++pass){
                double t0 = now();
                for(long q = 0; q < reps; ++q)
                    variant ? stage_mont(d, n, &T) : stage_shoup(d, n, &T);
                double t = (now() - t0) / reps;
                if(t < best) best = t;
            }
            printf("%-10s stage n=2^%-2u (%5.1f KB): %.3f ns / point-pair, %.2f GB/s\n",
                   variant ? "montgomery" : "shoup", lgn, n * 4 / 1024.0,
                   best / (n / 2.0) * 1e9,
                   /* traffic: data rw + twiddles r (shoup 2x) */
                   (n * 4.0 * 2 + (variant ? 1 : 2) * (n / 2.0) * 4.0) / best / 1e9);
            free(d); free(tw); free(twq);
        }
    }
    return barrett_main();
}

// ---- Barrett-by-constant (user variant, AVX-512 form): c_rec =
// floor(c*2^32/mod); for ANY u32 lane a, q = hi32(a*c_rec) under-
// estimates floor(a*c/mod) by at most 1 -> r = a*c - q*mod in [0,2p).
// The multiplier is a broadcast loop constant: no twiddle stream at all.
static inline __m512i mm_barrett_c(__m512i a, __m512i vc, __m512i vrec, __m512i p){
    __m512i pe = _mm512_mul_epu32(a, vrec);
    __m512i po = _mm512_mul_epu32(_mm512_srli_epi64(a, 32), vrec);
    __m512i q  = _mm512_mask_mov_epi32(_mm512_srli_epi64(pe, 32),
                                       (__mmask16)0xAAAA, po);
    return _mm512_sub_epi32(_mm512_mullo_epi32(a, vc),
                            _mm512_mullo_epi32(q, p));
}

// block-twiddle radix-2 level: each 256-point sub-transform shares one
// scalar twiddle (FLINT-style layer); constants hoisted per block
static void stage_barrett_blk(uint32_t* d, uint32_t n, const uint32_t* bw,
                              const uint32_t* bwr){
    const __m512i p  = _mm512_set1_epi32((int)P);
    const __m512i p2 = _mm512_set1_epi32((int)(2 * P));
    for(uint32_t base = 0; base < n; base += 256){
        __m512i vc   = _mm512_set1_epi32((int)bw[base >> 8]);
        __m512i vrec = _mm512_set1_epi32((int)bwr[base >> 8]);
        for(uint32_t j = 0; j < 128; j += 16){
            __m512i a = _mm512_load_si512(d + base + j);
            __m512i b = _mm512_load_si512(d + base + 128 + j);
            __m512i s = addmod2p(a, b, p2);
            __m512i t = _mm512_add_epi32(_mm512_sub_epi32(a, b), p2);
            _mm512_store_si512(d + base + j, s);
            _mm512_store_si512(d + base + 128 + j, mm_barrett_c(t, vc, vrec, p));
        }
    }
}

static int barrett_main(void){
    // validate: any u32 a (incl. 0xFFFFFFFF), c < P
    for(int t = 0; t < 100000; ++t){
        uint32_t a = (uint32_t)xr();
        uint32_t c = (uint32_t)(xr() % P);
        if(t == 0){ a = 0xFFFFFFFFu; c = P - 1; }
        if(t == 1){ a = 0; c = 0; }
        uint32_t cr = (uint32_t)(((unsigned __int128)c << 32) / P);
        __m512i r = mm_barrett_c(_mm512_set1_epi32((int)a), _mm512_set1_epi32((int)c),
                                 _mm512_set1_epi32((int)cr), _mm512_set1_epi32((int)P));
        uint32_t got;
        memcpy(&got, &r, 4);
        if(got >= 2 * P || got % P != (uint32_t)((uint64_t)(a % P) * c % P)){
            printf("BARRETT FAIL a=%u c=%u got=%u\n", a, c, got);
            return 1;
        }
    }
    printf("barrett-const validation OK (100k cases incl. a = 2^32-1)\n");

    for(uint32_t lgn = 13; lgn <= 23; lgn += 5){
        uint32_t n = 1u << lgn;
        uint32_t* d   = aligned_alloc(64, n * 4);
        uint32_t* bw  = aligned_alloc(64, (n >> 8) * 4);
        uint32_t* bwr = aligned_alloc(64, (n >> 8) * 4);
        for(uint32_t i = 0; i < n; ++i) d[i] = (uint32_t)(xr() % (2 * P));
        for(uint32_t i = 0; i < n >> 8; ++i){
            uint32_t w = (uint32_t)(xr() % P);
            bw[i] = w;
            bwr[i] = (uint32_t)(((unsigned __int128)w << 32) / P);
        }
        long reps = (long)(1u << 25) / n + 1;
        double best = 1e30;
        for(int pass = 0; pass < 7; ++pass){
            double t0 = now();
            for(long q = 0; q < reps; ++q)
                stage_barrett_blk(d, n, bw, bwr);
            double t = (now() - t0) / reps;
            if(t < best) best = t;
        }
        printf("barrett-blk stage n=2^%-2u (%5.1f KB): %.3f ns / point-pair, %.2f GB/s\n",
               lgn, n * 4 / 1024.0, best / (n / 2.0) * 1e9,
               (n * 4.0 * 2 + (n >> 8) * 8.0) / best / 1e9);
        free(d); free(bw); free(bwr);
    }
    return 0;
}
