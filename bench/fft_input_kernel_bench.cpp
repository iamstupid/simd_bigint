// Microbench for the fused FFT input stage (u64 limbs -> u16 digits ->
// doubles -> first DIF butterfly -> AoSoV tiles), AVX-512.
//
// Variants:
//   (a) radix-2^2, unroll 2: 4 input streams, one full 64B zmm load each
//       (8 limbs = 32 digits = 16 complex = 2 tiles per stream-read)
//   (b) radix-2^3: 8 input streams, 32B ymm loads (1 tile per stream-read)
//   (c) radix-2^2, unroll 2, but each 64B as two 32B loads (isolates
//       load width from shuffle pressure)
// All include the real twiddle multiplies and strided tile stores; (a) and
// (c) are checked for identical output.
//
// Build: clang++ -O3 -march=native fft_input_kernel_bench.cpp -o fikb
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
using std::size_t;
using std::uint64_t;

using vec8 = __m512d;
struct cv { vec8 re, im; };
static inline void cvstore(double* p, cv x){ _mm512_store_pd(p, x.re); _mm512_store_pd(p + 8, x.im); }
static inline cv operator+(cv a, cv b){ return { _mm512_add_pd(a.re, b.re), _mm512_add_pd(a.im, b.im) }; }
static inline cv operator-(cv a, cv b){ return { _mm512_sub_pd(a.re, b.re), _mm512_sub_pd(a.im, b.im) }; }
static inline cv cmul(cv a, cv w){
    return { _mm512_fmsub_pd(a.re, w.re, _mm512_mul_pd(a.im, w.im)),
             _mm512_fmadd_pd(a.re, w.im, _mm512_mul_pd(a.im, w.re)) };
}
static inline cv jw(cv w){ return { _mm512_sub_pd(_mm512_setzero_pd(), w.im), w.re }; }

// ---- digit decode -----------------------------------------------------------
// One raw zmm = 8 limbs = digits d0..d31 (u16). Even digits -> re, odd -> im,
// two tiles (complex 0..7 and 8..15). vpermb gathers the 8 even (odd) u16s of
// each tile into dword slots with a zero top half, then vcvtudq2pd.
struct dec2 { cv t0, t1; };
static inline __m512i dec_idx(int parity, int tile){
    alignas(64) char idx[64];
    for(int lane = 0; lane < 8; ++lane){
        int digit = tile * 16 + 2 * lane + parity;
        idx[lane * 4 + 0] = (char)(digit * 2);
        idx[lane * 4 + 1] = (char)(digit * 2 + 1);
        idx[lane * 4 + 2] = (char)0x40;   // zero (from zero second source)
        idx[lane * 4 + 3] = (char)0x40;
    }
    for(int i = 32; i < 64; ++i) idx[i] = (char)0x40;
    return _mm512_load_si512(idx);
}
static inline vec8 dec_one(__m512i raw, __m512i idx){
    __m512i d = _mm512_permutex2var_epi8(raw, idx, _mm512_setzero_si512());
    return _mm512_cvtepu32_pd(_mm512_castsi512_si256(d));
}
static inline dec2 decode_zmm(__m512i raw, const __m512i* IDX){
    return { { dec_one(raw, IDX[0]), dec_one(raw, IDX[1]) },
             { dec_one(raw, IDX[2]), dec_one(raw, IDX[3]) } };
}

// ---- variant (a): r22 unroll 2, full zmm loads ------------------------------
static void input_r4u2(double* out, const uint64_t* src, size_t n,
                       const double* tw, const __m512i* IDX){
    const size_t l = n / 4;                    // complex per quarter
    const size_t lt = l / 8;                   // tiles per quarter
    double* p0 = out;
    double* p1 = out + 2 * l;
    double* p2 = out + 4 * l;
    double* p3 = out + 6 * l;
    const uint64_t* s0 = src;
    const uint64_t* s1 = src + l / 2;          // 2 digits/complex, 4/limb
    const uint64_t* s2 = src + l;
    const uint64_t* s3 = src + 3 * l / 2;
    const double* twp = tw;
    for(size_t t = 0; t < lt; t += 2){
        dec2 a0 = decode_zmm(_mm512_loadu_si512(s0), IDX);
        dec2 a1 = decode_zmm(_mm512_loadu_si512(s1), IDX);
        dec2 a2 = decode_zmm(_mm512_loadu_si512(s2), IDX);
        dec2 a3 = decode_zmm(_mm512_loadu_si512(s3), IDX);
        s0 += 8; s1 += 8; s2 += 8; s3 += 8;

        cv w1a = { _mm512_load_pd(twp + 0),  _mm512_load_pd(twp + 8)  };
        cv w2a = { _mm512_load_pd(twp + 16), _mm512_load_pd(twp + 24) };
        cv w1b = { _mm512_load_pd(twp + 32), _mm512_load_pd(twp + 40) };
        cv w2b = { _mm512_load_pd(twp + 48), _mm512_load_pd(twp + 56) };
        twp += 64;

        cv b0 = a0.t0 + a2.t0, b2 = cmul(a0.t0 - a2.t0, w1a);
        cv b1 = a1.t0 + a3.t0, b3 = cmul(a1.t0 - a3.t0, jw(w1a));
        cvstore(p0,      b0 + b1);
        cvstore(p1,      cmul(b0 - b1, w2a));
        cvstore(p2,      b2 + b3);
        cvstore(p3,      cmul(b2 - b3, w2a));

        b0 = a0.t1 + a2.t1; b2 = cmul(a0.t1 - a2.t1, w1b);
        b1 = a1.t1 + a3.t1; b3 = cmul(a1.t1 - a3.t1, jw(w1b));
        cvstore(p0 + 16, b0 + b1);
        cvstore(p1 + 16, cmul(b0 - b1, w2b));
        cvstore(p2 + 16, b2 + b3);
        cvstore(p3 + 16, cmul(b2 - b3, w2b));
        p0 += 32; p1 += 32; p2 += 32; p3 += 32;
    }
}

// ---- variant (b): r8, 32B loads ---------------------------------------------
static inline cv decode_ymm(__m256i raw, const __m512i* IDX){
    __m512i r = _mm512_castsi256_si512(raw);
    return { dec_one(r, IDX[0]), dec_one(r, IDX[1]) };
}
static inline cv mnegi(cv a){ return { a.im, _mm512_sub_pd(_mm512_setzero_pd(), a.re) }; }
static inline cv rot8_1(cv a){
    const vec8 c = _mm512_set1_pd(0.70710678118654752440);
    return { _mm512_mul_pd(c, _mm512_add_pd(a.re, a.im)),
             _mm512_mul_pd(c, _mm512_sub_pd(a.im, a.re)) };
}
static inline cv rot8_3(cv a){
    const vec8 c = _mm512_set1_pd(0.70710678118654752440);
    return { _mm512_mul_pd(c, _mm512_sub_pd(a.im, a.re)),
             _mm512_sub_pd(_mm512_setzero_pd(), _mm512_mul_pd(c, _mm512_add_pd(a.re, a.im))) };
}
static void input_r8(double* out, const uint64_t* src, size_t n,
                     const double* tw, const __m512i* IDX){
    const size_t l = n / 8;
    const size_t lt = l / 8;
    double* p[8];
    const uint64_t* s[8];
    for(int j = 0; j < 8; ++j){ p[j] = out + 2 * l * j; s[j] = src + (l / 4) * j; }
    const double* twp = tw;
    for(size_t t = 0; t < lt; ++t){
        cv a0 = decode_ymm(_mm256_loadu_si256((const __m256i*)s[0]), IDX);
        cv a1 = decode_ymm(_mm256_loadu_si256((const __m256i*)s[1]), IDX);
        cv a2 = decode_ymm(_mm256_loadu_si256((const __m256i*)s[2]), IDX);
        cv a3 = decode_ymm(_mm256_loadu_si256((const __m256i*)s[3]), IDX);
        cv a4 = decode_ymm(_mm256_loadu_si256((const __m256i*)s[4]), IDX);
        cv a5 = decode_ymm(_mm256_loadu_si256((const __m256i*)s[5]), IDX);
        cv a6 = decode_ymm(_mm256_loadu_si256((const __m256i*)s[6]), IDX);
        cv a7 = decode_ymm(_mm256_loadu_si256((const __m256i*)s[7]), IDX);
        for(int j = 0; j < 8; ++j) s[j] += 4;

        cv s0 = a0 + a4, s1 = a1 + a5, s2 = a2 + a6, s3 = a3 + a7;
        cv d0 = a0 - a4, d1 = rot8_1(a1 - a5), d2 = mnegi(a2 - a6), d3 = rot8_3(a3 - a7);

        cv w2a = { _mm512_load_pd(twp + 0),  _mm512_load_pd(twp + 8)  };
        cv w4a = { _mm512_load_pd(twp + 16), _mm512_load_pd(twp + 24) };
        cv w2b = { _mm512_load_pd(twp + 32), _mm512_load_pd(twp + 40) };
        cv w4b = { _mm512_load_pd(twp + 48), _mm512_load_pd(twp + 56) };
        twp += 64;

        cv b0 = s0 + s2, b2 = cmul(s0 - s2, w2a);
        cv b1 = s1 + s3, b3 = cmul(s1 - s3, jw(w2a));
        cvstore(p[0], b0 + b1);
        cvstore(p[1], cmul(b0 - b1, w4a));
        cvstore(p[2], b2 + b3);
        cvstore(p[3], cmul(b2 - b3, w4a));
        cv c0 = d0 + d2, c2 = cmul(d0 - d2, w2b);
        cv c1 = d1 + d3, c3 = cmul(d1 - d3, jw(w2b));
        cvstore(p[4], c0 + c1);
        cvstore(p[5], cmul(c0 - c1, w4b));
        cvstore(p[6], c2 + c3);
        cvstore(p[7], cmul(c2 - c3, w4b));
        for(int j = 0; j < 8; ++j) p[j] += 16;
    }
}

// ---- variant (c): r22 unroll 2, two 32B loads per stream --------------------
static void input_r4u2_half(double* out, const uint64_t* src, size_t n,
                            const double* tw, const __m512i* IDX){
    const size_t l = n / 4;
    const size_t lt = l / 8;
    double* p0 = out;
    double* p1 = out + 2 * l;
    double* p2 = out + 4 * l;
    double* p3 = out + 6 * l;
    const uint64_t* s0 = src;
    const uint64_t* s1 = src + l / 2;
    const uint64_t* s2 = src + l;
    const uint64_t* s3 = src + 3 * l / 2;
    const double* twp = tw;
    for(size_t t = 0; t < lt; t += 2){
        #define LD2(s) _mm512_inserti64x4(_mm512_castsi256_si512( \
            _mm256_loadu_si256((const __m256i*)(s))), \
            _mm256_loadu_si256((const __m256i*)((s) + 4)), 1)
        dec2 a0 = decode_zmm(LD2(s0), IDX);
        dec2 a1 = decode_zmm(LD2(s1), IDX);
        dec2 a2 = decode_zmm(LD2(s2), IDX);
        dec2 a3 = decode_zmm(LD2(s3), IDX);
        #undef LD2
        s0 += 8; s1 += 8; s2 += 8; s3 += 8;

        cv w1a = { _mm512_load_pd(twp + 0),  _mm512_load_pd(twp + 8)  };
        cv w2a = { _mm512_load_pd(twp + 16), _mm512_load_pd(twp + 24) };
        cv w1b = { _mm512_load_pd(twp + 32), _mm512_load_pd(twp + 40) };
        cv w2b = { _mm512_load_pd(twp + 48), _mm512_load_pd(twp + 56) };
        twp += 64;

        cv b0 = a0.t0 + a2.t0, b2 = cmul(a0.t0 - a2.t0, w1a);
        cv b1 = a1.t0 + a3.t0, b3 = cmul(a1.t0 - a3.t0, jw(w1a));
        cvstore(p0,      b0 + b1);
        cvstore(p1,      cmul(b0 - b1, w2a));
        cvstore(p2,      b2 + b3);
        cvstore(p3,      cmul(b2 - b3, w2a));
        b0 = a0.t1 + a2.t1; b2 = cmul(a0.t1 - a2.t1, w1b);
        b1 = a1.t1 + a3.t1; b3 = cmul(a1.t1 - a3.t1, jw(w1b));
        cvstore(p0 + 16, b0 + b1);
        cvstore(p1 + 16, cmul(b0 - b1, w2b));
        cvstore(p2 + 16, b2 + b3);
        cvstore(p3 + 16, cmul(b2 - b3, w2b));
        p0 += 32; p1 += 32; p2 += 32; p3 += 32;
    }
}

static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmpd(const void* x, const void* y){ double a = *(const double*)x, b = *(const double*)y; return (a > b) - (a < b); }
static uint64_t rng = 42;
static uint64_t xr(){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
#define PASSES 9

int main(){
    alignas(64) __m512i IDX[4] = { dec_idx(0, 0), dec_idx(1, 0), dec_idx(0, 1), dec_idx(1, 1) };

    printf("n_complex,kib_out,r4u2_64B,r8_32B,r4u2_2x32B  (ns per input limb)\n");
    for(size_t n : {size_t(8192), size_t(65536), size_t(524288)}){
        size_t limbs = n / 2;                  // 2 complex per limb
        uint64_t* src = nullptr; double *outA, *outB, *outC, *tw;
        posix_memalign((void**)&src, 64, limbs * 8 + 64);
        posix_memalign((void**)&outA, 128, 2 * n * 8);
        posix_memalign((void**)&outB, 128, 2 * n * 8);
        posix_memalign((void**)&outC, 128, 2 * n * 8);
        posix_memalign((void**)&tw, 64, n * 8 * 2);
        for(size_t i = 0; i < limbs; ++i) src[i] = xr();
        for(size_t i = 0; i < n * 2; ++i){
            double th = -2.0 * 3.14159265358979323846 * (double)(i % n) / (double)n;
            tw[i] = (i & 8) ? std::sin(th) : std::cos(th);   // unit-modulus filler
        }
        long reps = (long)((1u << 23) / n) + 1;
        double ta[PASSES], tb[PASSES], tc[PASSES];
        for(int p = 0; p < PASSES; ++p){
            double t0 = now();
            for(long q = 0; q < reps; ++q) input_r4u2(outA, src, n, tw, IDX);
            ta[p] = (now() - t0) / reps;
            t0 = now();
            for(long q = 0; q < reps; ++q) input_r8(outB, src, n, tw, IDX);
            tb[p] = (now() - t0) / reps;
            t0 = now();
            for(long q = 0; q < reps; ++q) input_r4u2_half(outC, src, n, tw, IDX);
            tc[p] = (now() - t0) / reps;
        }
        if(memcmp(outA, outC, 2 * n * 8) != 0){ fprintf(stderr, "a/c mismatch\n"); return 1; }
        double sink = outB[1] + outA[7];
        if(sink == -1) printf("#");
        qsort(ta, PASSES, 8, cmpd); qsort(tb, PASSES, 8, cmpd); qsort(tc, PASSES, 8, cmpd);
        printf("%zu,%zu,%.4f,%.4f,%.4f\n", n, n * 16 / 1024,
               ta[PASSES/2] / limbs * 1e9, tb[PASSES/2] / limbs * 1e9, tc[PASSES/2] / limbs * 1e9);
        free(src); free(outA); free(outB); free(outC); free(tw);
    }
    return 0;
}
