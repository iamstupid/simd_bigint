// Microbench for the AVX-512 FFT port: 3 radix-4 full-array passes vs
// 2 radix-8 full-array passes (both = 6 levels), over spans from L1-spill to
// L3-resident. AoSoV tiles of 8 complex per zmm pair: [re0..7 | im0..7],
// 128 bytes. In-place, strided exactly like the real cascade's ancestor
// passes (power-of-2 stream separation -- the pattern that hurt on Zen4).
//
// Radix-4 pass = the reference r22_dif retiled to 8 lanes (4 streams,
// 2 twiddle cv per group). Radix-8 pass = level-1 radix-2 with constant
// omega_8 rotations feeding two r22 sub-butterflies (8 streams, 4 twiddle cv
// per group). Twiddle VALUES are unit-modulus dummies (indexing fidelity is
// irrelevant to throughput; magnitude matters to avoid denormals).
//
// Build: clang++ -O3 -march=native radix_pass_bench.cpp -o radix_pass_bench
#include <immintrin.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

using vec8 = __m512d;
struct cv { vec8 re, im; };

static inline vec8 vload(const double* p){ return _mm512_load_pd(p); }
static inline void vstore(double* p, vec8 x){ _mm512_store_pd(p, x); }
static inline cv cvload(const double* p){ return { vload(p), vload(p + 8) }; }
static inline void cvstore(double* p, cv x){ vstore(p, x.re); vstore(p + 8, x.im); }
static inline cv operator+(cv a, cv b){ return { _mm512_add_pd(a.re, b.re), _mm512_add_pd(a.im, b.im) }; }
static inline cv operator-(cv a, cv b){ return { _mm512_sub_pd(a.re, b.re), _mm512_sub_pd(a.im, b.im) }; }
static inline cv cmul(cv a, cv w){
    return { _mm512_fmsub_pd(a.re, w.re, _mm512_mul_pd(a.im, w.im)),
             _mm512_fmadd_pd(a.re, w.im, _mm512_mul_pd(a.im, w.re)) };
}
// multiply by -i: (re, im) -> (im, -re)
static inline cv mnegi(cv a){
    return { a.im, _mm512_sub_pd(_mm512_setzero_pd(), a.re) };
}
// multiply by e^{-i*pi/4} and e^{-3i*pi/4} (constant omega_8 rotations)
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

// ---- radix-4 (r22) pass over [0, n) at length len; tiles of 8 complex ----
static void r4_pass(double* d, size_t n, size_t len, const double* tw){
    const size_t qt = len / 32;                  // tiles per quarter-block
    for(size_t base = 0; base < n; base += len){
        double* p0 = d + 2 * base;               // 2 doubles per complex
        double* p1 = p0 + 2 * (len / 4);
        double* p2 = p1 + 2 * (len / 4);
        double* p3 = p2 + 2 * (len / 4);
        const double* twp = tw;
        for(size_t t = 0; t < qt; ++t){
            cv a0 = cvload(p0), a1 = cvload(p1), a2 = cvload(p2), a3 = cvload(p3);
            cv w1 = cvload(twp), w2 = cvload(twp + 16);
            cv b0 = a0 + a2;
            cv b2 = cmul(a0 - a2, w1);
            cv b1 = a1 + a3;
            cv b3 = cmul(a1 - a3, { _mm512_sub_pd(_mm512_setzero_pd(), w1.im), w1.re });
            cvstore(p0, b0 + b1);
            cvstore(p1, cmul(b0 - b1, w2));
            cvstore(p2, b2 + b3);
            cvstore(p3, cmul(b2 - b3, w2));
            p0 += 16; p1 += 16; p2 += 16; p3 += 16;
            twp += 32;
        }
    }
}

// ---- radix-8 pass: level-1 r2 (constant rotations) + two r22 halves -------
static void r8_pass(double* d, size_t n, size_t len, const double* tw){
    const size_t et = len / 64;                  // tiles per eighth-block
    for(size_t base = 0; base < n; base += len){
        double* p[8];
        p[0] = d + 2 * base;
        for(int j = 1; j < 8; ++j) p[j] = p[j - 1] + 2 * (len / 8);
        const double* twp = tw;
        for(size_t t = 0; t < et; ++t){
            cv a0 = cvload(p[0]), a1 = cvload(p[1]), a2 = cvload(p[2]), a3 = cvload(p[3]);
            cv a4 = cvload(p[4]), a5 = cvload(p[5]), a6 = cvload(p[6]), a7 = cvload(p[7]);

            cv s0 = a0 + a4, s1 = a1 + a5, s2 = a2 + a6, s3 = a3 + a7;
            cv d0 = a0 - a4;
            cv d1 = rot8_1(a1 - a5);
            cv d2 = mnegi (a2 - a6);
            cv d3 = rot8_3(a3 - a7);

            cv w2a = cvload(twp), w4a = cvload(twp + 16);
            cv w2b = cvload(twp + 32), w4b = cvload(twp + 48);

            cv b0 = s0 + s2;
            cv b2 = cmul(s0 - s2, w2a);
            cv b1 = s1 + s3;
            cv b3 = cmul(s1 - s3, { _mm512_sub_pd(_mm512_setzero_pd(), w2a.im), w2a.re });
            cvstore(p[0], b0 + b1);
            cvstore(p[1], cmul(b0 - b1, w4a));
            cvstore(p[2], b2 + b3);
            cvstore(p[3], cmul(b2 - b3, w4a));

            cv c0 = d0 + d2;
            cv c2 = cmul(d0 - d2, w2b);
            cv c1 = d1 + d3;
            cv c3 = cmul(d1 - d3, { _mm512_sub_pd(_mm512_setzero_pd(), w2b.im), w2b.re });
            cvstore(p[4], c0 + c1);
            cvstore(p[5], cmul(c0 - c1, w4b));
            cvstore(p[6], c2 + c3);
            cvstore(p[7], cmul(c2 - c3, w4b));

            for(int j = 0; j < 8; ++j) p[j] += 16;
            twp += 64;
        }
    }
}

static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmpd(const void* x, const void* y){ double a = *(const double*)x, b = *(const double*)y; return (a > b) - (a < b); }
#define PASSES 9

static double* amalloc(size_t doubles){
    void* p = nullptr;
    if(posix_memalign(&p, 128, doubles * 8)) abort();
    return (double*)p;
}

// unit-modulus twiddle fill: k-th cv lane gets e^{-i*theta_k}, powers spread
static void fill_tw(double* tw, size_t cvs, size_t len, int pw){
    for(size_t k = 0; k < cvs; ++k)
        for(int l = 0; l < 8; ++l){
            double th = -2.0 * 3.14159265358979323846 * double((k * 8 + l) * pw % len) / double(len);
            tw[k * 16 + l] = std::cos(th);
            tw[k * 16 + 8 + l] = std::sin(th);
        }
}

int main(){
    printf("n_complex,kib,r4x3_ns_cx,r8x2_ns_cx,r8/r4\n");
    const size_t spans[] = {4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152,
                            65536, 98304, 131072, 196608, 262144, 393216, 524288};
    for(size_t n : spans){
        if(n % 512) continue;
        double* data = amalloc(2 * n);
        for(size_t i = 0; i < 2 * n; ++i)
            data[i] = std::sin(0.001 * (double)i) + 1.0;   // benign magnitudes

        // r4: 3 passes at len n, n/4, n/16. r8: 2 passes at len n, n/8.
        // per-pass twiddle tables (table sized len-doubles; offsets per pass)
        double* tw4a = amalloc(n / 32 * 32); fill_tw(tw4a, n / 32 * 2, n, 1);
        double* tw4b = amalloc(n / 128 * 32); fill_tw(tw4b, n / 128 * 2, n / 4, 1);
        double* tw4c = amalloc(n / 512 * 32); fill_tw(tw4c, n / 512 * 2, n / 16, 1);
        double* tw8a = amalloc(n / 64 * 64); fill_tw(tw8a, n / 64 * 4, n, 1);
        double* tw8b = amalloc(n / 512 * 64); fill_tw(tw8b, n / 512 * 4, n / 8, 1);

        long reps = (long)((1u << 24) / n) + 1;
        double t4[PASSES], t8[PASSES];
        for(int p = 0; p < PASSES; ++p){
            double t0 = now();
            for(long q = 0; q < reps; ++q){
                r4_pass(data, n, n, tw4a);
                r4_pass(data, n, n / 4, tw4b);
                r4_pass(data, n, n / 16, tw4c);
            }
            t4[p] = (now() - t0) / reps;
            t0 = now();
            for(long q = 0; q < reps; ++q){
                r8_pass(data, n, n, tw8a);
                r8_pass(data, n, n / 8, tw8b);
            }
            t8[p] = (now() - t0) / reps;
        }
        double sink = 0;
        for(size_t i = 0; i < 2 * n; i += 97) sink += data[i];
        if(sink == -1) printf("#");                       // anti-DCE
        qsort(t4, PASSES, 8, cmpd); qsort(t8, PASSES, 8, cmpd);
        double a = t4[PASSES / 2] / n * 1e9, b = t8[PASSES / 2] / n * 1e9;
        printf("%zu,%zu,%.4f,%.4f,%.4f\n", n, n * 16 / 1024, a, b, b / a);
        fflush(stdout);
        free(data); free(tw4a); free(tw4b); free(tw4c); free(tw8a); free(tw8b);
    }
    return 0;
}
