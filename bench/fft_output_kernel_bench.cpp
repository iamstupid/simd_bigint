// Microbench for the fused FFT output stage (final inverse r22-unroll2 pass
// + U16 digit emit), AVX-512. Compares carry architectures:
//   (A) reference-style: pack to lo/hi, store an interleaved __int128 partial
//       array, then a separate serial scalar 128-bit carry pass
//   (B) fully fused: pack to lo/hi lanes, valignq hi-shift, vpcmpuq SWAR
//       ripple (base-2^64 canonize trick), final limbs stored directly --
//       no partial array. One carry chain per quarter-front.
// Outputs are checked identical (validates the SWAR carry logic itself).
//
// Build: clang++ -O3 -march=native fft_output_kernel_bench.cpp -o fokb
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
static inline cv cvload(const double* p){ return { _mm512_load_pd(p), _mm512_load_pd(p + 8) }; }
static inline cv operator+(cv a, cv b){ return { _mm512_add_pd(a.re, b.re), _mm512_add_pd(a.im, b.im) }; }
static inline cv operator-(cv a, cv b){ return { _mm512_sub_pd(a.re, b.re), _mm512_sub_pd(a.im, b.im) }; }
static inline cv cmul_conj(cv a, cv w){
    return { _mm512_fmadd_pd(a.im, w.im, _mm512_mul_pd(a.re, w.re)),
             _mm512_fmsub_pd(a.im, w.re, _mm512_mul_pd(a.re, w.im)) };
}
static inline cv jw(cv w){ return { _mm512_sub_pd(_mm512_setzero_pd(), w.im), w.re }; }

// magic-bias round of clipped non-negative doubles (< 2^51) to u64 lanes
static inline __m512i mround(vec8 x){
    const __m512i bias = _mm512_set1_epi64(0x4330000000000000LL);
    x = _mm512_max_pd(x, _mm512_setzero_pd());
    return _mm512_sub_epi64(_mm512_castpd_si512(_mm512_add_pd(x, _mm512_castsi512_pd(bias))), bias);
}

// pack two tiles (16 complex = 32 digits) into 8 limbs as lo/hi lane pairs
struct lohi { __m512i lo, hi; };
static inline lohi pack2(cv t0, cv t1){
    __m512i re0 = mround(t0.re), im0 = mround(t0.im);
    __m512i re1 = mround(t1.re), im1 = mround(t1.im);
    // u = re + (im << 16); u_hi = (im >> 48) + carry
    __m512i ul0 = _mm512_add_epi64(re0, _mm512_slli_epi64(im0, 16));
    __m512i uh0 = _mm512_add_epi64(_mm512_srli_epi64(im0, 48),
        _mm512_maskz_set1_epi64(_mm512_cmplt_epu64_mask(ul0, re0), 1));
    __m512i ul1 = _mm512_add_epi64(re1, _mm512_slli_epi64(im1, 16));
    __m512i uh1 = _mm512_add_epi64(_mm512_srli_epi64(im1, 48),
        _mm512_maskz_set1_epi64(_mm512_cmplt_epu64_mask(ul1, re1), 1));
    // limb L = u[2L] + (u[2L+1] << 32): even/odd complex lane compress
    const __m512i IE = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i IO = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    __m512i elo = _mm512_permutex2var_epi64(ul0, IE, ul1);
    __m512i olo = _mm512_permutex2var_epi64(ul0, IO, ul1);
    __m512i ehi = _mm512_permutex2var_epi64(uh0, IE, uh1);
    __m512i ohi = _mm512_permutex2var_epi64(uh0, IO, uh1);
    __m512i llo = _mm512_add_epi64(elo, _mm512_slli_epi64(olo, 32));
    __m512i lhi = _mm512_add_epi64(_mm512_add_epi64(ehi, _mm512_srli_epi64(olo, 32)),
        _mm512_add_epi64(_mm512_slli_epi64(ohi, 32),
        _mm512_maskz_set1_epi64(_mm512_cmplt_epu64_mask(llo, elo), 1)));
    return { llo, lhi };
}

// per-front carry state for the fused chain
struct chain {
    __m512i prev_hi;
    unsigned cin;
};
static inline __m512i chain_step(chain& c, lohi p){
    __m512i his = _mm512_alignr_epi64(p.hi, c.prev_hi, 7);
    __m512i sum = _mm512_add_epi64(p.lo, his);
    unsigned g = _mm512_cmplt_epu64_mask(sum, his);
    unsigned pr = _mm512_cmpeq_epu64_mask(sum, _mm512_set1_epi64(-1));
    unsigned cn = pr + ((g << 1) | c.cin);
    unsigned carries = cn ^ pr;
    sum = _mm512_mask_add_epi64(sum, (__mmask8)carries, sum, _mm512_set1_epi64(1));
    c.cin = (cn >> 8) & 1u;
    c.prev_hi = p.hi;
    return sum;
}

// ---- variant B: final DIT r22-unroll2 pass + fused in-vector emit ----------
static void out_fused(uint64_t* rp, const double* spec, size_t n, const double* tw){
    const size_t qt = n / 32;                  // tile-pairs per quarter... tiles per quarter
    const size_t limbs_q = n / 8;              // limbs per quarter (4 digits/limb)
    chain ch[4];
    for(int f = 0; f < 4; ++f){ ch[f].prev_hi = _mm512_setzero_si512(); ch[f].cin = 0; }
    const double* p0 = spec;
    const double* p1 = spec + 2 * (n / 4);
    const double* p2 = spec + 4 * (n / 4);
    const double* p3 = spec + 6 * (n / 4);
    uint64_t* r[4] = { rp, rp + limbs_q, rp + 2 * limbs_q, rp + 3 * limbs_q };
    const double* twp = tw;
    for(size_t t = 0; t < qt; t += 2){
        cv w1 = { _mm512_load_pd(twp + 0),  _mm512_load_pd(twp + 8)  };
        cv w2 = { _mm512_load_pd(twp + 16), _mm512_load_pd(twp + 24) };
        cv w1b = { _mm512_load_pd(twp + 32), _mm512_load_pd(twp + 40) };
        cv w2b = { _mm512_load_pd(twp + 48), _mm512_load_pd(twp + 56) };
        twp += 64;
        cv o0[2], o1[2], o2[2], o3[2];
        for(int u = 0; u < 2; ++u){
            cv y0 = cvload(p0 + 16 * u), y1 = cvload(p1 + 16 * u);
            cv y2 = cvload(p2 + 16 * u), y3 = cvload(p3 + 16 * u);
            cv wa = u ? w1b : w1, wb = u ? w2b : w2;
            cv s  = cmul_conj(y1, wb);
            cv b0 = y0 + s, b1 = y0 - s;
            s     = cmul_conj(y3, wb);
            cv b2 = y2 + s, b3 = y2 - s;
            s = cmul_conj(b2, wa);
            o0[u] = b0 + s; o2[u] = b0 - s;
            s = cmul_conj(b3, jw(wa));
            o1[u] = b1 + s; o3[u] = b1 - s;
        }
        p0 += 32; p1 += 32; p2 += 32; p3 += 32;
        _mm512_storeu_si512(r[0], chain_step(ch[0], pack2(o0[0], o0[1])));
        _mm512_storeu_si512(r[1], chain_step(ch[1], pack2(o1[0], o1[1])));
        _mm512_storeu_si512(r[2], chain_step(ch[2], pack2(o2[0], o2[1])));
        _mm512_storeu_si512(r[3], chain_step(ch[3], pack2(o3[0], o3[1])));
        r[0] += 8; r[1] += 8; r[2] += 8; r[3] += 8;
    }
    // junction fixups: front f's trailing (hi lane 7, cout) enters front f+1
    for(int f = 0; f < 3; ++f){
        unsigned __int128 c = (unsigned __int128)((uint64_t*)&ch[f].prev_hi)[7] + ch[f].cin;
        uint64_t* q = rp + (size_t)(f + 1) * limbs_q;
        while(c){ c += *q; *q++ = (uint64_t)c; c >>= 64; }
    }
}

// ---- variant A: same pass, partial-array emit + scalar carry ---------------
static void out_partial(uint64_t* rp, unsigned __int128* partial,
                        const double* spec, size_t n, const double* tw){
    const size_t qt = n / 32;
    const size_t limbs_q = n / 8;
    const double* p0 = spec;
    const double* p1 = spec + 2 * (n / 4);
    const double* p2 = spec + 4 * (n / 4);
    const double* p3 = spec + 6 * (n / 4);
    unsigned __int128* r[4] = { partial, partial + limbs_q, partial + 2 * limbs_q, partial + 3 * limbs_q };
    const __m512i ILO = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i IHI = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    const double* twp = tw;
    for(size_t t = 0; t < qt; t += 2){
        cv w1 = { _mm512_load_pd(twp + 0),  _mm512_load_pd(twp + 8)  };
        cv w2 = { _mm512_load_pd(twp + 16), _mm512_load_pd(twp + 24) };
        cv w1b = { _mm512_load_pd(twp + 32), _mm512_load_pd(twp + 40) };
        cv w2b = { _mm512_load_pd(twp + 48), _mm512_load_pd(twp + 56) };
        twp += 64;
        cv o0[2], o1[2], o2[2], o3[2];
        for(int u = 0; u < 2; ++u){
            cv y0 = cvload(p0 + 16 * u), y1 = cvload(p1 + 16 * u);
            cv y2 = cvload(p2 + 16 * u), y3 = cvload(p3 + 16 * u);
            cv wa = u ? w1b : w1, wb = u ? w2b : w2;
            cv s  = cmul_conj(y1, wb);
            cv b0 = y0 + s, b1 = y0 - s;
            s     = cmul_conj(y3, wb);
            cv b2 = y2 + s, b3 = y2 - s;
            s = cmul_conj(b2, wa);
            o0[u] = b0 + s; o2[u] = b0 - s;
            s = cmul_conj(b3, jw(wa));
            o1[u] = b1 + s; o3[u] = b1 - s;
        }
        p0 += 32; p1 += 32; p2 += 32; p3 += 32;
        lohi q0 = pack2(o0[0], o0[1]), q1 = pack2(o1[0], o1[1]);
        lohi q2 = pack2(o2[0], o2[1]), q3 = pack2(o3[0], o3[1]);
        // interleave lo/hi into i128 layout: two stores per front
        _mm512_storeu_si512(r[0],     _mm512_permutex2var_epi64(q0.lo, ILO, q0.hi));
        _mm512_storeu_si512(r[0] + 4, _mm512_permutex2var_epi64(q0.lo, IHI, q0.hi));
        _mm512_storeu_si512(r[1],     _mm512_permutex2var_epi64(q1.lo, ILO, q1.hi));
        _mm512_storeu_si512(r[1] + 4, _mm512_permutex2var_epi64(q1.lo, IHI, q1.hi));
        _mm512_storeu_si512(r[2],     _mm512_permutex2var_epi64(q2.lo, ILO, q2.hi));
        _mm512_storeu_si512(r[2] + 4, _mm512_permutex2var_epi64(q2.lo, IHI, q2.hi));
        _mm512_storeu_si512(r[3],     _mm512_permutex2var_epi64(q3.lo, ILO, q3.hi));
        _mm512_storeu_si512(r[3] + 4, _mm512_permutex2var_epi64(q3.lo, IHI, q3.hi));
        r[0] += 8; r[1] += 8; r[2] += 8; r[3] += 8;
    }
    unsigned __int128 carry = 0;
    const size_t total = n / 2;
    for(size_t i = 0; i < total; ++i){
        carry += partial[i];
        rp[i] = (uint64_t)carry;
        carry >>= 64;
    }
}

static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int cmpd(const void* x, const void* y){ double a = *(const double*)x, b = *(const double*)y; return (a > b) - (a < b); }
static uint64_t rng = 42;
static uint64_t xr(){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
#define PASSES 9

int main(){
    printf("n_complex,out_limbs,partialA_ns_limb,fusedB_ns_limb,B/A\n");
    for(size_t n : {size_t(8192), size_t(65536), size_t(524288)}){
        size_t limbs = n / 2;
        double *spec, *tw; uint64_t *ra, *rb; unsigned __int128* part;
        posix_memalign((void**)&spec, 128, 2 * n * 8);
        posix_memalign((void**)&tw, 64, n * 8 * 2);
        posix_memalign((void**)&ra, 64, limbs * 8 + 64);
        posix_memalign((void**)&rb, 64, limbs * 8 + 64);
        posix_memalign((void**)&part, 64, limbs * 16 + 64);
        // synthetic product spectrum: coefficient-scale values (~2^44)
        for(size_t i = 0; i < 2 * n; ++i)
            spec[i] = (double)(xr() >> 20);
        for(size_t i = 0; i < n * 2; ++i){
            double th = -2.0 * 3.14159265358979323846 * (double)(i % n) / (double)n;
            tw[i] = (i & 8) ? std::sin(th) : std::cos(th);
        }
        out_partial(ra, part, spec, n, tw);
        out_fused(rb, spec, n, tw);
        if(memcmp(ra, rb, limbs * 8) != 0){
            for(size_t i = 0; i < limbs; ++i) if(ra[i] != rb[i]){
                fprintf(stderr, "MISMATCH n=%zu limb=%zu A=%016llx B=%016llx\n", n, i,
                        (unsigned long long)ra[i], (unsigned long long)rb[i]);
                return 1;
            }
        }
        long reps = (long)((1u << 23) / n) + 1;
        double ta[PASSES], tb[PASSES];
        for(int p = 0; p < PASSES; ++p){
            double t0 = now();
            for(long q = 0; q < reps; ++q) out_partial(ra, part, spec, n, tw);
            ta[p] = (now() - t0) / reps;
            t0 = now();
            for(long q = 0; q < reps; ++q) out_fused(rb, spec, n, tw);
            tb[p] = (now() - t0) / reps;
        }
        qsort(ta, PASSES, 8, cmpd); qsort(tb, PASSES, 8, cmpd);
        double a = ta[PASSES/2] / limbs * 1e9, b = tb[PASSES/2] / limbs * 1e9;
        printf("%zu,%zu,%.4f,%.4f,%.4f\n", n, limbs, a, b, b / a);
        free(spec); free(tw); free(ra); free(rb); free(part);
    }
    return 0;
}
