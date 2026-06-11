// Sweep the reference int_fft codecs: balanced mul (an = bn = n) over
// n in [8, 2^17] u64 limbs at 1/16-octave steps, for the three u16-digit
// ranges: U16 (N <= 2^16), Wide U16 (N <= 2^17), Centered U16 (N <= 2^19).
// Each cell is verified bit-exactly against GMP mpn_mul before timing.
// Emits CSV: n,u16,wide,centered (median ns per input limb; blank = size
// beyond that codec's transform cap).
// Build: clang++ -O3 -march=native int_fft_sweep.cpp -I<gmp-build> \
//          <gmp-build>/.libs/libgmp.a -o int_fft_sweep
#define INT_FFT_IMPLEMENTATION
#include "../reference/int_fft.hpp"
#include <gmp.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>

static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint64_t rng = 42;
static uint64_t xr(){ rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }
static int cmpd(const void* x, const void* y){ double a = *(const double*)x, b = *(const double*)y; return (a > b) - (a < b); }
#define PASSES 5

// Force a specific codec at the auto-selected u16 transform size. These reach
// into the implementation TU's internals (anonymous namespace), which is the
// point: the public mul_auto would band-switch on its own.
static int mul_force_u16(uint64_t* rp, const uint64_t* a, ptrdiff_t n,
                         const uint64_t* b){
    return fft::mul(rp, a, n, b, n);
}
static int mul_force_wide(uint64_t* rp, const uint64_t* a, ptrdiff_t n,
                          const uint64_t* b){
    std::uint32_t na, nb;
    if(!digits_for_limbs(n, 16u, na) || !digits_for_limbs(n, 16u, nb)) return 0;
    fft_size fs = choose_fft_size((na + nb + 1u) >> 1);
    if(fs.N_full > MAX_U16_WIDE_TRANSFORM_N) return 0;
    U16WideCodec io;
    return mul_with_codec(rp, a, n, b, n, fs, io);
}
static int mul_force_centered(uint64_t* rp, const uint64_t* a, ptrdiff_t n,
                              const uint64_t* b){
    std::uint32_t na, nb;
    if(!digits_for_limbs(n, 16u, na) || !digits_for_limbs(n, 16u, nb)) return 0;
    fft_size fs = choose_fft_size((na + nb + 1u) >> 1);
    if(fs.N_full > MAX_U16_CENTERED_TRANSFORM_N) return 0;
    CenteredU16Codec io;
    return mul_with_codec(rp, a, n, b, n, fs, io);
}

typedef int (*forcefn)(uint64_t*, const uint64_t*, ptrdiff_t, const uint64_t*);

int main(){
    // 1/16-octave grid over [8, 2^17]
    std::vector<size_t> sizes;
    for(int k = 0; ; ++k){
        size_t n = (size_t)llround(8.0 * std::pow(2.0, k / 16.0));
        if(n > (1u << 17)) break;
        if(sizes.empty() || n != sizes.back()) sizes.push_back(n);
    }

    const size_t NMAX = 1u << 17;
    std::vector<uint64_t> A(NMAX), B(NMAX), R(2 * NMAX), REF(2 * NMAX);
    for(auto& x : A) x = xr();
    for(auto& x : B) x = xr();

    forcefn fns[3] = {mul_force_u16, mul_force_wide, mul_force_centered};
    const char* names[3] = {"u16", "wide", "centered"};

    printf("n,u16,wide,centered\n");
    for(size_t n : sizes){
        mpn_mul((mp_ptr)REF.data(), (mp_srcptr)A.data(), n, (mp_srcptr)B.data(), n);
        double res[3] = {-1, -1, -1};
        for(int c = 0; c < 3; ++c){
            std::memset(R.data(), 0xa5, 2 * n * 8);
            if(!fns[c](R.data(), A.data(), (ptrdiff_t)n, B.data())) continue;
            if(std::memcmp(R.data(), REF.data(), 2 * n * 8) != 0){
                fprintf(stderr, "MISMATCH codec=%s n=%zu\n", names[c], n);
                return 1;
            }
            long reps = (long)(200000 / n) + 1;
            double tp[PASSES];
            for(int p = 0; p < PASSES; ++p){
                double t0 = now();
                for(long q = 0; q < reps; ++q)
                    fns[c](R.data(), A.data(), (ptrdiff_t)n, B.data());
                tp[p] = (now() - t0) / reps;
            }
            qsort(tp, PASSES, 8, cmpd);
            res[c] = tp[PASSES / 2] * 1e9 / (double)n;
        }
        printf("%zu", n);
        for(int c = 0; c < 3; ++c)
            res[c] >= 0 ? printf(",%.3f", res[c]) : printf(",");
        printf("\n");
        fflush(stdout);
    }
    return 0;
}
