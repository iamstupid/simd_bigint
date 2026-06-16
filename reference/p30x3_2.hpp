// int_fft NTT backend implementation.
//
// Residue buffers are kept in standard form. The p30x3 CRT path uses ascending
// Garner reconstruction with AVX2 Barrett constant multiplication.

// int_fft NTT backend scaffolding.
#pragma once

#include <cstddef>
#include <cstdint>

namespace fft::ntt {

enum class p30_engine : unsigned {
    A = 0,
    B = 1,
    C = 2,
};

struct p30_engine_info {
    char name;
    std::uint32_t radix;
    std::uint32_t max_n_u32;
    std::uint32_t p0;
    std::uint32_t p1;
    std::uint32_t p2;
};

p30_engine_info get_p30_engine_info(p30_engine engine);

// Return the selected prime modulus for an engine slot 0..2.
std::uint32_t p30_modulus(p30_engine engine, unsigned prime_index);

// Reduce src[0..src_len) modulo one p30 prime and zero-pad to n.
// n must be the transform length accepted by ntt_p30_forward/inverse.
bool ntt_p30_reduce_and_pad(std::uint32_t* out,
                            const std::uint32_t* src,
                            std::size_t src_len,
                            std::size_t n,
                            p30_engine engine,
                            unsigned prime_index);

// In-place standard-domain NTT. Forward is DIF natural -> bit-reversed order;
// inverse is DIT bit-reversed -> natural order and includes the 1/n scale.
bool ntt_p30_forward(std::uint32_t* data,
                     std::size_t n,
                     p30_engine engine,
                     unsigned prime_index);
bool ntt_p30_inverse(std::uint32_t* data,
                     std::size_t n,
                     p30_engine engine,
                     unsigned prime_index);

// Pointwise product in the NTT domain. out may alias a or b.
bool ntt_p30_pointwise_mul(std::uint32_t* out,
                           const std::uint32_t* a,
                           const std::uint32_t* b,
                           std::size_t n,
                           p30_engine engine,
                           unsigned prime_index);

// One-prime convolution helper: reduce/pad, forward both inputs, multiply, and
// inverse. n is the transform size and must be large enough for the caller's
// desired convolution length.
bool ntt_p30_convolve(std::uint32_t* out,
                      const std::uint32_t* a,
                      std::size_t an,
                      const std::uint32_t* b,
                      std::size_t bn,
                      std::size_t n,
                      p30_engine engine,
                      unsigned prime_index);

// Reconstruct p30x3 CRT residues, pair adjacent base-2^32 coefficients, and
// propagate carries in base 2^64. Inputs r0/r1/r2 are coefficient residues in
// [0, p_i). out64 receives ceil(coeff_len / 2) limbs; the returned value is the
// final high carry limb.
std::uint64_t crt_p30_garner_u64(std::uint64_t* out64,
                                 const std::uint32_t* r0,
                                 const std::uint32_t* r1,
                                 const std::uint32_t* r2,
                                 std::size_t coeff_len,
                                 p30_engine engine);

}  // namespace fft::ntt

// Generated constants for int_fft p30x3 NTT engines.
//
// Prime selection rules:
//   p < 2^30, p = c * RADIX * 2^K + 1, prime.
//   The engine cap is N_u32 = 8 * RADIX * 2^(K - 1).
//   Three primes are selected per RADIX so the CRT product covers the
//   coefficient bound at that cap with lazy-u32 SIMD arithmetic slack.
//   Within each engine the primes are ordered ascending for Garner CRT, so
//   the first residue is already reduced modulo the later two primes.

#include <cstdint>

namespace fft::ntt::detail {

struct p30_modular_set {
    char name;
    std::uint32_t radix;
    std::uint32_t max_n_u32;

    std::uint32_t p0;
    std::uint32_t p1;
    std::uint32_t p2;

    // Garner constants in normal representation.
    std::uint32_t inv_p0_mod_p1;
    std::uint32_t p0_mod_p2;
    std::uint32_t inv_p0p1_mod_p2;

    // Barrett reciprocals for constant modular multiplication:
    //   floor(c * 2^32 / p)
    std::uint32_t inv_p0_mod_p1_rec;
    std::uint32_t p0_mod_p2_rec;
    std::uint32_t inv_p0p1_mod_p2_rec;

    std::uint64_t p0p1;
};

inline constexpr p30_modular_set P30_ENGINES[] = {
    {
        'A', 1u, 33554432u,
        880803841u, 897581057u, 998244353u,
        448790582u, 880803841u, 41593599u,
        2147483903u, 3789677026u, 178957333u,
        0x0af8c0006a000001ULL,
    },
    {
        'B', 3u, 25165824u,
        962592769u, 975175681u, 1012924417u,
        487587918u, 962592769u, 991822365u,
        2147483989u, 4081552772u, 4205491100u,
        0x0d06ec0073800001ULL,
    },
    {
        'C', 5u, 41943040u,
        943718401u, 975175681u, 985661441u,
        31u, 943718401u, 2209u,
        136u, 4112202730u, 9625u,
        0x0cc5880072600001ULL,
    },
};

}  // namespace fft::ntt::detail

#include <immintrin.h>

#include <cassert>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace fft::ntt {
namespace {

#if defined(_MSC_VER)
#  define NTT_INLINE __forceinline
#  define NTT_RESTRICT __restrict
#else
#  define NTT_INLINE __attribute__((always_inline)) inline
#  define NTT_RESTRICT __restrict__
#endif

using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u128 = unsigned __int128;
using vec = __m256i;
static constexpr u64 MASK32 = 0xffffffffULL;

struct prime_plan {
    u32 mod;
    u32 mod2;
    u32 niv;
    u32 one;
    u32 r2;
    u32 primitive;
    unsigned max_log;
};

NTT_INLINE vec loadu32x8(const u32* p) {
    return _mm256_loadu_si256(reinterpret_cast<const vec*>(p));
}

NTT_INLINE void storeu32x8(u32* p, vec x) {
    _mm256_storeu_si256(reinterpret_cast<vec*>(p), x);
}

NTT_INLINE vec splat32(u32 x) {
    return _mm256_set1_epi32(static_cast<int>(x));
}

NTT_INLINE vec low32(vec x) {
    return _mm256_and_si256(x, _mm256_set1_epi64x(static_cast<long long>(MASK32)));
}

NTT_INLINE vec reduce_once(vec x, u32 mod) {
    const vec m = splat32(mod);
    return _mm256_min_epu32(x, _mm256_sub_epi32(x, m));
}

NTT_INLINE vec add_mod_vec(vec a, vec b, const prime_plan& p) {
    return reduce_once(_mm256_add_epi32(a, b), p.mod);
}

NTT_INLINE vec sub_mod_vec(vec a, vec b, const prime_plan& p) {
    const vec d = _mm256_sub_epi32(a, b);
    return _mm256_min_epu32(d, _mm256_add_epi32(d, splat32(p.mod)));
}

NTT_INLINE vec mont_reduce_vec(vec even, vec odd, const prime_plan& p) {
    const vec vmod = splat32(p.mod);
    const vec vniv = splat32(p.niv);
    vec ce = _mm256_mul_epu32(even, vniv);
    vec co = _mm256_mul_epu32(odd, vniv);
    ce = _mm256_mul_epu32(ce, vmod);
    co = _mm256_mul_epu32(co, vmod);
    return _mm256_blend_epi32(
        _mm256_srli_epi64(_mm256_add_epi64(even, ce), 32),
        _mm256_add_epi64(odd, co),
        0xaa);
}

NTT_INLINE vec mont_mul_vec(vec a, vec b, const prime_plan& p) {
    const vec even = _mm256_mul_epu32(a, b);
    const vec odd = _mm256_mul_epu32(_mm256_srli_epi64(a, 32),
                                     _mm256_srli_epi64(b, 32));
    return reduce_once(mont_reduce_vec(even, odd, p), p.mod);
}

NTT_INLINE vec mont_mul_vec_broadcast(vec a, u32 b, const prime_plan& p) {
    const vec vb = splat32(b);
    const vec even = _mm256_mul_epu32(a, vb);
    const vec odd = _mm256_mul_epu32(_mm256_srli_epi64(a, 32), vb);
    return reduce_once(mont_reduce_vec(even, odd, p), p.mod);
}

// c_rec = floor(c * 2^32 / mod). For any u32 lane a and c < mod < 2^30,
// q = high32(a*c_rec) underestimates floor(a*c/mod) by at most one, so
// low32(a*c - q*mod) is already in [0, 2*mod).
NTT_INLINE vec barrett_mul_const_noredc(vec a, u32 c, u32 c_rec, u32 mod) {
    const vec vrec = splat32(c_rec);
    const vec q_even = _mm256_srli_epi64(_mm256_mul_epu32(a, vrec), 32);
    const vec q_odd = _mm256_slli_epi64(
        _mm256_srli_epi64(_mm256_mul_epu32(_mm256_srli_epi64(a, 32), vrec), 32),
        32);
    const vec q = _mm256_blend_epi32(q_even, q_odd, 0xaa);
    return _mm256_sub_epi32(_mm256_mullo_epi32(a, splat32(c)),
                            _mm256_mullo_epi32(q, splat32(mod)));
}

NTT_INLINE vec barrett_mul_const(vec a, u32 c, u32 c_rec, u32 mod) {
    return reduce_once(barrett_mul_const_noredc(a, c, c_rec, mod), mod);
}

NTT_INLINE const detail::p30_modular_set& modulars(p30_engine engine) {
    const unsigned idx = static_cast<unsigned>(engine);
    assert(idx < 3u);
    return detail::P30_ENGINES[idx < 3u ? idx : 0u];
}

NTT_INLINE u32 prime_at(const detail::p30_modular_set& e, unsigned prime_index) {
    return prime_index == 0 ? e.p0 : (prime_index == 1 ? e.p1 : e.p2);
}

inline unsigned ctz_u32(u32 x) {
#if defined(_MSC_VER)
    unsigned long r = 0;
    _BitScanForward(&r, x);
    return unsigned(r);
#else
    return unsigned(__builtin_ctz(x));
#endif
}

inline bool is_pow2(std::size_t n) {
    return n != 0 && (n & (n - 1)) == 0;
}

inline u32 compute_niv(u32 mod) {
    u32 n = 2u + mod;
    for (int i = 0; i < 4; ++i) n *= 2u + mod * n;
    return n;
}

inline u32 mont_reduce_scalar(u64 x, const prime_plan& p) {
    return u32((x + u64(u32(x) * p.niv) * p.mod) >> 32);
}

inline u32 shrink_scalar(u32 x, u32 mod) {
    return x >= mod ? x - mod : x;
}

inline u32 add_mod_scalar(u32 a, u32 b, u32 mod) {
    return shrink_scalar(a + b, mod);
}

inline u32 sub_mod_scalar(u32 a, u32 b, u32 mod) {
    return a >= b ? a - b : a + mod - b;
}

inline u32 mont_mul_scalar(u32 a, u32 b, const prime_plan& p) {
    return shrink_scalar(mont_reduce_scalar(u64(a) * b, p), p.mod);
}

inline u32 to_mont_scalar(u32 x, const prime_plan& p) {
    return mont_mul_scalar(x, p.r2, p);
}

inline u32 pow_mod_u32(u32 a, u64 e, u32 mod) {
    u64 r = 1;
    u64 x = a;
    while (e != 0) {
        if (e & 1u) r = (u128(r) * x) % mod;
        x = (u128(x) * x) % mod;
        e >>= 1;
    }
    return u32(r);
}

inline u32 inv_mod_u32(u32 a, u32 mod) {
    return pow_mod_u32(a, mod - 2u, mod);
}

inline u32 primitive_root_u32(u32 mod) {
    u32 factors[16];
    unsigned nf = 0;
    u32 n = mod - 1u;
    if ((n & 1u) == 0) {
        factors[nf++] = 2;
        while ((n & 1u) == 0) n >>= 1;
    }
    for (u32 d = 3; u64(d) * d <= n; d += 2) {
        if (n % d == 0) {
            factors[nf++] = d;
            while (n % d == 0) n /= d;
        }
    }
    if (n > 1) factors[nf++] = n;

    for (u32 g = 2; g < mod; ++g) {
        bool ok = true;
        for (unsigned i = 0; i < nf; ++i) {
            if (pow_mod_u32(g, (mod - 1u) / factors[i], mod) == 1u) {
                ok = false;
                break;
            }
        }
        if (ok) return g;
    }
    return 0;
}

prime_plan make_prime_plan(u32 mod) {
    prime_plan p{};
    p.mod = mod;
    p.mod2 = 2u * mod;
    p.niv = compute_niv(mod);
    p.one = u32((u64{1} << 32) % mod);
    p.r2 = u32((u64(0) - u64(mod)) % mod);
    p.primitive = primitive_root_u32(mod);
    p.max_log = ctz_u32(mod - 1u);
    return p;
}

const prime_plan& get_prime_plan(p30_engine engine, unsigned prime_index) {
    static const prime_plan plans[3][3] = {
        {
            make_prime_plan(detail::P30_ENGINES[0].p0),
            make_prime_plan(detail::P30_ENGINES[0].p1),
            make_prime_plan(detail::P30_ENGINES[0].p2),
        },
        {
            make_prime_plan(detail::P30_ENGINES[1].p0),
            make_prime_plan(detail::P30_ENGINES[1].p1),
            make_prime_plan(detail::P30_ENGINES[1].p2),
        },
        {
            make_prime_plan(detail::P30_ENGINES[2].p0),
            make_prime_plan(detail::P30_ENGINES[2].p1),
            make_prime_plan(detail::P30_ENGINES[2].p2),
        },
    };
    return plans[static_cast<unsigned>(engine)][prime_index];
}

inline bool valid_ntt_size(std::size_t n, const prime_plan& p) {
    return is_pow2(n) && n <= (std::size_t{1} << p.max_log);
}

vec make_twiddle_lanes(u32 step_mont, const prime_plan& p) {
    alignas(32) u32 lanes[8];
    u32 w = p.one;
    for (unsigned i = 0; i < 8; ++i) {
        lanes[i] = w;
        w = mont_mul_scalar(w, step_mont, p);
    }
    return _mm256_load_si256(reinterpret_cast<const vec*>(lanes));
}

vec make_repeating_twiddle_lanes(u32 step_mont, unsigned period,
                                 const prime_plan& p) {
    alignas(32) u32 lanes[8];
    u32 w = p.one;
    for (unsigned i = 0; i < period; ++i) {
        lanes[i] = w;
        w = mont_mul_scalar(w, step_mont, p);
    }
    for (unsigned i = period; i < 8; ++i) lanes[i] = lanes[i % period];
    return _mm256_load_si256(reinterpret_cast<const vec*>(lanes));
}

NTT_INLINE vec combine_128_low(vec lo, vec hi) {
    return _mm256_permute2x128_si256(lo, hi, 0x20);
}

NTT_INLINE vec forward_inlane_half1(vec x, const prime_plan& p) {
    const vec y = _mm256_shuffle_epi32(x, 0xb1);
    const vec sum = add_mod_vec(x, y, p);
    const vec diff = sub_mod_vec(x, y, p);
    return _mm256_blend_epi32(sum, _mm256_shuffle_epi32(diff, 0xb1), 0xaa);
}

NTT_INLINE vec forward_inlane_half2(vec x, vec tw, const prime_plan& p) {
    const vec y = _mm256_shuffle_epi32(x, 0x4e);
    const vec sum = add_mod_vec(x, y, p);
    const vec diff = mont_mul_vec(sub_mod_vec(x, y, p), tw, p);
    return _mm256_blend_epi32(sum, _mm256_shuffle_epi32(diff, 0x4e), 0xcc);
}

NTT_INLINE vec forward_inlane_half4(vec x, vec tw, const prime_plan& p) {
    const vec y = _mm256_permute2x128_si256(x, x, 0x01);
    const vec sum = add_mod_vec(x, y, p);
    const vec diff = mont_mul_vec(sub_mod_vec(x, y, p), tw, p);
    return combine_128_low(sum, diff);
}

void forward_inlane8(u32* data, std::size_t n, u32 root, const prime_plan& p) {
    const vec tw4 = make_repeating_twiddle_lanes(
        to_mont_scalar(pow_mod_u32(root, n / 8u, p.mod), p), 4, p);
    const vec tw2 = make_repeating_twiddle_lanes(
        to_mont_scalar(pow_mod_u32(root, n / 4u, p.mod), p), 2, p);

    for (std::size_t i = 0; i < n; i += 8) {
        vec x = loadu32x8(data + i);
        x = forward_inlane_half4(x, tw4, p);
        x = forward_inlane_half2(x, tw2, p);
        x = forward_inlane_half1(x, p);
        storeu32x8(data + i, x);
    }
}

NTT_INLINE vec inverse_inlane_half1(vec x, const prime_plan& p) {
    return forward_inlane_half1(x, p);
}

NTT_INLINE vec inverse_inlane_half2(vec x, vec tw, const prime_plan& p) {
    const vec y = _mm256_shuffle_epi32(x, 0x4e);
    const vec v = mont_mul_vec(y, tw, p);
    const vec sum = add_mod_vec(x, v, p);
    const vec diff = sub_mod_vec(x, v, p);
    return _mm256_blend_epi32(sum, _mm256_shuffle_epi32(diff, 0x4e), 0xcc);
}

NTT_INLINE vec inverse_inlane_half4(vec x, vec tw, const prime_plan& p) {
    const vec y = _mm256_permute2x128_si256(x, x, 0x01);
    const vec v = mont_mul_vec(y, tw, p);
    const vec sum = add_mod_vec(x, v, p);
    const vec diff = sub_mod_vec(x, v, p);
    return combine_128_low(sum, diff);
}

void inverse_inlane8(u32* data, std::size_t n, u32 inv_root, const prime_plan& p) {
    const vec tw2 = make_repeating_twiddle_lanes(
        to_mont_scalar(pow_mod_u32(inv_root, n / 4u, p.mod), p), 2, p);
    const vec tw4 = make_repeating_twiddle_lanes(
        to_mont_scalar(pow_mod_u32(inv_root, n / 8u, p.mod), p), 4, p);

    for (std::size_t i = 0; i < n; i += 8) {
        vec x = loadu32x8(data + i);
        x = inverse_inlane_half1(x, p);
        x = inverse_inlane_half2(x, tw2, p);
        x = inverse_inlane_half4(x, tw4, p);
        storeu32x8(data + i, x);
    }
}

void forward_radix4_pass(u32* data, std::size_t n, std::size_t half,
                         u32 root, u32 img_mont, const prime_plan& p) {
    const std::size_t quarter = half >> 1;
    const u32 step_std = pow_mod_u32(root, n / (2u * half), p.mod);
    const u32 step = to_mont_scalar(step_std, p);
    const u32 step8 = to_mont_scalar(pow_mod_u32(step_std, 8, p.mod), p);
    const u32 step2 = to_mont_scalar(pow_mod_u32(step_std, 2, p.mod), p);
    const u32 step16 = to_mont_scalar(pow_mod_u32(step_std, 16, p.mod), p);
    const vec step8v = splat32(step8);
    const vec step16v = splat32(step16);
    const vec w1_base = make_twiddle_lanes(step, p);
    const vec w2_base = make_twiddle_lanes(step2, p);
    const vec wi_base = mont_mul_vec_broadcast(w1_base, img_mont, p);

    for (std::size_t base = 0; base < n; base += 2u * half) {
        vec w1 = w1_base;
        vec w2 = w2_base;
        vec wi = wi_base;
        for (std::size_t j = 0; j < quarter; j += 8) {
            const vec a0 = loadu32x8(data + base + j);
            const vec a1 = loadu32x8(data + base + j + quarter);
            const vec a2 = loadu32x8(data + base + j + half);
            const vec a3 = loadu32x8(data + base + j + half + quarter);

            const vec b0 = add_mod_vec(a0, a2, p);
            const vec b2 = mont_mul_vec(sub_mod_vec(a0, a2, p), w1, p);
            const vec b1 = add_mod_vec(a1, a3, p);
            const vec b3 = mont_mul_vec(sub_mod_vec(a1, a3, p), wi, p);

            storeu32x8(data + base + j, add_mod_vec(b0, b1, p));
            storeu32x8(data + base + j + quarter,
                       mont_mul_vec(sub_mod_vec(b0, b1, p), w2, p));
            storeu32x8(data + base + j + half, add_mod_vec(b2, b3, p));
            storeu32x8(data + base + j + half + quarter,
                       mont_mul_vec(sub_mod_vec(b2, b3, p), w2, p));

            w1 = mont_mul_vec_broadcast(w1, step8, p);
            w2 = mont_mul_vec_broadcast(w2, step16, p);
            wi = mont_mul_vec_broadcast(wi, step8, p);
        }
    }
}

void forward_radix2_pass(u32* data, std::size_t n, std::size_t half,
                         u32 root, const prime_plan& p) {
    const u32 step_std = pow_mod_u32(root, n / (2u * half), p.mod);
    const u32 step = to_mont_scalar(step_std, p);
    const u32 step8 = to_mont_scalar(pow_mod_u32(step_std, 8, p.mod), p);
    const vec tw_base = make_twiddle_lanes(step, p);

    for (std::size_t base = 0; base < n; base += 2u * half) {
        vec tw = tw_base;
        for (std::size_t j = 0; j < half; j += 8) {
            const vec u = loadu32x8(data + base + j);
            const vec v = loadu32x8(data + base + j + half);
            storeu32x8(data + base + j, add_mod_vec(u, v, p));
            storeu32x8(data + base + j + half,
                       mont_mul_vec(sub_mod_vec(u, v, p), tw, p));
            tw = mont_mul_vec_broadcast(tw, step8, p);
        }
    }
}

void inverse_radix4_pass(u32* data, std::size_t n, std::size_t half,
                         u32 inv_root, u32 inv_img_mont, const prime_plan& p) {
    const u32 step_std = pow_mod_u32(inv_root, n / (4u * half), p.mod);
    const u32 step = to_mont_scalar(step_std, p);
    const u32 step8 = to_mont_scalar(pow_mod_u32(step_std, 8, p.mod), p);
    const u32 step2 = to_mont_scalar(pow_mod_u32(step_std, 2, p.mod), p);
    const u32 step16 = to_mont_scalar(pow_mod_u32(step_std, 16, p.mod), p);
    const vec q_base = make_twiddle_lanes(step, p);
    const vec q2_base = make_twiddle_lanes(step2, p);
    const vec qi_base = mont_mul_vec_broadcast(q_base, inv_img_mont, p);

    for (std::size_t base = 0; base < n; base += 4u * half) {
        vec q = q_base;
        vec q2 = q2_base;
        vec qi = qi_base;
        for (std::size_t j = 0; j < half; j += 8) {
            const vec a0 = loadu32x8(data + base + j);
            const vec a1 = loadu32x8(data + base + j + half);
            const vec a2 = loadu32x8(data + base + j + 2u * half);
            const vec a3 = loadu32x8(data + base + j + 3u * half);

            const vec t1 = mont_mul_vec(a1, q2, p);
            const vec t3 = mont_mul_vec(a3, q2, p);
            const vec b0 = add_mod_vec(a0, t1, p);
            const vec b1 = sub_mod_vec(a0, t1, p);
            const vec b2 = add_mod_vec(a2, t3, p);
            const vec b3 = sub_mod_vec(a2, t3, p);
            const vec v0 = mont_mul_vec(b2, q, p);
            const vec v1 = mont_mul_vec(b3, qi, p);

            storeu32x8(data + base + j, add_mod_vec(b0, v0, p));
            storeu32x8(data + base + j + half, add_mod_vec(b1, v1, p));
            storeu32x8(data + base + j + 2u * half, sub_mod_vec(b0, v0, p));
            storeu32x8(data + base + j + 3u * half, sub_mod_vec(b1, v1, p));

            q = mont_mul_vec_broadcast(q, step8, p);
            q2 = mont_mul_vec_broadcast(q2, step16, p);
            qi = mont_mul_vec_broadcast(qi, step8, p);
        }
    }
}

void inverse_radix2_pass(u32* data, std::size_t n, std::size_t half,
                         u32 inv_root, const prime_plan& p) {
    const u32 step_std = pow_mod_u32(inv_root, n / (2u * half), p.mod);
    const u32 step = to_mont_scalar(step_std, p);
    const u32 step8 = to_mont_scalar(pow_mod_u32(step_std, 8, p.mod), p);
    const vec tw_base = make_twiddle_lanes(step, p);

    for (std::size_t base = 0; base < n; base += 2u * half) {
        vec tw = tw_base;
        for (std::size_t j = 0; j < half; j += 8) {
            const vec u = loadu32x8(data + base + j);
            const vec v = mont_mul_vec(loadu32x8(data + base + j + half),
                                       tw, p);
            storeu32x8(data + base + j, add_mod_vec(u, v, p));
            storeu32x8(data + base + j + half, sub_mod_vec(u, v, p));
            tw = mont_mul_vec_broadcast(tw, step8, p);
        }
    }
}

void forward_dif_scalar_tail(u32* data, std::size_t n, std::size_t half,
                             u32 root, const prime_plan& p) {
    for (; half != 0; half >>= 1) {
        const u32 step = to_mont_scalar(
            pow_mod_u32(root, n / (2u * half), p.mod), p);
        for (std::size_t base = 0; base < n; base += 2u * half) {
            u32 w = p.one;
            for (std::size_t j = 0; j < half; ++j) {
                const u32 u = data[base + j];
                const u32 v = data[base + j + half];
                data[base + j] = add_mod_scalar(u, v, p.mod);
                data[base + j + half] =
                    mont_mul_scalar(sub_mod_scalar(u, v, p.mod), w, p);
                w = mont_mul_scalar(w, step, p);
            }
        }
    }
}

void inverse_dit_scalar_prefix(u32* data, std::size_t n, std::size_t last_half,
                               u32 inv_root, const prime_plan& p) {
    for (std::size_t half = 1; half < last_half; half <<= 1) {
        const u32 step = to_mont_scalar(
            pow_mod_u32(inv_root, n / (2u * half), p.mod), p);
        for (std::size_t base = 0; base < n; base += 2u * half) {
            u32 w = p.one;
            for (std::size_t j = 0; j < half; ++j) {
                const u32 u = data[base + j];
                const u32 v = mont_mul_scalar(data[base + j + half], w, p);
                data[base + j] = add_mod_scalar(u, v, p.mod);
                data[base + j + half] = sub_mod_scalar(u, v, p.mod);
                w = mont_mul_scalar(w, step, p);
            }
        }
    }
}

void ntt_forward_pow2(u32* data, std::size_t n, const prime_plan& p) {
    const u32 root = pow_mod_u32(p.primitive, (p.mod - 1u) / n, p.mod);
    if (n < 8) {
        forward_dif_scalar_tail(data, n, n >> 1, root, p);
        return;
    }

    std::vector<vec> twiddles((std::max)(std::size_t{1}, n >> 4));
    std::vector<vec> twiddles2((std::max)(std::size_t{1}, n >> 5));
    std::vector<vec> twiddles_i((std::max)(std::size_t{1}, n >> 5));
    const u32 img_mont = to_mont_scalar(pow_mod_u32(root, n / 4u, p.mod), p);

    std::size_t half = n >> 1;
    for (; half >= 16; half >>= 2) {
        forward_radix4_pass(data, n, half, root, img_mont, p,
                            twiddles, twiddles2, twiddles_i);
    }
    if (half >= 8) forward_radix2_pass(data, n, half, root, p, twiddles);
    forward_inlane8(data, n, root, p);
}

void ntt_inverse_pow2(u32* data, std::size_t n, const prime_plan& p) {
    const u32 root = pow_mod_u32(p.primitive, (p.mod - 1u) / n, p.mod);
    const u32 inv_root = inv_mod_u32(root, p.mod);
    if (n < 8) {
        inverse_dit_scalar_prefix(data, n, n, inv_root, p);
        const u32 inv_n = inv_mod_u32(u32(n % p.mod), p.mod);
        const u32 inv_n_mont = to_mont_scalar(inv_n, p);
        for (std::size_t i = 0; i < n; ++i) {
            data[i] = mont_mul_scalar(data[i], inv_n_mont, p);
        }
        return;
    }

    std::vector<vec> twiddles((std::max)(std::size_t{1}, n >> 4));
    std::vector<vec> twiddles2((std::max)(std::size_t{1}, n >> 5));
    std::vector<vec> twiddles_i((std::max)(std::size_t{1}, n >> 5));
    const u32 inv_img_mont = to_mont_scalar(pow_mod_u32(inv_root, n / 4u, p.mod), p);

    inverse_inlane8(data, n, inv_root, p);
    std::size_t half = 8;
    for (; half * 4u <= n; half <<= 2) {
        inverse_radix4_pass(data, n, half, inv_root, inv_img_mont, p,
                            twiddles, twiddles2, twiddles_i);
    }
    if (half < n) inverse_radix2_pass(data, n, half, inv_root, p, twiddles);

    const u32 inv_n = inv_mod_u32(u32(n % p.mod), p.mod);
    const u32 inv_n_mont = to_mont_scalar(inv_n, p);
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        storeu32x8(data + i, mont_mul_vec_broadcast(loadu32x8(data + i), inv_n_mont, p));
    }
    for (; i < n; ++i) data[i] = mont_mul_scalar(data[i], inv_n_mont, p);
}

void pointwise_mul(u32* out, const u32* a, const u32* b,
                   std::size_t n, const prime_plan& p) {
    std::size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const vec av = loadu32x8(a + i);
        const vec bv_mont = mont_mul_vec_broadcast(loadu32x8(b + i), p.r2, p);
        storeu32x8(out + i, mont_mul_vec(av, bv_mont, p));
    }
    for (; i < n; ++i) {
        const u32 bm = mont_mul_scalar(b[i], p.r2, p);
        out[i] = mont_mul_scalar(a[i], bm, p);
    }
}

struct pair64x4 {
    vec lo;
    vec hi;
};

NTT_INLINE pair64x4 compose_pairs_8(vec x0, vec t1, vec t2,
                                    const detail::p30_modular_set& e) {
    const vec p0 = _mm256_set1_epi64x(static_cast<long long>(e.p0));
    const vec p01_lo = _mm256_set1_epi64x(static_cast<long long>(u32(e.p0p1)));
    const vec p01_hi = _mm256_set1_epi64x(static_cast<long long>(u32(e.p0p1 >> 32)));

    const vec x1 = _mm256_srli_epi64(x0, 32);
    const vec t1_odd = _mm256_srli_epi64(t1, 32);
    const vec t2_odd = _mm256_srli_epi64(t2, 32);

    vec Lo0 = _mm256_add_epi64(low32(x0), _mm256_mul_epu32(t1, p0));
    Lo0 = _mm256_add_epi64(Lo0, _mm256_mul_epu32(t2, p01_lo));
    vec Hi0 = _mm256_mul_epu32(t2, p01_hi);

    vec Lo1 = _mm256_add_epi64(x1, _mm256_mul_epu32(t1_odd, p0));
    Lo1 = _mm256_add_epi64(Lo1, _mm256_mul_epu32(t2_odd, p01_lo));
    vec Hi1 = _mm256_mul_epu32(t2_odd, p01_hi);

    Hi0 = _mm256_add_epi64(Hi0, _mm256_srli_epi64(Lo0, 32));
    Hi1 = _mm256_add_epi64(Hi1, _mm256_srli_epi64(Lo1, 32));
    Lo1 = low32(Lo1);

    const vec mid = _mm256_add_epi64(Hi0, Lo1);
    const vec pair_lo = _mm256_blend_epi32(Lo0, _mm256_slli_epi64(mid, 32), 0xaa);
    const vec pair_hi = _mm256_add_epi64(Hi1, _mm256_srli_epi64(mid, 32));

    return {pair_lo, pair_hi};
}

NTT_INLINE void propagate_pairs_4_u64(u64* out64, std::size_t out_i,
                                      u128& carry, pair64x4 pairs) {
    alignas(32) u64 pl[4];
    alignas(32) u64 ph[4];
    _mm256_store_si256(reinterpret_cast<vec*>(pl), pairs.lo);
    _mm256_store_si256(reinterpret_cast<vec*>(ph), pairs.hi);

    for (unsigned j = 0; j < 4; ++j) {
        carry += u128(pl[j]) + (u128(ph[j]) << 64);
        out64[out_i + j] = u64(carry);
        carry >>= 64;
    }
}

NTT_INLINE void carry_pairs_8_u64(u64* out64, std::size_t out_i, u128& carry,
                                  vec x0, vec t1, vec t2,
                                  const detail::p30_modular_set& e) {
    propagate_pairs_4_u64(out64, out_i, carry,
                          compose_pairs_8(x0, t1, t2, e));
}

NTT_INLINE u128 recover_garner_scalar(const detail::p30_modular_set& e,
                                      u64 x0, u64 x1, u64 x2) {
    const u64 d1 = x1 >= x0 ? x1 - x0 : x1 + e.p1 - x0;
    const u64 s1 = u64((u128(d1) * e.inv_p0_mod_p1) % e.p1);

    const u64 p0t1 = u64((u128(e.p0_mod_p2) * s1) % e.p2);
    const u64 u = x0 + p0t1 >= e.p2 ? x0 + p0t1 - e.p2 : x0 + p0t1;
    const u64 d2 = x2 >= u ? x2 - u : x2 + e.p2 - u;
    const u64 s2 = u64((u128(d2) * e.inv_p0p1_mod_p2) % e.p2);

    return u128(x0) + u128(e.p0) * u128(s1) + u128(e.p0p1) * u128(s2);
}

void crt_p30_garner_u64_engine(u64* NTT_RESTRICT out64,
                               const u32* NTT_RESTRICT r0,
                               const u32* NTT_RESTRICT r1,
                               const u32* NTT_RESTRICT r2,
                               std::size_t coeff_len,
                               const detail::p30_modular_set& e,
                               u128& carry) {
    std::size_t i = 0;
    for (; i + 8 <= coeff_len; i += 8) {
        const vec x0 = loadu32x8(r0 + i);
        const vec x1 = loadu32x8(r1 + i);
        const vec x2 = loadu32x8(r2 + i);

        const vec d1 = _mm256_add_epi32(_mm256_sub_epi32(x1, x0), splat32(e.p1));
        const vec t1 = reduce_once(
            barrett_mul_const_noredc(d1, e.inv_p0_mod_p1,
                                     e.inv_p0_mod_p1_rec, e.p1),
            e.p1);

        const vec p0t1_mod_p2 = barrett_mul_const_noredc(t1, e.p0_mod_p2,
                                                         e.p0_mod_p2_rec, e.p2);
        const vec u = _mm256_add_epi32(x0, p0t1_mod_p2);
        const vec d2 = _mm256_sub_epi32(_mm256_add_epi32(x2, splat32(3u * e.p2)), u);
        const vec t2 = barrett_mul_const(d2, e.inv_p0p1_mod_p2,
                                          e.inv_p0p1_mod_p2_rec, e.p2);

        carry_pairs_8_u64(out64, i >> 1, carry, x0, t1, t2, e);
    }

    for (; i < coeff_len; i += 2) {
        u128 result = recover_garner_scalar(e, r0[i], r1[i], r2[i]);
        if (i + 1 < coeff_len) {
            result += recover_garner_scalar(e, r0[i + 1], r1[i + 1], r2[i + 1]) << 32;
        }

        carry += result;
        out64[i >> 1] = u64(carry);
        carry >>= 64;
    }
}

}  // namespace

p30_engine_info get_p30_engine_info(p30_engine engine) {
    const auto& e = modulars(engine);
    return {e.name, e.radix, e.max_n_u32, e.p0, e.p1, e.p2};
}

std::uint32_t p30_modulus(p30_engine engine, unsigned prime_index) {
    if (prime_index >= 3) return 0;
    return prime_at(modulars(engine), prime_index);
}

bool ntt_p30_reduce_and_pad(std::uint32_t* out,
                            const std::uint32_t* src,
                            std::size_t src_len,
                            std::size_t n,
                            p30_engine engine,
                            unsigned prime_index) {
    if (!out || (!src && src_len != 0) || prime_index >= 3) return false;
    const prime_plan& p = get_prime_plan(engine, prime_index);
    if (!valid_ntt_size(n, p) || src_len > n) return false;

    std::size_t i = 0;
    for (; i < src_len; ++i) out[i] = u32(u64(src[i]) % p.mod);
    if (i < n) std::memset(out + i, 0, (n - i) * sizeof(u32));
    return true;
}

bool ntt_p30_forward(std::uint32_t* data,
                     std::size_t n,
                     p30_engine engine,
                     unsigned prime_index) {
    if (!data || prime_index >= 3) return false;
    const prime_plan& p = get_prime_plan(engine, prime_index);
    if (!valid_ntt_size(n, p)) return false;
    ntt_forward_pow2(data, n, p);
    return true;
}

bool ntt_p30_inverse(std::uint32_t* data,
                     std::size_t n,
                     p30_engine engine,
                     unsigned prime_index) {
    if (!data || prime_index >= 3) return false;
    const prime_plan& p = get_prime_plan(engine, prime_index);
    if (!valid_ntt_size(n, p)) return false;
    ntt_inverse_pow2(data, n, p);
    return true;
}

bool ntt_p30_pointwise_mul(std::uint32_t* out,
                           const std::uint32_t* a,
                           const std::uint32_t* b,
                           std::size_t n,
                           p30_engine engine,
                           unsigned prime_index) {
    if (!out || !a || !b || prime_index >= 3) return false;
    const prime_plan& p = get_prime_plan(engine, prime_index);
    if (!valid_ntt_size(n, p)) return false;
    pointwise_mul(out, a, b, n, p);
    return true;
}

bool ntt_p30_convolve(std::uint32_t* out,
                      const std::uint32_t* a,
                      std::size_t an,
                      const std::uint32_t* b,
                      std::size_t bn,
                      std::size_t n,
                      p30_engine engine,
                      unsigned prime_index) {
    if (!out || (!a && an != 0) || (!b && bn != 0) || prime_index >= 3) return false;
    const prime_plan& p = get_prime_plan(engine, prime_index);
    if (!valid_ntt_size(n, p) || an > n || bn > n) return false;

    std::vector<u32> lhs(n);
    std::vector<u32> rhs(n);
    if (!ntt_p30_reduce_and_pad(lhs.data(), a, an, n, engine, prime_index)) return false;
    if (!ntt_p30_reduce_and_pad(rhs.data(), b, bn, n, engine, prime_index)) return false;
    ntt_forward_pow2(lhs.data(), n, p);
    ntt_forward_pow2(rhs.data(), n, p);
    pointwise_mul(lhs.data(), lhs.data(), rhs.data(), n, p);
    ntt_inverse_pow2(lhs.data(), n, p);
    std::memcpy(out, lhs.data(), n * sizeof(u32));
    return true;
}

std::uint64_t crt_p30_garner_u64(std::uint64_t* NTT_RESTRICT out64,
                                 const std::uint32_t* NTT_RESTRICT r0,
                                 const std::uint32_t* NTT_RESTRICT r1,
                                 const std::uint32_t* NTT_RESTRICT r2,
                                 std::size_t coeff_len,
                                 p30_engine engine) {
    if (coeff_len == 0) return 0;
    if (!out64 || !r0 || !r1 || !r2) return ~std::uint64_t{0};

    const auto& e = modulars(engine);
    u128 carry = 0;
    crt_p30_garner_u64_engine(out64, r0, r1, r2, coeff_len, e, carry);
    return static_cast<std::uint64_t>(carry);
}

}  // namespace fft::ntt