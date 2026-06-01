#include "ifma52_mul.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <gmp.h>
#include <vector>

extern "C" void __gmpn_mul_basecase(mp_limb_t *rp, const mp_limb_t *up,
                                    mp_size_t un, const mp_limb_t *vp,
                                    mp_size_t vn);

using simd_bigint::reference::Decoded52;
using simd_bigint::reference::Ifma52Workspace;
using simd_bigint::reference::ToomWorkspace;
using simd_bigint::reference::decode52;
using simd_bigint::reference::mul_basecase_u64;
using simd_bigint::reference::mul_predecoded_u64;
using simd_bigint::reference::mul_toom22_u64;
using simd_bigint::reference::mul_toom32_u64;
using simd_bigint::reference::mul_toom33_u64;

namespace {

std::uint64_t
splitmix64(std::uint64_t &x)
{
  std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

void
gmp_mul_basecase_norm(std::uint64_t *rp, const std::uint64_t *ap,
                      std::size_t an, const std::uint64_t *bp,
                      std::size_t bn)
{
  if (an >= bn)
    __gmpn_mul_basecase(reinterpret_cast<mp_limb_t *>(rp),
                        reinterpret_cast<const mp_limb_t *>(ap),
                        static_cast<mp_size_t>(an),
                        reinterpret_cast<const mp_limb_t *>(bp),
                        static_cast<mp_size_t>(bn));
  else
    __gmpn_mul_basecase(reinterpret_cast<mp_limb_t *>(rp),
                        reinterpret_cast<const mp_limb_t *>(bp),
                        static_cast<mp_size_t>(bn),
                        reinterpret_cast<const mp_limb_t *>(ap),
                        static_cast<mp_size_t>(an));
}

template <class F>
double
bench_one(F &&fn, double seconds, int repeats)
{
  std::vector<double> samples;
  volatile std::uint64_t guard = 0;

  for (int r = 0; r < repeats; ++r) {
    std::uint64_t iters = 0;
    const auto begin = std::chrono::steady_clock::now();
    auto now = begin;
    do {
      for (unsigned k = 0; k < 32; ++k)
        guard ^= fn();
      iters += 32;
      now = std::chrono::steady_clock::now();
    } while (std::chrono::duration<double>(now - begin).count() < seconds);

    const double elapsed = std::chrono::duration<double>(now - begin).count();
    samples.push_back(elapsed * 1.0e9 / static_cast<double>(iters));
  }

  std::sort(samples.begin(), samples.end());
  if (guard == 0x12345678)
    std::fprintf(stderr, "%llu\n", static_cast<unsigned long long>(guard));
  return samples[samples.size() / 2];
}

} // namespace

int
main(int argc, char **argv)
{
  const int max_n = argc > 1 ? std::atoi(argv[1]) : 160;
  const double seconds = argc > 2 ? std::atof(argv[2]) : 0.003;
  const int repeats = argc > 3 ? std::atoi(argv[3]) : 3;
  constexpr unsigned cases = 16;

  std::uint64_t seed = 0x2718281828459045ULL;
  std::puts("n,ifma_full_ns,ifma_core_ns,toom22_ns,toom32_ns,toom33_ns,"
            "gmp_basecase_ns,ifma_full_speedup,ifma_core_speedup,"
            "toom22_speedup,toom32_speedup,toom33_speedup");

  for (int n = 6; n <= max_n; ++n) {
    std::vector<std::vector<std::uint64_t>> a(cases), b(cases), out(cases);
    std::vector<Decoded52> da(cases), db(cases);

    for (unsigned c = 0; c < cases; ++c) {
      a[c].resize(n);
      b[c].resize(n);
      out[c].resize(2 * n + 2);
      for (int i = 0; i < n; ++i) {
        a[c][i] = splitmix64(seed);
        b[c][i] = splitmix64(seed);
      }
      a[c].back() |= std::uint64_t{1} << 63;
      b[c].back() |= std::uint64_t{1} << 63;
      decode52(da[c], a[c].data(), n);
      decode52(db[c], b[c].data(), n);
    }

    Ifma52Workspace ws_full;
    Ifma52Workspace ws_core;
    ToomWorkspace ws_t22;
    ToomWorkspace ws_t32;
    ToomWorkspace ws_t33;
    unsigned idx_full = 0, idx_core = 0, idx_t22 = 0, idx_t32 = 0, idx_t33 = 0;
    unsigned idx_base = 0;

    const double ifma_full = bench_one([&]() -> std::uint64_t {
      const unsigned c = idx_full++ & (cases - 1);
      mul_basecase_u64(out[c].data(), a[c].data(), n,
                       b[(5 * c) & (cases - 1)].data(), n, ws_full);
      return out[c][0] ^ out[c][n] ^ out[c][2 * n - 1];
    }, seconds, repeats);

    const double ifma_core = bench_one([&]() -> std::uint64_t {
      const unsigned c = idx_core++ & (cases - 1);
      mul_predecoded_u64(out[c].data(), da[c], db[(5 * c) & (cases - 1)],
                         2 * n, ws_core);
      return out[c][0] ^ out[c][n] ^ out[c][2 * n - 1];
    }, seconds, repeats);

    const double toom22 = bench_one([&]() -> std::uint64_t {
      const unsigned c = idx_t22++ & (cases - 1);
      mul_toom22_u64(out[c].data(), a[c].data(), n,
                     b[(5 * c) & (cases - 1)].data(), n, ws_t22);
      return out[c][0] ^ out[c][n] ^ out[c][2 * n - 1];
    }, seconds, repeats);

    const double toom32 = bench_one([&]() -> std::uint64_t {
      const unsigned c = idx_t32++ & (cases - 1);
      mul_toom32_u64(out[c].data(), a[c].data(), n,
                     b[(5 * c) & (cases - 1)].data(), n, ws_t32);
      return out[c][0] ^ out[c][n] ^ out[c][2 * n - 1];
    }, seconds, repeats);

    const double toom33 = bench_one([&]() -> std::uint64_t {
      const unsigned c = idx_t33++ & (cases - 1);
      mul_toom33_u64(out[c].data(), a[c].data(), n,
                     b[(5 * c) & (cases - 1)].data(), n, ws_t33);
      return out[c][0] ^ out[c][n] ^ out[c][2 * n - 1];
    }, seconds, repeats);

    const double gmp_base = bench_one([&]() -> std::uint64_t {
      const unsigned c = idx_base++ & (cases - 1);
      gmp_mul_basecase_norm(out[c].data(), a[c].data(), n,
                            b[(5 * c) & (cases - 1)].data(), n);
      return out[c][0] ^ out[c][n] ^ out[c][2 * n - 1];
    }, seconds, repeats);

    std::printf("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n", n,
                ifma_full, ifma_core, toom22, toom32, toom33, gmp_base,
                gmp_base / ifma_full, gmp_base / ifma_core, gmp_base / toom22,
                gmp_base / toom32, gmp_base / toom33);
    std::fflush(stdout);
  }

  return 0;
}
