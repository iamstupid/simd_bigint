#include "ifma52_mul.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gmp.h>
#include <vector>

extern "C" void __gmpn_mul_basecase(mp_limb_t *rp, const mp_limb_t *up,
                                    mp_size_t un, const mp_limb_t *vp,
                                    mp_size_t vn);

using simd_bigint::reference::Ifma52Workspace;
using simd_bigint::reference::ToomWorkspace;
using simd_bigint::reference::mul_basecase_u64;
using simd_bigint::reference::mul_toom22_raw_u64;
using simd_bigint::reference::mul_toom22_u64;
using simd_bigint::reference::mul_toom32_raw_u64;
using simd_bigint::reference::mul_toom32_u64;
using simd_bigint::reference::mul_toom33_raw_u64;
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
  enum class Mode {
    toom22_public,
    toom22_raw,
    toom32_public,
    toom32_raw,
    toom33_public,
    toom33_raw,
  };

  Mode mode = Mode::toom32_public;
  int arg = 1;
  if (argc > 1) {
    if (std::strcmp(argv[1], "toom22-square") == 0) {
      mode = Mode::toom22_public;
      arg = 2;
    } else if (std::strcmp(argv[1], "toom22-square-raw") == 0) {
      mode = Mode::toom22_raw;
      arg = 2;
    } else if (std::strcmp(argv[1], "toom32-public") == 0) {
      mode = Mode::toom32_public;
      arg = 2;
    } else if (std::strcmp(argv[1], "toom32-raw") == 0) {
      mode = Mode::toom32_raw;
      arg = 2;
    } else if (std::strcmp(argv[1], "toom33-square") == 0) {
      mode = Mode::toom33_public;
      arg = 2;
    } else if (std::strcmp(argv[1], "toom33-square-raw") == 0) {
      mode = Mode::toom33_raw;
      arg = 2;
    }
  }

  const bool square =
    mode == Mode::toom22_public || mode == Mode::toom22_raw ||
    mode == Mode::toom33_public || mode == Mode::toom33_raw;
  const bool raw =
    mode == Mode::toom22_raw || mode == Mode::toom32_raw || mode == Mode::toom33_raw;
  const char *label = "toom32";
  if (mode == Mode::toom22_public || mode == Mode::toom22_raw)
    label = raw ? "toom22_raw" : "toom22";
  else if (mode == Mode::toom32_raw)
    label = "toom32_raw";
  else if (mode == Mode::toom33_public || mode == Mode::toom33_raw)
    label = raw ? "toom33_raw" : "toom33";

  const int max_k = argc > arg ? std::atoi(argv[arg]) : 96;
  const double seconds = argc > arg + 1 ? std::atof(argv[arg + 1]) : 0.003;
  const int repeats = argc > arg + 2 ? std::atoi(argv[arg + 2]) : 3;
  const int min_k = argc > arg + 3 ? std::atoi(argv[arg + 3]) : 4;
  constexpr unsigned cases = 16;

  std::uint64_t seed = 0x9e3779b97f4a7c15ULL;
  if (square)
    std::printf("n,ifma_full_ns,%s_ns,gmp_basecase_ns,ifma_full_speedup,"
                "%s_speedup,scratch_allocs,scratch_bytes,scratch_peak_bytes\n",
                label, label);
  else
    std::printf("k,an,bn,ifma_full_ns,%s_ns,gmp_basecase_ns,ifma_full_speedup,"
                "%s_speedup,scratch_allocs,scratch_bytes,scratch_peak_bytes\n",
                label, label);

  for (int k = min_k; k <= max_k; ++k) {
    const std::size_t an = square ? static_cast<std::size_t>(k)
                                  : static_cast<std::size_t>(3 * k);
    const std::size_t bn = square ? static_cast<std::size_t>(k)
                                  : static_cast<std::size_t>(2 * k);
    std::vector<std::vector<std::uint64_t>> a(cases), b(cases), out(cases);

    for (unsigned c = 0; c < cases; ++c) {
      a[c].resize(an);
      b[c].resize(bn);
      out[c].resize(an + bn + 2);
      for (std::size_t i = 0; i < an; ++i)
        a[c][i] = splitmix64(seed);
      for (std::size_t i = 0; i < bn; ++i)
        b[c][i] = splitmix64(seed);
      a[c].back() |= std::uint64_t{1} << 63;
      b[c].back() |= std::uint64_t{1} << 63;
    }

    Ifma52Workspace ws_full;
    ToomWorkspace ws_toom;
    unsigned idx_full = 0, idx_t32 = 0, idx_base = 0;

    auto run_toom = [&](std::uint64_t *dst, const std::uint64_t *lhs,
                        const std::uint64_t *rhs,
                        ToomWorkspace &workspace) {
      if (mode == Mode::toom22_public)
        mul_toom22_u64(dst, lhs, an, rhs, bn, workspace);
      else if (mode == Mode::toom22_raw)
        mul_toom22_raw_u64(dst, lhs, an, rhs, bn, workspace);
      else if (mode == Mode::toom33_public)
        mul_toom33_u64(dst, lhs, an, rhs, bn, workspace);
      else if (mode == Mode::toom33_raw)
        mul_toom33_raw_u64(dst, lhs, an, rhs, bn, workspace);
      else if (mode == Mode::toom32_raw)
        mul_toom32_raw_u64(dst, lhs, an, rhs, bn, workspace);
      else
        mul_toom32_u64(dst, lhs, an, rhs, bn, workspace);
    };

    const double ifma_full = bench_one([&]() -> std::uint64_t {
      const unsigned c = idx_full++ & (cases - 1);
      mul_basecase_u64(out[c].data(), a[c].data(), an,
                       b[(5 * c) & (cases - 1)].data(), bn, ws_full);
      return out[c][0] ^ out[c][bn] ^ out[c][an + bn - 1];
    }, seconds, repeats);

    const double toom = bench_one([&]() -> std::uint64_t {
      const unsigned c = idx_t32++ & (cases - 1);
      run_toom(out[c].data(), a[c].data(), b[(5 * c) & (cases - 1)].data(),
               ws_toom);
      return out[c][0] ^ out[c][bn] ^ out[c][an + bn - 1];
    }, seconds, repeats);

    const double gmp_base = bench_one([&]() -> std::uint64_t {
      const unsigned c = idx_base++ & (cases - 1);
      gmp_mul_basecase_norm(out[c].data(), a[c].data(), an,
                            b[(5 * c) & (cases - 1)].data(), bn);
      return out[c][0] ^ out[c][bn] ^ out[c][an + bn - 1];
    }, seconds, repeats);

    ToomWorkspace scratch_probe;
    std::vector<std::uint64_t> scratch_out(an + bn + 2);
    run_toom(scratch_out.data(), a[0].data(), b[0].data(), scratch_probe);
    const auto scratch = scratch_probe.scratch_stats();

    if (square)
      std::printf("%d,%.6f,%.6f,%.6f,%.6f,%.6f,%zu,%zu,%zu\n", k,
                  ifma_full, toom, gmp_base, gmp_base / ifma_full,
                  gmp_base / toom, scratch.allocation_count,
                  scratch.allocated_bytes, scratch.peak_live_bytes);
    else
      std::printf("%d,%zu,%zu,%.6f,%.6f,%.6f,%.6f,%.6f,%zu,%zu,%zu\n", k,
                  an, bn, ifma_full, toom, gmp_base, gmp_base / ifma_full,
                  gmp_base / toom, scratch.allocation_count,
                  scratch.allocated_bytes, scratch.peak_live_bytes);
    std::fflush(stdout);
  }

  return 0;
}
