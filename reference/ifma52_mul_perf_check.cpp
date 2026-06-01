#include "ifma52_mul.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <gmp.h>
#include <iterator>
#include <string_view>
#include <vector>

extern "C" void __gmpn_mul_basecase(mp_limb_t *rp, const mp_limb_t *up,
                                    mp_size_t un, const mp_limb_t *vp,
                                    mp_size_t vn);

using simd_bigint::reference::Ifma52Workspace;
using simd_bigint::reference::ToomWorkspace;
using simd_bigint::reference::mul_auto_u64;
using simd_bigint::reference::mul_basecase_u64;
using simd_bigint::reference::mul_toom22_u64;
using simd_bigint::reference::mul_toom32_u64;
using simd_bigint::reference::mul_toom33_u64;

namespace {

struct PerfCase {
  std::string_view name;
  std::size_t an;
  std::size_t bn;
  void (*fn)(std::uint64_t *, const std::uint64_t *, std::size_t,
             const std::uint64_t *, std::size_t, void *);
};

struct PerfResult {
  double algo_ns = 0;
  double gmp_ns = 0;
  double speedup = 0;
  std::size_t scratch_allocs = 0;
  std::size_t scratch_bytes = 0;
};

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

void
run_basecase(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
             const std::uint64_t *bp, std::size_t bn, void *ctx)
{
  auto &ws = *static_cast<Ifma52Workspace *>(ctx);
  mul_basecase_u64(rp, ap, an, bp, bn, ws);
}

void
run_toom22(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
           const std::uint64_t *bp, std::size_t bn, void *ctx)
{
  auto &ws = *static_cast<ToomWorkspace *>(ctx);
  mul_toom22_u64(rp, ap, an, bp, bn, ws);
}

void
run_toom32(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
           const std::uint64_t *bp, std::size_t bn, void *ctx)
{
  auto &ws = *static_cast<ToomWorkspace *>(ctx);
  mul_toom32_u64(rp, ap, an, bp, bn, ws);
}

void
run_toom33(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
           const std::uint64_t *bp, std::size_t bn, void *ctx)
{
  auto &ws = *static_cast<ToomWorkspace *>(ctx);
  mul_toom33_u64(rp, ap, an, bp, bn, ws);
}

void
run_auto(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
         const std::uint64_t *bp, std::size_t bn, void *ctx)
{
  auto &ws = *static_cast<ToomWorkspace *>(ctx);
  mul_auto_u64(rp, ap, an, bp, bn, ws);
}

PerfResult
run_case(const PerfCase &tc, double seconds, int repeats, unsigned cases,
         std::uint64_t &seed)
{
  std::vector<std::vector<std::uint64_t>> a(cases), b(cases), out(cases);
  for (unsigned c = 0; c < cases; ++c) {
    a[c].resize(tc.an);
    b[c].resize(tc.bn);
    out[c].resize(tc.an + tc.bn + 2);
    for (std::uint64_t &x : a[c])
      x = splitmix64(seed);
    for (std::uint64_t &x : b[c])
      x = splitmix64(seed);
    a[c].back() |= std::uint64_t{1} << 63;
    b[c].back() |= std::uint64_t{1} << 63;
  }

  Ifma52Workspace base_ws;
  ToomWorkspace toom_ws;
  void *ctx = tc.fn == run_basecase ? static_cast<void *>(&base_ws)
                                    : static_cast<void *>(&toom_ws);
  unsigned idx_algo = 0, idx_gmp = 0;

  const double algo_ns = bench_one([&]() -> std::uint64_t {
    const unsigned c = idx_algo++ & (cases - 1);
    tc.fn(out[c].data(), a[c].data(), tc.an,
          b[(5 * c) & (cases - 1)].data(), tc.bn, ctx);
    return out[c][0] ^ out[c][tc.bn] ^ out[c][tc.an + tc.bn - 1];
  }, seconds, repeats);

  const double gmp_ns = bench_one([&]() -> std::uint64_t {
    const unsigned c = idx_gmp++ & (cases - 1);
    gmp_mul_basecase_norm(out[c].data(), a[c].data(), tc.an,
                          b[(5 * c) & (cases - 1)].data(), tc.bn);
    return out[c][0] ^ out[c][tc.bn] ^ out[c][tc.an + tc.bn - 1];
  }, seconds, repeats);

  ToomWorkspace scratch_probe;
  if (tc.fn != run_basecase)
    tc.fn(out[0].data(), a[0].data(), tc.an, b[0].data(), tc.bn,
          &scratch_probe);
  const auto scratch = scratch_probe.scratch_stats();

  return {algo_ns, gmp_ns, gmp_ns / algo_ns, scratch.allocation_count,
          scratch.allocated_bytes};
}

} // namespace

int
main(int argc, char **argv)
{
  bool sweep = false;
  int arg = 1;
  if (argc > 1 && std::strcmp(argv[1], "sweep") == 0) {
    sweep = true;
    arg = 2;
  }

  const double seconds = argc > arg ? std::atof(argv[arg]) : 0.006;
  const int repeats = argc > arg + 1 ? std::atoi(argv[arg + 1]) : 5;
  const double min_speedup = argc > arg + 2 ? std::atof(argv[arg + 2]) : 2.0;
  constexpr unsigned cases = 16;

  const PerfCase standard_checks[] = {
    {"ifma_basecase_64x64", 64, 64, run_basecase},
    {"toom22_public_72x72", 72, 72, run_toom22},
    {"toom32_public_144x96", 144, 96, run_toom32},
    {"toom33_public_144x144", 144, 144, run_toom33},
    {"auto_64x64", 64, 64, run_auto},
    {"auto_72x72", 72, 72, run_auto},
    {"auto_144x96", 144, 96, run_auto},
    {"auto_144x144", 144, 144, run_auto},
  };
  const PerfCase sweep_checks[] = {
    {"ifma_basecase_64x64", 64, 64, run_basecase},
    {"ifma_basecase_96x96", 96, 96, run_basecase},
    {"ifma_basecase_128x128", 128, 128, run_basecase},
    {"toom22_public_72x72", 72, 72, run_toom22},
    {"toom22_public_80x80", 80, 80, run_toom22},
    {"toom22_public_96x96", 96, 96, run_toom22},
    {"toom32_public_144x96", 144, 96, run_toom32},
    {"toom32_public_168x112", 168, 112, run_toom32},
    {"toom32_public_192x128", 192, 128, run_toom32},
    {"toom33_public_144x144", 144, 144, run_toom33},
    {"toom33_public_152x152", 152, 152, run_toom33},
    {"toom33_public_160x160", 160, 160, run_toom33},
    {"auto_64x64", 64, 64, run_auto},
    {"auto_72x72", 72, 72, run_auto},
    {"auto_96x96", 96, 96, run_auto},
    {"auto_144x96", 144, 96, run_auto},
    {"auto_168x112", 168, 112, run_auto},
    {"auto_144x144", 144, 144, run_auto},
    {"auto_160x160", 160, 160, run_auto},
  };

  std::uint64_t seed = 0x6a09e667f3bcc909ULL;
  bool ok = true;

  std::puts("case,an,bn,ifma_or_toom_ns,gmp_basecase_ns,speedup,min_speedup,"
            "scratch_allocs,scratch_bytes");

  const PerfCase *checks = sweep ? sweep_checks : standard_checks;
  const std::size_t check_count =
    sweep ? std::size(sweep_checks) : std::size(standard_checks);
  for (std::size_t i = 0; i < check_count; ++i) {
    const PerfCase &tc = checks[i];
    const PerfResult result = run_case(tc, seconds, repeats, cases, seed);
    std::printf("%.*s,%zu,%zu,%.6f,%.6f,%.6f,%.2f,%zu,%zu\n",
                static_cast<int>(tc.name.size()), tc.name.data(), tc.an,
                tc.bn, result.algo_ns, result.gmp_ns, result.speedup,
                min_speedup, result.scratch_allocs, result.scratch_bytes);
    if (result.speedup < min_speedup)
      ok = false;
  }

  if (!ok) {
    std::fprintf(stderr, "IFMA52 performance check failed threshold %.2fx\n",
                 min_speedup);
    return 1;
  }

  return 0;
}
