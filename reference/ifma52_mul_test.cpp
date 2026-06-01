#include "ifma52_mul.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

using simd_bigint::reference::mul_auto_u64;
using simd_bigint::reference::mul_basecase_u64;
using simd_bigint::reference::mul_toom22_raw_u64;
using simd_bigint::reference::mul_toom22_u64;
using simd_bigint::reference::mul_toom32_raw_u64;
using simd_bigint::reference::mul_toom32_u64;
using simd_bigint::reference::mul_toom33_raw_u64;
using simd_bigint::reference::mul_toom33_u64;
using simd_bigint::reference::ToomWorkspace;

namespace {

using MulFn = void (*)(std::uint64_t *, const std::uint64_t *, std::size_t,
                       const std::uint64_t *, std::size_t);
using WorkspaceMulFn = void (*)(std::uint64_t *, const std::uint64_t *,
                                std::size_t, const std::uint64_t *,
                                std::size_t, ToomWorkspace &);

std::uint64_t
splitmix64(std::uint64_t &x)
{
  std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

void
ref_mul(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
        const std::uint64_t *bp, std::size_t bn)
{
  if (an + bn != 0)
    std::memset(rp, 0, (an + bn) * sizeof(rp[0]));

  for (std::size_t j = 0; j < bn; ++j) {
    unsigned __int128 carry = 0;
    for (std::size_t i = 0; i < an; ++i) {
      const unsigned __int128 t =
        static_cast<unsigned __int128>(ap[i]) * bp[j] + rp[i + j] + carry;
      rp[i + j] = static_cast<std::uint64_t>(t);
      carry = t >> 64;
    }
    rp[j + an] = static_cast<std::uint64_t>(carry);
  }
}

void
fill_case(std::vector<std::uint64_t> &a, std::vector<std::uint64_t> &b,
          std::uint64_t &seed, std::size_t trial)
{
  for (std::uint64_t &x : a)
    x = splitmix64(seed);
  for (std::uint64_t &x : b)
    x = splitmix64(seed);

  if (!a.empty())
    a.back() |= std::uint64_t{1} << 63;
  if (!b.empty())
    b.back() |= std::uint64_t{1} << 63;

  if ((trial & 7) == 0) {
    for (std::uint64_t &x : a)
      x = ~std::uint64_t{0};
  }
  if ((trial & 15) == 0) {
    for (std::uint64_t &x : b)
      x = ~std::uint64_t{0};
  }
  if ((trial & 31) == 0 && !a.empty())
    a.back() = 0;
  if ((trial & 63) == 0 && !b.empty())
    b.back() = 0;
}

void
check_one(std::string_view name, MulFn fn, std::size_t an, std::size_t bn,
          std::size_t trials, std::uint64_t seed0)
{
  std::uint64_t seed = seed0 + an * 0x101 + bn * 0x10001;
  std::vector<std::uint64_t> a(an), b(bn), got(an + bn), ref(an + bn);

  for (std::size_t trial = 0; trial < trials; ++trial) {
    fill_case(a, b, seed, trial);
    std::memset(got.data(), 0xa5, got.size() * sizeof(got[0]));
    std::memset(ref.data(), 0x5a, ref.size() * sizeof(ref[0]));

    fn(got.data(), a.data(), an, b.data(), bn);
    ref_mul(ref.data(), a.data(), an, b.data(), bn);

    if (got != ref) {
      std::fprintf(stderr, "%.*s mismatch an=%zu bn=%zu trial=%zu\n",
                   static_cast<int>(name.size()), name.data(), an, bn, trial);
      for (std::size_t i = 0; i < got.size(); ++i)
        std::fprintf(stderr, "  [%zu] got=%016llx ref=%016llx\n", i,
                     static_cast<unsigned long long>(got[i]),
                     static_cast<unsigned long long>(ref[i]));
      std::exit(1);
    }
  }
}

void
check_grid(std::string_view name, MulFn fn, std::size_t max_an,
           std::size_t max_bn, std::size_t trials, std::uint64_t seed)
{
  for (std::size_t an = 1; an <= max_an; ++an)
    for (std::size_t bn = 1; bn <= max_bn; ++bn)
      check_one(name, fn, an, bn, trials, seed);
}

void
check_workspace_reuse()
{
  ToomWorkspace w22;
  ToomWorkspace w32;
  ToomWorkspace w33;
  std::uint64_t seed = 0xbb67ae8584caa73bULL;

  const std::pair<std::size_t, std::size_t> cases[] = {
    {37, 29}, {64, 64}, {97, 61}, {119, 120}, {160, 111}, {160, 160}
  };

  for (auto [an, bn] : cases) {
    std::vector<std::uint64_t> a(an), b(bn), got(an + bn), ref(an + bn);
    for (std::size_t trial = 0; trial < 12; ++trial) {
      fill_case(a, b, seed, trial);
      ref_mul(ref.data(), a.data(), an, b.data(), bn);

      mul_toom22_u64(got.data(), a.data(), an, b.data(), bn, w22);
      if (got != ref)
        std::exit(std::fprintf(stderr, "toom22 workspace mismatch\n"));

      mul_toom32_u64(got.data(), a.data(), an, b.data(), bn, w32);
      if (got != ref)
        std::exit(std::fprintf(stderr, "toom32 workspace mismatch\n"));

      mul_toom33_u64(got.data(), a.data(), an, b.data(), bn, w33);
      if (got != ref)
        std::exit(std::fprintf(stderr, "toom33 workspace mismatch\n"));
    }
  }
}

void
check_threshold_boundaries()
{
  ToomWorkspace w22;
  ToomWorkspace w32;
  ToomWorkspace w33;
  std::uint64_t seed = 0x3c6ef372fe94f82bULL;

  struct Case {
    std::string_view name;
    std::size_t an;
    std::size_t bn;
  };

  const Case cases[] = {
    {"toom22 below cutoff", 71, 71},
    {"toom22 at cutoff", 72, 72},
    {"toom32 below chunk cutoff", 141, 94},
    {"toom32 at chunk cutoff", 144, 96},
    {"toom32 swapped at chunk cutoff", 96, 144},
    {"toom33 below cutoff", 143, 143},
    {"toom33 at cutoff", 144, 144},
    {"toom33 above cutoff", 180, 180},
  };

  for (const Case &tc : cases) {
    std::vector<std::uint64_t> a(tc.an), b(tc.bn), got(tc.an + tc.bn),
      ref(tc.an + tc.bn);
    for (std::size_t trial = 0; trial < 10; ++trial) {
      fill_case(a, b, seed, trial);
      std::memset(got.data(), 0xa5, got.size() * sizeof(got[0]));
      ref_mul(ref.data(), a.data(), tc.an, b.data(), tc.bn);

      if (tc.name.starts_with("toom22")) {
        mul_toom22_u64(got.data(), a.data(), tc.an, b.data(), tc.bn, w22);
      } else if (tc.name.starts_with("toom32")) {
        mul_toom32_u64(got.data(), a.data(), tc.an, b.data(), tc.bn, w32);
      } else {
        mul_toom33_u64(got.data(), a.data(), tc.an, b.data(), tc.bn, w33);
      }

      if (got != ref) {
        std::fprintf(stderr, "%.*s threshold mismatch trial=%zu\n",
                     static_cast<int>(tc.name.size()), tc.name.data(), trial);
        std::exit(1);
      }
    }
  }
}

void
check_raw_kernels()
{
  ToomWorkspace w22;
  ToomWorkspace w32;
  ToomWorkspace w33;
  std::uint64_t seed = 0x510e527fade682d1ULL;

  struct Case {
    std::string_view name;
    std::size_t an;
    std::size_t bn;
  };

  const Case cases[] = {
    {"raw22 small", 37, 29},
    {"raw22 cutoff", 72, 72},
    {"raw32 below cutoff", 120, 80},
    {"raw32 cutoff", 144, 96},
    {"raw32 swapped", 96, 144},
    {"raw33 below cutoff", 120, 120},
    {"raw33 cutoff", 144, 144},
    {"raw33 medium", 180, 180},
  };

  for (const Case &tc : cases) {
    std::vector<std::uint64_t> a(tc.an), b(tc.bn), got(tc.an + tc.bn),
      ref(tc.an + tc.bn);
    for (std::size_t trial = 0; trial < 8; ++trial) {
      fill_case(a, b, seed, trial);
      std::memset(got.data(), 0xa5, got.size() * sizeof(got[0]));
      ref_mul(ref.data(), a.data(), tc.an, b.data(), tc.bn);

      if (tc.name.starts_with("raw22")) {
        mul_toom22_raw_u64(got.data(), a.data(), tc.an, b.data(), tc.bn, w22);
      } else if (tc.name.starts_with("raw32")) {
        mul_toom32_raw_u64(got.data(), a.data(), tc.an, b.data(), tc.bn, w32);
      } else {
        mul_toom33_raw_u64(got.data(), a.data(), tc.an, b.data(), tc.bn, w33);
      }

      if (got != ref) {
        std::fprintf(stderr, "%.*s raw mismatch trial=%zu\n",
                     static_cast<int>(tc.name.size()), tc.name.data(), trial);
        std::exit(1);
      }
    }
  }
}

void
check_padded_operands()
{
  struct Case {
    std::string_view name;
    std::size_t an;
    std::size_t bn;
    std::size_t used_an;
    std::size_t used_bn;
  };
  struct Algo {
    std::string_view name;
    MulFn fn;
  };

  const Case cases[] = {
    {"zero padded lhs", 96, 88, 0, 57},
    {"zero padded rhs", 88, 96, 57, 0},
    {"toom22 padded cutoff", 160, 155, 72, 72},
    {"toom32 padded cutoff", 210, 190, 144, 96},
    {"toom32 padded swapped", 190, 210, 96, 144},
    {"toom33 padded cutoff", 220, 210, 144, 144},
  };
  const Algo algos[] = {
    {"basecase", mul_basecase_u64},
    {"auto", mul_auto_u64},
    {"toom22", mul_toom22_u64},
    {"toom22 raw", mul_toom22_raw_u64},
    {"toom32", mul_toom32_u64},
    {"toom32 raw", mul_toom32_raw_u64},
    {"toom33", mul_toom33_u64},
    {"toom33 raw", mul_toom33_raw_u64},
  };

  std::uint64_t seed = 0x1f83d9abfb41bd6bULL;

  for (const Case &tc : cases) {
    std::vector<std::uint64_t> a(tc.an), b(tc.bn), got(tc.an + tc.bn),
      ref(tc.an + tc.bn);

    for (std::size_t trial = 0; trial < 8; ++trial) {
      std::fill(a.begin(), a.end(), 0);
      std::fill(b.begin(), b.end(), 0);
      for (std::size_t i = 0; i < tc.used_an; ++i)
        a[i] = splitmix64(seed);
      for (std::size_t i = 0; i < tc.used_bn; ++i)
        b[i] = splitmix64(seed);
      if (tc.used_an != 0)
        a[tc.used_an - 1] |= std::uint64_t{1} << 63;
      if (tc.used_bn != 0)
        b[tc.used_bn - 1] |= std::uint64_t{1} << 63;
      if ((trial & 3) == 0 && tc.used_an != 0)
        a[tc.used_an - 1] = 0;
      if ((trial & 5) == 0 && tc.used_bn != 0)
        b[tc.used_bn - 1] = 0;

      ref_mul(ref.data(), a.data(), tc.an, b.data(), tc.bn);

      for (const Algo &algo : algos) {
        std::memset(got.data(), 0xa5, got.size() * sizeof(got[0]));
        algo.fn(got.data(), a.data(), tc.an, b.data(), tc.bn);
        if (got != ref) {
          std::fprintf(stderr,
                       "%.*s %.*s padded mismatch an=%zu bn=%zu used=%zu/%zu "
                       "trial=%zu\n",
                       static_cast<int>(algo.name.size()), algo.name.data(),
                       static_cast<int>(tc.name.size()), tc.name.data(), tc.an,
                       tc.bn, tc.used_an, tc.used_bn, trial);
          std::exit(1);
        }
      }
    }
  }
}

void
check_scratch_stats()
{
  ToomWorkspace workspace;
  std::uint64_t seed = 0x5be0cd19137e2179ULL;
  std::vector<std::uint64_t> a(144), b(144), got(288);
  fill_case(a, b, seed, 1);

  mul_toom33_raw_u64(got.data(), a.data(), a.size(), b.data(), b.size(),
                     workspace);
  const auto used = workspace.scratch_stats();
  if (used.allocation_count == 0 || used.allocated_bytes == 0 ||
      used.peak_live_bytes == 0 ||
      used.current_live_bytes > used.allocated_bytes ||
      used.peak_live_bytes > used.allocated_bytes) {
    std::fprintf(stderr, "scratch stats did not record raw Toom allocation\n");
    std::exit(1);
  }

  std::vector<std::uint64_t> small_a(16), small_b(16), small_got(32);
  fill_case(small_a, small_b, seed, 2);
  mul_toom33_u64(small_got.data(), small_a.data(), small_a.size(),
                 small_b.data(), small_b.size(), workspace);
  const auto fallback = workspace.scratch_stats();
  if (fallback.allocation_count != 0 || fallback.allocated_bytes != 0 ||
      fallback.current_live_bytes != 0 || fallback.peak_live_bytes != 0) {
    std::fprintf(stderr, "scratch stats changed on Toom fallback\n");
    std::exit(1);
  }

  workspace.reset();
  const auto reset = workspace.scratch_stats();
  if (reset.allocation_count != 0 || reset.allocated_bytes != 0 ||
      reset.current_live_bytes != 0 || reset.peak_live_bytes != 0) {
    std::fprintf(stderr, "scratch stats reset mismatch\n");
    std::exit(1);
  }
}

void
check_scratch_footprints()
{
  struct Case {
    std::string_view name;
    WorkspaceMulFn fn;
    std::size_t an;
    std::size_t bn;
    std::size_t max_allocs;
    std::size_t max_bytes;
  };

  const Case cases[] = {
    {"raw22 footprint", mul_toom22_raw_u64, 96, 96, 2, 1176},
    {"raw32 footprint", mul_toom32_raw_u64, 144, 96, 3, 1992},
    {"raw33 footprint", mul_toom33_raw_u64, 144, 144, 4, 2792},
  };

  std::uint64_t seed = 0xcbbb9d5dc1059ed8ULL;
  for (const Case &tc : cases) {
    std::vector<std::uint64_t> a(tc.an), b(tc.bn), got(tc.an + tc.bn),
      ref(tc.an + tc.bn);
    fill_case(a, b, seed, 1);
    ref_mul(ref.data(), a.data(), tc.an, b.data(), tc.bn);

    ToomWorkspace workspace;
    tc.fn(got.data(), a.data(), tc.an, b.data(), tc.bn, workspace);
    if (got != ref) {
      std::fprintf(stderr, "%.*s correctness mismatch\n",
                   static_cast<int>(tc.name.size()), tc.name.data());
      std::exit(1);
    }

    const auto stats = workspace.scratch_stats();
    if (stats.allocation_count > tc.max_allocs ||
        stats.allocated_bytes > tc.max_bytes ||
        stats.current_live_bytes > stats.allocated_bytes ||
        stats.peak_live_bytes > stats.allocated_bytes) {
      std::fprintf(stderr,
                   "%.*s scratch regression allocs=%zu/%zu bytes=%zu/%zu "
                   "live=%zu peak=%zu\n",
                   static_cast<int>(tc.name.size()), tc.name.data(),
                   stats.allocation_count, tc.max_allocs, stats.allocated_bytes,
                   tc.max_bytes, stats.current_live_bytes, stats.peak_live_bytes);
      std::exit(1);
    }
  }
}

void
check_auto_dispatch()
{
  ToomWorkspace workspace;
  std::uint64_t seed = 0x629a292a367cd507ULL;

  struct Case {
    std::string_view name;
    std::size_t an;
    std::size_t bn;
    std::size_t max_allocs;
  };

  const Case cases[] = {
    {"auto basecase", 32, 31, 0},
    {"auto toom22", 72, 72, 2},
    {"auto toom32", 144, 96, 3},
    {"auto toom32 swapped", 96, 144, 3},
    {"auto toom33", 144, 144, 4},
    {"auto rectangular", 200, 180, 3},
    {"auto larger square", 180, 180, 4},
  };

  for (const Case &tc : cases) {
    std::vector<std::uint64_t> a(tc.an), b(tc.bn), got(tc.an + tc.bn),
      ref(tc.an + tc.bn);
    for (std::size_t trial = 0; trial < 8; ++trial) {
      fill_case(a, b, seed, trial);
      std::memset(got.data(), 0xa5, got.size() * sizeof(got[0]));
      ref_mul(ref.data(), a.data(), tc.an, b.data(), tc.bn);

      mul_auto_u64(got.data(), a.data(), tc.an, b.data(), tc.bn, workspace);
      if (got != ref) {
        std::fprintf(stderr, "%.*s auto mismatch trial=%zu\n",
                     static_cast<int>(tc.name.size()), tc.name.data(), trial);
        std::exit(1);
      }

      const auto stats = workspace.scratch_stats();
      if (stats.allocation_count > tc.max_allocs) {
        std::fprintf(stderr, "%.*s auto scratch allocs=%zu/%zu\n",
                     static_cast<int>(tc.name.size()), tc.name.data(),
                     stats.allocation_count, tc.max_allocs);
        std::exit(1);
      }
    }
  }
}

} // namespace

int
main()
{
  check_grid("basecase", mul_basecase_u64, 96, 96, 16, 0x243f6a8885a308d3ULL);
  check_grid("toom22", mul_toom22_u64, 96, 96, 8, 0x13198a2e03707344ULL);
  check_grid("toom32", mul_toom32_u64, 120, 96, 8, 0xa4093822299f31d0ULL);
  check_grid("toom33", mul_toom33_u64, 120, 120, 8, 0x082efa98ec4e6c89ULL);
  check_workspace_reuse();
  check_threshold_boundaries();
  check_raw_kernels();
  check_padded_operands();
  check_scratch_stats();
  check_scratch_footprints();
  check_auto_dispatch();

  std::puts("ok ifma52 basecase/toom reference");
  return 0;
}
