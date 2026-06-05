#include "../include/mul.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef MUL_U52_KARATSUBA_SWEEP_CASES
#define MUL_U52_KARATSUBA_SWEEP_CASES 4
#endif

namespace {

struct alignas(64) U52Block {
  std::uint64_t v[8];
};

std::uint64_t
splitmix64(std::uint64_t &x)
{
  std::uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

std::uint64_t
get_digit(const std::vector<U52Block> &x, std::size_t i)
{
  return x[i / 8].v[i & 7];
}

void
set_digit(std::vector<U52Block> &x, std::size_t i, std::uint64_t value)
{
  x[i / 8].v[i & 7] = value;
}

void
fill_u52(std::vector<U52Block> &x, std::size_t digits, std::uint64_t &seed)
{
  constexpr std::uint64_t mask = (std::uint64_t{1} << 52) - 1;
  const std::size_t blocks = (digits + 7) / 8;
  x.assign(blocks, {});
  for (std::size_t i = 0; i < digits; ++i) {
    std::uint64_t v = splitmix64(seed) & mask;
    if ((i & 17) == 0)
      v = mask;
    if ((i & 29) == 0)
      v = 0;
    set_digit(x, i, v);
  }
  set_digit(x, digits - 1, get_digit(x, digits - 1) | (std::uint64_t{1} << 51));
}

void
ref_mul(std::vector<std::uint64_t> &r, const std::vector<U52Block> &a,
        const std::vector<U52Block> &b, std::size_t n)
{
  constexpr unsigned __int128 mask =
    (static_cast<unsigned __int128>(1) << 52) - 1;
  std::vector<unsigned __int128> acc(2 * n + 1);
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j)
      acc[i + j] += static_cast<unsigned __int128>(get_digit(a, i)) *
                    static_cast<unsigned __int128>(get_digit(b, j));

  r.assign(2 * n, 0);
  unsigned __int128 carry = 0;
  for (std::size_t i = 0; i < 2 * n; ++i) {
    const unsigned __int128 t = acc[i] + carry;
    r[i] = static_cast<std::uint64_t>(t & mask);
    carry = t >> 52;
  }
  if (carry) {
    std::fprintf(stderr, "reference overflow\n");
    std::exit(1);
  }
}

void
check_grid()
{
  scratch s = scratch_create_ex(4096, 1u << 20, 1u << 20);
  std::uint64_t seed = 0x48d71f2ec0b9a536ULL;
  for (std::size_t n = 32; n <= 256; ++n) {
    for (std::size_t trial = 0; trial < 2; ++trial) {
      std::vector<U52Block> a, b, got;
      std::vector<std::uint64_t> ref;
      fill_u52(a, n, seed);
      fill_u52(b, n, seed);
      got.assign((2 * n + 7) / 8, {});
      mul_u52_dispatch(reinterpret_cast<pvec>(got.data()),
                       reinterpret_cast<cpvec>(a.data()),
                       reinterpret_cast<cpvec>(b.data()), n, n, &s);
      mpn_canon_neg(reinterpret_cast<pvec>(got.data()),
                    reinterpret_cast<cpvec>(got.data()), 2 * n);
      ref_mul(ref, a, b, n);
      for (std::size_t i = 0; i < 2 * n; ++i) {
        if (get_digit(got, i) != ref[i]) {
          std::fprintf(stderr,
                       "karatsuba mismatch n=%zu trial=%zu limb=%zu got=%016llx ref=%016llx\n",
                       n, trial, i,
                       static_cast<unsigned long long>(get_digit(got, i)),
                       static_cast<unsigned long long>(ref[i]));
          std::exit(1);
        }
      }
    }
  }
  scratch_destroy(&s);
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
      for (unsigned k = 0; k < 16; ++k)
        guard ^= fn();
      iters += 16;
      now = std::chrono::steady_clock::now();
    } while (std::chrono::duration<double>(now - begin).count() < seconds);

    const double elapsed = std::chrono::duration<double>(now - begin).count();
    samples.push_back(elapsed * 1.0e9 / static_cast<double>(iters));
  }

  std::sort(samples.begin(), samples.end());
  if (guard == 0x123456789abcdef0ULL)
    std::fprintf(stderr, "%llu\n", static_cast<unsigned long long>(guard));
  return samples[samples.size() / 2];
}

std::uint64_t
guard_output(const std::vector<U52Block> &out)
{
  return out[0].v[0] ^ out[out.size() / 2].v[3] ^ out.back().v[7];
}

void
run_case(std::size_t n, double seconds, int repeats, scratch *s)
{
  constexpr unsigned cases = MUL_U52_KARATSUBA_SWEEP_CASES;
  static_assert((cases & (cases - 1)) == 0,
                "MUL_U52_KARATSUBA_SWEEP_CASES must be a power of two");

  std::uint64_t seed = 0x8f4c2e5a9d3b7106ULL ^ (n << 17);
  std::vector<std::vector<U52Block>> a(cases), b(cases), base_out(cases),
    kara_out(cases);
  const std::size_t out_blocks = (2 * n + 7) / 8 + 1;

  for (unsigned c = 0; c < cases; ++c) {
    fill_u52(a[c], n, seed);
    fill_u52(b[c], n, seed);
    base_out[c].assign(out_blocks, {});
    kara_out[c].assign(out_blocks, {});
  }

  unsigned ibase = 0, ikara = 0;
  const double base_ns = bench_one([&]() -> std::uint64_t {
    const unsigned c = ibase++ & (cases - 1);
    mul_u52_basecase(reinterpret_cast<pvec>(base_out[c].data()),
                     reinterpret_cast<cpvec>(a[c].data()),
                     reinterpret_cast<cpvec>(b[(5 * c) & (cases - 1)].data()),
                     n, n);
    return guard_output(base_out[c]);
  }, seconds, repeats);

  const double kara_ns = bench_one([&]() -> std::uint64_t {
    const unsigned c = ikara++ & (cases - 1);
    mul_u52_dispatch(reinterpret_cast<pvec>(kara_out[c].data()),
                     reinterpret_cast<cpvec>(a[c].data()),
                     reinterpret_cast<cpvec>(b[(5 * c) & (cases - 1)].data()),
                     n, n, s);
    return guard_output(kara_out[c]);
  }, seconds, repeats);

  const double limbs = static_cast<double>(n);
  std::printf("%zu,%.6f,%.6f,%.9f,%.9f,%.6f\n", n, base_ns, kara_ns,
              base_ns / limbs, kara_ns / limbs, base_ns / kara_ns);
}

} // namespace

int
main(int argc, char **argv)
{
  if (argc > 1 && std::strcmp(argv[1], "--check") == 0) {
    check_grid();
    std::puts("ok mul_u52_karatsuba_sweep");
    return 0;
  }

  const std::size_t min_n = argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 0)) : 32;
  const std::size_t max_n = argc > 2 ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 0)) : 256;
  const std::size_t step = argc > 3 ? static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 0)) : 1;
  const double seconds = argc > 4 ? std::atof(argv[4]) : 0.0015;
  const int repeats = argc > 5 ? std::atoi(argv[5]) : 5;

  if (min_n == 0 || max_n < min_n || step == 0 || seconds <= 0.0 ||
      repeats <= 0) {
    std::fprintf(stderr,
                 "usage: %s [min_n=32] [max_n=256] [step=1] [seconds=0.0015] [repeats=5]\n"
                 "       %s --check\n",
                 argv[0], argv[0]);
    return 1;
  }

  scratch s = scratch_create_ex(4096, 1u << 20, 1u << 20);
  std::puts("n,basecase_ns,karatsuba_ns,basecase_ns_per_limb,"
            "karatsuba_ns_per_limb,speedup_base_over_karatsuba");
  for (std::size_t n = min_n; n <= max_n; n += step)
    run_case(n, seconds, repeats, &s);
  scratch_destroy(&s);
  return 0;
}
