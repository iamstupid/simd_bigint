#include "../include/canon.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef CANON_COMPARE_CASES
#define CANON_COMPARE_CASES 4
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
fill_pos_extended(std::vector<U52Block> &x, std::size_t n, std::uint64_t &seed)
{
  constexpr std::uint64_t mask = (std::uint64_t{1} << 52) - 1;
  constexpr std::uint64_t base = std::uint64_t{1} << 52;
  x.assign((n + 7) / 8, {});
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t lo = splitmix64(seed) & mask;
    const std::uint64_t hi = splitmix64(seed) & 0x7ff;
    if ((i & 31) == 0)
      lo = mask;
    if ((i & 63) == 0)
      lo = 0;
    set_digit(x, i, lo + hi * base);
  }
}

void
fill_neg_extended(std::vector<U52Block> &x, std::size_t n, std::uint64_t &seed)
{
  constexpr std::uint64_t mask = (std::uint64_t{1} << 52) - 1;
  constexpr __int128 base = __int128{1} << 52;
  x.assign((n + 7) / 8, {});
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t lo = splitmix64(seed) & mask;
    std::int64_t hi = static_cast<std::int64_t>(splitmix64(seed) & 0xff) - 128;
    if ((i & 31) == 0) {
      lo = 0;
      hi = -1;
    }
    if ((i & 63) == 0) {
      lo = mask;
      hi = -2;
    }
    const __int128 value = static_cast<__int128>(hi) * base + lo;
    set_digit(x, i, static_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
  }
}

void
ref_pos(std::vector<U52Block> &r, const std::vector<U52Block> &a, std::size_t n)
{
  constexpr std::uint64_t mask = (std::uint64_t{1} << 52) - 1;
  r.assign((n + 7) / 8, {});
  unsigned __int128 carry = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const unsigned __int128 t =
      static_cast<unsigned __int128>(get_digit(a, i)) + carry;
    set_digit(r, i, static_cast<std::uint64_t>(t) & mask);
    carry = t >> 52;
  }
}

void
ref_neg(std::vector<U52Block> &r, const std::vector<U52Block> &a, std::size_t n)
{
  constexpr std::int64_t mask = (std::int64_t{1} << 52) - 1;
  r.assign((n + 7) / 8, {});
  std::int64_t carry = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const std::int64_t t = static_cast<std::int64_t>(get_digit(a, i)) + carry;
    carry = t >> 52;
    set_digit(r, i, static_cast<std::uint64_t>(t & mask));
  }
}

void
check_one(std::size_t n, std::size_t trial, std::uint64_t &seed)
{
  std::vector<U52Block> pos_in, neg_in, got, ref;
  fill_pos_extended(pos_in, n, seed);
  fill_neg_extended(neg_in, n, seed);
  got.assign((n + 7) / 8, {});

  mpn_canon_pos(reinterpret_cast<pvec>(got.data()),
                reinterpret_cast<cpvec>(pos_in.data()), n);
  ref_pos(ref, pos_in, n);
  for (std::size_t i = 0; i < n; ++i) {
    if (get_digit(got, i) != get_digit(ref, i)) {
      std::fprintf(stderr,
                   "pos mismatch n=%zu trial=%zu limb=%zu in=%016llx got=%016llx ref=%016llx\n",
                   n, trial, i,
                   static_cast<unsigned long long>(get_digit(pos_in, i)),
                   static_cast<unsigned long long>(get_digit(got, i)),
                   static_cast<unsigned long long>(get_digit(ref, i)));
      std::exit(1);
    }
  }

  std::memset(got.data(), 0xa5, got.size() * sizeof(U52Block));
  mpn_canon_neg(reinterpret_cast<pvec>(got.data()),
                reinterpret_cast<cpvec>(neg_in.data()), n);
  ref_neg(ref, neg_in, n);
  for (std::size_t i = 0; i < n; ++i) {
    if (get_digit(got, i) != get_digit(ref, i)) {
      std::fprintf(stderr,
                   "neg mismatch n=%zu trial=%zu limb=%zu got=%016llx ref=%016llx\n",
                   n, trial, i,
                   static_cast<unsigned long long>(get_digit(got, i)),
                   static_cast<unsigned long long>(get_digit(ref, i)));
      std::exit(1);
    }
  }
}

void
check_grid()
{
  std::uint64_t seed = 0x123456789abcdef0ULL;
  for (std::size_t n = 1; n <= 4096; ++n)
    for (std::size_t trial = 0; trial < 4; ++trial)
      check_one(n, trial, seed);
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
  if (guard == 0x123456789abcdef0ULL)
    std::fprintf(stderr, "%llu\n", static_cast<unsigned long long>(guard));
  return samples[samples.size() / 2];
}

std::uint64_t
guard_output(const std::vector<U52Block> &out, std::size_t n)
{
  return get_digit(out, 0) ^ get_digit(out, n / 2) ^ get_digit(out, n - 1);
}

void
run_case(std::size_t n, double seconds, int repeats)
{
  constexpr unsigned cases = CANON_COMPARE_CASES;
  static_assert((cases & (cases - 1)) == 0,
                "CANON_COMPARE_CASES must be a power of two");

  std::uint64_t seed = 0x8f4c2e5a9d3b7106ULL ^ (n << 11);
  std::vector<std::vector<U52Block>> pos_in(cases), neg_in(cases), pos_out(cases),
    neg_out(cases);
  const std::size_t blocks = (n + 7) / 8;

  for (unsigned c = 0; c < cases; ++c) {
    fill_pos_extended(pos_in[c], n, seed);
    fill_neg_extended(neg_in[c], n, seed);
    pos_out[c].assign(blocks, {});
    neg_out[c].assign(blocks, {});
  }

  unsigned ipos = 0, ineg = 0;
  const double pos_ns = bench_one([&]() -> std::uint64_t {
    const unsigned c = ipos++ & (cases - 1);
    mpn_canon_pos(reinterpret_cast<pvec>(pos_out[c].data()),
                  reinterpret_cast<cpvec>(pos_in[c].data()), n);
    return guard_output(pos_out[c], n);
  }, seconds, repeats);

  const double neg_ns = bench_one([&]() -> std::uint64_t {
    const unsigned c = ineg++ & (cases - 1);
    mpn_canon_neg(reinterpret_cast<pvec>(neg_out[c].data()),
                  reinterpret_cast<cpvec>(neg_in[c].data()), n);
    return guard_output(neg_out[c], n);
  }, seconds, repeats);

  std::printf("%zu,%.6f,%.6f,%.9f,%.9f,%.6f\n", n, pos_ns, neg_ns,
              pos_ns / static_cast<double>(n), neg_ns / static_cast<double>(n),
              neg_ns / pos_ns);
}

} // namespace

int
main(int argc, char **argv)
{
  if (argc > 1 && std::strcmp(argv[1], "--check") == 0) {
    check_grid();
    std::puts("ok canon_compare");
    return 0;
  }

  const std::size_t min_n = argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 0)) : 8;
  const std::size_t max_n = argc > 2 ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 0)) : 4096;
  const std::size_t step = argc > 3 ? static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 0)) : 8;
  const double seconds = argc > 4 ? std::atof(argv[4]) : 0.002;
  const int repeats = argc > 5 ? std::atoi(argv[5]) : 5;

  if (min_n == 0 || max_n < min_n || step == 0 || seconds <= 0.0 ||
      repeats <= 0) {
    std::fprintf(stderr,
                 "usage: %s [min_n=8] [max_n=4096] [step=8] [seconds=0.002] [repeats=5]\n"
                 "       %s --check\n",
                 argv[0], argv[0]);
    return 1;
  }

  std::puts("n,pos_simd_ns,neg_scalar_ns,pos_simd_ns_per_limb,"
            "neg_scalar_ns_per_limb,neg_over_pos");
  for (std::size_t n = min_n; n <= max_n; n += step)
    run_case(n, seconds, repeats);
  return 0;
}
