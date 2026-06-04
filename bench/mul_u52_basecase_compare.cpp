#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" void bench_mul_u52_basecase_old(unsigned long long *rp,
                                           const unsigned long long *ap,
                                           const unsigned long long *bp,
                                           unsigned long long an,
                                           unsigned long long bn);

extern "C" void bench_mul_u52_basecase_new(unsigned long long *rp,
                                           const unsigned long long *ap,
                                           const unsigned long long *bp,
                                           unsigned long long an,
                                           unsigned long long bn);

#ifndef MUL_U52_BASECASE_COMPARE_CASES
#define MUL_U52_BASECASE_COMPARE_CASES 4
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

void
fill_u52(std::vector<U52Block> &x, std::size_t digits, std::uint64_t &seed)
{
  constexpr std::uint64_t mask = (std::uint64_t{1} << 52) - 1;
  const std::size_t blocks = (digits + 7) / 8;
  x.assign(blocks, {});
  for (std::size_t i = 0; i < digits; ++i)
    x[i / 8].v[i & 7] = splitmix64(seed) & mask;
  x[(digits - 1) / 8].v[(digits - 1) & 7] |= std::uint64_t{1} << 51;
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
guard_u52(const std::vector<U52Block> &out)
{
  return out[0].v[0] ^ out[out.size() / 2].v[3] ^ out.back().v[7];
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
ref_mul_u52(std::vector<U52Block> &r, const std::vector<U52Block> &a,
            const std::vector<U52Block> &b, std::size_t an, std::size_t bn)
{
  constexpr unsigned shift = 52;
  constexpr std::uint64_t mask = (std::uint64_t{1} << shift) - 1;

  for (U52Block &blk : r)
    for (std::uint64_t &x : blk.v)
      x = 0;

  for (std::size_t i = 0; i < an; ++i) {
    for (std::size_t j = 0; j < bn; ++j) {
      const unsigned __int128 p =
        static_cast<unsigned __int128>(get_digit(a, i)) * get_digit(b, j);
      set_digit(r, i + j,
                get_digit(r, i + j) + static_cast<std::uint64_t>(p & mask));
      set_digit(r, i + j + 1,
                get_digit(r, i + j + 1) + static_cast<std::uint64_t>(p >> shift));
    }
  }
}

std::size_t
rounded_div(std::size_t n, std::size_t d)
{
  return (n + d / 2) / d;
}

bool
check_case(std::size_t an52, std::size_t bn52, std::size_t trial,
           std::uint64_t &seed, bool *old_ok)
{
  std::vector<U52Block> a, b;
  fill_u52(a, an52, seed);
  fill_u52(b, bn52, seed);

  const std::size_t out_blocks = (an52 + 7) / 8 + (bn52 + 7) / 8 + 2;
  std::vector<U52Block> old_out(out_blocks), new_out(out_blocks), ref_out(out_blocks);

  bench_mul_u52_basecase_old(reinterpret_cast<unsigned long long *>(old_out.data()),
                             reinterpret_cast<const unsigned long long *>(a.data()),
                             reinterpret_cast<const unsigned long long *>(b.data()),
                             static_cast<unsigned long long>(an52),
                             static_cast<unsigned long long>(bn52));
  bench_mul_u52_basecase_new(reinterpret_cast<unsigned long long *>(new_out.data()),
                             reinterpret_cast<const unsigned long long *>(a.data()),
                             reinterpret_cast<const unsigned long long *>(b.data()),
                             static_cast<unsigned long long>(an52),
                             static_cast<unsigned long long>(bn52));

  ref_mul_u52(ref_out, a, b, an52, bn52);

  *old_ok = std::memcmp(old_out.data(), ref_out.data(),
                        out_blocks * sizeof(U52Block)) == 0;
  if (std::memcmp(new_out.data(), ref_out.data(),
                  out_blocks * sizeof(U52Block)) == 0)
    return true;

  for (std::size_t i = 0; i < out_blocks * 8; ++i) {
    const std::uint64_t nv = get_digit(new_out, i);
    const std::uint64_t rv = get_digit(ref_out, i);
    if (nv != rv)
      std::fprintf(stderr, "  [%zu] new=%016llx ref=%016llx\n", i,
                   static_cast<unsigned long long>(nv),
                   static_cast<unsigned long long>(rv));
  }
  return false;
}

void
check_grid()
{
  std::uint64_t seed = 0x123456789abcdef0ULL;
  std::size_t old_bad = 0;
  for (std::size_t an52 = 1; an52 <= 256; ++an52) {
    for (std::size_t bn52 = 1; bn52 <= an52 && bn52 <= 64; ++bn52) {
      for (std::size_t trial = 0; trial < 4; ++trial) {
        bool old_ok = false;
        if (!check_case(an52, bn52, trial, seed, &old_ok)) {
          std::fprintf(stderr, "new/ref mismatch an=%zu bn=%zu trial=%zu\n",
                       an52, bn52, trial);
          std::exit(1);
        }
        if (!old_ok)
          ++old_bad;
      }
    }
  }
  std::fprintf(stderr, "old/ref mismatches: %zu\n", old_bad);
}

void
run_case(const char *shape, std::size_t an52, std::size_t bn52,
         double seconds, int repeats)
{
  constexpr unsigned cases = MUL_U52_BASECASE_COMPARE_CASES;
  static_assert((cases & (cases - 1)) == 0,
                "MUL_U52_BASECASE_COMPARE_CASES must be a power of two");

  std::uint64_t seed = 0x8f4c2e5a9d3b7106ULL ^ (an52 << 17) ^ (bn52 << 3);
  std::vector<std::vector<U52Block>> au52(cases), bu52(cases), old_out(cases),
    new_out(cases);

  const std::size_t out_blocks = (an52 + 7) / 8 + (bn52 + 7) / 8 + 2;
  for (unsigned c = 0; c < cases; ++c) {
    fill_u52(au52[c], an52, seed);
    fill_u52(bu52[c], bn52, seed);
    old_out[c].assign(out_blocks, {});
    new_out[c].assign(out_blocks, {});
  }

  unsigned iold = 0, inew = 0;
  const double old_ns = bench_one([&]() -> std::uint64_t {
    const unsigned c = iold++ & (cases - 1);
    bench_mul_u52_basecase_old(reinterpret_cast<unsigned long long *>(old_out[c].data()),
                               reinterpret_cast<const unsigned long long *>(au52[c].data()),
                               reinterpret_cast<const unsigned long long *>(
                                 bu52[(5 * c) & (cases - 1)].data()),
                               static_cast<unsigned long long>(an52),
                               static_cast<unsigned long long>(bn52));
    return guard_u52(old_out[c]);
  }, seconds, repeats);

  const double new_ns = bench_one([&]() -> std::uint64_t {
    const unsigned c = inew++ & (cases - 1);
    bench_mul_u52_basecase_new(reinterpret_cast<unsigned long long *>(new_out[c].data()),
                               reinterpret_cast<const unsigned long long *>(au52[c].data()),
                               reinterpret_cast<const unsigned long long *>(
                                 bu52[(5 * c) & (cases - 1)].data()),
                               static_cast<unsigned long long>(an52),
                               static_cast<unsigned long long>(bn52));
    return guard_u52(new_out[c]);
  }, seconds, repeats);

  const double work = static_cast<double>(an52) * static_cast<double>(bn52);
  std::printf("%s,%zu,%zu,%.6f,%.6f,%.9f,%.9f,%.6f\n", shape, an52, bn52,
              old_ns, new_ns, old_ns / work, new_ns / work, old_ns / new_ns);
}

void
run_sweep(std::size_t min_an, std::size_t max_an, std::size_t step,
          double seconds, int repeats)
{
  std::puts("shape,an52,bn52,old_ns,new_ns,old_ns_per_digit_product,"
            "new_ns_per_digit_product,speedup_old_over_new");

  for (std::size_t ratio : {std::size_t{1}, std::size_t{2}, std::size_t{4},
                            std::size_t{8}}) {
    const std::string shape = "an=" + std::to_string(ratio) + "bn";
    for (std::size_t an52 = min_an; an52 <= max_an; an52 += step) {
      const std::size_t bn52 = std::max<std::size_t>(1, rounded_div(an52, ratio));
      run_case(shape.c_str(), an52, bn52, seconds, repeats);
    }
  }

  for (std::size_t fixed_bn : {std::size_t{4}, std::size_t{6},
                               std::size_t{8}, std::size_t{10},
                               std::size_t{12}}) {
    const std::string shape = "bn=" + std::to_string(fixed_bn);
    for (std::size_t an52 = std::max(min_an, fixed_bn); an52 <= max_an;
         an52 += step)
      run_case(shape.c_str(), an52, fixed_bn, seconds, repeats);
  }
}

} // namespace

int
main(int argc, char **argv)
{
  if (argc > 1 && std::strcmp(argv[1], "--check") == 0) {
    check_grid();
    std::puts("ok old/new mul_u52_basecase");
    return 0;
  }

  const std::size_t min_an = argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 0)) : 8;
  const std::size_t max_an = argc > 2 ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 0)) : 400;
  const std::size_t step = argc > 3 ? static_cast<std::size_t>(std::strtoull(argv[3], nullptr, 0)) : 8;
  const double seconds = argc > 4 ? std::atof(argv[4]) : 0.001;
  const int repeats = argc > 5 ? std::atoi(argv[5]) : 5;

  if (min_an == 0 || max_an < min_an || step == 0 || repeats <= 0 ||
      seconds <= 0.0) {
    std::fprintf(stderr,
                 "usage: %s [min_an=8] [max_an=400] [step=8] [seconds=0.001] [repeats=5]\n"
                 "       %s --check\n",
                 argv[0], argv[0]);
    return 1;
  }

  run_sweep(min_an, max_an, step, seconds, repeats);
  return 0;
}
