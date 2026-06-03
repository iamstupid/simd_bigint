#include "../include/mul_new.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

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

template <std::size_t BN>
void
run_case(std::size_t an52, double seconds, int repeats)
{
  constexpr unsigned cases = 32;
  std::uint64_t seed = 0x942042e4d073fc1bULL ^ (an52 << 17) ^ (BN << 3);

  std::vector<std::vector<U52Block>> au52(cases), bu52(cases), out52(cases);
  const std::size_t out_blocks = (an52 + 7) / 8 + 1;

  for (unsigned c = 0; c < cases; ++c) {
    fill_u52(au52[c], an52, seed);
    fill_u52(bu52[c], BN, seed);
    out52[c].assign(out_blocks, {});
  }

  unsigned iu52 = 0;
  const double total_ns = bench_one([&]() -> std::uint64_t {
    const unsigned c = iu52++ & (cases - 1);
    mul_u52_1(reinterpret_cast<pvec>(out52[c].data()),
              reinterpret_cast<cpvec>(au52[c].data()),
              reinterpret_cast<cpvec>(bu52[(5 * c) & (cases - 1)].data()),
              static_cast<std::uint64_t>(an52), BN);
    return guard_u52(out52[c]);
  }, seconds, repeats);

  const double work = static_cast<double>(an52) * static_cast<double>(BN);
  std::printf("%zu,%zu,%.6f,%.9f\n", an52, BN, total_ns, total_ns / work);
}

template <std::size_t BN>
void
run_bn(std::size_t min_an, std::size_t max_an, double seconds, int repeats)
{
  for (std::size_t an52 = min_an; an52 <= max_an; ++an52)
    run_case<BN>(an52, seconds, repeats);
}

} // namespace

int
main(int argc, char **argv)
{
  const std::size_t min_an = argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 0)) : 8;
  const std::size_t max_an = argc > 2 ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 0)) : 100;
  const double seconds = argc > 3 ? std::atof(argv[3]) : 0.001;
  const int repeats = argc > 4 ? std::atoi(argv[4]) : 3;

  if (min_an == 0 || max_an < min_an || repeats <= 0 || seconds <= 0.0) {
    std::fprintf(stderr, "usage: %s [min_an=8] [max_an=100] [seconds=0.001] [repeats=3]\n",
                 argv[0]);
    return 1;
  }

  std::puts("an52,bn52,total_ns,ns_per_limb");
  run_bn<1>(min_an, max_an, seconds, repeats);
  run_bn<2>(min_an, max_an, seconds, repeats);
  run_bn<3>(min_an, max_an, seconds, repeats);
  run_bn<4>(min_an, max_an, seconds, repeats);
  run_bn<5>(min_an, max_an, seconds, repeats);
  run_bn<6>(min_an, max_an, seconds, repeats);
  run_bn<7>(min_an, max_an, seconds, repeats);
  run_bn<8>(min_an, max_an, seconds, repeats);

  return 0;
}
