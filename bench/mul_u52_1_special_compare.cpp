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

inline static void
mul_u52_1_generic(pvec r, cpvec a, cpvec b, std::uint64_t an,
                  const std::uint64_t bn)
{
  _vec n0, n1, lo0, ax, m[8], bx[8];
  m[0] = m[1] = m[2] = m[3] = m[4] = m[5] = m[6] = m[7] = zero();
  if (bn > 4) {
    bx[7] = splat_load(b, 7);
    bx[6] = splat_load(b, 6);
    bx[5] = splat_load(b, 5);
    bx[4] = splat_load(b, 4);
  }
  bx[3] = splat_load(b, 3);
  bx[2] = splat_load(b, 2);
  bx[1] = splat_load(b, 1);
  bx[0] = splat_load(b, 0);
  for (an = (an + 7) >> 3; an; --an) {
    ax = load_vec(a++);
    n0 = zero();
    n1 = zero();
    switch (bn) {
    case 8: mul1_substep(ax, 7);
    case 7: mul1_substep(ax, 6);
    case 6: mul1_substep(ax, 5);
    case 5: mul1_substep(ax, 4);
    case 4: mul1_substep(ax, 3);
    case 3: mul1_substep(ax, 2);
    case 2: mul1_substep(ax, 1);
    case 1: mul1_substep0(ax);
    }
    n0 = add(n0, lo0);
    store_vec(r++, n0);
  }
  n0 = zero();
  n1 = zero();
  switch (bn) {
  case 8: n0 = add(n0, alignr64(zero(), m[7], 0));
  case 7: n1 = add(n1, alignr64(zero(), m[6], 1));
  case 6: n0 = add(n0, alignr64(zero(), m[5], 2));
  case 5: n1 = add(n1, alignr64(zero(), m[4], 3));
  case 4: n0 = add(n0, alignr64(zero(), m[3], 4));
  case 3: n1 = add(n1, alignr64(zero(), m[2], 5));
  case 2: n0 = add(n0, alignr64(zero(), m[1], 6));
  case 1: n1 = add(n1, alignr64(zero(), m[0], 7));
  }
  n0 = add(n0, n1);
  store_vec(r++, n0);
}

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

template <std::size_t BN, class F>
void
run_case(const char *variant, F &&fn, std::size_t an52, double seconds,
         int repeats)
{
  constexpr unsigned cases = 32;
  std::uint64_t seed = 0x6938f31326f8aa05ULL ^ (an52 << 17) ^ (BN << 3);

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
    fn(reinterpret_cast<pvec>(out52[c].data()),
       reinterpret_cast<cpvec>(au52[c].data()),
       reinterpret_cast<cpvec>(bu52[(5 * c) & (cases - 1)].data()),
       static_cast<std::uint64_t>(an52), BN);
    return guard_u52(out52[c]);
  }, seconds, repeats);

  const double work = static_cast<double>(an52) * static_cast<double>(BN);
  std::printf("%zu,%zu,%s,%.6f,%.9f\n", an52, BN, variant, total_ns,
              total_ns / work);
}

template <std::size_t BN>
void
run_bn(std::size_t min_an, std::size_t max_an, double seconds, int repeats)
{
  for (std::size_t an52 = min_an; an52 <= max_an; ++an52) {
    run_case<BN>("generic", mul_u52_1_generic, an52, seconds, repeats);
    run_case<BN>("special", mul_u52_1, an52, seconds, repeats);
  }
}

} // namespace

int
main(int argc, char **argv)
{
  const std::size_t min_an =
    argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 0)) : 8;
  const std::size_t max_an =
    argc > 2 ? static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 0)) : 100;
  const double seconds = argc > 3 ? std::atof(argv[3]) : 0.001;
  const int repeats = argc > 4 ? std::atoi(argv[4]) : 3;

  if (min_an == 0 || max_an < min_an || repeats <= 0 || seconds <= 0.0) {
    std::fprintf(stderr,
                 "usage: %s [min_an=8] [max_an=100] [seconds=0.001] "
                 "[repeats=3]\n",
                 argv[0]);
    return 1;
  }

  std::puts("an52,bn52,variant,total_ns,ns_per_limb");
  run_bn<1>(min_an, max_an, seconds, repeats);
  run_bn<2>(min_an, max_an, seconds, repeats);
  return 0;
}
