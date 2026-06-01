#include "../reference/ifma52_mul.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" void bench_mul_u52_basecase(unsigned long long *rp,
                                       const unsigned long long *ap,
                                       const unsigned long long *bp,
                                       unsigned long long an,
                                       unsigned long long bn);

namespace {

using simd_bigint::reference::Decoded52;
using simd_bigint::reference::Ifma52Workspace;
using simd_bigint::reference::decode52;
using simd_bigint::reference::mul_basecase_u64;
using simd_bigint::reference::mul_predecoded_u64;

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

void
fill_u64_for_u52_digits(std::vector<std::uint64_t> &x, std::size_t digits,
                        std::uint64_t &seed)
{
  const std::size_t bits = digits * 52;
  const std::size_t limbs = (bits + 63) / 64;
  x.resize(limbs);
  for (std::uint64_t &v : x)
    v = splitmix64(seed);

  const unsigned top_bits = static_cast<unsigned>(bits & 63);
  if (top_bits == 0) {
    x.back() |= std::uint64_t{1} << 63;
  } else {
    x.back() &= (std::uint64_t{1} << top_bits) - 1;
    x.back() |= std::uint64_t{1} << (top_bits - 1);
  }
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

std::size_t
rounded_div(std::size_t n, std::size_t d)
{
  return (n + d / 2) / d;
}

void
run_case(const char *shape, std::size_t an52, std::size_t bn52,
         double seconds, int repeats)
{
  constexpr unsigned cases = 16;
  std::uint64_t seed = 0x8f4c2e5a9d3b7106ULL ^ (an52 << 17) ^ (bn52 << 3);

  std::vector<std::vector<U52Block>> au52(cases), bu52(cases), out52(cases);
  std::vector<std::vector<std::uint64_t>> au64(cases), bu64(cases), out64(cases);
  std::vector<Decoded52> da(cases), db(cases);

  const std::size_t out_blocks = (an52 + 7) / 8 + (bn52 + 7) / 8 + 2;
  for (unsigned c = 0; c < cases; ++c) {
    fill_u52(au52[c], an52, seed);
    fill_u52(bu52[c], bn52, seed);
    out52[c].assign(out_blocks, {});

    fill_u64_for_u52_digits(au64[c], an52, seed);
    fill_u64_for_u52_digits(bu64[c], bn52, seed);
    out64[c].assign(au64[c].size() + bu64[c].size() + 2, 0);
    decode52(da[c], au64[c].data(), au64[c].size());
    decode52(db[c], bu64[c].data(), bu64[c].size());
  }

  Ifma52Workspace ref_core_ws;
  Ifma52Workspace ref_full_ws;
  unsigned iu52 = 0, icore = 0, ifull = 0;

  const double u52_ns = bench_one([&]() -> std::uint64_t {
    const unsigned c = iu52++ & (cases - 1);
    std::memset(out52[c].data(), 0, out52[c].size() * sizeof(U52Block));
    bench_mul_u52_basecase(reinterpret_cast<unsigned long long *>(out52[c].data()),
                           reinterpret_cast<const unsigned long long *>(au52[c].data()),
                           reinterpret_cast<const unsigned long long *>(
                             bu52[(5 * c) & (cases - 1)].data()),
                           static_cast<unsigned long long>(an52),
                           static_cast<unsigned long long>(bn52));
    return guard_u52(out52[c]);
  }, seconds, repeats);

  const double ref_core_ns = bench_one([&]() -> std::uint64_t {
    const unsigned c = icore++ & (cases - 1);
    auto &out = out64[c];
    mul_predecoded_u64(out.data(), da[c], db[(5 * c) & (cases - 1)],
                       au64[c].size() + bu64[(5 * c) & (cases - 1)].size(),
                       ref_core_ws);
    return out[0] ^ out[out.size() / 2] ^ out.back();
  }, seconds, repeats);

  const double ref_full_ns = bench_one([&]() -> std::uint64_t {
    const unsigned c = ifull++ & (cases - 1);
    auto &out = out64[c];
    const auto &b = bu64[(5 * c) & (cases - 1)];
    mul_basecase_u64(out.data(), au64[c].data(), au64[c].size(), b.data(),
                     b.size(), ref_full_ws);
    return out[0] ^ out[out.size() / 2] ^ out.back();
  }, seconds, repeats);

  const double work = static_cast<double>(an52) * static_cast<double>(bn52);
  std::printf("%s,%zu,%zu,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f\n", shape, an52,
              bn52, u52_ns, ref_core_ns, ref_full_ns, u52_ns / work,
              ref_core_ns / work, ref_full_ns / work);
}

} // namespace

int
main(int argc, char **argv)
{
  const std::size_t max_an = argc > 1 ? static_cast<std::size_t>(std::strtoull(argv[1], nullptr, 0)) : 200;
  const double seconds = argc > 2 ? std::atof(argv[2]) : 0.002;
  const int repeats = argc > 3 ? std::atoi(argv[3]) : 3;
  const std::size_t min_an = argc > 4 ? static_cast<std::size_t>(std::strtoull(argv[4], nullptr, 0)) : 8;
  const std::size_t step = argc > 5 ? static_cast<std::size_t>(std::strtoull(argv[5], nullptr, 0)) : 2;

  std::puts("shape,an52,bn52,u52_ns,ref_core_ns,ref_full_ns,"
            "u52_ns_per_digit_product,ref_core_ns_per_digit_product,"
            "ref_full_ns_per_digit_product");

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
    for (std::size_t an52 = min_an; an52 <= max_an; an52 += step)
      run_case(shape.c_str(), an52, fixed_bn, seconds, repeats);
  }

  return 0;
}
