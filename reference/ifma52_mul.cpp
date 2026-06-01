#include "ifma52_mul.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <immintrin.h>
#include <memory_resource>
#include <utility>

namespace simd_bigint::reference {

CountingMemoryResource::CountingMemoryResource(std::pmr::memory_resource *upstream)
  : upstream_(upstream)
{
}

void
CountingMemoryResource::reset_stats()
{
  stats_ = {};
}

ToomScratchStats
CountingMemoryResource::stats() const
{
  return stats_;
}

void *
CountingMemoryResource::do_allocate(std::size_t bytes, std::size_t alignment)
{
  void *p = upstream_->allocate(bytes, alignment);
  ++stats_.allocation_count;
  stats_.allocated_bytes += bytes;
  stats_.current_live_bytes += bytes;
  stats_.peak_live_bytes =
    std::max(stats_.peak_live_bytes, stats_.current_live_bytes);
  return p;
}

void
CountingMemoryResource::do_deallocate(void *p, std::size_t bytes,
                                      std::size_t alignment)
{
  upstream_->deallocate(p, bytes, alignment);
  stats_.current_live_bytes =
    bytes < stats_.current_live_bytes ? stats_.current_live_bytes - bytes : 0;
}

bool
CountingMemoryResource::do_is_equal(const std::pmr::memory_resource &other) const
  noexcept
{
  return this == &other;
}

ToomWorkspace::ToomWorkspace(std::size_t initial_bytes)
  : storage(initial_bytes), arena(storage.data(), storage.size()),
    resource(&arena)
{
}

void
ToomWorkspace::reset()
{
  arena.release();
  resource.reset_stats();
}

ToomScratchStats
ToomWorkspace::scratch_stats() const
{
  return resource.stats();
}

namespace {

constexpr std::uint64_t DIGIT_MASK = (std::uint64_t{1} << 52) - 1;
constexpr __mmask64 STORE52_MASK = (std::uint64_t{1} << 52) - 1;
constexpr std::size_t TOOM22_MIN_LIMBS = 72;
constexpr std::size_t TOOM32_MIN_CHUNK = 48;
constexpr std::size_t TOOM33_MIN_LIMBS = 144;

class LimbVec {
public:
  LimbVec() = default;
  explicit LimbVec(std::pmr::memory_resource *mem) : mem_(mem) {}
  LimbVec(std::size_t n, std::uint64_t value, std::pmr::memory_resource *mem)
    : mem_(mem)
  {
    resize_for_overwrite(n);
    for (std::size_t i = 0; i < n_; ++i)
      p_[i] = value;
  }

  LimbVec(const LimbVec &) = delete;
  LimbVec &operator=(const LimbVec &) = delete;

  LimbVec(LimbVec &&other) noexcept { move_from(other); }
  LimbVec &operator=(LimbVec &&other) noexcept
  {
    if (this != &other) {
      drop();
      move_from(other);
    }
    return *this;
  }

  ~LimbVec() { drop(); }

  bool empty() const { return n_ == 0; }
  std::size_t size() const { return n_; }
  std::size_t capacity() const { return cap_; }
  std::uint64_t *data() { return p_; }
  const std::uint64_t *data() const { return p_; }
  std::uint64_t &operator[](std::size_t i) { return p_[i]; }
  std::uint64_t operator[](std::size_t i) const { return p_[i]; }
  std::uint64_t &back() { return p_[n_ - 1]; }
  std::uint64_t back() const { return p_[n_ - 1]; }

  void clear() { n_ = 0; }
  void pop_back()
  {
    assert(n_ != 0);
    --n_;
  }

  void reserve(std::size_t want)
  {
    if (want <= cap_)
      return;
    assert(mem_ != nullptr);

    void *raw = mem_->allocate(want * sizeof(std::uint64_t), 64);
    auto *np = static_cast<std::uint64_t *>(raw);
    for (std::size_t i = 0; i < n_; ++i)
      np[i] = p_[i];
    p_ = np;
    cap_ = want;
  }

  void resize(std::size_t want)
  {
    const std::size_t old = n_;
    reserve(want);
    n_ = want;
    for (std::size_t i = old; i < n_; ++i)
      p_[i] = 0;
  }

  void resize_for_overwrite(std::size_t want)
  {
    reserve(want);
    n_ = want;
  }

private:
  void drop()
  {
    p_ = nullptr;
    n_ = 0;
    cap_ = 0;
  }

  void move_from(LimbVec &other)
  {
    mem_ = other.mem_;
    p_ = other.p_;
    n_ = other.n_;
    cap_ = other.cap_;
    other.mem_ = nullptr;
    other.p_ = nullptr;
    other.n_ = 0;
    other.cap_ = 0;
  }

  std::pmr::memory_resource *mem_ = nullptr;
  std::uint64_t *p_ = nullptr;
  std::size_t n_ = 0;
  std::size_t cap_ = 0;
};

struct LimbSpan {
  const std::uint64_t *p = nullptr;
  std::size_t n = 0;

  bool empty() const { return n == 0; }
  std::size_t size() const { return n; }
  const std::uint64_t *data() const { return p; }
  std::uint64_t operator[](std::size_t i) const { return p[i]; }
};

struct SignedMag {
  int sign = 0;
  LimbVec limbs;
};

static_assert(alignof(Block52) == 64);
static_assert(sizeof(Block52) == 64);

std::size_t
normalized_size(const std::uint64_t *xp, std::size_t n)
{
  while (n != 0 && xp[n - 1] == 0)
    --n;
  return n;
}

void
trim(LimbVec &x)
{
  while (!x.empty() && x.back() == 0)
    x.pop_back();
}

LimbSpan
chunk_span(const std::uint64_t *xp, std::size_t n, std::size_t off,
           std::size_t len)
{
  if (off >= n || len == 0)
    return {};

  std::size_t avail = std::min(len, n - off);
  while (avail != 0 && xp[off + avail - 1] == 0)
    --avail;
  return {xp + off, avail};
}

template <class A, class B>
int
cmp_abs(const A &a, const B &b)
{
  if (a.size() != b.size())
    return a.size() < b.size() ? -1 : 1;
  for (std::size_t i = a.size(); i-- != 0;) {
    if (a[i] != b[i])
      return a[i] < b[i] ? -1 : 1;
  }
  return 0;
}

template <class A, class B>
void
add_abs_into(LimbVec &out, const A &a, const B &b,
             std::size_t reserve_limbs = 0)
{
  const std::size_t n = std::max(a.size(), b.size());
  out.reserve(std::max(n + 1, reserve_limbs));
  out.resize(n + 1);
  unsigned __int128 carry = 0;

  for (std::size_t i = 0; i < n; ++i) {
    const unsigned __int128 av = i < a.size() ? a[i] : 0;
    const unsigned __int128 bv = i < b.size() ? b[i] : 0;
    const unsigned __int128 s = av + bv + carry;
    out[i] = static_cast<std::uint64_t>(s);
    carry = s >> 64;
  }
  out[n] = static_cast<std::uint64_t>(carry);
  trim(out);
}

template <class A, class B>
LimbVec
add_abs(const A &a, const B &b, std::pmr::memory_resource *mem,
        std::size_t reserve_limbs = 0)
{
  LimbVec out(mem);
  add_abs_into(out, a, b, reserve_limbs);
  return out;
}

template <class X>
LimbVec
clone_abs(const X &x, std::pmr::memory_resource *mem,
          std::size_t reserve_limbs = 0)
{
  LimbVec out(mem);
  out.reserve(std::max(x.size(), reserve_limbs));
  out.resize(x.size());
  for (std::size_t i = 0; i < x.size(); ++i)
    out[i] = x[i];
  return out;
}

template <class B>
void
add_assign_abs(LimbVec &a, const B &b)
{
  const std::size_t n = std::max(a.size(), b.size());
  a.resize(n + 1);
  unsigned __int128 carry = 0;

  for (std::size_t i = 0; i < n; ++i) {
    const unsigned __int128 av = a[i];
    const unsigned __int128 bv = i < b.size() ? b[i] : 0;
    const unsigned __int128 s = av + bv + carry;
    a[i] = static_cast<std::uint64_t>(s);
    carry = s >> 64;
  }
  a[n] = static_cast<std::uint64_t>(carry);
  trim(a);
}

template <class X0, class X1>
void
lincomb2_abs_into(LimbVec &out, const X0 &x0, const X1 &x1,
                  std::uint32_t c1, std::size_t reserve_limbs = 0)
{
  const std::size_t n = std::max(x0.size(), x1.size());
  out.reserve(std::max(n + 1, reserve_limbs));
  out.resize(n + 1);
  unsigned __int128 carry = 0;

  for (std::size_t i = 0; i < n; ++i) {
    const unsigned __int128 v0 = i < x0.size() ? x0[i] : 0;
    const unsigned __int128 v1 = i < x1.size() ? x1[i] : 0;
    const unsigned __int128 s = v0 + v1 * c1 + carry;
    out[i] = static_cast<std::uint64_t>(s);
    carry = s >> 64;
  }
  out[n] = static_cast<std::uint64_t>(carry);
  trim(out);
}

template <class X0, class X1>
LimbVec
lincomb2_abs(const X0 &x0, const X1 &x1, std::uint32_t c1,
             std::pmr::memory_resource *mem, std::size_t reserve_limbs = 0)
{
  LimbVec out(mem);
  lincomb2_abs_into(out, x0, x1, c1, reserve_limbs);
  return out;
}

template <class X0, class X1, class X2>
void
lincomb3_abs_into(LimbVec &out, const X0 &x0, const X1 &x1,
                  std::uint32_t c1, const X2 &x2, std::uint32_t c2,
                  std::size_t reserve_limbs = 0)
{
  const std::size_t n = std::max({x0.size(), x1.size(), x2.size()});
  out.reserve(std::max(n + 1, reserve_limbs));
  out.resize(n + 1);
  unsigned __int128 carry = 0;

  for (std::size_t i = 0; i < n; ++i) {
    const unsigned __int128 v0 = i < x0.size() ? x0[i] : 0;
    const unsigned __int128 v1 = i < x1.size() ? x1[i] : 0;
    const unsigned __int128 v2 = i < x2.size() ? x2[i] : 0;
    const unsigned __int128 s = v0 + v1 * c1 + v2 * c2 + carry;
    out[i] = static_cast<std::uint64_t>(s);
    carry = s >> 64;
  }
  out[n] = static_cast<std::uint64_t>(carry);
  trim(out);
}

template <class X0, class X1, class X2>
LimbVec
lincomb3_abs(const X0 &x0, const X1 &x1, std::uint32_t c1,
             const X2 &x2, std::uint32_t c2, std::pmr::memory_resource *mem,
             std::size_t reserve_limbs = 0)
{
  LimbVec out(mem);
  lincomb3_abs_into(out, x0, x1, c1, x2, c2, reserve_limbs);
  return out;
}

template <class B>
void
sub_assign_abs(LimbVec &a, const B &b)
{
  assert(cmp_abs(a, b) >= 0);

  unsigned __int128 borrow = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const unsigned __int128 av = a[i];
    const unsigned __int128 bv = (i < b.size() ? b[i] : 0) + borrow;
    if (av < bv) {
      a[i] =
        static_cast<std::uint64_t>((static_cast<unsigned __int128>(1) << 64) + av - bv);
      borrow = 1;
    } else {
      a[i] = static_cast<std::uint64_t>(av - bv);
      borrow = 0;
    }
  }

  assert(borrow == 0);
  trim(a);
}

template <class A>
void
sub_current_from_abs(LimbVec &dst, const A &a)
{
  assert(cmp_abs(a, dst) > 0);

  const std::size_t old_size = dst.size();
  dst.resize(a.size());
  unsigned __int128 borrow = 0;
  for (std::size_t i = 0; i < dst.size(); ++i) {
    const unsigned __int128 av = a[i];
    const unsigned __int128 bv = (i < old_size ? dst[i] : 0) + borrow;
    if (av < bv) {
      dst[i] =
        static_cast<std::uint64_t>((static_cast<unsigned __int128>(1) << 64) + av - bv);
      borrow = 1;
    } else {
      dst[i] = static_cast<std::uint64_t>(av - bv);
      borrow = 0;
    }
  }

  assert(borrow == 0);
  trim(dst);
}

template <class B>
void
sub_mul_small_assign_abs(LimbVec &a, const B &b, std::uint32_t m)
{
  if (b.empty() || m == 0)
    return;

  assert(b.size() <= a.size());
  unsigned __int128 borrow = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    unsigned __int128 p = borrow;
    if (i < b.size())
      p += static_cast<unsigned __int128>(b[i]) * m;

    const std::uint64_t lo = static_cast<std::uint64_t>(p);
    const unsigned __int128 hi = p >> 64;
    const std::uint64_t av = a[i];
    if (av < lo) {
      a[i] = static_cast<std::uint64_t>(
        (static_cast<unsigned __int128>(1) << 64) + av - lo);
      borrow = hi + 1;
    } else {
      a[i] = av - lo;
      borrow = hi;
    }
  }

  assert(borrow == 0);
  trim(a);
}

void
divexact_abs_2_inplace(LimbVec &x)
{
  if (x.empty())
    return;

  std::uint64_t carry = 0;
  for (std::size_t i = x.size(); i-- != 0;) {
    const std::uint64_t next_carry = x[i] & 1;
    x[i] = (x[i] >> 1) | (carry << 63);
    carry = next_carry;
  }
  assert(carry == 0);
  trim(x);
}

void
divexact_abs_3_inplace(LimbVec &x)
{
  if (x.empty())
    return;

  constexpr std::uint64_t INV3 = 0xaaaaaaaaaaaaaaabULL;
  std::uint64_t carry = 0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    const std::uint64_t q = (x[i] - carry) * INV3;
    const unsigned __int128 product = static_cast<unsigned __int128>(q) * 3 + carry;
    assert(static_cast<std::uint64_t>(product) == x[i]);
    x[i] = q;
    carry = static_cast<std::uint64_t>(product >> 64);
  }
  assert(carry == 0);
  trim(x);
}

template <class X0, class X1, class X2>
SignedMag
eval_neg1_3_into(LimbVec &even, const X0 &x0, const X1 &x1,
                 const X2 &x2, std::size_t reserve_limbs = 0)
{
  add_abs_into(even, x0, x2, reserve_limbs);
  const int c = cmp_abs(even, x1);
  if (c == 0)
    even.clear();
  else if (c > 0) {
    sub_assign_abs(even, x1);
    return {1, std::move(even)};
  } else {
    sub_current_from_abs(even, x1);
    return {-1, std::move(even)};
  }

  return {0, std::move(even)};
}

template <class X0, class X1, class X2>
SignedMag
eval_neg1_3(const X0 &x0, const X1 &x1, const X2 &x2,
            std::pmr::memory_resource *mem, std::size_t reserve_limbs = 0)
{
  LimbVec even(mem);
  return eval_neg1_3_into(even, x0, x1, x2, reserve_limbs);
}

template <class Y>
void
sub_positive_from_signed_inplace(SignedMag &x, const Y &y,
                                 std::pmr::memory_resource *mem)
{
  if (y.empty())
    return;

  if (x.sign == 0) {
    x.sign = -1;
    x.limbs = clone_abs(y, mem);
    return;
  }

  if (x.sign < 0) {
    add_assign_abs(x.limbs, y);
    return;
  }

  const int c = cmp_abs(x.limbs, y);
  if (c == 0) {
    x.sign = 0;
    x.limbs.clear();
    return;
  }
  if (c > 0) {
    sub_assign_abs(x.limbs, y);
    return;
  }

  sub_current_from_abs(x.limbs, y);
  x.sign = -1;
}

void
add_shifted_to_output(std::uint64_t *rp, std::size_t rn, const LimbVec &x,
                      std::size_t shift)
{
  if (x.empty())
    return;

  assert(shift < rn);

  unsigned __int128 carry = 0;
  std::size_t i = 0;
  for (; i < x.size() && shift + i < rn; ++i) {
    const unsigned __int128 s =
      static_cast<unsigned __int128>(rp[shift + i]) + x[i] + carry;
    rp[shift + i] = static_cast<std::uint64_t>(s);
    carry = s >> 64;
  }

  for (std::size_t j = shift + i; carry != 0 && j < rn; ++j) {
    const unsigned __int128 s = static_cast<unsigned __int128>(rp[j]) + carry;
    rp[j] = static_cast<std::uint64_t>(s);
    carry = s >> 64;
  }
  assert(i == x.size());
  assert(carry == 0);
}

template <class X>
void
sub_shifted_from_output(std::uint64_t *rp, std::size_t rn, const X &x,
                        std::size_t shift)
{
  if (x.empty())
    return;

  assert(shift < rn);

  unsigned __int128 borrow = 0;
  std::size_t i = 0;
  for (; i < x.size() && shift + i < rn; ++i) {
    const unsigned __int128 av = rp[shift + i];
    const unsigned __int128 bv = x[i] + borrow;
    if (av < bv) {
      rp[shift + i] =
        static_cast<std::uint64_t>((static_cast<unsigned __int128>(1) << 64) + av - bv);
      borrow = 1;
    } else {
      rp[shift + i] = static_cast<std::uint64_t>(av - bv);
      borrow = 0;
    }
  }

  for (std::size_t j = shift + i; borrow != 0 && j < rn; ++j) {
    if (rp[j] == 0) {
      rp[j] = ~std::uint64_t{0};
    } else {
      --rp[j];
      borrow = 0;
    }
  }
  assert(i == x.size());
  assert(borrow == 0);
}

std::uint64_t
load64u(const std::uint8_t *p)
{
  std::uint64_t x;
  std::memcpy(&x, p, sizeof(x));
  return x;
}

std::uint64_t
load64_bounded(const std::uint8_t *p, std::size_t bytes, std::size_t off)
{
  if (off + 8 <= bytes)
    return load64u(p + off);
  if (off < bytes) {
    const unsigned rem = static_cast<unsigned>(bytes - off);
    const __m128i v = _mm_maskz_loadu_epi8((std::uint16_t{1} << rem) - 1, p + off);
    return static_cast<std::uint64_t>(_mm_cvtsi128_si64(v));
  }
  return 0;
}

std::size_t
bit_length(const std::uint64_t *xp, std::size_t n)
{
  n = normalized_size(xp, n);
  if (n == 0)
    return 0;
  return (n - 1) * 64 + (64 - static_cast<std::size_t>(__builtin_clzll(xp[n - 1])));
}

unsigned
block_len(const Decoded52 &x, std::size_t i)
{
  return i + 1 == x.nblocks ? x.tail_len : 8;
}

__m512i
setr_epi64(std::uint64_t e0, std::uint64_t e1, std::uint64_t e2,
           std::uint64_t e3, std::uint64_t e4, std::uint64_t e5,
           std::uint64_t e6, std::uint64_t e7)
{
  return _mm512_set_epi64(static_cast<long long>(e7), static_cast<long long>(e6),
                          static_cast<long long>(e5), static_cast<long long>(e4),
                          static_cast<long long>(e3), static_cast<long long>(e2),
                          static_cast<long long>(e1), static_cast<long long>(e0));
}

template <int S>
__m512i
lane_shl(const __m512i x)
{
  if constexpr (S == 0)
    return x;
  else
    return _mm512_alignr_epi64(x, _mm512_setzero_si512(), 8 - S);
}

template <int S>
__m512i
lane_shr(const __m512i x)
{
  if constexpr (S == 0)
    return x;
  else
    return _mm512_alignr_epi64(_mm512_setzero_si512(), x, S);
}

void
muladd_lane(const __m512i a, const std::uint64_t b, __m512i &lo, __m512i &hi)
{
  const __m512i bj = _mm512_set1_epi64(static_cast<long long>(b));
  lo = _mm512_madd52lo_epu64(lo, a, bj);
  hi = _mm512_madd52hi_epu64(hi, a, bj);
}

void
muladd8(const __m512i a, const std::uint64_t *b, __m512i acc[9])
{
  muladd_lane(a, b[0], acc[0], acc[1]);
  muladd_lane(a, b[1], acc[1], acc[2]);
  muladd_lane(a, b[2], acc[2], acc[3]);
  muladd_lane(a, b[3], acc[3], acc[4]);
  muladd_lane(a, b[4], acc[4], acc[5]);
  muladd_lane(a, b[5], acc[5], acc[6]);
  muladd_lane(a, b[6], acc[6], acc[7]);
  muladd_lane(a, b[7], acc[7], acc[8]);
}

void
muladd_duff(const __m512i a, const std::uint64_t *b, unsigned len,
            __m512i acc[9])
{
  switch (len) {
  case 8:
    muladd_lane(a, b[7], acc[7], acc[8]);
    [[fallthrough]];
  case 7:
    muladd_lane(a, b[6], acc[6], acc[7]);
    [[fallthrough]];
  case 6:
    muladd_lane(a, b[5], acc[5], acc[6]);
    [[fallthrough]];
  case 5:
    muladd_lane(a, b[4], acc[4], acc[5]);
    [[fallthrough]];
  case 4:
    muladd_lane(a, b[3], acc[3], acc[4]);
    [[fallthrough]];
  case 3:
    muladd_lane(a, b[2], acc[2], acc[3]);
    [[fallthrough]];
  case 2:
    muladd_lane(a, b[1], acc[1], acc[2]);
    [[fallthrough]];
  case 1:
    muladd_lane(a, b[0], acc[0], acc[1]);
    [[fallthrough]];
  default:
    break;
  }
}

void
muladd_block(const Block52 &ab, unsigned alen, const Block52 &bb,
             unsigned blen, __m512i acc[9])
{
  if (alen == 8) {
    const __m512i a = _mm512_load_si512(reinterpret_cast<const void *>(ab.v));
    if (blen == 8)
      muladd8(a, bb.v, acc);
    else
      muladd_duff(a, bb.v, blen, acc);
  } else if (blen == 8) {
    const __m512i b = _mm512_load_si512(reinterpret_cast<const void *>(bb.v));
    muladd_duff(b, ab.v, alen, acc);
  } else {
    const __m512i a = _mm512_load_si512(reinterpret_cast<const void *>(ab.v));
    muladd_duff(a, bb.v, blen, acc);
  }
}

void
harvest(__m512i acc[9], __m512i &lo, __m512i &hi)
{
  lo = acc[0];
  hi = acc[8];

  lo = _mm512_add_epi64(lo, lane_shl<1>(acc[1]));
  hi = _mm512_add_epi64(hi, lane_shr<7>(acc[1]));
  lo = _mm512_add_epi64(lo, lane_shl<2>(acc[2]));
  hi = _mm512_add_epi64(hi, lane_shr<6>(acc[2]));
  lo = _mm512_add_epi64(lo, lane_shl<3>(acc[3]));
  hi = _mm512_add_epi64(hi, lane_shr<5>(acc[3]));
  lo = _mm512_add_epi64(lo, lane_shl<4>(acc[4]));
  hi = _mm512_add_epi64(hi, lane_shr<4>(acc[4]));
  lo = _mm512_add_epi64(lo, lane_shl<5>(acc[5]));
  hi = _mm512_add_epi64(hi, lane_shr<3>(acc[5]));
  lo = _mm512_add_epi64(lo, lane_shl<6>(acc[6]));
  hi = _mm512_add_epi64(hi, lane_shr<2>(acc[6]));
  lo = _mm512_add_epi64(lo, lane_shl<7>(acc[7]));
  hi = _mm512_add_epi64(hi, lane_shr<1>(acc[7]));
}

__m512i
pack52_vec(const __m512i d)
{
  alignas(64) static constexpr std::uint8_t rperm[64] = {
    0,  1,  2,  3,  4,  5,  6,  7,  9,  10, 11, 12, 13,
    14, 15, 7,  19, 20, 21, 22, 23, 7,  7,  7,  28, 29,
    30, 31, 7,  7,  7,  7,  38, 7,  7,  7,  7,  7,  7,
    48, 49, 50, 51, 52, 53, 54, 55, 7,  58, 59, 60, 61,
    62, 63, 7,  7,  7,  7,  7,  7,  7,  7,  7,  7};
  alignas(64) static constexpr std::uint8_t lperm[64] = {
    7,  7,  7,  7,  7,  7,  8,  9,  7,  7,  7,  7,  7,
    16, 17, 18, 7,  7,  7,  24, 25, 26, 27, 28, 7,  7,
    32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45, 46,
    7,  7,  7,  7,  7,  7,  56, 57, 58, 7,  7,  7,  7,
    7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7};

  const __m512i rsrc =
    _mm512_permutexvar_epi8(_mm512_load_si512(reinterpret_cast<const void *>(rperm)), d);
  const __m512i lsrc =
    _mm512_permutexvar_epi8(_mm512_load_si512(reinterpret_cast<const void *>(lperm)), d);
  const __m512i rsh = setr_epi64(0, 4, 0, 4, 0, 0, 4, 0);
  const __m512i lsh = setr_epi64(4, 0, 4, 0, 4, 4, 0, 0);

  return _mm512_or_si512(_mm512_srlv_epi64(rsrc, rsh),
                         _mm512_sllv_epi64(lsrc, lsh));
}

void
store52_full(std::uint8_t *dst, const __m512i d)
{
  _mm512_mask_storeu_epi8(dst, STORE52_MASK, pack52_vec(d));
}

void
store52_tail(std::uint8_t *dst, std::size_t bytes, const __m512i d)
{
  if (bytes >= 52)
    store52_full(dst, d);
  else if (bytes != 0)
    _mm512_mask_storeu_epi8(dst, (std::uint64_t{1} << bytes) - 1, pack52_vec(d));
}

__m512i
normalize52(const __m512i x0, __m512i &carry_to_next)
{
  const __m512i mask = _mm512_set1_epi64(static_cast<long long>(DIGIT_MASK));
  const __m512i one = _mm512_set1_epi64(1);
  const __m512i q = _mm512_srli_epi64(x0, 52);
  __m512i s = _mm512_add_epi64(_mm512_and_si512(x0, mask), lane_shl<1>(q));
  __m512i d = _mm512_and_si512(s, mask);

  std::uint16_t c = static_cast<std::uint16_t>(
                      _mm512_cmpneq_epu64_mask(_mm512_srli_epi64(s, 52),
                                               _mm512_setzero_si512()))
                    << 1;
  const std::uint16_t m =
    static_cast<std::uint16_t>(_mm512_cmpeq_epu64_mask(d, mask));
  c += m;

  d = _mm512_and_si512(
    _mm512_add_epi64(d, _mm512_maskz_mov_epi64((c ^ m) & 0xff, one)), mask);
  carry_to_next =
    _mm512_add_epi64(lane_shr<7>(q), _mm512_maskz_mov_epi64((c >> 8) & 1, one));
  return d;
}

__m512i
normalize_store(std::uint8_t *dst, std::size_t bytes, const __m512i lo,
                const __m512i hi)
{
  __m512i carry;
  const __m512i digits = normalize52(lo, carry);
  store52_tail(dst, bytes, digits);
  return _mm512_add_epi64(hi, carry);
}

template <class A, class B>
void
mul_mag_ifma_into(LimbVec &out, const A &a, const B &b, Ifma52Workspace &ws)
{
  if (a.empty() || b.empty()) {
    out.clear();
    return;
  }

  const std::size_t an = a.size();
  const std::size_t bn = b.size();
  decode52(ws.a, a.data(), an);
  decode52(ws.b, b.data(), bn);
  out.resize_for_overwrite(an + bn);
  mul_predecoded_u64(out.data(), ws.a, ws.b, out.size(), ws);
  trim(out);
}

template <class A, class B>
LimbVec
mul_mag_ifma(const A &a, const B &b, std::pmr::memory_resource *mem,
             Ifma52Workspace &ws)
{
  LimbVec out(mem);
  mul_mag_ifma_into(out, a, b, ws);
  return out;
}

template <class A, class B>
LimbSpan
mul_mag_ifma_to_output(std::uint64_t *dst, const A &a, const B &b,
                       Ifma52Workspace &ws)
{
  if (a.empty() || b.empty())
    return {};

  const std::size_t n = a.size() + b.size();
  mul_basecase_u64(dst, a.data(), a.size(), b.data(), b.size(), ws);
  return {dst, normalized_size(dst, n)};
}

} // namespace

void
decode52(Decoded52 &dst, const std::uint64_t *xp, std::size_t n)
{
  const std::size_t bits = bit_length(xp, n);
  const std::size_t digits = std::max<std::size_t>(1, (bits + 51) / 52);
  const std::size_t blocks = (digits + 7) / 8;
  const auto *bytes = reinterpret_cast<const std::uint8_t *>(xp);
  const std::size_t byte_count = n * sizeof(xp[0]);

  dst.nblocks = blocks;
  dst.digits = digits;
  dst.tail_len = static_cast<std::uint8_t>(((digits - 1) & 7) + 1);
  dst.blocks.resize(blocks);

  for (std::size_t bi = 0; bi < blocks; ++bi) {
    Block52 &blk = dst.blocks[bi];
    const std::size_t base_digit = 8 * bi;
    const std::size_t len = std::min<std::size_t>(8, digits - base_digit);

    for (std::size_t lane = 0; lane < 8; ++lane) {
      const std::size_t digit = base_digit + lane;
      if (lane >= len) {
        blk.v[lane] = 0;
        continue;
      }

      const std::size_t bit = 52 * digit;
      const std::size_t byte = bit >> 3;
      const unsigned shift = static_cast<unsigned>(bit & 7);
      blk.v[lane] = (load64_bounded(bytes, byte_count, byte) >> shift) & DIGIT_MASK;
    }
  }
}

void
mul_predecoded_u64(std::uint64_t *rp, const Decoded52 &aa, const Decoded52 &bb,
                   std::size_t out_limbs, Ifma52Workspace &ws)
{
  (void)ws;

  if (out_limbs == 0)
    return;

  const Decoded52 *ap = &aa;
  const Decoded52 *bp = &bb;
  if (ap->nblocks < bp->nblocks)
    std::swap(ap, bp);

  const std::size_t an = ap->nblocks;
  const std::size_t bn = bp->nblocks;
  const std::size_t diagonals = an + bn - 1;
  const std::size_t out_bytes = out_limbs * sizeof(rp[0]);
  auto *dst_bytes = reinterpret_cast<std::uint8_t *>(rp);

  __m512i prev_hi = _mm512_setzero_si512();

  for (std::size_t k = 0; k < diagonals; ++k) {
    __m512i acc[9] = {
      _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(),
      _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512(),
      _mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512()};

    const std::size_t i0 = k >= bn - 1 ? k - (bn - 1) : 0;
    const std::size_t i1 = std::min(k, an - 1);
    for (std::size_t i = i0; i <= i1; ++i) {
      const std::size_t j = k - i;
      muladd_block(ap->blocks[i], block_len(*ap, i), bp->blocks[j],
                   block_len(*bp, j), acc);
    }

    __m512i lo, hi;
    harvest(acc, lo, hi);
    lo = _mm512_add_epi64(lo, prev_hi);
    const std::size_t off = 52 * k;
    prev_hi =
      normalize_store(dst_bytes + off, out_bytes > off ? out_bytes - off : 0, lo, hi);
  }

  __m512i carry;
  const __m512i digits = normalize52(prev_hi, carry);
  const std::size_t final_off = 52 * diagonals;
  if (final_off < out_bytes)
    store52_tail(dst_bytes + final_off, out_bytes - final_off, digits);
}

void
mul_basecase_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
                 const std::uint64_t *bp, std::size_t bn, Ifma52Workspace &ws)
{
  const std::size_t orig_an = an;
  const std::size_t orig_bn = bn;
  const std::size_t out_limbs = an + bn;

  an = normalized_size(ap, an);
  bn = normalized_size(bp, bn);

  if (an == 0 || bn == 0) {
    if (out_limbs != 0)
      std::memset(rp, 0, out_limbs * sizeof(rp[0]));
    return;
  }

  if (an != orig_an || bn != orig_bn)
    std::memset(rp, 0, out_limbs * sizeof(rp[0]));

  decode52(ws.a, ap, an);
  decode52(ws.b, bp, bn);
  mul_predecoded_u64(rp, ws.a, ws.b, out_limbs, ws);
}

void
mul_basecase_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
                 const std::uint64_t *bp, std::size_t bn)
{
  Ifma52Workspace ws;
  mul_basecase_u64(rp, ap, an, bp, bn, ws);
}

void
mul_auto_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
             const std::uint64_t *bp, std::size_t bn)
{
  ToomWorkspace workspace;
  mul_auto_u64(rp, ap, an, bp, bn, workspace);
}

void
mul_auto_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
             const std::uint64_t *bp, std::size_t bn, ToomWorkspace &workspace)
{
  const std::size_t norm_an = normalized_size(ap, an);
  const std::size_t norm_bn = normalized_size(bp, bn);

  if (norm_an == 0 || norm_bn == 0) {
    workspace.reset();
    if (an + bn != 0)
      std::memset(rp, 0, (an + bn) * sizeof(rp[0]));
    return;
  }

  std::size_t long_n = norm_an;
  std::size_t short_n = norm_bn;
  if (long_n < short_n)
    std::swap(long_n, short_n);

  const std::size_t k32 =
    std::max((long_n + 2) / 3, (short_n + 1) / 2);
  const bool true_toom32 =
    2 * k32 < long_n && k32 < short_n && k32 >= TOOM32_MIN_CHUNK;
  if (true_toom32) {
    mul_toom32_u64(rp, ap, an, bp, bn, workspace);
    return;
  }

  if (short_n >= TOOM33_MIN_LIMBS) {
    mul_toom33_u64(rp, ap, an, bp, bn, workspace);
    return;
  }

  if (short_n >= TOOM22_MIN_LIMBS) {
    mul_toom22_u64(rp, ap, an, bp, bn, workspace);
    return;
  }

  workspace.reset();
  mul_basecase_u64(rp, ap, an, bp, bn, workspace.point_ws);
}

void
mul_toom22_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
               const std::uint64_t *bp, std::size_t bn)
{
  ToomWorkspace workspace;
  mul_toom22_u64(rp, ap, an, bp, bn, workspace);
}

void
mul_toom22_raw_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
                   const std::uint64_t *bp, std::size_t bn)
{
  ToomWorkspace workspace;
  mul_toom22_raw_u64(rp, ap, an, bp, bn, workspace);
}

void
mul_toom22_raw_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
                   const std::uint64_t *bp, std::size_t bn, ToomWorkspace &workspace)
{
  const std::size_t out_limbs = an + bn;
  workspace.reset();
  const std::size_t norm_an = normalized_size(ap, an);
  const std::size_t norm_bn = normalized_size(bp, bn);
  if (norm_an == 0 || norm_bn == 0) {
    if (out_limbs != 0)
      std::memset(rp, 0, out_limbs * sizeof(rp[0]));
    return;
  }
  an = norm_an;
  bn = norm_bn;

  std::pmr::memory_resource *mem = &workspace.resource;

  const std::size_t k = (std::max(an, bn) + 1) / 2;
  const LimbSpan a0 = chunk_span(ap, an, 0, k);
  const LimbSpan a1 = chunk_span(ap, an, k, an - std::min(k, an));
  const LimbSpan b0 = chunk_span(bp, bn, 0, k);
  const LimbSpan b1 = chunk_span(bp, bn, k, bn - std::min(k, bn));

  if (out_limbs != 0)
    std::memset(rp, 0, out_limbs * sizeof(rp[0]));

  Ifma52Workspace &point_ws = workspace.point_ws;
  LimbVec s_a = add_abs(a0, a1, mem, 2 * k + 2);
  const LimbVec s_b = add_abs(b0, b1, mem);
  const LimbSpan c0 = mul_mag_ifma_to_output(rp, a0, b0, point_ws);
  const LimbSpan c2 = mul_mag_ifma_to_output(rp + 2 * k, a1, b1, point_ws);
  mul_mag_ifma_into(s_a, s_a, s_b, point_ws);
  sub_assign_abs(s_a, c0);
  sub_assign_abs(s_a, c2);

  add_shifted_to_output(rp, out_limbs, s_a, k);
}

void
mul_toom22_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
               const std::uint64_t *bp, std::size_t bn, ToomWorkspace &workspace)
{
  const std::size_t norm_an = normalized_size(ap, an);
  const std::size_t norm_bn = normalized_size(bp, bn);
  if (std::min(norm_an, norm_bn) < TOOM22_MIN_LIMBS) {
    workspace.reset();
    mul_basecase_u64(rp, ap, an, bp, bn, workspace.point_ws);
    return;
  }

  mul_toom22_raw_u64(rp, ap, an, bp, bn, workspace);
}

void
mul_toom32_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
               const std::uint64_t *bp, std::size_t bn)
{
  ToomWorkspace workspace;
  mul_toom32_u64(rp, ap, an, bp, bn, workspace);
}

void
mul_toom32_raw_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
                   const std::uint64_t *bp, std::size_t bn)
{
  ToomWorkspace workspace;
  mul_toom32_raw_u64(rp, ap, an, bp, bn, workspace);
}

void
mul_toom32_raw_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
                   const std::uint64_t *bp, std::size_t bn, ToomWorkspace &workspace)
{
  const std::size_t norm_an = normalized_size(ap, an);
  const std::size_t norm_bn = normalized_size(bp, bn);
  if (norm_an < norm_bn) {
    mul_toom32_raw_u64(rp, bp, bn, ap, an, workspace);
    return;
  }

  const std::size_t out_limbs = an + bn;
  workspace.reset();
  if (norm_an == 0 || norm_bn == 0) {
    if (out_limbs != 0)
      std::memset(rp, 0, out_limbs * sizeof(rp[0]));
    return;
  }
  an = norm_an;
  bn = norm_bn;

  const std::size_t k = std::max((an + 2) / 3, (bn + 1) / 2);

  std::pmr::memory_resource *mem = &workspace.resource;

  const LimbSpan a0 = chunk_span(ap, an, 0, k);
  const LimbSpan a1 = chunk_span(ap, an, k, k);
  const LimbSpan a2 = chunk_span(ap, an, 2 * k, an > 2 * k ? an - 2 * k : 0);
  const LimbSpan b0 = chunk_span(bp, bn, 0, k);
  const LimbSpan b1 = chunk_span(bp, bn, k, bn > k ? bn - k : 0);

  if (out_limbs != 0)
    std::memset(rp, 0, out_limbs * sizeof(rp[0]));

  const std::size_t eval_product_limbs = 2 * k + 4;
  LimbVec s1 = lincomb3_abs(a0, a1, 1, a2, 1, mem, eval_product_limbs);
  LimbVec s2 = add_abs(b0, b1, mem, eval_product_limbs);
  const LimbVec eval_b2 = lincomb2_abs(b0, b1, 2, mem);

  Ifma52Workspace &point_ws = workspace.point_ws;
  const LimbSpan c0 = mul_mag_ifma_to_output(rp, a0, b0, point_ws);
  const LimbSpan c3 = a2.empty() || b1.empty()
                       ? LimbSpan{}
                       : mul_mag_ifma_to_output(rp + 3 * k, a2, b1, point_ws);
  mul_mag_ifma_into(s1, s1, s2, point_ws);
  lincomb3_abs_into(s2, a0, a1, 2, a2, 4, eval_product_limbs);
  mul_mag_ifma_into(s2, s2, eval_b2, point_ws);

  sub_assign_abs(s1, c0);
  sub_assign_abs(s1, c3);
  sub_assign_abs(s2, c0);
  sub_mul_small_assign_abs(s2, c3, 8);
  divexact_abs_2_inplace(s2);
  sub_assign_abs(s2, s1);
  sub_assign_abs(s1, s2);

  add_shifted_to_output(rp, out_limbs, s1, k);
  add_shifted_to_output(rp, out_limbs, s2, 2 * k);
}

void
mul_toom32_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
               const std::uint64_t *bp, std::size_t bn, ToomWorkspace &workspace)
{
  const std::size_t norm_an = normalized_size(ap, an);
  const std::size_t norm_bn = normalized_size(bp, bn);
  if (norm_an < norm_bn) {
    mul_toom32_u64(rp, bp, bn, ap, an, workspace);
    return;
  }

  if (norm_an == 0 || norm_bn == 0) {
    workspace.reset();
    mul_basecase_u64(rp, ap, an, bp, bn, workspace.point_ws);
    return;
  }

  const std::size_t k = std::max((norm_an + 2) / 3, (norm_bn + 1) / 2);
  if (2 * k >= norm_an) {
    mul_toom22_u64(rp, ap, an, bp, bn, workspace);
    return;
  }
  if (k >= norm_bn || k < TOOM32_MIN_CHUNK) {
    workspace.reset();
    mul_basecase_u64(rp, ap, an, bp, bn, workspace.point_ws);
    return;
  }

  mul_toom32_raw_u64(rp, ap, an, bp, bn, workspace);
}

void
mul_toom33_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
               const std::uint64_t *bp, std::size_t bn)
{
  ToomWorkspace workspace;
  mul_toom33_u64(rp, ap, an, bp, bn, workspace);
}

void
mul_toom33_raw_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
                   const std::uint64_t *bp, std::size_t bn)
{
  ToomWorkspace workspace;
  mul_toom33_raw_u64(rp, ap, an, bp, bn, workspace);
}

void
mul_toom33_raw_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
                   const std::uint64_t *bp, std::size_t bn, ToomWorkspace &workspace)
{
  const std::size_t out_limbs = an + bn;
  workspace.reset();
  const std::size_t norm_an = normalized_size(ap, an);
  const std::size_t norm_bn = normalized_size(bp, bn);
  if (norm_an == 0 || norm_bn == 0) {
    if (out_limbs != 0)
      std::memset(rp, 0, out_limbs * sizeof(rp[0]));
    return;
  }
  an = norm_an;
  bn = norm_bn;

  std::pmr::memory_resource *mem = &workspace.resource;

  const std::size_t k = (std::max(an, bn) + 2) / 3;
  const LimbSpan a0 = chunk_span(ap, an, 0, k);
  const LimbSpan a1 = chunk_span(ap, an, k, k);
  const LimbSpan a2 = chunk_span(ap, an, 2 * k, an > 2 * k ? an - 2 * k : 0);
  const LimbSpan b0 = chunk_span(bp, bn, 0, k);
  const LimbSpan b1 = chunk_span(bp, bn, k, k);
  const LimbSpan b2 = chunk_span(bp, bn, 2 * k, bn > 2 * k ? bn - 2 * k : 0);

  if (out_limbs != 0)
    std::memset(rp, 0, out_limbs * sizeof(rp[0]));

  const std::size_t eval_product_limbs = 2 * k + 4;
  LimbVec s = lincomb3_abs(a0, a1, 1, a2, 1, mem, eval_product_limbs);
  LimbVec r = lincomb3_abs(b0, b1, 1, b2, 1, mem, eval_product_limbs);
  LimbVec vm1_scratch = lincomb3_abs(b0, b1, 2, b2, 4, mem, eval_product_limbs);
  const SignedMag eval_bm1 = eval_neg1_3(b0, b1, b2, mem);

  Ifma52Workspace &point_ws = workspace.point_ws;
  const LimbSpan c0 = mul_mag_ifma_to_output(rp, a0, b0, point_ws);
  const LimbSpan c4 = a2.empty() || b2.empty()
                       ? LimbSpan{}
                       : mul_mag_ifma_to_output(rp + 4 * k, a2, b2, point_ws);
  mul_mag_ifma_into(s, s, r, point_ws);
  lincomb3_abs_into(r, a0, a1, 2, a2, 4, eval_product_limbs);
  mul_mag_ifma_into(r, r, vm1_scratch, point_ws);
  SignedMag vm1 = eval_neg1_3_into(vm1_scratch, a0, a1, a2, eval_product_limbs);
  vm1.sign *= eval_bm1.sign;
  mul_mag_ifma_into(vm1.limbs, vm1.limbs, eval_bm1.limbs, point_ws);
  if (vm1.limbs.empty())
    vm1.sign = 0;

  sub_assign_abs(s, c0);
  sub_assign_abs(s, c4);
  sub_positive_from_signed_inplace(vm1, c0, mem);
  sub_positive_from_signed_inplace(vm1, c4, mem);

  sub_assign_abs(r, c0);
  sub_mul_small_assign_abs(r, c4, 16);
  divexact_abs_2_inplace(r);

  add_shifted_to_output(rp, out_limbs, s, k);
  sub_assign_abs(r, s);

  if (vm1.sign < 0)
    sub_assign_abs(s, vm1.limbs);
  else if (vm1.sign > 0)
    add_assign_abs(s, vm1.limbs);
  divexact_abs_2_inplace(s);

  sub_assign_abs(r, s);
  divexact_abs_3_inplace(r);

  sub_shifted_from_output(rp, out_limbs, s, k);
  sub_shifted_from_output(rp, out_limbs, r, k);
  add_shifted_to_output(rp, out_limbs, s, 2 * k);
  add_shifted_to_output(rp, out_limbs, r, 3 * k);
}

void
mul_toom33_u64(std::uint64_t *rp, const std::uint64_t *ap, std::size_t an,
               const std::uint64_t *bp, std::size_t bn, ToomWorkspace &workspace)
{
  const std::size_t norm_an = normalized_size(ap, an);
  const std::size_t norm_bn = normalized_size(bp, bn);
  if (std::min(norm_an, norm_bn) < TOOM33_MIN_LIMBS) {
    workspace.reset();
    mul_basecase_u64(rp, ap, an, bp, bn, workspace.point_ws);
    return;
  }

  mul_toom33_raw_u64(rp, ap, an, bp, bn, workspace);
}

} // namespace simd_bigint::reference
