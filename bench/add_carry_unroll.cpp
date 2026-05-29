#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>

#if defined(__GNUC__) || defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

static constexpr size_t MAX_LIMBS = 256;

static inline uint64_t rdtsc() {
  unsigned aux;
  _mm_lfence();
  const uint64_t t = __rdtscp(&aux);
  _mm_lfence();
  return t;
}

static inline __m512i load512(const uint64_t *p) {
  return _mm512_load_si512((const __m512i *)p);
}

static inline void store512(uint64_t *p, __m512i x) {
  _mm512_store_si512((__m512i *)p, x);
}

static inline __m512i add_carry_fix(__m512i r, __mmask8 fix) {
  const __m512i max = _mm512_set1_epi64(-1LL);
  return _mm512_mask_sub_epi64(r, fix, r, max);
}

static inline __m512i addcarry_loop1_step(__m512i a, __m512i b,
                                          __mmask64 *carry) {
  const __m512i max = _mm512_set1_epi64(-1LL);
  __m512i r = _mm512_add_epi64(a, b);

  __mmask64 c = *carry;
  c = _kor_mask64(
      c, _kshiftli_mask64((__mmask64)_mm512_cmplt_epu64_mask(r, a), 1));
  __mmask64 m = (__mmask64)_mm512_cmpeq_epu64_mask(r, max);

  c = _kadd_mask64(c, m);
  m = _kxor_mask64(m, c);
  r = add_carry_fix(r, (__mmask8)m);
  *carry = _kshiftri_mask64(c, 8);

  return r;
}

static inline __m512i addcarry_loop1_gpr_step(__m512i a, __m512i b,
                                              uint64_t *carry) {
  const __m512i max = _mm512_set1_epi64(-1LL);
  __m512i r = _mm512_add_epi64(a, b);

  uint64_t c = *carry;
  c |= (uint64_t)_mm512_cmplt_epu64_mask(r, a) << 1;
  uint64_t m = (uint64_t)_mm512_cmpeq_epu64_mask(r, max);

  c += m;
  m ^= c;
  r = add_carry_fix(r, (__mmask8)m);
  *carry = c >> 8;

  return r;
}

template <size_t LIMBS>
NOINLINE uint64_t add_l1_base(uint64_t *rp, const uint64_t *ap,
                              const uint64_t *bp, uint64_t reps) {
  static_assert(LIMBS <= MAX_LIMBS && LIMBS % 32 == 0);
  constexpr size_t VECS = LIMBS / 8;
  uint64_t acc = 0;

  for (uint64_t rep = 0; rep < reps; rep++) {
    for (size_t j = 0; j < VECS; j++) {
      const __m512i a = load512(ap + 8 * j);
      const __m512i b = load512(bp + 8 * j);
      store512(rp + 8 * j, _mm512_add_epi64(a, b));
    }

    asm volatile("" ::: "memory");
    acc += rp[rep & (LIMBS - 1)];
  }

  return acc;
}

template <size_t LIMBS>
NOINLINE uint64_t add_l1_unroll4(uint64_t *rp, const uint64_t *ap,
                                 const uint64_t *bp, uint64_t reps) {
  static_assert(LIMBS <= MAX_LIMBS && LIMBS % 32 == 0);
  constexpr size_t VECS = LIMBS / 8;
  const __m512i max = _mm512_set1_epi64(-1LL);
  uint64_t acc = 0;

  for (uint64_t rep = 0; rep < reps; rep++) {
    __mmask64 carry = 0;

    for (size_t j = 0; j < VECS; j += 4) {
      const __m512i a0 = load512(ap + 8 * (j + 0));
      const __m512i a1 = load512(ap + 8 * (j + 1));
      const __m512i a2 = load512(ap + 8 * (j + 2));
      const __m512i a3 = load512(ap + 8 * (j + 3));
      const __m512i b0 = load512(bp + 8 * (j + 0));
      const __m512i b1 = load512(bp + 8 * (j + 1));
      const __m512i b2 = load512(bp + 8 * (j + 2));
      const __m512i b3 = load512(bp + 8 * (j + 3));

      __m512i r0 = _mm512_add_epi64(a0, b0);
      __m512i r1 = _mm512_add_epi64(a1, b1);
      __m512i r2 = _mm512_add_epi64(a2, b2);
      __m512i r3 = _mm512_add_epi64(a3, b3);

      __mmask64 c = carry;
      c = _kor_mask64(
          c, _kshiftli_mask64((__mmask64)_mm512_cmplt_epu64_mask(r0, a0), 1));
      c = _kor_mask64(
          c, _kshiftli_mask64((__mmask64)_mm512_cmplt_epu64_mask(r1, a1), 9));
      c = _kor_mask64(
          c, _kshiftli_mask64((__mmask64)_mm512_cmplt_epu64_mask(r2, a2), 17));
      c = _kor_mask64(
          c, _kshiftli_mask64((__mmask64)_mm512_cmplt_epu64_mask(r3, a3), 25));

      __mmask64 m = (__mmask64)_mm512_cmpeq_epu64_mask(r0, max);
      m = _kor_mask64(
          m, _kshiftli_mask64((__mmask64)_mm512_cmpeq_epu64_mask(r1, max), 8));
      m = _kor_mask64(
          m, _kshiftli_mask64((__mmask64)_mm512_cmpeq_epu64_mask(r2, max), 16));
      m = _kor_mask64(
          m, _kshiftli_mask64((__mmask64)_mm512_cmpeq_epu64_mask(r3, max), 24));

      c = _kadd_mask64(c, m);
      m = _kxor_mask64(m, c);

      r0 = add_carry_fix(r0, (__mmask8)m);
      r1 = add_carry_fix(r1, (__mmask8)_kshiftri_mask64(m, 8));
      r2 = add_carry_fix(r2, (__mmask8)_kshiftri_mask64(m, 16));
      r3 = add_carry_fix(r3, (__mmask8)_kshiftri_mask64(m, 24));

      carry = _kshiftri_mask64(c, 32);

      store512(rp + 8 * (j + 0), r0);
      store512(rp + 8 * (j + 1), r1);
      store512(rp + 8 * (j + 2), r2);
      store512(rp + 8 * (j + 3), r3);
    }

    asm volatile("" ::: "memory");
    acc += rp[rep & (LIMBS - 1)] + (uint64_t)carry;
  }

  return acc;
}

template <size_t LIMBS>
NOINLINE uint64_t add_l1_loop1x4(uint64_t *rp, const uint64_t *ap,
                                 const uint64_t *bp, uint64_t reps) {
  static_assert(LIMBS <= MAX_LIMBS && LIMBS % 32 == 0);
  constexpr size_t VECS = LIMBS / 8;
  uint64_t acc = 0;

  for (uint64_t rep = 0; rep < reps; rep++) {
    __mmask64 carry = 0;

    for (size_t j = 0; j < VECS; j += 4) {
      const __m512i a0 = load512(ap + 8 * (j + 0));
      const __m512i a1 = load512(ap + 8 * (j + 1));
      const __m512i a2 = load512(ap + 8 * (j + 2));
      const __m512i a3 = load512(ap + 8 * (j + 3));
      const __m512i b0 = load512(bp + 8 * (j + 0));
      const __m512i b1 = load512(bp + 8 * (j + 1));
      const __m512i b2 = load512(bp + 8 * (j + 2));
      const __m512i b3 = load512(bp + 8 * (j + 3));

      const __m512i r0 = addcarry_loop1_step(a0, b0, &carry);
      const __m512i r1 = addcarry_loop1_step(a1, b1, &carry);
      const __m512i r2 = addcarry_loop1_step(a2, b2, &carry);
      const __m512i r3 = addcarry_loop1_step(a3, b3, &carry);

      store512(rp + 8 * (j + 0), r0);
      store512(rp + 8 * (j + 1), r1);
      store512(rp + 8 * (j + 2), r2);
      store512(rp + 8 * (j + 3), r3);
    }

    asm volatile("" ::: "memory");
    acc += rp[rep & (LIMBS - 1)] + (uint64_t)carry;
  }

  return acc;
}

template <size_t LIMBS>
NOINLINE uint64_t add_l1_unroll4_gpr_or(uint64_t *rp, const uint64_t *ap,
                                        const uint64_t *bp, uint64_t reps) {
  static_assert(LIMBS <= MAX_LIMBS && LIMBS % 32 == 0);
  constexpr size_t VECS = LIMBS / 8;
  const __m512i max = _mm512_set1_epi64(-1LL);
  uint64_t acc = 0;

  for (uint64_t rep = 0; rep < reps; rep++) {
    uint64_t carry = 0;

    for (size_t j = 0; j < VECS; j += 4) {
      const __m512i a0 = load512(ap + 8 * (j + 0));
      const __m512i a1 = load512(ap + 8 * (j + 1));
      const __m512i a2 = load512(ap + 8 * (j + 2));
      const __m512i a3 = load512(ap + 8 * (j + 3));
      const __m512i b0 = load512(bp + 8 * (j + 0));
      const __m512i b1 = load512(bp + 8 * (j + 1));
      const __m512i b2 = load512(bp + 8 * (j + 2));
      const __m512i b3 = load512(bp + 8 * (j + 3));

      __m512i r0 = _mm512_add_epi64(a0, b0);
      __m512i r1 = _mm512_add_epi64(a1, b1);
      __m512i r2 = _mm512_add_epi64(a2, b2);
      __m512i r3 = _mm512_add_epi64(a3, b3);

      uint64_t c = carry;
      c |= (uint64_t)_mm512_cmplt_epu64_mask(r0, a0) << 1;
      c |= (uint64_t)_mm512_cmplt_epu64_mask(r1, a1) << 9;
      c |= (uint64_t)_mm512_cmplt_epu64_mask(r2, a2) << 17;
      c |= (uint64_t)_mm512_cmplt_epu64_mask(r3, a3) << 25;

      uint64_t m = (uint64_t)_mm512_cmpeq_epu64_mask(r0, max);
      m |= (uint64_t)_mm512_cmpeq_epu64_mask(r1, max) << 8;
      m |= (uint64_t)_mm512_cmpeq_epu64_mask(r2, max) << 16;
      m |= (uint64_t)_mm512_cmpeq_epu64_mask(r3, max) << 24;

      c += m;
      m ^= c;

      r0 = add_carry_fix(r0, (__mmask8)m);
      r1 = add_carry_fix(r1, (__mmask8)(m >> 8));
      r2 = add_carry_fix(r2, (__mmask8)(m >> 16));
      r3 = add_carry_fix(r3, (__mmask8)(m >> 24));

      carry = c >> 32;

      store512(rp + 8 * (j + 0), r0);
      store512(rp + 8 * (j + 1), r1);
      store512(rp + 8 * (j + 2), r2);
      store512(rp + 8 * (j + 3), r3);
    }

    asm volatile("" ::: "memory");
    acc += rp[rep & (LIMBS - 1)] + carry;
  }

  return acc;
}

template <size_t LIMBS>
NOINLINE uint64_t add_l1_unroll4_gpr_lea(uint64_t *rp, const uint64_t *ap,
                                         const uint64_t *bp, uint64_t reps) {
  static_assert(LIMBS <= MAX_LIMBS && LIMBS % 32 == 0);
  constexpr size_t VECS = LIMBS / 8;
  const __m512i max = _mm512_set1_epi64(-1LL);
  uint64_t acc = 0;

  for (uint64_t rep = 0; rep < reps; rep++) {
    uint64_t carry = 0;

    for (size_t j = 0; j < VECS; j += 4) {
      const __m512i a0 = load512(ap + 8 * (j + 0));
      const __m512i a1 = load512(ap + 8 * (j + 1));
      const __m512i a2 = load512(ap + 8 * (j + 2));
      const __m512i a3 = load512(ap + 8 * (j + 3));
      const __m512i b0 = load512(bp + 8 * (j + 0));
      const __m512i b1 = load512(bp + 8 * (j + 1));
      const __m512i b2 = load512(bp + 8 * (j + 2));
      const __m512i b3 = load512(bp + 8 * (j + 3));

      __m512i r0 = _mm512_add_epi64(a0, b0);
      __m512i r1 = _mm512_add_epi64(a1, b1);
      __m512i r2 = _mm512_add_epi64(a2, b2);
      __m512i r3 = _mm512_add_epi64(a3, b3);

      uint64_t c = (uint64_t)_mm512_cmplt_epu64_mask(r0, a0);
      c |= (uint64_t)_mm512_cmplt_epu64_mask(r1, a1) << 8;
      c |= (uint64_t)_mm512_cmplt_epu64_mask(r2, a2) << 16;
      c |= (uint64_t)_mm512_cmplt_epu64_mask(r3, a3) << 24;

      c = carry + c * 2;

      uint64_t m = (uint64_t)_mm512_cmpeq_epu64_mask(r0, max);
      m |= (uint64_t)_mm512_cmpeq_epu64_mask(r1, max) << 8;
      m |= (uint64_t)_mm512_cmpeq_epu64_mask(r2, max) << 16;
      m |= (uint64_t)_mm512_cmpeq_epu64_mask(r3, max) << 24;

      c += m;
      m ^= c;

      r0 = add_carry_fix(r0, (__mmask8)m);
      r1 = add_carry_fix(r1, (__mmask8)(m >> 8));
      r2 = add_carry_fix(r2, (__mmask8)(m >> 16));
      r3 = add_carry_fix(r3, (__mmask8)(m >> 24));

      carry = c >> 32;

      store512(rp + 8 * (j + 0), r0);
      store512(rp + 8 * (j + 1), r1);
      store512(rp + 8 * (j + 2), r2);
      store512(rp + 8 * (j + 3), r3);
    }

    asm volatile("" ::: "memory");
    acc += rp[rep & (LIMBS - 1)] + carry;
  }

  return acc;
}

template <size_t LIMBS>
NOINLINE uint64_t add_l1_unroll4_gpr_byte(uint64_t *rp, const uint64_t *ap,
                                          const uint64_t *bp, uint64_t reps) {
  static_assert(LIMBS <= MAX_LIMBS && LIMBS % 32 == 0);
  constexpr size_t VECS = LIMBS / 8;
  const __m512i max = _mm512_set1_epi64(-1LL);
  uint64_t acc = 0;

  for (uint64_t rep = 0; rep < reps; rep++) {
    uint64_t carry = 0;

    for (size_t j = 0; j < VECS; j += 4) {
      const __m512i a0 = load512(ap + 8 * (j + 0));
      const __m512i a1 = load512(ap + 8 * (j + 1));
      const __m512i a2 = load512(ap + 8 * (j + 2));
      const __m512i a3 = load512(ap + 8 * (j + 3));
      const __m512i b0 = load512(bp + 8 * (j + 0));
      const __m512i b1 = load512(bp + 8 * (j + 1));
      const __m512i b2 = load512(bp + 8 * (j + 2));
      const __m512i b3 = load512(bp + 8 * (j + 3));

      __m512i r0 = _mm512_add_epi64(a0, b0);
      __m512i r1 = _mm512_add_epi64(a1, b1);
      __m512i r2 = _mm512_add_epi64(a2, b2);
      __m512i r3 = _mm512_add_epi64(a3, b3);

      const uint64_t c0 = (uint8_t)_mm512_cmplt_epu64_mask(r0, a0);
      const uint64_t c1 = (uint8_t)_mm512_cmplt_epu64_mask(r1, a1);
      const uint64_t c2 = (uint8_t)_mm512_cmplt_epu64_mask(r2, a2);
      const uint64_t c3 = (uint8_t)_mm512_cmplt_epu64_mask(r3, a3);
      uint64_t c = c0 | (c1 << 8) | (c2 << 16) | (c3 << 24);

      c = carry + (c << 1);

      uint64_t m = (uint64_t)_mm512_cmpeq_epu64_mask(r0, max);
      m |= (uint64_t)_mm512_cmpeq_epu64_mask(r1, max) << 8;
      m |= (uint64_t)_mm512_cmpeq_epu64_mask(r2, max) << 16;
      m |= (uint64_t)_mm512_cmpeq_epu64_mask(r3, max) << 24;

      c += m;
      m ^= c;

      r0 = add_carry_fix(r0, (__mmask8)m);
      r1 = add_carry_fix(r1, (__mmask8)(m >> 8));
      r2 = add_carry_fix(r2, (__mmask8)(m >> 16));
      r3 = add_carry_fix(r3, (__mmask8)(m >> 24));

      carry = c >> 32;

      store512(rp + 8 * (j + 0), r0);
      store512(rp + 8 * (j + 1), r1);
      store512(rp + 8 * (j + 2), r2);
      store512(rp + 8 * (j + 3), r3);
    }

    asm volatile("" ::: "memory");
    acc += rp[rep & (LIMBS - 1)] + carry;
  }

  return acc;
}

template <size_t LIMBS>
NOINLINE uint64_t add_l1_loop1x4_gpr(uint64_t *rp, const uint64_t *ap,
                                     const uint64_t *bp, uint64_t reps) {
  static_assert(LIMBS <= MAX_LIMBS && LIMBS % 32 == 0);
  constexpr size_t VECS = LIMBS / 8;
  uint64_t acc = 0;

  for (uint64_t rep = 0; rep < reps; rep++) {
    uint64_t carry = 0;

    for (size_t j = 0; j < VECS; j += 4) {
      const __m512i a0 = load512(ap + 8 * (j + 0));
      const __m512i a1 = load512(ap + 8 * (j + 1));
      const __m512i a2 = load512(ap + 8 * (j + 2));
      const __m512i a3 = load512(ap + 8 * (j + 3));
      const __m512i b0 = load512(bp + 8 * (j + 0));
      const __m512i b1 = load512(bp + 8 * (j + 1));
      const __m512i b2 = load512(bp + 8 * (j + 2));
      const __m512i b3 = load512(bp + 8 * (j + 3));

      const __m512i r0 = addcarry_loop1_gpr_step(a0, b0, &carry);
      const __m512i r1 = addcarry_loop1_gpr_step(a1, b1, &carry);
      const __m512i r2 = addcarry_loop1_gpr_step(a2, b2, &carry);
      const __m512i r3 = addcarry_loop1_gpr_step(a3, b3, &carry);

      store512(rp + 8 * (j + 0), r0);
      store512(rp + 8 * (j + 1), r1);
      store512(rp + 8 * (j + 2), r2);
      store512(rp + 8 * (j + 3), r3);
    }

    asm volatile("" ::: "memory");
    acc += rp[rep & (LIMBS - 1)] + carry;
  }

  return acc;
}

template <size_t LIMBS>
NOINLINE uint64_t add_l1_loop1x4_gpr_store_early(uint64_t *rp,
                                                 const uint64_t *ap,
                                                 const uint64_t *bp,
                                                 uint64_t reps) {
  static_assert(LIMBS <= MAX_LIMBS && LIMBS % 32 == 0);
  constexpr size_t VECS = LIMBS / 8;
  uint64_t acc = 0;

  for (uint64_t rep = 0; rep < reps; rep++) {
    uint64_t carry = 0;

    for (size_t j = 0; j < VECS; j += 4) {
      const __m512i a0 = load512(ap + 8 * (j + 0));
      const __m512i a1 = load512(ap + 8 * (j + 1));
      const __m512i a2 = load512(ap + 8 * (j + 2));
      const __m512i a3 = load512(ap + 8 * (j + 3));
      const __m512i b0 = load512(bp + 8 * (j + 0));
      const __m512i b1 = load512(bp + 8 * (j + 1));
      const __m512i b2 = load512(bp + 8 * (j + 2));
      const __m512i b3 = load512(bp + 8 * (j + 3));

      const __m512i r0 = addcarry_loop1_gpr_step(a0, b0, &carry);
      store512(rp + 8 * (j + 0), r0);

      const __m512i r1 = addcarry_loop1_gpr_step(a1, b1, &carry);
      store512(rp + 8 * (j + 1), r1);

      const __m512i r2 = addcarry_loop1_gpr_step(a2, b2, &carry);
      store512(rp + 8 * (j + 2), r2);

      const __m512i r3 = addcarry_loop1_gpr_step(a3, b3, &carry);
      store512(rp + 8 * (j + 3), r3);
    }

    asm volatile("" ::: "memory");
    acc += rp[rep & (LIMBS - 1)] + carry;
  }

  return acc;
}

template <size_t LIMBS>
NOINLINE uint64_t add_l1_loop1x4_gpr_stream(uint64_t *rp, const uint64_t *ap,
                                            const uint64_t *bp,
                                            uint64_t reps) {
  static_assert(LIMBS <= MAX_LIMBS && LIMBS % 32 == 0);
  constexpr size_t VECS = LIMBS / 8;
  uint64_t acc = 0;

  for (uint64_t rep = 0; rep < reps; rep++) {
    uint64_t carry = 0;

    for (size_t j = 0; j < VECS; j += 4) {
      const __m512i a0 = load512(ap + 8 * (j + 0));
      const __m512i b0 = load512(bp + 8 * (j + 0));
      const __m512i r0 = addcarry_loop1_gpr_step(a0, b0, &carry);
      store512(rp + 8 * (j + 0), r0);

      const __m512i a1 = load512(ap + 8 * (j + 1));
      const __m512i b1 = load512(bp + 8 * (j + 1));
      const __m512i r1 = addcarry_loop1_gpr_step(a1, b1, &carry);
      store512(rp + 8 * (j + 1), r1);

      const __m512i a2 = load512(ap + 8 * (j + 2));
      const __m512i b2 = load512(bp + 8 * (j + 2));
      const __m512i r2 = addcarry_loop1_gpr_step(a2, b2, &carry);
      store512(rp + 8 * (j + 2), r2);

      const __m512i a3 = load512(ap + 8 * (j + 3));
      const __m512i b3 = load512(bp + 8 * (j + 3));
      const __m512i r3 = addcarry_loop1_gpr_step(a3, b3, &carry);
      store512(rp + 8 * (j + 3), r3);
    }

    asm volatile("" ::: "memory");
    acc += rp[rep & (LIMBS - 1)] + carry;
  }

  return acc;
}

template <size_t LIMBS>
NOINLINE uint64_t add_l1_loop1_gpr(uint64_t *rp, const uint64_t *ap,
                                   const uint64_t *bp, uint64_t reps) {
  static_assert(LIMBS <= MAX_LIMBS && LIMBS % 32 == 0);
  constexpr size_t VECS = LIMBS / 8;
  uint64_t acc = 0;

  for (uint64_t rep = 0; rep < reps; rep++) {
    uint64_t carry = 0;

#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for (size_t j = 0; j < VECS; j++) {
      const __m512i a = load512(ap + 8 * j);
      const __m512i b = load512(bp + 8 * j);
      const __m512i r = addcarry_loop1_gpr_step(a, b, &carry);
      store512(rp + 8 * j, r);
    }

    asm volatile("" ::: "memory");
    acc += rp[rep & (LIMBS - 1)] + carry;
  }

  return acc;
}

static uint64_t splitmix64(uint64_t *x) {
  uint64_t z = (*x += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}

static void fill_operands(uint64_t *ap, uint64_t *bp) {
  uint64_t seed = 0x123456789abcdef0ull;

  for (size_t i = 0; i < MAX_LIMBS; i++) {
    ap[i] = splitmix64(&seed);
    bp[i] = splitmix64(&seed);

    switch (i & 31) {
    case 0:
      ap[i] = UINT64_MAX - 3;
      bp[i] = 19;
      break;
    case 1:
    case 2:
    case 3:
      ap[i] = UINT64_MAX;
      bp[i] = 0;
      break;
    case 4:
      ap[i] = 0x1234;
      bp[i] = 0x5678;
      break;
    case 17:
      ap[i] = UINT64_MAX;
      bp[i] = UINT64_MAX;
      break;
    case 18:
      ap[i] = UINT64_MAX;
      bp[i] = 0;
      break;
    default:
      break;
    }
  }
}

template <size_t LIMBS>
static uint64_t add_ref(uint64_t *rp, const uint64_t *ap, const uint64_t *bp) {
  unsigned __int128 carry = 0;
  for (size_t i = 0; i < LIMBS; i++) {
    const unsigned __int128 s = (unsigned __int128)ap[i] + bp[i] + carry;
    rp[i] = (uint64_t)s;
    carry = s >> 64;
  }
  return (uint64_t)carry;
}

template <size_t LIMBS> static uint64_t hash_limbs(const uint64_t *p) {
  uint64_t h = 0x9e3779b97f4a7c15ull;
  for (size_t i = 0; i < LIMBS; i++)
    h = (h ^ p[i]) * 0xbf58476d1ce4e5b9ull;
  return h;
}

template <size_t LIMBS>
static int verify(uint64_t *tmp, uint64_t *ref, const uint64_t *ap,
                  const uint64_t *bp) {
  const uint64_t ref_carry = add_ref<LIMBS>(ref, ap, bp);

  memset(tmp, 0, LIMBS * sizeof(tmp[0]));
  const uint64_t unroll_ret = add_l1_unroll4<LIMBS>(tmp, ap, bp, 1);
  const uint64_t unroll_carry = unroll_ret - tmp[0];
  if (memcmp(tmp, ref, LIMBS * sizeof(tmp[0])) != 0 ||
      unroll_carry != ref_carry)
    return 0;

  memset(tmp, 0, LIMBS * sizeof(tmp[0]));
  const uint64_t loop_ret = add_l1_loop1x4<LIMBS>(tmp, ap, bp, 1);
  const uint64_t loop_carry = loop_ret - tmp[0];
  if (memcmp(tmp, ref, LIMBS * sizeof(tmp[0])) != 0 || loop_carry != ref_carry)
    return 0;

  memset(tmp, 0, LIMBS * sizeof(tmp[0]));
  const uint64_t unroll_or_ret = add_l1_unroll4_gpr_or<LIMBS>(tmp, ap, bp, 1);
  const uint64_t unroll_or_carry = unroll_or_ret - tmp[0];
  if (memcmp(tmp, ref, LIMBS * sizeof(tmp[0])) != 0 ||
      unroll_or_carry != ref_carry)
    return 0;

  memset(tmp, 0, LIMBS * sizeof(tmp[0]));
  const uint64_t unroll_gpr_ret =
      add_l1_unroll4_gpr_lea<LIMBS>(tmp, ap, bp, 1);
  const uint64_t unroll_gpr_carry = unroll_gpr_ret - tmp[0];
  if (memcmp(tmp, ref, LIMBS * sizeof(tmp[0])) != 0 ||
      unroll_gpr_carry != ref_carry)
    return 0;

  memset(tmp, 0, LIMBS * sizeof(tmp[0]));
  const uint64_t unroll_byte_ret =
      add_l1_unroll4_gpr_byte<LIMBS>(tmp, ap, bp, 1);
  const uint64_t unroll_byte_carry = unroll_byte_ret - tmp[0];
  if (memcmp(tmp, ref, LIMBS * sizeof(tmp[0])) != 0 ||
      unroll_byte_carry != ref_carry)
    return 0;

  memset(tmp, 0, LIMBS * sizeof(tmp[0]));
  const uint64_t loop_gpr_ret = add_l1_loop1x4_gpr<LIMBS>(tmp, ap, bp, 1);
  const uint64_t loop_gpr_carry = loop_gpr_ret - tmp[0];
  if (memcmp(tmp, ref, LIMBS * sizeof(tmp[0])) != 0 ||
      loop_gpr_carry != ref_carry)
    return 0;

  memset(tmp, 0, LIMBS * sizeof(tmp[0]));
  const uint64_t early_gpr_ret =
      add_l1_loop1x4_gpr_store_early<LIMBS>(tmp, ap, bp, 1);
  const uint64_t early_gpr_carry = early_gpr_ret - tmp[0];
  if (memcmp(tmp, ref, LIMBS * sizeof(tmp[0])) != 0 ||
      early_gpr_carry != ref_carry)
    return 0;

  memset(tmp, 0, LIMBS * sizeof(tmp[0]));
  const uint64_t stream_gpr_ret =
      add_l1_loop1x4_gpr_stream<LIMBS>(tmp, ap, bp, 1);
  const uint64_t stream_gpr_carry = stream_gpr_ret - tmp[0];
  if (memcmp(tmp, ref, LIMBS * sizeof(tmp[0])) != 0 ||
      stream_gpr_carry != ref_carry)
    return 0;

  memset(tmp, 0, LIMBS * sizeof(tmp[0]));
  const uint64_t loop1_gpr_ret = add_l1_loop1_gpr<LIMBS>(tmp, ap, bp, 1);
  const uint64_t loop1_gpr_carry = loop1_gpr_ret - tmp[0];
  if (memcmp(tmp, ref, LIMBS * sizeof(tmp[0])) != 0 ||
      loop1_gpr_carry != ref_carry)
    return 0;

  return 1;
}

template <size_t LIMBS>
using bench_fn = uint64_t (*)(uint64_t *, const uint64_t *, const uint64_t *,
                              uint64_t);

template <size_t LIMBS>
static uint64_t time_one(bench_fn<LIMBS> fn, uint64_t *rp, const uint64_t *ap,
                         const uint64_t *bp, uint64_t reps, uint64_t *hash) {
  memset(rp, 0, LIMBS * sizeof(rp[0]));
  const uint64_t t0 = rdtsc();
  const uint64_t acc = fn(rp, ap, bp, reps);
  const uint64_t t1 = rdtsc();
  *hash ^= hash_limbs<LIMBS>(rp) ^ acc;
  return t1 - t0;
}

template <size_t LIMBS>
static void bench_case(uint64_t reps, unsigned rounds, uint64_t *rp,
                       uint64_t *ref, const uint64_t *ap, const uint64_t *bp) {
  if (!verify<LIMBS>(rp, ref, ap, bp)) {
    fprintf(stderr, "carry benchmark verification failed at %zu limbs\n",
            LIMBS);
    exit(1);
  }

  uint64_t best_base = UINT64_MAX;
  struct Candidate {
    const char *name;
    bench_fn<LIMBS> fn;
    uint64_t best;
    uint64_t hash;
  };

  Candidate candidates[] = {
      {"base", add_l1_base<LIMBS>, UINT64_MAX, 0},
      {"k-unroll4", add_l1_unroll4<LIMBS>, UINT64_MAX, 0},
      {"k-loop1x4", add_l1_loop1x4<LIMBS>, UINT64_MAX, 0},
      {"gpr-unroll4-or", add_l1_unroll4_gpr_or<LIMBS>, UINT64_MAX, 0},
      {"gpr-unroll4-lea", add_l1_unroll4_gpr_lea<LIMBS>, UINT64_MAX, 0},
      {"gpr-unroll4-byte", add_l1_unroll4_gpr_byte<LIMBS>, UINT64_MAX, 0},
      {"gpr-loop1x4", add_l1_loop1x4_gpr<LIMBS>, UINT64_MAX, 0},
      {"gpr-loop1x4-early", add_l1_loop1x4_gpr_store_early<LIMBS>,
       UINT64_MAX, 0},
      {"gpr-loop1x4-stream", add_l1_loop1x4_gpr_stream<LIMBS>, UINT64_MAX, 0},
      {"gpr-loop1", add_l1_loop1_gpr<LIMBS>, UINT64_MAX, 0},
  };
  constexpr size_t N = sizeof(candidates) / sizeof(candidates[0]);

  for (unsigned r = 0; r < rounds; r++) {
    for (size_t i = 0; i < N; i++) {
      Candidate &c = candidates[(i + r) % N];
      const uint64_t t = time_one<LIMBS>(c.fn, rp, ap, bp, reps + r, &c.hash);
      if (t < c.best)
        c.best = t;
    }
  }

  for (size_t i = 0; i < N; i++)
    if (candidates[i].name[0] == 'b')
      best_base = candidates[i].best;

  const double base = (double)best_base / (double)reps;

  printf("%4zu limbs: base %8.3f cyc\n", LIMBS, base);
  size_t best = 1;
  for (size_t i = 1; i < N; i++) {
    const double cyc = (double)candidates[i].best / (double)reps;
    printf("            %-16s %8.3f cyc  %.4f cyc/limb  delta %8.3f\n",
           candidates[i].name, cyc, cyc / LIMBS, cyc - base);
    if (candidates[i].best < candidates[best].best)
      best = i;
  }
  printf("            best carry path: %s, hashes", candidates[best].name);
  for (size_t i = 0; i < N; i++)
    printf(" %016llx", (unsigned long long)candidates[i].hash);
  printf("\n");
}

int main(int argc, char **argv) {
  const uint64_t reps = argc > 1 ? strtoull(argv[1], nullptr, 0) : 2000000ull;
  const unsigned rounds =
      argc > 2 ? (unsigned)strtoul(argv[2], nullptr, 0) : 12;

  alignas(64) uint64_t a[MAX_LIMBS];
  alignas(64) uint64_t b[MAX_LIMBS];
  alignas(64) uint64_t r[MAX_LIMBS];
  alignas(64) uint64_t ref[MAX_LIMBS];

  fill_operands(a, b);

  printf(
      "reps=%llu rounds=%u, hot-L1 operands, max=%zu limbs (%zu bytes/array)\n",
      (unsigned long long)reps, rounds, MAX_LIMBS,
      MAX_LIMBS * sizeof(uint64_t));
  printf("columns compare k-register carry arithmetic against GPR carry "
         "arithmetic\n");

  bench_case<32>(reps, rounds, r, ref, a, b);
  bench_case<64>(reps, rounds, r, ref, a, b);
  bench_case<128>(reps, rounds, r, ref, a, b);
  bench_case<256>(reps, rounds, r, ref, a, b);

  return 0;
}
