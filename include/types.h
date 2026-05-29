#pragma once

#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef uint64_t _limb;
typedef uint32_t _hlimb;
typedef uint64_t *plimb;
typedef uint32_t *phlimb;

#define EVAL(x) x

#define CAT_(x, y) x##y
#define CAT(x, y) CAT_(x, y)

#define CAT3_(x, y, z) x##y##z
#define CAT3(x, y, z) CAT3_(x, y, z)

#define NARGS_(_1, _2, _3, _4, _5, _6, N, ...) N
#define NARGS(...) NARGS_(__VA_ARGS__, 6, 5, 4, 3, 2, 1)

#define vec_bits 512
#define vec_base CAT(__m, vec_bits)
#define vec_int_token CAT(si, vec_bits)
#define vec_fn_typ CAT3(_mm, vec_bits, _)
#define vec_fn(x) CAT(vec_fn_typ, x)

#define vec_itype CAT(vec_base, i)
#define vec_dtype CAT(vec_base, d)

typedef vec_itype _vec;
typedef vec_itype *pvec;
typedef vec_dtype _dvec;
typedef vec_dtype *pdvec;

#define vec_castpd_i CAT(castpd_, vec_int_token)
#define vec_casti_pd CAT3(cast, vec_int_token, _pd)

#define as_ivec(x) vec_fn(vec_castpd_i)(x)
#define as_dvec(x) vec_fn(vec_casti_pd)(x)

#define vec_loadu_i CAT(loadu_, vec_int_token)
#define vec_storeu_i CAT(storeu_, vec_int_token)

#define load_vec(...) CAT(load_vec_, NARGS(__VA_ARGS__))(__VA_ARGS__)
#define load_vec_1(p) vec_fn(vec_loadu_i)((const void *)(p))
#define load_vec_2(p, mask)                                                    \
  vec_fn(maskz_loadu_epi64)((__mmask8)(mask), (const void *)(p))

#define store_vec(...) CAT(store_vec_, NARGS(__VA_ARGS__))(__VA_ARGS__)
#define store_vec_2(p, v) vec_fn(vec_storeu_i)((void *)(p), (v))
#define store_vec_3(p, v, mask)                                                \
  vec_fn(mask_storeu_epi64)((void *)(p), (__mmask8)(mask), (v))

// Automatically choose raw op, masked op blended with zero vec, or masked op
// with a src vector, based on paramater list length
#define mask_fn(len, x) CAT(mask_fn_, len)(x)
#define mask_fn_1(x) vec_fn(x)
#define mask_fn_2(x) vec_fn(CAT(maskz_, x))
#define mask_fn_3(x) vec_fn(CAT(mask_, x))
#define mask_call(...) CAT(mask_call_, NARGS(__VA_ARGS__))(__VA_ARGS__)
#define mask_call_2(a, b) a, b
#define mask_call_3(a, b, mask) mask, a, b
#define mask_call_4(a, b, mask, src) src, mask, a, b

#define call_mask_fn(x, a, b, ...)                                             \
  mask_fn(NARGS(0, ##__VA_ARGS__), x)(mask_call(a, b, ##__VA_ARGS__))
#define generic_mask_fn(ivec_fn, dvec_fn, a, b, ...)                           \
  _Generic((a),                                                                \
      _vec: mask_fn(NARGS(0, ##__VA_ARGS__), ivec_fn),                         \
      _dvec: mask_fn(NARGS(0, ##__VA_ARGS__), dvec_fn))(                       \
      mask_call(a, b, ##__VA_ARGS__))

#define add(a, b, ...)                                                         \
  generic_mask_fn(add_epi64, add_pd, (a), (b), ##__VA_ARGS__)
#define sub(a, b, ...)                                                         \
  generic_mask_fn(sub_epi64, sub_pd, (a), (b), ##__VA_ARGS__)
#define mul(a, b, ...)                                                         \
  generic_mask_fn(mul_epi32, mul_pd, (a), (b), ##__VA_ARGS__)
#define andnot(a, b, ...) call_mask_fn(andnot_epi64, (a), (b), ##__VA_ARGS__)
#define and_v(a, b, ...) call_mask_fn(and_epi64, (a), (b), ##__VA_ARGS__)
#define or_v(a, b, ...) call_mask_fn(or_epi64, (a), (b), ##__VA_ARGS__)
#define xor_v(a, b, ...) call_mask_fn(xor_epi64, (a), (b), ##__VA_ARGS__)
#define srlv(a, b, ...) call_mask_fn(srlv_epi64, (a), (b), ##__VA_ARGS__)
#define sllv(a, b, ...) call_mask_fn(sllv_epi64, (a), (b), ##__VA_ARGS__)
#define srli(a, b, ...) call_mask_fn(srli_epi64, (a), (b), ##__VA_ARGS__)
#define slli(a, b, ...) call_mask_fn(slli_epi64, (a), (b), ##__VA_ARGS__)
#define minu(a, b, ...) call_mask_fn(min_epu64, (a), (b), ##__VA_ARGS__)
#define maxu(a, b, ...) call_mask_fn(max_epu64, (a), (b), ##__VA_ARGS__)

#define maskz(mask, a) vec_fn(maskz_mov_epi64)((__mmask8)(mask), (a))
#define blend(src, mask, a) vec_fn(mask_mov_epi64)((src), (__mmask8)(mask), (a))
#define blend8(mask, off, on)                                                  \
  vec_fn(mask_blend_epi8)((__mmask64)(mask), (off), (on))
#define blend16(mask, off, on)                                                 \
  vec_fn(mask_blend_epi16)((__mmask32)(mask), (off), (on))
#define blend32(mask, off, on)                                                 \
  vec_fn(mask_blend_epi32)((__mmask16)(mask), (off), (on))
#define blend64(mask, off, on)                                                 \
  vec_fn(mask_blend_epi64)((__mmask8)(mask), (off), (on))

// ternary(a, b, c, imm8)             -> ternlog(a, b, c)
// ternary(a, b, c, imm8, mask)       -> inactive lanes keep a
// ternary(a, b, c, imm8, mask, zero) -> inactive lanes are zeroed
#define ternary(...) CAT(ternary_, NARGS(__VA_ARGS__))(__VA_ARGS__)
#define ternary_4(a, b, c, imm8)                                               \
  vec_fn(ternarylogic_epi64)((a), (b), (c), (imm8))
#define ternary_5(a, b, c, imm8, mask)                                         \
  vec_fn(mask_ternarylogic_epi64)((a), (__mmask8)(mask), (b), (c), (imm8))
#define ternary_6(a, b, c, imm8, mask, zero)                                   \
  vec_fn(maskz_ternarylogic_epi64)((__mmask8)(mask), (a), (b), (c), (imm8))

// Constructors and simple constants.
#define zero() vec_fn(setzero_si512)()
#define dzero() vec_fn(setzero_pd)()
#define ones() vec_fn(set1_epi64)(-1LL)
#define set1_64(x) vec_fn(set1_epi64)((long long)(x))
#define set1_32(x) vec_fn(set1_epi32)((int)(x))
#define set1_d(x) vec_fn(set1_pd)((double)(x))
#define setr_64(e0, e1, e2, e3, e4, e5, e6, e7)                                \
  vec_fn(setr_epi64)((long long)(e0), (long long)(e1), (long long)(e2),        \
                     (long long)(e3), (long long)(e4), (long long)(e5),        \
                     (long long)(e6), (long long)(e7))
#define setr_32(e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13,    \
                e14, e15)                                                      \
  vec_fn(setr_epi32)((int)(e0), (int)(e1), (int)(e2), (int)(e3), (int)(e4),    \
                     (int)(e5), (int)(e6), (int)(e7), (int)(e8), (int)(e9),    \
                     (int)(e10), (int)(e11), (int)(e12), (int)(e13),           \
                     (int)(e14), (int)(e15))

// Unsigned epi64 comparisons returning mask8.
#define eq(a, b) vec_fn(cmpeq_epu64_mask)((a), (b))
#define neq(a, b) vec_fn(cmpneq_epu64_mask)((a), (b))
#define ltu(a, b) vec_fn(cmplt_epu64_mask)((a), (b))
#define leu(a, b) vec_fn(cmple_epu64_mask)((a), (b))
#define gtu(a, b) vec_fn(cmpgt_epu64_mask)((a), (b))
#define geu(a, b) vec_fn(cmpge_epu64_mask)((a), (b))
#define is_zero(a) eq((a), zero())

// Common ternlog names.  Operand order matters for mux.
#define or3(a, b, c) ternary((a), (b), (c), 0xfe)
#define xor3(a, b, c) ternary((a), (b), (c), 0x96)
#define maj(a, b, c) ternary((a), (b), (c), 0xe8)
#define mux(s, a, b) ternary((s), (a), (b), 0xd8)
#define bitselect(s, a, b) mux((s), (a), (b))

// Mask casts, widening, arithmetic, and boolean ops.
#define k8(x) ((__mmask8)(x))
#define k16(x) ((__mmask16)(x))
#define k32(x) ((__mmask32)(x))
#define k64(x) ((__mmask64)(x))

#define kadd(a, b)                                                             \
  _Generic((a),                                                                \
      __mmask8: _kadd_mask8,                                                   \
      __mmask16: _kadd_mask16,                                                 \
      __mmask32: _kadd_mask32,                                                 \
      __mmask64: _kadd_mask64)((a), (b))
#define kand(a, b)                                                             \
  _Generic((a),                                                                \
      __mmask8: _kand_mask8,                                                   \
      __mmask16: _kand_mask16,                                                 \
      __mmask32: _kand_mask32,                                                 \
      __mmask64: _kand_mask64)((a), (b))
#define kandn(a, b)                                                            \
  _Generic((a),                                                                \
      __mmask8: _kandn_mask8,                                                  \
      __mmask16: _kandn_mask16,                                                \
      __mmask32: _kandn_mask32,                                                \
      __mmask64: _kandn_mask64)((a), (b))
#define kor(a, b)                                                              \
  _Generic((a),                                                                \
      __mmask8: _kor_mask8,                                                    \
      __mmask16: _kor_mask16,                                                  \
      __mmask32: _kor_mask32,                                                  \
      __mmask64: _kor_mask64)((a), (b))
#define kxor(a, b)                                                             \
  _Generic((a),                                                                \
      __mmask8: _kxor_mask8,                                                   \
      __mmask16: _kxor_mask16,                                                 \
      __mmask32: _kxor_mask32,                                                 \
      __mmask64: _kxor_mask64)((a), (b))
#define kxnor(a, b)                                                            \
  _Generic((a),                                                                \
      __mmask8: _kxnor_mask8,                                                  \
      __mmask16: _kxnor_mask16,                                                \
      __mmask32: _kxnor_mask32,                                                \
      __mmask64: _kxnor_mask64)((a), (b))
#define knot(a)                                                                \
  _Generic((a),                                                                \
      __mmask8: _knot_mask8,                                                   \
      __mmask16: _knot_mask16,                                                 \
      __mmask32: _knot_mask32,                                                 \
      __mmask64: _knot_mask64)(a)

#define klsh(a, imm)                                                           \
  _Generic((a),                                                                \
      __mmask8: _kshiftli_mask8(k8(a), (imm)),                                 \
      __mmask16: _kshiftli_mask16(k16(a), (imm)),                              \
      __mmask32: _kshiftli_mask32(k32(a), (imm)),                              \
      __mmask64: _kshiftli_mask64(k64(a), (imm)))
#define krsh(a, imm)                                                           \
  _Generic((a),                                                                \
      __mmask8: _kshiftri_mask8(k8(a), (imm)),                                 \
      __mmask16: _kshiftri_mask16(k16(a), (imm)),                              \
      __mmask32: _kshiftri_mask32(k32(a), (imm)),                              \
      __mmask64: _kshiftri_mask64(k64(a), (imm)))

// Lane movement and full-vector permutes.
#define alignr64(a, b, imm) vec_fn(alignr_epi64)((a), (b), (imm))
#define alignr8(...) CAT(alignr8_, NARGS(__VA_ARGS__))(__VA_ARGS__)
#define alignr8_3(a, b, imm) vec_fn(alignr_epi8)((a), (b), (imm))
#define alignr8_4(a, b, imm, mask)                                             \
  vec_fn(maskz_alignr_epi8)((__mmask64)(mask), (a), (b), (imm))
#define alignr8_5(a, b, imm, mask, src)                                        \
  vec_fn(mask_alignr_epi8)((src), (__mmask64)(mask), (a), (b), (imm))
#define lane_lsh(a, n) alignr64((a), zero(), 8 - (n))
#define lane_rsh(a, n) alignr64(zero(), (a), (n))
#define shuffle8(a, idx, ...)                                                  \
  call_mask_fn(shuffle_epi8, (a), (idx), ##__VA_ARGS__)
#define perm64(idx, a) vec_fn(permutexvar_epi64)((idx), (a))
#define perm32(idx, a) vec_fn(permutexvar_epi32)((idx), (a))
#define permb(idx, a) vec_fn(permutexvar_epi8)((idx), (a))

// VBMI2 double shifts.  Immediate forms take an immediate shift count; variable
// forms take a vector of per-lane counts.
#define fshl64i(...) CAT(fshl64i_, NARGS(__VA_ARGS__))(__VA_ARGS__)
#define fshl64i_3(a, b, imm) vec_fn(shldi_epi64)((a), (b), (imm))
#define fshl64i_4(a, b, imm, mask)                                             \
  vec_fn(maskz_shldi_epi64)((__mmask8)(mask), (a), (b), (imm))
#define fshl64i_5(a, b, imm, mask, src)                                        \
  vec_fn(mask_shldi_epi64)((src), (__mmask8)(mask), (a), (b), (imm))

#define fshr64i(...) CAT(fshr64i_, NARGS(__VA_ARGS__))(__VA_ARGS__)
#define fshr64i_3(a, b, imm) vec_fn(shrdi_epi64)((a), (b), (imm))
#define fshr64i_4(a, b, imm, mask)                                             \
  vec_fn(maskz_shrdi_epi64)((__mmask8)(mask), (a), (b), (imm))
#define fshr64i_5(a, b, imm, mask, src)                                        \
  vec_fn(mask_shrdi_epi64)((src), (__mmask8)(mask), (a), (b), (imm))

#define fshl64v(...) CAT(fshl64v_, NARGS(__VA_ARGS__))(__VA_ARGS__)
#define fshl64v_3(a, b, counts) vec_fn(shldv_epi64)((a), (b), (counts))
#define fshl64v_4(a, b, counts, mask)                                          \
  vec_fn(maskz_shldv_epi64)((__mmask8)(mask), (a), (b), (counts))
#define fshl64v_5(a, b, counts, mask, src)                                     \
  blend((src), (mask), fshl64v_3((a), (b), (counts)))

#define fshr64v(...) CAT(fshr64v_, NARGS(__VA_ARGS__))(__VA_ARGS__)
#define fshr64v_3(a, b, counts) vec_fn(shrdv_epi64)((a), (b), (counts))
#define fshr64v_4(a, b, counts, mask)                                          \
  vec_fn(maskz_shrdv_epi64)((__mmask8)(mask), (a), (b), (counts))
#define fshr64v_5(a, b, counts, mask, src)                                     \
  blend((src), (mask), fshr64v_3((a), (b), (counts)))

// Explicit IFMA helpers.
#define madd52lo(acc, a, b) vec_fn(madd52lo_epu64)((acc), (a), (b))
#define madd52hi(acc, a, b) vec_fn(madd52hi_epu64)((acc), (a), (b))
#define madd52lo1(acc, a, x) madd52lo((acc), (a), set1_64(x))
#define madd52hi1(acc, a, x) madd52hi((acc), (a), set1_64(x))

#define adc(carry, a, b, r)                                                    \
  _addcarry_u64((carry), (a), (b), (unsigned long long *)(r))

#define sbb(carry, a, b, r)                                                    \
  _subborrow_u64((carry), (a), (b), (unsigned long long *)(r))