#pragma once
#include "types.h"

#define add_step_u64_vec(r, a, b, carry, nmask)                                \
  {                                                                            \
    const _vec max_word = set1_64(-1);                                         \
    r = add(a, b);                                                             \
    __mmask16 c = ltu(r, a);                                                   \
    __mmask16 m = eq(r, max_word);                                             \
    c = carry | (c << 1);                                                      \
    c += m; m ^= c;                                                            \
    carry = c >> nmask;                                                        \
    r = sub(r, max_word, m, r);                                                \
  }

#define sub_step_u64_vec(r, a, b, carry, nmask)                                \
  {                                                                            \
    const _vec max_word = set1_64(-1);                                         \
    r = sub(a, b);                                                             \
    __mmask16 c = gtu(r, a);                                                   \
    __mmask16 m = eq(r, zero());                                               \
    __mmask16 active = (__mmask16)((1u << nmask) - 1u);                        \
    c &= active; m &= active;                                                  \
    c = carry | (c << 1);                                                      \
    c += m; m ^= c;                                                            \
    carry = c >> nmask;                                                        \
    r = add(r, max_word, m, r);                                                \
  }

#define _MPN_ADDSUB_N mpn_add_n
#define ADDSUB adc
#define FLEX_add_step_u64_vec add_step_u64_vec
#include "addsub_n_impl"
#undef _MPN_ADDSUB_N
#undef ADDSUB
#undef FLEX_add_step_u64_vec

#define _MPN_ADDSUB_N mpn_sub_n
#define ADDSUB sbb
#define FLEX_add_step_u64_vec sub_step_u64_vec
#include "addsub_n_impl"
#undef _MPN_ADDSUB_N
#undef ADDSUB
#undef FLEX_add_step_u64_vec
