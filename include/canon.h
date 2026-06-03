#pragma once
#include "types.h"
#define _MPN_CANON canon_u52_pos
#define canonize(vec, carry, sigcarry) {\
    const _vec all1 = set1_64((1ull << 52) - 1); \
    _vec overflow = srli(vec, 52); \
    _vec anded = and_v(vec, all1); \
    _vec ofshifted = alignr64(overflow, carry, 7); \
    _vec added = add(anded, ofshifted); \
    __mmask16 c = gtu(added, all1); \
    __mmask16 m = eq(added, all1); \
    c = m + (sigcarry | (c << 1)); \
    m ^= c; \
    