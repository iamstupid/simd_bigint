// r = a + b, canonical (each limb < 2^52). Add is commutative so operands are
// swapped to put the longer first. The carry out of limb (max-1) lands in limb
// `max`: when extra_limb is set it is written (result occupies max+1 limbs, the
// top one possibly 0) and max+1 is returned; otherwise it is dropped and max is
// returned. extra_limb is a runtime bool, but inlined at a constant call site it
// folds away, so callers in the "always carried" regime pay no branch.
static inline uint64_t u52_add_carry(pvec r, cpvec a, cpvec b, uint64_t na, uint64_t nb, bool extra_limb){
    assert(na > 0 && nb > 0);
    if(na < nb){ cpvec t = a; a = b; b = t; uint64_t n = na; na = nb; nb = n; }

    const uint8_t  at = (uint8_t)((na - 1) & 7), bt = (uint8_t)((nb - 1) & 7);
    const __mmask8 am = (__mmask8)((1u << (at + 1)) - 1);
    const __mmask8 bm = (__mmask8)((1u << (bt + 1)) - 1);
    const uint64_t nav = (na - 1) >> 3, nbv = (nb - 1) >> 3;
    uint16_t carry = 0;
    _vec ax, bx;
    uint64_t i;

    // full a+b vectors below b's top vector
    for(i = 0; i < nbv; ++i){
        ax = load_vec(a++), bx = load_vec(b++);
        carry_prop(r, ax, bx, carry);
    }
    // reach a's top vector, with bx = the matching b lanes (zero beyond b)
    if(nav == nbv){
        ax = load_vec(a, am), bx = load_vec(b, bm);
    }else{
        ax = load_vec(a++), bx = load_vec(b, bm);   // vector nbv: a full, b partial
        carry_prop(r, ax, bx, carry);
        for(++i; i < nav; ++i){                      // a-only full vectors
            ax = load_vec(a++);
            carry_prop(r, ax, zero(), carry);
        }
        ax = load_vec(a, am), bx = zero();           // a's top vector
    }

    // top vector: store a's lanes, plus the carry limb if requested
    if(extra_limb && at != 7){
        // carry-out terminates in lane at+1 (a zero lane); widen the store to keep it
        const __mmask8 sm = (__mmask8)((1u << (at + 2)) - 1);
        carry_prop(r, ax, bx, carry, sm);
        return na + 1;
    }
    carry_prop(r, ax, bx, carry, am);
    if(extra_limb){                                  // at == 7: carry spills into a new limb
        store_vec(r, set1_64((uint64_t)(carry & 1)), (__mmask8)0x01);
        return na + 1;
    }
    return na;
}
