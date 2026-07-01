#pragma once
#include "types.h"
#include <cstdint>

/**
 * Middle product formulation:
 * convention: B = |b| > A = |a|
 * MP(a,b) = sum_{t in [A-1, B-1]} sum_{i in [0, A-1]} a[i]*b[t-i]
 * E.G. the middle part of a polynomial multiplication that is rectangular.
 * Example:
 * a\b | 0 1 2 3 4 5 6 7
 *  0  | O O O * * * * *
 *  1  | O O * * * * * .
 *  2  | O * * * * * . .
 *  3  | * * * * * . . .
 * DIAG| 3 4 5 6 7 8 9 a
 *      [SUMMED UP] <- those diagonals
 */

static inline void mulmid_basecase(plimb r, const _limb *a, const _limb *b, int64_t an, int64_t bn){
    /**
     * Design of basecase kernel:
     * a. We have to accumulate lo and hi sums separately, since hi sum leak from 
     * below, and lo sum leak from above the boundary.
     * This affect our semantical correctness, also invalidates divide and conquer.
     * b. We first iterate via b indices. This is because B is larger and the result
     * length is under B+1. (Actually B-A+1+1; the final +1 come from the hi offset)
     * c. Thus, we indeed iterate via result index. Every time we want to accumulate
     * products for r[i..i+8], which has a lower bound on b that involved in the sum
     * of b[i..i+8]
     * d. We then decrease a pointer. Process by block of 8 in a value, however we
     * accumulate like dividing a by 4. The intuition is the following:
     * [case a] accumulate by 8: we cannot avoid that, for the first block, we have
     * to waste half of the multiplication for useless garbage. The shape is:
     * | notice the    * | * * * * * * * * |
     * | upper tri-  * * | * * * * * * * * |
     * | angle     * * * | * * * * * * *   |
     * | here    * * * * | * * * * * *     |
     * |       * * * * * | * * * * *       |
     * |     * * * * * * | * * * *         |
     * |   * * * * * * * | * * *           |
     * | * * * * * * * * | * *             |
     * [bptr .. bptr+8  ][bptr+8 .. bptr+16]
     * The upper triangle is indeed not reducible under this formulation, because we
     * need to do this diagonal triangle anyways. We also have to take care of the
     * lo/hi separation. A naive idea would be doing 16 accumulator vectors; a more
     * advanced idea is to load bptr and bptr+1 vector, then do even/odd split.
     *
     * This is why we want to do step 4 on a.
     *                   | * * * * * * * * | * ...
     *                   [bptr+8 .. bptr+16] <- next block, at here swaps for the 2nd
     *                                          vector; and load [bptr+16 .. bptr+24]
     * |       * * * * * | * * * * *       |   /\  # Figure of this 4 step is shifted
     * |     * * * * * * | * * * *         |  /||\ # 4 limbs front
     * |   * * * * * * * | * * *           |   || Direction of computation
     * | * * * * * * * * | * *             |   ||
     * [bptr+ 4..bptr+12] <- We don't load this vector; instead, since we need to do
     *                       next block as [bptr+8 .. bptr+16], we can pre-load this
     *                       vector, then compute [bptr+ 4 .. bptr+12] as a simple
     *                       vpalignq(hi, lo, 4). This reduce the load occupation due
     *                       to potential unaligned load jankiness.
     * |       * * * * * | * * * * *       |   /\
     * |     * * * * * * | * * * *         |  /||\
     * |   * * * * * * * | * * *           |   || Direction of computation
     * | * * * * * * * * | * *             |   ||
     * [bptr .. bptr+8  ][bptr+8 .. bptr+16] <- load two vectors right now
     * E. Tail computation: we do it vertically
     *                   | * * * * * * * * |   * ...
     * ------------------------------------|a  ||                   
     * |       * * * * * | * * * * *       |0 \||/ in this direction a increases
     * |     * * * * * * | * * * * |       |1  \/  We load a in vector A
     * |   * * * * * * * | * * * | |       |2 
     * | * * * * * * * * | * * | | |       |3  If we want to do vertical point
     *              bptr + 4 5 6 7 8           multiplication, then we have to load B
     *                                         in the underlying form: [44440000]
     *                                         [55551111][66662222][77773333], etc.
     * |       * * * * * | * * * * *       |4 
     * |     * * * * * * | * * * *         |5  Thing is essentially [44440000] *
     * |   * * * * * * * | * * *           |6  [01234567] -> [45674567]
     * | * * * * * * * * | * *             |7  Note that:
     *              bptr + 0 1 2 3 4           [456789ab] -> acc[3]
     *                                         [56789abc] -> acc[2]
     *                                         [6789abcd] -> acc[1]
     *                                         [789abcde] -> acc[0]
     */
    const int64_t rn = bn - an + 1; // output length = rn+1
    _vec acc_lo[4], acc_hi[4], last_lo[4], last_hi[4], b0, b1, bh;
    last_lo[0] = zero(); last_lo[1] = zero(); last_lo[2] = zero(); last_lo[3] = zero();
    last_hi[0] = zero(); last_hi[1] = zero(); last_hi[2] = zero(); last_hi[3] = zero();
    acc_lo[0] = zero(); acc_lo[1] = zero(); acc_lo[2] = zero(); acc_lo[3] = zero();
    acc_hi[0] = zero(); acc_hi[1] = zero(); acc_hi[2] = zero(); acc_hi[3] = zero();
    #define mm_acc(ind, aind, b) { \
        acc_lo[ind] = madd52lo(acc_lo[ind], splat_load(aptr, aind), b); \
        acc_hi[ind] = madd52hi(acc_hi[ind], splat_load(aptr, aind), b); \
    }
    #define mm_acc_2(ind, aind, b) {\
        \
    }
    for(int64_t i = 0; i < rn; i += 8){
        const _limb *bptr = b + i;
        if(i + 8 < rn){
            b0 = load_vec(bptr);
            for(int64_t j = an; j > 0; j -= 8){
                const _limb *aptr = a + (j-8);
                b1 = load_vec(bptr += 8);
                bh = alignr64(b1, b0, 4);
                if(j <= 8) switch(j){
                    full:
                    case 8: mm_acc(3, 0, bh);
                    case 7: mm_acc(2, 1, bh);
                    case 6: mm_acc(1, 2, bh);
                    case 5: mm_acc(0, 3, bh);
                    case 4: mm_acc(3, 4, b0);
                    case 3: mm_acc(2, 5, b0);
                    case 2: mm_acc(1, 6, b0);
                    case 1: mm_acc(0, 7, b0);
                }
                else goto full;
                b0 = b1;
            }
        }else{
            // tail
        }
    }
    #undef mm_acc
    #undef mm_acc_2
}

static inline void midmul_basecase_simple(plimb r, const _limb *a, const _limb *b, int64_t an, int64_t bn){
    // A simpler version with unaligned vectors (load, then valignq-ed into unaligned)
    // should be a bit faster with smaller inputs
    // due to no shape loss
    _vec acc[8], b0, b1, sum, sum1, last1 = zero();
    // acc[i] is basically, only for dependency elimination
    // acc[even] is aligned  acc lo
    // acc[ odd] is off-by-1 acc hi
    const int64_t rn = bn - an + 1;
    #define mm_acc(ind) {\
        const _vec _v = alignr64(b1, b0, ind); \
        const _vec _a = splat_load(aptr, 7 - ind); \
        acc[(ind&3)*2  ] = madd52lo(acc[(ind&3)*2  ] , _v, _a); \
        acc[(ind&3)*2+1] = madd52hi(acc[(ind&3)*2+1] , _v, _a); \
    }
    for(int64_t i = 0; i < rn; i += 8){
        const _limb *bptr = b + i;
        sum = zero(); sum1 = zero();
        for(int _ = 0; _ < 8; ++_) acc[_] = zero();
        b0 = load_vec(bptr);
        for(int64_t j = an; j > 0; j -= 8){
            const _limb *aptr = a + (j-8);
            b1 = load_vec(bptr += 8);
            if(j <= 8) switch(j){
                full:
                case 8: mm_acc(7);
                case 7: mm_acc(6);
                case 6: mm_acc(5);
                case 5: mm_acc(4);
                case 4: mm_acc(3);
                case 3: mm_acc(2);
                case 2: mm_acc(1);
                case 1: mm_acc(0);
            }else goto full;
            b0 = b1;
        }
        acc[5] = add(acc[5], acc[7]);
        acc[4] = add(acc[4], acc[6]);
        acc[1] = add(acc[3], acc[1]);
        acc[0] = add(acc[2], acc[0]);
        acc[1] = add(acc[5], acc[1]);
        acc[0] = add(acc[4], acc[0]);
        if(i + 8 < rn){
            store_vec(r, add(acc[0], alignr64(acc[1], last1, 7)));
            r+=8;
            last1 = acc[1];
        }else{
            _vec hi1 = alignr64(acc[1], last1, 7);
            const uint8_t mask = uint8_t((1 << (rn - i)) - 1);
            const uint8_t write_mask = uint8_t((1 << (rn - i + 1)) - 1);
            _vec re1 = add(hi1, acc[0], mask, hi1);
            store_vec(r, re1, write_mask);
            if(rn == i + 8) store_vec(r+1, acc[1], 0x80);
        }
    }
    #undef mm_acc
}

static inline void midmul_basecase_simple_wtail(plimb r, const _limb *a, const _limb *b, int64_t an, int64_t bn){
    // A simpler version with unaligned vectors (load, then valignq-ed into unaligned)
    // should be a bit faster with smaller inputs
    // due to no shape loss
    _vec acc[8], b0, b1, last1 = zero(), lo = zero(), hi = zero(), hi1 = zero();
    // acc[i] is basically, only for dependency elimination
    // acc[even] is aligned  acc lo
    // acc[ odd] is off-by-1 acc hi
    const int64_t rn = bn - an + 1;
    #define mm_acc(ind) {\
        const _vec _v = alignr64(b1, b0, ind); \
        const _vec _a = splat_load(aptr, 7 - ind); \
        acc[(ind&3)*2  ] = madd52lo(acc[(ind&3)*2  ] , _v, _a); \
        acc[(ind&3)*2+1] = madd52hi(acc[(ind&3)*2+1] , _v, _a); \
    }
    for(int64_t i = 0; i < rn; i += 8){
        const _limb *bptr = b + i, *aptr;
        for(int _ = 0; _ < 8; ++_) acc[_] = zero();
        int64_t j = an;
        {
            if(i+8 <= rn){
                b0 = load_vec(bptr);
            }else{
                // i+8 > rn: only one non-full block left
                // We only pre-process the tail block, which
                // we can't process it vertically.
                j = an & 7; bptr += an - j;
                b0 = load_vec(bptr); b1 = load_vec(bptr + 8);
                aptr = a + j - 8;
                if(j) goto htail; else{
                    // the last block is also full
                    // we skip over all the unnecessary compute
                    // directly into vertical sums part
                    lo = zero();
                    hi1 = alignr64(zero(), last1, 7);
                    goto vsums;
                }
            }
            for(; j > 0; j -= 8){
                aptr = a + (j-8);
                b1 = load_vec(bptr += 8);
                if(j <= 8){
                    htail:
                    switch(j){
                        full:
                        case 8: mm_acc(7);
                        case 7: mm_acc(6);
                        case 6: mm_acc(5);
                        case 5: mm_acc(4);
                        case 4: mm_acc(3);
                        case 3: mm_acc(2);
                        case 2: mm_acc(1);
                        case 1: mm_acc(0);
                    }
                }else goto full;
                b0 = b1;
            }
            acc[5] = add(acc[5], acc[7]);
            acc[4] = add(acc[4], acc[6]);
            acc[1] = add(acc[3], acc[1]);
            acc[0] = add(acc[2], acc[0]);
            acc[1] = add(acc[5], acc[1]);
            acc[0] = add(acc[4], acc[0]);
        }
        if(i + 8 < rn){
            store_vec(r, add(acc[0], alignr64(acc[1], last1, 7)));
            r+=8;
            last1 = acc[1];
        }else{
            lo = acc[0]; hi = acc[1];
            hi1 = alignr64(hi, last1, 7);
            if(i + 8 > rn){
                // process the remaining parts
                // accumulators now have different (vertical) semantics
                // so we need to clean the accumulators
                for(int _ = 0; _ < 8; ++_) acc[_] = zero();
                vsums:
                acc[8] = zero();
                const _vec rev = setr_64(7,6,5,4,3,2,1,0);
                const uint8_t sel_ni = rn - i; // [1,7]
                bptr = b + i; b0 = load_vec(bptr);
                for(j = an - 8; j >= 0; j -= 8){
                    b1 = load_vec(bptr += 8);
                    _vec ax = load_vec(a + j);
                    ax = perm64(rev, ax); // ax is reverse a+j loaded
                    #define mm_acc_2(ind) case ind: {\
                        const _vec _v = alignr64(b1, b0, ind - 1); \
                        acc[ind-1] = madd52lo(acc[ind-1], _v, ax); \
                        acc[ ind ] = madd52hi(acc[ ind ], _v, ax); \
                    }
                    switch(sel_ni){
                        mm_acc_2(7)
                        mm_acc_2(6)
                        mm_acc_2(5)
                        mm_acc_2(4)
                        mm_acc_2(3)
                        mm_acc_2(2)
                        mm_acc_2(1)
                    }
                    #undef mm_acc_2
                    b0 = b1;
                }
                // batch vertical sum
                // fixed cost
                acc[0]=add(unpacklo64(acc[0], acc[1]), unpackhi64(acc[0], acc[1]));
                acc[2]=add(unpacklo64(acc[2], acc[3]), unpackhi64(acc[2], acc[3]));
                acc[4]=add(unpacklo64(acc[4], acc[5]), unpackhi64(acc[4], acc[5]));
                acc[6]=add(unpacklo64(acc[6], acc[7]), unpackhi64(acc[6], acc[7]));
                acc[0]=add(shufi64x2(acc[0], acc[2], 0x88), shufi64x2(acc[0], acc[2], 0xDD));
                acc[4]=add(shufi64x2(acc[4], acc[6], 0x88), shufi64x2(acc[4], acc[6], 0xDD));
                acc[0]=add(shufi64x2(acc[0], acc[4], 0x88), shufi64x2(acc[0], acc[4], 0xDD));
                hi1 = add(hi1, acc[0]);
            }
            const uint8_t mask = uint8_t((1 << (rn - i)) - 1);
            const uint8_t write_mask = uint8_t((1 << (rn - i + 1)) - 1);
            _vec re1 = add(hi1, lo, mask, hi1);
            store_vec(r, re1, write_mask);
            if(rn == i + 8) store_vec(r+1, hi, 0x80);
        }
    }
    #undef mm_acc
}
