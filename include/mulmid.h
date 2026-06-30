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
}