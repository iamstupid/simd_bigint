A weird formulation mitigates the problem.

Let's formulate the problem first:

Suppose we have A = {a[0], ..., a[an]}, B = {b[0], ..., b[bn]}, |A| = an+1, |B| = bn+1 > an+1. We want to compute:

Forall k in [an, bn], R[k] = sum{i+j=k} A[i]*b[j]

Due to madd52 formulation, we have to separate it into high and low:

Forall k in [an, bn], Rh[k] = sum{i+j=k} Hi(A[i]*b[j]), Rl[k] = sum{i+j=k} Lo(A[i]*b[j]), we want to have R[k] = Rh[k-1] + Rl[k]

Problem arises, when we consider the final endpoints of R, if we do the split:

R[an] = Rh[an-1] + Rl[an] # Rh[an-1] should not exist
R[bn] = Rh[bn] + Rl[bn+1] # Rl[bn+1] should not exist

One obvious fix:

1. Compute Rh[an-1], Rl[bn+1], and subtract from the final sum. This is easy; however not desired due to the extra work.

---

So consider this approach.

We load B0 = B[8k + 0..7], B1 = B[8k + 1..8] each time. We also have a slice of As: A0 = A[an - (8j + 0..7)]
(Index reversed; A0[0] = A[an - 8j], A0[7] = A[an - 8j - 7]).

We allocate 8 acc slots: acc[2..9]
For even slots, we accumulate low parts. For odd slots, we accumulate high parts.
For this block:

acc[9] = hi(B0 * A0[0]) + hi(B1 * A0[1])
acc[8] = lo(B0 * A0[0]) + lo(B1 * A0[1])
acc[7] = hi(B0 * A0[2]) + hi(B1 * A0[3])
acc[6] = lo(B0 * A0[2]) + lo(B1 * A0[3])
acc[5] = hi(B0 * A0[4]) + hi(B1 * A0[5])
acc[4] = lo(B0 * A0[4]) + lo(B1 * A0[5])
acc[3] = hi(B0 * A0[6]) + hi(B1 * A0[7])
acc[2] = lo(B0 * A0[6]) + lo(B1 * A0[7])

The merit of this method, is we separated the sums from lo part and hi part, thus we can apply masking to acc[9]+acc[7]+acc[5]+acc[3] or acc[8]+acc[6]+acc[4]+acc[2],
to mask for specifically lo/hi, to mask out the boundary case.

---

We have to accomodate the same optimized block looping structure. We loop over the output vector:

Output offset: O[0] = Rl[an], O[bn-an] = Rl[bn], O[bn-an+1] = Rh[bn]. So length bn-an+2/+3. +3 is due to overflow;
since we use redundant representation we doesn't need it all the time.

```
for i = 0 .. bn - an : step 8{
    # starting index of A is an, starting index of B is 0
    # corresponding output vector is O[0..7]
    j = an;
    for k = i .. i + an : step 8{
        B0 = load(B + k);
        B1 = load(B + k + 1);
        A0 = A + j; j -= 8;
        # acc[0] = lo(B0 * A0[0]) + ...
        acc[1] += hi(B0 * A0[0]) + hi(B1 * A0[-1])
        acc[0] += lo(B0 * A0[0]) + lo(B1 * A0[-1])
        acc[7-8] += hi(B0 * A0[6-8]) + hi(B1 * A0[5-8])
        acc[6-8] += lo(B0 * A0[6-8]) + lo(B1 * A0[5-8])
        acc[5-8] += hi(B0 * A0[4-8]) + hi(B1 * A0[3-8])
        acc[4-8] += lo(B0 * A0[4-8]) + lo(B1 * A0[3-8])
        acc[3-8] += hi(B0 * A0[2-8]) + hi(B1 * A0[1-8])
        acc[2-8] += lo(B0 * A0[2-8]) + lo(B1 * A0[1-8])
    }
    # process j residues ...
    ...
    # mask and harvest
    sum_even = acc[0] + alignq(acc[2], last[2], 6) + alignq(acc[4], last[4], 4) + alignq(acc[6], last[6], 6);
    sum_odd = alignq(acc[1], last[1], 7) + alignq(acc[3], last[3], 5) + alignq(acc[5], last[5], 3) + alignq(acc[7], last[7], 1);
    # if i == 0, mask (sum of acc[odd])'s lane 0
    # if i is last, mask (sum off acc[even])'s corresponding lane
    # then do the sum of even and odd
    # and do masked store

    # move acc to last
}
```

---

Doing a bit of manual simplification. Let's work with positive indices.

Idea: forward the block sum

Let a = |A|, b = |B|. (e.g. an+1, bn+1)

A enumerator is j; B enumerator is i.

A\B | 0 1 2 3 4 5 6 7 | offset | acc pos | hi pos | eo acc pos | eo hi pos |
 0  |               * |   -7   |   7+1   |   7    |     6+1    |     6     |
 1  |             * * |   -6   |   6+1   |   6    |     6+1    |     6     |
 2  |           * * * |   -5   |   5+1   |   5    |     4+1    |     4     |
 3  |         * * * * |   -4   |   4+1   |   4    |     4+1    |     4     |
 4  |       * * * * * |   -3   |   3+1   |   3    |     2+1    |     2     |
 5  |     * * * * * * |   -2   |   2+1   |   2    |     2+1    |     2     |
 6  |   * * * * * * * |   -1   |   1+1   |   1    |     0+1    |     0     |
 7  | * * * * * * * * |    0   |   0+1   |   0    |     0+1    |     0     |

So, we enumerate B(i) at step of 8, then A. A decreasing in blocks. So the base ptr of A is j = a-8. The block enumerates j + (0..7).

The identity is: j + i + 7 = an. So i = an - 7 - j = a - 8 - j = a - 8 - (a - 8) = 0.

Even/odd load = j/j+1.

acc[1] is shift=0 position; store has to delay by one block. So for the next block, the harvest is:

```
losum = last[1] + alignq(last[3], acc[3], 2) + alignq(last[5], acc[5], 4) + alignq(last[7], acc[7], 6);
hisum =           alignq(last[2], acc[2], 1) + alignq(last[4], acc[4], 3) + alignq(last[6], acc[6], 5) + alignq(last[0], last[8], 7);
# alignq of last[0], last[8] is one block delayed
# because acc[0] can not get harvest this block
if(i == 8){
    hisum = hisum.mask(0xfe); # mask out least significant limb
}
if(i + a > b){
    # last block; mask lo and harvest highest
    # we need to mask lo for entry bn + 1
    # index arithmetics: last[1] base index is i + a - 8 + 7 = i + a - 1.
    # we need to mask the position bn - (i + a - 1) + 1 = b - a - i + 1.
    mask = 1 << (b - a - i + 1);
    mask = (~mask) & 0xFF; # if b - a - i + 1 = 8, then there's no need to mask
    losum = losum.mask(mask)
    result.push_store(losum + hisum)

    if(!mask){
        # one last row; that consists of only hisum via acc[0]
        result.push_store(acc[0], zero, 7)
    }
    break;
}
result.push_store(losum + hisum);

last[1..7] = acc[1..7]
last[0] = last[8]
last[8] = acc[0]
```

---

Now we design the productional shape of the kernel, taking into account the non-multiple-of-8 lengths. This is important for smooth scaling of the kernel.

Suppose we have a mod 8 = 4. The desired shape of our initial step is: (diagonal sums up to an + i = a + i - 1)

A\B | 0 1 2 3 4 5 6 7 | offset | acc pos | hi pos | eo acc pos | eo hi pos |
 0  | o o o * * * * * |   -3   |   3+1   |   3    |     2+1    |     2     | < jptr = a - 4
 1  | o o * * * * * * |   -2   |   2+1   |   2    |     2+1    |     2     |
 2  | o * * * * * * * |   -1   |   1+1   |   1    |     0+1    |     0     |
 3  | * * * * * * * * |    0   |   0+1   |   0    |     0+1    |     0     | < an
            ^ a + i - 1 - (a - 4) = i + 3
      ^ iptr

So for the first block, iptr start from i and decrements by 8; jptr = a - 4. Generally, if we have (a - k) mod 8 = 0, k in [1,8], we just let jptr = a - k,
and that's it. Caveat here: the A in group offsets changes. However, if instead, we just do jptr = a - 8, but skip the small steps, it is effectively the same;
only the next time we need to start from a & (~7). So we just add a target variable nxtj to indicate this.

**Important:** Claim above was falsified. Because if we treat this way, we will lose a triangle of values:

A\B | 0 1 2 3 4 5 6 7 8 9 a b c d e f g
 0  |                   * * * * * * * *
 1  |                 * * * * * * * *
 2  |               o * * * * * * *
 3  |             o o * * * * * *
 4  |           o o o * * * * *
 5  |         o o o o * * * *
 6  |       o o o o o * * *
 7  |     o o o o o o * * < Block 1, the o marked multiples are not calculated correctly
 8  |   * * * * * * * * <
 9  | * * * * * * * *   < Block 0

So it seems we can only go the other way round.

A\B | 0 1 2 3 4 5 6 7 | 8 9 a b c d e f | g h i j
 0  |   * * * * * * * | * * * *
 1  | * * * * * * * * | * * *
--------------------------------------------------
 2  |       * * * * * | * * * * * *
 3  |     * * * * * * | * * * * *
 4  |   * * * * * * * | * * * *
 5  | * * * * * * * * | * * *  
 -------------------------------------------------
 6  |       * * * * * | * * * * * *     |
 7  |     * * * * * * | * * * * *       |
 8  |   * * * * * * * | * * * *         |
 9  | * * * * * * * * | * * *           |


A\B | 0 1 2 3 4 5 6 7 | 8 9 a b c d e f | g h i j
 0  |   * * * * * * * | *
 1  | * * * * * * * * |
----------------------------------------------
 2  |       * * * * * | * * *           |
 3  |     * * * * * * | * *             |
 4  |   * * * * * * * | *               |
 5  | * * * * * * * * |                 |
-----------------------------------------------
 6  |       * * * * * | * * *           |
 7  |     * * * * * * | * *             |
 8  |   * * * * * * * | *  Partial      |
 9  | * * * * * * * * |    Block        |

```
for i = 0 .. b - a step 8{    
    iptr, jptr = i, a;
    for(;jptr>0;){
        B0, B1 = load(iptr), load(iptr+1);
        iptr += 8;
        p = jptr;
        jptr -= 8;
        if(jptr <= 0){
            switch(p){
                start:
                case 8: acc[7] += lo(B1 * splat(A[jptr + 0])), acc[6] += hi(B1 * splat(A[jptr + 0]));
                case 7: acc[7] += lo(B0 * splat(A[jptr + 1])), acc[6] += hi(B0 * splat(A[jptr + 1]));
                case 6: acc[5] += lo(B1 * splat(A[jptr + 2])), acc[4] += hi(B1 * splat(A[jptr + 2]));
                case 5: acc[5] += lo(B0 * splat(A[jptr + 3])), acc[4] += hi(B0 * splat(A[jptr + 3]));
                case 4: acc[3] += lo(B1 * splat(A[jptr + 4])), acc[2] += hi(B1 * splat(A[jptr + 4]));
                case 3: acc[3] += lo(B0 * splat(A[jptr + 5])), acc[2] += hi(B0 * splat(A[jptr + 5]));
                case 2: acc[1] += lo(B1 * splat(A[jptr + 6])), acc[0] += hi(B1 * splat(A[jptr + 6]));
                case 1: acc[1] += lo(B0 * splat(A[jptr + 7])), acc[0] += hi(B0 * splat(A[jptr + 7]));
            }
        }else goto start;
    }
    losum = last[1] + alignq(last[3], acc[3], 2) + alignq(last[5], acc[5], 4) + alignq(last[7], acc[7], 6);
    hisum =           alignq(last[2], acc[2], 1) + alignq(last[4], acc[4], 3) + alignq(last[6], acc[6], 5) + alignq(last[0], last[8], 7);
    # alignq of last[0], last[8] is one block delayed
    # because acc[0] can not get harvest this block
    if(i == 8){
        hisum = hisum.mask(0xfe); # mask out least significant limb
    }
    if(i + a >= b){
        # last block; mask lo and harvest highest
        # we need to mask lo for entry bn + 1
        # index arithmetics: last[1] base index is i + a - 8 + 7 = i + a - 1.
        # we need to mask the position bn - (i + a - 1) + 1 = b - a - i + 1.
        mask = 1 << (b - a - i + 1);
        mask = (~mask) & 0xFF; # if b - a - i + 1 = 8, then there's no need to mask
        losum = losum.mask(mask);
        result.push_store(losum + hisum);

        if(!mask){
            # one last row; that consists of only hisum via acc[0]
            result.push_store(alignq(acc[0], zero, 7));
        }
        break;
    }
    result.push_store(losum + hisum);

    last[1..7] = acc[1..7];
    last[0] = last[8];
    last[8] = acc[0];
    acc = 0;
}
```

---


