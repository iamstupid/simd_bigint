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
        acc[7] += hi(B0 * A0[6]) + hi(B1 * A0[5])
        acc[6] += lo(B0 * A0[6]) + lo(B1 * A0[5])
        acc[5] += hi(B0 * A0[4]) + hi(B1 * A0[3])
        acc[4] += lo(B0 * A0[4]) + lo(B1 * A0[3])
        acc[3] += hi(B0 * A0[2]) + hi(B1 * A0[1])
        acc[2] += lo(B0 * A0[2]) + lo(B1 * A0[1])
        acc[1] += hi(B0 * A0[0]) + hi(B1 * A0[-1])
        acc[0] += lo(B0 * A0[0]) + lo(B1 * A0[-1])
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