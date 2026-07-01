# Align-reduced middle-product basecase (`mulmid_basecase`)

Goal: same IFMA work as `midmul_basecase_simple`, but **one `valignq` per a-block instead of
eight**. `simple` issues `alignr64(b1,b0,ind)` for `ind=0..7` (8 shuffles per a-block); the
reduced kernel issues a single `bh = alignr64(b1,b0,4)` and lets **4 accumulators + a cheap
per-output-block combine** recover the eight diagonals. On Zen5 the shuffle port (where `valignq`
lives) is what makes `simple` ~6% slower than square `mul_basecase`; killing 7/8 of the shuffles
is the point.

Everything below fixes the convention **`B = |b| ≥ |a| = A`, radix `R = 2^52`**, limbs `< 2^52`.

---

## 1. Middle-product decomposition

Product split: `a[i']·b[j'] = lo + R·hi`, `lo = (a·b) mod 2^52`, `hi = (a·b) >> 52`, both `< 2^52`.

Per **diagonal** `k = i'+j'`:

```
Sl[k] = Σ_{i'+j'=k} lo(a[i']·b[j'])          Sh[k] = Σ_{i'+j'=k} hi(a[i']·b[j'])
```

The middle product keeps diagonals `k ∈ [an-1, bn-1]` (`rn = bn-an+1` of them) plus a top carry.
Writing output index `p = k-(an-1)` (`p = 0 .. rn-1`, `r[rn]` = carry):

```
r[p] = Sl[an-1+p] + Sh[an-2+p]                        (†)
```

`Sh[an-2]` (the `p=0` term) is *below band* and must be dropped — that is the bottom boundary.

---

## 2. The shift-4 accumulation (the index arithmetic)

Process output in blocks of 8. Fix a block whose first output index is `i` (outputs `i..i+7`,
lane `L = 0..7`). Walk `a` top-down in blocks of 8: iteration variable `j = an, an-8, …`, with
`aptr = a + (j-8)` (so `aptr[aind] = a[j-8+aind]`). The b-window base for this a-block is

```
cur = i + (an - j)          (b0 = b[cur .. cur+7],  b1 = b[cur+8 .. cur+15])
bh  = alignr64(b1,b0,4)     (= b[cur+4 .. cur+11],  ONE shuffle)
```

The kernel accumulates each product into one of four `acc_lo[d]/acc_hi[d]` (`d=0..3`) with
either `b0` or `bh`:

```
mm_acc(d, aind, bvec):
    acc_lo[d] = madd52lo(acc_lo[d], splat(aptr,aind), bvec)
    acc_hi[d] = madd52hi(acc_hi[d], splat(aptr,aind), bvec)

# bh group (shift 4):  d=3←aind0, d=2←aind1, d=1←aind2, d=0←aind3
# b0 group (shift 0):  d=3←aind4, d=2←aind5, d=1←aind6, d=0←aind7
```

**Diagonal of each product.** For `b0`: `a[j-8+aind]·b[cur+L]`, diagonal
`= (j-8+aind) + (cur+L) = an-8+aind + i + L`. For `bh` (b index `+4`): `an-4+aind + i + L`.
Plugging the group assignments, **every product routed to `acc[d]` lands on the same diagonal**:

| `d` | diagonal of `acc[*][d][L]` | a-residue class it sums | via `b0` (`aind`) | via `bh` (`aind`) |
|----|------|------|----|----|
| 0 | `an-1 + i + L` | `i' ≡ (an-1) mod 4` | 7 | 3 |
| 1 | `an-2 + i + L` | `i' ≡ (an-2) mod 4` | 6 | 2 |
| 2 | `an-3 + i + L` | `i' ≡ (an-3) mod 4` | 5 | 1 |
| 3 | `an-4 + i + L` | `i' ≡ (an-4) mod 4` | 4 | 0 |

The shift-4 is exactly half of 8: it folds the two aind-groups `{0..3}` and `{4..7}` onto the same
four diagonals. Since `j` steps by 8, `j mod 4` is constant, so each `acc[d]` sums a **fixed
a-residue class** across the whole a-walk. Net result — a **polyphase decomposition**:

```
acc_lo[d][L] = Σ_{i' ≡ (an-1-d) mod 4}  lo(a[i']·b[k-i'])   for diagonal k = an-1-d + i + L
acc_hi[d][L] = (same with hi)
```

i.e. the 4 accumulators hold **four adjacent diagonals** (`an-1 … an-4`, offset by `-d`), each a
**quarter (one residue class)** of that diagonal's full `Sl/Sh`. The full diagonal is recovered by
summing the four quarters, which sit at **staggered lanes**.

---

## 3. The combine (recover `Sl`, `Sh`, then merge)

Full diagonal from the four quarters. `Sl[an-1+p]` needs `acc_lo[d]` at the lane where its diagonal
equals `an-1+p`, i.e. `an-1-d+i+L = an-1+p ⇒ L = (p-i)+d`. So for output lane `ℓ = p-i`:

```
Rl_vec[ℓ] := Sl[an-1+i+ℓ] = Σ_{d=0}^{3} acc_lo[d][ℓ+d]        (lane shifts 0,1,2,3)
Rh_vec[ℓ] := Sh[an-1+i+ℓ] = Σ_{d=0}^{3} acc_hi[d][ℓ+d]
```

Vectorized, `acc[d][ℓ+d]` is a down-shift by `d`; lanes `ℓ+d > 7` come from the **next** block's
`acc[d]` low lanes. With `cur[d]` = this block and `last[d]` = the block being finalized:

```
Rl_vec = Σ_{d=0}^{3} alignr64(cur_lo[d], last_lo[d], d)     # d=0 term is just last_lo[0]
Rh_vec = Σ_{d=0}^{3} alignr64(cur_hi[d], last_hi[d], d)
```

Then apply (†) — hi is one diagonal below lo, so shift `Rh_vec` up by one lane, lane 0 from the
previously finalized block:

```
r[block] = Rl_vec + alignr64(Rh_vec, prevRh, 7)            # prevRh = previous block's Rh_vec
```

Proof lane 0 is right: `Rh_vec_i[-1] = Σ_d acc_hi[d][d-1]` has diagonal `an-2+i` over all four
residues `= Sh[an-2+i] = Rh_vec_{i-8}[7] = prevRh[7]`. ✓

### One-block delay

`Rl_vec/Rh_vec` need the **next** block's acc, so finalize with a one-block lag:

```
for i = 0,8,…:  compute acc[*][0..3] for block i
                if i>0: finalize block (i-8) using last[*]=block(i-8), cur[*]=block i
                last[*] = block i;   prevRh = Rh_vec(i-8)
```

State carried between blocks: `last_lo[4], last_hi[4]` (8 vectors) + `prevRh` (1) — exactly the
declared `last_lo[4]/last_hi[4]` (`prevRh` is one extra). Combine cost/block: **7 `valignq` + 7
adds** (amortized over 8 outputs), vs `simple`'s `an` shuffles in the inner loop.

### Combine op budget vs `simple`

| per output block | `simple` | reduced |
|---|---|---|
| `valignq` | `an` (8 per a-block) | `an/8` (inner) + 7 (combine) |
| `madd52`  | `2·an` | `2·an` (identical) |
| `splat`   | `an`   | `an` |

For `an=64`: 64 → 15 shuffles. That is the whole point.

---

## 4. Boundaries

- **Bottom (block 0).** `last[*] = 0`, `prevRh = 0`. `Rl_vec[0] = Σ_d acc_lo[d][d]`, all diagonal
  `an-1` ⇒ full `Sl[an-1]`. `alignr64(Rh_vec,0,7)[0] = 0` ⇒ `Sh[an-2]` dropped. Matches (†) at `p=0`.
  The below-band quarters (`acc_lo[1..3]` at lane 0 = diagonals `an-2,an-3,an-4`) map to outputs
  `-1,-2,-3` and are simply never read.
- **a-tail (`an mod 8 ≠ 0`).** Handled *inside* the inner loop by the same `case j … case 1`
  fall-through as `simple` (see §5.1). No change needed.
- **Top.** The forward dependency means the **last block has no next** — see §5.2.

---

## 5. Tail (efficient)

### 5.1 a-tail — the switch (already efficient)

The bottom a-block has `j = an mod 8` valid limbs `a[0..j-1]`. The `switch(j)` fall-through runs
cases `j..1` → `aind = 8-j .. 7` → `a[0..j-1]`, using `bh` for cases `5..8` and `b0` for `1..4`.
Exactly the needed limbs, no waste. (The *first* a-block `j=an` takes `goto full`, all 8.) `b1`
for `bh` reads past `bn` on the bottom block → **pad `b` by ≥ 16 limbs** (same as `simple`).

### 5.2 output-tail — the last block is done vertically

Because the combine pulls the **next** block's low lanes, the last real block `I_last =
8·⌊(rn-1)/8⌋` cannot be finalized by the polyphase path (there is no `I_last+8`). Its missing
contributions are exactly the top diagonals near `bn-1` (the a-values whose residue quarter
"wrapped" into the phantom block). Computing a full phantom block would cost a whole extra block —
the staircase we want to avoid.

**Do the last block vertically instead** — reuse the validated `midmul_basecase_simple_wtail`
vertical top: for the last `t = rn - I_last` outputs (plus carry `r[rn]`), compute each output
diagonal as a straight dot product

```
S(an-1+p) = Σ_k arev[k]·b[(p) + k]      (arev = reversed a),   lo→r[p], hi→r[p+1]
```

merged with the carry-in `Sh[an-2+I_last]`, which the bulk already produced as
**`prevRh[7]`** (the last finalized block's `Rh_vec`, lane 7). This is the vertical kernel's
`last1`. Cost ∝ `t` (no phantom, no wasted lanes, and one batched transpose-reduce), so the tail is
a fixed small overhead independent of the staircase.

**Division of labour.** The bulk loop still *computes* `I_last`'s polyphase acc (needed to finalize
`I_last-8`), then the vertical step *finalizes* `I_last`. The last block is thus touched ~1.5×
(polyphase for `I_last-8`'s high lanes + vertical for its own outputs) — a constant tail cost.

> Design choice recap: **align-reduced bulk** (fewest shuffles, interior blocks) + **vertical last
> block** (no phantom, no staircase). They are complementary; the coupling is a single vector
> `prevRh` handed from bulk to tail.

---

## 6. Pseudocode

```c
static inline void mulmid_basecase(plimb r, const _limb *a, const _limb *b,
                                   int64_t an, int64_t bn) {
    const int64_t rn = bn - an + 1;               // outputs r[0..rn], length rn+1
    _vec accL[4], accH[4], lastL[4], lastH[4], prevRh = zero();
    for (int d=0; d<4; ++d) lastL[d]=lastH[d]=zero();

    // ---- accumulate one output block at output-index i -> accL/accH ----
    #define mm_acc(d,aind,bv) { \
        accL[d]=madd52lo(accL[d], splat_load(aptr,aind), bv); \
        accH[d]=madd52hi(accH[d], splat_load(aptr,aind), bv); }
    #define ACC_BLOCK(i) do {                                                  \
        const _limb *bptr=b+(i), *aptr; _vec b0,b1,bh;                         \
        for(int d=0;d<4;++d){accL[d]=zero();accH[d]=zero();}                   \
        b0=load_vec(bptr);                                                     \
        for(int64_t j=an;j>0;j-=8){                                           \
            aptr=a+(j-8); b1=load_vec(bptr+=8); bh=alignr64(b1,b0,4);          \
            if(j<=8) switch(j){                                               \
                full:                                                          \
                case 8: mm_acc(3,0,bh); case 7: mm_acc(2,1,bh);                \
                case 6: mm_acc(1,2,bh); case 5: mm_acc(0,3,bh);                \
                case 4: mm_acc(3,4,b0); case 3: mm_acc(2,5,b0);                \
                case 2: mm_acc(1,6,b0); case 1: mm_acc(0,7,b0);                \
            } else goto full;                                                  \
            b0=b1;                                                             \
        }                                                                      \
    } while(0)

    // ---- finalize block bi (outputs bi..bi+7) into r, using last=block bi, cur=accL/accH ----
    // Rl = Σ_d alignr(cur_lo[d], last_lo[d], d);  Rh similarly;  r = Rl + alignr(Rh,prevRh,7)
    #define FINALIZE(dst, mask) do {                                           \
        _vec Rl=lastL[0], Rh=lastH[0];                                         \
        Rl=add(Rl, alignr64(accL[1],lastL[1],1));                              \
        Rl=add(Rl, alignr64(accL[2],lastL[2],2));                              \
        Rl=add(Rl, alignr64(accL[3],lastL[3],3));                              \
        Rh=add(Rh, alignr64(accH[1],lastH[1],1));                              \
        Rh=add(Rh, alignr64(accH[2],lastH[2],2));                              \
        Rh=add(Rh, alignr64(accH[3],lastH[3],3));                              \
        store_vec((dst), add(Rl, alignr64(Rh, prevRh, 7)), (mask));           \
        prevRh=Rh;                                                             \
    } while(0)

    const int64_t Ilast = ((rn-1) & ~(int64_t)7);      // first output of last block
    // bulk: compute block i, finalize block i-8
    for (int64_t i=0; i<=Ilast; i+=8) {
        ACC_BLOCK(i);
        if (i>0) FINALIZE(r + (i-8), 0xff);            // full interior block
        for(int d=0;d<4;++d){lastL[d]=accL[d];lastH[d]=accH[d];}
    }
    // NOTE: after the loop, blocks 0..Ilast-8 are stored; prevRh = Rh_vec(Ilast-8),
    //       whose lane 7 = Sh[an-2+Ilast] = the carry-in for the last block.

    // tail: finalize the last block Ilast (outputs Ilast..rn-1 and carry r[rn]) VERTICALLY,
    //       with carry-in = prevRh (see §5.2). Reuse the wtail vertical-top kernel:
    midmul_tail_vertical(r, a, b, an, bn, /*i0=*/Ilast, /*carry=*/prevRh);
    #undef mm_acc
    #undef ACC_BLOCK
    #undef FINALIZE
}
```

`midmul_tail_vertical(...)` is `midmul_basecase_simple_wtail`'s vertical block, generalized to take
`i0` and the carry-in vector (its `hi1 = alignr64(zero(), carry, 7)` seed). It emits `r[i0..rn-1]`
and `r[rn]`.

Edge cases folded in above: `rn ≤ 8` (only the vertical tail runs, `last=prevRh=0` ⇒ correct bottom
boundary); `rn ≡ 1 (mod 8)` (`t=1`, cheapest vertical tail); `an ≤ 7` (`ACC_BLOCK`'s first
iteration is already the a-tail switch).

---

## 7. Expected payoff

Bulk shuffles drop ~8× (`an` → `an/8 + 7` per block). `madd52`/`splat` counts are unchanged, so if
`simple` is shuffle-port-bound this should recover most of the ~6% gap to `mul_basecase` and hold
throughput at ~14 ps/product. The vertical tail keeps the ns/limb curve off the staircase
(§5.2). Validate with the existing GMP harness (`value(r)` vs the exact band) and benchmark against
`simple`/`wtail` on `bn=2·an`.

Open knobs: (a) whether to keep `prevRh` or refold the lo/hi merge into per-`d` vectors `M[d] =
acc_lo[d] + alignr64(acc_hi[d],last_hi[d],7)` (saves the separate `Rh_vec` at the cost of a longer
dependency chain); (b) small-`an` gate for the vertical tail (same `an ≥ ~32` threshold measured for
`wtail`).
```
