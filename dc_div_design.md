# Divide-and-conquer division — design

Goal: turn the schoolbook `divrem_core` (O(n²), rank-1 `block_submul_vec` at
~0.39 cyc/u52²) into a recursive divider whose bulk work is **full
multiplication** (`mul_u52_dispatch`, ~0.17 cyc/u52² basecase, Karatsuba/Toom
beyond), recovering the ~4× mul advantage that the rank-1 submul strands at
~1.8×, and going sub-quadratic for large divisors.

Everything operates in the **u52 block domain**: 1 block = β = 2^416 = 8 u52
digits = one zmm. The divisor is normalized once by the u64 wrapper (top digit
bit 51 set); the recursion never re-normalizes. The "limb" of the recursion is a
block.

--------------------------------------------------------------------------------
## 1. Invariants & conventions

- Sizes are in **blocks** (8 u52 digits). Arrays are `_limb*` (u52 digits),
  block k at digit offset `8*k`.
- Divisor `B` is normalized: top digit (index `8*n-1`) has bit 51 set, so the
  top 2 blocks form an 832-bit value in [2^831, 2^832).
- One reciprocal `v = recip_3by2_seed(top 2 blocks of B)` threads through the
  whole recursion: every sub-divisor that appears (`B`, and `B1` = top half of
  `B`) shares B's top 2 blocks, hence the same `v`. No recomputation.
- Remainders are produced in place over the dividend storage where possible;
  quotients are written to a caller buffer. Scratch from the arena (`SALLOC`).

--------------------------------------------------------------------------------
## 2. Kernel inventory

### Already exist (reuse)
- `divrem_core(qp, np, nn, dp, dn, v)` — schoolbook block division. **The base
  case.** General (m+n)/n; in-place remainder in low `dn` blocks.
- `mul_u52_dispatch(r, a, b, an, bn, sc)` (mul.h:319) — full u52 multiply,
  routes basecase / Karatsuba / Toom by size. Returns redundant lanes.
- `block_add_n`, `block_addc`, `block_subb`, `block_ge`, `block_ge2`, `canon2`,
  `block_dec` (divrem.h / div.h).
- u64 boundary: `u52_from_u64_lsh`, `u52_rshift`, `u64_from_u52_canon`,
  `recip_3by2_seed`.

### New, small (add to divrem.h)
- `block_sub_n(np, dp, len) -> borrow` — multi-block subtract, mirror of
  `block_add_n` (uses `block_subb` per block, threads borrow).
- `block_cmp(a, b, len) -> {-1,0,1}` — compare two canonical len-block arrays,
  top block first (extends `block_ge`/`block_ge2`).
- (multiply) — **no new wrapper needed**: `mul_u52_dispatch_canon(r, a, b, an,
  bn, sc)` (mul.h:322) already does dispatch + class-1 canonicalize, returning a
  canonical (an+bn)-digit product. Caveats baked into the contract: `an`/`bn`
  are in **digits** (= 8·blocks), and **`a` must be the larger** operand (here
  Q and B0 are both h blocks → equal, fine).

### New, the recursion (new file `dc_div.h`, includes divrem.h)
- `dc_2n1n` — divide 2n by n (the workhorse).
- `dc_3n2n` — divide 3·(n/2) by n (the helper that issues the full mul).
- `dc_divrem` — general (m+n)/n dispatcher with the chunk loop + threshold.

--------------------------------------------------------------------------------
## 3. Recursive routines (Burnikel–Ziegler D2n1n / D3n2n)

Chosen over the fused GMP `dcpi1` form for clarity/verifiability; `dcpi1` is a
later micro-opt (saves a few block adds). Split convention for odd n:
`hi = (n+1)/2`, `lo = n/2` (B1 is the *high* half so it stays normalized).

### dc_2n1n(Q, A, B, n, v)  →  Q[n], R in A[0..n)
Pre: A is 2n blocks, B is n blocks normalized, `<A high n> < B` (⇒ Q < β^n).

```
if n <= DC_THRESHOLD:
    divrem_core(Q, A, 2n, B, n, v)         # base case; R lands in A[0..n)
    return
hi = (n+1)/2 ; lo = n/2
# A = [A3 A2 A1 A0] in lo-sized pieces (A3 may be hi-sized); B = [B1 B0]
(Q1, R1) = dc_3n2n(A[top 3 halves], B, n, v)   # Q1 = hi blocks
write R1 back over A's high region
(Q0, R ) = dc_3n2n([R1, A0], B, n, v)          # Q0 = lo blocks
Q = [Q1, Q0]                                   # n blocks
# R already in place in A[0..n)
```

### dc_3n2n(A, B, n, v)  →  Q[n/2], R[n]
Pre: A is 3·(n/2) blocks = [A2 A1 A0] (each n/2), B = [B1 B0] (each n/2),
`<A2,A1> < B` (⇒ Q < β^(n/2)). Let h = n/2.

```
if A2 < B1:                                    # block_cmp on h blocks
    (Q, R1) = dc_2n1n(<A2,A1>, B1, h, v)       # 2h / h ; Q,R1 = h blocks
else:                                          # saturation (rare)
    Q  = β^h - 1                               # all-ones h blocks
    R1 = <A2,A1> - Q*B1                         # = <A2,A1> - B1*β^h + B1, via add-back
D  = Q * B0                                     # h × h -> 2h blocks  [FULL MUL]
R  = <R1, A0> - D                               # 2h-block subtract, borrow b
while b:                                        # R<0 ⇒ Q too big; ≤ 2 iters
    Q = Q - 1
    R = R + B                                   # 2h-block add; carry cancels b
return (Q, R)
```

Mutual recursion `dc_2n1n(n) → 2·dc_3n2n(n) → 2·[dc_2n1n(n/2) + MUL(n/2)]`.
The `Q*B0` full multiply is the whole point: it replaces the level's rank-1
submul with one balanced multiply, and `mul_u52_dispatch` makes it
Karatsuba/Toom once h crosses its thresholds → sub-quadratic division.

--------------------------------------------------------------------------------
## 4. General dividend dispatcher

`dc_divrem(Q, A, m_plus_n, B, n, v)` for a general (m+n)-block dividend. Mirrors
GMP `mpn_dcpi1_div_qr`: peel the quotient in ≤ n-block chunks from the top so
every recursive call is a clean 2n/n.

```
dc_divrem(Q, A, an, B, n, v):
    if n <= DC_THRESHOLD or (an - n) <= DC_THRESHOLD:
        return divrem_core(...)                # whole thing schoolbook
    qn = an - n                                # quotient blocks
    # process top-down: maintain an n-block running remainder window
    if qn > n:
        # long dividend: repeated 2n/n on a sliding window, R fed down
        i = qn - n                             # first (partial) chunk
        dc_2n1n(Q+i, A[i .. i+2n), B, n, v)    # may need an initial cmp/sub for the top
        ... loop: i -= n; dc_2n1n(Q+i, [R, A_chunk], B, n, v) ...
    else:
        # qn <= n: single 2n/n on the top 2*qn-ish blocks, remainder is the rest
        dc_2n1n on the top (qn + n) window after the standard high-quotient cmp
    # R left in A[0..n)
```

Details to pin down at implementation time (all standard dcpi1 bookkeeping):
- The very first chunk handles the possible `<A top n> ≥ B` high-quotient block
  (one `block_cmp` + `block_sub_n`, contributing the top quotient block), exactly
  as `divrem_core`'s padding handles it now.
- Chunk size n keeps each `dc_2n1n` at the 2n/n shape its precondition needs.

--------------------------------------------------------------------------------
## 5. Integration with the u64 wrapper

`divrem_u64` is unchanged through normalization + decode + reciprocal. Only the
core call swaps:

```
- divrem_core((_limb*)Q52, (_limb*)N52, nn, (_limb*)D52, dn, v);
+ dc_divrem ((_limb*)Q52, (_limb*)N52, nn, (_limb*)D52, dn, v);
```

`dc_divrem` itself falls back to `divrem_core` below threshold, so small
divisions are byte-identical to today. The `v`, normalization, fused
decode/encode and `u52_rshift` denormalize all carry over verbatim.

--------------------------------------------------------------------------------
## 6. Thresholds

- `DC_THRESHOLD` (blocks): below this, `divrem_core` wins (recursion overhead +
  mul setup not amortized). Expect ~6–12 blocks (GMP `DC_DIV_QR_THRESHOLD` ≈ 50
  u64 limbs ≈ 8 blocks). Tune by sweeping the scaling bench at the crossover.
- The multiply inside `dc_3n2n` self-dispatches (basecase < 96 u52 ≈ 12 blocks,
  Karatsuba/Toom above) — no separate knob; the division inherits the mul's
  thresholds.
- Provide as a `#define` (mutable global under a tune build, like the
  `MUL_U52_T22_THRESHOLD` pattern in mul.h:21).

--------------------------------------------------------------------------------
## 7. Correctness & testing

Reuse the existing harnesses, extended to large/odd sizes that exercise the
recursion and both `dc_3n2n` branches:
1. `divrem_test` / `divrem_edge` vs GMP `mpz_tdiv_qr` — already cover dn≤~64;
   raise the size sweep to force ≥3 recursion levels and odd n.
2. Targeted: force the **saturation branch** (`A2 ≥ B1`) — construct
   dividends whose top half-block equals the divisor's, like the existing
   `n2==d1` forcing in `divrem_test`.
3. Differential: `dc_divrem` vs `divrem_core` on identical inputs at sizes above
   threshold (must be bit-identical Q and R) — isolates recursion bugs from
   reciprocal/convert bugs.
4. Run with `DC_THRESHOLD = 2` (force maximal recursion) to stress the recursion
   independent of the basecase.

--------------------------------------------------------------------------------
## 8. Scratch / memory

Per `dc_3n2n` frame: the product `D` (2h blocks) + the multiply's own scratch
(via `mul_u52_dispatch`'s `scratch*`). Quotient halves write into the caller's Q.
Use one `SCRATCH(sc)` scope at `dc_divrem` entry; recursion threads `sc`. Peak
extra space O(n) (geometric series of the D buffers) — same order as GMP.

--------------------------------------------------------------------------------
## 9. Expected payoff

- Mid sizes (dn 8–32, mul still basecase): full-mul vs rank-1 submul →
  division constant drops ~2× toward the mul floor; expect the full-vs-GMP
  ratio to climb from ~2× toward ~3×.
- Large sizes (dn ≥ ~24, mul = Karatsuba/Toom): division becomes sub-quadratic,
  so the ratio stops *decaying* at the top (today 2.3×→1.8× over dn 16→64) and
  instead holds/grows — this is the regime where GMP's own DC currently catches
  us.

Build order: (1) `block_sub_n`, `block_cmp` + unit tests (multiply reuses
`mul_u52_dispatch_canon`);
(2) `dc_3n2n`/`dc_2n1n` with `DC_THRESHOLD` huge (never taken) to wire types,
then small to exercise; (3) `dc_divrem` dispatcher + differential test;
(4) thresholds sweep; (5) swap into `divrem_u64`.
