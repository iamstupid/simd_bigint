# simd_bigint Algorithm Design And Implementation Audit

## Abstract

`experimental/simd_bigint` is a SIMD-first unsigned bigint experiment.  The
current implementation is no longer only a draft plan: the header tree contains
working kernels for vectorized limb addition and subtraction, base-`2^52`
canonicalization, `u64 <-> u52` conversion, IFMA basecase multiplication,
Karatsuba/Toom multiplication, a PQ complex FFT multiplier, two truncated NTT
engines, and a normalized block division prototype.

The implementation is intentionally not a GMP fork.  It follows several of
GMP's algorithmic decompositions, especially Toom evaluation/interpolation and
Knuth-style long division, but the representation, carry strategy, scratch
policy, and transform layouts are built around AVX-512 vectors.  This document
records the algorithms actually implemented under
`experimental/simd_bigint/include` as of this audit.

## Current Progress

The project currently has a complete experimental low-level multiplication
stack for many operand sizes:

- Tiny public `u64` multiplication is delegated to the declared
  `simd_mpn_mul_basecase_le6` assembly entry for operands up to six limbs.
- Medium public `u64` multiplication can dispatch to `fft16_mul` when the
  empirical FFT profitability rule fires.
- The default medium path decodes operands to base `2^52`, uses IFMA basecase,
  Karatsuba, Toom-32, Toom-33, Toom-42, or strip-mined Toom-42, and encodes the
  product back to `u64`.
- `pq16_mul_r` is a cleaner scratch-backed rewrite of the `fft16` PQ FFT
  algorithm.  It is implemented as a separate entry point but is not the FFT
  target used by `simd_mpn_mul`; `u64cvt.h` currently includes `fft16.h`.
- `p30_mul_r`, `p30_sqr_r`, `p50_mul_r`, `p50_sqr_r`, and the `ntmul.h`
  dispatcher are implemented as independent large-product NTT engines.  They
  are not yet wired into `simd_mpn_mul`.
- `divrem_u64` implements a normalized long-division path for divisors spanning
  at least two `u52` blocks, using a 3-by-2 reciprocal seed and quotient block
  correction.
- The support layer includes a two-tier scratch arena, vector intrinsic
  abstraction macros, u52 canonicalization, exact division by 3, folded
  interpolation canonicalization, and public `u64` add/subtract kernels.

Important incomplete or transitional areas:

- There is no high-level signed bigint object layer.
- Public ABI names are still experimental (`_limb`, `plimb`, `simd_mpn_*`,
  `mpn_u52_*`) rather than a final namespace.
- The division path handles the large normalized case but does not yet present
  a complete public division family for all sizes.
- FFT/NTT crossovers are partly measured but not unified into one public
  dispatch ladder.
- `p30_ref.h` is deliberately a reference superset with dormant validated
  paths; `p30.h` is the cleaner production NTT engine.

## Data Model

The low-level data model is unsigned magnitude arithmetic over little-endian
64-bit limbs.  The current headers define:

```c
typedef uint64_t _limb;
typedef uint32_t _hlimb;
typedef uint64_t *plimb;
typedef uint32_t *phlimb;
```

Public operands are pointer-plus-length arrays of `_limb`, with limb zero as
the least significant limb.  Internal SIMD multiplication uses vectorized
base-`2^52` digits, stored as eight 64-bit lanes per `__m512i`.  One vector
therefore carries 416 bits of payload.

The vector abstraction in `types.h` fixes the active vector width at 512 bits:

- `_vec` is `__m512i`, used for integer lanes.
- `_dvec` is `__m512d`, used for FFT double lanes.
- `pvec` and `cpvec` are vector pointers.
- The macro layer wraps masked and unmasked AVX-512 operations behind
  consistent names such as `load_vec`, `store_vec`, `add`, `sub`, `mul`,
  `and_v`, `or_v`, `srlv`, `sllv`, `fshl64v`, `fshr64v`, `madd52lo`, and
  `madd52hi`.

The vector layer also provides mask arithmetic and lane movement:

- mask shifts and boolean operations (`kadd`, `kand`, `kor`, `klsh`, `krsh`);
- unsigned compare masks (`ltu`, `gtu`, `eq`, `neq`);
- byte, word, dword, and qword blends;
- ternary logic helpers for common boolean formulas;
- lane shifts/rotates via `alignr64`;
- VBMI/VBMI2 permutations and funnel shifts.

This layer is intentionally macro-heavy.  The implementation favors generating
direct intrinsics at each call site over introducing an abstraction that hides
mask operands or vector type selection from the compiler.

## Scratch Allocation

`scratch.h` implements a header-only two-tier bump allocator:

- A small "stack" tier routes allocations smaller than `SCRATCH_PAGE`.
  It is 64-byte aligned and defaults to a 256 MiB reservation.
- A large "heap" tier routes allocations at least `SCRATCH_PAGE`.
  It is page-aligned, defaults to a 64 GiB reservation, and may be THP-backed.
- Both tiers are `mmap` reservations.  The default uses `MAP_NORESERVE`; an
  optional `SCRATCH_EXPLICIT_COMMIT` mode reserves with `PROT_NONE` and commits
  with `mprotect` as the bump pointer grows.
- `SCRATCH(s)` is a scope guard.  It records both bump pointers and restores
  them on scope exit.
- AddressSanitizer integration poisons restored ranges, so use-after-scope
  scratch bugs are more likely to be detected.
- `salloc_zeroed` avoids clearing pages that have never been dirtied because
  anonymous pages already fault in as zero.

The multiplication and transform engines rely on this allocator for recursive
Toom temporaries, FFT spectra, and NTT residue buffers.  The policy is central:
large kernels use explicit short-lived workspaces without hidden heap traffic
inside hot loops.

## Carry And Canonicalization

### `u64` Add/Subtract

`addsub_n.h` implements public `u64` vector addition and subtraction through a
shared template in `addsub_n_impl`.

For a vector addition:

1. Compute `res = a + b` lane-wise.
2. Generate a carry mask with `res < a`.
3. Generate a propagate mask with `res == UINT64_MAX`.
4. Inject the incoming carry at lane zero.
5. Use one integer mask addition to ripple generate/propagate through the
   eight lanes.
6. Add one to lanes that received a carry.
7. Return the carry out of the active lane count.

Subtraction is the dual:

1. Compute `res = a - b`.
2. Generate borrow with `res > a`.
3. Generate propagate with `res == 0`.
4. Mask off inactive tail lanes.
5. Ripple the incoming borrow through active lanes.
6. Subtract one from lanes receiving a borrow.

The template has scalar fast paths for one to three limbs, one-vector masked
paths through seven limbs, two-vector handling through fifteen limbs, and an
unrolled four-vector loop for larger sizes.

### Non-Canonical `u52` Add/Subtract

`addsub_nc_impl` implements non-canonical base-`2^52` vector add/subtract used
inside interpolation.  It performs plain lane-wise addition or subtraction,
preserves redundant lanes, and masks only the final vector.  This is not a
carry-propagating arithmetic primitive; it is a deliberate redundant-format
building block.

### Positive `u52` Canonicalization

`canon.h` defines the main `canonize` macro for positive redundant base-`2^52`
digits.

For each vector:

1. `hi = vec >> 52` captures per-lane overflow.
2. `his = alignr64(hi, carry, 7)` shifts the previous lane's overflow into the
   current lane; lane zero receives the previous vector's lane-seven overflow.
3. `t = (vec & MASK52) + his` forms a one-level carry-absorbed value.
4. A generate mask marks `t > MASK52`.
5. A propagate mask marks `t == MASK52`.
6. `chain = propagate + ((generate << 1) | scalar_carry)` performs the
   eight-lane carry ripple in a scalar mask register.
7. Lanes selected by `propagate ^ chain` receive the carry correction and are
   masked back to 52 bits.

The return state is split between a vector carry (`carry = hi`) and a scalar
carry bit (`sigcarry`) that links vector blocks.

`mpn_canon_pos`, `mpn_canon_pos_c`, and `mpn_u52_add_canon` are positive
canonicalization passes and fused add-canonicalize passes.

### Signed/Negative `u52` Canonicalization

Toom interpolation often creates signed redundant lanes.  `canon.h` implements
`canonneg`, a dual-chain normalizer that simultaneously handles:

- lane borrow created by negative low 52-bit parts;
- lane carry from high bits above bit 51;
- propagation across vectors.

The macro tracks `bw_v` for borrow bits, `hi_v` for high carry fragments,
`bin` for scalar borrow, and `cin` for scalar carry.  A lane is first adjusted
for the borrowed low word, then receives high carry from the previous lane.
This permits signed redundant values to be carried into canonical non-negative
base-`2^52` form in one streaming pass when the whole region is known to be
non-negative.

The tail-aware entry `mpn_canon_neg_tail` is critical: it reads and writes
exactly `n` live u52 limbs, masking the final vector so unrelated lanes beyond
the region cannot be accidentally absorbed into the carry chain.

### Folded Canonicalization

The Toom code avoids standalone canonicalization passes by folding the final
interpolation operation into canonicalization:

- `mpn_u52_fold_add_canon`
- `mpn_u52_fold_sub_canon`
- `mpn_u52_fold_addsub_canon`

These functions compute `r = canonneg(r +/- x)` or
`r = canonneg(r + x - y)` over a full mixed region.  The contract is strong:
the span must extend to the top of the value so the final carry/borrow is
genuinely zero by whole-value non-negativity.  A mid-number fold would drop a
live boundary carry.

This design is one of the main departures from GMP-style scalar carry
threading.  It reduces interpolation traffic by making the last arithmetic
application also resolve the signed redundant representation.

### Exact Halving

The code has two halving families:

- canonical halving in `mpn_u52_add_rsh1_canon` and
  `mpn_u52_sub_rsh1_canon`;
- redundant signed halving in `mpn_u52_add_rsh1_nc`,
  `mpn_u52_sub_rsh1_nc`, and `mpn_u52_subadd_rsh1_nc`.

Canonical halving first resolves carries or borrows, then shifts a canonical
base-`2^52` stream right by one digit-bit using a cross-lane bit transfer.

Redundant signed halving uses the identity:

```text
r_i = arithmetic_shift_right(d_i, 1) + ((d_{i+1} & 1) << 51)
```

where `d` is a lane-wise signed redundant value.  It performs no carry
resolution and is therefore used only where the consumer can tolerate redundant
lanes.

### Exact Division By 3

`mpn_u52_divexact3` divides a canonical base-`2^52` vector stream exactly by
three.  It uses:

- lane residues modulo 3 from popcounts of alternating bit masks, since
  `2^i = (-1)^i mod 3`;
- in-vector inclusive prefix sums to determine borrow residues;
- the modular inverse `3^-1 mod 2^52`;
- IFMA `madd52lo` to multiply by that inverse;
- a vectorized inter-block residue carry, splatting lane seven into the next
  vector.

The hot loop is four-vector unrolled.  This routine is used by shared Toom
5-point interpolation.

## `u64 <-> u52` Boundary

`u64cvt.h` implements conversion and the public multiplication entry
`simd_mpn_mul`.

The key layout fact is:

```text
8 u52 digits = 416 bits = 52 bytes
```

That means every u52 vector maps to a byte-aligned 52-byte window in the packed
`u64` representation.

### Decode

`u52_from_u64` computes the input bit length, derives the required number of
u52 digits, then decodes each 52-byte block:

1. Load up to 64 bytes, using masked loads near the end.
2. Use a fixed `vpermb` table so each lane gathers the eight-byte window that
   contains one 52-bit digit.
3. Shift odd digits right by four bits.
4. Mask every lane with `2^52 - 1`.
5. Store full vectors; lanes beyond the logical digit count are zero-padded by
   construction or caller slack.

`u52_from_u64_lsh` is the division-specific variant.  It fuses decode with a
left normalization shift, splitting the shift into a whole-u52-digit offset and
a subdigit funnel shift.

### Encode

Encoding is fused with canonicalization in `canon.h`:

- `u64_from_u52_canon` canonicalizes a non-negative redundant u52 product and
  packs it to `u64`.
- `u64_from_u52_canonneg` handles a signed-redundant class-1 top-level result
  using the negative canonicalization chain while packing.

Packing uses two byte permutes.  A canonical u52 digit has byte seven clear, so
that byte acts as a zero filler during the 52-byte block repack.

## Multiplication Dispatch

`simd_mpn_mul` is the public low-level multiplication entry in `u64cvt.h`.

The current order is:

1. Swap operands so `an >= bn`.
2. If `an <= 6`, call `simd_mpn_mul_basecase_le6`.
3. If FFT is enabled and the empirical FFT profitability rule says yes, try
   `fft16_mul`.
4. Allocate scratch, decode both operands to u52 vectors, run
   `mul_u52_dispatch`, zero padding lanes needed by the encoder, and encode
   with positive or negative canonicalization depending on the returned output
   class.

The FFT profitability table is indexed by the smaller operand rounded in steps
of eight limbs.  The rule models FFT cost as proportional to `an + bn` and the
Toom route as approximately `S(bn) * an` for rectangular inputs.  The FFT path
is only considered below the `fft16` centered-band cap.

## Tiny Scalar Multiplication

`mul_basecase_le6.h` declares two assembly kernels:

- `simd_mpn_mul_basecase_le6`
- `simd_mpn_addmul_basecase_le6`

They cover public `u64` operands where both lengths are between one and six
limbs.  The design intent is a scalar `mulx/adcx/adox` Comba front end before
IFMA decode/setup becomes profitable.  The header defines the ABI and
non-overlap contract; the implementation lives outside the include tree.

## IFMA Basecase Multiplication

The IFMA basecase lives mainly in `mul.h`, with single-vector helper kernels in
`mul_vec.h`.

The base representation is u52.  Every vector has eight digits.  IFMA gives:

- `madd52lo(acc, a, b)` for low 52-bit product accumulation;
- `madd52hi(acc, a, b)` for high 52-bit product accumulation.

### Specialized `bn = 1` And `bn = 2`

`mul_u52_bn1` multiplies an arbitrary-length u52 operand by one u52 digit:

- each vector of `a` is multiplied by the splatted digit;
- low products are combined with the previous vector's high products shifted by
  one lane;
- the tail stores `tail + 1` digits.

`mul_u52_bn2` tracks two product diagonals using `pMid` and `pH1`.  It emits
one result vector per input vector and stores up to two high tail digits.

### `bn <= 8`

`mul_u52_1` handles multipliers up to one full vector.  It splats each digit of
`b`, uses a Duff-style fall-through switch to accumulate live diagonals, and
rotates high products through `m[0..7]` so that every output vector receives
the correct shifted contribution.  The tail path writes either one vector or a
final partial high vector depending on `tail + bn`.

### General Basecase

`mul_u52_basecase` handles rectangular `an >= bn`.  It is a vector Comba
algorithm:

1. Treat each operand as groups of eight u52 digits plus a final partial group.
2. Walk product diagonals over vector blocks.
3. For each block pair, apply `madd52lo` and `madd52hi` into nine accumulator
   vectors.
4. Shift high halves into their diagonal offsets with `alignr64`.
5. Harvest one output vector per diagonal.
6. Finish the tail by rotating partial vectors and storing either one or two
   vectors depending on the final live digit count.

This basecase emits redundant u52 lanes.  Full canonicalization is deferred to
Toom folds or to the public encode boundary.

`mul_vec.h` contains older or auxiliary single-vector multiply routines,
including `mul_u52_vec` and `mul_u52_vec_fast`.  The fast variant adds a
common-case canonicalization path: if no lane overflows after absorbing the
previous high lane, it injects the scalar carry into lane zero and avoids the
full SWAR ripple.

## Toom And Karatsuba Multiplication In u52

`mul.h` implements a shape-aware multiplication hierarchy.  The dispatcher
returns an output class:

- class 0: canonical or non-negative redundant within the basecase lane bound;
- class 1: composite signed-redundant lanes, requiring a consumer fold or
  negative canonicalization at the boundary.

`mul_u52_dispatch_canon` forces class 0 by running `mpn_canon_neg_tail` when a
child returns class 1.

### Karatsuba / Toom-22

`mul_u52_karatsuba` splits the larger operand at `an / 2` digits.  If the
smaller operand does not reach the high half, it falls back to basecase.
Otherwise it computes:

- `point_0 = a0 * b0`;
- `point_inf = a1 * b1`;
- `point_n1 = |a0 - a1| * |b0 - b1|`;
- the middle product as `point_0 + point_inf -/+ point_n1`.

The middle recomposition is non-canonical when children are class 0.  If any
child is class 1, the final application of `point_n1` is folded through
`mpn_u52_fold_add_canon` or `mpn_u52_fold_sub_canon`.

The implementation carefully returns the correct class after a fold: the low
region retains `point_0`'s class, so the output is not automatically class 0
just because the high mixed span was folded.

### Toom-32

`mul_u52_toom32` handles shapes roughly `bn < an < 3*bn`, especially around
the 3:2 ratio.  It uses GMP-style points `{0, +1, -1, inf}`.

Operand split:

```text
A(x) = a0 + a1*x + a2*x^2
B(x) = b0 + b1*x
```

Evaluations:

- `A(1) = a0 + a1 + a2`
- `A(-1) = a0 - a1 + a2`, stored as magnitude plus sign flag
- `B(1) = b0 + b1`
- `B(-1) = b0 - b1`, stored as magnitude plus sign flag
- `A(0)B(0)` and `A(inf)B(inf)`

The interpolation relies on redundant halving:

- `W = (v1 + signed(vm1)) / 2`
- construct the middle coefficient groups from `W`, `vm1`, `v0`, and `vinf`
- place the final low-middle contribution using `mpn_u52_fold_add_canon`

The scratch layout deliberately reuses output regions after their evaluation
operands are dead.

### Shared 5-Point Interpolation

`u52_toom_interp_5pts` is shared by Toom-33 and Toom-42.  It follows GMP's
`toom_interpolate_5pts` order over points `{0, +1, -1, +2, inf}` while using
redundant arithmetic:

1. Canonicalize `v2 +/- vm1`, then exact-divide by 3.
2. Compute `(v1 -/+ vm1) / 2` in redundant form.
3. Compute a fused `t2 = (v2 - (v1 - v0)) / 2`, either redundant or canonical
   depending on whether high lanes fit.
4. Derive `c4 + c2`, then subtract `2*vinf` to obtain `c3`.
5. Subtract `vinf` to obtain `c2`.
6. Restore parked `vinf` lanes.
7. Place `c3` by non-canonical add.
8. Fold `c1 = tm1 - c3` into the output span with
   `mpn_u52_fold_addsub_canon`.

The final fold is the only signed canonicalization pass for the mixed span.

### Toom-33

`mul_u52_toom33_n` handles near-balanced 3-way splits:

```text
A(x) = a0 + a1*x + a2*x^2
B(x) = b0 + b1*x + b2*x^2
```

It evaluates:

- `A(0), B(0)`
- `A(1), B(1)`
- `A(-1), B(-1)` as magnitudes plus sign
- `A(2), B(2)` using `mpn_u52_eval2_canon`
- `A(inf), B(inf)`

The products are computed in a GMP-inspired order so temporary buffers in the
output region die exactly when their product storage is needed.  The shared
5-point interpolation then assembles the result.

### Toom-42

`mul_u52_toom42` handles shapes around `an ~= 2*bn`:

```text
A(x) = a0 + a1*x + a2*x^2 + a3*x^3
B(x) = b0 + b1*x
```

It uses the same point set and shared interpolation as Toom-33.  Evaluation
uses:

- `p13 = a1 + a3`
- `p02 = a0 + a2`
- `A(1) = p02 + p13`
- `A(-1) = p02 - p13`
- `A(2) = a0 + 2*a1 + 4*a2 + 8*a3`
- `B(1), B(-1), B(2)`

`mpn_u52_eval2_canon` computes the `x = 2` evaluation for both 3-part and
4-part polynomials with a streaming carry chain.

### Rectangular Strip-Mining

For extreme imbalance, `mul_u52_stripmine` multiplies the larger operand in
strips against the smaller operand.  The normal strip is `2*bn x bn` using
Toom-42.  The overlap seam is just a non-canonical add over `bn` digits plus a
copy of the non-overlapping high region.

There is a sliver guard: if `an mod 2*bn <= 8`, the code derives equalized
strip widths from `an` so it avoids a near-empty tail strip.

### Thresholds

`u52-mparam.h` contains tuned thresholds:

- `MUL_U52_T22_THRESHOLD = 96`
- `MUL_U52_T33_THRESHOLD = 618`
- `MUL_U52_T32_OK_THRESHOLD = 108`
- `MUL_U52_T42_OK_THRESHOLD = 132`

In `U52_TUNE_BUILD`, these become mutable globals so the tuner can recurse
through candidate thresholds.

The dispatcher logic is:

- below `T22`, use IFMA basecase;
- below `T33`, use the ToomX2 band with Karatsuba, Toom-32, Toom-42, or
  strip-mining depending on ratio;
- at and above `T33`, use Toom-33, Toom-32, Toom-42, or strip-mining;
- fall back to Karatsuba when shape predicates reject the ideal Toom form.

## FFT Multiplication

There are two PQ complex FFT implementations.

### `fft16.h`

`fft16.h` is the active FFT header included by `u64cvt.h`.

Representation:

- Inputs are viewed as base-`2^16` digits, four per `u64` limb.
- The PQ trick uses `2*(an + bn)` complex points.
- Tiles are AoSoV: eight complex numbers per pair of ZMM registers,
  `[re0..7 | im0..7]`.
- Forward transform is DIF with root `exp(+2*pi*i/N)`, natural input, and
  bit-reversed output.
- Leaves intentionally store a "pi layout", where the transpose-back is
  skipped.  The pointwise table is generated in that layout and inverse leaves
  consume it directly, so the omitted transposes cancel.

Transform structure:

1. Fused input decode plus radix-2^2 first stage.
2. Radix-2^3 ancestor stages.
3. Leaves of size 32x2, 64, or 128 according to transform size parity.
4. PQ pointwise multiplication over mirror pairs.
5. Inverse leaves fused into pointwise while data is cache-hot.
6. Inverse radix-8 stages.
7. Final inverse radix-2^2 fused with rounding, base-`2^64` carry, and output
   emission.

Power-of-two and Good-Thomas PFA sizes are supported.  The PFA branch factors
are `M in {3, 5, 7}`; they fill the octave between powers of two and reduce
padding.  Branch 0 is self-paired, and branch pairs `(b, M-b)` use rotated PQ
twiddles.

Precision bands:

- Uncentered transforms are capped by empirical adversarial probes.
- Above that band, centered digits `d - 2^15` are used up to
  `F16_MAX_N_C = 2^19` complex points.
- Centered output corrects signed convolution coefficients with:

```text
c_k = c_hat_k + (S_k << 15) - (C_k << 30)
```

where `S_k` is a running digit-window sum and `C_k` is the overlap count.
Segmented correction paths reduce loads in rising, middle, and falling
regions.

Final emit:

- uncentered power-of-two uses four quarter-front carry chains, then junction
  fixups;
- uncentered PFA uses one carry chain per branch region;
- centered output uses a flat natural-order emit for prefetch friendliness in
  the bandwidth-bound band.

### `pq16.h`

`pq16.h` is a cleaned-up rewrite of the same algorithm family.  It shares the
same main ideas but improves integration with the rest of the project:

- per-call spectra come from `scratch.h`;
- persistent twiddle tables live in a thread-local `pq16_plan`;
- both spectra are manually 128-byte aligned inside one scratch allocation;
- the centered band stages operands in scratch, where the centering XOR also
  normalizes memory layout;
- the code uses the common `_vec`/`_dvec` macro layer from `types.h`.

It implements:

- radix-2^2 and radix-2^3 butterflies;
- 8x8 transposes and DFT8 leaves;
- compact twiddle tables for large stages;
- three-tier recursive blocking for cache residency;
- Good-Thomas PFA for `M = 3, 5, 7`;
- PQ pointwise self-pairs and cross-pairs;
- fused inverse leaves;
- uncentered fused emit and centered flat emit.

`pq16_mul_r` returns `1` on success and `0` outside the supported band.  It is
implemented but not currently selected by `simd_mpn_mul`.

## NTT Multiplication

The include tree contains two integer NTT engines plus a dispatcher.

### `p30.h`

`p30.h` is the production 5-prime, 30-bit NTT engine.

Representation:

- One `u64` limb is one NTT coefficient.
- Sixteen coefficients are packed per ZMM vector (`P30_W = 16`).
- Five primes are used, in ascending order for Garner reconstruction:

```text
918552577, 935329793, 943718401, 985661441, 998244353
```

The supported binary-transform band is total limbs up to `2^22`, with a CRT
minimum-operand cap of `2,344,425` limbs.

Pipeline:

1. Convert limbs to residues for each prime.  Full vectors share limb loads
   across all five primes and use a fused REDC-style input transform:
   `(hi*R^2 + lo*R)*R^-1`.
2. Run a van der Hoeven truncated NTT on vectors, with sixteen scalar TFTs in
   lockstep.
3. Use FLINT's twiddle convention: node twiddle `w2[j] = w[2j]`.
4. Store prefix-stable bit-reversed twiddle tables as Barrett `(c, rec)` pairs.
5. Run full subtrees through an iterative DFS flattened into a leaf-block walk.
6. Keep the truncation spine radix-2; no truncated radix-8 moth variants are
   used.
7. Fuse `forward(b) + pointwise + inverse` so the `b` leaf block transforms,
   convolves with the matching `a` spectrum block, and inverse-transforms while
   cache-hot.
8. Carry inverse-tail side data in the dead tail of the `a` spectrum,
   positions `[tv, M)`.
9. Scale each residue by `NV^-1 * R`, canceling the pointwise REDC stray
   factor.
10. Run Garner CRT and compose to `u64` limbs.

Modular arithmetic:

- Between-pass values are lazy `[0, 4p)`.
- Barrett multiplication accepts any `u32` lane as input and returns `[0, 2p)`.
- The unmultiplied butterfly operand is shrunk once per level.
- Pointwise convolution uses a 16-point twisted cyclic convolution modulo
  `x^16 - w[k]`.
- The production pointwise kernel is parity-split Karatsuba over two spectral
  positions per call.

Output:

- Garner runs in batched step-major order through a 256-coefficient L1 buffer.
- Mixed-radix digits are composed into three limb-aligned streams.
- Two SWAR carry chains combine those streams into final `u64` limbs.
- A scalar tail handles leftovers and flushes delayed high streams.

Mixed ladder:

- One measured octave uses a mixed `3*2^k` transform.
- A separate prime set supports `3*2^20 | p-1` for radix-3 twists.
- Children are twisted to cyclic form by substituting `x -> nu*x`.
- The mixed ladder is gated by default to the L3-crossing window where it was
  measured to win.

`p30_sqr_r` specializes squaring by using one input pass and fused traversal
with `sq = 1`, so each convolution leaf multiplies against itself.

### `p30_ref.h`

`p30_ref.h` is the reference superset of `p30.h`.  It retains validated but
dormant experimental paths:

- standalone general truncated inverse routines;
- mixed fused-input variants;
- pre-depth drivers for those variants.

It is intended for internal benchmarks and regression tests.  New production
work should go into `p30.h`.

### `p50.h`

`p50.h` is the 4-prime, 50-bit NTT engine using AVX-512 IFMA52.

Representation:

- Coefficients are variable-width "trunks" with `T in {88, 84, 80}` bits.
- The implementation prefers the largest `T` whose minimum-operand cap fits.
- Eight coefficient lanes are stored per ZMM vector (`P50_W = 8`).
- The four primes are about 50 bits and ordered for Garner reconstruction.

The skeleton follows the p30 truncated-TFT structure:

- prefix-stable bit-reversed twiddle tables;
- lazy `[0, 4p)` values;
- radix-8 full-node moths;
- in-register 16/32-vector bottoms;
- fused forward/pointwise/inverse traversal;
- inverse-tail side channel in the `a` spectrum dead tail.

Arithmetic differences:

- Barrett52 reduction uses IFMA (`madd52hi`, `madd52lo`) with base `2^52`.
- Wide REDC52 reduces 104-bit product columns.
- Pointwise convolution is an 8-point schoolbook twisted cyclic convolution
  modulo `x^8 - w[k]`.  Comments note that Karatsuba lost under IFMA because
  fused multiply-add made the product columns cheap.
- Inputs and outputs use table-driven trunk extraction and regrouping.

Input:

- For full blocks, vector funnel shifts extract low and high trunk pieces from
  caller limbs.
- The high 52-bit piece is multiplied by `2^T mod p`, REDC-reduced, then the
  low piece is added.
- Ragged tails are handled through a stack-padded scalar path to avoid
  over-reading caller memory.

Output:

- Four residue arrays are converted by trunk-radix Garner.
- IFMA composes CRT digits in radix `2^52`.
- The result is regrouped into radix `2^T`, then carried by SWAR trunk chains.
- A final table-driven funnel regroup writes `u64` limbs.

`p50_mul_r` and `p50_sqr_r` are implemented.  Compared with p30, p50 currently
does not include the p30 fused input column pass or the mixed `3*2^k` ladder.

### `ntmul.h`

`ntmul.h` dispatches between p30 and p50.

The selection rule is measurement-derived:

- p50 wins for small in-cache totals;
- p30 wins deep in DRAM while inside its CRT and transform caps;
- in the seam band, the rule compares transform fill ratios because p50 and
  p30 step at different transform sizes.

`nt_pick50` computes a total-limb window.  Below `NT_Z_LO`, it chooses p50.
Above `NT_Z_HI`, it chooses p30 first.  In between, it compares p50 and p30
transform occupancy and picks the fuller transform.

The public helpers `nt_mul_r`, `nt_sqr_r`, `nt_mul`, and `nt_sqr` try the
preferred engine and fall back to the other engine if the preferred one reports
out of band.

## Division

Division is implemented in two layers: a 3-by-2 block reciprocal path in
`div.h`, and a full `u64` wrapper in `divrem.h`.

### Reciprocal Kernels

`recip_3by2.h` declares an AVX-512 assembly reciprocal:

- input: a normalized 13-limb, 832-bit divisor block `I`;
- output: one u52 vector `V`;
- meaning:

```text
V = floor((2^1248 - 1) / I) - 2^416
```

The implicit leading base `B = 2^416` is dropped.  This is the
Moeller-Granlund 3-by-2 block reciprocal for dividing a 3-block value by a
2-block divisor.

`recip_3by2_seed.h` declares a stripped seed variant returning `V-1` or `V`.
`divrem.h` uses the seed and lets quotient correction absorb the possible
one-unit reciprocal error.

### Block Arithmetic

`div.h` defines one block as:

```text
beta = 2^416 = 8 u52 digits
```

The helper type `_u832` stores two blocks as `{lo, hi}`.  The block helpers
include:

- `mul_zmm`: one block times one u52 vector;
- `mul_zmm_2`: two blocks times one u52 vector;
- `canon2`: canonicalize a redundant two-block product;
- `block_addc`: block add with carry;
- `block_subb`: block subtract with borrow;
- `block_ge` and `block_ge2`: canonical block comparisons.

### `div3by2`

`div3by2` implements Edamatsu Algorithm 3 at block granularity:

```text
<u2,u1,u0> / <d1,d0>
```

Preconditions:

- `beta/2 <= d1 < beta`;
- `<u2,u1> < <d1,d0>`;
- `v` is the 3-by-2 reciprocal of `<d1,d0>`.

High-level steps:

1. Estimate `<q1,q0> = v*u2 + <u2,u1>`.
2. Canonicalize the two-block estimate.
3. Increment `q1` and detect overflow.
4. If overflow occurred, form the special remainder path.
5. Otherwise subtract `<d1,d0>*q1` from `<u1,u0>`.
6. If `r1 >= q0`, decrement `q1` and add back the divisor.
7. If `<r1,r0> >= <d1,d0>`, increment `q1` and subtract the divisor.

The result is one quotient block and a two-block remainder.

### Long Division Core

`divrem_core` in `divrem.h` is a normalized Knuth-style long division over
base `beta = 2^416`.

Inputs:

- `np`: `nn` dividend blocks, modified in place;
- `dp`: `dn` normalized divisor blocks, `dn >= 2`;
- `v`: reciprocal seed of the top two divisor blocks;
- `qp`: quotient output of `nn - dn` blocks.

For each quotient position from high to low:

1. Load the top dividend block `n2`.
2. If `n2 == d1`, use the rare overflow case with `qhat = beta - 1` and
   subtract the full divisor.
3. Otherwise call `div3by2` on the top three dividend blocks and top two
   divisor blocks.
4. Subtract `qhat * lower_divisor` using `block_submul_vec`.
5. Borrow from the returned product carry into the two-block remainder.
6. While a borrow remains, decrement `qhat` and add the divisor back.
7. Store the corrected quotient block.

`block_submul_vec` is a fused streaming submul.  It mirrors the IFMA Comba
diagonal schedule used in multiplication, but each product block is
canonicalized and immediately subtracted from the dividend block with a borrow
chain.  The returned high block is the carry/borrow contribution beyond the
processed span.

### `divrem_u64`

`divrem_u64` wraps the block core for packed `u64` operands.

It:

1. Computes bit lengths and handles `N < D`.
2. Chooses `dn = ceil(bitlen(D) / 416)`.
3. Computes a normalization shift `s = 416*dn - bitlen(D)`.
4. Allocates u52 block buffers from scratch.
5. Decodes and left-shifts `D` and `N` in one pass using `u52_from_u64_lsh`.
6. Builds the 13-limb normalized reciprocal seed input from the top bits of
   the original divisor.
7. Calls `recip_3by2_seed`.
8. Runs `divrem_core`.
9. Converts the quotient from u52 to u64.
10. Converts the normalized remainder from u52 to u64 and right-shifts by `s`.
11. Trims logical quotient and remainder lengths.

The current wrapper requires the divisor to span at least two `u52` blocks
(`bitlen(D) > 416`) and uses thread scratch.

## Header-by-Header Inventory

This section records each include file's role.

`types.h`:

- defines scalar limb aliases and AVX-512 vector aliases;
- provides masked/unmasked vector load/store, integer/double arithmetic,
  comparisons, logical ops, lane movement, byte permutations, ternary logic,
  IFMA helpers, and scalar `adc`/`sbb` wrappers;
- defines `MASK52`.

`scratch.h`:

- implements the two-tier mmap bump allocator, scope marks, thread-local
  default scratch, ASan poisoning, zeroed allocation, trimming, and mlock
  helpers.

`addsub_n.h`, `addsub_n_impl`, `addsub_nc_impl`:

- implement public `u64` add/sub with vector carry/borrow propagation;
- instantiate non-canonical u52 add/sub helpers for interpolation.

`canon.h`:

- implements positive and signed u52 canonicalization;
- implements folded add/sub canonicalization;
- implements exact halving;
- implements exact division by 3 support macros;
- implements fused canonicalize-and-encode to packed `u64`;
- implements `mpn_u52_eval2_canon`.

`canon_impl_deprecated`:

- retains older positive canonicalization templates; marked deprecated.

`interp_helpers.h`:

- implements u52 masks, compare-and-subtract by magnitude, add-with-carry,
  non-canonical add/sub variants, redundant halving helpers, three-operand
  interpolation helpers, and exact division by 3.

`mul_basecase_le6.h`:

- declares tiny scalar assembly multiplication and addmul kernels for operands
  up to six `u64` limbs.

`mul_vec.h`:

- implements auxiliary one-vector IFMA multiplication and fast
  canonicalization variants.

`mul.h`:

- implements u52 IFMA basecase, Karatsuba/Toom-22, Toom-32, Toom-33, Toom-42,
  shared 5-point interpolation, rectangular strip-mining, output class
  tracking, and shape-aware dispatch.

`u52-mparam.h`:

- stores tuned u52 multiplication thresholds.

`u64cvt.h`:

- implements `u64` bit length, `u64 -> u52` decode, FFT profitability, and
  the public `simd_mpn_mul` entry.

`fft16.h`:

- implements the active PQ complex FFT multiplier used by `simd_mpn_mul`.

`pq16.h`:

- implements a scratch-backed cleaned-up PQ complex FFT multiplier with its
  own public entry.

`p30.h`:

- implements the production 5-prime 30-bit truncated NTT multiplier and
  squarer.

`p30_ref.h`:

- implements the p30 reference superset and dormant validated experimental
  paths.

`p50.h`:

- implements the 4-prime 50-bit IFMA truncated NTT multiplier and squarer.

`ntmul.h`:

- implements p30/p50 dispatch for large multiplication and squaring.

`recip_3by2.h`, `recip_3by2_seed.h`:

- declare AVX-512 assembly reciprocal kernels for block division.

`div.h`:

- implements block reciprocal division helpers and `div3by2`.

`divrem.h`:

- implements normalized block long division and the packed `u64` wrapper.

## Design Invariants

The current implementation relies on these invariants:

- Public arithmetic uses unsigned little-endian `u64` limbs.
- Internal IFMA multiplication uses u52 digits with vector padding.
- One u52 vector is one base-`2^416` block for division.
- u52 tails are often stored as full vectors; callers must provide padding
  unless a tail-masked variant is explicitly used.
- Toom children may return class-1 signed-redundant results; consumers must
  fold or canonicalize before using them in lane-budget-sensitive operations.
- Folded signed canonicalization must run to the top of the whole mixed span.
- FFT output correctness is guarded by final carry-out checks; failure returns
  zero from the FFT/transform entry.
- NTT output correctness is guarded by CRT spill checks; spill returns zero
  from the NTT entry.
- Scratch allocations are scoped and intentionally not individually freed.

## Open Work

The main architectural work remaining is integration and surface cleanup:

- finalize a public naming convention and ABI;
- add complete public add/sub/mul/div entry coverage and documentation;
- integrate `pq16` or retire `fft16` after an A/B decision;
- integrate `ntmul` into the top-level multiplication dispatch;
- define FFT/NTT/Toom crossover policy in one table or generated model;
- add small and medium division paths below the two-block divisor threshold;
- decide whether `p30_ref.h` remains in include or moves to tests/bench
  support;
- produce a single benchmark artifact format that can regenerate
  `u52-mparam.h`, FFT profitability data, and NTT dispatch gates;
- document aliasing rules for every public entry;
- add a stable test oracle story for every header-only public path.

## Milestone Interpretation

The first milestone originally called for public add/sub, u64/u52 conversion,
IFMA Comba, GMP differential tests, and benchmark threshold notes.  The code
has progressed beyond that in multiplication and transform experimentation:

- the u52 basecase and Toom hierarchy are implemented;
- FFT and NTT engines exist;
- division has a serious large-divisor prototype;
- scratch allocation and carry/canonicalization are substantial.

The current milestone should be reframed as consolidation:

1. decide the public multiplication ladder: tiny scalar, u52 Toom, PQ FFT,
   p30/p50 NTT;
2. choose the FFT implementation to keep (`fft16` versus `pq16`);
3. complete division coverage below the current normalized block path;
4. standardize ABI names and aliasing contracts;
5. generate and check in reproducible tuning data.
