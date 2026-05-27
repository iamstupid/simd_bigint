# simd_bigint Design Draft

This project is a clean experimental bigint library, not a GMP fork.  The
intent is to keep GMP's proven algorithm formation where it still makes sense,
but design the data movement, thresholds, and scratch formats around wide SIMD
from the beginning.

## Goals

- First-class AVX-512 implementation, especially IFMA/VBMI/VBMI2/BW/VL.
- Keep the public low-level ABI simple: unsigned limb pointers plus explicit
  sizes.
- Use SIMD-native temporary formats instead of forcing every kernel through
  `u64` limb arithmetic.
- Make multiplication decisions from the actual operand shape, not from a
  single size parameter.
- Keep short and medium multiplication scratch local and cheap; avoid hidden
  heap traffic in hot paths.
- Use reproducible benchmarks and per-microarchitecture threshold generation.

## Non-Goals For The First Pass

- ABI compatibility with GMP.
- Supporting every old x86 target.
- A high-level `mpz` replacement before the low-level kernels are correct.
- Constant-time cryptographic behavior in the first arithmetic kernels.

## Low-Level Data Model

The low-level library should work like GMP's `mpn` layer: it operates directly
on unsigned magnitude arrays, not on signed bigint objects.

```c
typedef uint64_t sb_limb_t;
typedef int64_t sb_size_t;

typedef sb_limb_t *sb_ptr;
typedef const sb_limb_t *sb_srcptr;
```

Operands are passed as pointer-plus-length pairs:

```c
sb_add_n(rp, ap, bp, n);
sb_mul(rp, ap, an, bp, bn);
```

`limbs` is little-endian: `limbs[0]` is the least significant limb.

All low-level arithmetic is unsigned.  Signed integers, sign/magnitude object
wrappers, allocation policy, and canonical object normalization belong above
this layer.

This removes sign extension from hot kernels and keeps multiplication,
division, Toom splitting, FFT, and NTT working on plain magnitudes.

## Normalization Invariants

For a normalized public magnitude:

- `n == 0` means zero and the limb pointer may be null.
- If `n > 0`, `ap[n - 1] != 0`.
- The exact bit length is:

```c
bitlen = n * 64 - clz64(ap[n - 1]);
```

Kernel-level functions may accept fixed-size temporary slices whose top limb is
zero.  That must be documented per function.  Algorithmic kernels should keep
logical length, allocated capacity, and scratch length separate.

## Temporary Internal Formats

### u64x8

Native 64-bit limb vectors in `__m512i`.  This is the natural format for
addition, subtraction, compare, normalization scans, and public input/output.

Addition and subtraction need dedicated vector carry/borrow handling.  The
carry design should be based on generate/propagate masks rather than scalar
lane-by-lane carry chains.

### u52x8

Eight base-`2^52` digits per ZMM register.  This is the short multiplication
format for AVX-512 IFMA:

- one vector holds 416 bits, equal to 6.5 `u64` limbs;
- each digit keeps 12 spare bits before normalization;
- `vpmadd52luq` and `vpmadd52huq` are the primitive multiply-add operations.

The decode path must be reusable.  Decoding the same operand block repeatedly
is not acceptable in a competitive basecase/Toom implementation.

### PQ-Packed Double Complex AoSoV

Medium multiplication should use a packed complex FFT layout designed around
SIMD vectors rather than scalar complex structs.

The working assumption is an array-of-structs-of-vectors layout where multiple
complex lanes are transformed together and the `P/Q` split is chosen to reduce
carry pressure and rounding error.

The exact packing, radix schedule, error bounds, and carry reconstruction are
still open design choices.

### p50xn NTT

Large multiplication should use an NTT representation with approximately
50-bit payload chunks.  The transform may use several machine-word primes plus
CRT reconstruction.

The exact prime set, transform radix, lazy reduction policy, and crossover
against FFT are open design choices.

## Addition And Subtraction

Addition/subtraction should have separate kernels for:

- public `u64` limb arrays;
- internal `u52x8` digit arrays;
- raw magnitude add/sub for multiplication carry propagation.

For `u64x8`, the carry chain should use a mask-based generate/propagate pass:

- compute vector sums;
- generate carry masks from unsigned wraparound;
- generate propagate masks from all-ones lanes;
- prefix the carry across the eight lanes using integer mask arithmetic;
- adjust lanes with a masked vector add/sub.

For `u52x8`, the same idea applies with base `2^52`, but the correction must
wrap modulo `2^52`, not modulo `2^64`.

Subtraction needs the borrow equivalent.  Low-level subtraction assumes the
caller has selected the correct larger magnitude when a non-negative result is
required.  Signed dispatch belongs above this layer.

## Performance-Critical MPN Kernels

The first implementation should not blindly copy every GMP `mpn` entry point.
The kernels below are the ones that matter for a SIMD-first bigint core.

### u52-Only Internal Kernels

These kernels operate only on temporary base-`2^52` vector arrays.  They should
not be public ABI unless benchmarking proves a reason to expose them.

- `sb52_carry_normalize`: propagate carries across `u52x8` vectors.
- `sb52_add_n`, `sb52_sub_n`: vector add/sub for internal Toom and carry work.
- `sb52_lshift`, `sb52_rshift`: digit and bit shifts inside base `2^52`.
- `sb52_mul_basecase`: IFMA Comba rectangular multiplication.
- `sb52_sqr_basecase`: IFMA Comba squaring with symmetry exploitation.
- `sb52_addmul_1`, `sb52_submul_1`: single-vector multiply accumulation for
  Toom interpolation and division helpers.
- `sb52_toom22_mul`, `sb52_toom32_mul`, `sb52_toom42_mul`,
  `sb52_toom33_mul`: recursive multiplication in decoded form.
- `sb52_toom_eval_*`, `sb52_toom_interp_*`: evaluation/interpolation helpers
  that avoid converting back to `u64` between recursive calls.
- `sb52_divexact_1`, `sb52_divexact_by3`, `sb52_divexact_by5`: exact small
  constant division for Toom interpolation.
- `sb52_inv_1_precomp`: reciprocal/preinverse helpers if division starts using
  decoded arithmetic.
- `sb52_div_qr_basecase`: optional internal division once multiplication and
  Toom need reciprocal-based division experiments.

### u64-Only Interface Kernels

These are the exposed pointer-format kernels.  They operate on little-endian
`uint64_t` limbs and define the stable low-level ABI.

- `sb_copyi`, `sb_copyd`, `sb_zero`: overlap-aware copy and zeroing.
- `sb_normalized_size`: trim leading zero limbs.
- `sb_bitlen`: exact bit length using the top limb `clz`.
- `sb_cmp`, `sb_cmp_n`, `sb_zero_p`: magnitude comparison and zero tests.
- `sb_add_1`, `sb_sub_1`: add/subtract one limb.
- `sb_mul_1`: public multiply by one limb.
- `sb_div_qr_1`, `sb_mod_1`, `sb_preinv_div_qr_1`: public single-limb
  division and remainder.
- `sb_invert_limb`: reciprocal helper for division.
- `sb_get/set` helpers for bytes, if import/export becomes part of this layer.

### Kernels Needed In Both u64 And u52 Forms

These operations are performance-critical at the public boundary and inside
decoded multiplication/Toom.  They need separate implementations for each
representation, not one generic abstraction.

- `add_n`, `add`: vectorized addition with carry-out.
- `sub_n`, `sub`: vectorized subtraction with borrow-out.
- `lshift`, `rshift`: bit shifts with carry bits across limbs/digits.
- `addmul_1`, `submul_1`: core building blocks for basecase, division, and
  interpolation.
- `decode_blocks`, `pack_blocks`: `u64 <-> u52x8` conversion at algorithm
  boundaries.
- `mul_basecase`: scalar/`u64` tiny kernels and IFMA/`u52` kernels.
- `sqr_basecase`: tiny scalar squaring and IFMA/`u52` squaring.
- `mul`, `mul_n`, `sqr`: public `u64` entry points with internal dispatch to
  scalar, `u52`, FFT, or NTT implementations.
- `toom*_mul`: public `u64` entry points that may immediately decode and then
  recurse in `u52`.
- `div_qr_basecase`: public `u64` division plus possible `u52` reciprocal
  experiments.
- `carry_normalize`: public `u64` result normalization and internal `u52`
  carry propagation.
- `scan/cmp high limbs`: top-limb search and compare logic for dispatch,
  normalization, and division.

## Multiplication Plan

### 1. Tiny Scalar Kernel

For inputs where the longer side is below one full `u52x8` block, use hand
written scalar assembly.

Initial target:

- up to `6 x 6` `u64` limbs;
- tuned `mulx/adcx/adox` Comba kernels;
- separate shapes for rectangular operands where worthwhile.

This avoids SIMD setup overhead for tiny products.

### 2. SIMD Basecase And Toom In u52x8

For sizes from roughly `6.5 x 6.5` limbs upward, use IFMA basecase and Toom
kernels in `u52x8` format.

The basecase direction is Comba-style:

- decode each operand into `u52x8` blocks once;
- keep one accumulator vector for each lane offset, `acc[0..8]`;
- use IFMA directly into the offset accumulators;
- harvest offset accumulators into output vectors once per Comba diagonal;
- propagate carry in vector form;
- pack back to `u64` only at the output boundary.

Toom variants should follow GMP's broad formation:

- Toom-22;
- Toom-32;
- Toom-42;
- Toom-33;
- higher variants only after basecase/Toom-3 behavior is understood.

The Toom evaluation/interpolation format should avoid premature conversion
back to public `u64` limbs.

Scratch for this size class should use stack-local C VLAs with explicit
alignment.  Heap allocation in these kernels should be treated as a fallback,
not the normal path.

### 3. FFT

Medium and large-enough products should transition to a SIMD complex FFT.

The FFT path should be benchmarked against the best Toom variant over its
range, not only against basecase.

Open items:

- transform radix and blocking;
- exact `P/Q` packed layout;
- whether the data is kept split through carry reconstruction;
- error analysis and guard bits;
- AVX-512 FMA scheduling on Zen 5 and later.

### 4. NTT

The largest products should transition to an integer NTT.

Open items:

- prime set;
- radix schedule;
- lazy reduction depth;
- CRT reconstruction strategy;
- crossover against FFT;
- memory traffic model.

## Operand-Shape Dispatch

Multiplication is not a one-parameter kernel.  Dispatch should use at least:

- `an`, `bn`;
- exact bit lengths of both operands;
- rectangularity ratio;
- output aliasing constraints;
- available scratch;
- target microarchitecture.

Threshold tables should support two-dimensional or small-rule dispatch rather
than a single `MUL_TOOM22_THRESHOLD` style cutoff.

## Stack Scratch Policy

Short and medium kernels should prefer stack-local scratch:

```c
alignas(64) u52x8 a_tmp[a_blocks];
alignas(64) u52x8 b_tmp[b_blocks];
```

The project will require compilers with C VLA support for these kernels.

Design choices still needed:

- maximum stack scratch before falling back to caller-provided workspace;
- whether public APIs expose scratch estimation;
- whether recursive Toom receives one pre-sliced workspace or uses nested VLAs;
- alignment syntax for the chosen C standard and compiler set.

## Initial Source Layout

```text
include/        public C headers
src/            portable and target-specific library code
bench/          benchmark drivers and threshold tooling
tests/          correctness tests and randomized differential tests
docs/           design notes
```

Target-specific source should be explicit, for example:

```text
src/x86_64/avx512/
src/x86_64/znver5/
```

## Design Choices To Solve

- Should the size type be signed like GMP's `mp_size_t` for compatibility with
  local conventions, or unsigned because this layer has no signed object
  lengths?
- How much normalization is required at public API boundaries?
- What is the exact borrow propagation algorithm for `u64x8` subtraction?
- What is the exact carry/borrow propagation algorithm for `u52x8` after IFMA
  accumulation and Toom interpolation?
- What is the optimal `u64 <-> u52x8` decode/encode schedule for half-limb
  block boundaries?
- Should decoded operands be stored as plain 64-byte blocks only, or should
  tail lengths live in side metadata?
- What is the exact Comba traversal shape for rectangular operands?
- Where should Duff-style tail dispatch live so full-vector hot paths remain
  branch-free?
- Which scalar tiny kernels deserve handwritten assembly first?
- How should Toom evaluation/interpolation represent negative temporary values
  without leaking signed object semantics into the public ABI?
- Which Toom variants should be implemented before FFT becomes worthwhile?
- How should recursive Toom scratch be laid out to avoid repeated decoding and
  unnecessary writes?
- What is the PQ-packed complex FFT layout?
- What are the FFT rounding-error limits for each payload width?
- What NTT primes and radix schedule should be used for the `p50xn` format?
- How should FFT and NTT carry reconstruction share code with `u52x8` carry
  propagation?
- Should runtime CPU dispatch exist, or should builds be target-specific first?
- What benchmark format should threshold generation consume and emit?
- What external oracle should tests use: GMP, randomized schoolbook, or both?

## First Milestone

Build a narrow but complete low-level path:

1. public magnitude add/sub for `u64` limbs;
2. `u64` to `u52x8` decode and `u52x8` to `u64` encode;
3. IFMA Comba basecase for rectangular operands;
4. GMP-differential test harness;
5. benchmark sweep against GMP `mpn_mul_basecase` and `mpn_mul_n`;
6. threshold notes for Zen 5.
