# simd_bigint Design Draft

This project is a clean experimental bigint library, not a GMP fork.  The
intent is to keep GMP's proven algorithm formation where it still makes sense,
but design the data movement, thresholds, and scratch formats around wide SIMD
from the beginning.

## Goals

- First-class AVX-512 implementation, especially IFMA/VBMI/VBMI2/BW/VL.
- Keep the public binary representation simple and stable.
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

## Public Binary Format

The external integer format is:

```c
typedef struct sb_int {
    uint64_t *limbs;
    int64_t limb_size;
} sb_int;
```

`limbs` is little-endian: `limbs[0]` is the least significant limb.

`limb_size` is not a two's-complement signed length.  Its top bit is the sign
bit and the remaining low 63 bits are the absolute limb count:

```c
sign = (uint64_t)limb_size >> 63;
size = (uint64_t)limb_size & INT64_MAX;
```

The magnitude is not stored in two's complement.  Negative numbers have the
same limb array as their absolute value, with only the sign bit set in
`limb_size`.

Canonical zero has `limb_size == 0`.  Negative zero is invalid.

## Normalization Invariants

- `size == 0` means the value is zero and `limbs` may be null.
- If `size > 0`, `limbs[size - 1] != 0`.
- The exact bit length is:

```c
bitlen = size * 64 - clz64(limbs[size - 1]);
```

- Public API entry points may accept non-normalized temporaries only when
  explicitly documented.
- Kernel-level functions should separate logical operand length from allocated
  capacity.

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
- signed public values using sign/magnitude dispatch;
- raw magnitude add/sub for multiplication carry propagation.

For `u64x8`, the carry chain should use a mask-based generate/propagate pass:

- compute vector sums;
- generate carry masks from unsigned wraparound;
- generate propagate masks from all-ones lanes;
- prefix the carry across the eight lanes using integer mask arithmetic;
- adjust lanes with a masked vector add/sub.

For `u52x8`, the same idea applies with base `2^52`, but the correction must
wrap modulo `2^52`, not modulo `2^64`.

Subtraction needs the borrow equivalent.  Since the public format is
sign/magnitude, subtraction at the high level dispatches to magnitude compare
plus magnitude subtract instead of relying on sign extension.

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

- Should the public `limb_size` type remain `int64_t`, or should the ABI use
  `uint64_t` while preserving the top-bit sign encoding?
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
- How should Toom evaluation/interpolation represent signed temporary values?
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
