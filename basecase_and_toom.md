# AVX512 IFMA Basecase And Toom Reference

This note tracks the experimental multiplication implementation kept under
`simd_bigint/reference`.  The code in that directory is a reference and tuning
workspace; it is not the intended final shape of the main library API.

## Current Artifacts

- `reference/ifma52_mul.h`: reference API for decoded IFMA52 basecase and
  Toom entry points, including reusable Toom scratch, raw Toom kernels, and an
  auto-dispatch reference multiply.
- `reference/ifma52_mul.cpp`: AVX512 IFMA52 basecase plus Toom-22, Toom-32,
  and Toom-33 interpolation.
- `reference/ifma52_mul_test.cpp`: randomized differential test against a
  scalar `unsigned __int128` schoolbook oracle.
- `reference/ifma52_mul_bench.cpp`: GMP scalar basecase comparison benchmark
  for square products.
- `reference/ifma52_mul_shape_bench.cpp`: focused GMP comparison benchmark for
  shape-specific public/raw crossovers: square Toom-22, `3k x 2k` Toom-32, and
  square Toom-33.
- `reference/ifma52_mul_perf_check.cpp`: small O3 performance gate that exits
  nonzero if IFMA basecase or the public Toom cutoff cases fall below the
  configured speedup threshold against GMP scalar basecase.

## Basecase

The basecase converts public little-endian `uint64_t` limbs into packed
base-`2^52` digits.  Eight digits fit in one `__m512i`, giving 416 payload
bits per vector.  The IFMA kernel works over block diagonals:

1. Decode each operand into `Decoded52` blocks once.
2. For each block diagonal, keep nine vector accumulators `acc[0..8]`.
3. Use `vpmadd52luq` and `vpmadd52huq` to accumulate every block product into
   the lane-offset accumulators.
4. Harvest the nine offset accumulators into one low vector and one high vector.
5. Normalize carries in base `2^52`.
6. Store each 8-digit chunk directly back into the public `uint64_t` byte
   layout with VBMI byte permutation and masked stores.

The exposed reference functions are:

```c++
decode52(decoded, ap, an);
mul_predecoded_u64(rp, decoded_a, decoded_b, out_limbs, workspace);
mul_basecase_u64(rp, ap, an, bp, bn, workspace);
mul_basecase_u64(rp, ap, an, bp, bn);
mul_auto_u64(rp, ap, an, bp, bn, toom_workspace);
mul_toom*_u64(rp, ap, an, bp, bn, toom_workspace);
mul_toom*_raw_u64(rp, ap, an, bp, bn, toom_workspace);
```

`mul_predecoded_u64` is the core timing target when operands are already in the
temporary format.  `mul_basecase_u64` includes decode cost.
`mul_auto_u64` is the convenience reference dispatcher: it selects true
Toom-32 for measured rectangular `3k x 2k`-style shapes, Toom-33 for remaining
large square-ish shapes from 144 limbs, Toom-22 from 72 limbs, and IFMA
basecase below those cutoffs.

## Toom-22

Split at `k = ceil(max(an, bn) / 2)` public limbs:

```text
A(x) = a0 + a1*x
B(x) = b0 + b1*x
```

Evaluate at `0`, `1`, and infinity:

```text
v0   = a0*b0
v1   = (a0+a1)*(b0+b1)
vinf = a1*b1

c0 = v0
c2 = vinf
c1 = v1 - c0 - c2
```

The result is assembled as `c0 + c1*B^k + c2*B^(2k)`, where
`B = 2^64`.

## Toom-32

The longer operand is split into three chunks and the shorter into two chunks:

```text
A(x) = a0 + a1*x + a2*x^2
B(x) = b0 + b1*x
```

The current interpolation uses positive points `0`, `1`, `2`, and infinity:

```text
v0   = a0*b0
v1   = A(1)*B(1)
v2   = A(2)*B(2)
vinf = a2*b1

s1 = v1 - v0 - vinf        = c1 + c2
s2 = (v2 - v0 - 8*vinf)/2 = c1 + 2*c2

c0 = v0
c2 = s2 - s1
c1 = 2*s1 - s2
c3 = vinf
```

All Toom-32 evaluation and interpolation values in this formulation are
non-negative magnitudes.  If the selected split leaves the three-way side with
no high chunk, the public Toom-32 entry point falls back to Toom-22 instead of
running a degenerate 3x2 interpolation.  True Toom-32 is also cutoff until the
selected chunk size is at least 48 public limbs.

## Toom-33

Both operands are split into three chunks:

```text
A(x) = a0 + a1*x + a2*x^2
B(x) = b0 + b1*x + b2*x^2
```

The current interpolation uses points `0`, `1`, `-1`, `2`, and infinity:

```text
v0   = a0*b0
v1   = A(1)*B(1)
vm1  = A(-1)*B(-1)
v2   = A(2)*B(2)
vinf = a2*b2

c0 = v0
c4 = vinf
s = v1 - c0 - c4          = c1 + c2 + c3
t = vm1 - c0 - c4         = -c1 + c2 - c3
q = (s - t)/2             = c1 + c3
c2 = (s + t)/2
r = (v2 - c0 - 16*c4)/2  = c1 + 2*c2 + 4*c3
c3 = (r - q - 2*c2)/3
c1 = q - c3
```

The `-1` evaluation is represented as sign plus magnitude, but all other
Toom-33 values remain unsigned magnitudes.  This avoids the larger `x=3`
evaluation point while keeping the signed layer small and isolated.
The implementation uses the equivalent identities
`c3 = (r - s - c2) / 3` and `c1 = s - c2 - c3` so `q` never needs to be
materialized.

## Current Implementation Status

- Basecase uses AVX512 IFMA for all point products.
- Toom-22, Toom-32, and Toom-33 are implemented as clear reference algorithms
  over public `uint64_t` limb chunks.  Input chunks are represented as
  non-owning spans into the caller operands, so point products no longer require
  copying the split pieces into temporary vectors first.
- Toom-22 and Toom-32 interpolation use only unsigned magnitude arithmetic.
  Toom-33 uses one small sign/magnitude temporary for the `-1` point.
- Exact interpolation divisions by 2 use a limb right shift.  Exact division by
  3 uses the modular inverse of 3 modulo `2^64`, avoiding scalar integer
  division in the Toom-33 interpolation path.
- Interpolation now mutates point-product temporaries in place for subtract,
  add, small-scale, and exact-division steps where the old value is dead.
  Toom-32 derives `c2` and `c1` directly from the live `s2` and `s1`
  temporaries.  Toom-33 keeps `s = c1 + c2 + c3` live long enough to assemble
  part of `c1`, then turns that same buffer into `c2`; `c3` is derived from
  `r - s - c2`.  The Toom-33 signed `-1` product is also adjusted by `c0` and
  `c4` in place instead of materializing `c0+c4`.  This removes several
  one-use vectors while keeping the formulas easy to audit.
- Toom evaluation buffers that are dead after point-product decoding are now
  reserved with enough capacity for the corresponding product and overwritten
  in place.  Toom-32 computes the `x=1` product first and then reuses that dead
  RHS evaluation buffer for the `x=2` A-side/product buffer.  Toom-33 similarly
  reuses the dead `x=1` and `x=2` RHS buffers for later A-side point products.
- Sign-changing subtract paths reuse the existing destination scratch when
  the destination magnitude is known dead.  This lets Toom-33 compute a
  negative `x=-1` evaluation and signed endpoint subtraction without cloning
  the larger operand first.
- Endpoint coefficients that are already final output terms are computed
  directly into the caller output buffer before interpolation: `c0` at shift
  zero and the infinity coefficient at its final shift.  Interpolation reads
  those coefficients through non-owning spans and skips their final add pass.
- The temporary magnitude container inside `ifma52_mul.cpp` is now a small
  move-only scratch vector over `ToomWorkspace`'s memory resource, with
  64-byte-aligned allocations and an explicit resize-for-overwrite path for
  IFMA point products.  It drops ownership without per-vector deallocation and
  relies on `ToomWorkspace::reset()` to reclaim the arena between calls.  It
  replaces `std::pmr::vector` in the Toom arithmetic layer while keeping the
  public reference API unchanged.
- Small-multiple subtractions in interpolation are fused into the destination
  pass.  Terms such as `8*c3`, `16*c4`, and `2*c2` are no longer materialized
  as separate magnitude vectors before subtraction.
- Interpolation coefficients are assembled directly into the caller's output
  buffer with shifted add-with-carry passes.  Earlier reference revisions built
  a separate PMR result vector and copied it out; direct assembly removes that
  large temporary from every raw Toom kernel.  The endpoint-coefficient storage
  now removes two more point-product temporaries from each raw Toom kernel.
- Toom-33 no longer materializes either `q = c1 + c3` or
  `c1 = q - c3` as separate magnitudes.  It adds `s = c1 + c2 + c3` at the
  `B^k` shift, converts `s` into `c2`, subtracts `c2` and `c3` at the same
  shift, then assembles `c2` and `c3` at their final shifts.
- `ToomWorkspace` provides reusable PMR-backed scratch for evaluation and
  interpolation, and retains one `Ifma52Workspace` so decoded-vector capacity is
  reused both across point products and across repeated Toom calls.
- `ToomWorkspace::scratch_stats()` reports per-call scratch arena allocation
  count, requested bytes, current live bytes, and peak live bytes.  The shape
  benchmark appends allocation columns so cutoff sweeps can separate arithmetic
  cost from scratch traffic; IFMA fallback rows report zero Toom scratch bytes.
- Public and raw Toom entry points normalize significant limb lengths before
  choosing split points.  They still clear and write the full caller-requested
  output range, so padded operands do not pay for zero high limbs and do not
  leave stale high output.
- The public Toom entry points are thresholded reference algorithms:
  Toom-22 falls back below 72 limbs, true Toom-32 below a 48-limb split chunk,
  and Toom-33 below 144 limbs.  Fallbacks use IFMA basecase or the lower Toom
  entry point.
- `mul_auto_u64` preserves those public thresholds but adds a shape policy:
  choose true Toom-32 first when the normalized operand lengths form a real
  `3 x 2` split with `k >= 48`; otherwise choose Toom-33 when both operands are
  at least 144 limbs, Toom-22 when both are at least 72 limbs, and IFMA
  basecase below that.
- `mul_toom*_raw_u64` entry points bypass thresholds and run the corresponding
  Toom formula directly.  These exist for correctness audit and crossover
  measurement, not as tuned dispatch choices.
- Toom point products currently call the IFMA basecase rather than recursively
  dispatching to smaller Toom variants.  Larger-shape measurements still show
  the IFMA point products beating recursive use of the current reference Toom
  kernels in most measured ranges, so recursion is left as a future tuning step.
- The scratch model is still reference-grade: input chunks now use spans,
  several evaluation/interpolation temporaries are reused in place, and
  magnitudes use a local aligned scratch vector, but each magnitude still owns
  an independently sized scratch allocation.  Production should replace those
  dynamic scratch vectors with pre-sized arena spans.

## Verification

Current local test command:

```sh
g++ -O2 -std=c++20 -mavx512f -mavx512ifma -mavx512bw -mavx512vl \
  -mavx512vbmi -mavx512vbmi2 \
  simd_bigint/reference/ifma52_mul.cpp \
  simd_bigint/reference/ifma52_mul_test.cpp \
  -o /tmp/ifma52_mul_test
/tmp/ifma52_mul_test
```

Result on 2026-06-01:

```text
ok ifma52 basecase/toom reference
```

The test covers:

- basecase grids up to `96 x 96` limbs;
- Toom-22 grids up to `96 x 96` limbs;
- Toom-32 grids up to `120 x 96` limbs, including swapped rectangular input;
- Toom-33 grids up to `120 x 120` limbs;
- explicit reusable-workspace checks for Toom-22/32/33 up to `160 x 160`
  rectangular products;
- cutoff boundary checks for Toom-22 at `71/72`, Toom-32 at the 48-limb split
  threshold, and Toom-33 at `143/144`, plus a true Toom-33 `180 x 180`
  public case;
- raw-kernel checks for Toom-22, Toom-32, and Toom-33 below and at their public
  cutoffs, plus a raw Toom-33 `180 x 180` case;
- padded-operand checks where the physical input arrays are larger than the
  significant length, including public and raw Toom paths and zero operands;
- auto-dispatch checks for basecase, Toom-22, Toom-32 including swapped
  rectangular input, Toom-33, and larger rectangular/square shapes;
- scratch-accounting checks that raw Toom records arena allocation, reset clears
  counters, and public fallback rows leave Toom scratch at zero;
- scratch-footprint regression checks that cap raw Toom-22/32/33 at the current
  allocation counts and byte ceilings for representative cutoff-size cases;
- random data, all-ones stress cases, and deliberately non-normalized top
  limbs.

## Performance Snapshot

The earlier standalone IFMA52 experiment on this Zen 5-class AVX512 machine
showed the predecoded IFMA core exceeding 2x GMP scalar basecase once operands
are above the tiny range, and the full decode-plus-core path exceeding 2x once
decode setup is amortized over larger basecase products.  The reference
benchmark reproduces this shape:

```sh
g++ -O3 -std=c++20 -mavx512f -mavx512ifma -mavx512bw -mavx512vl \
  -mavx512vbmi -mavx512vbmi2 \
  simd_bigint/reference/ifma52_mul.cpp \
  simd_bigint/reference/ifma52_mul_bench.cpp \
  -lgmp -o /tmp/ifma52_mul_bench
/tmp/ifma52_mul_bench 260 0.001 3
```

Shape-specific paths are benchmarked separately:

```sh
g++ -O3 -std=c++20 -mavx512f -mavx512ifma -mavx512bw -mavx512vl \
  -mavx512vbmi -mavx512vbmi2 \
  simd_bigint/reference/ifma52_mul.cpp \
  simd_bigint/reference/ifma52_mul_shape_bench.cpp \
  -lgmp -o /tmp/ifma52_mul_shape_bench
/tmp/ifma52_mul_shape_bench toom32-public 96 0.003 3
/tmp/ifma52_mul_shape_bench toom22-square 96 0.003 5
/tmp/ifma52_mul_shape_bench toom22-square-raw 96 0.003 5
/tmp/ifma52_mul_shape_bench toom33-square 180 0.003 3
/tmp/ifma52_mul_shape_bench toom32-raw 64 0.003 3
/tmp/ifma52_mul_shape_bench toom33-square-raw 180 0.003 3
```

The shape benchmark appends `scratch_allocs`, `scratch_bytes`, and
`scratch_peak_bytes` columns.  These are measured with a single untimed probe
call for each row, using the same public/raw Toom mode as the timed sample.
Rows that dispatch to IFMA basecase fallback have zero Toom scratch bytes.
An optional final argument starts the sweep at a specific row, for example
`/tmp/ifma52_mul_shape_bench toom33-square-raw 144 0.006 5 144`.

The repeatable 2x threshold gate checks the main public crossover points and
the matching auto-dispatch rows:

```sh
g++ -O3 -std=c++20 -mavx512f -mavx512ifma -mavx512bw -mavx512vl \
  -mavx512vbmi -mavx512vbmi2 \
  simd_bigint/reference/ifma52_mul.cpp \
  simd_bigint/reference/ifma52_mul_perf_check.cpp \
  -lgmp -o /tmp/ifma52_mul_perf_check
/tmp/ifma52_mul_perf_check
/tmp/ifma52_mul_perf_check sweep 0.006 5 2.0
```

Current result on 2026-06-01:

```text
case,an,bn,ifma_or_toom_ns,gmp_basecase_ns,speedup,min_speedup,scratch_allocs,scratch_bytes
ifma_basecase_64x64,64,64,320.510239,1498.394593,4.675029,2.00,0,0
toom22_public_72x72,72,72,741.537549,1916.265306,2.584178,2.00,2,888
toom32_public_144x96,144,96,1706.763636,5282.661458,3.095134,2.00,3,1992
toom33_public_144x144,144,144,2755.238225,7923.236979,2.875699,2.00,4,2792
auto_64x64,64,64,330.076230,1514.490423,4.588305,2.00,0,0
auto_72x72,72,72,752.954625,1927.757015,2.560257,2.00,2,888
auto_144x96,144,96,1724.795585,5299.500868,3.072539,2.00,3,1992
auto_144x144,144,144,2741.470562,7936.934896,2.895138,2.00,4,2792
```

Current sweep result on 2026-06-01:

```text
case,an,bn,ifma_or_toom_ns,gmp_basecase_ns,speedup,min_speedup,scratch_allocs,scratch_bytes
ifma_basecase_64x64,64,64,321.862082,1500.287000,4.661273,2.00,0,0
ifma_basecase_96x96,96,96,601.436098,3612.330529,6.006175,2.00,0,0
ifma_basecase_128x128,128,128,960.804847,6226.323589,6.480321,2.00,0,0
toom22_public_72x72,72,72,733.403076,1919.262436,2.616927,2.00,2,888
toom22_public_80x80,80,80,810.918238,2396.446598,2.955226,2.00,2,984
toom22_public_96x96,96,96,1012.497984,3476.468750,3.433556,2.00,2,1176
toom32_public_144x96,144,96,1743.376157,5294.691840,3.037034,2.00,3,1992
toom32_public_168x112,168,112,2012.266290,7223.831731,3.589898,2.00,3,2312
toom32_public_192x128,192,128,2329.732639,9561.943750,4.104309,2.00,3,2632
toom33_public_144x144,144,144,2752.022645,7997.751302,2.906136,2.00,4,2792
toom33_public_152x152,152,152,2946.473145,8908.622159,3.023487,2.00,4,2960
toom33_public_160x160,160,160,3075.383197,9892.480263,3.216666,2.00,4,3128
auto_64x64,64,64,328.924475,1520.606603,4.622966,2.00,0,0
auto_72x72,72,72,740.588952,1937.231959,2.615799,2.00,2,888
auto_96x96,96,96,980.235677,3489.523727,3.559882,2.00,2,1176
auto_144x96,144,96,1718.902273,5321.375000,3.095798,2.00,3,1992
auto_168x112,168,112,2030.227487,7287.423077,3.589461,2.00,3,2312
auto_144x144,144,144,2753.865942,7977.843750,2.896962,2.00,4,2792
auto_160x160,160,160,3164.768750,9901.296053,3.128600,2.00,4,3128
```

The sweep is intended to use normal median samples.  Very short one-repeat
micro-sweeps are useful for smoke testing the command path, but cutoff rows are
too noisy for pass/fail conclusions at that duration.

Short scratch probes after sequential evaluation-buffer reuse:

```text
raw Toom-22, n=96:  scratch_allocs=2, scratch_bytes=1176
raw Toom-32, k=64:  scratch_allocs=3, scratch_bytes=2632
raw Toom-33, n=144: scratch_allocs=4, scratch_bytes=2792
```

Short local square smoke run on 2026-06-01:

```text
n=32:  ifma_full 137.20 ns, ifma_core 94.87 ns, gmp_basecase 350.17 ns
       speedup full 2.55x, core 3.69x
       public toom22 2.44x, toom32 2.50x, toom33 2.53x
n=72:  toom22 712.81 ns, toom32 716.80 ns, toom33 388.48 ns,
       gmp_basecase 1907.62 ns
       speedup toom22 2.68x, toom32 2.66x, toom33 4.91x
n=96:  toom22 969.54 ns, toom32 966.47 ns, toom33 592.42 ns,
       gmp_basecase 3443.33 ns
       speedup toom22 3.55x, toom32 3.56x, toom33 5.81x
```

The short aggregate square sweep is useful as a smoke check but is noisy when
several variants are measured back-to-back.  Cutoff decisions use the focused
shape modes below.

For square inputs, `toom32` falls back to Toom-22 because the 3-way side would
otherwise have no high chunk.  The dedicated `3k x 2k` Toom-32 run shows the
thresholded public Toom-32 path; rows below `k=48` are IFMA basecase fallbacks,
and rows from `k=48` onward are true Toom-32:

```text
k=40: an=120, bn=80,  toom32 648.41 ns,  gmp_basecase 3690.38 ns,
      speedup 5.69x
k=48: an=144, bn=96,  toom32 1740.00 ns, gmp_basecase 5329.33 ns,
      speedup 3.06x
k=64: an=192, bn=128, toom32 2383.37 ns, gmp_basecase 9593.94 ns,
      speedup 4.03x
k=96: an=288, bn=192, toom32 3953.14 ns, gmp_basecase 21748.68 ns,
      speedup 5.50x
```

The square Toom-22 shape mode separates the thresholded public path from the raw
formula.  Rows below `n=72` are IFMA basecase fallbacks for the public entry
point, and rows from `n=72` onward are true Toom-22:

```text
public Toom-22, n=32: speedup 2.22x
public Toom-22, n=64: speedup 4.71x
public Toom-22, n=72: speedup 2.61x
public Toom-22, n=96: speedup 3.57x

raw Toom-22, n=32: speedup 1.05x
raw Toom-22, n=64: speedup 2.48x
raw Toom-22, n=71: speedup 2.57x
raw Toom-22, n=72: speedup 2.63x
raw Toom-22, n=96: speedup 3.59x
```

The square Toom-33 threshold mode isolates that public entry point; rows below
`n=144` are IFMA basecase fallbacks, and rows from `n=144` onward are true
Toom-33:

```text
n=136: toom33 speedup 6.67x (IFMA fallback)
n=140: toom33 speedup 6.68x (IFMA fallback)
n=144: toom33 speedup 2.88x
n=146: toom33 speedup 3.11x
n=150: toom33 speedup 2.97x
```

Raw modes bypass cutoffs and measure the formulas directly:

```text
raw Toom-22, n=64:  speedup 2.48x
raw Toom-22, n=72:  speedup 2.63x
raw Toom-22, n=96:  speedup 3.59x

raw Toom-32, k=24: an=72,  bn=48,  speedup 1.52x
raw Toom-32, k=40: an=120, bn=80,  speedup 2.61x
raw Toom-32, k=48: an=144, bn=96,  speedup 3.10x
raw Toom-32, k=64: an=192, bn=128, speedup 3.83x

raw Toom-33, n=136: speedup 2.72x
raw Toom-33, n=144: speedup 2.89x
raw Toom-33, n=160: speedup 3.09x
raw Toom-33, n=180: speedup 3.61x
```

With these cutoffs, the focused public Toom shape modes and the auto-dispatch
rows clear the 2x GMP scalar target at their intended crossover points in this
run.  The raw Toom-22 kernel clears from about 64 square limbs; raw rectangular
Toom-32 clears from about `120 x 80`, while the public true-Toom32 cutoff
remains at `144 x 96` to keep margin.  Raw Toom-33 clears the 2x GMP target
below its public cutoff in this run, but it is still slower than the IFMA
basecase fallback there.  Tiny products below that range still use the IFMA
basecase and are not part of the Toom crossover target.

## Completion Audit

Current evidence satisfies the stated reference objective:

- The AVX512 IFMA basecase is implemented in `reference/ifma52_mul.cpp` using
  `_mm512_madd52lo_epu64` and `_mm512_madd52hi_epu64`.
- Toom-22, Toom-32, and Toom-33 public and raw reference kernels are implemented
  in `reference/ifma52_mul.cpp` and exposed from `reference/ifma52_mul.h`.
- The reference implementation is isolated under `simd_bigint/reference`; the
  main library can use it as design and tuning material without absorbing the
  experimental code.
- `basecase_and_toom.md` records the algorithms, implementation decisions,
  verification commands, scratch accounting, and performance data.
- The differential test command above passes against an independent
  `unsigned __int128` schoolbook oracle and includes boundary, padding,
  workspace-reuse, raw-kernel, auto-dispatch, and scratch-footprint checks.
- The O3 performance gate above passes with every checked basecase, public Toom,
  and auto-dispatch row above the configured `2.0x` GMP scalar basecase
  threshold.

## Optional Next Refinements

1. Replace the remaining dynamically sized scratch vectors with pre-sized arena
   spans, especially the live Toom-33 evaluation, point-product, and signed
   `x=-1` magnitudes.
2. Add recursive point-product dispatch once measured Toom thresholds justify
   it; current measurements favor IFMA point products until after the remaining
   Toom scratch/traffic reductions.
3. Use the scratch allocation counters in longer cutoff sweeps to decide which
   interpolation temporaries should be converted to fixed arena spans first.
4. Tune cutoffs separately for full decode cost, predecoded core cost, and each
   Toom variant.
5. Add broader shape-specific benchmarks for additional rectangular ratios and
   longer threshold sweeps.
