# AVX512 IFMA52 multiplication reference

This directory holds experimental reference code for the AVX512 IFMA bigint
multiplication path.  It is intentionally separate from `include/` and `src/`
so the main library can stay concise while the basecase and Toom designs are
still being tested.

The Toom entry points have overloads that accept `ToomWorkspace`.  Use those
for benchmarking or repeated calls; the no-workspace overloads are convenience
wrappers.  The workspace retains both PMR scratch storage and decoded IFMA
point-product capacity between calls.  It also exposes per-call scratch
allocation counters through `scratch_stats()`, reset by `ToomWorkspace::reset()`.
The public Toom entry points are thresholded and fall back to IFMA basecase or a
lower Toom variant below the measured crossover range.

`mul_auto_u64` is the convenience dispatcher for the reference path.  It chooses
true Toom-32 for measured rectangular `3k x 2k`-style shapes, Toom-33 for
remaining large square-ish shapes, Toom-22 from its cutoff, and IFMA basecase
below Toom cutoffs.

Each Toom variant also has a `*_raw_u64` entry point.  These bypass thresholds
and run the formula directly, which is useful for audit and crossover
benchmarks.

Build and run the differential test:

The test includes randomized correctness grids, cutoff boundaries, padded
operands, reusable workspace cases, scratch accounting, and scratch-footprint
regression guards for raw Toom-22/32/33.  It also covers `mul_auto_u64` across
basecase, Toom-22, Toom-32, Toom-33, and swapped rectangular dispatch cases.

```sh
g++ -O2 -std=c++20 -mavx512f -mavx512ifma -mavx512bw -mavx512vl \
  -mavx512vbmi -mavx512vbmi2 \
  simd_bigint/reference/ifma52_mul.cpp \
  simd_bigint/reference/ifma52_mul_test.cpp \
  -o /tmp/ifma52_mul_test
/tmp/ifma52_mul_test
```

Build and run the GMP comparison benchmark:

```sh
g++ -O3 -std=c++20 -mavx512f -mavx512ifma -mavx512bw -mavx512vl \
  -mavx512vbmi -mavx512vbmi2 \
  simd_bigint/reference/ifma52_mul.cpp \
  simd_bigint/reference/ifma52_mul_bench.cpp \
  -lgmp -o /tmp/ifma52_mul_bench
/tmp/ifma52_mul_bench 160 0.003 3
```

The benchmark prints IFMA basecase, predecoded IFMA core, Toom-22, Toom-32,
Toom-33, GMP scalar basecase, and per-variant speedups over GMP.

Build and run the O3 performance gate.  It checks IFMA basecase, the public
Toom cutoff cases, and matching `mul_auto_u64` rows against GMP scalar basecase
and exits nonzero below the configured speedup threshold, default `2.0`.

```sh
g++ -O3 -std=c++20 -mavx512f -mavx512ifma -mavx512bw -mavx512vl \
  -mavx512vbmi -mavx512vbmi2 \
  simd_bigint/reference/ifma52_mul.cpp \
  simd_bigint/reference/ifma52_mul_perf_check.cpp \
  -lgmp -o /tmp/ifma52_mul_perf_check
/tmp/ifma52_mul_perf_check
/tmp/ifma52_mul_perf_check sweep 0.006 5 2.0
```

Build and run the shape/crossover benchmark.  The default mode is the `3k x 2k`
Toom-32 public path.  Its CSV output appends `scratch_allocs`, `scratch_bytes`,
and `scratch_peak_bytes`; fallback rows report zero scratch allocation.  Pass an
optional final start row to avoid rerunning the whole sweep.

```sh
g++ -O3 -std=c++20 -mavx512f -mavx512ifma -mavx512bw -mavx512vl \
  -mavx512vbmi -mavx512vbmi2 \
  simd_bigint/reference/ifma52_mul.cpp \
  simd_bigint/reference/ifma52_mul_shape_bench.cpp \
  -lgmp -o /tmp/ifma52_mul_shape_bench
/tmp/ifma52_mul_shape_bench 96 0.001 1
/tmp/ifma52_mul_shape_bench toom33-square-raw 144 0.006 5 144
```

Run square Toom-22 threshold and raw modes:

```sh
/tmp/ifma52_mul_shape_bench toom22-square 96 0.001 3
/tmp/ifma52_mul_shape_bench toom22-square-raw 96 0.001 3
```

Run the square Toom-33 threshold mode:

```sh
/tmp/ifma52_mul_shape_bench toom33-square 180 0.003 3
```

Other raw modes bypass public cutoffs:

```sh
/tmp/ifma52_mul_shape_bench toom32-raw 64 0.003 3
/tmp/ifma52_mul_shape_bench toom33-square-raw 180 0.003 3
```
