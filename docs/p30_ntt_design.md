# p30: 5×30-bit-prime truncated NTT multiplier — design v2

Large-band engine: van-der-Hoeven-truncated CRT-NTT over five 30-bit
primes with 64-bit chunking, AVX-512 epi32 throughout, radix-8 interior.
Takes over from pq16 (cap 131072 limbs) and extends the band to total
2^22 limbs. A later 4×p50 engine (80-bit chunking, IFMA or double-FMA,
Bailey 4-step) covers sizes beyond — out of scope here.

First-class constraints (fixed before any kernel choice):
  T1. TRUNCATED transform, FLINT-style — smooth ns/limb in n, no pow2 or
      PFA grid padding. Everything else must compose with truncation.
  T2. AVX-512 — 16-lane epi32, 32 zmm registers — radix-8 interior
      passes (the references are AVX2/16-ymm and stop at radix-4).

Numeric claims are measured (bench/p30_kernel_bench.c) or exact.

## 0. Reference analysis (what we take, what we leave)

Sources: reference/zint (github.com/iamstupid/zint — AVX2 p30x3 +
p50x4, ~75 ns/limb at 1M limbs ≈ 3.6× GMP on Zen4) and
reference/p30x3_2.hpp (AVX2 p30x3 with SIMD Garner CRT).

ADOPTED:
- Lazy-range discipline beyond ours (zint radix4.hpp): values flow in
  [0,4M) BETWEEN butterfly layers; only operands of a multiply or a
  store get shrunk to [0,2M). add2/sub2/lazy_add/lazy_sub + min-trick.
  Our bench kernels reduced every step — adopting the 4M-flow removes
  ~1 sub+min per point per level.
- Twiddle-free j=0 path (zint Phase 2, FLINT *_1 vs *_0): the first
  block of every level has all-ones twiddles; dedicated notw kernels.
- SIMD Garner CRT with Barrett-by-constant (p30x3_2 lines 793-832):
  exploit "any u32 input" to skip every pre-reduction (d1 = x1−x0+p1
  feeds Barrett unreduced); ascending prime order so x0 needs no
  reduction; compose mixed-radix digits with vpmuludq Horner. We extend
  3→5 primes and replace their scalar u128 carry loop with our SWAR
  3-stream carry chains (pq16 q_chain shape) — the one piece of their
  CRT that didn't vectorize.
- Single-pass reduce-and-pad input staging (zint api.hpp) shape, fused
  with our centered-band lesson: staging normalizes layout.
- Stop-the-NTT-at-the-vector basecase + twisted cyclic convolution
  pointwise (zint cyclic_conv.hpp) as the PRIMARY basecase candidate
  (see §3c): the transform becomes pure vertical SIMD — zero lane
  crossings end to end — and 4 levels (lg 16) vanish from both forward
  and inverse, in exchange for a 16-point twisted conv per vector pair
  in the pointwise (2 vpmuludq/point at zmm, same rate as their 8-pt
  at ymm).
- Montgomery mul with precomputed (b, b·niv) pairs (zint mul_bsmfxd)
  where a VECTOR twiddle is unavoidable (basecase option A): breaks the
  reduce dependency chain (a·b and a·bniv run in parallel).

REJECTED, with reasons:
- Separated radix-{1,3,5} engines with per-class prime triples (zint
  engines.hpp): that is their answer to grid padding (worst ×1.5);
  truncation answers it strictly better (measured +14% worst, decaying)
  with ONE prime set and no cross-engine code. T1 kills it.
- Ruler-sequence on-the-fly root updates (zint scheduler.hpp st_1/rt3):
  elegant zero-table scheme, but it requires visiting butterfly groups
  in exact sequential order — vdH truncation skips subtrees, breaking
  the update chain. FLINT's random-access layout wins under T1; our
  block scalars make tables ≈ free anyway (§5).
- 32-bit chunking with 3 primes (both references): input stage becomes
  multiply-free (two min-subs reduce u32 into lazy range — admirable),
  but per total-limb it costs 6·(lgN+1) point-levels and 24 B spectra
  vs 5·lgN and 20 B for 64-bit chunking with 5 primes (~20% more ALU,
  ~17% more BW, and twice the Garner invocations). 64-bit chunking
  also makes the carry-out limb-aligned (no 32-bit pair packing).
- Their pointwise normalization (p30x3_2: to_mont(b) then mont-mul =
  2 mulmods/point): ours folds the stray R^−1 into the mandatory 1/N
  scaling constant — 1 mulmod/point.
- u128 float-quotient CRT (zint crt_recover_e): needs a double
  estimate + conditional correction; Garner is exact, all-integer,
  and SIMD-cleaner at 5 primes.
- Bailey 4-step with explicit cache-oblivious transposes (zint p50x4):
  right call at their L≥27 sizes; at our N ≤ 2^22 (16 MB/prime) the
  vdH column/row recursion over 1-KB block units IS the locality
  scheme (block-strided columns, TLB-friendly). Revisit for p50.

## 1. Parameters (fixed profile — no chooser)

Primes p = k·2^22 + 1 < 2^30 (2^22-th roots; 22 exist), largest five:

    p0 = 918552577 = 219*2^22+1   (29.7748 bits)
    p1 = 935329793 = 223*2^22+1   (29.8009 bits)
    p2 = 943718401 = 225*2^22+1   (29.8138 bits)   [also zint C.p0]
    p3 = 985661441 = 235*2^22+1   (29.8765 bits)   [also zint C.p2]
    p4 = 998244353 = 238*2^22+1   (29.8948 bits, 2-val 23)

ASCENDING order is load-bearing for Garner (x0 < p_i for all i, no
reduction of the first digit — p30x3_2's trick). Product P = 2^149.16.
64-bit chunking: one u64 limb = one coefficient.

  - CRT bound: min(an, bn) ≤ floor(P / (2^64−1)^2) = 2,344,425
  - transform bound: N = trunc-grid(an+bn−1) ≤ 2^22
  - headroom at the balanced cap (an = bn = 2^21): 1.118 — exact, no
    rounding analysis exists in an integer pipeline; the all-0xFF probe
    at total = 2^22 exercises it directly.

If the vector basecase (§3c) is adopted, the NTT depth requirement
drops by lg(16) = 4 (only 2^18-th roots needed) — the prime CONSTRAINT
relaxes, but we keep this set: it is optimal among p < 2^30 anyway.

## 2. Core arithmetic (measured; validated 100k cases at the bounds)

Coefficients in NORMAL form end to end; lazy flow [0,4M) between
layers, shrink to [0,2M) only at multiply inputs and stores (zint
discipline). 4p < 2^32 so sums never wrap; corrections are min_epu32.

Three multiply forms by multiplier kind:
1. BLOCK TWIDDLES (the bulk under T1's block recursion) — Barrett-by-
   constant: c_rec = floor(c·2^32/p); q = hi32(a·c_rec) under-
   estimates by ≤1 for ANY u32 a; r = lo32(a·c − q·p) ∈ [0,2p).
   (c, c_rec) broadcast per block node; ~8 ops/vector, no table stream.
2. BASECASE VECTOR TWIDDLES (only if option §3c-A is chosen) — zint
   mul_bsmfxd: Montgomery with precomputed (b, b·niv) vector pairs,
   parallel product/reduce chains; tables tiny + L1-resident.
3. POINTWISE (variable × variable) — Montgomery REDC; single uniform
   stray R^−1 folded into the iNTT scaling constant ŝ = N^−1·R mod p
   applied as a Barrett constant. J = −p^−1 mod 2^32 per prime
   (p4: 0x3B7FFFFF). Under §3c-B the pointwise is the twisted conv:
   same REDC accumulation structure, same single stray R^−1 per
   output — the fold is unchanged.

Measured (radix-2 stage, data RW + twiddle traffic, this machine):

                          ALU-chain     L1 stage   L2 stage   DRAM stage
    shoup table (8B/tw)   0.48 ns/vec   0.054      0.117      0.32-0.34
    mont  table (4B/tw)   0.63 ns/vec   0.055      0.081      0.209
    barrett const (none)       —        0.040      0.054      0.225  ns/pair

## 3. Transform structure (T1 + T2 together)

a) BACKBONE: vdH truncated FFT exactly as FLINT structures it — DIF
   forward natural→bit-reversed-blocks, matrix split k = k1 + k2 with
   truncated COLUMN transforms (per-column itrunc/otrunc counts) and
   recursive ROW transforms; inverse mirrors with the force-zero tail
   logic. Truncation grid = one BLK of 256 points (1 KB) — measured
   FLINT scalloping at this grid: +14% at a depth step, decaying.
   The column/row recursion IS the cache blocking; tier thresholds
   follow our measured L1/L2/L3 ladder rather than fixed k1 = k/2.

b) INTERIOR RADIX 8 (T2): with 32 zmm registers a radix-2^3 pass (3
   levels per memory round-trip) fits with room — 8 data vectors +
   3 broadcast twiddle pairs (W1, W2, W4 block scalars; W1·ω8 derived
   by one Barrett-const mul with the fixed ω8) + p/2p + temps ≈ 22
   registers. 12 mulmods per 8 vector-points = 0.5 mulmod/point/level,
   identical count to radix-4 but 1.5× fewer memory passes than
   radix-4 and 3× fewer than radix-2. The AVX2 references cap at
   radix-4 because 16 ymm cannot hold the radix-8 working set —
   that constraint is gone on zmm.
   TRUNCATION × RADIX-8 RULE: full (non-truncated) nodes — the bulk —
   run the radix-8 kernels; the O(1)-per-level boundary nodes (the one
   partial block column/row at the truncation edge) run radix-4/2
   trunc-variant moths, FLINT-macro style (16 (itrunc,otrunc) variants
   at radix-4 vs 64 at radix-8 — variant explosion stays contained
   while >95% of work flows through radix-8).

c) BASECASE — two candidates, bench-decided (both T1-compatible since
   truncation granularity is ≥ BLK = 16 vectors):
   A. In-vector levels: last 4 levels via shuffled butterflies on
      16-lane vectors (p30x3_2 inlane8 extended: valignd/vpermd pairs
      + repeating-lane twiddle vectors, mul_bsmfxd form). Pointwise
      stays 1 REDC/point. Costs ~3 shuffle + 1 mulmod per point over
      the 4 levels, ×3 transforms.
   B. Vector-stop + twisted conv (zint): the NTT stops at whole-vector
      granularity (N/16 vector-points, NO in-vector levels, NO lane
      crossings anywhere in the transform); the pointwise becomes a
      16-point twisted cyclic convolution per vector pair, mod
      (x^16 − ww) with per-vector twist ww from the block-scalar
      table. At zmm: 32 vpmuludq per 16 points = 2 mul/point (same
      rate as zint's 8-pt at ymm) + ~3 shuffle/point (valignd sliding
      window over [a | a·ww]), ONCE per point — vs option A paying
      its shuffles+muls in BOTH forward transforms and the inverse.
      Static count favors B: conv16 ≈ 32 vpmuludq + ~33 shuffles per
      16 points ≈ 1.1–1.3 cyc/point ONCE, vs A paying ~4 in-vector
      levels (~0.2–0.3 cyc/point/level, shuffle-laden) in ALL THREE
      transform passes ≈ 3 cyc/point — roughly −1 ns/limb across 5
      primes, plus B drops the root-depth requirement by 4 bits and
      deletes the in-vector trunc bookkeeping (§3d). Risk: shuffle-
      port pressure of the conv window — bench confirms.
   Recommendation: build B first (zint proves it end-to-end; §3d
   settles the truncation math), keep A as the fallback kernel set.
   VERIFIED IN CODE: tests/p30_tft_model_test.c is a scalar model of
   exactly this pipeline (radix-2 vdH TFT fwd+inv over y, W=4 lanes,
   twisted conv pointwise at w[k], 5-prime Garner, 64-bit chunking) —
   2570 GMP multiply differentials green (exhaustive 1..48^2, pow2-
   boundary totals lg 4..12, all-0xFF to 30000x30000, random shapes),
   plus exhaustive-trunc TFT round-trips and forward==DFT checks.

d) WHY §3c-B COMMUTES WITH TRUNCATION (the math, so it is settled):
   With y = x^16:  Z_p[x]/(x^N−1) ≅ S[x]/(x^16−y), S = Z_p[y]/(y^M−1),
   M = N/16. Operand a(x) = Σ_l A_l(y)·x^l; lane polynomial A_l
   collects x-indices ≡ l (mod 16); one memory vector = the lanes'
   common y-coefficient = 16 CONSECUTIVE LIMBS (input codec unchanged).
   - The y-NTT applies one length-M transform per lane with broadcast
     twiddles: SIXTEEN INDEPENDENT SCALAR TFTs IN LOCKSTEP. The vdH
     forward/inverse math (including the inverse's force-zero steps)
     applies verbatim per lane — no new transform math exists.
   - Truncation translation: deg A_l < ceil((zlen−l)/16) ≤ trunc_vec
     = ceil(zlen/16). The uniform bound is used per lane; the partial
     last input vector zero-fills lanes (free via the masked input
     stage); the inverse precondition "lane coefficients ≥ trunc_vec
     are zero" holds since 16j+l ≥ 16·trunc_vec ≥ zlen. Recovered:
     16·trunc_vec ≥ zlen x-coefficients. Grid = 16 x-coefficients.
   - Pointwise at BR spectral position k: everything is evaluated at
     y = ω^{rev k}, so the product lives in Z_p[x]/(x^16 − ω^{rev k}):
       c_l = Σ_{i+j=l} a_i b_j + ω^{rev k}·Σ_{i+j=16+l} a_i b_j
     (zint's twisted conv, ww_k = ω^{rev k}). ww is indexed in BR
     order ⇒ PREFIX-STABLE table; a truncated run (k < otrunc_vec)
     reads exactly a prefix. Spectral vectors are never partially
     valid (lanes are whole transforms) ⇒ no conv edge variants.
   - Scale: iTFT scale is M^{-1}; the conv's single REDC stray folds
     as ŝ = M^{-1}·R, same mechanism as before.
   - Simplifications vs FLINT: no basecase reordering (spectral
     vector k sits at slot k — no trunc_index swizzle) and no
     in-vector trunc-variant kernels.

e) PASS/PRIME ORDER: per cache tier, prime-major over tier chunks
   (5 spectra at N = 2^22 total 80 MB > L3; one prime's chunk stays
   resident through its stage group). Spectra in one scratch.h
   allocation, prime strides padded off power-of-two alignment.

## 4. I/O codecs

INPUT (limb → 5 residues, normal form): per prime one fused REDC:
t = hi32·R̂₂ + lo32·R̂₁ (R̂₂ = R² mod p, R̂₁ = R mod p; t < 2Rp),
REDC(t) < 3p, one min → [0,2p) — directly in the NTT's lazy domain.
(32-bit chunking's multiply-free reduce — two min-subs — is
acknowledged and declined; see §0.) Fusable with the first NTT level.
IMPLEMENTED + MEASURED (bench/p30_io_bench.c, validated incl. edge
limbs): one fused pass, all 5 primes per 16-limb iteration (the two
limb loads shared across primes), results packed to u32 by one
vpermt2d: 0.34 ns/limb cache-resident, 0.89 at 4.19M limbs (DRAM,
28 B/limb). Fusion beats prime-at-a-time 5-pass by 6% (L1), 22% (L2),
30% (DRAM).

INPUT x FIRST-STAGE FUSION (bench/p30_fuse_probe.c): fusing the decode
into the FIRST BLOCKED-COLUMN pass of the y-NTT removes one full
spectra RW pass. Traffic to "first k1 levels done":
    U unfused             28 + 5*(4+4) = 68 B/limb
    P per-prime fused     5*(8+4)      = 60 B/limb
    F all-prime blockcol  8 + 5*4      = 28 B/limb
F holds the 2^k1 strided LIMB chunks L1/L2-hot across all five primes
(and the top y-NTT levels are twiddle-free/constant in our convention,
so the fused butterflies are nearly free). Measured at 4.19M limbs:
naive F LOSES (4.7 ns/limb, 5.9 GB/s effective: 128-way strided
single-vector visits defeat the prefetchers and the stores pay RFO).
With COLUMN TILING (T = 16 consecutive vectors per row visit, 1-2 KB
bursts) + NT stores for the spectra writes:
    U 3.8-5.4   P 4.2-7.2   F 4.5-4.9   F2-tiled 1.39-1.45
    F2-tiled+NT 1.21-1.43 ns/limb   (k1 in {5,7,9}, robust)
    L3-resident (512K limbs): F2+NT 1.07 vs U 2.58
DECISIONS: (1) input stage = all-prime block-column fused with the
first k1 column levels, T=16 column tiles, NT stores; symmetric on the
inverse side (fuse Garner + carry-out into the LAST inverse column
pass) — together ~80 B/limb off the §7 DRAM budget (~384 -> ~264).
CONFIRMED (user) with the parallelization rationale made explicit:
(a) thread scaling is governed by compute-per-byte — unfused input is
~0.35 ns/limb compute per 28 B (demands ~80 GB/s/core: saturates BW
below one core; threads buy nothing), fused is ~1.1-1.2 ns/limb per
the same 28 B (~3-4x more scaling headroom before the BW wall);
(b) one homogeneous tile loop, zero shared mutable state, no
input->stage barrier — embarrassingly parallel at tile granularity;
(c) per-thread working sets are core-private (L1/L2), produced spectra
chunks are schedulable to the same core for later passes, NT stores
avoid cross-core coherence traffic.
(2) PRODUCTION RULE: every strided column pass in the NTT must be
column-tiled (~1 KB bursts per row visit) — the single-vector
block-column walk loses 3x at DRAM. (FLINT's BLK_SZ = 256 doubles
embeds exactly this: 2 KB per row visit.) (3) k1 is a free parameter
in 5..9 — choose by the real kernel's L1/L2 working set.
P50 NOTE: for the future 4xp50 engine the same probe shape answers the
4-step question — there the input pass writes the TRANSPOSED matrix
directly (decode + tiled transpose through L2), which is the same
tiled-strided kernel skeleton just validated; fused is always better
there (40 vs 104 B/limb) since coefficients are 8 B doubles.

OUTPUT (5 residues → 150-bit coefficient → limbs):
1. No domain exit: iNTT output is normal form; stray R^−1 and 1/N are
   both inside ŝ (§2.3).
2. Garner mixed-radix, ascending primes, all Barrett-by-constant with
   NO pre-reductions (p30x3_2 discipline at 5 primes): digits v0..v4
   via 1+2+3+4 = 10 Barrett-const muls per coefficient, 16 coefficients
   per vector batch.
3. Horner compose c = (((v4·p3+v3)·p2+v2)·p1+v1)·p0+v0 in 64-bit
   chunks (widths 30→60→90→120→150): ~12 vpmuludq + adds per batch,
   emitting three u64 streams (lo, mid, hi) — coefficient j is exactly
   limb-aligned at j (64-bit chunking).
4. Carry-out: z[j] = lo(c_j) + mid(c_{j−1}) + hi(c_{j−2}) + carry —
   two sequential SWAR chain steps (the L2 stream enters the second
   chain pre-delayed one lane via valignd vs the previous vector,
   giving the two-coefficient offset). No scalar u128 loop (the one
   weakness of the reference CRT).

IMPLEMENTED + MEASURED (bench/p30_io_bench.c; validated against
ground-truth 150-bit coefficients in random/all-max/zero patterns,
lazy [0,2p) residue presentation, tails crossing every batch
boundary): 1.02 ns/limb FLAT across L1..DRAM working sets (ALU-bound,
~4.8 cyc/limb), 1.16 at 4.19M limbs.
LESSON (same shape as pq16's flat-emit discovery): the natural fused
16-coefficient loop holds ~30 live broadcast constants and SPILLS
(101 stack refs, 2.35 ns/limb). Restructured STEP-MAJOR through a
256-coefficient L1 batch buffer — each Garner digit step sweeps the
batch with only its own ~10 constants live, then one compose+chain
sweep — 2.3× faster; batch size insensitive (128..512 within 1.5%).
Production rule: keep per-loop constant working sets under the
register budget BY CONSTRUCTION.
Combined I/O ≈ 1.4 ns/limb cache-resident, ≈ 2.0 at the DRAM band —
inside the §7 budget.

## 5. Twiddle storage

Per block node: scalars only — radix-8 nodes store (W1, W2, W4) +
their c_recs = 24 B/node; one node per ≥256-point block per level →
≈ 3·N/32 bytes total across all levels (sub-0.4 MB at N = 2^22, mostly
L2-resident; the stream is ~0.1% of data traffic). FLINT-style
bit-reversed indexed layout (prefix-stable, lazily grown, random-
access — required by T1, see §0 ruler-update rejection). Twist scalars
for §3c-B live in the same BR layout: (ww, ww_rec) Barrett pairs,
8 B per spectral vector = M·8 = N/2 bytes (2 MB at cap, read once per
pointwise ≈ 3% of one spectra pass; derivable smaller via broadcast-
anchor if it ever shows in a profile).
Basecase-A vector tables (if A wins): few KB, L1-permanent.

## 6. Plan, integration, validation

- Thread-local persistent tables; scratch.h for spectra + result
  staging (one allocation, 128B-aligned, power-of-two-stagger padding).
- Entry p30_mul(rp, ap, an, bp, bn, sc) → 0 outside band; dispatch:
  pq16 keeps small/mid band; crossover measured by sweep (expect
  2^15..2^17 total limbs), then u64cvt gets a third leg.
- Validation: GMP differential on the truncation grid (every BLK step
  around each depth boundary — the trunc-variant moths are the
  highest-risk code), adversarial all-0xFF at total = 2^22 (headroom
  1.118), sparse/zero, 300+ random shapes an≷bn, Garner edge
  residues (0 and p−1 everywhere), ASan, pq16 differential in the
  overlap band. Truncation-specific gate: itrunc/otrunc sweep at every
  (k1, k2) split the recursion can produce for N ∈ {2^9..2^22}.

## 7. Roofline (this machine, per limb, top of band)

- ALU: ~22 levels × 5 primes × 0.020 ns/point/level (Barrett block
  stage measured; radix-8 fusion and §3c-B both push it down) ≈ 6.6
  ns + I/O+CRT ~1.5-2 ≈ **8–9 ns/limb** at total = 2^22.
- DRAM: ~3 passes × 20 B spectra + I/O ≈ 250 B/limb ≈ 4–5 ns/limb at
  ~60 GB/s — ALU-bound, thread-scaling headroom.
- Yardsticks at 2^22 total: FLINT fft_small 60–65 ns/limb (measured
  here), zint p30x3 ~75 ns/limb (their Zen4/AVX2 numbers), GMP ~214
  (measured); pq16 ~21 at its 2^17-limb cap.

## 8. Stage-1 implementation (include/p30.h) — status

IMPLEMENTED AND VALIDATED (tests/p30_test.c: 1843 GMP differentials —
exhaustive 1..40^2, totals straddling every pow2-vector boundary
lg 1..14, adversarial all-0xFF including the 2^21 x 2^21 CAP (CRT
headroom worst case), random shapes, band-rejection checks; ASan
clean). Stage-1 shape: radix-2 recursion, canonical [0,p) range
discipline, linear input pass, conv16 pointwise per §3c-B, batched
Garner output, scratch.h workspace, thread-local prefix-stable tables.

One real bug found by the adversarial gate (random data never hit it):
the conv16 wide-REDC 64-bit lane result reaches hi32(16p^2) + p + 1
~ 4.64e9 > 2^32 at all-max products — truncated by the 32-bit pack.
Fix: one min_epu64 fold vs 2p per half BEFORE packing.

Measured (balanced an = bn = n, best-of-8, ns per input limb):
    n:      2048   8192  32768  65536 131072 262144 524288  1M    2M
    p30:    24.5   25.2   26.8   27.8   31.4   32.1   35.9  37.2  39.1
Truncation scalloping: +7-9% just past a depth step, decaying through
the octave (147456: 33.7 vs 131072: 31.4) — no FLINT-style np spikes.
Yardsticks: FLINT fft_small 44.6 (n=65536) .. ~65 (n=2M) -> stage 1 is
already ~1.6x faster across the band and reaches 16x beyond pq16's
cap; pq16 still wins its own band (16.9 at n=65536), as expected.

STAGE 2 QUEUE (the gap to the 8-9 ns/limb roofline): radix-8 interior
passes (3 levels/memory round-trip), input x first-column fusion (T=16
tiles + size-gated NT stores, §4), lazy [0,4p) ranges (remove ~2 mins
per butterfly), leaf flattening (recursion call overhead at m <= 32),
output fusion into the last inverse column pass, then the pq16<->p30
crossover sweep and the u64cvt dispatch leg.

## 9. Stage-2a (radix-8 + lazy ranges) — status

IMPLEMENTED AND VALIDATED (same 1843-case battery + ASan, green):
radix-8 fwd/inv moths for full nodes (3 fused levels/round-trip, 7
broadcast node constants), truncation spine stays radix-2 (containment
rule), lazy [0,4p) between-pass invariant (Barrett eats multiplied
operands raw; the unmultiplied operand shrinks once per level), conv16
b-lanes broadcast from memory (vpbroadcastd m32) with a pre-shrunk b
spectrum, no-op leaf recursions elided.

Measured: overall median gain over stage 1 only 1.06x (best octaves
1.18x); FLINT margin now 1.68x median (min 1.30x). The phase profile
(n = 262144, ns/limb) explains precisely where the time is:
    input 1.7 | fwd 4.3 (x2) | conv 7.1 | inv 4.6 | scale 0.3 | out 2.2
- conv16 is the LARGEST phase: codegen is clean (no spills); it sits
  at its mul-throughput + spectra-pass cost. Stage-2b: fuse conv+scale
  into fwd(a) leaf completion / inv entry (saves one full ra RW pass
  and the rb pre-shrink pass); revisit accumulate width.
- fwd/inv remain ~3x their stage-kernel roofline: per-node setup (14
  broadcast constants per moth amortized over as little as e=1) and
  call structure. Stage-2b: flatten m<=64 subtrees into single calls,
  j==0 specialization, software-pipeline the moth v-loop.
- input/output fusion (probe-validated, §4) still pending.
Stage-2b target: ~12-15 ns/limb; the 8-9 roofline needs the full
flattened schedule.

## 10. Conv-pass microbench: parity-split Karatsuba (adopted)

bench/p30_conv_bench.c. A LINEAR Karatsuba split (a0 + x^8 a1) cannot
pack: 15-wide sub-products overflow an 8-lane half. The PARITY split
works: with y = x^2,  c_even = ae*be + y*(ao*bo),  c_odd =
(ae+ao)(be+bo) - ae*be - ao*bo — all three sub-products are TWISTED
8-point convs mod (y^8 - ww), exactly 8 lanes wide, so TWO spectral
positions batch into zmm halves at full lane use: 24 vpmuludq/position
vs 32 schoolbook. Windows via vpermt2d over [X | ww*X] (8 shared
constant index vectors); b-pair broadcasts via two immediate shuffles.

Measured (single prime, validated vs scalar incl. the Montgomery
stray): schoolbook 10.95 / kara2 9.95 ns/position L1-hot (1.10x),
10.83 / 9.78 streaming (1.11x) — the 25% mul cut nets ~10% after
shuffle/compose overhead.

Two correctness traps found while building it (both documented in the
kernel): (1) msum_acc - m0_acc - m2_acc is NOT exact in the raw
accumulator domain — the wrapped window terms use REDUCED ww*x values,
so the Karatsuba identity holds only mod p and the u64 difference can
go negative; combine AFTER REDC in u32 with a +2p offset. (2) the
engine feeds fa LAZY [0,4p): the kernel must shrink on entry (the
bench validated with canonical inputs and hid this).

INTEGRATED into p30.h (conv loop runs pairs + schoolbook tail).
End-to-end: 32768: 23.7 -> 21.5, 262144: 27.7 -> 26.0, 2M: 34.9 ->
32.3 ns/limb (-6..9%). The remaining conv lever is structural: fusion
into fwd(a) leaf completion / inv entry (kills the spectra pass).

## 11. Stage-2b: fused fwd(b)+pointwise+inverse traversal (user schedule)

IMPLEMENTED + VALIDATED (battery + ASan green; an exhaustive per-(M,
trunc) old-vs-fused bisect harness pinned the one real bug).
Structure: fwd(a) + canonicalize a-spectrum, then ONE depth-first
traversal of b: full subtrees run fwd8 -> (at the 8-vector leaf)
conv-kara pairs -> inv8 while L1-hot, inverse combines on the ascent —
exactly the "0-8 -> pw -> inv 0-8 -> 8-16 -> ..." schedule. The
separate conv pass, its pre-shrink pass, the tail memset and the
standalone inverse descent are gone; the result lands in b's buffer.

THE TRUNCATION TRAP (cost: one wrong implementation round): the
zero-tail property propagates only through z <= h spine chains. A
z > h node's RIGHT child has inverse tails h*d_i = h*c_i — the LEFT
child's outputs, NOT zero. Resolution: fuse full subtrees and z <= h
chains (tails provably zero, the inverse tail-fill vanishes and the
combine collapses to 2*sh2(lo)); the z > h right subtree runs plain
fwd -> conv -> GENERAL truncated inverse, with the parent writing its
tails in between (general inv_rec handles nested cascades itself).
A fully-fused spine needs side-buffer tail propagation — designed,
deferred (worst case today: ~half the transform unfused at tv just
under a power of 2; the max/min 1.19-1.31 in the upper octaves shows
it).

Measured (sweep p30_sweep_s2b.csv vs s2a): median 1.027x, top octave
1.058x — the pass elimination pays where DRAM matters and is neutral
in-cache, consistent with the engine being kernel-bound below L3.

User idea status: (3) fused schedule = DONE (above). (1) in-register
unrolled leaves = PARTIAL (leaf-8 conv fusion in place; the fully
unrolled multi-level leaf is the next kernel lever — fwd/inv are
still ~3x their stage roofline from per-node overhead). (2) NTT
bottom layers via 16-vector batch transpose instead of the twisted
conv: designed — transposing 16 spectral vectors turns the 4 in-lane
levels into vertical ops with CONTIGUOUS table-vector twiddles (w2[j]
for 16 consecutive j loads directly, no broadcasts), pointwise becomes
1 REDC mul/point; op-count is close to the conv but with better mul
structure; needs the bench before adoption.

## 12. Stage-2c: in-register bottoms + iterative DFS scheduler

Idea-1 microbench (bench/p30_leaf_bench.c, all variants validated
against the engine paths mod p). The full-tree leaf-block size is
forced by lg M mod 3 (radix-8 moths from the top): 8 / 16 / 32
vectors. The recursive m <= 32 remainders were the "3x roofline"
overhead — per-node calls, vcc broadcast setup, and (inverse m=2)
the general z>=h path with extra hi/lo writes. L1-hot, ns/pt:

  fused leaf   engine -> winner          plain bottoms  engine -> winner
  m=8   0.919 -> 0.922 (reg, neutral)    fwd16  0.226 -> 0.102 (reg, 2.2x)
  m=16  1.173 -> 0.921 (reg,    1.27x)   fwd32  0.288 -> 0.125 (r2+16r, 2.3x)
  m=32  1.278 -> 0.972 (r2+16r, 1.32x)   inv16  0.331 -> 0.108 (reg, 3.1x)

Adopted kernels (include/p30.h): p30_kara2r (register-input kara2),
p30_fwd16x/p30_inv16x (4 unrolled levels on a live register frame),
p30_fwd16r/inv16r/fwd32r/inv32r (m=32 = radix-2 seam + two reg-16s,
which BEAT the moth + reg-4 split), p30_fused8l/fused16r/fused32r.

Idea-2 microbench (bench/p30_bottom_bench.c): batch-transpose NTT
bottom. Identity verified in code (scalar + SIMD, 512 blocks): the
twisted conv mod (x^16 - w[k]) == 4 more in-lane levels (after a
16x16 transpose) + 1-REDC/pt product + 4 inverse levels, with
twiddles being the SAME prefix-stable BR table read 4 levels deeper
(table VECTORS w[2k], w[4k+2c2], w[8k+2c], w[16k+2c] over 16
lockstep positions, pre-swizzled to 15x[c[16]|rec[16]] block tables).
Result: b-side middle 8.09 vs kara 10.38 ns/pos (1.28x) — but the
mandatory a-side extension costs 3.41 ns/pos, so the seam total is
0.90x L1-hot / 0.84x streaming (3.8KB/block twiddle tables vs kara's
64B). REJECTED; kara conv stays. TRAP for future vector-twiddle
kernels: p30_bar2p assumes a BROADCAST rec — its odd-lane quotient
uses the even lane's rec; per-lane twiddles need bar2p_v with an
extra vpsrlq on the rec plane (bench file has it).

Iterative scheduler (user request: keep DFS order, drop recursion,
zint-style): p30_fwd_full / p30_inv_full / p30_fused_full flatten a
full subtree to a leaf-block walk b = 0..m/LS-1. A size-s moth fires
BEFORE block b iff b == 0 mod s/LS (pre-order, largest first) and
AFTER block b iff b+1 == 0 mod s/LS (post-order, smallest first) —
bit-test on b, node index j*(m/s) + o/s from the prefix-stable
tables (this is where rejecting ruler-sequence updates pays: random
node access is free). Pass sequence is IDENTICAL to the recursion;
only the call tree is gone. The truncation spine stays recursive
(one node per level, depth lg M) and routes its full children into
the iterative engines.

Validation: 1843-case GMP battery + ASan green. Sweep
(p30_sweep_s2c.csv vs s2b): overall median 1.087x (max 1.18x); the
lg = 1,2 mod 3 octaves gain 1.10-1.15x as predicted, lg = 0 octaves
~1.0x (m=8 leaves were already one-pass), single shared top-octave
point -1.3% (DRAM-bound, noise-level). Curve now 20.6 (4K) - 35.8
(2^20 oct median) ns/limb; FLINT/p30 median 1.83x (min 1.44) over 81
shared points; pq16 ahead only in its small-n band (fft16/p30 max
1.21 at the bottom). Scalloping unchanged (1.23-1.32 upper octaves)
— still the unfused z>h spine fraction, next lever = side-buffer
spine tails; after that the fused input column pass (T=16 + NT).

## 13. Stage-2d: side-buffer spine tails + split fused leaves

Side-buffer spine fusion (user: "proceed the z>h spine fraction"):
the z > h right children now run FULLY FUSED via p30_fused_sb, a
traversal carrying GENERAL (nonzero) inverse tails out-of-band. The
anti-dependence that forced the old fwd -> conv -> general-inverse
detour was tails overwriting coefficient storage the forward still
needed; with tails beside the data it disappears. Recurrences (from
the model-test inverse derivation):
  z <= h: child tails  h*c_v = (T_v + w*T_{v+h})*inv2,  v in [z, h)
          combine      m*u_v = 2*h*c_v - w*T_{v+h},     v in [0, z)
  z >  h: right tails  h*d_v = h*c_v - w*T_{v+h},       v in [zr, h)
          parent out   m*u_v = h*c_v + h*d_v
Tail slot frames nest exactly (right child frame = parent + 16h;
child slots coincide with the parent tails consumed in the same
read-then-write loop). The buffer is FREE: every spine node sits at
global spectral base g with g + z == tv, so all tail slots map to
global positions [tv, M) — the a-spectrum's DEAD tail (conv reads
only [0, tv)); since A walks the recursion with the same offsets,
SB == (uint32_t*)A. First cut used a separate bv/2 scratch buffer and
LOST ~2% in the DRAM octaves (extra unique footprint) — the dead-tail
form removed that. p30_inv_rec is no longer on the multiply path
(kept as the reference inverse).

Measured: ~NEUTRAL (median 1.00 vs s2c). The scalloping (1.23-1.34
upper octaves) did NOT move => the unfused-spine hypothesis is
FALSIFIED: scalloping is INTRINSIC truncation cost (just past 2^k
the top level still passes over all M vectors while demand is M/2+e,
and utilization recovers as tv grows toward M). Kept anyway: simpler
single traversal, no general-inverse detour, zero allocation.

Split fused leaves (the actual win hiding underneath): the leaf
bench's fused16/32 "reg" variants had only been compared against the
OLD recursive engine, not against the NEW unfused composition. Added
"split" variants: fwd16r + 8x kara2 (memory) + inv16r = 0.873 ns/pt
vs 0.925 all-register (m=32: 0.915 vs 0.975) — holding the 16-vector
data frame across the register-hungry kara costs more in spills than
the block's L1 load/stores. Bench-matrix lesson: when a baseline
component improves, re-run the fusion comparison against the NEW sum
of parts. Engine leaves switched to split form
(p30_fused16l/p30_fused32l).

Validation: 1843-case battery + ASan green (both steps). Sweep
p30_sweep_s2f.csv vs s2c: overall median 1.023x (split leaves ~+2.6%,
sb fusion ~neutral), positive in 10/11 octaves; [2^20,2^21) -0.6%
within the +-2% run-to-run noise measured there. Curve 19.5 (4K) -
36.5 (2^20 oct median) ns/limb.

## 14. Stage-2e: fused input column pass + canonicalize elimination

Canonicalize elimination (free win at every size): the standalone
a-spectrum shrink pass is gone -- each b...a-spectrum vector is read
by exactly ONE conv, so kara2 (both forms) now shrinks its fb block
in registers ([0,4p) -> [0,p), 2 mins per vector = the same ops as
the old pass with zero memory traffic); the two conv16 sites (m == 1
spine leaves) canonicalize through a 64B stack temp because conv16
broadcasts fb lanes from memory. Worth +0.3-2.7% below the fusion
gate, included in the gated octaves' gains.

Fused input (p30_input_f3 + p30_cfwd8x): one tiled block-column pass
does the 64-bit decode AND the first 3 forward levels (k1 = 3 = one
radix-8 moth depth -- radix-8 alignment makes the downstream seam
trivial). Matrix view M = 8 rows x C = M/8 columns; per 16-column
tile, per prime: decode row-bursts (2KB limb reads, the 16KB limb
tile stays cache-hot across the 5 primes) into an 8KB L1 staging
tile, run the TRUNCATED 3-level network in-tile (node twiddles w[0],
w[2], w[4], w[6]; row demand zrow = ceil(tv/C), rows >= zrow neither
finished nor stored), then row-burst stores (NT when gated). Two
implementation lessons, both measured: (a) the column-INNER loop
(64B per row visit) lost ~12% in DRAM octaves vs the probe's
row-burst pattern -- the L1 staging tile restores it (the probe's
production rule, re-learned in vivo); (b) NT stores below ~L3-sized
working sets FORCE DRAM round trips the cache would have absorbed:
NT at M = 2^16 cost -22% in [2^19,2^20).

Downstream seam: pre-depth drivers (p30_fwd_pre / p30_fused_pre /
p30_fused_sb_pre / p30_fused_full_pre) enter the traversals d = 3
levels down -- fwd passes those levels would have done are skipped,
ALL inverse-side work (spine tails, combines) is unchanged; full
nodes at d == 3 dispatch 8 fused_full children + one inv8 moth.
p30_fwd_full/p30_inv_full gained m in {1,2,4} branches (pre-entry at
small M produces tiny subtrees; their silent no-op was caught by the
forced-low-gate battery: 28/1843 failures, spill-guard rejections).

Gating (tuned by sweep grid): fused input + NT both at M >= 2^17
(P30_FIN_MIN_M / P30_FIN_NT_MIN_M, overridable). On THIS machine the
64MB L3 absorbs the unfused passes below that (fused lost 2-12%
mid-band when gated low); the win region is where spectra + limbs
exceed L3. Tests force the gates low (M >= 8) AND keep defaults, so
both paths stay battery + ASan green.

Measured (p30_sweep_s2h.csv vs s2f): gated octaves +4.0% / +7.7% /
+3.5% ([2^19..2^22)); top-octave medians now 31.9 / 34.0 / 32.0
ns/limb. BONUS: scalloping in the gated octaves collapsed (1.265 ->
1.121, 1.337 -> 1.129) -- the column pass's cost is proportional to
zrow, so the truncation overhead of the absorbed top levels
disappears. Cumulative stage1 -> s2e: median 1.272x (max 1.371);
FLINT/p30 median 1.903x (min 1.447). Next: output fusion (the
symmetric trick on the last inverse + Garner), crossover sweep,
threading (the fused input pass is the tile-parallel template).

## 15. EXPERIMENTAL: truncated 3*2^k NTT (mixed-radix ladder)

Idea (user): cover tv in (2*2^k, 3*2^k] with N = 3*2^k transforms and
(3*2^k, 2^(k+2)] with binary ones -- transform-size overage drops
from max 2.0x to max 1.5x, the smoothest dispatch an FFT multiply
can have. SCALAR MODEL VALIDATED: tests/p30_tft3_model_test.c.

Why it composes cleanly: the coverage window (2L, 3L] guarantees the
radix-3 level is ALWAYS FULL (all 3 children demanded) -- no
truncated radix-3 variants exist anywhere; truncation lives entirely
inside binary child 2, where the standard vdH machinery applies
unchanged. The ladder picker even proves this automatically: 3*2^j
is chosen over 2^m only when tv > 2*2^j.

The ONLY new math is the inverse-side radix-3 seam. Children moduli
(y^L - zeta^t): b^t_i = a_i + zeta^t a_{i+L} + zeta^{2t} a_{i+2L}.
After FULL inverses of children 0,1 and with parent tails zero
(a_{i+2L} = 0 for i >= z2 = z-2L), a 2x2 Vandermonde solve per i:
    a_{i+L} = (b0_i - b1_i)/(1 - zeta);  a_i = b0_i - a_{i+L}
yields the demanded outputs AND child-2's coefficient tails
b2_i = a_i + zeta^2 a_{i+L}; child 2 then runs the standard truncated
binary inverse, and a plain 3-point combine finishes i < z2. No
order-of-operations hazard (the seam reads only children 0,1 regions
and writes only child-2 tails / final outputs). Scaling: subtree
inverses return L*b^t, the /3 sits in the 3-point combine, one
global /L at the end.

Model results (both 3-capable primes 918552577, 943718401; roots as
exponents of Omega, order 3*2^16, no table-design commitments):
eval gate, round trips over EVERY z2 (k <= 6), 186 product-vs-
schoolbook cases per prime, and a tv = 5..192 ladder-dispatch sweep
(every size verified against both binary-only dispatch and
schoolbook) -- ALL GREEN. Model harness trap worth recording: the
truncated forward leaves intermediates above tv; the inverse
contract needs those tails ZEROED for a product (the engine's fused
traversal does this implicitly; a standalone pipeline must memset).

Mult-count smoothness (model, conv-dominated so understated):
max/min over tv >= 48: ladder 1.18-1.19 vs binary-only 1.22-1.24.
The real prize is at DRAM-bound sizes where full-span top passes
cost ~ M/tv: worst overage 2.0 -> 1.5.

PRODUCTION PRIME REALITY: only 918552577 and 943718401 of the
current set have 3 | p-1. Scan of < 2^30 prime sets:
  - 3*2^22 | p-1 (one unified set, full band + ladder): top-5 product
    2^148.1 -> min-operand cap 1.13M limbs: LOSES half the balanced
    band. Rejected.
  - TWO PRIME SETS (the design answer): keep the current set for
    binary sizes; mixed sizes use the 3*2^20-capable top-5
    {962592769, 972029953, 975175681, 1012924417, 1053818881},
    product 2^149.45 -> cap 2.86M limbs (BETTER than the current
    2.34M). Mixed rungs need root order 16*3*2^16 = 3*2^20 exactly
    (largest in-band rung is 3*2^16 vectors). Cost: a second plan's
    twiddle tables, zero runtime overhead per multiply.

Engine integration (stage 3, not started): radix-3 vector butterfly
pass kernel (broadcast zeta twiddles); per-child root tables (the
prefix-stable BR construction seeded at zeta^t, i.e. Omega-offset
tables); the fused traversals/iterative DFS/kara conv apply to the
binary children unchanged; fused-input column pass needs a 3x- or
24-row variant; dispatch by ladder.

## 16. Stage 3: mixed-radix 3*2^k ladder IN THE ENGINE

Integration of §15 into include/p30.h (gated production + full
machinery for the future):

Architecture: TWO complete prime-set engines in the plan (p30_pset:
primes, twiddles, Garner): binary sizes use P30_PR as before; mixed
sizes use P30_MPR = {962592769, 972029953, 975175681, 1012924417,
1053818881} (3*2^20 | p-1, cap 2.86M limbs; bounds re-audited for
p up to 1.054e9: conv wide-REDC reaches 2.93p < 3p -- kshrp3 covers
it; the Garner acc 4p < 2^32 holds with ~2% margin).

Children TWISTED to cyclic form so ALL binary machinery (tables,
iterative DFS, fused traversals, fused_sb, kara) applies verbatim
with local j = 0. THE KEY CORRECTION found during integration: the
twist must be the RING SUBSTITUTION x -> nu*x (nu^(16L) = zeta^t),
NOT a per-vector mu^v scaling -- a vector-only twist commutes with
the y-transform but BREAKS ON THE CONV'S X-CARRIES (lanes l1+l2 >= 16
carry into vector v+1 and pick up mu^(v-1)). Factorization
nu^(16v+l) = mu^v * nu^l: per-vector broadcast tables (packed per
rung -- strided 8B reads into the master table thrashed) plus one
FIXED 16-lane vector per child per rung applied with p30_bar2pv (the
per-lane-rec Barrett). The scalar model never caught this because it
passed roots as exponents and twisted nothing.

Inverse seam (p30_mfused): children 0,1 fused_full; per-vector 2x2
solve with untwist/retwist folded in; child-2 tails ride the
a-spectrum dead tail exactly like the binary spine (the frames line
up at A + 2L); child 2 runs fused_sb; unscaled 3-point combine gives
the x3 for free, final scale (3L)^-1 * R.

Bugs found by the gates: (1) the x-carry twist (above) -- caught by
the GMP battery, localized by a per-prime bisect harness
(/tmp/m3dbg.c: children-vs-scalar-ref forward, per-prime product);
(2) p30_output's SCALAR TAIL hardcoded P30_PR in the Garner walk and
the u128 CRT compose (m0 AND the easy-to-miss m1 line) -- mixed
results were correct except limbs in the last zlen mod 256 block /
the final carry limb.

MEASUREMENT LESSON: cross-process sweep comparisons on this machine
carry up to 9% layout/THP luck in DRAM octaves (identical binary
code, s3 vs s3b). The decisive numbers come from an IN-PROCESS A/B
(bench/p30_ab.c, runtime p30_ladder switch): ladder wins +8.3..11%
across the [2^18, 2^19) limb octave -- binary M = 2^16 working set
(~44MB) spills L3 while mixed 3*2^14 (~33MB) fits -- and loses
1-13% elsewhere (cache sizes: ~2 extra RW passes + ~8 twist barretts
per vector outweigh the 25% size cut; top octaves: the mixed path
has no fused-input column pass yet). Within each window the ratio
improves monotonically toward tv = 3L, as the utilization model
predicts.

Production gate: P30_MIX_MIN_LG = P30_MIX_MAX_LG = 16 (the win
octave). Gated sweep p30_sweep_s3c.csv: [2^18,2^19) median +6.9%,
scallop 1.333 -> 1.218; other octaves unchanged. Battery + ASan
green at the default gate AND with -DP30_MIX_MIN_LG=5 (every rung
exercised, all z2 shapes via gates 2/3).

To widen the window (queued): port the fused input column pass to
mixed (decode + radix-3 + twist + first binary levels in one pass --
would attack the top-octave losses), and merge the twist tables to
full per-vector lane vectors (one bar2pv instead of bar2p+bar2pv per
twist site, 4 sites saved ~4 barretts/vector) for the cache-size
losses.

## 16b. Widening attempts: mixed fused input (p30_input_m3) -- measured, DISABLED

Built and validated (battery + ASan green with forced gates): a
24-row tiled pass (3 children x 8 sub-rows, C = L/8 columns) doing
decode + radix-3 + twist + the first 3 binary levels of each child
in one sweep; downstream enters per child through the existing
pre-depth drivers (p30_mfwd_pre / p30_mfused_pre, d = 3) -- the
whole r3fwd pass disappears. Also added: pre-depth mixed drivers and
the P30_MFIN_* gates.

MEASUREMENT CONFOUND (the session's hard lesson): the in-process A/B
that first suggested m3+NT "closed the top-octave gap to 1.00" was
built while P30_MIX_MAX_LG still defaulted to 16 -- at lg 17/18 BOTH
arms dispatched binary. ALWAYS VERIFY THE EXPERIMENTAL ARM ENGAGES
(instrument the dispatch, don't infer from ratios). Interleaved
single-config process runs (3x each, /tmp/p30_pt.c) give the truth:
  lg 17/18 mixed, old path (input + r3fwd):   0.91-0.96 vs binary
  lg 17/18 mixed, m3 + NT:                    ~0.84-0.88
  lg 16 mixed, old path:                      +8-10%  (the win)
  m3 at lg 16 (no NT, L3-resident):           ~0.85
m3 loses at every size: below L3 its 24 strided RFO store streams
cost more than the pass it saves; past L3 with NT it still trails
the old path. Suspect: the store pattern (24 streams x row bursts)
plus in-tile staging; future work = restructure before re-enabling.

FINAL PRODUCTION CONFIG: P30_MIX_MIN_LG = P30_MIX_MAX_LG = 16 (the
L3-crossing octave), old mixed path, m3 disabled
(P30_MFIN_MIN_L = 1<<20; tests force it low so the code stays
battery + ASan green). Final sweep p30_sweep_s3e.csv: [2^18,2^19)
median +6.5%, scallop 1.333 -> 1.216; other octaves unchanged within
the documented cross-process noise.

## 16c. x6 mixed fused input (user-shaped: 3x2 first layer, 6 streams)

User call: 24 output streams was too aggressive; absorb only the
3x2 first-layer decomposition -> SIX row streams (binary f3's 8 is
the proven regime). Implemented as p30_input_x6: NV = 3L vectors as
6 rows (child t, half r1) x C = L/2 columns; per column: decode,
radix-3 + twist, then the binary top level of each child -- which is
TWIDDLE-FREE (cyclic child, w[0] = 1, pure add/sub). Bonus from the
d = 1 entry: the absorbed level's inverse combine (also twiddle-free)
FUSES into the radix-3 seam pass (p30_mfused_x6: one pass does the
d=1 combines of children 0,1 AND the seam solve, region-split at
b = max(z2-C, 0) and e = min(z2, C)), so the inverse side costs no
extra pass. Children enter the existing pre-depth machinery at d = 1
(fwd_pre / fused_sb_pre; child halves are nodes (L/2, j=0/1) of the
standard tables). Battery + ASan green at all gate combinations.

Three-way interleaved single-config runs (bin / old-mixed / x6):
  small + (66..96k):  x6 beats old-mixed by 2-4% (m3's pathology
                      fixed) but binary still wins -> gated off
  lg=16 (267..383k):  old-mixed +8-10% over binary; x6 LOSES 15-18%
                      to old -- the working set is L3-RESIDENT, so
                      column fusion has no bandwidth to harvest and
                      pays its access pattern (the exact same lesson
                      as the binary f3 gate at M >= 2^17)
  lg=17 (534..765k):  x6+NT == old == binary (0.995..1.007)
CONFIG: P30_MFIN_MIN_L = 2^15 (x6 only past L3, where it is the best
mixed form); production mix gate unchanged at lg = 16 with the plain
path. Net production behavior identical to §16b's; the x6 machinery
is the correct input-fusion form whenever the mix window widens.

Standing conclusion on the ladder: the 3*2^k rung pays where it
moves the working set across a cache boundary (+8-10% at lg 16) and
is otherwise neutral-to-slightly-negative on this machine's deep
cache hierarchy; its real remaining upside is multi-threading (less
footprint per multiply) and machines with smaller L3.

## 17. Stage-3 cost-structure analysis (phase profile + untried levers)

Phase profile (bench/p30_phase.c, ns per input limb, balanced):
  n=32K  (M=2^12):    in 1.27 | fwd(a) 3.45 | fused(b) 14.30 | scale 0.35 | out 2.40 | sum 21.8
  n=256K (M=2^15):    in 1.74 | fwd(a) 4.40 | fused(b) 16.49 | scale 0.34 | out 2.25 | sum 25.2
  n=2M   (M=2^18 f3): in 4.48 | fwd(a) 5.09 | fused(b) 19.15 | scale 1.21 | out 2.49 | sum 32.4

Roofline status: input/output at their io-bench kernel pace (out is
2x because the bench counted output limbs); fwd at leaf-kernel pace
+~11%; conv inside fused(b) ~= 7-8 ns/limb = kara kernel pace +15-20%.
CONCLUSION: at cache sizes the engine is KERNEL-BOUND end to end --
further single-thread gains there require WORK REDUCTION, not
scheduling. At 2M, ~10 ns/limb is DRAM-attributable; remaining
fusible passes ~= 3-4 ns/limb. The original §7 roofline (9-11
ns/limb) undercounted the pointwise: kara conv ALONE is 6-8 ns/limb
(windows + REDC + combines, not just products). Revised realistic
single-thread floor for this algorithm family: ~18-20 ns/limb in
cache, ~26-28 at the band top.

## 18. Cleanup: p30.h (clean production) / p30_ref.h (superset)

Pre-Tier-1 cleanup (user call; also: Tier-2 items 4 and 5 from §17
are REJECTED on review -- full output fusion needs 10+ concurrent
streams in the final all-prime block pass, and a/b subtree
interleaving doubles the effective transform footprint in cache).

p30_ref.h = frozen superset: production + every validated-but-
dormant path (p30_inv_rec / p30_inv_full standalone general inverse,
m3 24-row and x6 6-row mixed fused-input passes + their pre-depth
drivers + gates). Internals benches (conv / leaf / bottom) build
against it. p30_conv_bench's local development copy of conv16_kara2
renamed (it had silently collided with the integrated engine since
stage 2a -- the bench had not been rebuilt since).

p30.h = clean engine, 1927 lines (from 2438): exactly the production
choices -- fused input f3 (gated M >= 2^17 + NT), iterative DFS with
in-register bottoms, fused traversal + side-channel spine, kara
conv, scale + batched Garner output, and the 3*2^k ladder (gate
lg = 16, plain mixed path with packed rung twist tables). Header
rewritten from changelog form into an architecture document; the
truncated-inverse invariant kept as load-bearing documentation with
a pointer to the reference implementation.

Parity: battery + ASan green (default and -DP30_MIX_MIN_LG=5),
phase profile and full sweep identical to pre-cleanup (median 1.005,
within process noise). Next: Tier 1 (scale-into-Garner fold,
p30_sqr squaring path, pq16 crossover dispatch) on the clean file.
