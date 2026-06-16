# p50x4 NTT design — I/O stages (v1)

Shape (user-specified): **80-bit trunks, 4 primes < 2^50, IFMA52
arithmetic, W = 8 lanes (64-bit), band to ~2^40 limbs** — virtually
unbounded for practical inputs. This document settles the I/O design
(input reduce, SIMD Garner output) plus the arithmetic toolkit; the
transform itself reuses the p30 machinery shapes (vdH truncated y-TFT,
fused traversal, iterative DFS) and is specified only where p50
differs.

## 0. Parameters (computed)

Primes: the largest four p < 2^50 with 2-adic valuation >= 40
(ascending — Garner order is load-bearing):

    p0 = 1025844348715009 = 933*2^40 + 1   (lg 49.866)
    p1 = 1072023837081601 = 975*2^40 + 1   (lg 49.929)
    p2 = 1086317488242689 = 247*2^42 + 1   (lg 49.948)
    p3 = 1108307720798209 =  63*2^44 + 1   (lg 49.977)

- product P = 2^199.72; min-operand cap = P / (2^80-1)^2 =
  2^39.72 trunks = **2^40.04 limbs** (~9 TB operand).
- all satisfy 4p < 2^52: the p30 lazy range discipline [0,4p)
  transfers VERBATIM into 52-bit IFMA lanes — this is *why* p50, not
  p52.
- roots: min valuation 40 -> transforms to 2^40 points = 2^37 vectors
  of W=8, covering totals to 2^40 * 80/64 = 2^40.3 limbs. Root limit
  and CRT cap coincide at ~2^40 limbs; nothing is wasted.

Measured on this machine (Zen 5): vpmadd52{lo,hi} throughput
~1.75/cycle (nominal 2, dual pipe), latency 4 cycles — full-rate.
AVX512-VBMI2 (vpshrdvq/vpshldvq funnel shifts) available: the
trunk/limb regrouping below leans on it.

**IFMA vs DOUBLE-FMA shootout** (user question: is the FLINT
fft_small double-NTT arithmetic faster? bench/p50_arith_bench.c,
validated kernels, ping-pong streams -- xor-reduce and offset tricks
both got algebraically folded by clang; values must EVOLVE across
reps for an honest mult-pipe-bound measurement):

    T1 mulmod stream   IFMA 0.0373 | dbl 0.0499 ns/elem   -> 1.34x
    T2 lazy butterfly  IFMA 0.0369 | dbl 0.0472 ns/elem   -> 1.28x
    T3 dot8 (conv acc) IFMA 0.329  | dbl 0.568  ns/out    -> 1.72x

T1 IFMA sits exactly on the madd-pipe bound (3 madd52 / 2 per cycle
per 8 lanes). The decisive gap is T3: madd52lo/hi accumulate 104-bit
columns IN-INSTRUCTION, while doubles must fully reduce every product
before a float add can touch it (the (h,l) splits are exact but their
sums are not) -- and the p50 design is conv-heavy by construction.
Per payload bit, IFMA mulmod = 1.87 ps/bit vs p30's 1.92 (parity, as
estimated) vs double's 2.5. VERDICT: IFMA52, not double, on Zen 5;
double would re-enter consideration only on cores without full-rate
vpmadd52 (and would then also push toward FLINT's depth-full
transform shape to avoid convs entirely).

## 1. Arithmetic toolkit (52-bit lanes)

**Barrett-by-constant, no pre-reduction (the p30 lemma at R = 2^52).**
rec = floor(c * 2^52 / p); q = madd52hi(0, a, rec) = floor(a*rec/2^52).
For ANY a < 2^52: a*rec/2^52 > a*c/p - a/2^52 > a*c/p - 1, so
q >= floor(a*c/p) - 1 and r = a*c - q*p in [0, 2p). Compute r in the
low 52 bits (r < 2p < 2^51, exact):

    q = madd52hi(0, a, rec)
    r = madd52lo(madd52lo(0, a, c), q, pneg) & M52     // pneg = 2^52 - p

3 madd52 + 1 and per vector of 8. Same role as p30_bar2p: block
twiddles, Garner constants.

**REDC52** (pointwise / input only, R = 2^52, J = -p^-1 mod 2^52):
same wide-REDC pattern as p30_kredc with the carry-from-low trick;
accumulators for conv8 fit: 8 * (2p)^2 < 2^105 -> two 52-bit column
accumulators per output (lo/hi columns accumulate madd52lo/hi
DIRECTLY — IFMA's fused accumulate eats the adds p30 spends).

**Ranges**: between-pass lazy [0,4p) < 2^52; sh2-analog folds via
64-bit min; butterflies identical in structure to p30 (sh2 + barrett
+ add/sub + mask).

## 2. Input stage (settled, user's shape + Montgomery refinement)

Trunk t = bits [80t, 80t+80) of the operand. 80*4 = 64*5: groups of
4 trunks align to 5 limbs; in-group bit offsets cycle {0,16,32,48}.

1. **Extract**: lo-limb index of trunk t is floor(5t/4) — gather limb
   pairs (A, B) with two vpermt2q from limb vectors, then ONE
   vpshrdvq(A, B, {0,16,32,48,...}) funnel shift per output. Split
   hi = trunk >> 28 (52 bits), lo = trunk & (2^28 - 1).
2. **Reduce** (per prime, trunk registers shared across the 4 primes
   like p30's input): with C = 2^80 mod p precomputed,

       res = REDC52(hi * C) + lo        // = hi*2^28 + lo (mod p)

   since REDC52(hi*C) = hi * 2^80 * 2^-52 = hi * 2^28 (mod p), < 2p,
   and lo < 2^28: one fold to [0,2p), two to canonical. ~7 madd-class
   ops per (trunk, prime).

Traffic: 8B/limb read + 4 * 8B residue per trunk = 32B/limb — vs
p30's 28B (52-bit residues ride in 64-bit lanes; 12 idle bits per
lane is the price of IFMA). Same fused-first-levels / NT-store
options apply as in p30 (gate past L3).

## 3. Output stage: the SIMD Garner (the hard part, settled)

KEY DECISION: **commit to TRUNK RADIX (2^80) for the carry domain.**
p30's pattern — 149-bit per-coefficient values -> 3 limb-aligned
64-bit streams -> SWAR carries — generalizes verbatim: p50's 199.7-bit
values become **3 trunk-aligned 80-bit streams**; u64 limbs only
appear in a final bit-regroup of carry-NORMALIZED trunks, where
4 trunks = 5 limbs exactly. Five stages, all batched through an L1
buffer of 8-coefficient vectors (step-major, only each stage's
constants live — the p30 register-budget rule):

**(a) Digits.** Standard Garner, ascending primes, no-pre-reduction
Barrett52: v0 = r0; v_i = (r_i + 2p_i - t_i) * inv_i mod p_i with
t_i accumulated via ppm[i][k] constants. 6 Barrett muls per
coefficient (i = 1..3), digits canonical < p_i < 2^50.

**(b) Compose, radix-2^52 IFMA columns.** Weights in 52-bit words:
W1 = p0 (1 word), W2 = p0*p1 (2), W3 = p0*p1*p2 (3) — 6 words total:

    col_m   += madd52lo(v_k, w_{k,m})
    col_m+1 += madd52hi(v_k, w_{k,m})

12 madd52 per 8 coefficients; each column <= 6 terms * 2^52 < 2^55.

**(c) Normalize.** Three sequential carry steps
(c_{m+1} += c_m >> 52; c_m &= M52) -> exact radix-52 digits c0..c3
of V < 2^199.8 (c3 < 2^44).

**(d) Regroup 52 -> 80.** Pure shifts/ors (no arithmetic): each trunk
word held as a (lo64, hi16) lane pair:

    b0 = V[ 79:  0] = c0 | c1[27:0]<<52          (+ hi16 from c1)
    b1 = V[159: 80] = c1[51:28] | c2<<24 | c3[3:0]<<76
    b2 = V[199:160] = c3 >> 4                    (< 2^40, one lane)

**(e) Streams + carry + limbs.** T_t = b0_t + b1_{t-1} + b2_{t-2}
(delayed adds via valignq, exactly p30's L2-delay pattern) < 3*2^80;
radix-2^80 SWAR carry chains on the (lo64, hi16) pairs (carry-out =
hi >> 16; generate/propagate masks as in p30_chain_step, propagate
condition trunk == 2^80 - 1). Then the only place limbs exist:
carry-normalized trunks regroup 4 -> 5 limbs with vpshldvq funnels
(shift cycle {0,16,32,48}) + one compaction permute, and a scalar
tail + flush identical in shape to p30's.

Cost estimate per coefficient (8-lane batched): ~6 barretts (a) +
12 madd (b) + ~12 ops (c,d) + ~10 ops (e) ≈ comparable per payload
bit to p30's output (which runs 1.0-1.2 ns/output-limb); target
~1.5 ns per output limb-equivalent.

## 4. Transform notes (delta from p30 only)

W = 8, y = x^8; pointwise = conv8 mod (x^8 - w[k]): schoolbook is 8
windows x madd52lo/hi pairs accumulating IN-INSTRUCTION (no separate
adds); parity-split Karatsuba applies as in p30 (3 twisted-4 convs)
if the bench says so. Tables: (c, rec) u64 pairs, 16B/node — same
bytes per payload bit as p30. Twiddle convention, prefix-stable BR
tables, iterative DFS, fused traversal, side-channel spine tails:
all transfer unchanged (the machinery is radix-agnostic given the
toolkit above).

## 5. Honest in-band comparison vs p30

Per payload bit: butterflies ~parity (p50: ~8 instr / 8 lanes / 20
payload bits; p30: ~9 / 16 / 12.8); conv ~20-35% cheaper (length 8
vs 16, IFMA fused accumulate); spectra bandwidth 28% WORSE (25.6 vs
20 B/limb). Verdict: p50 is **the beyond-band engine** — its reason
to exist is the 2^40-limb cap vs p30's 2^21.2 — and roughly
performance-parity in-band on this machine (ALU win eaten by
bandwidth at DRAM sizes). Expected dispatch: p30 to its cap,
p50 above; crossover measurement decides the exact seam.

## 6. Protocol (next steps, in order)

1. Scalar model test (mirror tests/p30_tft_model_test.c): W=2-lane
   model with 80-bit trunks, 4-prime Garner, GMP differentials —
   prove trunk I/O + transform math end to end before SIMD.
2. Kernel microbenches: Barrett52 butterfly, conv8 (school vs kara),
   input reduce, output stages (a)-(e) — pin ns/limb before engine
   work, p30-protocol style.
3. Engine: port the p30 skeleton (it is deliberately radix-agnostic).

## 7. I/O IMPLEMENTED + MEASURED (bench/p50_io_bench.c)

Both kernels implemented exactly as designed, parameterized by trunk
size T -- the user's ladder conjecture holds: T in {80, 84, 88} runs
through ONE table-driven code path (plan constants per T; alignment
cycles 4/5, 16/21, 8/11 trunks/limbs). Caps: 2^40 / 2^32 / 2^24
limbs: pick the largest T that fits the operands, ~64/T of the
transform work per limb (88 vs 80 = 9% less).

Validation: input vs scalar trunk%p (65536 trunks, all T); output vs
scalar limb accumulation, random AND all-max coefficients (the carry
chain's adversarial case). One real bug found by the all-max/period
pattern: VPSHRDVQ counts are MOD 64, so limbs with in-trunk offset
s >= 64 silently funneled from the wrong source -- fixed with a
per-lane source-swap mask ((lo,hi) -> (hi,0)); everything else
worked as designed, including the sllv auto-zero trick.

Measured (ns per limb of the processed operand, solo):

            input  L2 / DRAM      output L2 / DRAM
    T = 80   0.516 / 0.772         1.764 / 3.199
    T = 84   0.490 / 0.767         1.688 / 2.980
    T = 88   0.471 / 0.757         1.630 / 2.749

Input is at p30's class (0.34-0.89) despite 60% more residue bytes.
Output v1 is a 3-pass structure (b-streams spilled, trunks spilled,
then limbs); the DRAM number is pass-traffic-bound -- production
fusing (chain in-flight valignq-style like p30, pass 2+3 merged)
should land ~1.5-2.0 ns/limb. Good enough to green-light the
transform work; the I/O problem is solved.

## 8. Transform leaf kernels measured (bench/p50_kernel_bench.c)

    conv8: schoolbook 4.32 | kara 6.28 ns/pos  -> SCHOOLBOOK WINS 1.45x
    fwd8 moth: 0.0406 ns/coeff/level

- The kara optimum FLIPPED vs p30: with madd52's fused accumulate,
  products are nearly free; kara's 3 REDCs + permute traffic + extra
  twiddle muls cost more than the 25% products it saves. p50 uses
  SCHOOLBOOK conv8 (sliding [aw | a] valignq windows, b broadcasts,
  one wide REDC52 -- uniform R^-1 stray as in p30). Validated incl.
  lazy/adversarial inputs; vector (c, rec) twiddles are safe here
  (madd52 is lane-wise; no p30 broadcast-rec trap).
- Butterfly parity with p30 is EXACT per payload bit:
  0.0406/20 = 0.026/12.8 = 2.03 ps/bit-level.
- Conv per limb: 4.32/8 * 0.8 coeff/limb * 4 primes = **1.73 ns/limb
  vs p30's 6.5-8** -- conv cost scales with W, and W = 8 halves it
  while payload/coeff rises 56%.

REVISED in-band verdict: the §5 "parity" estimate assumed conv
parity; it is actually ~3.7x cheaper per limb. Projection at cache
sizes: transforms ~10.5 (per-bit parity, +5% extra levels) + conv
1.7 + I/O 2.7 = ~15 ns/limb vs p30's 21.8 -- p50 is likely ~25-30%
FASTER IN BAND on compute-bound sizes (bandwidth still 28% worse:
DRAM octaves to be measured). The full engine port is justified on
in-band merit, not just range.

## 9. ENGINE v1 IMPLEMENTED + MEASURED (include/p50.h)

Full port of the p30 skeleton (1100 lines): Barrett52 butterflies,
radix-8 moths, in-register m <= 32 bottoms, iterative DFS, fused
traversal + side-channel spine, SCHOOLBOOK conv8 leaves, the
table-driven trunk I/O with the T = {88, 84, 80} ladder dispatch
(largest fitting trunk). v1 omits: input-stage level fusion, the
3*2^k ladder, squaring. GMP battery green (exhaustive small,
boundary totals, all-0xFF adversarial incl. 1M x 1M, random,
forced T = 80/84) + ASan.

TWO REAL BUGS, both bound violations the random kernel bench missed
and the GMP battery caught:
1. conv8 fb at lazy [0,4p): column sum 32p^2 = 2^104.94 OVERFLOWS
   the 104-bit accumulator domain (value-dependent). Fix: load fb as
   a vector, fold once to [0,2p) (16p^2 < 2^104), broadcast from
   register (permutexvar) -- op-count neutral.
2. conv8 REDC output reaches 16p^2/2^52 + p ~ 4.86p > 2^52 at
   adversarial magnitudes -- downstream madd52 inputs TRUNCATE. Fix:
   one fold before the store (range chain 4.86p -> 2.86p -> sh2 ->
   2p -> butterfly args < 4p < 2^52).
LESSON (now thrice-learned): random validation does not probe
accumulator bounds; only adversarial all-max inputs do. Write the
bound chain into comments and test it.

Sweep (p50_sweep_v1.csv, GMP-verified points) vs p30 stage 3:
    [2^11..2^16) limbs: p50 WINS 1.12-1.19x  (conv advantage, as
                        projected -- in-cache, compute-bound)
    [2^16,2^17):        ~parity (1.03)
    [2^17..2^21):       p50 loses 0.87-0.93x (28% spectra bandwidth
                        penalty + p50 v1 has no input fusion while
                        p30 has f3+NT and the 3*2^k octave)
    vs FLINT: 1.5-2.0x faster everywhere measured.
Crossover ~2^16.5 limbs. Natural combined dispatch: p50 in-cache,
p30 to its 2^21.2-limb cap, p50 beyond (its raison d'etre: the
2^40-limb band). p50 top-octave deficit is addressable: port the
fused input column pass (the largest p30 lever at DRAM sizes) and
revisit. The "25-30% faster in-band" projection was half right:
measured +12-19% in-cache; the transforms' extra level and conv at
~2x (not 3.7x) account for the difference.

## 10. Fused input: NEGATIVE result (measured, removed)

Port of p30's f3 (8-row column-fused input + truncated 3-level
register network + pre-depth traversal entry, NT-gated row-burst
stores) was implemented, battery+ASan green at default AND forced-low
gates, then A/B'd IN-PROCESS with a runtime switch (cross-process
sweeps carry up to 9% layout noise; the first sweep-vs-sweep diff was
misleadingly mild). Combos (mismatch probe confirmed fused input and
pre-walk are inseparable -- f3 bakes the first 3 NTT levels into the
rows, so only plain / fused / fused+NT are valid arms):

    n (limbs)   plain   fused   fused+NT
    524288      35.50   45.70   37.68
    655360      34.12   43.41   36.35
    917504      39.49   51.63   43.03
    1048576     37.97   48.86   41.03
    1310720     36.41   46.54   39.91
    1966080     42.85   52.32   44.26     (ns/limb, best-of)

Fused loses ~22-30%; NT stores claw most of that back (the row-burst
store traffic was the bulk), but fused+NT is STILL a strict 3-6% loss
at every size. Root cause (user diagnosis, confirmed by the numbers):
TRUNKS ARE WIDER THAN A LIMB (T = 80/84/88 > 64). In p30 a column row
decodes from limbs at aligned, periodic offsets -- the f3 decode is a
few cheap ops on top of loads the pass must do anyway. In p50 every
trunk straddles limb boundaries at arbitrary bit offsets, so each
8-row tile column pays the full funnel machinery (2 unaligned vector
loads + permutex2var windows per piece + shrdv + REDC fold) on
STRIDED windows, plus scalar edge groups per row. The decode is no
longer amortizable against the saved spectra pass: input is ~0.5-0.8
ns/limb of a ~35-43 ns/limb pipeline, so the theoretical upside was
~2-3% while the strided-decode overhead is ~2x the entire input cost.
p30's f3 physics DOES NOT TRANSFER once the digit radix exceeds the
limb width.

Code removed from p50.h (kept in git history / this doc). p50's DRAM
octaves stay 0.87-0.93x vs p30 -- acceptable: combined dispatch runs
p30 there anyway; p50's domain is in-cache (+12-19%) and beyond
p30's 2^21.2-limb cap. Remaining levers: squaring path, three-way
crossover dispatch.

## 11. Squaring paths (p50_sqr, p30_sqr)

One observation makes squaring nearly free to implement: in BOTH
engines the fused traversal uses its A argument as genuine a-spectrum
data ONLY at the leaf convolutions; everywhere else (the sb spine,
the ladder seam) A is side-buffer scratch frames at matching offsets.
And the leaf convs are alias-safe with fa == fb (all fb loads precede
the fa store; p50_conv8 and p30_conv16_kara2 both fold fb to their
required range internally). So squaring = the same traversal with an
`sq` flag that redirects leaf convs to the leaf's OWN just-computed
spectrum, plus ONE shared scratch buffer standing in for the A
frames (per-prime passes complete sequentially, so one buffer serves
all primes). p30's m = 2/4 register leaves store the forward result
once so kara2r can self-read; p50's small-m path is memory-based
already. What disappears per call: one input pass and ALL standalone
forward transforms (4-5 per multiply).

The batteries also established that the side channel never READS a
dead-tail position it did not write (sqr runs with an uninitialized
scratch buffer and passes both engines' full batteries, forced-gate
f3 and forced-everywhere ladder variants included, plus ASan).

Measured (balanced, solo, in-process):
    p50_sqr/p50_mul: 0.80 in-cache -> 0.68-0.73 at DRAM octaves
    p30_sqr/p30_mul: 0.85 in-cache -> 0.72-0.75 at DRAM octaves
(DRAM benefits more: 5 buffers instead of 8-10, less bandwidth.)
The sqr crossover between engines sits at the same TOTAL length as
mul (~2^18 limbs).

## 12. Combined dispatch (include/ntmul.h): fill-aware three-band

nt_mul/nt_sqr pick an engine per call (bench/nt_sweep_v2.csv, svg):
  - total <= 180224 limbs (11*2^14, p50's first in-band M-step):
    p50 -- compute-bound, transform occupancy barely matters;
  - total >= 2^20 limbs: p30 to its caps (bandwidth + f3 + ladder
    dominate even a fuller p50 transform), p50 beyond the caps;
  - BETWEEN: the engines' power-of-two transform steps interleave
    (p50 steps at 11*2^k totals at T = 88, p30 at 2^(k+4) totals),
    and the winner at EVERY swept point was simply the engine with
    the fuller truncated transform: compare tv/M and take the
    larger. A single-threshold rule leaves 9-14% on seam points;
    the fill rule's mean regret vs the per-point minimum is 1.0003
    (worst 3%, noise level). In the seam octaves nt_mul beats the
    better single engine's octave median by 2-6% because it rides
    each engine's full-transform windows.
  p50_MAX_LG raised 19 -> 22 (root tables now grow lazily with a
  prefix-preserving copy -- idle headroom costs nothing); p50 covers
  totals to 2^25 limbs by default and the design band beyond with a
  larger override.
Validation: tests/nt_test.c -- GMP differential at every dispatch
seam (band edges +-2, an in-band M-step, p30 CRT cap + 1, p30
transform cap exceeded with in-cap min operand, adversarial all-max
at the caps), 658 cases green.

Performance summary (nt_sweep_v2, ns/limb octave medians):
    n in [2^11,2^16): mul 17.2-20.5, sqr 13.9-16.1
    n in [2^16,2^19): mul 22.2-30.4, sqr 17.4-22.0
    n in [2^19,2^21): mul 31.9-33.6, sqr 23.4-25.1
