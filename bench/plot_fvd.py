#!/usr/bin/env python3
"""Crossover plot: fft16 vs u52 toom dispatch, n in [128, 512] step 2."""
import csv
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p):
    with open(p) as f:
        return {int(r["n"]): float(r["ns_limb"]) for r in csv.DictReader(f)}

f16 = load("fvd_f16.csv")
u52 = load("fvd_u52.csv")
ns = sorted(set(f16) & set(u52))
out = sys.argv[1] if len(sys.argv) > 1 else "fft16_vs_dispatch.svg"

fig, ax = plt.subplots(figsize=(13, 7))
ax.plot(ns, [u52[n] for n in ns], lw=1.0, color="tab:red",
        label="u52 toom dispatch (simd_mpn_mul)")
ax.plot(ns, [f16[n] for n in ns], lw=1.0, color="tab:blue",
        label="fft16 (AVX-512 PQ FFT)")
ax.set_xlabel("n (u64 limbs, an = bn = n)")
ax.set_ylabel("ns / input limb")
ax.set_title("crossover sweep, balanced mul, step 2 (solo-per-process)")
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(out)
print(f"wrote {out}")

# crossover: first n where fft16 wins sustainedly (>= 7 of 8 consecutive)
wins = [1 if f16[n] < u52[n] else 0 for n in ns]
cross = None
for i in range(len(ns) - 8):
    if sum(wins[i:i + 8]) >= 7:
        cross = ns[i]
        break
print(f"sustained crossover (fft16 wins >=7/8): n = {cross}")
for lo, hi in ((128, 192), (192, 256), (256, 320), (320, 384), (384, 448), (448, 513)):
    fs = [f16[n] for n in ns if lo <= n < hi]
    us = [u52[n] for n in ns if lo <= n < hi]
    w = sum(1 for n in ns if lo <= n < hi and f16[n] < u52[n])
    t = sum(1 for n in ns if lo <= n < hi)
    print(f"[{lo:>3},{hi-1:>3}] u52 {statistics.median(us):6.2f}  "
          f"fft16 {statistics.median(fs):6.2f}  fft16 wins {w}/{t}")
