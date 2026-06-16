#!/usr/bin/env python3
"""Plot the 1/16-octave p30 truncated-NTT sweep with yardstick overlays.
Usage: plot_p30_sweep.py [p30_csv] [out]"""
import csv
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "p30_sweep.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "p30_sweep.svg"

def load(p):
    try:
        with open(p) as f:
            return [(int(r["n"]), float(r["ns_limb"])) for r in csv.DictReader(f)]
    except OSError:
        return []

p30 = load(src)
f16 = load("fft16_sweep_f16.csv")          # pq16/fft16 (caps at n = 131072)
fnt = load("flint_ntt_sweep.csv")          # FLINT fft_small (n >= 65536)
ref = load("fft16_sweep_ref.csv")          # int_fft reference (AVX2)

fig, ax = plt.subplots(figsize=(13, 6.5))
if ref:
    ax.plot(*zip(*ref), lw=0.9, color="tab:gray", alpha=0.7,
            label="int_fft reference (AVX2)")
if fnt:
    ax.plot(*zip(*fnt), lw=0.9, color="tab:purple", alpha=0.8,
            label="FLINT fft_small (1 thread)")
if f16:
    ax.plot(*zip(*f16), lw=1.0, color="tab:blue",
            label="fft16/pq16 (caps at n = 2^17)")
ax.plot(*zip(*p30), lw=1.4, color="tab:red", marker=".", ms=3,
        label="p30 truncated NTT (stage 1)")
ax.set_xscale("log", base=2)
ax.set_xlabel("n (u64 limbs, balanced an = bn = n)")
ax.set_ylabel("ns / input limb")
ax.set_title("p30 5-prime truncated NTT, stage 1 — 1/16-octave grid "
             "(solo process, median of 7, all points GMP-verified)")
ax.grid(alpha=0.3, which="both")
for p in range(11, 22):
    ax.axvline(1 << p, color="gray", lw=0.5, ls=":")
ax.set_ylim(bottom=0)
ax.legend()
fig.tight_layout()
fig.savefig(out)
print(f"wrote {out}")

d30 = dict(p30)
print(f"{'octave':>16} | {'med':>6} | {'min':>6} | {'max':>6} | max/min")
lo = min(n for n, _ in p30)
import math
p0 = int(math.log2(lo))
for p in range(p0, 21):
    band = [t for n, t in p30 if (1 << p) <= n < (1 << (p + 1))]
    if not band:
        continue
    print(f"[2^{p:>2}, 2^{p+1:>2}) {'':>3} | {statistics.median(band):6.2f} |"
          f" {min(band):6.2f} | {max(band):6.2f} | {max(band)/min(band):.3f}")
for name, data in (("fft16", f16), ("FLINT", fnt)):
    common = sorted(set(d30) & {n for n, _ in data})
    if common:
        dd = dict(data)
        rat = [dd[n] / d30[n] for n in common]
        print(f"{name}/p30 over {len(common)} shared points: "
              f"median {statistics.median(rat):.3f}, "
              f"min {min(rat):.3f}, max {max(rat):.3f}")
