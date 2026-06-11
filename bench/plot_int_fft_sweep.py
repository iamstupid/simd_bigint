#!/usr/bin/env python3
"""Plot int_fft codec sweep: ns per input limb vs n (balanced mul, an=bn=n).
Usage: plot_int_fft_sweep.py [csv] [out_svg]"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "int_fft_sweep.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "int_fft_sweep.svg"

cols = {"u16": [], "wide": [], "centered": []}
with open(src) as f:
    for r in csv.DictReader(f):
        n = int(r["n"])
        for k in cols:
            if r[k]:
                cols[k].append((n, float(r[k])))

fig, ax = plt.subplots(figsize=(12, 7))
style = {"u16": ("tab:blue", "U16 (N <= 2^16)"),
         "wide": ("tab:orange", "Wide U16 (N <= 2^17)"),
         "centered": ("tab:green", "Centered U16 (N <= 2^19)")}
for k, pts in cols.items():
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs, ys, lw=1.1, color=style[k][0], label=style[k][1])

ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_xlabel("n (u64 limbs, an = bn = n)")
ax.set_ylabel("ns / input limb")
ax.set_title("int_fft reference: balanced mul, per-codec cost (1/16-octave grid)")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout()
fig.savefig(out)
print(f"wrote {out}")

# band summary
import statistics
print(f"{'band':>16} | {'u16':>7} | {'wide':>7} | {'centered':>8}")
for lo, hi in ((8, 64), (64, 512), (512, 4096), (4096, 16385), (16385, 32769), (32769, 131073)):
    row = []
    for k in ("u16", "wide", "centered"):
        v = [y for x, y in cols[k] if lo <= x < hi]
        row.append(f"{statistics.median(v):7.2f}" if v else "      -")
    print(f"[{lo:>6},{hi - 1:>6}] | {row[0]} | {row[1]} | {row[2]:>8}")
