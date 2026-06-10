#!/usr/bin/env python3
"""Plot gmp_vs_simd_sweep CSV: speedup (gmp/simd) vs an (u64 limbs), one curve
per relative size class. Usage: plot_gmp_vs_simd.py [csv] [out_svg]"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "gmp_vs_simd_sweep.csv"
out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".svg"

series = {}
with open(src) as f:
    for r in csv.DictReader(f):
        series.setdefault(float(r["ratio"]), []).append((int(r["an"]), float(r["speedup"])))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
labels = {0.125: "bn = an/8", 0.2: "bn = an/5", 0.25: "bn = an/4",
          0.3333: "bn = an/3", 0.4: "bn = 0.4 an", 0.5: "bn = 0.5 an",
          0.65: "bn = 0.65 an", 0.8: "bn = 0.8 an", 1.0: "bn = an"}
cmap = plt.get_cmap("viridis")
keys = sorted(series)
for i, k in enumerate(keys):
    pts = sorted(series[k])
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    lbl = labels.get(round(k, 4), f"bn={k:.3f}an")
    ax1.plot(xs, ys, lw=0.9, color=cmap(i / 8), label=lbl)
    # smoothed for the right panel (rolling median over 9 points)
    sm = [sorted(ys[max(0, j-4):j+5])[len(ys[max(0, j-4):j+5])//2] for j in range(len(ys))]
    ax2.plot(xs, sm, lw=1.4, color=cmap(i / 8), label=lbl)

for ax in (ax1, ax2):
    ax.axhline(1.0, color="black", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("an (u64 limbs)")
    ax.set_ylabel("speedup (gmp_ns / simd_ns)")
    ax.grid(alpha=0.3)
ax1.set_title("raw")
ax2.set_title("rolling median (9)")
ax2.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig(out)

# console summary: per-class medians by size band
import statistics
print(f"wrote {out}")
print(f"{'class':>12} | " + " | ".join(f"{b:>9}" for b in ["8-64", "64-256", "256-1024", "1024-2048"]))
for k in keys:
    pts = sorted(series[k])
    row = []
    for lo, hi in ((8, 64), (64, 256), (256, 1024), (1024, 2049)):
        v = [s for a, s in pts if lo <= a < hi]
        row.append(f"{statistics.median(v):9.2f}" if v else "        -")
    print(f"{labels.get(round(k,4), k):>12} | " + " | ".join(row))
