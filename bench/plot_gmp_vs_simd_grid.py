#!/usr/bin/env python3
"""3x3 grid: per ratio class, ns per an-limb for GMP vs simd_mpn_mul.
Usage: plot_gmp_vs_simd_grid.py [csv] [out_svg]"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "gmp_vs_simd_sweep.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "gmp_vs_simd_grid_3x3.svg"

series = {}
with open(src) as f:
    for r in csv.DictReader(f):
        an = int(r["an"])
        series.setdefault(float(r["ratio"]), []).append(
            (an, float(r["gmp_ns"]) / an, float(r["simd_ns"]) / an))

labels = {0.125: "bn = an/8", 0.2: "bn = an/5", 0.25: "bn = an/4",
          0.3333: "bn = an/3", 0.4: "bn = 0.4 an", 0.5: "bn = 0.5 an",
          0.65: "bn = 0.65 an", 0.8: "bn = 0.8 an", 1.0: "bn = an"}

fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True)
for i, k in enumerate(sorted(series)):
    ax = axes[i // 3][i % 3]
    pts = sorted(series[k])
    xs = [p[0] for p in pts]
    ax.plot(xs, [p[1] for p in pts], lw=0.9, color="tab:red", label="gmp mpn_mul")
    ax.plot(xs, [p[2] for p in pts], lw=0.9, color="tab:blue", label="simd_mpn_mul")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_title(labels.get(round(k, 4), f"bn={k}an"), fontsize=11)
    ax.grid(alpha=0.3, which="both")
    if i == 0:
        ax.legend(fontsize=9)
    if i // 3 == 2:
        ax.set_xlabel("an (u64 limbs)")
    if i % 3 == 0:
        ax.set_ylabel("ns / an-limb")
fig.suptitle("GMP (zen3, clang) vs simd_mpn_mul -- ns per an-limb", fontsize=13)
fig.tight_layout()
fig.savefig(out)
print(f"wrote {out}")
