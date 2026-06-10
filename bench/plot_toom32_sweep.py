#!/usr/bin/env python3
"""Plot toom32_vs_dispatch_sweep CSV: speedup heatmap over (an, bn/an) plus
ratio-slice curves. Usage: plot_toom32_sweep.py [csv] [out_svg]"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

src = sys.argv[1] if len(sys.argv) > 1 else "toom32_vs_dispatch_sweep_60_400.csv"
out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".svg"

rows = []
with open(src) as f:
    for r in csv.DictReader(f):
        rows.append((int(r["an"]), int(r["bn"]), float(r["ratio"]),
                     float(r["dispatch_ns"]), float(r["toom32_ns"]), float(r["speedup"])))

an = [r[0] for r in rows]
ratio = [r[2] for r in rows]
spd = [r[5] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

lo, hi = min(spd), max(spd)
norm = TwoSlopeNorm(vmin=min(lo, 0.95), vcenter=1.0, vmax=max(hi, 1.05))
sc = ax1.scatter(an, ratio, c=spd, cmap="RdBu_r", norm=norm, s=26, marker="s")
fig.colorbar(sc, ax=ax1, label="speedup (dispatch / toom32)")
ax1.set_xlabel("an (limbs)")
ax1.set_ylabel("bn / an")
ax1.set_title("toom32 speedup vs dispatch")
ax1.axhline(2 / 3, color="gray", lw=0.6, ls="--")
ax1.text(an[-1], 2 / 3, " 2/3", va="center", fontsize=8, color="gray")

# ratio slices: nearest bn to target ratio per an
targets = [0.50, 0.58, 0.667, 0.75]
byan = {}
for r in rows:
    byan.setdefault(r[0], []).append(r)
for tgt in targets:
    xs, ys = [], []
    for a in sorted(byan):
        best = min(byan[a], key=lambda r: abs(r[2] - tgt))
        if abs(best[2] - tgt) > 0.03:
            continue
        xs.append(a)
        ys.append(best[5])
    ax2.plot(xs, ys, marker=".", ms=4, lw=1, label=f"bn/an ~ {tgt}")
ax2.axhline(1.0, color="black", lw=0.8)
ax2.set_xlabel("an (limbs)")
ax2.set_ylabel("speedup")
ax2.set_title("speedup by shape ratio")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(out)
print(f"wrote {out} ({len(rows)} points, speedup {lo:.3f}..{hi:.3f})")
