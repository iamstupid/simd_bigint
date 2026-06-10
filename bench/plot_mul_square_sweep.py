#!/usr/bin/env python3
"""Plot mul_square_sweep CSV: ns/limb for basecase / toom-22 / toom-33 on
square shapes. Usage: plot_mul_square_sweep.py [csv] [out_svg]"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "mul_square_sweep_8_500.csv"
out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".svg"

n, base, t22, t33 = [], [], [], []
with open(src) as f:
    for r in csv.DictReader(f):
        n.append(int(r["n"]))
        base.append(float(r["basecase_ns"]) / int(r["n"]))
        t22.append(float(r["toom22_ns"]) / int(r["n"]))
        t33.append(float(r["toom33_ns"]) / int(r["n"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for ax in (ax1, ax2):
    ax.plot(n, base, lw=1, label="basecase (IFMA)")
    ax.plot(n, t22, lw=1, label="toom-22 (karatsuba)")
    ax.plot(n, t33, lw=1, label="toom-33")
    ax.set_xlabel("n = an = bn (limbs)")
    ax.set_ylabel("ns / limb")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

ax1.set_title("square mul, full range")

# zoom: cap y to make the crossovers readable
cap = 2.5 * min(min(base), min(t22), min(t33)) + max(t22[len(t22)//2], t33[len(t33)//2])
ax2.set_ylim(0, cap)
ax2.set_title("zoom (crossover region)")

# annotate crossovers
def cross(a, b):
    for i in range(len(n) - 8):
        if a[i] > b[i] and all(a[j] >= b[j] for j in range(i, min(i + 8, len(n)))):
            return n[i]
    return None
c22 = cross(base, t22)
c33 = cross(t22, t33)
for c, lbl in ((c22, "base/t22"), (c33, "t22/t33")):
    if c:
        ax2.axvline(c, color="gray", lw=0.7, ls="--")
        ax2.text(c, ax2.get_ylim()[1] * 0.95, f" {lbl}@{c}", fontsize=8, color="gray", va="top")

fig.tight_layout()
fig.savefig(out)
print(f"wrote {out} ({len(n)} points); crossovers: base->t22 @ {c22}, t22->t33 @ {c33}")
