#!/usr/bin/env python3
"""3x3 grid: fft16 vs u52 dispatch per relative size class, with crossover.
Usage: plot_fvdr.py [out]"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p):
    d = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            d.setdefault(float(r["ratio"]), {})[int(r["an"])] = (
                int(r["bn"]), float(r["ns_call"]))
    return d

f16 = load("fvdr_f16.csv")
u52 = load("fvdr_u52.csv")
out = sys.argv[1] if len(sys.argv) > 1 else "fft16_vs_dispatch_ratios.svg"
labels = {1.0: "bn = an", 0.8: "bn = 0.8 an", 0.65: "bn = 0.65 an",
          0.5: "bn = 0.5 an", 0.4: "bn = 0.4 an", 0.3333: "bn = an/3",
          0.25: "bn = an/4", 0.2: "bn = an/5", 0.125: "bn = an/8"}

fig, axes = plt.subplots(3, 3, figsize=(16, 11), sharex=True)
ratios = sorted(f16, reverse=True)
crossings = []
for i, r in enumerate(ratios):
    ax = axes[i // 3][i % 3]
    ans = sorted(set(f16[r]) & set(u52[r]))
    # ns per total operand limb
    fs = [f16[r][a][1] / (a + f16[r][a][0]) for a in ans]
    us = [u52[r][a][1] / (a + u52[r][a][0]) for a in ans]
    ax.plot(ans, us, lw=0.9, color="tab:red", label="u52 dispatch")
    ax.plot(ans, fs, lw=0.9, color="tab:blue", label="fft16")
    # sustained crossover: fft16 wins >= 7 of 8 consecutive
    wins = [1 if fs[k] < us[k] else 0 for k in range(len(ans))]
    cross = next((ans[k] for k in range(len(ans) - 8)
                  if sum(wins[k:k + 8]) >= 7), None)
    if cross is not None:
        ax.axvline(cross, color="black", lw=0.8, ls="--")
        bn = f16[r][cross][0]
        ax.set_title(f"{labels.get(round(r, 4), r)}   cross an={cross} (s={cross + bn})")
        crossings.append((r, cross, bn, cross + bn))
    else:
        ax.set_title(f"{labels.get(round(r, 4), r)}   no crossover <= an=4096")
        crossings.append((r, None, None, None))
    ax.grid(alpha=0.3)
    if i == 0:
        ax.legend(fontsize=8)
    if i // 3 == 2:
        ax.set_xlabel("an (u64 limbs)")
    if i % 3 == 0:
        ax.set_ylabel("ns / (an+bn)")
fig.tight_layout()
fig.savefig(out)
print(f"wrote {out}")
print(f"{'ratio':>8} | {'an*':>6} | {'bn*':>6} | {'s*':>6}")
for r, a, b, s in crossings:
    print(f"{r:8.4f} | " + (f"{a:>6} | {b:>6} | {s:>6}" if a else "  none |   none |   none"))
