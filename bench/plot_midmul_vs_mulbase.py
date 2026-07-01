#!/usr/bin/env python3
"""Plot midmul (bn=2*an) vs mul_basecase (an=bn) at matched ~n^2 work.
Left:  ns / n_limb, both kernels overlaid.
Right: midmul/mul time ratio -- the price of the extra valignq (alignr64)
       ops midmul needs to build shifted b-vectors (1.0 == identical throughput).
Usage: plot_midmul_vs_mulbase.py [csv] [out_svg]"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "midmul_vs_mulbase_sweep.csv"
out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".svg"

n, mid_per, mul_per, ratio = [], [], [], []
with open(src) as f:
    for r in csv.DictReader(f):
        x = int(r["n"]); m = float(r["midmul_ns"]); s = float(r["mul_ns"])
        n.append(x); mid_per.append(m / x); mul_per.append(s / x); ratio.append(m / s)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.plot(n, mid_per, lw=1, color="C0", label="midmul_basecase_simple  (bn=2·an)")
ax1.plot(n, mul_per, lw=1, color="C2", label="mul_u52_basecase  (an=bn)")
ax1.set_xlabel("n limbs   (midmul: an=n, bn=2n  |  mul: an=bn=n)")
ax1.set_ylabel("ns / n_limb")
ax1.set_title("matched ~n² work: ns per n-limb")
ax1.grid(alpha=0.3)
ax1.legend(fontsize=9)
ax1.margins(x=0.02)

ax2.axhline(1.0, color="gray", lw=0.8, ls="--")
ax2.plot(n, ratio, lw=1, color="C3")
ax2.set_xlabel("n limbs")
ax2.set_ylabel("midmul / mul_basecase  (time ratio)")
ax2.set_title("cost of midmul's extra align ops")
ax2.grid(alpha=0.3)
ax2.margins(x=0.02)
# steady-state mean ratio over n>=32
ss = [r for x, r in zip(n, ratio) if x >= 32]
if ss:
    mean_ss = sum(ss) / len(ss)
    ax2.axhline(mean_ss, color="C3", lw=0.7, ls=":")
    ax2.text(n[-1], mean_ss, f" mean(n≥32)={mean_ss:.3f}", fontsize=8,
             color="C3", va="bottom", ha="right")
    ax2.set_ylim(0.9, max(1.3, min(1.6, max(ss) * 1.05)))

fig.tight_layout()
fig.savefig(out)
png = out.rsplit(".", 1)[0] + ".png"
fig.savefig(png, dpi=110)
print(f"wrote {out} and {png} ({len(n)} points); "
      f"steady-state (n≥32) mean midmul/mul = {mean_ss:.3f}")
