#!/usr/bin/env python3
"""Plot midmul_simple_sweep CSV (shape bn = 2*an).
Left:  ns / an_limb  (the requested metric).
Right: ns / (an*bn) i.e. ns per scalar 52x52 product -- reveals steady-state
       kernel throughput, ~constant once vector/overhead amortizes.
Usage: plot_midmul_simple_sweep.py [csv] [out_svg]"""
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "midmul_simple_sweep.csv"
out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".svg"

an, ns_per_an, ns_per_prod = [], [], []
with open(src) as f:
    for r in csv.DictReader(f):
        a = int(r["an"]); b = int(r["bn"]); t = float(r["ns"])
        an.append(a)
        ns_per_an.append(t / a)
        ns_per_prod.append(t / (a * b))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

ax1.plot(an, ns_per_an, lw=1, color="C0")
ax1.set_xlabel("an (limbs)   [bn = 2*an]")
ax1.set_ylabel("ns / an_limb")
ax1.set_title("midmul_basecase_simple: ns per an-limb")
ax1.grid(alpha=0.3)
# O(an^2) work / an => linear; show the trend
ax1.margins(x=0.02)

ax2.plot(an, [p * 1000 for p in ns_per_prod], lw=1, color="C1")
ax2.set_xlabel("an (limbs)   [bn = 2*an]")
ax2.set_ylabel("ps / (52x52 product)   [ns/(an*bn) * 1000]")
ax2.set_title("steady-state kernel throughput")
ax2.grid(alpha=0.3)
ax2.set_ylim(bottom=0)
ax2.margins(x=0.02)

fig.tight_layout()
fig.savefig(out)

# also drop a png for quick viewing
png = out.rsplit(".", 1)[0] + ".png"
fig.savefig(png, dpi=110)
print(f"wrote {out} and {png} ({len(an)} points); "
      f"ns/an_limb: {ns_per_an[0]:.3f} @an={an[0]} -> {ns_per_an[-1]:.3f} @an={an[-1]}")
