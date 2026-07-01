#!/usr/bin/env python3
"""Plot midmul simple (masked horizontal tail) vs vtail (vertical dot-product
tail for a small rn%8 tail) vs mul_basecase, shape bn=2*an.  ns/n_limb.
Usage: plot_midmul_tail_compare.py [csv] [out_svg]"""
import csv, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "midmul_tail_compare.csv"
out = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".svg"

n, s, v, m = [], [], [], []
with open(src) as f:
    for r in csv.DictReader(f):
        x = int(r["n"]); n.append(x)
        s.append(float(r["simple_ns"]) / x)
        v.append(float(r["vtail_ns"]) / x)
        m.append(float(r["mul_ns"]) / x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
for ax in (ax1, ax2):
    ax.plot(n, s, lw=1.1, color="C0", label="midmul simple (horizontal masked tail)")
    ax.plot(n, v, lw=1.1, color="C3", label="midmul vtail (vertical tail, t≤3)")
    ax.plot(n, m, lw=0.9, color="C2", alpha=0.7, label="mul_u52_basecase (an=bn)")
    ax.set_xlabel("n  (an=n, bn=2n)"); ax.set_ylabel("ns / n_limb")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax1.set_title("full range 8..%d" % n[-1])
# zoom on the staircase region where the steps are clearest
lo, hi = 40, min(96, n[-1])
ax2.set_xlim(lo, hi)
sel = [y for x, y in zip(n, s) if lo <= x <= hi]
ax2.set_ylim(min(sel) * 0.96, max(sel) * 1.04)
ax2.set_title("zoom %d..%d: vtail shaves each step's leading edge (t=1)" % (lo, hi))
# mark the step boundaries (an ≡ 0 mod 8 => rn ≡ 1 mod 8 => t=1 peak of simple)
for x in n:
    if lo <= x <= hi and x % 8 == 0:
        ax2.axvline(x, color="gray", lw=0.5, ls=":", alpha=0.6)

fig.tight_layout(); fig.savefig(out)
png = out.rsplit(".", 1)[0] + ".png"; fig.savefig(png, dpi=115)
gain = (sum(s) - sum(v)) / sum(s) * 100
print(f"wrote {out} and {png} ({len(n)} pts); mean simple->vtail = {gain:+.2f}%")
