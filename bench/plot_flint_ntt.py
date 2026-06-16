#!/usr/bin/env python3
"""Plot the 1/16-octave FLINT fft_small (small-primes truncated NTT) sweep.
Usage: plot_flint_ntt.py [csv] [out]"""
import csv
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "flint_ntt_sweep.csv"
out = sys.argv[2] if len(sys.argv) > 2 else "flint_ntt_sweep.svg"

with open(src) as f:
    rows = [(int(r["n"]), float(r["ns_limb"])) for r in csv.DictReader(f)]
ns = [n for n, _ in rows]
ts = [t for _, t in rows]

try:  # np,bits profile choice per point (from the best_profile replica probe)
    with open("flint_ntt_profiles.csv") as f:
        prof = {int(l.split(",")[0]): int(l.split(",")[1]) for l in f}
except OSError:
    prof = {}

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.plot(ns, ts, lw=1.1, color="tab:purple",
        label="FLINT fft_small (mpn_mul_default_mpn_ctx, 1 thread)")
if prof:
    npcol = {4: "tab:green", 5: "tab:blue", 6: "tab:olive",
             7: "tab:orange", 8: "tab:red"}
    for npv, c in npcol.items():
        sel = [(n, t) for n, t in rows if prof.get(n) == npv]
        if sel:
            ax.scatter([n for n, _ in sel], [t for _, t in sel], s=14, color=c,
                       zorder=3, label=f"np = {npv} primes")
ax.set_xscale("log", base=2)
ax.set_xlabel("n (u64 limbs)")
ax.set_ylabel("ns / input limb")
ax.set_title("balanced mul, an = bn = n, 1/16-octave grid, n in [2^16, 2^22] "
             "(solo process, median of 7)")
ax.grid(alpha=0.3, which="both")
for p in range(16, 23):
    ax.axvline(1 << p, color="gray", lw=0.5, ls=":")
ax.set_ylim(bottom=0)
ax.legend()
fig.tight_layout()
fig.savefig(out)
print(f"wrote {out}")

print(f"{'band':>20} | {'med':>6} | {'min':>6} | {'max':>6}")
for lo, hi in ((1 << 16, 1 << 17), (1 << 17, 1 << 18), (1 << 18, 1 << 19),
               (1 << 19, 1 << 20), (1 << 20, 1 << 21), (1 << 21, (1 << 22) + 1)):
    band = [t for n, t in rows if lo <= n < hi]
    print(f"[2^{lo.bit_length()-1:>2}, 2^{hi.bit_length()-1:>2}) {'':>6} |"
          f" {statistics.median(band):6.2f} | {min(band):6.2f} | {max(band):6.2f}")
print(f"{'overall median':>20} : {statistics.median(ts):.2f} ns/limb")
print(f"{'smoothness (max/min within octave)':>20}")
for p in range(16, 22):
    band = [t for n, t in rows if (1 << p) <= n < (1 << (p + 1))]
    print(f"    2^{p}..2^{p+1}: {max(band)/min(band):.3f}")
