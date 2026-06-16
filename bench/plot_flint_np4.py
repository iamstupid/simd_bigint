#!/usr/bin/env python3
"""Overlay: FLINT fft_small default profile policy vs forced 4-prime NTTs.
Usage: plot_flint_np4.py [default_csv] [np4_csv] [out]"""
import csv
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dft_csv = sys.argv[1] if len(sys.argv) > 1 else "flint_ntt_sweep.csv"
np4_csv = sys.argv[2] if len(sys.argv) > 2 else "flint_ntt_np4.csv"
out = sys.argv[3] if len(sys.argv) > 3 else "flint_ntt_np4.svg"

def load(p):
    with open(p) as f:
        return [(int(r["n"]), float(r["ns_limb"])) for r in csv.DictReader(f)]

dft = load(dft_csv)
np4 = load(np4_csv)

fig, ax = plt.subplots(figsize=(13, 6.5))
ax.plot([n for n, _ in dft], [t for _, t in dft], lw=1.0, marker=".", ms=3,
        color="tab:purple", alpha=0.8,
        label="default policy (np = 4..8, profile chooser)")
ax.plot([n for n, _ in np4], [t for _, t in np4], lw=1.2, marker=".", ms=3,
        color="tab:cyan",
        label="forced np = 4 (bits = 88 everywhere)")
ax.set_xscale("log", base=2)
ax.set_xlabel("n (u64 limbs)")
ax.set_ylabel("ns / input limb")
ax.set_title("FLINT fft_small, balanced mul, 1 thread, 1/16-octave grid "
             "(solo process, median of 7)")
ax.grid(alpha=0.3, which="both")
for p in range(16, 23):
    ax.axvline(1 << p, color="gray", lw=0.5, ls=":")
ax.set_ylim(bottom=0)
ax.legend()
fig.tight_layout()
fig.savefig(out)
print(f"wrote {out}")

d4 = dict(np4)
dd = dict(dft)
common = sorted(set(d4) & set(dd))
print(f"{'band':>14} | {'default':>8} | {'np4':>7} | {'np4/dft':>7} | np4 max/min")
for lo, hi in ((1 << 16, 1 << 17), (1 << 17, 1 << 18), (1 << 18, 1 << 19),
               (1 << 19, 1 << 20), (1 << 20, 1 << 21), (1 << 21, (1 << 22) + 1)):
    cs = [n for n in common if lo <= n < hi]
    md = statistics.median(dd[n] for n in cs)
    m4 = statistics.median(d4[n] for n in cs)
    band4 = [d4[n] for n in cs]
    print(f"[2^{lo.bit_length()-1}, 2^{hi.bit_length()-1}) {'':>2} | {md:8.2f} |"
          f" {m4:7.2f} | {m4/md:7.3f} | {max(band4)/min(band4):.3f}")
rat = [d4[n] / dd[n] for n in common]
print(f"overall median np4/default : {statistics.median(rat):.3f}")
print(f"worst-case spike removed   : default max {max(t for _, t in dft):.1f}"
      f" -> np4 max {max(t for _, t in np4):.1f} ns/limb")
