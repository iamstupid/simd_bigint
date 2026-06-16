#!/usr/bin/env python3
"""Plot the 1/16-octave fft16 (AVX-512) vs reference int_fft (AVX2) sweep.
Usage: plot_fft16_sweep.py [f16_csv] [ref_csv] [out]"""
import csv
import statistics
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

f16_csv = sys.argv[1] if len(sys.argv) > 1 else "fft16_sweep_f16.csv"
ref_csv = sys.argv[2] if len(sys.argv) > 2 else "fft16_sweep_ref.csv"
out = sys.argv[3] if len(sys.argv) > 3 else "fft16_vs_intfft_sweep.svg"

def load(p):
    with open(p) as f:
        return {int(r["n"]): float(r["ns_limb"]) for r in csv.DictReader(f)}

f16 = load(f16_csv)
ref = load(ref_csv)
ns = sorted(set(f16) & set(ref))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9), sharex=True,
                               height_ratios=[2.2, 1])
ax1.plot(ns, [ref[n] for n in ns], lw=1.1, color="tab:orange",
         label="int_fft reference (AVX2, mul_auto bands)")
ax1.plot(ns, [f16[n] for n in ns], lw=1.1, color="tab:blue",
         label="fft16 (AVX-512, U16/centered bands)")
ax1.set_xscale("log", base=2)
ax1.set_ylabel("ns / input limb")
ax1.set_title("balanced mul, an = bn = n, 1/16-octave grid (solo-per-process)")
ax1.grid(alpha=0.3, which="both")
ax1.legend()

ratio = [ref[n] / f16[n] for n in ns]
ax2.plot(ns, ratio, lw=1.1, color="tab:green")
ax2.axhline(1.0, color="black", lw=0.8)
ax2.set_xscale("log", base=2)
ax2.set_xlabel("n (u64 limbs)")
ax2.set_ylabel("speedup (ref / fft16)")
ax2.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(out)
print(f"wrote {out}")

print(f"{'band':>16} | {'ref':>6} | {'fft16':>6} | {'ratio':>6}")
for lo, hi in ((128, 512), (512, 2048), (2048, 8192), (8192, 16385),
               (16385, 32769), (32769, 65537), (65537, 131073)):
    rs = [ref[n] for n in ns if lo <= n < hi]
    fs = [f16[n] for n in ns if lo <= n < hi]
    qs = [ref[n] / f16[n] for n in ns if lo <= n < hi]
    print(f"[{lo:>6},{hi - 1:>6}] | {statistics.median(rs):6.2f} |"
          f" {statistics.median(fs):6.2f} | {statistics.median(qs):6.3f}")
print(f"{'overall median ratio':>33} : {statistics.median(ratio):.3f}")
print(f"{'min / max ratio':>33} : {min(ratio):.3f} / {max(ratio):.3f}")
