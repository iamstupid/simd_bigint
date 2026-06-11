#!/usr/bin/env python3
import csv, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p):
    with open(p) as f:
        return {int(r["n"]): float(r["ns_limb"]) for r in csv.DictReader(f)}

f16 = load("fvd_f16.csv")
u52a = load("fvd_u52.csv")      # T22 = 128
u52b = load("fvd_u52_t90.csv")  # T22 = 90
ns = sorted(set(f16) & set(u52a) & set(u52b))

fig, ax = plt.subplots(figsize=(13, 7))
ax.plot(ns, [u52a[n] for n in ns], lw=1.0, color="tab:red",   label="u52 dispatch, T22=128")
ax.plot(ns, [u52b[n] for n in ns], lw=1.0, color="tab:purple", label="u52 dispatch, T22=90")
ax.plot(ns, [f16[n] for n in ns],  lw=1.0, color="tab:blue",  label="fft16")
ax.set_xlabel("n (u64 limbs)"); ax.set_ylabel("ns / input limb")
ax.set_title("crossover sweep with T22 variants (step 2, solo-per-process)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout(); fig.savefig(sys.argv[1] if len(sys.argv) > 1 else "fft16_vs_dispatch_t90.svg")

for tag, u in (("T22=128", u52a), ("T22=90 ", u52b)):
    wins = [1 if f16[n] < u[n] else 0 for n in ns]
    cross = None
    for i in range(len(ns) - 8):
        if sum(wins[i:i+8]) >= 7: cross = ns[i]; break
    print(f"{tag}: crossover n = {cross}")
for lo, hi in ((128,192),(192,256),(256,320),(320,384),(384,448),(448,513)):
    a = statistics.median([u52a[n] for n in ns if lo <= n < hi])
    b = statistics.median([u52b[n] for n in ns if lo <= n < hi])
    f = statistics.median([f16[n] for n in ns if lo <= n < hi])
    print(f"[{lo:>3},{hi-1:>3}] T128 {a:6.2f}  T90 {b:6.2f}  fft16 {f:6.2f}")
