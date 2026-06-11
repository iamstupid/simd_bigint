#!/usr/bin/env python3
import csv, statistics, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load(p):
    with open(p) as f:
        return {int(r["n"]): float(r["ns_limb"]) for r in csv.DictReader(f)}

f16 = load("fvd_f16.csv")
u128 = load("fvd_u52.csv")
u90 = load("fvd_u52_t90.csv")
u84 = load("fvd_u52_t84.csv")
ns = sorted(set(f16) & set(u128) & set(u90) & set(u84))

fig, ax = plt.subplots(figsize=(13, 7))
ax.plot(ns, [u128[n] for n in ns], lw=0.9, color="tab:red",    label="u52, T22=128")
ax.plot(ns, [u90[n] for n in ns],  lw=0.9, color="tab:purple", label="u52, T22=90")
ax.plot(ns, [u84[n] for n in ns],  lw=1.1, color="tab:green",  label="u52, T22=84")
ax.plot(ns, [f16[n] for n in ns],  lw=1.0, color="tab:blue",   label="fft16")
ax.set_xlabel("n (u64 limbs)"); ax.set_ylabel("ns / input limb")
ax.set_title("crossover sweep, T22 in {128, 90, 84} (step 2, solo-per-process)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig.savefig("fft16_vs_dispatch_t84.svg")
if len(sys.argv) > 1: fig.savefig(sys.argv[1])
print("wrote fft16_vs_dispatch_t84.svg")

# smoothness: median |second difference| (lower = smoother), plus medians
def rough(u): 
    v = [u[n] for n in ns]
    return statistics.median(abs(v[i+1] - 2*v[i] + v[i-1]) for i in range(1, len(v)-1))
for tag, u in (("T128", u128), ("T90 ", u90), ("T84 ", u84)):
    wins = [1 if f16[n] < u[n] else 0 for n in ns]
    cross = next((ns[i] for i in range(len(ns)-8) if sum(wins[i:i+8]) >= 7), None)
    print(f"{tag}: roughness {rough(u):.4f}  crossover {cross}")
for lo, hi in ((128,192),(192,256),(256,320),(320,384),(384,448),(448,513)):
    a = statistics.median([u128[n] for n in ns if lo <= n < hi])
    b = statistics.median([u90[n] for n in ns if lo <= n < hi])
    c = statistics.median([u84[n] for n in ns if lo <= n < hi])
    f = statistics.median([f16[n] for n in ns if lo <= n < hi])
    print(f"[{lo:>3},{hi-1:>3}] T128 {a:6.2f}  T90 {b:6.2f}  T84 {c:6.2f}  fft16 {f:6.2f}")
