#!/usr/bin/env python3
# Plot DC block-division speedup over GMP (and over the u52 schoolbook wrapper)
# for an = k*bn, bn = divisor vecs (zmm blocks, 416 bits each).
import csv, sys
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "dcsweep.csv"
rows = list(csv.DictReader(open(path)))
ks = sorted({float(r["k"]) for r in rows})

# group by k: bn -> ratios
def series(metric):
    d = defaultdict(lambda: ([], []))
    for r in rows:
        k = float(r["k"]); bn = int(r["bn"])
        gmp = float(r["gmp_ns"]); sb = float(r["sb_ns"]); dc = float(r["dc_ns"])
        val = {"gmp": gmp/dc, "sb": sb/dc}[metric]
        d[k][0].append(bn); d[k][1].append(val)
    return d

# empirical GMP dc->mu knee: smallest bn past the floor where DC/GMP rises >=15%
def gmp_mu_knee(k):
    pts = sorted(zip(*[series("gmp")[k][j] for j in (0,1)]))  # (bn, gmp/dc)
    inv = [(bn, 1.0/r) for bn, r in pts]                       # dc/gmp (lower=better)
    fb, fv = min(inv[5:] if len(inv) > 6 else inv, key=lambda t: t[1])
    for bn, v in inv:
        if bn > fb and v >= fv*1.15:
            # return (bn, speedup gmp/dc at that bn)
            return bn, next(r for b, r in pts if b == bn)
    return None

fig, ax = plt.subplots(1, 2, figsize=(15, 6), sharex=True)
cmap = plt.cm.viridis
for i, k in enumerate(ks):
    c = cmap(i/max(1,len(ks)-1))
    g = series("gmp"); s = series("sb")
    ax[0].plot(g[k][0], g[k][1], "-o", ms=3, color=c, label=f"k={k:g}")
    ax[1].plot(s[k][0], s[k][1], "-o", ms=3, color=c, label=f"k={k:g}")
    kn = gmp_mu_knee(k)
    if kn:
        ax[0].plot(kn[0], kn[1], marker="*", ms=16, color=c, mec="k", mew=0.6, zorder=5)
ax[0].plot([], [], marker="*", ms=12, color="w", mec="k", lw=0,
           label="GMP → mpn_mu_div_qr\n(block Barrett) knee")
ax[0].annotate("flat ~3× over GMP's mpn_dcpi1_div_qr;\n"
               "tail dropoff = GMP going sub-quadratic\n(Newton inverse + fast-mul Barrett)",
               xy=(0.02, 0.03), xycoords="axes fraction", fontsize=9,
               va="bottom", ha="left",
               bbox=dict(boxstyle="round", fc="#ffffcc", ec="0.6", alpha=0.9))

for a, title, base in ((ax[0], "DC speedup over GMP  (mpn_tdiv_qr)", "GMP"),
                       (ax[1], "DC speedup over u52 schoolbook wrapper", "schoolbook")):
    a.set_xscale("log", base=2)
    a.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.6)
    a.set_xlabel("divisor size  bn  (vecs = 416-bit blocks)")
    a.set_ylabel(f"speedup  ({base}_ns / DC_ns)   >1 = DC faster")
    a.set_title(title)
    a.grid(True, which="both", alpha=0.25)
    a.legend(title="an = k·bn", fontsize=9)
    a.set_ylim(bottom=0)

fig.suptitle("AVX-512 IFMA u52 divide-and-conquer division  —  an = k·bn sweep (Zen5, clang -O2 -march=native)",
             fontsize=13)
fig.tight_layout(rect=[0,0,1,0.97])
out = path.rsplit(".",1)[0] + ".svg"
fig.savefig(out)
print("wrote", out)

# quick text summary
print("\nDC/GMP time ratio (lower=DC faster):")
print(f"{'bn':>5} | " + " ".join(f"k={k:<4g}" for k in ks))
bysz = defaultdict(dict)
for r in rows: bysz[int(r['bn'])][float(r['k'])] = float(r['dc_ns'])/float(r['gmp_ns'])
for bn in sorted(bysz):
    print(f"{bn:>5} | " + " ".join(f"{bysz[bn].get(k,float('nan')):<6.2f}" for k in ks))
