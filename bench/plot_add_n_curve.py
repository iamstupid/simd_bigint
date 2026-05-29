#!/usr/bin/env python3

import csv
import sys

import matplotlib.pyplot as plt


def read_csv(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "n": int(row["n"]),
                    "simd": float(row["simd_ns_per_limb"]),
                    "gmp": float(row["gmp_ns_per_limb"]) if "gmp_ns_per_limb" in row else None,
                    "scalar": float(row["scalar_ns_per_limb"]),
                }
            )
    return rows


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/simd_bigint_add_n_curve.csv"
    png_path = sys.argv[2] if len(sys.argv) > 2 else "/tmp/simd_bigint_add_n_curve.png"

    rows = read_csv(csv_path)
    n = [r["n"] for r in rows]
    simd = [r["simd"] for r in rows]
    gmp = [r["gmp"] for r in rows]
    scalar = [r["scalar"] for r in rows]

    plt.figure(figsize=(13, 7))
    plt.plot(n, simd, label="simd_mpn_add_n", linewidth=1.8)
    if any(x is not None for x in gmp):
        plt.plot(n, gmp, label="GMP mpn_add_n", linewidth=1.6)
    plt.plot(n, scalar, label="scalar reference", linewidth=1.2, alpha=0.75)

    for x in (8, 16, 24, 32, 64, 128, 256, 512):
        if n[0] <= x <= n[-1]:
            plt.axvline(x, color="0.85", linewidth=0.8, zorder=0)

    plt.title("mpn_add_n hot-L1 throughput")
    plt.xlabel("limbs")
    plt.ylabel("ns / limb")
    plt.grid(True, which="major", alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    print(png_path)


if __name__ == "__main__":
    main()
