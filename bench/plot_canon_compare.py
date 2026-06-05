#!/usr/bin/env python3
import argparse

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("out")
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("LOW", "HIGH"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv).sort_values("n")

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(df["n"], df["pos_simd_ns_per_limb"], label="pos SIMD canon", linewidth=1.6)
    axes[0].plot(df["n"], df["neg_scalar_ns_per_limb"], label="neg scalar canon", linewidth=1.6)
    axes[0].set_ylabel("ns / limb")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    if args.ylim is not None:
        axes[0].set_ylim(args.ylim[0], args.ylim[1])

    axes[1].axhline(2.0, color="black", linewidth=0.9, alpha=0.5, label="2x threshold")
    axes[1].plot(df["n"], df["neg_over_pos"], label="neg scalar ns / pos SIMD ns", linewidth=1.8)
    axes[1].set_xlabel("u52 limbs")
    axes[1].set_ylabel("ratio")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.suptitle("canon positive SIMD vs negative scalar")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)


if __name__ == "__main__":
    main()
