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

    df = pd.read_csv(args.csv)

    fig, ax = plt.subplots(figsize=(12, 7))
    for bn in sorted(df["bn52"].unique()):
        d = df[df["bn52"] == bn].sort_values("an52")
        ax.plot(d["an52"], d["ns_per_limb"], label=f"bn={bn}", linewidth=1.6)

    ax.set_title("mul_u52_1 sweep")
    ax.set_xlabel("an u52 limbs")
    ax.set_ylabel("total_ns / (an * bn)")
    ax.grid(True, alpha=0.25)
    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)


if __name__ == "__main__":
    main()
