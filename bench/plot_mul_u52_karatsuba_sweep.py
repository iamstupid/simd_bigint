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
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df["n"], df["basecase_ns_per_limb"], label="basecase", linewidth=1.6)
    ax.plot(df["n"], df["karatsuba_ns_per_limb"], label="1-level karatsuba", linewidth=1.8)
    ax.set_title("u52 multiplication: basecase vs one-level Karatsuba")
    ax.set_xlabel("an = bn (u52 limbs)")
    ax.set_ylabel("ns / limb")
    if args.ylim is not None:
        ax.set_ylim(args.ylim[0], args.ylim[1])
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)


if __name__ == "__main__":
    main()
