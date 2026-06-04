#!/usr/bin/env python3
import argparse
import math

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv")
    parser.add_argument("out")
    parser.add_argument("--metric", choices=("time", "speedup"), default="time")
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("LOW", "HIGH"))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    wanted = [
        "an=1bn",
        "an=2bn",
        "an=4bn",
        "an=8bn",
        "bn=4",
        "bn=6",
        "bn=8",
        "bn=10",
        "bn=12",
    ]
    shapes = [shape for shape in wanted if shape in set(df["shape"])]

    ncols = 3
    nrows = math.ceil(len(shapes) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    axes = axes.ravel()

    for ax, shape in zip(axes, shapes):
        d = df[df["shape"] == shape].sort_values("an52")
        x = d["an52"]
        if args.metric == "time":
            ax.plot(x, d["old_ns_per_digit_product"], label="old mul.h", linewidth=1.6)
            ax.plot(x, d["new_ns_per_digit_product"], label="new mul_new.h", linewidth=1.8)
            ax.set_ylabel("ns / (an * bn)")
        else:
            ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
            ax.plot(x, d["speedup_old_over_new"], label="old / new", linewidth=1.8)
            ax.set_ylabel("speedup")
        ax.set_title(shape)
        ax.set_xlabel("an u52 digits")
        if args.ylim is not None:
            ax.set_ylim(args.ylim[0], args.ylim[1])
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    for ax in axes[len(shapes):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)


if __name__ == "__main__":
    main()
