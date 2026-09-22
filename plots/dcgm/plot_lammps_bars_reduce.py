#!/usr/bin/env python3
"""Plot measured vs. SMOCC runtime comparison from a CSV file."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLUMNS = ["Measurement", "SMOCC Lower", "SMOCC Mid", "SMOCC Upper", "SMOCC Mock"]

# 'Measurement' gets its own distinct color and no hatch.
# All SMOCC bars share a similar color, distinguished by hatch.
COLORS = [
    "silver",  # measurement (distinct)
    "mediumslateblue",  # smocc_lower
    "mediumpurple",  # smocc_mid
    "darkslateblue",  # smocc_upper
    "indigo",  # mock_smocc
]
HATCHES = [
    "",  # measurement (no hatch)
    "//",  # smocc_lower
    "\\\\",  # smocc_mid
    "xx",  # smocc_upper
    "O",  # mock_smocc
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv", type=Path, help="Path to the input CSV file (must contain a 'category' column)."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to the output image. Defaults to the CSV filename with a .png extension.",
    )
    parser.add_argument(
        "-e",
        "--exclude",
        action="append",
        default=None,
        metavar="CATEGORY",
        help="Category to exclude from the plot; repeat to exclude several. (default: A100-40G)",
    )
    parser.add_argument(
        "--ylabel", default="Overall Runtime (second)", help="Label for the y axis."
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output image DPI.")
    return parser.parse_args()


def load_data(csv_path, excluded):
    df = pd.read_csv(csv_path)
    if "category" not in df.columns:
        raise SystemExit(f"{csv_path}: missing required 'category' column")

    # Preserve the order in which categories appear in the file.
    categories = [cat for cat in dict.fromkeys(df["category"]) if cat not in excluded]
    if not categories:
        raise SystemExit(f"{csv_path}: no categories left after exclusions")

    value_cols = [c for c in df.columns if c != "category"]
    data = np.array(
        [
            df[df["category"] == cat][value_cols].mean(axis=0).round().astype(int).values
            for cat in categories
        ]
    )
    return categories, data


def make_plot(categories, data, ylabel):
    plt.rcParams["hatch.linewidth"] = 3  # default is 1.0

    x = np.arange(len(categories))
    n_cols = data.shape[1]
    width = 0.17
    bar_width = width * 0.9  # leave a small gap between bars

    fig, ax = plt.subplots(figsize=(12, 4.5))

    # Reference measurement values (column 0) for each category
    measurement = data[:, 0]

    for i in range(n_cols):
        label = COLUMNS[i] if i < len(COLUMNS) else f"Series {i}"
        offset = (i - n_cols / 2) * width + width / 2
        if i == 0:
            # First bar: keep the solid fill color, grey edge.
            ax.bar(
                x + offset,
                data[:, i],
                bar_width,
                label=label,
                color=COLORS[i],
                hatch=HATCHES[i],
                edgecolor="darkgrey",
                linewidth=3,
            )
        else:
            # Other bars: white fill, colored edge + hatch
            bars = ax.bar(
                x + offset,
                data[:, i],
                bar_width,
                label=label,
                color="white",  # or "none" for transparent
                hatch=HATCHES[i],
                edgecolor=COLORS[i],
                linewidth=3,
            )
            # Signed error percentage relative to the measurement bar
            err_pct = (data[:, i] - measurement) / measurement * 100
            ax.bar_label(bars, labels=[f"{p:+.1f}%" for p in err_pct], padding=2, fontsize=17)

    ax.set_ylabel(ylabel, fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(ncol=3, loc="upper left", columnspacing=0.6, fontsize=19)
    ax.set_ylim(0, np.max(data) * 1.5)

    # Remove ticks on x axis (keep the labels)
    ax.tick_params(axis="x", length=0, labelsize=22)
    # Make y axis ticks point inward
    ax.tick_params(axis="y", direction="in", labelsize=22)

    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_linewidth(3)

    fig.tight_layout()
    return fig


def main():
    args = parse_args()
    excluded = set(args.exclude) if args.exclude is not None else {"A100-40G"}
    output = args.output or args.csv.with_suffix(".png")

    categories, data = load_data(args.csv, excluded)
    fig = make_plot(categories, data, args.ylabel)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
